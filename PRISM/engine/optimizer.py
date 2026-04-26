"""Parameterized Circuit Optimization -- VQE/QAOA with gradient estimation.

Provides:
- ParameterBinding: maps a circuit gate parameter to an optimization variable
- CostFunction: static methods for building cost functions
- GradientEstimator: parameter-shift rule and finite difference
- CircuitOptimizer: Adam optimizer with barren plateau detection
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .circuit import QuantumCircuit, GateInstance
from .simulator import Simulator
from .state_vector import StateVector
from .gate_registry import GateRegistry
from .gates import GateType
from .analysis import StateAnalysis


# ---- Parameter binding ----------------------------------------------------

@dataclass
class ParameterBinding:
    """Maps an optimization variable to a gate parameter slot."""

    gate_index: int  # index into circuit.gates
    param_index: int  # which param of that gate
    name: str = ""  # human-readable name


class ParameterizedCircuitConfig:
    """A circuit with identified tunable parameters.

    Wraps a QuantumCircuit and tracks which gate parameters are variable.
    """

    def __init__(self, circuit: QuantumCircuit, bindings: list[ParameterBinding]):
        self._circuit = circuit
        self._bindings = bindings

    @property
    def circuit(self) -> QuantumCircuit:
        return self._circuit

    @property
    def bindings(self) -> list[ParameterBinding]:
        return self._bindings

    @property
    def num_params(self) -> int:
        return len(self._bindings)

    def get_values(self) -> np.ndarray:
        """Read current parameter values from the circuit."""
        vals = np.zeros(self.num_params)
        for i, b in enumerate(self._bindings):
            gate = self._circuit.gates[b.gate_index]
            vals[i] = gate.params[b.param_index]
        return vals

    def bind_values(self, values: np.ndarray) -> QuantumCircuit:
        """Return a copy of the circuit with parameters set to given values."""
        from copy import deepcopy
        qc = deepcopy(self._circuit)
        for i, b in enumerate(self._bindings):
            qc.gates[b.gate_index].params[b.param_index] = float(values[i])
        return qc

    @classmethod
    def auto_detect(cls, circuit: QuantumCircuit) -> ParameterizedCircuitConfig:
        """Automatically detect parameterized gates and create bindings."""
        registry = GateRegistry.instance()
        bindings = []
        for gi, gate in enumerate(circuit.gates):
            try:
                gate_def = registry.get(gate.gate_name)
            except KeyError:
                continue
            if gate_def.num_params > 0:
                for pi in range(gate_def.num_params):
                    name = f"{gate.gate_name}[{gi}].p{pi}"
                    bindings.append(ParameterBinding(gi, pi, name))
        return cls(circuit, bindings)


# ---- Cost functions -------------------------------------------------------

class CostFunction:
    """Static factory methods for building cost functions.

    Each returns a callable: (state: StateVector) -> float
    """

    @staticmethod
    def expectation_value(
        observable: np.ndarray,
        target_qubits: list[int],
    ) -> Callable[[StateVector], float]:
        """Cost = <psi|O|psi> for given observable matrix and qubits."""

        def _cost(state: StateVector) -> float:
            return float(np.real(
                StateAnalysis.expectation_value(state, observable, target_qubits)
            ))

        return _cost

    @staticmethod
    def state_fidelity(target_state: np.ndarray) -> Callable[[StateVector], float]:
        """Cost = 1 - |<target|psi>|^2  (minimize to reach target)."""

        def _cost(state: StateVector) -> float:
            fid = StateAnalysis.state_fidelity(target_state, state.data)
            return 1.0 - fid

        return _cost

    @staticmethod
    def z_expectation(qubit: int) -> Callable[[StateVector], float]:
        """Cost = <Z_qubit> (common VQE cost function)."""
        from .gates import Z_MATRIX

        def _cost(state: StateVector) -> float:
            return float(StateAnalysis.pauli_expectation(state, "Z", qubit))

        return _cost

    @staticmethod
    def vqe_hamiltonian(
        terms: list[tuple[float, str, list[int]]],
    ) -> Callable[[StateVector], float]:
        """Cost = sum_i coeff_i * <P_i> where P_i is a Pauli string.

        Args:
            terms: list of (coefficient, pauli_label, qubit_list).
                   pauli_label is a string of X/Y/Z/I, one per qubit in qubit_list.
                   E.g. [(-0.5, "ZZ", [0,1]), (0.3, "X", [0])]
        """
        from .gates import X_MATRIX, Y_MATRIX, Z_MATRIX, I_MATRIX
        pauli_map = {"I": I_MATRIX, "X": X_MATRIX, "Y": Y_MATRIX, "Z": Z_MATRIX}

        def _cost(state: StateVector) -> float:
            total = 0.0
            for coeff, pauli_str, qubits in terms:
                # Build the multi-qubit Pauli tensor product
                if len(pauli_str) == 1 and len(qubits) == 1:
                    val = StateAnalysis.pauli_expectation(state, pauli_str, qubits[0])
                else:
                    # Multi-qubit: compute tensor product matrix
                    matrices = [pauli_map[p] for p in pauli_str]
                    obs = matrices[0]
                    for m in matrices[1:]:
                        obs = np.kron(obs, m)
                    val = float(np.real(
                        StateAnalysis.expectation_value(state, obs, qubits)
                    ))
                total += coeff * val
            return total

        return _cost

    @staticmethod
    def qaoa_maxcut(edges: list[tuple[int, int]]) -> Callable[[StateVector], float]:
        """QAOA MaxCut cost: C = sum_{(i,j)} (1 - Z_i Z_j) / 2."""
        from .gates import Z_MATRIX

        def _cost(state: StateVector) -> float:
            total = 0.0
            for i, j in edges:
                zi = StateAnalysis.pauli_expectation(state, "Z", i)
                zj = StateAnalysis.pauli_expectation(state, "Z", j)
                # <Z_i Z_j> via product only works for separable states
                # For entangled states we need the full correlator
                zz_obs = np.kron(Z_MATRIX, Z_MATRIX)
                zz = float(np.real(
                    StateAnalysis.expectation_value(state, zz_obs, [i, j])
                ))
                total += (1 - zz) / 2
            return total

        return _cost


# ---- Gradient estimation --------------------------------------------------

class GradientEstimator:
    """Static methods for estimating gradients of parameterized circuits."""

    @staticmethod
    def parameter_shift(
        config: ParameterizedCircuitConfig,
        cost_fn: Callable[[StateVector], float],
        values: np.ndarray,
        shift: float = np.pi / 2,
        seed: int | None = None,
    ) -> np.ndarray:
        """Parameter-shift rule gradient.

        Exact for Rx, Ry, Rz. For U3, use with caution or prefer finite_difference.

        grad_i = [f(theta_i + shift) - f(theta_i - shift)] / (2 * sin(shift))
        """
        sim = Simulator()
        grad = np.zeros(len(values))
        coeff = 1.0 / (2.0 * np.sin(shift))

        for i in range(len(values)):
            # Forward shift
            vals_plus = values.copy()
            vals_plus[i] += shift
            qc_plus = config.bind_values(vals_plus)
            result_plus = sim.run(qc_plus, shots=0, seed=seed)
            cost_plus = cost_fn(result_plus.final_state)

            # Backward shift
            vals_minus = values.copy()
            vals_minus[i] -= shift
            qc_minus = config.bind_values(vals_minus)
            result_minus = sim.run(qc_minus, shots=0, seed=seed)
            cost_minus = cost_fn(result_minus.final_state)

            grad[i] = (cost_plus - cost_minus) * coeff

        return grad

    @staticmethod
    def finite_difference(
        config: ParameterizedCircuitConfig,
        cost_fn: Callable[[StateVector], float],
        values: np.ndarray,
        epsilon: float = 1e-4,
        seed: int | None = None,
    ) -> np.ndarray:
        """Central finite difference gradient estimation."""
        sim = Simulator()
        grad = np.zeros(len(values))

        for i in range(len(values)):
            vals_plus = values.copy()
            vals_plus[i] += epsilon
            qc_plus = config.bind_values(vals_plus)
            cost_plus = cost_fn(sim.run(qc_plus, shots=0, seed=seed).final_state)

            vals_minus = values.copy()
            vals_minus[i] -= epsilon
            qc_minus = config.bind_values(vals_minus)
            cost_minus = cost_fn(sim.run(qc_minus, shots=0, seed=seed).final_state)

            grad[i] = (cost_plus - cost_minus) / (2 * epsilon)

        return grad


# ---- Adam optimizer -------------------------------------------------------

@dataclass
class BarrenPlateauAnalysis:
    """Layer-wise barren plateau analysis result."""

    per_layer_variance: list[list[float]]  # [layer][param_in_layer]
    per_layer_mean_variance: list[float]   # mean variance per layer
    per_qubit_variance: list[float]        # mean gradient variance per qubit
    depth_scaling: list[tuple[int, float]] # (layer_depth, mean_var) for plotting
    overall_mean_variance: float
    overall_is_barren: bool
    threshold: float
    n_samples: int
    param_layer_map: list[int]             # param_idx -> layer_idx


@dataclass
class OptimizationResult:
    """Result of a parameter optimization run."""

    optimal_values: np.ndarray
    optimal_cost: float
    history: list[tuple[np.ndarray, float]]  # (values, cost) per iteration
    converged: bool
    iterations: int


class CircuitOptimizer:
    """Adam optimizer for parameterized quantum circuits.

    Args:
        config: Parameterized circuit configuration.
        cost_fn: Cost function to minimize.
        learning_rate: Adam learning rate.
        beta1: Adam first moment decay.
        beta2: Adam second moment decay.
        max_iterations: Maximum optimization steps.
        tolerance: Convergence threshold (cost change).
        gradient_method: "parameter_shift" or "finite_difference".
    """

    def __init__(
        self,
        config: ParameterizedCircuitConfig,
        cost_fn: Callable[[StateVector], float],
        learning_rate: float = 0.1,
        beta1: float = 0.9,
        beta2: float = 0.999,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        gradient_method: str = "parameter_shift",
    ):
        self._config = config
        self._cost_fn = cost_fn
        self._lr = learning_rate
        self._beta1 = beta1
        self._beta2 = beta2
        self._max_iter = max_iterations
        self._tol = tolerance
        self._grad_method = gradient_method

        n = config.num_params
        self._values = config.get_values().copy()
        self._m = np.zeros(n)  # first moment
        self._v = np.zeros(n)  # second moment
        self._t = 0  # timestep
        self._history: list[tuple[np.ndarray, float]] = []
        self._stop_requested = False

    @property
    def values(self) -> np.ndarray:
        return self._values.copy()

    @property
    def history(self) -> list[tuple[np.ndarray, float]]:
        return self._history

    def request_stop(self) -> None:
        """Request the optimizer to stop after the current step."""
        self._stop_requested = True

    def step(self, seed: int | None = None) -> tuple[np.ndarray, float]:
        """Perform one optimization step.

        Returns:
            (current_values, current_cost)
        """
        self._t += 1

        # Compute gradient
        if self._grad_method == "parameter_shift":
            grad = GradientEstimator.parameter_shift(
                self._config, self._cost_fn, self._values, seed=seed
            )
        else:
            grad = GradientEstimator.finite_difference(
                self._config, self._cost_fn, self._values, seed=seed
            )

        # Adam update
        self._m = self._beta1 * self._m + (1 - self._beta1) * grad
        self._v = self._beta2 * self._v + (1 - self._beta2) * grad ** 2

        m_hat = self._m / (1 - self._beta1 ** self._t)
        v_hat = self._v / (1 - self._beta2 ** self._t)

        self._values -= self._lr * m_hat / (np.sqrt(v_hat) + 1e-8)

        # Evaluate cost
        qc = self._config.bind_values(self._values)
        sim = Simulator()
        result = sim.run(qc, shots=0, seed=seed)
        cost = self._cost_fn(result.final_state)

        self._history.append((self._values.copy(), cost))
        return self._values.copy(), cost

    def run(
        self,
        callback: Callable[[int, np.ndarray, float], None] | None = None,
        seed: int | None = None,
    ) -> OptimizationResult:
        """Run full optimization loop.

        Args:
            callback: Optional function called each step: callback(iteration, values, cost)
            seed: Reproducibility seed.

        Returns:
            OptimizationResult with optimal values, cost, and convergence info.
        """
        self._stop_requested = False
        converged = False

        for i in range(self._max_iter):
            if self._stop_requested:
                break

            values, cost = self.step(seed=seed)

            if callback is not None:
                callback(i, values, cost)

            # Check convergence
            if len(self._history) >= 2:
                prev_cost = self._history[-2][1]
                if abs(cost - prev_cost) < self._tol:
                    converged = True
                    break

        # Find best
        best_idx = min(range(len(self._history)), key=lambda i: self._history[i][1])
        optimal_values = self._history[best_idx][0]
        optimal_cost = self._history[best_idx][1]

        return OptimizationResult(
            optimal_values=optimal_values,
            optimal_cost=optimal_cost,
            history=self._history,
            converged=converged,
            iterations=len(self._history),
        )

    def detect_barren_plateau(
        self,
        n_samples: int = 50,
        seed: int | None = None,
    ) -> dict:
        """Estimate gradient variance to detect barren plateaus.

        Returns:
            {
                "mean_variance": float,
                "per_param": list[float],  # variance of each parameter's gradient
                "is_barren": bool,  # True if mean variance < threshold
            }
        """
        rng = np.random.default_rng(seed)
        n_params = self._config.num_params
        grad_samples = np.zeros((n_samples, n_params))

        for s in range(n_samples):
            # Random parameter point
            random_vals = rng.uniform(-np.pi, np.pi, size=n_params)
            grad = GradientEstimator.parameter_shift(
                self._config, self._cost_fn, random_vals,
                seed=int(rng.integers(0, 2**63)),
            )
            grad_samples[s] = grad

        per_param_var = np.var(grad_samples, axis=0)
        mean_var = float(np.mean(per_param_var))

        # Threshold: if mean gradient variance < 1e-4, likely barren
        threshold = 1e-4

        return {
            "mean_variance": mean_var,
            "per_param": per_param_var.tolist(),
            "is_barren": mean_var < threshold,
        }

    def detect_barren_plateau_layered(
        self,
        n_samples: int = 50,
        seed: int | None = None,
    ) -> BarrenPlateauAnalysis:
        """Layer-wise barren plateau analysis.

        Groups parameters by their gate's column (circuit depth layer),
        then computes gradient variance per layer and per qubit.

        Args:
            n_samples: Number of random parameter points to sample.
            seed: Reproducibility seed.

        Returns:
            BarrenPlateauAnalysis with layer-wise and qubit-wise variance.
        """
        rng = np.random.default_rng(seed)
        n_params = self._config.num_params
        circuit = self._config.circuit

        # Map each parameter to its layer and qubit using the shared
        # layer definition from circuit.gate_to_layer_map()
        g2l = circuit.gate_to_layer_map()
        param_layer_map: list[int] = []
        param_qubit_map: list[int] = []
        for binding in self._config.bindings:
            gate = circuit.gates[binding.gate_index]
            param_layer_map.append(g2l[binding.gate_index])
            # Use first target qubit for per-qubit attribution
            param_qubit_map.append(
                gate.target_qubits[0] if gate.target_qubits else 0
            )

        # Collect gradient samples
        grad_samples = np.zeros((n_samples, n_params))
        for s in range(n_samples):
            random_vals = rng.uniform(-np.pi, np.pi, size=n_params)
            grad = GradientEstimator.parameter_shift(
                self._config, self._cost_fn, random_vals,
                seed=int(rng.integers(0, 2**63)),
            )
            grad_samples[s] = grad

        per_param_var = np.var(grad_samples, axis=0)

        # Group by layer
        layer_indices: dict[int, list[int]] = {}
        for pi, layer in enumerate(param_layer_map):
            if layer not in layer_indices:
                layer_indices[layer] = []
            layer_indices[layer].append(pi)

        sorted_layers = sorted(layer_indices.keys())
        per_layer_variance: list[list[float]] = []
        per_layer_mean: list[float] = []
        depth_scaling: list[tuple[int, float]] = []

        for layer in sorted_layers:
            pis = layer_indices[layer]
            layer_vars = [float(per_param_var[pi]) for pi in pis]
            per_layer_variance.append(layer_vars)
            mean_v = float(np.mean(layer_vars))
            per_layer_mean.append(mean_v)
            depth_scaling.append((layer, mean_v))

        # Group by qubit
        qubit_indices: dict[int, list[int]] = {}
        for pi, q in enumerate(param_qubit_map):
            if q not in qubit_indices:
                qubit_indices[q] = []
            qubit_indices[q].append(pi)

        max_qubit = max(qubit_indices.keys()) if qubit_indices else 0
        per_qubit_variance: list[float] = []
        for q in range(max_qubit + 1):
            if q in qubit_indices:
                pis = qubit_indices[q]
                per_qubit_variance.append(
                    float(np.mean([per_param_var[pi] for pi in pis]))
                )
            else:
                per_qubit_variance.append(0.0)

        overall_mean = float(np.mean(per_param_var))
        threshold = 1e-4

        return BarrenPlateauAnalysis(
            per_layer_variance=per_layer_variance,
            per_layer_mean_variance=per_layer_mean,
            per_qubit_variance=per_qubit_variance,
            depth_scaling=depth_scaling,
            overall_mean_variance=overall_mean,
            overall_is_barren=overall_mean < threshold,
            threshold=threshold,
            n_samples=n_samples,
            param_layer_map=param_layer_map,
        )
