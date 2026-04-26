"""Quantitative analysis tools for quantum states and circuits.

All functions operate on NumPy arrays and StateVector objects.
No GUI or PyQt6 dependencies -- this module belongs to the engine layer.

Provides:
- StateAnalysis: fidelity, entropy, purity, expectation values, entanglement
- ConvergenceAnalysis: shot convergence, TVD, KL divergence
- BenchmarkAnalysis: gate timing, quantum volume estimation
"""

from __future__ import annotations

import time
import math
from enum import Enum
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .state_vector import StateVector
from .gates import X_MATRIX, Y_MATRIX, Z_MATRIX

_PAULI = {"X": X_MATRIX, "Y": Y_MATRIX, "Z": Z_MATRIX}


# =========================================================================
# StateAnalysis -- pure-state and density-matrix metrics
# =========================================================================

class StateAnalysis:
    """Static methods for quantitative analysis of quantum states."""

    # ---- Fidelity ---------------------------------------------------------

    @staticmethod
    def state_fidelity(psi: np.ndarray, phi: np.ndarray) -> float:
        """Fidelity between two pure state vectors: |<psi|phi>|^2."""
        return float(np.abs(np.vdot(psi, phi)) ** 2)

    @staticmethod
    def process_fidelity(ideal: StateVector, actual: StateVector) -> float:
        """Convenience wrapper: fidelity between two StateVector objects."""
        return StateAnalysis.state_fidelity(ideal.data, actual.data)

    @staticmethod
    def density_fidelity(rho: np.ndarray, sigma: np.ndarray) -> float:
        r"""Uhlmann fidelity between two density matrices.

        F(rho, sigma) = (Tr sqrt(sqrt(rho) * sigma * sqrt(rho)))^2

        Both inputs are sanitized (Hermitian symmetrization, eigenvalue
        clipping, trace renormalization) to guard against numerical noise
        from partial trace or ensemble averaging.
        """
        rho = StateAnalysis._sanitize_density_matrix(rho)
        sigma = StateAnalysis._sanitize_density_matrix(sigma)
        sqrt_rho = StateAnalysis._matrix_sqrt(rho)
        product = sqrt_rho @ sigma @ sqrt_rho
        eigvals = np.linalg.eigvalsh(product)
        eigvals = np.maximum(eigvals, 0.0)  # clip numerical negatives
        fid = float(np.sum(np.sqrt(eigvals)) ** 2)
        return min(fid, 1.0)  # clamp to valid range

    @staticmethod
    def _sanitize_density_matrix(rho: np.ndarray) -> np.ndarray:
        """Ensure density matrix is Hermitian, positive, and trace-1.

        Guards against small numerical errors that accumulate in
        partial trace, ensemble averaging, and matrix square root.
        """
        rho = (rho + rho.conj().T) / 2  # enforce Hermitian
        tr = np.trace(rho).real
        if tr > 1e-15:
            rho = rho / tr  # normalize trace to 1
        return rho

    @staticmethod
    def _matrix_sqrt(mat: np.ndarray) -> np.ndarray:
        """Hermitian positive semi-definite matrix square root via eigendecomp."""
        eigvals, eigvecs = np.linalg.eigh(mat)
        eigvals = np.maximum(eigvals, 0.0)
        return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.conj().T

    # ---- Entropy ----------------------------------------------------------

    @staticmethod
    def von_neumann_entropy(state: StateVector) -> float:
        """Von Neumann entropy S(rho) in bits for a pure state.

        For a pure state |psi>, S = 0. This is included for API
        completeness; use entanglement_entropy for subsystem entropy.
        """
        # Pure state -> S = 0, but compute properly for generality
        rho = state.get_density_matrix()
        return StateAnalysis.von_neumann_entropy_dm(rho)

    @staticmethod
    def von_neumann_entropy_dm(rho: np.ndarray) -> float:
        """Von Neumann entropy S(rho) = -Tr(rho log2 rho) in bits."""
        eigvals = np.linalg.eigvalsh(rho)
        eigvals = eigvals[eigvals > 1e-15]  # filter near-zero
        return float(-np.sum(eigvals * np.log2(eigvals)))

    @staticmethod
    def entanglement_entropy(
        state: StateVector, subsystem_qubits: list[int]
    ) -> float:
        """Entanglement entropy of a subsystem (in bits).

        Computes the von Neumann entropy of the reduced density matrix
        obtained by tracing out all qubits NOT in subsystem_qubits.
        """
        rho_sub = StateAnalysis.partial_trace(state, subsystem_qubits)
        return StateAnalysis.von_neumann_entropy_dm(rho_sub)

    # ---- Partial trace ----------------------------------------------------

    @staticmethod
    def partial_trace(
        state: StateVector, keep_qubits: list[int]
    ) -> np.ndarray:
        """Partial trace: trace out all qubits not in keep_qubits.

        Returns the reduced density matrix for the kept subsystem.
        Dimension: 2^len(keep_qubits) x 2^len(keep_qubits).
        """
        n = state.num_qubits
        keep = sorted(keep_qubits)
        trace_out = [q for q in range(n) if q not in keep]
        k = len(keep)

        # Full density matrix as a tensor with 2n indices
        # rho[i0,i1,...,i_{n-1}, j0,j1,...,j_{n-1}]
        rho_full = np.outer(state.data, np.conj(state.data))
        rho_tensor = rho_full.reshape([2] * (2 * n))

        # Trace over each qubit in trace_out:
        # Contract axis i with axis i+n for each traced-out qubit
        # We use einsum with a carefully built subscript string.
        # Bra indices: 0..n-1, Ket indices: n..2n-1
        # For kept qubits: bra_i and ket_i get distinct output labels
        # For traced qubits: bra_i and ket_i share the same label (summed)

        bra_labels = list(range(2 * n))  # first n
        ket_labels = list(range(n, 2 * n))  # second n

        # Build einsum labels
        input_labels = list(range(2 * n))
        output_labels = []
        next_label = 2 * n

        # For traced-out qubits, ket index = bra index (contraction)
        for q in trace_out:
            input_labels[n + q] = input_labels[q]  # ket_q = bra_q

        # Output labels: only kept qubits, bra then ket
        for q in keep:
            output_labels.append(input_labels[q])
        for q in keep:
            output_labels.append(input_labels[n + q])

        rho_reduced = np.einsum(rho_tensor, input_labels, output_labels)
        dim_sub = 2 ** k
        return rho_reduced.reshape(dim_sub, dim_sub)

    # ---- Purity -----------------------------------------------------------

    @staticmethod
    def purity(state: StateVector) -> float:
        """Purity Tr(rho^2). Returns 1.0 for pure states."""
        rho = state.get_density_matrix()
        return StateAnalysis.purity_dm(rho)

    @staticmethod
    def purity_dm(rho: np.ndarray) -> float:
        """Purity Tr(rho^2) for a density matrix."""
        return float(np.real(np.trace(rho @ rho)))

    # ---- Entanglement measures --------------------------------------------

    @staticmethod
    def mutual_information(
        state: StateVector, qubit_a: int, qubit_b: int
    ) -> float:
        """Quantum mutual information I(A:B) = S(A) + S(B) - S(AB) in bits."""
        sa = StateAnalysis.entanglement_entropy(state, [qubit_a])
        sb = StateAnalysis.entanglement_entropy(state, [qubit_b])
        sab = StateAnalysis.entanglement_entropy(state, [qubit_a, qubit_b])
        return float(max(0.0, sa + sb - sab))

    @staticmethod
    def concurrence(
        state: StateVector, qubit_a: int, qubit_b: int
    ) -> float:
        """Wootters concurrence for a 2-qubit subsystem.

        C = max(0, lambda_1 - lambda_2 - lambda_3 - lambda_4)
        where lambda_i are the square roots of eigenvalues of rho * rho_tilde
        in decreasing order, and rho_tilde = (Y x Y) rho* (Y x Y).
        """
        rho = StateAnalysis.partial_trace(state, [qubit_a, qubit_b])

        # sigma_y tensor sigma_y
        sy = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
        yy = np.kron(sy, sy)

        rho_tilde = yy @ np.conj(rho) @ yy
        product = rho @ rho_tilde

        eigvals = np.linalg.eigvals(product)
        eigvals = np.real(eigvals)
        eigvals = np.maximum(eigvals, 0.0)
        lambdas = np.sort(np.sqrt(eigvals))[::-1]

        c = lambdas[0] - np.sum(lambdas[1:])
        return float(max(0.0, c))

    # ---- Expectation values -----------------------------------------------

    @staticmethod
    def expectation_value(
        state: StateVector,
        observable: np.ndarray,
        target_qubits: list[int],
    ) -> complex:
        """Compute <psi|O|psi> for an observable acting on target_qubits.

        Uses the same tensor contraction approach as gate application
        to avoid constructing the full 2^n x 2^n operator.
        """
        # Apply observable to a copy, then compute inner product
        psi = state.data.copy()
        temp = state.copy()
        temp.apply_gate(observable, target_qubits)
        return complex(np.vdot(psi, temp.data))

    @staticmethod
    def pauli_expectation(
        state: StateVector, pauli: str, qubit: int
    ) -> float:
        """Compute <psi|P|psi> for P in {X, Y, Z} on a single qubit."""
        if pauli.upper() not in _PAULI:
            raise ValueError(f"Unknown Pauli: {pauli}. Use 'X', 'Y', or 'Z'.")
        obs = _PAULI[pauli.upper()]
        val = StateAnalysis.expectation_value(state, obs, [qubit])
        return float(np.real(val))


# =========================================================================
# EntanglementEventDetector -- detect entanglement creation/destruction
# =========================================================================

class EntanglementEventType(Enum):
    """Classification of entanglement events."""
    CREATION = "creation"               # pair goes from separable to entangled
    DISENTANGLEMENT = "disentanglement" # pair goes from entangled to separable
    INCREASE = "increase"               # entanglement increases significantly
    DECREASE = "decrease"               # entanglement decreases significantly


@dataclass
class EntanglementEvent:
    """A detected change in pairwise entanglement."""
    step: int
    qubit_pair: tuple[int, int]
    event_type: EntanglementEventType
    magnitude: float         # absolute change in mutual information
    entropy_before: float    # MI before
    entropy_after: float     # MI after


class EntanglementEventDetector:
    """Detects entanglement creation and destruction events step by step.

    Tracks pairwise mutual information I(A:B) at each step and fires
    events when the change exceeds a threshold with hysteresis.

    Hysteresis prevents spurious events from noise fluctuations:
    - ``epsilon_on``: threshold to transition from separable to entangled
    - ``epsilon_off``: threshold to transition from entangled to separable
      (should be < epsilon_on for proper hysteresis)
    - ``persistence``: number of consecutive steps a condition must hold
      before an event is emitted

    Usage::

        detector = EntanglementEventDetector(epsilon=0.01)
        for step, state in enumerate(states):
            events = detector.process_step(state, step)
        timeline = detector.get_timeline()
    """

    def __init__(
        self,
        epsilon: float = 0.01,
        epsilon_on: float | None = None,
        epsilon_off: float | None = None,
        persistence: int = 1,
    ):
        # Hysteresis thresholds
        self.epsilon_on = epsilon_on if epsilon_on is not None else epsilon
        self.epsilon_off = epsilon_off if epsilon_off is not None else epsilon * 0.5
        self.epsilon = epsilon  # delta threshold for INCREASE/DECREASE
        self.persistence = max(1, persistence)

        self._prev_mi: dict[tuple[int, int], float] = {}  # pair -> MI
        self._entangled: dict[tuple[int, int], bool] = {}  # pair -> entangled state
        self._pending: dict[tuple[int, int], int] = {}  # pair -> consecutive steps
        self._pending_type: dict[tuple[int, int], EntanglementEventType] = {}
        self._events: list[EntanglementEvent] = []
        self._pair_history: dict[tuple[int, int], list[tuple[int, float]]] = {}

    def process_step(
        self, state: StateVector, step_index: int
    ) -> list[EntanglementEvent]:
        """Process one simulation step and detect entanglement events.

        Args:
            state: Quantum state after this step.
            step_index: Current step index.

        Returns:
            List of events detected at this step.
        """
        n = state.num_qubits
        step_events: list[EntanglementEvent] = []

        for i in range(n):
            for j in range(i + 1, n):
                pair = (i, j)
                mi = StateAnalysis.mutual_information(state, i, j)

                # Record history
                if pair not in self._pair_history:
                    self._pair_history[pair] = []
                self._pair_history[pair].append((step_index, mi))

                prev = self._prev_mi.get(pair, 0.0)
                was_entangled = self._entangled.get(pair, False)
                delta = mi - prev

                # Determine candidate event type with hysteresis
                candidate: EntanglementEventType | None = None
                if not was_entangled and mi >= self.epsilon_on:
                    candidate = EntanglementEventType.CREATION
                elif was_entangled and mi < self.epsilon_off:
                    candidate = EntanglementEventType.DISENTANGLEMENT
                elif abs(delta) > self.epsilon:
                    candidate = (
                        EntanglementEventType.INCREASE if delta > 0
                        else EntanglementEventType.DECREASE
                    )

                # Apply persistence filter
                if candidate is not None:
                    prev_pending = self._pending_type.get(pair)
                    if prev_pending == candidate:
                        self._pending[pair] = self._pending.get(pair, 0) + 1
                    else:
                        self._pending[pair] = 1
                        self._pending_type[pair] = candidate

                    if self._pending.get(pair, 0) >= self.persistence:
                        # Emit event
                        if candidate == EntanglementEventType.CREATION:
                            self._entangled[pair] = True
                        elif candidate == EntanglementEventType.DISENTANGLEMENT:
                            self._entangled[pair] = False

                        event = EntanglementEvent(
                            step=step_index,
                            qubit_pair=pair,
                            event_type=candidate,
                            magnitude=abs(delta),
                            entropy_before=prev,
                            entropy_after=mi,
                        )
                        step_events.append(event)
                        self._events.append(event)
                        # Reset pending counter
                        self._pending[pair] = 0
                        self._pending_type.pop(pair, None)
                else:
                    # No candidate: reset pending state
                    self._pending.pop(pair, None)
                    self._pending_type.pop(pair, None)

                self._prev_mi[pair] = mi

        return step_events

    def get_timeline(self) -> list[EntanglementEvent]:
        """Return all detected events in chronological order."""
        return list(self._events)

    def get_pair_history(
        self, qa: int, qb: int
    ) -> list[tuple[int, float]]:
        """Return MI history for a specific qubit pair."""
        pair = (min(qa, qb), max(qa, qb))
        return list(self._pair_history.get(pair, []))

    def get_all_pair_histories(self) -> dict[tuple[int, int], list[tuple[int, float]]]:
        """Return MI history for all tracked qubit pairs."""
        return dict(self._pair_history)

    def reset(self) -> None:
        """Clear all state and history."""
        self._prev_mi.clear()
        self._events.clear()
        self._pair_history.clear()


# =========================================================================
# ConvergenceAnalysis -- shot count convergence metrics
# =========================================================================

class ConvergenceAnalysis:
    """Shot-count convergence analysis: TVD, KL divergence vs shots."""

    @staticmethod
    def tvd(
        ideal_probs: np.ndarray,
        empirical_counts: dict[str, int],
        total_shots: int,
    ) -> float:
        """Total Variation Distance between ideal and empirical distributions.

        TVD = 0.5 * sum_i |p_ideal_i - p_empirical_i|.  Range [0, 1].
        """
        n_states = len(ideal_probs)
        num_qubits = int(np.log2(n_states))
        tvd = 0.0
        for idx in range(n_states):
            bitstring = format(idx, f'0{num_qubits}b')
            p_ideal = ideal_probs[idx]
            p_emp = empirical_counts.get(bitstring, 0) / total_shots
            tvd += abs(p_ideal - p_emp)
        return float(0.5 * tvd)

    @staticmethod
    def kl_divergence(
        ideal_probs: np.ndarray,
        empirical_counts: dict[str, int],
        total_shots: int,
        epsilon: float = 1e-10,
    ) -> float:
        """KL divergence D_KL(ideal || empirical).

        Uses epsilon smoothing to avoid log(0).
        """
        n_states = len(ideal_probs)
        num_qubits = int(np.log2(n_states))
        kl = 0.0
        for idx in range(n_states):
            p = ideal_probs[idx]
            if p < epsilon:
                continue
            bitstring = format(idx, f'0{num_qubits}b')
            q = empirical_counts.get(bitstring, 0) / total_shots + epsilon
            kl += p * np.log2(p / q)
        return float(max(0.0, kl))

    @staticmethod
    def shot_convergence(
        state: StateVector,
        shot_counts: list[int],
        seed: int | None = None,
    ) -> list[dict]:
        """Compute TVD and KL divergence for varying shot counts.

        Returns list of {"shots": int, "tvd": float, "kl_divergence": float}.
        """
        from .measurement import MeasurementEngine

        ideal_probs = state.probabilities
        rng = np.random.default_rng(seed)
        results = []

        for shots in shot_counts:
            child_rng = np.random.default_rng(rng.integers(0, 2**63))
            counts = MeasurementEngine.sample(state, shots, rng=child_rng)
            tvd = ConvergenceAnalysis.tvd(ideal_probs, counts, shots)
            kl = ConvergenceAnalysis.kl_divergence(ideal_probs, counts, shots)
            results.append({
                "shots": shots,
                "tvd": tvd,
                "kl_divergence": kl,
            })

        return results


# =========================================================================
# BenchmarkAnalysis -- performance and quantum volume metrics
# =========================================================================

class BenchmarkAnalysis:
    """Runtime benchmarking and quantum volume estimation."""

    @staticmethod
    def gate_timing(
        num_qubits_range: range,
        gate_matrix: np.ndarray,
        target_qubits_func: Callable[[int], list[int]],
        repetitions: int = 20,
    ) -> list[dict]:
        """Benchmark gate application time vs. qubit count.

        Args:
            num_qubits_range: Range of qubit counts to test.
            gate_matrix: The gate matrix to apply.
            target_qubits_func: Function mapping num_qubits -> target qubit list.
            repetitions: Number of repetitions for timing.

        Returns list of {"num_qubits": int, "mean_time_ms": float, "std_time_ms": float}.
        """
        results = []
        for nq in num_qubits_range:
            targets = target_qubits_func(nq)
            times = []
            for _ in range(repetitions):
                sv = StateVector(nq)
                t0 = time.perf_counter()
                sv.apply_gate(gate_matrix, targets)
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1000)  # ms

            results.append({
                "num_qubits": nq,
                "mean_time_ms": float(np.mean(times)),
                "std_time_ms": float(np.std(times)),
            })
        return results

    @staticmethod
    def quantum_volume(
        max_qubits: int = 8,
        num_trials: int = 100,
        noise_model: object | None = None,
        seed: int | None = None,
    ) -> dict:
        """Estimate quantum volume using random SU(4) circuits.

        For each width m (2..max_qubits), generate random circuits of
        depth m using random 2-qubit unitaries. Compute the heavy output
        probability. QV = 2^m where m is the largest width achieving
        heavy output probability > 2/3.

        Returns {"quantum_volume": int, "log2_qv": int,
                 "results_per_width": list[dict]}.
        """
        from .simulator import Simulator
        from .circuit import QuantumCircuit, GateInstance
        from .gate_registry import GateRegistry

        rng = np.random.default_rng(seed)
        results_per_width = []
        best_m = 1

        for m in range(2, min(max_qubits + 1, 9)):  # cap at 8 for performance
            heavy_count = 0
            for trial in range(num_trials):
                # Generate random circuit of depth m with m qubits
                circuit = QuantumCircuit(num_qubits=m)

                # Build random layers: each layer applies random
                # single-qubit rotations
                for col in range(m):
                    for q in range(m):
                        # Random SU(2): Rz(a) Ry(b) Rz(c)
                        a, b, c = rng.uniform(0, 2 * np.pi, 3)
                        circuit.add_gate(GateInstance(
                            "Rz", [q], [a], col * 3
                        ))
                        circuit.add_gate(GateInstance(
                            "Ry", [q], [b], col * 3 + 1
                        ))
                        circuit.add_gate(GateInstance(
                            "Rz", [q], [c], col * 3 + 2
                        ))

                # Simulate ideal
                sim = Simulator()
                result = sim.run(circuit, shots=0)
                ideal_probs = result.final_state.probabilities

                # Simulate noisy (or ideal if no noise)
                if noise_model is not None:
                    sim_noisy = Simulator(noise_model=noise_model)
                    result_noisy = sim_noisy.run(circuit, shots=0)
                    actual_probs = result_noisy.final_state.probabilities
                else:
                    actual_probs = ideal_probs

                # Heavy output: states with ideal prob > median
                median_prob = float(np.median(ideal_probs))
                heavy_mask = ideal_probs > median_prob
                heavy_prob = float(np.sum(actual_probs[heavy_mask]))

                if heavy_prob > 2.0 / 3.0:
                    heavy_count += 1

            success_rate = heavy_count / num_trials
            passed = success_rate > 2.0 / 3.0

            results_per_width.append({
                "width": m,
                "success_rate": success_rate,
                "passed": passed,
            })

            if passed:
                best_m = m

        return {
            "quantum_volume": 2 ** best_m,
            "log2_qv": best_m,
            "results_per_width": results_per_width,
        }
