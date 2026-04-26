"""Algorithm Comparison -- run two circuits side-by-side and compare metrics.

Provides CircuitMetrics, ComparisonResult, and CircuitComparator.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict

import numpy as np

from .circuit import QuantumCircuit
from .simulator import Simulator, SimulationResult
from .state_vector import StateVector
from .gate_registry import GateRegistry
from .gates import GateType
from .analysis import StateAnalysis, ConvergenceAnalysis


@dataclass
class CircuitMetrics:
    """Resource metrics for a single circuit."""

    gate_count: int = 0
    depth: int = 0
    single_qubit_gates: int = 0
    multi_qubit_gates: int = 0
    num_qubits: int = 0
    parameterized_gates: int = 0
    measurement_gates: int = 0


@dataclass
class ComparisonResult:
    """Complete comparison between two circuits."""

    # Metrics
    metrics_a: CircuitMetrics
    metrics_b: CircuitMetrics

    # Simulation results
    result_a: SimulationResult
    result_b: SimulationResult

    # Distribution comparison
    output_fidelity: float  # |<psi_a|psi_b>|^2
    distribution_tvd: float  # TVD between measurement distributions
    distribution_kl_ab: float  # KL(A || B)
    distribution_kl_ba: float  # KL(B || A)

    # State properties
    entropy_a: float
    entropy_b: float
    purity_a: float
    purity_b: float


class CircuitComparator:
    """Compare two quantum circuits on metrics, output fidelity, and distributions."""

    def __init__(self):
        self._registry = GateRegistry.instance()

    def compute_metrics(self, circuit: QuantumCircuit) -> CircuitMetrics:
        """Compute resource metrics for a circuit."""
        m = CircuitMetrics(num_qubits=circuit.num_qubits)

        for gate in circuit.gates:
            try:
                gate_def = self._registry.get(gate.gate_name)
            except KeyError:
                continue

            if gate_def.gate_type == GateType.MEASUREMENT:
                m.measurement_gates += 1
                continue
            if gate_def.gate_type == GateType.BARRIER:
                continue

            m.gate_count += 1
            if gate_def.num_qubits <= 1:
                m.single_qubit_gates += 1
            else:
                m.multi_qubit_gates += 1
            if gate_def.num_params > 0:
                m.parameterized_gates += 1

        m.depth = circuit.get_column_count()
        return m

    def compare(
        self,
        circuit_a: QuantumCircuit,
        circuit_b: QuantumCircuit,
        shots: int = 1024,
        noise_model=None,
        seed: int | None = None,
    ) -> ComparisonResult:
        """Run both circuits and produce a full comparison.

        Args:
            circuit_a: First circuit.
            circuit_b: Second circuit.
            shots: Number of measurement shots for each.
            noise_model: Optional shared noise model.
            seed: Reproducibility seed.

        Returns:
            ComparisonResult with all metrics and comparisons.
        """
        rng = np.random.default_rng(seed)

        # Compute metrics
        metrics_a = self.compute_metrics(circuit_a)
        metrics_b = self.compute_metrics(circuit_b)

        # Run simulations
        sim = Simulator(noise_model=noise_model)

        seed_a = int(rng.integers(0, 2**63))
        seed_b = int(rng.integers(0, 2**63))

        if noise_model is not None:
            result_a = sim.run_with_noise(circuit_a, shots=shots, seed=seed_a)
            result_b = sim.run_with_noise(circuit_b, shots=shots, seed=seed_b)
            # Get ideal states too
            ideal_sim = Simulator()
            ideal_a = ideal_sim.run(circuit_a, shots=0, seed=seed_a)
            ideal_b = ideal_sim.run(circuit_b, shots=0, seed=seed_b)
            state_a = ideal_a.final_state
            state_b = ideal_b.final_state
        else:
            result_a = sim.run(circuit_a, shots=shots, seed=seed_a)
            result_b = sim.run(circuit_b, shots=shots, seed=seed_b)
            state_a = result_a.final_state
            state_b = result_b.final_state

        # Output fidelity: only meaningful if same number of qubits
        if circuit_a.num_qubits == circuit_b.num_qubits:
            output_fidelity = StateAnalysis.state_fidelity(state_a.data, state_b.data)
        else:
            output_fidelity = float("nan")

        # Distribution comparison
        # Normalize counts to probability distributions
        all_keys = set(result_a.measurement_counts.keys()) | set(result_b.measurement_counts.keys())
        n_bits = max(circuit_a.num_qubits, circuit_b.num_qubits)

        # Build probability arrays aligned by bitstring
        dim = 2 ** n_bits
        prob_a = np.zeros(dim)
        prob_b = np.zeros(dim)

        for key, count in result_a.measurement_counts.items():
            idx = int(key, 2)
            if idx < dim:
                prob_a[idx] = count / shots

        for key, count in result_b.measurement_counts.items():
            idx = int(key, 2)
            if idx < dim:
                prob_b[idx] = count / shots

        # TVD
        tvd = 0.5 * float(np.sum(np.abs(prob_a - prob_b)))

        # KL divergences (with epsilon smoothing)
        eps = 1e-10
        with np.errstate(divide="ignore", invalid="ignore"):
            kl_ab = float(np.nansum(
                np.where(prob_a > eps, prob_a * np.log2(prob_a / (prob_b + eps)), 0.0)
            ))
            kl_ba = float(np.nansum(
                np.where(prob_b > eps, prob_b * np.log2(prob_b / (prob_a + eps)), 0.0)
            ))

        # State properties
        entropy_a = StateAnalysis.von_neumann_entropy(state_a)
        entropy_b = StateAnalysis.von_neumann_entropy(state_b)
        purity_a = StateAnalysis.purity(state_a)
        purity_b = StateAnalysis.purity(state_b)

        return ComparisonResult(
            metrics_a=metrics_a,
            metrics_b=metrics_b,
            result_a=result_a,
            result_b=result_b,
            output_fidelity=output_fidelity,
            distribution_tvd=tvd,
            distribution_kl_ab=max(0.0, kl_ab),
            distribution_kl_ba=max(0.0, kl_ba),
            entropy_a=entropy_a,
            entropy_b=entropy_b,
            purity_a=purity_a,
            purity_b=purity_b,
        )

    @staticmethod
    def export_report(result: ComparisonResult, filepath: str) -> None:
        """Export comparison result to JSON file."""
        data = {
            "metrics_a": asdict(result.metrics_a),
            "metrics_b": asdict(result.metrics_b),
            "output_fidelity": result.output_fidelity,
            "distribution_tvd": result.distribution_tvd,
            "distribution_kl_ab": result.distribution_kl_ab,
            "distribution_kl_ba": result.distribution_kl_ba,
            "entropy_a": result.entropy_a,
            "entropy_b": result.entropy_b,
            "purity_a": result.purity_a,
            "purity_b": result.purity_b,
            "counts_a": result.result_a.measurement_counts,
            "counts_b": result.result_b.measurement_counts,
            "shots_a": result.result_a.num_shots,
            "shots_b": result.result_b.num_shots,
        }

        def _default(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            raise TypeError(f"Not serializable: {type(obj)}")

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=_default)
