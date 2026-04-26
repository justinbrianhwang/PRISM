"""Benchmark suite for the quantum circuit simulator.

Provides predefined benchmark circuits and a runner that validates
simulator correctness by checking state fidelity, total variation
distance (TVD), and expected measurement outcomes.

Classes:
    BenchmarkResult: Dataclass holding the outcome of a single benchmark.
    BenchmarkSuite: Collection of predefined benchmarks with a runner.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .circuit import QuantumCircuit, GateInstance


@dataclass
class BenchmarkResult:
    """Result of running a single benchmark circuit.

    Attributes:
        name: Human-readable benchmark name.
        passed: Whether the benchmark met all acceptance criteria.
        fidelity: State fidelity between ideal and (optionally noisy) states.
        tvd: Total variation distance between ideal probabilities and
            empirical measurement distribution.
        runtime_ms: Wall-clock time for the ideal simulation in milliseconds.
        details: Optional free-form string with additional information.
    """

    name: str
    passed: bool
    fidelity: float
    tvd: float
    runtime_ms: float
    details: str = ""


class BenchmarkSuite:
    """Predefined quantum circuit benchmarks for validation."""

    # ------------------------------------------------------------------
    # Individual benchmark definitions
    # ------------------------------------------------------------------

    @staticmethod
    def _bell_benchmark() -> dict:
        """Bell state: H on q0, CNOT q0->q1. Expect |00>+|11>/sqrt(2)."""
        circuit = QuantumCircuit(num_qubits=2)
        circuit.add_gate(GateInstance("H", [0], [], 0))
        circuit.add_gate(GateInstance("CNOT", [0, 1], [], 1))
        return {
            "name": "Bell State",
            "circuit": circuit,
            "expected_nonzero": {"00", "11"},
            "expected_fidelity_min": 0.99,
        }

    @staticmethod
    def _ghz3_benchmark() -> dict:
        """GHZ-3: H on q0, CNOT q0->q1, CNOT q0->q2."""
        circuit = QuantumCircuit(num_qubits=3)
        circuit.add_gate(GateInstance("H", [0], [], 0))
        circuit.add_gate(GateInstance("CNOT", [0, 1], [], 1))
        circuit.add_gate(GateInstance("CNOT", [0, 2], [], 2))
        return {
            "name": "GHZ-3",
            "circuit": circuit,
            "expected_nonzero": {"000", "111"},
            "expected_fidelity_min": 0.99,
        }

    @staticmethod
    def _hadamard1_benchmark() -> dict:
        """Single Hadamard: H on q0. Expect |+> state."""
        circuit = QuantumCircuit(num_qubits=1)
        circuit.add_gate(GateInstance("H", [0], [], 0))
        return {
            "name": "Hadamard-1",
            "circuit": circuit,
            "expected_nonzero": {"0", "1"},
            "expected_fidelity_min": 0.99,
        }

    @staticmethod
    def _qft3_benchmark() -> dict:
        """QFT on 3 qubits (using algorithm template if available)."""
        from quantum_sim.engine.algorithms import AlgorithmTemplate

        circuit = AlgorithmTemplate.quantum_fourier_transform(3)
        return {
            "name": "QFT-3",
            "circuit": circuit,
            "expected_nonzero": None,  # All states should be nonzero for QFT on |0>
            "expected_fidelity_min": 0.99,
        }

    @staticmethod
    def _identity_benchmark() -> dict:
        """Identity: no gates. Expect |0...0>."""
        circuit = QuantumCircuit(num_qubits=2)
        return {
            "name": "Identity",
            "circuit": circuit,
            "expected_nonzero": {"00"},
            "expected_fidelity_min": 0.9999,
        }

    @staticmethod
    def _xgate_benchmark() -> dict:
        """X gate on q0. Expect |10>."""
        circuit = QuantumCircuit(num_qubits=2)
        circuit.add_gate(GateInstance("X", [0], [], 0))
        return {
            "name": "X-Gate",
            "circuit": circuit,
            "expected_nonzero": {"10"},
            "expected_fidelity_min": 0.99,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @classmethod
    def get_all_benchmarks(cls) -> list[dict]:
        """Return the list of all predefined benchmark definitions.

        Each entry is a dict with keys:
        - ``name``: human-readable benchmark name
        - ``circuit``: a :class:`QuantumCircuit` instance
        - ``expected_nonzero``: set of bitstrings expected in measurements,
          or ``None`` if all basis states are expected
        - ``expected_fidelity_min``: minimum acceptable fidelity
        """
        return [
            cls._bell_benchmark(),
            cls._ghz3_benchmark(),
            cls._hadamard1_benchmark(),
            cls._qft3_benchmark(),
            cls._identity_benchmark(),
            cls._xgate_benchmark(),
        ]

    @classmethod
    def run_all(
        cls,
        noise_model: object | None = None,
        seed: int | None = None,
    ) -> list[BenchmarkResult]:
        """Run all benchmarks and return results.

        For each benchmark the runner will:

        1. Run an ideal simulation (no noise) and record wall-clock time.
        2. If *noise_model* is provided, run a noisy simulation and compute
           the state fidelity between the ideal and noisy final states.
           Otherwise the fidelity is trivially 1.0.
        3. Compute the total variation distance (TVD) between the ideal
           probability distribution and the empirical measurement counts.
        4. Check that the expected nonzero basis states appear in the
           measurement results.

        Args:
            noise_model: Optional noise model instance passed to the
                :class:`Simulator`.  When ``None``, only ideal simulations
                are performed.
            seed: Optional seed for reproducibility.

        Returns:
            A list of :class:`BenchmarkResult` objects, one per benchmark.
        """
        from quantum_sim.engine.simulator import Simulator
        from quantum_sim.engine.analysis import StateAnalysis, ConvergenceAnalysis

        rng = np.random.default_rng(seed)
        results: list[BenchmarkResult] = []

        for bench in cls.get_all_benchmarks():
            name: str = bench["name"]
            circuit: QuantumCircuit = bench["circuit"]
            expected_nonzero: set[str] | None = bench["expected_nonzero"]
            fidelity_min: float = bench["expected_fidelity_min"]

            # --- Ideal run ---------------------------------------------------
            sim_ideal = Simulator()
            child_rng = np.random.default_rng(rng.integers(0, 2**63))
            t0 = time.perf_counter()
            result_ideal = sim_ideal.run(circuit, shots=1024, rng=child_rng)
            t1 = time.perf_counter()
            runtime_ms = (t1 - t0) * 1000
            ideal_state = result_ideal.final_state

            # --- Noisy run (if applicable) -----------------------------------
            if noise_model is not None:
                sim_noisy = Simulator(noise_model=noise_model)
                child_rng2 = np.random.default_rng(rng.integers(0, 2**63))
                result_noisy = sim_noisy.run(circuit, shots=0, rng=child_rng2)
                noisy_state = result_noisy.final_state
                fidelity = StateAnalysis.state_fidelity(
                    ideal_state.data, noisy_state.data
                )
            else:
                fidelity = 1.0

            # --- TVD ---------------------------------------------------------
            tvd = ConvergenceAnalysis.tvd(
                ideal_state.probabilities,
                result_ideal.measurement_counts,
                result_ideal.num_shots,
            )

            # --- Pass / fail decision ----------------------------------------
            passed = fidelity >= fidelity_min
            if expected_nonzero is not None:
                actual_nonzero = set(result_ideal.measurement_counts.keys())
                if not expected_nonzero.issubset(actual_nonzero):
                    passed = False

            details = (
                f"Fidelity={fidelity:.6f}, TVD={tvd:.4f}, "
                f"Time={runtime_ms:.1f}ms"
            )

            results.append(
                BenchmarkResult(
                    name=name,
                    passed=passed,
                    fidelity=fidelity,
                    tvd=tvd,
                    runtime_ms=runtime_ms,
                    details=details,
                )
            )

        return results
