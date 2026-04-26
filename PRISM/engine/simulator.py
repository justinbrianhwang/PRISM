"""Quantum circuit simulator - applies circuits to state vectors."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generator

import numpy as np

from .state_vector import StateVector
from .circuit import QuantumCircuit, GateInstance
from .gate_registry import GateRegistry
from .gates import GateType
from .measurement import MeasurementEngine, MeasurementBasis


@dataclass
class SimulationResult:
    """Result of a full simulation run."""
    final_state: StateVector
    measurement_counts: dict[str, int]
    step_states: list[StateVector] | None = None
    num_shots: int = 1024
    seed: int | None = None
    reference_state: StateVector | None = None


class Simulator:
    """Executes a QuantumCircuit on a StateVector."""

    def __init__(self, noise_model: object | None = None):
        self._gate_registry = GateRegistry.instance()
        self._noise_model = noise_model

    def run(self, circuit: QuantumCircuit, shots: int = 1024,
            record_steps: bool = False,
            seed: int | None = None,
            rng: np.random.Generator | None = None,
            measurement_basis: MeasurementBasis = MeasurementBasis.Z) -> SimulationResult:
        """Full simulation: apply all gates, then sample measurements.

        Args:
            circuit: The quantum circuit to simulate.
            shots: Number of measurement samples.
            record_steps: Whether to record state after each column.
            seed: Optional seed for reproducibility (creates rng if not given).
            rng: Optional pre-seeded Generator (takes precedence over seed).
            measurement_basis: Measurement basis (Z, X, or Y).
        """
        if rng is None:
            rng = np.random.default_rng(seed)

        state = StateVector.from_initial_states(circuit.initial_states)
        step_states = [] if record_steps else None

        # Apply gates column by column
        has_measurement = False
        for column_gates in circuit.get_ordered_gates():
            for gate_inst in column_gates:
                gate_def = self._gate_registry.get(gate_inst.gate_name)
                if gate_def.gate_type == GateType.MEASUREMENT:
                    has_measurement = True
                    continue  # Measurements handled separately via sampling
                if gate_def.gate_type == GateType.BARRIER:
                    continue  # Barriers are visual only
                self._apply_gate_instance(state, gate_inst)
                if self._noise_model is not None:
                    self._noise_model.apply(state, gate_inst)

            if record_steps:
                step_states.append(state.copy())

        # Sample measurements
        if has_measurement or shots > 0:
            readout_err = None
            if self._noise_model is not None and hasattr(self._noise_model, "readout_error"):
                readout_err = self._noise_model.readout_error
            counts = MeasurementEngine.sample_with_basis(
                state, shots, basis=measurement_basis,
                readout_error=readout_err, rng=rng,
            )
        else:
            counts = {}

        return SimulationResult(
            final_state=state,
            measurement_counts=counts,
            step_states=step_states,
            num_shots=shots,
            seed=seed,
        )

    def run_step_by_step(self, circuit: QuantumCircuit,
                         rng: np.random.Generator | None = None) -> Generator[tuple[StateVector, int], None, None]:
        """Yields (state_vector, column_index) after each column."""
        state = StateVector.from_initial_states(circuit.initial_states)
        yield state.copy(), -1  # Initial state

        ordered = circuit.get_ordered_gates()
        for col_idx, column_gates in enumerate(ordered):
            for gate_inst in column_gates:
                gate_def = self._gate_registry.get(gate_inst.gate_name)
                if gate_def.gate_type in (GateType.MEASUREMENT, GateType.BARRIER):
                    continue
                self._apply_gate_instance(state, gate_inst)
                if self._noise_model is not None:
                    self._noise_model.apply(state, gate_inst)
            yield state.copy(), col_idx

    def _apply_gate_instance(self, state: StateVector, gate: GateInstance):
        """Apply a single gate instance to the state."""
        gate_def = self._gate_registry.get(gate.gate_name)
        matrix = gate_def.matrix_func(*gate.params)
        state.apply_gate(matrix, gate.target_qubits)

    def run_with_noise(self, circuit: QuantumCircuit, shots: int = 1024,
                       seed: int | None = None,
                       rng: np.random.Generator | None = None) -> SimulationResult:
        """Run simulation with noise, re-simulating for each shot.

        Args:
            circuit: The quantum circuit to simulate.
            shots: Number of measurement shots.
            seed: Optional seed for reproducibility (creates rng if not given).
            rng: Optional pre-seeded Generator (takes precedence over seed).
        """
        if self._noise_model is None:
            return self.run(circuit, shots, seed=seed, rng=rng)

        if rng is None:
            rng = np.random.default_rng(seed)

        all_counts: dict[str, int] = {}
        for _ in range(shots):
            state = StateVector.from_initial_states(circuit.initial_states)
            for column_gates in circuit.get_ordered_gates():
                for gate_inst in column_gates:
                    gate_def = self._gate_registry.get(gate_inst.gate_name)
                    if gate_def.gate_type in (GateType.MEASUREMENT, GateType.BARRIER):
                        continue
                    self._apply_gate_instance(state, gate_inst)
                    self._noise_model.apply(state, gate_inst)

            bitstring = state.measure_all(rng)
            all_counts[bitstring] = all_counts.get(bitstring, 0) + 1

        final_state = StateVector.from_initial_states(circuit.initial_states)  # Not meaningful for noisy
        return SimulationResult(
            final_state=final_state,
            measurement_counts=all_counts,
            num_shots=shots,
            seed=seed,
        )

    def ensemble_density_matrix(
        self,
        circuit: QuantumCircuit,
        n_trials: int = 50,
        seed: int | None = None,
    ) -> np.ndarray:
        """Estimate the mixed-state density matrix via ensemble averaging.

        Stochastic noise produces a pure state per run. The ensemble
        average rho = (1/N) sum_i |psi_i><psi_i| approximates the true
        mixed state density matrix.

        Args:
            circuit: Circuit to simulate.
            n_trials: Number of stochastic runs to average.
            seed: Base seed for reproducibility.

        Returns:
            Estimated density matrix (2^n x 2^n complex array).
        """
        rng = np.random.default_rng(seed)
        dim = 2 ** circuit.num_qubits
        rho = np.zeros((dim, dim), dtype=np.complex128)

        for _ in range(n_trials):
            trial_seed = int(rng.integers(0, 2**63))
            if self._noise_model is not None:
                self._noise_model.set_seed(trial_seed)

            state = StateVector.from_initial_states(circuit.initial_states)
            for column_gates in circuit.get_ordered_gates():
                for gate_inst in column_gates:
                    gate_def = self._gate_registry.get(gate_inst.gate_name)
                    if gate_def.gate_type in (GateType.MEASUREMENT, GateType.BARRIER):
                        continue
                    self._apply_gate_instance(state, gate_inst)
                    if self._noise_model is not None:
                        self._noise_model.apply(state, gate_inst)

            # Accumulate |psi><psi|
            psi = state.data
            rho += np.outer(psi, np.conj(psi))

        rho /= n_trials
        return rho
