"""Measurement logic and sampling for quantum states."""

from __future__ import annotations

from enum import Enum

import numpy as np
from .state_vector import StateVector
from .gates import H_MATRIX, S_DAG_MATRIX


class MeasurementBasis(Enum):
    """Measurement basis selection."""
    Z = "Z"  # computational basis (default)
    X = "X"  # Hadamard basis (apply H before measure)
    Y = "Y"  # Y basis (apply S-dagger then H before measure)


class MeasurementEngine:
    """Handles measurement simulation with proper state collapse."""

    @staticmethod
    def measure_qubit(state: StateVector, qubit: int,
                      rng: np.random.Generator | None = None) -> tuple[int, StateVector]:
        """Measure a single qubit. Returns (outcome, collapsed_state)."""
        collapsed = state.copy()
        outcome = collapsed.measure_qubit(qubit, rng)
        return outcome, collapsed

    @staticmethod
    def measure_all(state: StateVector,
                    rng: np.random.Generator | None = None) -> tuple[str, StateVector]:
        """Measure all qubits. Returns (bitstring, collapsed_state)."""
        collapsed = state.copy()
        bitstring = collapsed.measure_all(rng)
        return bitstring, collapsed

    @staticmethod
    def sample(state: StateVector, shots: int,
               rng: np.random.Generator | None = None) -> dict[str, int]:
        """Sample 'shots' measurement outcomes without collapse.

        Uses numpy multinomial for efficiency.
        """
        rng = rng or np.random.default_rng()
        probs = state.probabilities
        # Ensure normalization
        total = probs.sum()
        if total > 1e-15:
            probs = probs / total
        else:
            # Uniform distribution if state is all zeros (shouldn't happen)
            probs = np.ones_like(probs) / len(probs)

        counts_array = rng.multinomial(shots, probs)
        n = state.num_qubits
        return {format(i, f'0{n}b'): int(c)
                for i, c in enumerate(counts_array) if c > 0}

    @staticmethod
    def sample_with_basis(
        state: StateVector,
        shots: int,
        basis: MeasurementBasis = MeasurementBasis.Z,
        readout_error=None,
        readout_mode: str = "shot",
        rng: np.random.Generator | None = None,
    ) -> dict[str, int]:
        """Sample measurements in the specified basis with optional readout error.

        For X-basis: apply H to all qubits before measurement.
        For Y-basis: apply S-dagger then H to all qubits before measurement.
        For Z-basis: standard computational basis.

        Args:
            state: Quantum state to measure.
            shots: Number of measurement shots.
            basis: Measurement basis (Z, X, or Y).
            readout_error: Optional ReadoutError model.
            readout_mode: "shot" for per-shot bitstring corruption (default),
                          "distribution" for confusion-matrix transform on the
                          probability distribution before sampling.
            rng: Random number generator.

        Returns:
            Dictionary mapping bitstrings to counts.
        """
        rng = rng or np.random.default_rng()

        # Rotate to measurement basis (on a copy)
        if basis != MeasurementBasis.Z:
            rotated = state.copy()
            for q in range(rotated.num_qubits):
                if basis == MeasurementBasis.Y:
                    rotated.apply_gate(S_DAG_MATRIX, [q])
                rotated.apply_gate(H_MATRIX, [q])
        else:
            rotated = state

        # Distribution-transform readout error: apply before sampling
        if readout_error is not None and readout_mode == "distribution":
            probs = rotated.probabilities.copy()
            total = probs.sum()
            if total > 1e-15:
                probs /= total
            noisy_probs = readout_error.apply_to_distribution(
                probs, rotated.num_qubits
            )
            # Sample from transformed distribution
            counts_array = rng.multinomial(shots, noisy_probs)
            n = rotated.num_qubits
            return {
                format(i, f'0{n}b'): int(c)
                for i, c in enumerate(counts_array) if c > 0
            }

        # Sample in computational basis
        counts = MeasurementEngine.sample(rotated, shots, rng=rng)

        # Shot-based readout error: apply after sampling
        if readout_error is not None and readout_mode == "shot":
            noisy_counts: dict[str, int] = {}
            for bitstring, count in counts.items():
                for _ in range(count):
                    noisy_bs = readout_error.apply_to_bitstring(bitstring, rng)
                    noisy_counts[noisy_bs] = noisy_counts.get(noisy_bs, 0) + 1
            counts = noisy_counts

        return counts
