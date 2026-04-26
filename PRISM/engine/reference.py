"""Reference state management for consistent fidelity comparisons.

Provides a centralized reference state that all fidelity computations
can use, ensuring consistent comparison baselines across the simulator.

Two reference types are maintained:

- **State reference**: noiseless final |psi>, used for fidelity computations.
  Invalidation key: ``circuit_hash`` only (basis-independent).

- **Measurement reference**: noiseless probability distribution p_b(x) in a
  specific measurement basis.  Invalidation key: ``(circuit_hash, basis)``.
  Distributions are lazily computed and cached per basis.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .state_vector import StateVector


@dataclass
class ReferenceData:
    """Immutable snapshot of a reference state for fidelity comparisons.

    ``measurement_distribution`` holds the Z-basis distribution (default).
    For other bases, use ``ReferenceManager.get_distribution(basis)``.
    """

    state: StateVector
    density_matrix: np.ndarray
    measurement_distribution: np.ndarray  # Z-basis (default)
    label: str = "reference"
    circuit_hash: int = 0
    # Lazily cached distributions for non-Z bases: {"X": array, "Y": array}
    _basis_distributions: dict[str, np.ndarray] = field(
        default_factory=dict, repr=False
    )


class ReferenceManager:
    """Stores and manages a reference quantum state.

    All panels and engine components can query the current reference
    state for fidelity comparison.

    Invalidation policy:

    - **State reference** is invalidated when the circuit structure changes
      (detected via ``circuit_hash``).  Measurement basis does *not* affect
      the state vector, so it is intentionally excluded from this key.

    - **Measurement distributions** are lazily computed per basis and cached
      inside ``ReferenceData._basis_distributions``.  Changing the basis
      triggers a one-time recomputation, not a full invalidation.
    """

    def __init__(self):
        self._reference: ReferenceData | None = None

    @property
    def reference(self) -> ReferenceData | None:
        return self._reference

    @property
    def has_reference(self) -> bool:
        return self._reference is not None

    def store(
        self,
        state: StateVector,
        label: str = "reference",
        circuit_hash: int = 0,
    ) -> ReferenceData:
        """Store a state vector as the reference.

        Eagerly computes and caches the density matrix and Z-basis
        probability distribution.  Other basis distributions are
        computed lazily via ``get_distribution(basis)``.
        """
        ref = ReferenceData(
            state=state.copy(),
            density_matrix=state.get_density_matrix(),
            measurement_distribution=state.probabilities.copy(),
            label=label,
            circuit_hash=circuit_hash,
        )
        ref._basis_distributions["Z"] = ref.measurement_distribution
        self._reference = ref
        return ref

    def clear(self) -> None:
        """Clear the stored reference."""
        self._reference = None

    def check_invalidation(self, circuit_hash: int) -> bool:
        """Check if the reference is still valid for the given circuit.

        If the circuit hash differs from the stored reference's hash,
        the reference is automatically cleared.

        Returns:
            True if the reference was invalidated (cleared), False if still valid.
        """
        if self._reference is None:
            return False
        if self._reference.circuit_hash != 0 and self._reference.circuit_hash != circuit_hash:
            self._reference = None
            return True
        return False

    def get_distribution(self, basis: str = "Z") -> np.ndarray | None:
        """Get the reference measurement distribution for a specific basis.

        Lazily computes and caches the distribution for non-Z bases by
        applying the appropriate basis rotation to the stored state.

        Args:
            basis: "Z" (computational), "X" (Hadamard), or "Y" (S-dag + H).

        Returns:
            Probability distribution array, or None if no reference is stored.
        """
        if self._reference is None:
            return None

        basis = basis.upper()
        cached = self._reference._basis_distributions.get(basis)
        if cached is not None:
            return cached

        # Compute rotated probabilities
        from .gates import H_MATRIX, S_DAG_MATRIX

        rotated = self._reference.state.copy()
        for q in range(rotated.num_qubits):
            if basis == "Y":
                rotated.apply_gate(S_DAG_MATRIX, [q])
            if basis in ("X", "Y"):
                rotated.apply_gate(H_MATRIX, [q])

        dist = rotated.probabilities.copy()
        self._reference._basis_distributions[basis] = dist
        return dist

    def fidelity_to_reference(self, state: StateVector) -> float | None:
        """Compute fidelity between a given state and the reference.

        Uses pure-state fidelity |<psi|phi>|^2 (basis-independent).
        Returns None if no reference is stored.
        """
        if self._reference is None:
            return None
        from .analysis import StateAnalysis

        return StateAnalysis.state_fidelity(
            self._reference.state.data, state.data
        )
