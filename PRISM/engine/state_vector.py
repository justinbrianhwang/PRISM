"""Core quantum state representation using state vectors."""

from __future__ import annotations

import numpy as np


class StateVector:
    """Represents an n-qubit quantum state as a complex numpy array.

    Uses tensor contraction for efficient gate application,
    avoiding construction of the full 2^n x 2^n unitary matrix.
    """

    def __init__(self, num_qubits: int):
        if num_qubits < 1 or num_qubits > 16:
            raise ValueError(f"num_qubits must be 1-16, got {num_qubits}")
        self._num_qubits = num_qubits
        self._data = np.zeros(2 ** num_qubits, dtype=np.complex128)
        self._data[0] = 1.0 + 0.0j  # |00...0>

    @property
    def num_qubits(self) -> int:
        return self._num_qubits

    @property
    def data(self) -> np.ndarray:
        return self._data

    @data.setter
    def data(self, value: np.ndarray):
        if value.shape != (2 ** self._num_qubits,):
            raise ValueError(f"Expected shape ({2**self._num_qubits},), got {value.shape}")
        self._data = value.astype(np.complex128)

    @property
    def probabilities(self) -> np.ndarray:
        """Returns |amplitude|^2 for each basis state."""
        return np.abs(self._data) ** 2

    def apply_gate(self, gate_matrix: np.ndarray, target_qubits: list[int]):
        """Applies a gate matrix to the specified target qubits using tensor contraction.

        This is O(2^n * 4^k) where k = len(target_qubits), much more efficient
        than constructing the full 2^n x 2^n unitary for large n.
        """
        n = self._num_qubits
        k = len(target_qubits)

        for q in target_qubits:
            if q < 0 or q >= n:
                raise ValueError(f"Qubit index {q} out of range [0, {n-1}]")

        # Reshape state to (2, 2, ..., 2) tensor with n axes
        state_tensor = self._data.reshape([2] * n)

        # Reshape gate to (2, 2, ..., 2) tensor with 2k axes
        gate_tensor = gate_matrix.reshape([2] * (2 * k))

        # Contract: input axes of gate (k..2k-1) with target axes of state
        input_axes = list(range(k, 2 * k))
        result = np.tensordot(gate_tensor, state_tensor,
                              axes=(input_axes, target_qubits))

        # Result has target axes moved to front; transpose back to canonical order
        non_target = [i for i in range(n) if i not in target_qubits]
        dest_order = [0] * n
        for i, q in enumerate(target_qubits):
            dest_order[q] = i
        for i, q in enumerate(non_target):
            dest_order[q] = k + i

        result = np.transpose(result, np.argsort(dest_order))
        self._data = result.reshape(2 ** n)

    def measure_qubit(self, qubit: int,
                      rng: np.random.Generator | None = None) -> int:
        """Measures a single qubit. Collapses state. Returns 0 or 1."""
        if qubit < 0 or qubit >= self._num_qubits:
            raise ValueError(f"Qubit {qubit} out of range")

        rng = rng or np.random.default_rng()
        n = self._num_qubits
        probs = self.probabilities

        # Calculate P(qubit=0): sum probs over all states where bit 'qubit' is 0
        # Bit position: qubit 0 is the most significant bit
        bit_position = n - 1 - qubit
        mask = 1 << bit_position
        p0 = sum(probs[i] for i in range(len(probs)) if not (i & mask))

        outcome = 0 if rng.random() < p0 else 1

        # Collapse: zero out amplitudes inconsistent with outcome
        for i in range(len(self._data)):
            bit_val = (i >> bit_position) & 1
            if bit_val != outcome:
                self._data[i] = 0.0

        # Renormalize
        norm = np.sqrt(np.sum(np.abs(self._data) ** 2))
        if norm > 1e-15:
            self._data /= norm

        return outcome

    def measure_all(self, rng: np.random.Generator | None = None) -> str:
        """Measures all qubits. Returns bitstring like '0110'."""
        rng = rng or np.random.default_rng()
        probs = self.probabilities
        probs = probs / probs.sum()  # Ensure normalization
        idx = rng.choice(len(probs), p=probs)
        bitstring = format(idx, f'0{self._num_qubits}b')

        # Collapse to measured state
        self._data = np.zeros_like(self._data)
        self._data[idx] = 1.0 + 0.0j

        return bitstring

    def get_reduced_density_matrix(self, qubit: int) -> np.ndarray:
        """Computes the reduced density matrix for a single qubit by partial trace.

        Returns a 2x2 complex matrix.
        """
        n = self._num_qubits
        if qubit < 0 or qubit >= n:
            raise ValueError(f"Qubit {qubit} out of range")

        # Reshape state to (2^a, 2, 2^b) where a = qubits before, b = qubits after
        a = qubit
        b = n - qubit - 1
        dim_a = 2 ** a
        dim_b = 2 ** b

        psi = self._data.reshape(dim_a, 2, dim_b)

        # rho_reduced[i, j] = sum over environment indices of psi[env, i] * conj(psi[env, j])
        rho = np.einsum('aib,ajb->ij', psi, np.conj(psi))
        return rho

    def get_bloch_coordinates(self, qubit: int) -> tuple[float, float, float]:
        """Returns (x, y, z) Bloch sphere coordinates for a single qubit."""
        rho = self.get_reduced_density_matrix(qubit)
        x = 2.0 * np.real(rho[0, 1])
        y = 2.0 * np.imag(rho[1, 0])  # Note: y = 2*Im(rho[1,0]) = -2*Im(rho[0,1])
        z = np.real(rho[0, 0] - rho[1, 1])
        return (float(x), float(y), float(z))

    def get_density_matrix(self) -> np.ndarray:
        """Returns full density matrix rho = |psi><psi|."""
        return np.outer(self._data, np.conj(self._data))

    def copy(self) -> StateVector:
        """Deep copy of this state vector."""
        sv = StateVector.__new__(StateVector)
        sv._num_qubits = self._num_qubits
        sv._data = self._data.copy()
        return sv

    @classmethod
    def from_initial_states(cls, initial_states: list[int]) -> StateVector:
        """Create a StateVector from a list of per-qubit initial states (0 or 1).

        Args:
            initial_states: List of 0s and 1s, one per qubit.
                            E.g. [0, 1, 0] creates |010>.
        """
        n = len(initial_states)
        sv = cls(n)
        # Compute basis state index from bitstring (qubit 0 = MSB)
        index = 0
        for i, bit in enumerate(initial_states):
            if bit:
                index |= (1 << (n - 1 - i))
        sv._data = np.zeros(2 ** n, dtype=np.complex128)
        sv._data[index] = 1.0 + 0.0j
        return sv

    def reset(self, initial_states: list[int] | None = None):
        """Reset to |00...0> or to a specific computational basis state."""
        self._data = np.zeros(2 ** self._num_qubits, dtype=np.complex128)
        if initial_states and any(s != 0 for s in initial_states):
            index = 0
            for i, bit in enumerate(initial_states):
                if bit:
                    index |= (1 << (self._num_qubits - 1 - i))
            self._data[index] = 1.0 + 0.0j
        else:
            self._data[0] = 1.0 + 0.0j

    def __repr__(self) -> str:
        return f"StateVector(num_qubits={self._num_qubits})"
