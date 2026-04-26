"""Regression tests for the axis-permutation logic in
:pymeth:`PRISM.engine.state_vector.StateVector.apply_gate`.

Background
----------
A latent bug shipped in earlier revisions used ``np.argsort`` to derive
the post-tensordot transpose permutation, which produces the *inverse*
of the intended permutation whenever the placement of the target qubits
is not its own inverse.  The bug never surfaced because every existing
unit test happened to use either a single-qubit gate or a CNOT with
control on qubit 0, both of which give a self-inverse placement.

QAOA on the 4-vertex cycle exercises edges like ``(1, 2)`` and
``(3, 0)``, which exposed the bug as a 12% probability discrepancy
versus :func:`scipy.linalg.expm`.  These regression tests cover every
non-self-inverse placement we can construct on three and four qubits
with two-qubit gates so that the same regression cannot reappear
silently.
"""

from __future__ import annotations

import numpy as np
import pytest

from PRISM.engine.gates import CNOT_MATRIX, SWAP_MATRIX
from PRISM.engine.state_vector import StateVector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _basis_state(n: int, bits: tuple[int, ...]) -> StateVector:
    """``StateVector`` whose computational-basis amplitude at ``bits`` is 1."""
    if len(bits) != n:
        raise ValueError("bits length must equal n")
    sv = StateVector(n)
    sv.data = np.zeros(2 ** n, dtype=np.complex128)
    idx = 0
    for i, b in enumerate(bits):
        if b:
            idx |= 1 << (n - 1 - i)
    sv.data[idx] = 1.0
    return sv


def _bitstring_index(n: int, bits: tuple[int, ...]) -> int:
    idx = 0
    for i, b in enumerate(bits):
        if b:
            idx |= 1 << (n - 1 - i)
    return idx


def _kron(*ops: np.ndarray) -> np.ndarray:
    out = ops[0]
    for o in ops[1:]:
        out = np.kron(out, o)
    return out


def _embed_two_qubit_gate(
    gate: np.ndarray, qubits: tuple[int, int], n: int
) -> np.ndarray:
    """Build the explicit 2^n x 2^n unitary for a two-qubit ``gate`` acting
    on the ordered pair ``qubits = (a, b)`` of qubits in an ``n``-qubit
    register, treating qubit 0 as the most-significant bit.

    Used as the ground-truth oracle in tests.
    """
    a, b = qubits
    dim = 2 ** n
    out = np.zeros((dim, dim), dtype=np.complex128)
    # Iterate over every basis state of the n-qubit register.
    for in_idx in range(dim):
        bits = [(in_idx >> (n - 1 - q)) & 1 for q in range(n)]
        a_in, b_in = bits[a], bits[b]
        gate_col = (a_in << 1) | b_in
        for gate_row in range(4):
            amp = gate[gate_row, gate_col]
            if amp == 0:
                continue
            a_out, b_out = (gate_row >> 1) & 1, gate_row & 1
            new_bits = bits.copy()
            new_bits[a] = a_out
            new_bits[b] = b_out
            out_idx = 0
            for k, bb in enumerate(new_bits):
                if bb:
                    out_idx |= 1 << (n - 1 - k)
            out[out_idx, in_idx] += amp
    return out


# ---------------------------------------------------------------------------
# Pinned regression: the case that originally exposed the bug.
# ---------------------------------------------------------------------------


def test_cnot_1_2_on_010_produces_011():
    """CNOT(control=q1, target=q2) applied to |010> must give |011>.

    Prior to the apply_gate fix, this returned the wrong basis state
    |101> because the inverse axis permutation was applied.
    """
    sv = _basis_state(3, (0, 1, 0))
    sv.apply_gate(CNOT_MATRIX, [1, 2])
    expected_idx = _bitstring_index(3, (0, 1, 1))
    assert np.argmax(np.abs(sv.data)) == expected_idx
    assert sv.data[expected_idx] == pytest.approx(1.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Exhaustive: CNOT on every ordered pair of qubits, every basis state.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [3, 4])
def test_cnot_basis_action_matches_explicit_unitary(n):
    """For every ordered pair of qubits, applying CNOT to every basis
    state must agree with the explicit 2^n x 2^n CNOT unitary."""
    for ctrl in range(n):
        for tgt in range(n):
            if ctrl == tgt:
                continue
            U = _embed_two_qubit_gate(CNOT_MATRIX, (ctrl, tgt), n)
            for in_idx in range(2 ** n):
                sv = StateVector(n)
                sv.data = np.zeros(2 ** n, dtype=np.complex128)
                sv.data[in_idx] = 1.0
                sv.apply_gate(CNOT_MATRIX, [ctrl, tgt])
                expected = U[:, in_idx]
                assert np.allclose(sv.data, expected, atol=1e-12), (
                    f"CNOT({ctrl},{tgt}) on |{in_idx:0{n}b}> mismatched"
                )


@pytest.mark.parametrize("n", [3, 4])
def test_swap_basis_action_matches_explicit_unitary(n):
    """SWAP on every unordered pair of qubits should match the
    permutation matrix that swaps those two bit positions."""
    for a in range(n):
        for b in range(a + 1, n):
            U = _embed_two_qubit_gate(SWAP_MATRIX, (a, b), n)
            for in_idx in range(2 ** n):
                sv = StateVector(n)
                sv.data = np.zeros(2 ** n, dtype=np.complex128)
                sv.data[in_idx] = 1.0
                sv.apply_gate(SWAP_MATRIX, [a, b])
                expected = U[:, in_idx]
                assert np.allclose(sv.data, expected, atol=1e-12), (
                    f"SWAP({a},{b}) on |{in_idx:0{n}b}> mismatched"
                )


# ---------------------------------------------------------------------------
# Numerical: arbitrary 2-qubit unitary must agree with kron-embedded version.
# ---------------------------------------------------------------------------


def test_random_two_qubit_gate_matches_kron_embedding():
    """A random 4x4 unitary applied to a random 4-qubit state must match
    the result of multiplying by the explicitly Kronecker-embedded
    operator -- for every choice of target-qubit pair."""
    rng = np.random.default_rng(2024)
    n = 4
    # Random unitary via QR
    M = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
    Q, _ = np.linalg.qr(M)
    gate = Q

    psi = rng.standard_normal(2 ** n) + 1j * rng.standard_normal(2 ** n)
    psi /= np.linalg.norm(psi)

    for a in range(n):
        for b in range(n):
            if a == b:
                continue
            sv = StateVector(n)
            sv.data = psi.copy()
            sv.apply_gate(gate, [a, b])

            U = _embed_two_qubit_gate(gate, (a, b), n)
            expected = U @ psi
            assert np.allclose(sv.data, expected, atol=1e-10), (
                f"gate({a},{b}) on random psi mismatched at "
                f"max diff = {np.max(np.abs(sv.data - expected)):.3e}"
            )


# ---------------------------------------------------------------------------
# QAOA(C_4): full-circuit ground truth via scipy.linalg.expm.
# ---------------------------------------------------------------------------


def test_qaoa_c4_matches_scipy_expm():
    """End-to-end check that QAOA on the 4-cycle reproduces the
    scipy ``expm(-i beta H_M) @ expm(-i gamma H_C)`` ground truth."""
    pytest.importorskip("scipy")
    from scipy.linalg import expm

    from PRISM.engine.algorithms import AlgorithmTemplate
    from PRISM.engine.simulator import Simulator

    I = np.eye(2, dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)

    def kr(*ops):
        out = ops[0]
        for o in ops[1:]:
            out = np.kron(out, o)
        return out

    psi0 = np.ones(16, dtype=complex) / 4.0
    HC = (kr(Z, Z, I, I) + kr(I, I, Z, Z)
          + kr(I, Z, Z, I) + kr(Z, I, I, Z))
    HM = (kr(X, I, I, I) + kr(I, X, I, I)
          + kr(I, I, X, I) + kr(I, I, I, X))

    gamma, beta = 0.5, 0.6
    expected = expm(-1j * beta * HM) @ expm(-1j * gamma * HC) @ psi0

    qc = AlgorithmTemplate.qaoa_maxcut_4cycle(gamma, beta)
    sim = Simulator()
    actual = sim.run(qc, shots=0, seed=0).final_state.data

    # Pure states are equal up to a global phase; compare via fidelity.
    overlap = np.vdot(expected, actual)
    fidelity = abs(overlap) ** 2
    assert fidelity == pytest.approx(1.0, abs=1e-10), (
        f"QAOA(C_4) fidelity vs scipy expm = {fidelity:.10f}"
    )
