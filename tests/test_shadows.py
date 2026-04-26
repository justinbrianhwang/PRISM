"""Tests for :mod:`PRISM.engine.shadows`.

The classical-shadow protocol has two non-negotiable correctness
properties: it converges to the true Pauli expectation value as
$N \\to \\infty$, and the standard error of the mean shrinks as
$1/\\sqrt{N}$.  Both are verified here on a Bell state, where
$\\langle X X \\rangle = +1$, $\\langle Y Y \\rangle = -1$,
$\\langle Z Z \\rangle = +1$ are exactly known.

Additional tests pin down:

* The trivial observable ``"II...I"`` returns ``(1.0, 0.0)`` exactly.
* Length / character validation for the queried Pauli string.
* Shadow snapshots have the right shape and basis labels.
* Reproducibility under a fixed RNG.
"""

from __future__ import annotations

import numpy as np
import pytest

from PRISM.engine.circuit import GateInstance, QuantumCircuit
from PRISM.engine.shadows import (
    PAULI_LABELS_NONIDENTITY,
    ShadowSet,
    ShadowSnapshot,
    estimate_pauli_string,
    shadow_pauli_observables,
    take_pauli_shadows,
)
from PRISM.engine.simulator import Simulator
from PRISM.engine.state_vector import StateVector


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _bell_state() -> StateVector:
    """Standard Bell state |00> + |11> / sqrt(2)."""
    qc = QuantumCircuit(num_qubits=2)
    qc.add_gate(GateInstance("H", [0], [], 0))
    qc.add_gate(GateInstance("CNOT", [0, 1], [], 1))
    return Simulator().run(qc, shots=0, seed=0).final_state


def _zero_state(n: int) -> StateVector:
    sv = StateVector(n)
    return sv


# ---------------------------------------------------------------------------
# Sampler basics
# ---------------------------------------------------------------------------


class TestTakePauliShadows:

    def test_returns_correct_shot_count(self):
        sv = _bell_state()
        rng = np.random.default_rng(0)
        shadows = take_pauli_shadows(sv, n_shots=50, rng=rng)
        assert shadows.n_shots == 50

    def test_negative_n_rejected(self):
        sv = _bell_state()
        with pytest.raises(ValueError, match="non-negative"):
            take_pauli_shadows(sv, n_shots=-1)

    def test_zero_shots_returns_empty_set(self):
        sv = _bell_state()
        shadows = take_pauli_shadows(sv, n_shots=0)
        assert shadows.n_shots == 0
        assert shadows.n_qubits == 2

    def test_basis_labels_only_non_identity(self):
        sv = _bell_state()
        rng = np.random.default_rng(0)
        shadows = take_pauli_shadows(sv, n_shots=20, rng=rng)
        for snap in shadows.snapshots:
            for p in snap.basis:
                assert p in PAULI_LABELS_NONIDENTITY

    def test_snapshot_widths_match_n_qubits(self):
        sv = _zero_state(3)
        rng = np.random.default_rng(0)
        shadows = take_pauli_shadows(sv, n_shots=10, rng=rng)
        for snap in shadows.snapshots:
            assert len(snap.basis) == 3
            assert len(snap.outcomes) == 3

    def test_state_argument_unmodified(self):
        sv = _bell_state()
        before = sv.data.copy()
        rng = np.random.default_rng(0)
        take_pauli_shadows(sv, n_shots=20, rng=rng)
        assert np.allclose(sv.data, before, atol=1e-12)

    def test_reproducibility(self):
        sv = _bell_state()
        a = take_pauli_shadows(sv, n_shots=15, rng=np.random.default_rng(7))
        b = take_pauli_shadows(sv, n_shots=15, rng=np.random.default_rng(7))
        for sa, sb in zip(a.snapshots, b.snapshots):
            assert sa.basis == sb.basis
            assert sa.outcomes == sb.outcomes


# ---------------------------------------------------------------------------
# Pauli-string estimator
# ---------------------------------------------------------------------------


class TestEstimatePauliString:

    def test_identity_observable_returns_one_with_zero_variance(self):
        sv = _bell_state()
        shadows = take_pauli_shadows(
            sv, n_shots=30, rng=np.random.default_rng(0),
        )
        mean, sem = estimate_pauli_string(shadows, "II")
        assert mean == 1.0
        assert sem == 0.0

    def test_length_mismatch_rejected(self):
        sv = _bell_state()
        shadows = take_pauli_shadows(
            sv, n_shots=10, rng=np.random.default_rng(0),
        )
        with pytest.raises(ValueError, match="length"):
            estimate_pauli_string(shadows, "XYZ")  # 3 chars, 2 qubits

    def test_invalid_character_rejected(self):
        sv = _bell_state()
        shadows = take_pauli_shadows(
            sv, n_shots=10, rng=np.random.default_rng(0),
        )
        with pytest.raises(ValueError, match=r"\{I, X, Y, Z\}"):
            estimate_pauli_string(shadows, "XW")

    def test_bell_state_two_qubit_paulis_converge(self):
        """At N = 2000 shots on a Bell state, the three sign-fixed
        two-qubit correlators must land within 0.1 of their analytical
        values.  Bell expectations are <XX> = +1, <YY> = -1, <ZZ> = +1.
        """
        sv = _bell_state()
        rng = np.random.default_rng(2024)
        shadows = take_pauli_shadows(sv, n_shots=2000, rng=rng)

        for ps, expected in [("XX", 1.0), ("YY", -1.0), ("ZZ", 1.0)]:
            mean, sem = estimate_pauli_string(shadows, ps)
            assert abs(mean - expected) < 0.10, (
                f"<{ps}> = {mean:.3f} +/- {sem:.3f}, "
                f"expected ~ {expected}"
            )

    def test_bell_state_off_diagonal_paulis_vanish(self):
        """Bell state has <XY> = <XZ> = <YX> = <YZ> = <ZX> = <ZY> = 0
        analytically.  The shadow estimate should be small at large N."""
        sv = _bell_state()
        rng = np.random.default_rng(4321)
        shadows = take_pauli_shadows(sv, n_shots=2000, rng=rng)

        for ps in ("XY", "XZ", "YX", "YZ", "ZX", "ZY"):
            mean, _ = estimate_pauli_string(shadows, ps)
            assert abs(mean) < 0.20, f"<{ps}> = {mean:.3f}"

    def test_zero_state_single_qubit_paulis(self):
        """|00> has <Z_0> = +1, <Z_1> = +1, <X_0> = <Y_0> = 0."""
        sv = _zero_state(2)
        rng = np.random.default_rng(42)
        shadows = take_pauli_shadows(sv, n_shots=1500, rng=rng)

        for ps, expected in [("ZI", 1.0), ("IZ", 1.0), ("XI", 0.0), ("YI", 0.0)]:
            mean, _ = estimate_pauli_string(shadows, ps)
            assert abs(mean - expected) < 0.20, (
                f"<{ps}> = {mean:.3f}, expected {expected}"
            )


# ---------------------------------------------------------------------------
# Standard-error scaling
# ---------------------------------------------------------------------------


class TestErrorScaling:

    def test_sem_decreases_with_more_shots(self):
        """Standard error should drop roughly as 1/sqrt(N).  We check
        that doubling the shot count cuts SEM by at least a factor of
        1.2 (very loose to avoid false flags from RNG variance)."""
        sv = _bell_state()
        rng = np.random.default_rng(0)
        small = take_pauli_shadows(sv, n_shots=400, rng=rng)
        # Reset RNG so the larger run is independent.
        big = take_pauli_shadows(sv, n_shots=1600, rng=np.random.default_rng(1))

        _, sem_small = estimate_pauli_string(small, "XX")
        _, sem_big = estimate_pauli_string(big, "XX")
        assert sem_big < sem_small / 1.2, (
            f"SEM did not shrink with N: small={sem_small:.4f}, "
            f"big={sem_big:.4f}"
        )


# ---------------------------------------------------------------------------
# Batch helper
# ---------------------------------------------------------------------------


class TestShadowPauliObservables:

    def test_batch_keys_match_input(self):
        sv = _bell_state()
        rng = np.random.default_rng(0)
        shadows = take_pauli_shadows(sv, n_shots=200, rng=rng)
        out = shadow_pauli_observables(shadows, ["XX", "YY", "ZZ"])
        assert set(out.keys()) == {"XX", "YY", "ZZ"}
        for k, (mean, sem) in out.items():
            assert isinstance(mean, float)
            assert isinstance(sem, float)
            assert sem >= 0.0
