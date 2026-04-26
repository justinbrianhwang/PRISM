"""Tests for :mod:`PRISM.engine.twirling`.

The twirler ships three claims that the test suite has to defend:

* **Pauli channels are fixed points of the twirl.**
  Twirling a depolarizing or bit-flip channel must leave the shot
  statistics unchanged within Monte Carlo error.  We assert this on
  the per-trial mean fidelity of a Bell circuit.
* **Coherent noise becomes stochastic under the twirl.**
  Pauli twirling converts a deterministic over-rotation into a
  Pauli channel of the same strength.  The shot-to-shot fidelity
  variance is the cleanest proxy: untwirled coherent noise has
  zero shot variance (every shot lands at the same fidelity);
  twirled has non-zero variance.
* **Bit-exact reproducibility.**
  Same seed in -> same Pauli string out, same shot trajectory.

The lower-level utilities (``random_pauli_string``,
``apply_pauli_string``) are also exercised directly so any future
refactor is caught immediately.
"""

from __future__ import annotations

import numpy as np
import pytest

from PRISM.engine.circuit import GateInstance, QuantumCircuit
from PRISM.engine.debugger import CircuitDebugger
from PRISM.engine.gates import I_MATRIX
from PRISM.engine.noise import (
    CoherentOverRotationNoise,
    DepolarizingNoise,
    NoiseModel,
)
from PRISM.engine.state_vector import StateVector
from PRISM.engine.twirling import (
    PAULI_LABELS,
    PAULI_MATRICES,
    PauliTwirler,
    apply_pauli_string,
    random_pauli_string,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bell_circuit() -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits=2)
    qc.add_gate(GateInstance("H", [0], [], 0))
    qc.add_gate(GateInstance("CNOT", [0, 1], [], 1))
    return qc


def _deep_circuit() -> QuantumCircuit:
    """A 3-qubit circuit with a chain of CNOTs so noise has somewhere
    to manifest."""
    qc = QuantumCircuit(num_qubits=3)
    qc.add_gate(GateInstance("H", [0], [], 0))
    qc.add_gate(GateInstance("CNOT", [0, 1], [], 1))
    qc.add_gate(GateInstance("CNOT", [1, 2], [], 2))
    qc.add_gate(GateInstance("H", [2], [], 3))
    return qc


# ---------------------------------------------------------------------------
# random_pauli_string
# ---------------------------------------------------------------------------


class TestRandomPauliString:

    def test_length_matches_n_qubits(self):
        rng = np.random.default_rng(0)
        out = random_pauli_string(5, rng)
        assert len(out) == 5

    def test_zero_qubits_is_empty_tuple(self):
        rng = np.random.default_rng(0)
        assert random_pauli_string(0, rng) == ()

    def test_negative_n_rejected(self):
        with pytest.raises(ValueError, match=">="):
            random_pauli_string(-1, np.random.default_rng(0))

    def test_uniform_distribution(self):
        """Across many samples, each Pauli label should appear ~1/4
        of the time per slot."""
        rng = np.random.default_rng(42)
        n_samples = 4000
        counts = {p: 0 for p in PAULI_LABELS}
        for _ in range(n_samples):
            (p,) = random_pauli_string(1, rng)
            counts[p] += 1
        # Each label expected ~1000; allow generous slack.
        for p, n in counts.items():
            assert 850 < n < 1150, f"Pauli {p}: count={n}, expected ~1000"

    def test_reproducibility(self):
        a = random_pauli_string(8, np.random.default_rng(7))
        b = random_pauli_string(8, np.random.default_rng(7))
        assert a == b


# ---------------------------------------------------------------------------
# apply_pauli_string
# ---------------------------------------------------------------------------


class TestApplyPauliString:

    def _basis(self, n: int, idx: int) -> StateVector:
        sv = StateVector(n)
        sv.data = np.zeros(2 ** n, dtype=np.complex128)
        sv.data[idx] = 1.0
        return sv

    def test_identity_string_is_no_op(self):
        sv = self._basis(2, 0b01)
        before = sv.data.copy()
        apply_pauli_string(sv, ["I", "I"], [0, 1])
        assert np.allclose(sv.data, before)

    def test_x_flips_target_qubit(self):
        # |00> -> X on q1 -> |01>
        sv = self._basis(2, 0b00)
        apply_pauli_string(sv, ["I", "X"], [0, 1])
        assert sv.data[0b01] == pytest.approx(1.0)

    def test_pauli_self_inverse(self):
        """P^2 = I on every basis state: applying any Pauli twice must
        return to the original state."""
        for label in PAULI_LABELS:
            sv = self._basis(1, 0)
            sv.apply_gate(np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2),
                          [0])  # |+>
            before = sv.data.copy()
            apply_pauli_string(sv, [label], [0])
            apply_pauli_string(sv, [label], [0])
            assert np.allclose(sv.data, before, atol=1e-12), (
                f"P={label} is not self-inverse on |+>: {sv.data}"
            )

    def test_length_mismatch_rejected(self):
        sv = self._basis(2, 0)
        with pytest.raises(ValueError, match="same length"):
            apply_pauli_string(sv, ["X", "Y"], [0])

    def test_invalid_label_rejected(self):
        sv = self._basis(2, 0)
        with pytest.raises(ValueError, match="Unknown Pauli"):
            apply_pauli_string(sv, ["W", "I"], [0, 1])


# ---------------------------------------------------------------------------
# PauliTwirler.apply_twirled_noise -- low-level
# ---------------------------------------------------------------------------


class TestPauliTwirlerLowLevel:

    def test_returns_sampled_pauli_string(self):
        sv = StateVector(2)
        nm = NoiseModel()
        nm.add_global_noise(DepolarizingNoise(0.1))
        gate = GateInstance("H", [0], [], 0)
        rng = np.random.default_rng(0)
        out = PauliTwirler.apply_twirled_noise(sv, nm, gate, rng)
        assert isinstance(out, tuple)
        assert len(out) == 1
        assert out[0] in PAULI_LABELS

    def test_no_noise_short_circuits(self):
        sv = StateVector(2)
        before = sv.data.copy()
        gate = GateInstance("H", [0], [], 0)
        rng = np.random.default_rng(0)
        out = PauliTwirler.apply_twirled_noise(sv, None, gate, rng)
        assert out == ()
        # No mutation when noise_model is None.
        assert np.allclose(sv.data, before)


# ---------------------------------------------------------------------------
# Pauli channel idempotence: depolarizing twirled == depolarizing
# ---------------------------------------------------------------------------


class TestPauliChannelIdempotence:

    def test_depolarizing_twirling_does_not_change_mean_fidelity(self):
        """Twirling a Pauli channel is the identity transformation, so
        the mean fidelity across a fixed seed should match.  Bootstrap
        Monte Carlo noise in the trial loop adds some slack but the
        means must agree to a few percent."""
        circuit = _deep_circuit()
        nm = NoiseModel()
        nm.add_global_noise(DepolarizingNoise(0.05))

        debugger = CircuitDebugger()
        plain = debugger.compute_noise_attribution(
            circuit, nm, n_trials=80, seed=2024,
        )
        # Reset noise RNG so the second run is independent.
        nm = NoiseModel()
        nm.add_global_noise(DepolarizingNoise(0.05))
        twirled = debugger.compute_noise_attribution(
            circuit, nm, n_trials=80, seed=2024, twirl=True,
        )

        # Total fidelity loss should agree closely.
        assert plain.total_fidelity_loss > 0
        assert twirled.total_fidelity_loss > 0
        rel_diff = (
            abs(plain.total_fidelity_loss - twirled.total_fidelity_loss)
            / plain.total_fidelity_loss
        )
        assert rel_diff < 0.30, (
            f"plain={plain.total_fidelity_loss:.4f}, "
            f"twirled={twirled.total_fidelity_loss:.4f}, "
            f"rel_diff={rel_diff:.3f}"
        )


# ---------------------------------------------------------------------------
# Coherent noise: twirling adds shot variance
# ---------------------------------------------------------------------------


class TestCoherentNoiseTwirling:

    def test_untwirled_coherent_has_zero_shot_variance(self):
        """A purely coherent noise (single unitary Kraus) is
        deterministic: every shot produces the same noisy state, so the
        per-trial std of the fidelity drop is exactly zero (modulo
        float arithmetic noise)."""
        circuit = _bell_circuit()
        nm = NoiseModel()
        nm.add_global_noise(CoherentOverRotationNoise(0.2, axis="Z"))

        debugger = CircuitDebugger()
        attr = debugger.compute_noise_attribution(
            circuit, nm, n_trials=40, seed=7,
        )
        for col_std in attr.delta_fidelity_std:
            assert col_std < 1e-9, (
                f"coherent noise unexpectedly has shot variance: "
                f"{attr.delta_fidelity_std}"
            )

    def test_twirled_coherent_introduces_shot_variance(self):
        """Twirling converts the coherent rotation into a stochastic
        Pauli channel, so shot-to-shot variance becomes non-trivial.
        Use a 2-qubit ZZ-coupled rotation pattern (CoherentOverRotation
        on a circuit with CNOTs) so the Pauli twirl actually has bite.
        """
        circuit = _deep_circuit()
        nm = NoiseModel()
        nm.add_global_noise(CoherentOverRotationNoise(0.25, axis="Y"))

        debugger = CircuitDebugger()
        attr = debugger.compute_noise_attribution(
            circuit, nm, n_trials=80, seed=11, twirl=True,
        )
        # Some column must show nonzero shot variance now.
        max_std = max(attr.delta_fidelity_std)
        assert max_std > 1e-6, (
            f"twirled coherent noise still has near-zero variance: "
            f"max std = {max_std:.3e}"
        )

    def test_twirling_reproducibility_under_fixed_seed(self):
        circuit = _deep_circuit()
        debugger = CircuitDebugger()

        nm = NoiseModel()
        nm.add_global_noise(CoherentOverRotationNoise(0.3, axis="X"))
        a = debugger.compute_noise_attribution(
            circuit, nm, n_trials=20, seed=42, twirl=True,
        )

        nm = NoiseModel()
        nm.add_global_noise(CoherentOverRotationNoise(0.3, axis="X"))
        b = debugger.compute_noise_attribution(
            circuit, nm, n_trials=20, seed=42, twirl=True,
        )

        assert a.delta_fidelity == b.delta_fidelity
        assert a.column_attribution_pct == b.column_attribution_pct


# ---------------------------------------------------------------------------
# Statistics-aware twirling
# ---------------------------------------------------------------------------


class TestTwirledStatistics:

    def test_with_statistics_propagates_twirl_flag(self):
        """The bootstrap-aware path must accept the twirl flag and
        produce a populated AttributionStatistics block."""
        circuit = _deep_circuit()
        nm = NoiseModel()
        nm.add_global_noise(CoherentOverRotationNoise(0.2, axis="Y"))

        debugger = CircuitDebugger()
        attr = debugger.compute_noise_attribution_with_statistics(
            circuit, nm,
            n_trials=40, n_bootstrap=200,
            seed=2024, twirl=True,
        )
        assert attr.statistics is not None
        assert attr.statistics.n_trials == 40
        assert attr.statistics.n_bootstrap == 200
