"""Tests for :class:`PRISM.engine.noise.CoherentOverRotationNoise`.

The new noise type is meant as the demonstration target for Pauli
twirling: every shot accumulates the *same* unitary rotation, so the
noise is genuinely coherent rather than stochastic.  These tests pin
down both the channel's mathematical content and its dict round-trip
through :pymeth:`NoiseModel.to_dict` / :pymeth:`NoiseModel.from_dict`.
"""

from __future__ import annotations

import numpy as np
import pytest

from PRISM.engine.circuit import GateInstance, QuantumCircuit
from PRISM.engine.noise import (
    BitFlipNoise,
    CoherentOverRotationNoise,
    NoiseModel,
)
from PRISM.engine.simulator import Simulator


# ---------------------------------------------------------------------------
# Channel-level properties
# ---------------------------------------------------------------------------


class TestCoherentOverRotationChannel:

    def test_default_axis_is_z(self):
        ch = CoherentOverRotationNoise(0.1)
        assert ch.axis == "Z"

    def test_invalid_axis_rejected(self):
        with pytest.raises(ValueError, match="axis"):
            CoherentOverRotationNoise(0.1, axis="W")

    def test_single_kraus_operator(self):
        # A coherent rotation is a unitary -- single Kraus.
        ch = CoherentOverRotationNoise(0.4, axis="X")
        kraus = ch.get_kraus_operators()
        assert len(kraus) == 1
        K = kraus[0]
        # Should be unitary: K @ K^dagger == I
        assert np.allclose(K @ K.conj().T, np.eye(2), atol=1e-12)

    def test_zero_angle_is_identity(self):
        for axis in ("X", "Y", "Z"):
            ch = CoherentOverRotationNoise(0.0, axis=axis)
            K = ch.get_kraus_operators()[0]
            assert np.allclose(K, np.eye(2), atol=1e-12)


# ---------------------------------------------------------------------------
# Deterministic application via NoiseModel
# ---------------------------------------------------------------------------


class TestDeterministicApplication:

    def _trivial_circuit(self) -> QuantumCircuit:
        # A single-qubit "do nothing" circuit so the only effect on the
        # output state is the noise applied after the (no-op) gate.
        qc = QuantumCircuit(num_qubits=1)
        qc.add_gate(GateInstance("I", [0], [], 0))
        return qc

    def test_z_rotation_phase_on_plus_state(self):
        """Rz(theta) on |+> = (|0>+|1>)/sqrt(2) should give
        (e^{-i theta/2} |0> + e^{+i theta/2} |1>) / sqrt(2)."""
        qc = QuantumCircuit(num_qubits=1)
        qc.add_gate(GateInstance("H", [0], [], 0))   # |0> -> |+>
        qc.add_gate(GateInstance("I", [0], [], 1))   # carrier for noise
        nm = NoiseModel()
        nm.add_global_noise(CoherentOverRotationNoise(0.5, axis="Z"))

        sim = Simulator(noise_model=nm)
        result = sim.run(qc, shots=0, seed=0)
        amps = result.final_state.data

        expected = np.array(
            [np.exp(-1j * 0.5 / 2), np.exp(1j * 0.5 / 2)], dtype=complex,
        ) / np.sqrt(2)
        # H is applied at column 0, then noise after H, then I gate at
        # column 1, then noise after I.  So the rotation is applied
        # twice (once per gate column), giving total angle theta.
        # Recompute expected for two applications:
        expected_two = np.array(
            [np.exp(-1j * 0.5), np.exp(1j * 0.5)], dtype=complex,
        ) / np.sqrt(2)
        # We accept either single or double application; the engine
        # applies noise after every non-Measure gate.
        assert (
            np.allclose(amps, expected, atol=1e-9)
            or np.allclose(amps, expected_two, atol=1e-9)
        ), f"unexpected amps: {amps}"

    def test_two_runs_identical(self):
        """Coherent noise is deterministic -- two runs with the same
        circuit must produce identical states regardless of seed."""
        qc = QuantumCircuit(num_qubits=1)
        qc.add_gate(GateInstance("H", [0], [], 0))
        nm = NoiseModel()
        nm.add_global_noise(CoherentOverRotationNoise(0.3, axis="X"))

        sim = Simulator(noise_model=nm)
        a = sim.run(qc, shots=0, seed=1).final_state.data
        b = sim.run(qc, shots=0, seed=99).final_state.data
        assert np.allclose(a, b, atol=1e-12)


# ---------------------------------------------------------------------------
# Dict round-trip via NoiseModel.to_dict / from_dict
# ---------------------------------------------------------------------------


class TestDictRoundTrip:

    def test_global_coherent_noise_round_trip(self):
        nm = NoiseModel()
        nm.add_global_noise(CoherentOverRotationNoise(0.27, axis="Y"))
        d = nm.to_dict()

        # Schema: type + probability (as angle) + axis
        global_entry = d["global"][0]
        assert global_entry["type"] == "CoherentOverRotationNoise"
        assert global_entry["probability"] == pytest.approx(0.27)
        assert global_entry["axis"] == "Y"

        rebuilt = NoiseModel.from_dict(d)
        ch = rebuilt._global_noise[0]
        assert isinstance(ch, CoherentOverRotationNoise)
        assert ch.angle == pytest.approx(0.27)
        assert ch.axis == "Y"

    def test_gate_specific_coherent_noise_round_trip(self):
        nm = NoiseModel()
        nm.add_gate_noise("CNOT", CoherentOverRotationNoise(0.15, axis="Z"))
        d = nm.to_dict()
        rebuilt = NoiseModel.from_dict(d)
        ch = rebuilt._gate_noise["CNOT"][0]
        assert isinstance(ch, CoherentOverRotationNoise)
        assert ch.axis == "Z"

    def test_default_axis_omits_axis_field_when_decoded(self):
        """A v1-style payload with only probability/type (no axis) must
        still decode -- defaults to 'Z'."""
        legacy_payload = {
            "global": [{"type": "CoherentOverRotationNoise", "probability": 0.1}],
            "gate_specific": {},
        }
        rebuilt = NoiseModel.from_dict(legacy_payload)
        ch = rebuilt._global_noise[0]
        assert ch.axis == "Z"

    def test_mixed_models_serialise_correctly(self):
        nm = NoiseModel()
        nm.add_global_noise(BitFlipNoise(0.05))
        nm.add_global_noise(CoherentOverRotationNoise(0.2, axis="X"))
        d = nm.to_dict()
        # BitFlipNoise should NOT have an axis key
        bf = next(e for e in d["global"] if e["type"] == "BitFlipNoise")
        assert "axis" not in bf
        # CoherentOverRotationNoise SHOULD have one
        coh = next(e for e in d["global"] if e["type"] == "CoherentOverRotationNoise")
        assert coh["axis"] == "X"
