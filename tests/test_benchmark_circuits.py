"""Tests for the new benchmark circuit factories in algorithms.py.

The QAOA MaxCut and bit-flip-encoder factories are used to build the
PRISM paper figures, so we verify both *structural* correctness
(qubit count, no measurement gates, expected column layout) and
*physical* correctness (final state matches the expected analytical
output for a noiseless run).
"""

from __future__ import annotations

import numpy as np
import pytest

from PRISM.engine.algorithms import AlgorithmTemplate
from PRISM.engine.gates import GateType
from PRISM.engine.gate_registry import GateRegistry
from PRISM.engine.simulator import Simulator


# ---------------------------------------------------------------------------
# QAOA-MaxCut(4-cycle)
# ---------------------------------------------------------------------------


class TestQaoaMaxcut4Cycle:

    def test_default_shape(self):
        qc = AlgorithmTemplate.qaoa_maxcut_4cycle()
        assert qc.num_qubits == 4
        # Parallel mode -> 8 columns: H | CNOT | Rz | CNOT | CNOT | Rz | CNOT | Rx
        assert qc.get_column_count() == 8

    def test_no_measurement_gates(self):
        """The benchmark factories must omit measurements so attribution
        does not see phantom zero-weight columns."""
        qc = AlgorithmTemplate.qaoa_maxcut_4cycle()
        registry = GateRegistry.instance()
        for g in qc.gates:
            gd = registry.get(g.gate_name)
            assert gd.gate_type not in (GateType.MEASUREMENT, GateType.BARRIER)

    def test_sequential_mode_has_more_columns(self):
        qc = AlgorithmTemplate.qaoa_maxcut_4cycle(parallel_edges=False)
        # Sequential mode: H | per-edge {CNOT, Rz, CNOT} x4 | Rx -> 14 columns
        assert qc.get_column_count() == 14

    def test_zero_angles_produce_uniform_superposition(self):
        """When gamma = beta = 0 the circuit reduces to H^{tensor 4},
        which produces the uniform superposition |+>^4 = sum |x> / 4."""
        qc = AlgorithmTemplate.qaoa_maxcut_4cycle(gamma=0.0, beta=0.0)
        sim = Simulator()
        result = sim.run(qc, shots=0, seed=0)
        amps = result.final_state.data
        expected = 1.0 / np.sqrt(16)
        # Every basis state has the same magnitude
        assert np.allclose(np.abs(amps), expected, atol=1e-9)

    def test_custom_angles_change_state(self):
        """Two distinct (gamma, beta) settings must give measurably
        different states."""
        sim = Simulator()
        a = sim.run(
            AlgorithmTemplate.qaoa_maxcut_4cycle(0.7, 0.4),
            shots=0, seed=0,
        ).final_state.data
        b = sim.run(
            AlgorithmTemplate.qaoa_maxcut_4cycle(0.3, 1.0),
            shots=0, seed=0,
        ).final_state.data
        assert not np.allclose(a, b, atol=1e-6)

    def test_maxcut_invariance_under_bitflip(self):
        """MaxCut is invariant under flipping every spin: P(x) = P(~x)
        for any QAOA(C_4) state."""
        qc = AlgorithmTemplate.qaoa_maxcut_4cycle(0.5, 0.6)
        sim = Simulator()
        probs = sim.run(qc, shots=0, seed=0).final_state.probabilities
        n = 4
        for idx in range(2 ** n):
            inv = idx ^ ((1 << n) - 1)
            assert probs[idx] == pytest.approx(probs[inv], abs=1e-9)


# ---------------------------------------------------------------------------
# Bit-flip encoder
# ---------------------------------------------------------------------------


class TestBitFlipEncoder:

    def test_shape(self):
        qc = AlgorithmTemplate.bit_flip_encoder()
        assert qc.num_qubits == 3
        assert qc.get_column_count() == 2

    def test_produces_logical_zero(self):
        """Starting from |000>, the encoder maps the input through two
        CNOTs that act trivially on |000>, so the output is still |000>."""
        qc = AlgorithmTemplate.bit_flip_encoder()
        sim = Simulator()
        result = sim.run(qc, shots=0, seed=0)
        amps = result.final_state.data
        assert amps[0b000] == pytest.approx(1.0, abs=1e-9)
        # Every other amplitude must be zero
        for idx in range(1, 8):
            assert amps[idx] == pytest.approx(0.0, abs=1e-9)

    def test_applied_after_x_produces_logical_one(self):
        """X|0> = |1>, then the encoder fans out to |111>."""
        from PRISM.engine.circuit import GateInstance, QuantumCircuit
        qc = QuantumCircuit(num_qubits=3)
        qc.add_gate(GateInstance("X", [0], [], 0))
        # Now stack the encoder columns shifted by 1
        qc.add_gate(GateInstance("CNOT", [0, 1], [], 1))
        qc.add_gate(GateInstance("CNOT", [0, 2], [], 2))

        sim = Simulator()
        amps = sim.run(qc, shots=0, seed=0).final_state.data
        assert amps[0b111] == pytest.approx(1.0, abs=1e-9)

    def test_no_measurement_gates(self):
        qc = AlgorithmTemplate.bit_flip_encoder()
        registry = GateRegistry.instance()
        for g in qc.gates:
            gd = registry.get(g.gate_name)
            assert gd.gate_type not in (GateType.MEASUREMENT, GateType.BARRIER)


# ---------------------------------------------------------------------------
# Algorithm template registry: new entries listed
# ---------------------------------------------------------------------------


class TestTemplateRegistry:

    def test_qaoa_listed(self):
        names = [t["name"] for t in AlgorithmTemplate.list_templates()]
        assert "qaoa_maxcut_4cycle" in names

    def test_bit_flip_encoder_listed(self):
        names = [t["name"] for t in AlgorithmTemplate.list_templates()]
        assert "bit_flip_encoder" in names
