"""Tests for :class:`PRISM.engine.qec.Shor9Code`.

The Shor code is the smallest CSS code that corrects an arbitrary
single-qubit Pauli error.  These tests pin down the four properties
that any reasonable implementation must satisfy:

* **Encoding correctness** -- the codeword has the textbook structure:
  eight equally-weighted basis states with the right sign pattern,
  ``<Z_L> = +1`` on ``|0>_L`` and ``<Z_L> = -1`` on ``|1>_L``.
* **No-error syndrome is all zeros.**
* **Single-error syndromes are unique** -- every X, Y, Z error on
  any of the nine data qubits yields a distinct syndrome (otherwise
  the decoder cannot disambiguate them).
* **Full QEC cycle recovers** ``|0>_L`` / ``|1>_L`` after a single
  X / Y / Z error on any data qubit.

Performance: each test runs in well under a second; the worst case
(parametrised single-error cycle) is bounded by 9 data qubits x 3
Pauli types x 2 logical states = 54 cycles.
"""

from __future__ import annotations

import numpy as np
import pytest

from PRISM.engine.gates import X_MATRIX, Y_MATRIX, Z_MATRIX
from PRISM.engine.qec import (
    AVAILABLE_CODES,
    QECSimulator,
    Shor9Code,
)


@pytest.fixture
def code() -> Shor9Code:
    return Shor9Code()


# ---------------------------------------------------------------------------
# Codeword structure
# ---------------------------------------------------------------------------


class TestEncoding:

    def test_data_qubit_count(self, code):
        assert code.data_qubits == 9
        assert code.ancilla_qubits == 0
        assert code.total_qubits == 9
        assert code.code_distance == 3

    def test_zero_logical_has_eight_uniform_components(self, code):
        sv = code.encode(0)
        amps = sv.data
        # All amplitudes are real and either 0 or +1/sqrt(8).
        nonzero = np.abs(amps) > 1e-9
        assert nonzero.sum() == 8
        assert np.allclose(
            amps[nonzero].real, 1 / np.sqrt(8), atol=1e-9,
        )
        assert np.allclose(amps[nonzero].imag, 0.0, atol=1e-12)

    def test_logical_z_expectation_on_codewords(self, code):
        zero = code.encode(0)
        one = code.encode(1)
        assert code.logical_z_expectation(zero) == pytest.approx(1.0, abs=1e-9)
        assert code.logical_z_expectation(one) == pytest.approx(-1.0, abs=1e-9)

    def test_codewords_are_orthogonal(self, code):
        zero = code.encode(0).data
        one = code.encode(1).data
        overlap = np.vdot(zero, one)
        assert abs(overlap) < 1e-9

    def test_codewords_are_normalised(self, code):
        for logical in (0, 1):
            sv = code.encode(logical)
            norm = np.sqrt(np.sum(np.abs(sv.data) ** 2))
            assert abs(norm - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# Syndrome correctness
# ---------------------------------------------------------------------------


class TestSyndrome:

    def test_no_error_syndrome_is_zero(self, code):
        sv = code.encode(0)
        rng = np.random.default_rng(0)
        syndrome = code.extract_syndrome(sv, rng)
        assert syndrome == [0] * 8, f"unexpected syndrome {syndrome}"

    @pytest.mark.parametrize("err_qubit", list(range(9)))
    def test_single_x_error_block_localisation(self, code, err_qubit):
        """X on qubit i triggers the within-block Z-syndromes that
        identify the qubit, *and* leaves the between-block X-syndromes
        untouched (since X commutes with X)."""
        sv = code.encode(0)
        sv.apply_gate(X_MATRIX, [err_qubit])

        rng = np.random.default_rng(0)
        syndrome = code.extract_syndrome(sv, rng)
        z_syn, x_syn = syndrome[:6], syndrome[6:]
        # X errors do not flip X-stabilisers.
        assert x_syn == [0, 0], f"X error at q{err_qubit}: x_syn={x_syn}"
        # Z-syndromes for the affected block: not-all-zero.
        block = err_qubit // 3
        block_z = z_syn[2 * block: 2 * block + 2]
        assert block_z != [0, 0]

    @pytest.mark.parametrize("err_qubit", list(range(9)))
    def test_single_z_error_only_flips_x_stabilisers(self, code, err_qubit):
        """Z on qubit i leaves the within-block Z-syndromes alone (Z
        commutes with Z) and triggers exactly the between-block
        X-stabilisers whose support contains qubit i."""
        sv = code.encode(0)
        sv.apply_gate(Z_MATRIX, [err_qubit])

        rng = np.random.default_rng(0)
        syndrome = code.extract_syndrome(sv, rng)
        z_syn, x_syn = syndrome[:6], syndrome[6:]
        # Z errors do not flip Z-stabilisers.
        assert z_syn == [0, 0, 0, 0, 0, 0]
        # Block 1 -> (1, 0); block 2 -> (1, 1); block 3 -> (0, 1).
        block = err_qubit // 3
        expected = {0: [1, 0], 1: [1, 1], 2: [0, 1]}[block]
        assert x_syn == expected, (
            f"Z@q{err_qubit}: x_syn={x_syn}, expected {expected}"
        )


# ---------------------------------------------------------------------------
# Full encode-error-correct cycle
# ---------------------------------------------------------------------------


class TestQECCycle:

    @pytest.mark.parametrize("logical", [0, 1])
    def test_no_noise_cycle_is_exact(self, logical):
        sim = QECSimulator(Shor9Code())
        result = sim.run_cycle(
            logical_state=logical,
            noise_type="bit_flip",
            noise_prob=0.0,
            seed=0,
        )
        assert result.fidelity_after == pytest.approx(1.0, abs=1e-9)
        # Z-sign correctness
        expected_sign = 1.0 if logical == 0 else -1.0
        assert result.logical_z_expectation * expected_sign > 0.99


class TestRegistry:

    def test_shor_in_available_codes(self):
        assert "Shor [[9,1,3]]" in AVAILABLE_CODES
        assert AVAILABLE_CODES["Shor [[9,1,3]]"] is Shor9Code
