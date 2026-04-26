"""Tests for :pymeth:`PRISM.engine.qec.QECSimulator.analyze_metric_agreement`.

The metric-agreement analysis classifies each QEC trial into one of
four bins (both pass, both fail, F-only, Z-only) and returns one
aggregate per physical-error rate.  These tests verify:

* The four bins partition the trials -- they always sum to ``n_trials``.
* Output is deterministic under a fixed seed.
* For a near-zero error rate every trial lands in the ``both_pass``
  bin (perfect-correction regime).
* For a noiseless cycle every trial passes both metrics.
* The convenience properties (fractions) compute correctly.
"""

from __future__ import annotations

import pytest

from PRISM.engine.qec import (
    BitFlipCode,
    MetricAgreement,
    QECSimulator,
    Shor9Code,
)


# ---------------------------------------------------------------------------
# Shape and partitioning
# ---------------------------------------------------------------------------


class TestPartitioning:

    def test_bins_sum_to_n_trials(self):
        sim = QECSimulator(BitFlipCode())
        rates = [0.05, 0.15, 0.30]
        breakdowns = sim.analyze_metric_agreement(
            rates, n_trials=40, noise_type="bit_flip", seed=0,
        )
        assert len(breakdowns) == len(rates)
        for b in breakdowns:
            assert (
                b.n_both_pass + b.n_both_fail + b.n_f_only + b.n_z_only
                == b.n_trials
            ), f"bins don't sum at p={b.physical_rate}"
            assert b.n_trials == 40

    def test_rate_order_preserved(self):
        sim = QECSimulator(BitFlipCode())
        rates = [0.20, 0.05, 0.10]
        breakdowns = sim.analyze_metric_agreement(
            rates, n_trials=20, noise_type="bit_flip", seed=0,
        )
        assert [b.physical_rate for b in breakdowns] == rates


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestReproducibility:

    def test_same_seed_same_output(self):
        sim = QECSimulator(BitFlipCode())
        rates = [0.10, 0.20]
        a = sim.analyze_metric_agreement(
            rates, n_trials=30, noise_type="bit_flip", seed=42,
        )
        b = sim.analyze_metric_agreement(
            rates, n_trials=30, noise_type="bit_flip", seed=42,
        )
        for ai, bi in zip(a, b):
            assert ai.n_both_pass == bi.n_both_pass
            assert ai.n_both_fail == bi.n_both_fail
            assert ai.n_f_only == bi.n_f_only
            assert ai.n_z_only == bi.n_z_only

    def test_different_seeds_can_differ(self):
        sim = QECSimulator(BitFlipCode())
        rates = [0.30]  # high enough to produce variable outcomes
        a = sim.analyze_metric_agreement(rates, n_trials=80, seed=1)
        b = sim.analyze_metric_agreement(rates, n_trials=80, seed=2)
        # Not strictly guaranteed but extremely likely that >= one bin
        # differs at p=0.3 with depolarizing noise.
        assert (
            (a[0].n_both_pass != b[0].n_both_pass)
            or (a[0].n_both_fail != b[0].n_both_fail)
        )


# ---------------------------------------------------------------------------
# Boundary regimes
# ---------------------------------------------------------------------------


class TestBoundaryRegimes:

    def test_zero_noise_yields_all_both_pass(self):
        """At p = 0 every cycle restores the codeword exactly."""
        sim = QECSimulator(BitFlipCode())
        breakdowns = sim.analyze_metric_agreement(
            [0.0], n_trials=20, noise_type="bit_flip", seed=0,
        )
        b = breakdowns[0]
        assert b.n_both_pass == 20
        assert b.n_both_fail == 0
        assert b.n_f_only == 0
        assert b.n_z_only == 0

    def test_zero_noise_shor_yields_all_both_pass(self):
        sim = QECSimulator(Shor9Code())
        b = sim.analyze_metric_agreement(
            [0.0], n_trials=10, noise_type="depolarizing", seed=0,
        )[0]
        assert b.n_both_pass == 10


# ---------------------------------------------------------------------------
# Convenience properties
# ---------------------------------------------------------------------------


class TestConvenienceProperties:

    def test_fraction_disagreement(self):
        b = MetricAgreement(
            physical_rate=0.1, n_trials=100,
            n_both_pass=70, n_both_fail=20,
            n_f_only=7, n_z_only=3,
        )
        assert b.fraction_disagreement == pytest.approx(0.10)

    def test_fraction_both_pass_and_fail(self):
        b = MetricAgreement(
            physical_rate=0.1, n_trials=100,
            n_both_pass=60, n_both_fail=15,
            n_f_only=15, n_z_only=10,
        )
        assert b.fraction_both_pass == pytest.approx(0.60)
        assert b.fraction_both_fail == pytest.approx(0.15)

    def test_zero_trials_does_not_div_by_zero(self):
        b = MetricAgreement(
            physical_rate=0.1, n_trials=0,
            n_both_pass=0, n_both_fail=0,
            n_f_only=0, n_z_only=0,
        )
        assert b.fraction_disagreement == 0.0
        assert b.fraction_both_pass == 0.0
        assert b.fraction_both_fail == 0.0
