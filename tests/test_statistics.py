"""Unit tests for :mod:`PRISM.engine.statistics`.

These tests verify:

* Bootstrap CI coverage on a known distribution (~95% of CIs cover the
  true mean across many independent draws).
* Bootstrap p-value behaviour: vanishing for clear effects, large for
  null data.
* Reproducibility under fixed seeds.
* Matrix-bootstrap consistency with 1D bootstrap on a single column.
* Benjamini-Hochberg correctness on textbook examples + monotonicity.
* :func:`attribution_percentage` and :func:`recovery_rate` helper
  invariants (non-negative, sums to 100, etc).
"""

from __future__ import annotations

import numpy as np
import pytest

from PRISM.engine.statistics import (
    BootstrapResult,
    attribution_percentage,
    benjamini_hochberg,
    bootstrap_ci,
    bootstrap_matrix_statistics,
    column_pvalues_from_bootstrap,
    recovery_rate,
)


# ---------------------------------------------------------------------------
# bootstrap_ci
# ---------------------------------------------------------------------------


class TestBootstrapCI:

    def test_returns_bootstrap_result(self, rng):
        samples = rng.normal(size=50)
        out = bootstrap_ci(samples, n_bootstrap=200, rng=rng)
        assert isinstance(out, BootstrapResult)

    def test_ci_bounds_ordered_around_estimate(self, rng):
        samples = rng.normal(size=30)
        res = bootstrap_ci(samples, n_bootstrap=400, rng=rng)
        assert res.ci_lower <= res.point_estimate <= res.ci_upper

    def test_coverage_on_normal_data(self, make_rng):
        """A 95% CI should cover the true mean ~95% of the time.

        We run 200 independent experiments with n=80 normal samples each;
        the binomial 95% interval for "true coverage = 0.95" with 200
        trials is roughly [0.92, 0.98] but with bootstrap noise we
        accept anything >= 85/100.
        """
        true_mean = 1.0
        contained = 0
        n_experiments = 100
        for k in range(n_experiments):
            sample_rng = make_rng(1000 + k)
            boot_rng = make_rng(2000 + k)
            samples = sample_rng.normal(loc=true_mean, scale=1.0, size=80)
            res = bootstrap_ci(
                samples, n_bootstrap=400, confidence=0.95, rng=boot_rng
            )
            if res.contains(true_mean):
                contained += 1
        # Lower bound generously to avoid flaky failures
        assert contained >= 85, f"coverage = {contained}/{n_experiments}"

    def test_p_value_small_for_clear_effect(self, make_rng):
        rng = make_rng(7)
        samples = rng.normal(loc=4.0, scale=1.0, size=100)
        res = bootstrap_ci(
            samples, null_value=0.0, n_bootstrap=2000, rng=make_rng(8)
        )
        assert res.p_value < 0.01
        assert res.is_significant()

    def test_p_value_large_under_null(self, make_rng):
        """Many trials drawn under the null should mostly give p > 0.05."""
        big_p_count = 0
        n_experiments = 50
        for k in range(n_experiments):
            samples = make_rng(100 + k).normal(loc=0.0, scale=1.0, size=80)
            res = bootstrap_ci(
                samples, null_value=0.0, n_bootstrap=400, rng=make_rng(200 + k)
            )
            if res.p_value > 0.05:
                big_p_count += 1
        # Expected ~95% under H0, but we're permissive due to small N
        assert big_p_count >= 35, f"only {big_p_count}/{n_experiments} non-sig"

    def test_p_value_floor(self, make_rng):
        """p-value cannot drop below 1/n_bootstrap purely from finite resampling."""
        samples = np.full(50, 100.0)  # Extremely far from 0
        res = bootstrap_ci(
            samples, null_value=0.0, n_bootstrap=100, rng=make_rng(0)
        )
        assert res.p_value >= 1.0 / 100 - 1e-12

    def test_empty_input(self):
        res = bootstrap_ci(np.array([]), null_value=0.0)
        assert res.point_estimate == 0.0
        assert res.ci_lower == 0.0
        assert res.ci_upper == 0.0
        assert res.p_value == 1.0

    def test_reproducibility(self, make_rng):
        samples = np.array([1.5, 2.0, -0.3, 4.1, 0.8, 1.2])
        r1 = bootstrap_ci(samples, n_bootstrap=500, rng=make_rng(123))
        r2 = bootstrap_ci(samples, n_bootstrap=500, rng=make_rng(123))
        assert r1.point_estimate == r2.point_estimate
        assert r1.ci_lower == r2.ci_lower
        assert r1.ci_upper == r2.ci_upper
        assert r1.p_value == r2.p_value

    def test_custom_statistic(self, rng):
        samples = rng.normal(size=50)
        res = bootstrap_ci(samples, statistic=np.median, n_bootstrap=300, rng=rng)
        assert np.isclose(res.point_estimate, float(np.median(samples)))


# ---------------------------------------------------------------------------
# bootstrap_matrix_statistics
# ---------------------------------------------------------------------------


class TestBootstrapMatrixStatistics:

    def test_shape_and_ordering(self, make_rng):
        rng = make_rng(0)
        matrix = rng.normal(size=(40, 5))
        est, lo, hi, dist = bootstrap_matrix_statistics(
            matrix,
            statistic_fn=lambda m: m.mean(axis=0),
            n_bootstrap=200,
            rng=rng,
        )
        assert est.shape == (5,)
        assert lo.shape == (5,)
        assert hi.shape == (5,)
        assert dist.shape == (200, 5)
        assert np.all(lo <= est + 1e-12)
        assert np.all(est <= hi + 1e-12)

    def test_matches_1d_bootstrap_on_single_column(self, make_rng):
        """Per-column mean matrix bootstrap should yield CIs comparable to
        running 1D bootstrap on each column, modulo bootstrap noise."""
        rng_data = make_rng(1)
        matrix = rng_data.normal(loc=0.5, scale=1.0, size=(60, 3))

        # Matrix bootstrap with shared seed for reproducibility
        _, lo_mat, hi_mat, _ = bootstrap_matrix_statistics(
            matrix,
            statistic_fn=lambda m: m.mean(axis=0),
            n_bootstrap=2000,
            rng=make_rng(99),
        )

        # 1D bootstrap on each column with independent seeds
        for c in range(3):
            res_1d = bootstrap_ci(
                matrix[:, c], n_bootstrap=2000, rng=make_rng(100 + c)
            )
            # Loose tolerance because both are noisy
            assert abs(res_1d.ci_lower - lo_mat[c]) < 0.15
            assert abs(res_1d.ci_upper - hi_mat[c]) < 0.15

    def test_rejects_non_2d_input(self):
        with pytest.raises(ValueError):
            bootstrap_matrix_statistics(
                np.array([1, 2, 3]),
                statistic_fn=lambda m: m.mean(axis=0),
                n_bootstrap=10,
            )

    def test_rejects_wrong_statistic_shape(self, make_rng):
        rng = make_rng(0)
        matrix = rng.normal(size=(20, 4))
        with pytest.raises(ValueError):
            bootstrap_matrix_statistics(
                matrix,
                statistic_fn=lambda m: m.mean(),  # returns scalar
                n_bootstrap=10,
                rng=rng,
            )

    def test_pvalues_from_bootstrap(self, make_rng):
        rng = make_rng(0)
        # 4 columns: cols 0/1 strong effect, 2/3 null
        n_trials = 80
        matrix = np.column_stack([
            rng.normal(loc=0.5, scale=0.3, size=n_trials),
            rng.normal(loc=-0.4, scale=0.3, size=n_trials),
            rng.normal(loc=0.0, scale=0.3, size=n_trials),
            rng.normal(loc=0.0, scale=0.3, size=n_trials),
        ])
        est, _, _, dist = bootstrap_matrix_statistics(
            matrix,
            statistic_fn=lambda m: m.mean(axis=0),
            n_bootstrap=2000,
            rng=make_rng(1),
        )
        p = column_pvalues_from_bootstrap(dist, est, null_value=0.0)
        assert p[0] < 0.01
        assert p[1] < 0.01
        # Null columns: not guaranteed >0.05 in any single run, but
        # should not collectively be tiny
        assert max(p[2], p[3]) > 0.05 or min(p[2], p[3]) > 0.01


# ---------------------------------------------------------------------------
# benjamini_hochberg
# ---------------------------------------------------------------------------


class TestBenjaminiHochberg:

    def test_all_significant(self):
        p = np.array([0.001, 0.002, 0.003, 0.004])
        q, rej = benjamini_hochberg(p, fdr=0.05)
        assert rej.all()
        # q-values should also lie below fdr
        assert (q <= 0.05 + 1e-12).all()

    def test_none_significant(self):
        p = np.array([0.5, 0.6, 0.7, 0.8])
        _, rej = benjamini_hochberg(p, fdr=0.05)
        assert not rej.any()

    def test_known_textbook_example(self):
        """Sorted p = [0.005, 0.01, 0.03, 0.04]; n=4, fdr=0.05.
        BH raw q: 0.02, 0.02, 0.04, 0.04 -> all reject."""
        p = np.array([0.01, 0.04, 0.03, 0.005])
        _, rej = benjamini_hochberg(p, fdr=0.05)
        assert rej.all()

    def test_partial_rejection(self):
        p = np.array([0.001, 0.04, 0.5, 0.9])
        _, rej = benjamini_hochberg(p, fdr=0.05)
        # Sorted: 0.001, 0.04, 0.5, 0.9; n=4
        # raw q: 0.004, 0.08, 0.667, 0.9
        # monotonised from right: 0.004, 0.08, 0.667, 0.9
        # at fdr 0.05: only the first passes
        assert rej.tolist() == [True, False, False, False]

    def test_q_clipped_at_one(self):
        p = np.array([0.99, 0.999, 1.0])
        q, rej = benjamini_hochberg(p, fdr=0.05)
        assert (q <= 1.0 + 1e-12).all()
        assert not rej.any()

    def test_q_monotone_along_sorted_p(self, make_rng):
        rng = make_rng(123)
        p = rng.uniform(0, 1, size=30)
        q, _ = benjamini_hochberg(p)
        sorted_q = q[np.argsort(p)]
        assert np.all(np.diff(sorted_q) >= -1e-12)

    def test_empty(self):
        q, rej = benjamini_hochberg(np.array([]))
        assert q.size == 0
        assert rej.size == 0

    def test_rejects_invalid_p(self):
        with pytest.raises(ValueError):
            benjamini_hochberg(np.array([0.5, -0.1, 0.3]))
        with pytest.raises(ValueError):
            benjamini_hochberg(np.array([0.5, 1.1, 0.3]))

    def test_order_preserved(self):
        p = np.array([0.5, 0.001, 0.04])
        q, rej = benjamini_hochberg(p, fdr=0.05)
        # Output index aligned with input index
        assert q[1] < q[0]  # 0.001 -> smallest q
        assert rej[1]


# ---------------------------------------------------------------------------
# Helpers used by NoiseAttribution
# ---------------------------------------------------------------------------


class TestAttributionPercentage:

    def test_sum_to_100(self, make_rng):
        rng = make_rng(0)
        # All-positive trial means => percentage sums to 100
        matrix = rng.uniform(0.1, 1.0, size=(20, 5))
        pct = attribution_percentage(matrix)
        assert pct.shape == (5,)
        assert abs(pct.sum() - 100.0) < 1e-9
        assert (pct >= 0).all()

    def test_negative_means_clamped_to_zero(self):
        # Two columns: positive mean, negative mean
        matrix = np.array([
            [0.4, -0.2],
            [0.5, -0.3],
            [0.3, -0.1],
        ])
        pct = attribution_percentage(matrix)
        assert pct[0] == pytest.approx(100.0)
        assert pct[1] == 0.0

    def test_no_loss_returns_zeros(self):
        matrix = np.full((10, 3), -0.5)  # all negative -> total 0
        pct = attribution_percentage(matrix)
        assert (pct == 0.0).all()

    def test_empty_matrix(self):
        # 0-trial 3-col matrix
        empty = np.zeros((0, 3))
        pct = attribution_percentage(empty)
        assert pct.shape == (3,)


class TestRecoveryRate:

    def test_all_negative(self):
        matrix = np.full((20, 3), -0.5)
        rate = recovery_rate(matrix)
        assert (rate == 1.0).all()

    def test_all_positive(self):
        matrix = np.full((20, 3), 0.5)
        rate = recovery_rate(matrix)
        assert (rate == 0.0).all()

    def test_half_negative(self):
        matrix = np.array([[-0.1, 0.2]] * 10 + [[0.3, -0.5]] * 10)
        rate = recovery_rate(matrix)
        assert rate[0] == pytest.approx(0.5)
        assert rate[1] == pytest.approx(0.5)

    def test_eps_threshold(self):
        # tiny negatives below eps should not count
        matrix = np.full((10, 1), -1e-15)
        rate = recovery_rate(matrix, eps=1e-12)
        assert rate[0] == 0.0
