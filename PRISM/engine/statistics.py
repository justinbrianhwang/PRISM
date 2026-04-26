"""Statistical inference utilities for noise attribution analysis.

This module provides the statistical layer used by
:class:`PRISM.engine.debugger.NoiseAttribution` to produce
publication-quality error bars and significance markers on per-column
noise attribution.

The functions are intentionally kept dependency-free (NumPy only) so
that headless replay scripts and the GUI can use them identically.

Three primitives are exposed:

* :func:`bootstrap_ci` -- percentile bootstrap CI + two-sided p-value
  for a scalar statistic of a 1D sample.
* :func:`bootstrap_matrix_statistics` -- row-resampling bootstrap for a
  ``(n_trials, n_cols)`` matrix, returning per-column CIs for any
  derived statistic.  Used to obtain joint CIs for quantities like
  per-column attribution percentages whose denominator depends on the
  whole matrix.
* :func:`benjamini_hochberg` -- BH FDR correction for a vector of
  raw p-values.

A small ``BootstrapResult`` dataclass bundles the CI + p-value together
so that downstream code does not need to juggle parallel arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BootstrapResult:
    """Bootstrap confidence interval + two-sided p-value for one statistic.

    Attributes
    ----------
    point_estimate : float
        The statistic computed on the original sample.
    ci_lower, ci_upper : float
        Percentile bootstrap CI bounds at the requested ``confidence``.
    p_value : float
        Two-sided p-value for ``H0: statistic(population) == null_value``,
        computed by recentering the bootstrap distribution on
        ``null_value`` and counting tail mass beyond ``point_estimate``.
        Floored at ``1 / n_bootstrap`` to avoid the meaningless ``p=0``
        artifact of finite resampling.
    null_value : float
        Hypothesised value used for the p-value computation.
    confidence : float
        Two-sided CI level (e.g. ``0.95``).
    n_bootstrap : int
        Number of bootstrap resamples drawn.
    """

    point_estimate: float
    ci_lower: float
    ci_upper: float
    p_value: float
    null_value: float
    confidence: float
    n_bootstrap: int

    def is_significant(self, alpha: float = 0.05) -> bool:
        """Return ``True`` if ``p_value <= alpha``."""
        return self.p_value <= alpha

    def contains(self, value: float) -> bool:
        """Return ``True`` if ``value`` lies within the CI."""
        return self.ci_lower <= value <= self.ci_upper


# ---------------------------------------------------------------------------
# 1D bootstrap
# ---------------------------------------------------------------------------


def bootstrap_ci(
    samples: np.ndarray,
    statistic: Callable[[np.ndarray], float] = np.mean,
    null_value: float = 0.0,
    confidence: float = 0.95,
    n_bootstrap: int = 1000,
    rng: np.random.Generator | None = None,
) -> BootstrapResult:
    """Percentile-bootstrap CI and two-sided p-value for a sample statistic.

    Resamples ``samples`` with replacement ``n_bootstrap`` times, computes
    ``statistic`` on each resample, and returns:

    * The ``(1 - confidence)/2`` and ``1 - (1 - confidence)/2`` percentiles
      of the resampled statistics as the CI.
    * A two-sided p-value for ``H0: statistic(population) == null_value``,
      obtained by shifting the bootstrap distribution to be centered on
      ``null_value`` and computing
      ``2 * min(P(shifted <= observed), P(shifted >= observed))``.

    Parameters
    ----------
    samples : np.ndarray
        1D array of i.i.d. samples.
    statistic : callable, optional
        Function mapping a 1D array to a scalar.  Defaults to
        :func:`numpy.mean`.
    null_value : float, optional
        Value of the statistic under the null hypothesis (default ``0.0``).
    confidence : float, optional
        Two-sided CI level (default ``0.95``).
    n_bootstrap : int, optional
        Number of bootstrap resamples (default ``1000``).
    rng : numpy.random.Generator, optional
        Generator for reproducibility.  If ``None``, OS entropy is used.

    Returns
    -------
    BootstrapResult
        Point estimate, CI bounds and p-value.

    Notes
    -----
    Empty input returns a degenerate result with ``point_estimate ==
    null_value`` and ``p_value == 1.0`` rather than raising, so callers
    can apply this uniformly across columns that may have no trials.
    """
    rng = rng or np.random.default_rng()
    samples = np.asarray(samples, dtype=float).ravel()
    n = samples.size

    if n == 0:
        return BootstrapResult(
            point_estimate=null_value,
            ci_lower=null_value,
            ci_upper=null_value,
            p_value=1.0,
            null_value=null_value,
            confidence=confidence,
            n_bootstrap=n_bootstrap,
        )

    observed = float(statistic(samples))

    # Resample n_bootstrap copies of size-n samples with replacement,
    # vectorised in a single integers() call to avoid Python-level loops.
    indices = rng.integers(0, n, size=(n_bootstrap, n))
    # Apply the statistic per row.  np.mean is the common case and admits
    # an axis argument; fall back to a Python loop for arbitrary callables.
    if statistic is np.mean:
        boot_stats = samples[indices].mean(axis=1)
    else:
        boot_stats = np.fromiter(
            (statistic(samples[idx]) for idx in indices),
            dtype=float,
            count=n_bootstrap,
        )

    alpha = 1.0 - confidence
    ci_lower = float(np.percentile(boot_stats, 100.0 * alpha / 2.0))
    ci_upper = float(np.percentile(boot_stats, 100.0 * (1.0 - alpha / 2.0)))

    p_value = _bootstrap_pvalue(boot_stats, observed, null_value)

    return BootstrapResult(
        point_estimate=observed,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        p_value=p_value,
        null_value=null_value,
        confidence=confidence,
        n_bootstrap=n_bootstrap,
    )


def _bootstrap_pvalue(
    boot_stats: np.ndarray,
    observed: float,
    null_value: float,
) -> float:
    """Two-sided p-value from a bootstrap distribution.

    Recenters the bootstrap distribution on ``null_value`` (it is centred
    on ``observed`` by construction) and reports
    ``2 * min(P(shifted >= observed), P(shifted <= observed))``.

    The result is floored at ``1 / n_bootstrap`` so that no test ever
    reports the impossible value ``p == 0`` purely as an artefact of
    finite resampling.
    """
    n_bootstrap = boot_stats.size
    if n_bootstrap == 0:
        return 1.0

    shifted = boot_stats - observed + null_value
    p_low = float(np.mean(shifted <= observed))
    p_high = float(np.mean(shifted >= observed))
    p_value = 2.0 * min(p_low, p_high)
    p_value = min(max(p_value, 1.0 / n_bootstrap), 1.0)
    return p_value


# ---------------------------------------------------------------------------
# Matrix bootstrap (joint resampling across columns)
# ---------------------------------------------------------------------------


def bootstrap_matrix_statistics(
    trials_matrix: np.ndarray,
    statistic_fn: Callable[[np.ndarray], np.ndarray],
    confidence: float = 0.95,
    n_bootstrap: int = 1000,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Row-resampling bootstrap for a ``(n_trials, n_cols)`` trials matrix.

    Resamples *rows* of ``trials_matrix`` with replacement, applies
    ``statistic_fn`` to each resampled matrix to obtain a per-column
    statistic vector, and returns percentile CIs and the raw bootstrap
    distribution for every column.

    This is the right tool when the statistic of interest is *coupled*
    across columns -- e.g. attribution percentages where the denominator
    depends on the entire row.  Independent column-wise bootstrapping
    would break that coupling and underestimate uncertainty.

    Parameters
    ----------
    trials_matrix : np.ndarray
        ``(n_trials, n_cols)`` matrix of per-trial per-column values.
    statistic_fn : callable
        Function mapping a ``(n_trials, n_cols)`` matrix to a 1D
        ``(n_cols,)`` vector of column statistics.
    confidence : float, optional
        Two-sided CI level (default ``0.95``).
    n_bootstrap : int, optional
        Number of bootstrap resamples (default ``1000``).
    rng : numpy.random.Generator, optional
        Generator for reproducibility.

    Returns
    -------
    point_estimates : np.ndarray, shape ``(n_cols,)``
        ``statistic_fn`` applied to the original matrix.
    ci_lower, ci_upper : np.ndarray, shape ``(n_cols,)``
        Percentile bootstrap CI bounds for each column.
    boot_stats : np.ndarray, shape ``(n_bootstrap, n_cols)``
        Raw bootstrap distribution.  Useful for additional inference
        (e.g. p-values via :func:`_bootstrap_pvalue`) without resampling
        twice.
    """
    rng = rng or np.random.default_rng()
    trials_matrix = np.asarray(trials_matrix, dtype=float)
    if trials_matrix.ndim != 2:
        raise ValueError(
            f"trials_matrix must be 2D, got shape {trials_matrix.shape}"
        )
    n_trials, n_cols = trials_matrix.shape

    if n_trials == 0 or n_cols == 0:
        empty = np.zeros(n_cols)
        return empty, empty.copy(), empty.copy(), np.zeros((n_bootstrap, n_cols))

    point_estimates = np.asarray(statistic_fn(trials_matrix), dtype=float)
    if point_estimates.shape != (n_cols,):
        raise ValueError(
            "statistic_fn must return a vector of length n_cols, "
            f"got shape {point_estimates.shape}"
        )

    boot_stats = np.empty((n_bootstrap, n_cols), dtype=float)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n_trials, size=n_trials)
        boot_stats[b] = statistic_fn(trials_matrix[idx])

    alpha = 1.0 - confidence
    ci_lower = np.percentile(boot_stats, 100.0 * alpha / 2.0, axis=0)
    ci_upper = np.percentile(boot_stats, 100.0 * (1.0 - alpha / 2.0), axis=0)

    return point_estimates, ci_lower, ci_upper, boot_stats


def column_pvalues_from_bootstrap(
    boot_stats: np.ndarray,
    point_estimates: np.ndarray,
    null_value: float = 0.0,
) -> np.ndarray:
    """Two-sided per-column p-values from a precomputed bootstrap matrix.

    Reuses the bootstrap distribution produced by
    :func:`bootstrap_matrix_statistics` to avoid resampling twice.  Each
    column is tested independently against ``null_value``.

    Parameters
    ----------
    boot_stats : np.ndarray, shape ``(n_bootstrap, n_cols)``
        Bootstrap distribution returned by
        :func:`bootstrap_matrix_statistics`.
    point_estimates : np.ndarray, shape ``(n_cols,)``
        Observed column statistics.
    null_value : float, optional
        Hypothesised statistic value under H0 (default ``0.0``).

    Returns
    -------
    p_values : np.ndarray, shape ``(n_cols,)``
        Two-sided p-values, floored at ``1 / n_bootstrap``.
    """
    boot_stats = np.asarray(boot_stats, dtype=float)
    point_estimates = np.asarray(point_estimates, dtype=float)
    n_bootstrap, n_cols = boot_stats.shape
    if n_bootstrap == 0:
        return np.ones(n_cols)

    p_values = np.empty(n_cols)
    for c in range(n_cols):
        p_values[c] = _bootstrap_pvalue(
            boot_stats[:, c], float(point_estimates[c]), null_value
        )
    return p_values


# ---------------------------------------------------------------------------
# Multiple-comparison correction
# ---------------------------------------------------------------------------


def benjamini_hochberg(
    p_values: np.ndarray,
    fdr: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg FDR correction for multiple comparisons.

    Adjusts a vector of raw p-values to control the expected false
    discovery rate at level ``fdr``.

    Parameters
    ----------
    p_values : np.ndarray
        1D array of raw p-values in ``[0, 1]``.
    fdr : float, optional
        Target false discovery rate (default ``0.05``).

    Returns
    -------
    q_values : np.ndarray
        BH-adjusted p-values (q-values).  Smallest q-value across the
        family is at most ``1.0``.
    rejected : np.ndarray of bool
        Boolean mask marking columns whose ``q_value <= fdr``.

    Notes
    -----
    Implemented via the standard step-up procedure: sort p-values
    ascending, set ``raw_q_i = sorted_p_i * n / i``, then enforce
    monotonicity from the largest rank downwards by taking the running
    minimum.  Order of the input is preserved in the output.
    """
    p = np.asarray(p_values, dtype=float).ravel()
    n = p.size
    if n == 0:
        return np.array([], dtype=float), np.array([], dtype=bool)
    if np.any((p < 0) | (p > 1)):
        raise ValueError("p_values must lie in [0, 1]")

    order = np.argsort(p)
    sorted_p = p[order]

    ranks = np.arange(1, n + 1, dtype=float)
    raw_q = sorted_p * n / ranks

    # Enforce monotonic non-decreasing q by taking running minimum from
    # the right (largest-rank end) towards the left.
    q_sorted = np.minimum.accumulate(raw_q[::-1])[::-1]
    q_sorted = np.minimum(q_sorted, 1.0)

    q_values = np.empty(n, dtype=float)
    q_values[order] = q_sorted

    rejected = q_values <= fdr
    return q_values, rejected


# ---------------------------------------------------------------------------
# Convenience helpers used by NoiseAttribution.with_statistics
# ---------------------------------------------------------------------------


def attribution_percentage(matrix: np.ndarray) -> np.ndarray:
    """Per-column attribution % from a ``(n_trials, n_cols)`` delta-F matrix.

    Implements the same convention as
    :class:`PRISM.engine.debugger.NoiseAttribution`: negative
    contributions (recovery) are clamped to zero, then the row-mean is
    normalised by the sum of clamped means.  Returned in percent.

    If the total positive contribution is below ``1e-12`` the function
    returns a zero vector rather than dividing by a tiny number.
    """
    matrix = np.asarray(matrix, dtype=float)
    if matrix.size == 0:
        return np.zeros(matrix.shape[1] if matrix.ndim == 2 else 0)
    mean_contrib = matrix.mean(axis=0)
    clamped = np.maximum(mean_contrib, 0.0)
    total = clamped.sum()
    if total <= 1e-12:
        return np.zeros_like(mean_contrib)
    return 100.0 * clamped / total


def recovery_rate(matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Per-column empirical probability of fidelity recovery.

    Defined as the fraction of trials in which a column's contribution
    is *negative* by more than ``eps``.  A column with high recovery
    rate but small mean contribution typically reflects noise jitter
    rather than a real systematic effect.
    """
    matrix = np.asarray(matrix, dtype=float)
    if matrix.size == 0:
        return np.zeros(matrix.shape[1] if matrix.ndim == 2 else 0)
    return np.mean(matrix < -eps, axis=0)
