"""Smoke tests for :mod:`PRISM.figures`.

The plotting layer cannot be exercised in a headful CI environment, so
we run all tests against the matplotlib ``Agg`` backend.  We verify
that:

* Each public plotting function produces a valid Axes / Figure.
* The composite ``attribution_summary_figure`` runs without raising
  for both stat-bearing and stat-less attributions.
* Saving to PDF / PNG produces a non-empty file.
* Significance markers are emitted for highly-significant columns.
"""

from __future__ import annotations

# Headless backend BEFORE importing pyplot.
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from PRISM.engine.circuit import GateInstance, QuantumCircuit
from PRISM.engine.debugger import CircuitDebugger
from PRISM.engine.noise import BitFlipNoise, DepolarizingNoise, NoiseModel
from PRISM.figures import (
    AttributionPalette,
    DEFAULT_PALETTE,
    attribution_summary_figure,
    plot_attribution_percent,
    plot_delta_fidelity,
    plot_recovery_rate,
    save_figure,
    use_paper_style,
)


# ---------------------------------------------------------------------------
# Fixture: small attribution that always has a clear signal
# ---------------------------------------------------------------------------


def _bell_circuit() -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits=2)
    qc.add_gate(GateInstance("H", [0], [], 0))
    qc.add_gate(GateInstance("CNOT", [0, 1], [], 1))
    return qc


def _strong_noise() -> NoiseModel:
    """Loud depolarizing channel chosen to make every Bell-circuit
    column reliably register as significant under FDR(0.05) given the
    fixture's bootstrap budget."""
    nm = NoiseModel()
    nm.add_global_noise(DepolarizingNoise(0.20))
    return nm


@pytest.fixture
def attribution_with_stats():
    """Bell-state attribution with statistics populated.

    Trials and bootstrap budget are sized so that, at 20% depolarizing
    noise, both columns clear FDR(0.05) deterministically under the
    fixed seed -- the significance-star plotting test depends on this.
    """
    return CircuitDebugger().compute_noise_attribution_with_statistics(
        _bell_circuit(),
        _strong_noise(),
        n_trials=120,
        n_bootstrap=500,
        seed=2024,
    )


@pytest.fixture
def attribution_no_stats():
    """Same circuit but the cheap attribution method (no stats)."""
    return CircuitDebugger().compute_noise_attribution(
        _bell_circuit(),
        _strong_noise(),
        n_trials=120,
        seed=2024,
    )


# ---------------------------------------------------------------------------
# plot_attribution_percent
# ---------------------------------------------------------------------------


class TestPlotAttributionPercent:

    def test_returns_axes(self, attribution_with_stats):
        ax = plot_attribution_percent(attribution_with_stats)
        assert ax is not None
        plt.close(ax.figure)

    def test_emits_one_bar_per_column(self, attribution_with_stats):
        fig, ax = plt.subplots()
        plot_attribution_percent(attribution_with_stats, ax=ax)
        bars = [
            p for p in ax.patches
            if p.__class__.__name__ == "Rectangle"
        ]
        # There may be tick / spine rectangles; filter to bar-coloured ones.
        assert len(bars) >= len(attribution_with_stats.delta_fidelity)
        plt.close(fig)

    def test_works_without_statistics(self, attribution_no_stats):
        """Plotting must not crash for the cheap attribution type."""
        fig, ax = plt.subplots()
        plot_attribution_percent(attribution_no_stats, ax=ax)
        plt.close(fig)

    def test_significance_stars_emitted(self, attribution_with_stats):
        """A heavily-noised Bell state should produce at least one
        column whose contribution is significant after FDR
        correction; the plot should annotate that column with stars."""
        fig, ax = plt.subplots()
        plot_attribution_percent(attribution_with_stats, ax=ax)
        star_glyphs = {"*", "**", "***"}
        text_strings = {t.get_text() for t in ax.texts}
        assert text_strings & star_glyphs, (
            f"No significance stars in {text_strings}"
        )
        plt.close(fig)


# ---------------------------------------------------------------------------
# plot_delta_fidelity
# ---------------------------------------------------------------------------


class TestPlotDeltaFidelity:

    def test_returns_axes(self, attribution_with_stats):
        ax = plot_delta_fidelity(attribution_with_stats)
        assert ax is not None
        plt.close(ax.figure)

    def test_falls_back_to_std_band_without_stats(self, attribution_no_stats):
        """The plot should still render using +/- 1 std fallback."""
        fig, ax = plt.subplots()
        plot_delta_fidelity(attribution_no_stats, ax=ax)
        # Legend label should reference the std fallback
        legend = ax.get_legend()
        labels = [t.get_text() for t in legend.get_texts()]
        assert any("std" in lbl for lbl in labels)
        plt.close(fig)

    def test_uses_bootstrap_label_with_stats(self, attribution_with_stats):
        fig, ax = plt.subplots()
        plot_delta_fidelity(attribution_with_stats, ax=ax)
        legend = ax.get_legend()
        labels = [t.get_text() for t in legend.get_texts()]
        assert any("bootstrap" in lbl.lower() for lbl in labels)
        plt.close(fig)


# ---------------------------------------------------------------------------
# plot_recovery_rate
# ---------------------------------------------------------------------------


class TestPlotRecoveryRate:

    def test_returns_axes(self, attribution_with_stats):
        ax = plot_recovery_rate(attribution_with_stats)
        assert ax is not None
        plt.close(ax.figure)

    def test_y_axis_bounded_to_unit_interval(self, attribution_with_stats):
        fig, ax = plt.subplots()
        plot_recovery_rate(attribution_with_stats, ax=ax)
        ymin, ymax = ax.get_ylim()
        assert ymin == pytest.approx(0.0, abs=1e-9)
        assert ymax == pytest.approx(1.0, abs=1e-9)
        plt.close(fig)


# ---------------------------------------------------------------------------
# attribution_summary_figure
# ---------------------------------------------------------------------------


class TestAttributionSummaryFigure:

    def test_three_panels(self, attribution_with_stats):
        fig = attribution_summary_figure(attribution_with_stats, title="Bell")
        assert len(fig.axes) == 3
        plt.close(fig)

    def test_runs_without_stats(self, attribution_no_stats):
        fig = attribution_summary_figure(attribution_no_stats)
        assert len(fig.axes) == 3
        plt.close(fig)

    def test_save_pdf(self, tmp_path, attribution_with_stats):
        fig = attribution_summary_figure(attribution_with_stats, title="Bell")
        out = tmp_path / "out.pdf"
        save_figure(fig, str(out))
        assert out.exists() and out.stat().st_size > 1000
        plt.close(fig)

    def test_save_png(self, tmp_path, attribution_with_stats):
        fig = attribution_summary_figure(attribution_with_stats, title="Bell")
        out = tmp_path / "out.png"
        save_figure(fig, str(out))
        assert out.exists() and out.stat().st_size > 1000
        plt.close(fig)


# ---------------------------------------------------------------------------
# Style helpers
# ---------------------------------------------------------------------------


def test_use_paper_style_idempotent():
    """Repeated calls should not raise and should leave rcParams stable."""
    use_paper_style()
    snapshot = dict(matplotlib.rcParams)
    use_paper_style()
    # Equality on the whole dict is brittle; sample a few keys instead.
    for key in ("font.size", "pdf.fonttype", "axes.spines.top"):
        assert matplotlib.rcParams[key] == snapshot[key]


def test_palette_is_dataclass_and_hashable():
    p = AttributionPalette()
    # Frozen dataclass -> hashable
    assert hash(p) == hash(DEFAULT_PALETTE)
