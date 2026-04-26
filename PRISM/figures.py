"""Publication-quality plotting helpers for PRISM noise attribution.

This module is **GUI-free** -- it imports only ``matplotlib`` and
operates on :class:`PRISM.engine.debugger.NoiseAttribution` objects
(with optional :class:`AttributionStatistics`).  Both the headless
benchmark scripts and the Qt panel layer can use it without dragging
each other in.

The exposed functions are designed to compose into a single
"attribution summary" figure but can also be used standalone in a
caller-supplied :class:`matplotlib.axes.Axes`.

Conventions
-----------
* Significance markers (``'*'``, ``'**'``, ``'***'``) follow the usual
  thresholds at ``q <= 0.05 / 0.01 / 0.001`` after Benjamini-Hochberg
  correction.
* Recovery columns (where ``mean(delta_F) < 0``) are coloured with the
  recovery palette and annotated with a small caret.
* All percent values are plotted on a 0-100 scale; CI errorbars use
  the bootstrap percentile interval at the configured confidence
  level (default 95%).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from PRISM.engine.debugger import NoiseAttribution


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AttributionPalette:
    """Colour palette used by the attribution plots.

    Defaults are tuned for both light and dark publication backgrounds
    and intentionally avoid colour-blind-unfriendly red/green pairings.
    """

    primary: str = "#2563eb"      # blue: significant contribution
    secondary: str = "#94a3b8"    # slate: non-significant contribution
    recovery: str = "#f59e0b"     # amber: fidelity recovery (delta_F < 0)
    error_bar: str = "#475569"    # dark slate: errorbar caps
    grid: str = "#cbd5e1"
    text: str = "#0f172a"


DEFAULT_PALETTE = AttributionPalette()


def _significance_stars(q_value: float) -> str:
    """Return ``'*'`` / ``'**'`` / ``'***'`` for the standard thresholds."""
    if q_value <= 0.001:
        return "***"
    if q_value <= 0.01:
        return "**"
    if q_value <= 0.05:
        return "*"
    return ""


def _column_labels(attr: "NoiseAttribution") -> list[str]:
    """One short label per column.

    Joins the gate-instance labels at each column with "+" and wraps
    over multiple lines if more than three gates share a column.
    """
    out: list[str] = []
    for col_idx, labels in enumerate(attr.gate_labels):
        if not labels:
            out.append(f"col {col_idx}")
            continue
        if len(labels) <= 3:
            out.append("\n+\n".join(labels))
        else:
            head = labels[:2]
            out.append("\n+\n".join(head) + f"\n+{len(labels) - 2} more")
    return out


# ---------------------------------------------------------------------------
# Single-axes plots
# ---------------------------------------------------------------------------


def plot_attribution_percent(
    attr: "NoiseAttribution",
    ax: "Axes | None" = None,
    palette: AttributionPalette = DEFAULT_PALETTE,
    show_significance: bool = True,
    annotate_recovery: bool = True,
) -> "Axes":
    """Bar chart of per-column attribution % with bootstrap CI errorbars.

    If the attribution carries :class:`AttributionStatistics`,
    significant columns (``q <= 0.05`` after BH-FDR) are shaded with
    the primary palette colour and annotated with stars; non-significant
    columns are shaded with the secondary colour.  Recovery columns
    (where the mean ``delta_F`` was negative) are shaded with the
    recovery colour and given a small caret.

    Parameters
    ----------
    attr : NoiseAttribution
        Attribution with optional ``statistics``.
    ax : matplotlib.axes.Axes, optional
        Target axes.  If ``None`` a new figure/axes are created.
    palette : AttributionPalette, optional
    show_significance : bool, optional
        Annotate significant bars with stars when statistics are
        available.  No-op for stat-less attributions.
    annotate_recovery : bool, optional
        Tag recovery bars with a small caret (``"^"``) above the bar.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    pct = np.asarray(attr.column_attribution_pct, dtype=float)
    n_cols = pct.size
    xs = np.arange(n_cols)

    # Choose per-bar colour based on recovery / significance flags.
    colours: list[str] = []
    for i in range(n_cols):
        if attr.is_recovery[i]:
            colours.append(palette.recovery)
        elif attr.statistics and attr.statistics.column_significant[i]:
            colours.append(palette.primary)
        else:
            colours.append(palette.secondary)

    # CI errorbars only when statistics are available
    yerr_lower = yerr_upper = None
    if attr.statistics is not None:
        lo = np.asarray(attr.statistics.attribution_pct_ci_lower, dtype=float)
        hi = np.asarray(attr.statistics.attribution_pct_ci_upper, dtype=float)
        yerr_lower = np.maximum(pct - lo, 0.0)
        yerr_upper = np.maximum(hi - pct, 0.0)

    bars = ax.bar(xs, pct, color=colours, edgecolor=palette.text,
                  linewidth=0.6, zorder=2)
    if yerr_lower is not None:
        ax.errorbar(
            xs, pct,
            yerr=np.vstack([yerr_lower, yerr_upper]),
            fmt="none", ecolor=palette.error_bar, capsize=3, capthick=0.8,
            elinewidth=0.8, zorder=3,
        )

    # Star annotations and recovery carets
    if attr.statistics is not None and show_significance:
        for i, bar in enumerate(bars):
            stars = _significance_stars(attr.statistics.delta_fidelity_q_value[i])
            if not stars:
                continue
            top = pct[i] + (yerr_upper[i] if yerr_upper is not None else 0.0)
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                top + 1.0,
                stars,
                ha="center", va="bottom",
                fontsize=10, fontweight="bold",
                color=palette.text, zorder=4,
            )

    if annotate_recovery:
        for i, bar in enumerate(bars):
            if not attr.is_recovery[i]:
                continue
            top = pct[i] + (yerr_upper[i] if yerr_upper is not None else 0.0)
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                top + 0.8,
                "^",
                ha="center", va="bottom",
                fontsize=10, color=palette.recovery, zorder=4,
            )

    ax.set_xticks(xs)
    ax.set_xticklabels(_column_labels(attr), fontsize=8)
    ax.set_ylabel("Attribution (%)", color=palette.text)
    ax.set_xlabel("Circuit column", color=palette.text)
    ax.grid(True, axis="y", linestyle=":", linewidth=0.6,
            color=palette.grid, zorder=0)
    ax.set_axisbelow(True)
    return ax


def plot_delta_fidelity(
    attr: "NoiseAttribution",
    ax: "Axes | None" = None,
    palette: AttributionPalette = DEFAULT_PALETTE,
) -> "Axes":
    """Per-column mean delta-F with a shaded bootstrap CI band.

    Falls back to a mean +/- 1 std band when no
    :class:`AttributionStatistics` is attached.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 3))

    mean = np.asarray(attr.delta_fidelity, dtype=float)
    xs = np.arange(mean.size)

    if attr.statistics is not None:
        lo = np.asarray(attr.statistics.delta_fidelity_ci_lower, dtype=float)
        hi = np.asarray(attr.statistics.delta_fidelity_ci_upper, dtype=float)
        band_label = (
            f"{int(round(attr.statistics.confidence * 100))}% bootstrap CI"
        )
    else:
        std = np.asarray(attr.delta_fidelity_std, dtype=float)
        lo = mean - std
        hi = mean + std
        band_label = "+/- 1 std"

    ax.fill_between(xs, lo, hi, color=palette.primary, alpha=0.18,
                    label=band_label, zorder=1)
    ax.plot(xs, mean, "-o", color=palette.primary,
            markersize=4, linewidth=1.5, label="mean delta_F", zorder=2)
    ax.axhline(0.0, color=palette.text, linewidth=0.6, linestyle="--", zorder=0)

    ax.set_xticks(xs)
    ax.set_xticklabels(_column_labels(attr), fontsize=8)
    ax.set_ylabel("delta_F per column", color=palette.text)
    ax.set_xlabel("Circuit column", color=palette.text)
    ax.grid(True, axis="y", linestyle=":", linewidth=0.6,
            color=palette.grid, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="best", fontsize=8, frameon=False)
    return ax


def plot_recovery_rate(
    attr: "NoiseAttribution",
    ax: "Axes | None" = None,
    palette: AttributionPalette = DEFAULT_PALETTE,
) -> "Axes":
    """Per-column empirical recovery rate ``P(delta_F < 0)`` with CI."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 3))

    if attr.statistics is None:
        # Without statistics we can show a binary indicator only.
        rates = np.array(
            [1.0 if r else 0.0 for r in attr.is_recovery], dtype=float
        )
        lo = hi = rates
    else:
        rates = np.asarray(attr.statistics.recovery_rate, dtype=float)
        lo = np.asarray(attr.statistics.recovery_rate_ci_lower, dtype=float)
        hi = np.asarray(attr.statistics.recovery_rate_ci_upper, dtype=float)

    xs = np.arange(rates.size)
    yerr = np.vstack([np.maximum(rates - lo, 0.0),
                      np.maximum(hi - rates, 0.0)])

    ax.bar(xs, rates, color=palette.recovery, edgecolor=palette.text,
           linewidth=0.6, alpha=0.85, zorder=2)
    if attr.statistics is not None:
        ax.errorbar(
            xs, rates, yerr=yerr,
            fmt="none", ecolor=palette.error_bar, capsize=3, capthick=0.8,
            elinewidth=0.8, zorder=3,
        )
    ax.axhline(0.5, color=palette.text, linewidth=0.6, linestyle="--", zorder=0)

    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(xs)
    ax.set_xticklabels(_column_labels(attr), fontsize=8)
    ax.set_ylabel("Recovery rate", color=palette.text)
    ax.set_xlabel("Circuit column", color=palette.text)
    ax.grid(True, axis="y", linestyle=":", linewidth=0.6,
            color=palette.grid, zorder=0)
    ax.set_axisbelow(True)
    return ax


# ---------------------------------------------------------------------------
# Composite figure
# ---------------------------------------------------------------------------


def attribution_summary_figure(
    attr: "NoiseAttribution",
    title: str = "",
    palette: AttributionPalette = DEFAULT_PALETTE,
    figsize: tuple[float, float] = (10.0, 8.0),
) -> "Figure":
    """Three-panel summary of an attribution result.

    Layout
    ------
    Top:    attribution % bar chart with bootstrap CI errorbars.
    Middle: per-column delta-F line with CI band.
    Bottom: recovery-rate bars with CI.

    Parameters
    ----------
    attr : NoiseAttribution
    title : str
        Suptitle for the figure.
    palette : AttributionPalette
    figsize : tuple[float, float]

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, (ax_top, ax_mid, ax_bot) = plt.subplots(
        3, 1,
        figsize=figsize,
        gridspec_kw={"height_ratios": [3, 2, 2], "hspace": 0.55},
    )

    plot_attribution_percent(attr, ax=ax_top, palette=palette)
    plot_delta_fidelity(attr, ax=ax_mid, palette=palette)
    plot_recovery_rate(attr, ax=ax_bot, palette=palette)

    if title:
        fig.suptitle(title, y=0.995, fontsize=12, fontweight="bold")

    # Hide x-tick labels on top two panels to reduce visual noise --
    # the bottom panel already labels the columns.
    for ax in (ax_top, ax_mid):
        ax.set_xlabel("")
        ax.tick_params(axis="x", labelbottom=False)

    return fig


# ---------------------------------------------------------------------------
# I/O helper
# ---------------------------------------------------------------------------


def save_figure(
    fig: "Figure",
    path: str,
    dpi: int = 300,
) -> None:
    """Save a figure with sensible publication defaults.

    Wraps :pymeth:`Figure.savefig` with ``bbox_inches='tight'``,
    ``pad_inches=0.05`` and the supplied DPI.  The output format is
    inferred from the file extension, so ``.pdf`` / ``.png`` / ``.svg``
    all work.

    Resolution policy
    -----------------
    * **PDF / SVG** are vector formats; the ``dpi`` argument has no
      effect on their visual fidelity, but is still used to size any
      embedded raster elements (we do not generate any in PRISM, so
      the output is fully vector and scales infinitely).
    * **PNG** is rasterised at the given DPI.  300 DPI -- the paper
      default -- meets typical journal print requirements and is
      visibly sharper than the on-screen 200 DPI default that earlier
      revisions used.
    """
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.05)


def use_paper_style() -> None:
    """Apply matplotlib rcParams suitable for paper figures.

    Idempotent.  Call once at the top of a figure-generation script;
    individual figures can still override with ``plt.rc_context``.
    """
    matplotlib.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.format": "pdf",
        "pdf.fonttype": 42,   # TrueType fonts in PDF (avoid Type-3 issues)
        "ps.fonttype": 42,
    })
