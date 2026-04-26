"""Generate the QEC 3-metric disagreement analysis for paper Section 6.7.

For each of the four codes (BitFlip, PhaseFlip, Steane, Shor) and a
sweep over physical depolarizing-error rates, we run
``analyze_metric_agreement`` to obtain the per-rate 4-way breakdown of
F-success vs Z-success outcomes, and render a stacked-bar figure where
each bar shows the fraction of trials in each agreement bin
(both_pass / both_fail / F_only / Z_only).

Outputs land under :file:`paper/summary/`:

* ``qec_disagreement.pdf``   -- 2x2 panel of the four codes
* ``qec_disagreement.png``   -- inline preview
* ``qec_disagreement.csv``   -- tidy ``(code, p, bin, count)`` table
* ``qec_disagreement.json``  -- experiment metadata + headline numbers

Usage::

    python scripts/qec_disagreement.py
    python scripts/qec_disagreement.py --quick
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from PRISM.engine.qec import (  # noqa: E402
    BitFlipCode,
    PhaseFlipCode,
    QECSimulator,
    Shor9Code,
    SteaneCode,
)
from PRISM.figures import use_paper_style  # noqa: E402


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


# Each entry: (display_name, factory) -- factory takes no args and
# returns an instance of the QEC code.
CODES: list[tuple[str, type]] = [
    ("BitFlip [3,1,1]", BitFlipCode),
    ("PhaseFlip [3,1,1]", PhaseFlipCode),
    ("Steane [[7,1,3]]", SteaneCode),
    ("Shor [[9,1,3]]", Shor9Code),
]

DEFAULT_PHYSICAL_RATES = (0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30)


# Stacked-bar palette: ordered to read intuitively (best -> worst).
COLOR_BOTH_PASS = "#10b981"   # emerald
COLOR_F_ONLY = "#f59e0b"      # amber (amplitude correct, phase wrong)
COLOR_Z_ONLY = "#3b82f6"      # blue (phase correct, amplitude weak)
COLOR_BOTH_FAIL = "#dc2626"   # red


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------


def run_all_codes(
    physical_rates: tuple[float, ...],
    n_trials: int,
    noise_type: str,
    seed: int,
) -> dict[str, list]:
    """Run ``analyze_metric_agreement`` for every code in :data:`CODES`.

    Returns a dict ``{code_name -> [MetricAgreement, ...]}``.
    """
    out: dict[str, list] = {}
    for k, (name, factory) in enumerate(CODES):
        sim = QECSimulator(factory())
        # Stable per-code seed so adding a code later does not change
        # the others' numbers.
        per_code_seed = seed + 1000 * k
        t0 = time.perf_counter()
        breakdowns = sim.analyze_metric_agreement(
            list(physical_rates),
            n_trials=n_trials,
            noise_type=noise_type,
            seed=per_code_seed,
        )
        elapsed = time.perf_counter() - t0
        out[name] = breakdowns
        # One-line summary per code: peak disagreement fraction.
        peak_disagreement = max(b.fraction_disagreement for b in breakdowns)
        peak_at = next(
            b.physical_rate for b in breakdowns
            if b.fraction_disagreement == peak_disagreement
        )
        print(
            f"  {name:<22s}  peak disagreement = "
            f"{peak_disagreement:5.2%} at p={peak_at:.3f}  "
            f"({elapsed:5.2f}s)"
        )
    return out


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def write_csv(
    out: dict[str, list], path: Path, n_trials: int, noise_type: str,
) -> None:
    """Tidy long-form CSV: one row per (code, rate, bin)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "code", "noise_type", "physical_rate", "n_trials",
            "bin", "count", "fraction",
        ])
        for code, breakdowns in out.items():
            for b in breakdowns:
                for bin_name, count in (
                    ("both_pass", b.n_both_pass),
                    ("both_fail", b.n_both_fail),
                    ("f_only", b.n_f_only),
                    ("z_only", b.n_z_only),
                ):
                    writer.writerow([
                        code, noise_type,
                        f"{b.physical_rate:.4f}", b.n_trials,
                        bin_name, count,
                        f"{count / b.n_trials:.4f}",
                    ])


def write_json(
    out: dict[str, list], path: Path, *,
    n_trials: int, noise_type: str, seed: int,
) -> None:
    payload = {
        "kind": "qec_disagreement",
        "n_trials": n_trials,
        "noise_type": noise_type,
        "seed": seed,
        "codes": {},
    }
    for code, breakdowns in out.items():
        payload["codes"][code] = [
            {
                "physical_rate": b.physical_rate,
                "n_trials": b.n_trials,
                "n_both_pass": b.n_both_pass,
                "n_both_fail": b.n_both_fail,
                "n_f_only": b.n_f_only,
                "n_z_only": b.n_z_only,
                "fraction_disagreement": b.fraction_disagreement,
            }
            for b in breakdowns
        ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def render_figure(
    out: dict[str, list], path_pdf: Path, path_png: Path,
    noise_type: str,
) -> None:
    use_paper_style()

    n_codes = len(out)
    fig, axes = plt.subplots(
        2, 2, figsize=(11, 8),
        sharex=True, sharey=True,
    )
    axes_flat = axes.flatten()

    for ax, (code, breakdowns) in zip(axes_flat, out.items()):
        rates = np.array([b.physical_rate for b in breakdowns])
        n_trials = breakdowns[0].n_trials

        bp = np.array([b.n_both_pass for b in breakdowns]) / n_trials
        fo = np.array([b.n_f_only for b in breakdowns]) / n_trials
        zo = np.array([b.n_z_only for b in breakdowns]) / n_trials
        bf = np.array([b.n_both_fail for b in breakdowns]) / n_trials

        # Stack bottom-to-top: success at the bottom, failure at top.
        bar_w = (rates[1] - rates[0]) * 0.7 if len(rates) > 1 else 0.02
        ax.bar(rates, bp, width=bar_w,
               color=COLOR_BOTH_PASS, edgecolor="#0f172a", linewidth=0.4,
               label="both pass (F+ Z+)")
        ax.bar(rates, fo, width=bar_w, bottom=bp,
               color=COLOR_F_ONLY, edgecolor="#0f172a", linewidth=0.4,
               label="F only (F+ Z-)")
        ax.bar(rates, zo, width=bar_w, bottom=bp + fo,
               color=COLOR_Z_ONLY, edgecolor="#0f172a", linewidth=0.4,
               label="Z only (F- Z+)")
        ax.bar(rates, bf, width=bar_w, bottom=bp + fo + zo,
               color=COLOR_BOTH_FAIL, edgecolor="#0f172a", linewidth=0.4,
               label="both fail (F- Z-)")

        ax.set_title(code, fontsize=11, fontweight="bold", loc="left")
        ax.set_ylim(0, 1.0)
        ax.grid(True, axis="y", linestyle=":", linewidth=0.6, alpha=0.7)
        ax.set_axisbelow(True)

    # Outer labels
    for ax in axes[-1, :]:
        ax.set_xlabel(f"physical {noise_type} rate")
    for ax in axes[:, 0]:
        ax.set_ylabel("fraction of trials")

    # One legend for the whole figure, placed below the plots.
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=4, fontsize=9, frameon=False,
    )

    fig.tight_layout(rect=(0, 0.04, 1, 1))

    path_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path_pdf), bbox_inches="tight", pad_inches=0.05)
    fig.savefig(str(path_png), dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run the F-vs-Z metric agreement analysis on every QEC "
            "code in PRISM and emit the stacked-bar figure plus tidy "
            "long-form CSV."
        ),
    )
    p.add_argument(
        "--rates", nargs="+", type=float,
        default=list(DEFAULT_PHYSICAL_RATES),
        help="Physical error rates to sweep.",
    )
    p.add_argument(
        "--n-trials", type=int, default=200,
        help="Trials per (code, rate) pair.",
    )
    p.add_argument(
        "--noise-type", default="depolarizing",
        choices=["bit_flip", "phase_flip", "depolarizing"],
        help="Physical noise channel.",
    )
    p.add_argument(
        "--seed", type=int, default=20260430,
    )
    p.add_argument(
        "--quick", action="store_true",
        help="Smoke mode: rates=[0.05, 0.15, 0.30], n_trials=60.",
    )
    p.add_argument(
        "--output", type=Path,
        default=_PROJECT_ROOT / "paper" / "summary",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.quick:
        rates = (0.05, 0.15, 0.30)
        n_trials = 60
    else:
        rates = tuple(args.rates)
        n_trials = args.n_trials

    print(
        f"QEC disagreement analysis: {len(CODES)} codes, "
        f"{len(rates)} rates, T={n_trials}, noise={args.noise_type}"
    )
    out = run_all_codes(rates, n_trials, args.noise_type, args.seed)

    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(
        out, args.output / "qec_disagreement.csv",
        n_trials=n_trials, noise_type=args.noise_type,
    )
    write_json(
        out, args.output / "qec_disagreement.json",
        n_trials=n_trials, noise_type=args.noise_type, seed=args.seed,
    )
    render_figure(
        out,
        args.output / "qec_disagreement.pdf",
        args.output / "qec_disagreement.png",
        noise_type=args.noise_type,
    )
    print()
    print(f"Figure : {args.output / 'qec_disagreement.pdf'}")
    print(f"CSV    : {args.output / 'qec_disagreement.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
