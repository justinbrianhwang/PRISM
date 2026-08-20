"""Sweep noise strength on a representative circuit and chart the
attribution methodology's response.

We run :pymeth:`compute_noise_attribution_with_statistics` on
QAOA(C_4) under global depolarizing noise at seven probabilities
spanning two orders of magnitude (0.005 - 0.20) and record three
quantities per setting:

* total fidelity loss ``g_L``
* number of FDR-significant columns
* mean attribution percentage of FDR-significant columns

The resulting two-panel figure goes into Section 6 of the paper to
defend two claims:

1. The methodology degrades gracefully across the noise spectrum --
   even at ``p = 0.005`` the bootstrap is wide enough to suppress
   spurious significance, and at ``p = 0.20`` the FDR control still
   limits the false-discovery proportion.
2. The choice of ``p = 0.05`` for the main figure suite is well
   inside the regime where attribution is informative (every column
   has a measurable signal) without saturating fidelity.

Usage::

    python scripts/strength_sweep.py
    python scripts/strength_sweep.py --quick      # smoke run (fewer trials)
    python scripts/strength_sweep.py --circuit qft4

Outputs land under :file:`paper/summary/strength_sweep.{pdf,png,csv,json}`.
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

from PRISM.engine.debugger import CircuitDebugger  # noqa: E402
from PRISM.engine.noise import (  # noqa: E402
    DepolarizingNoise,
    NoiseModel,
)
from PRISM.figures import save_figure, use_paper_style  # noqa: E402

# Import the same factories the figure-generation script uses so the
# circuit definitions stay in lock-step.
sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))
from generate_attribution_figures import CIRCUITS, CIRCUIT_TITLES  # noqa: E402


DEFAULT_PROBABILITIES = (0.005, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20)


def run_sweep(
    circuit_name: str,
    probabilities: tuple[float, ...],
    n_trials: int,
    n_bootstrap: int,
    seed_base: int,
) -> list[dict]:
    """Run attribution at each ``p`` and return one summary dict per
    probability."""
    if circuit_name not in CIRCUITS:
        raise ValueError(
            f"Unknown circuit {circuit_name!r}; choose from "
            f"{sorted(CIRCUITS)}"
        )
    circuit_factory = CIRCUITS[circuit_name]
    debugger = CircuitDebugger()
    rows: list[dict] = []

    for k, p in enumerate(probabilities):
        # Fresh circuit + noise model per setting so cross-pair RNG
        # state never leaks.
        circuit = circuit_factory()
        nm = NoiseModel()
        nm.add_global_noise(DepolarizingNoise(p))

        seed = seed_base + 100 * k
        t0 = time.perf_counter()
        attr = debugger.compute_noise_attribution_with_statistics(
            circuit, nm,
            n_trials=n_trials,
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
        elapsed = time.perf_counter() - t0

        n_cols = len(attr.delta_fidelity)
        sig_idx = [i for i, s in enumerate(attr.statistics.column_significant) if s]
        n_sig = len(sig_idx)
        sig_attr_mean = (
            float(np.mean([attr.column_attribution_pct[i] for i in sig_idx]))
            if sig_idx else 0.0
        )
        sig_attr_max = (
            float(np.max([attr.column_attribution_pct[i] for i in sig_idx]))
            if sig_idx else 0.0
        )

        rows.append({
            "p": p,
            "n_columns": n_cols,
            "n_significant": n_sig,
            "fraction_significant": n_sig / n_cols if n_cols else 0.0,
            "total_fidelity_loss": float(attr.total_fidelity_loss),
            "max_attribution_pct": float(max(attr.column_attribution_pct)),
            "max_attribution_pct_significant": sig_attr_max,
            "mean_attribution_pct_significant": sig_attr_mean,
            "elapsed_seconds": elapsed,
            "seed": seed,
        })
        print(
            f"  p={p:5.3f}  cols={n_cols:2d}  sig={n_sig:2d}  "
            f"g_L={attr.total_fidelity_loss:.4f}  "
            f"max(A_sig)={sig_attr_max:5.1f}%  "
            f"({elapsed:5.2f}s)"
        )
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        for r in rows:
            writer.writerow({
                k: (f"{v:.6f}" if isinstance(v, float) else v)
                for k, v in r.items()
            })


def write_json(rows: list[dict], path: Path, meta: dict) -> None:
    payload = {"meta": meta, "rows": rows}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def render_figure(
    rows: list[dict],
    circuit_label: str,
    out_pdf: Path,
    out_png: Path,
) -> None:
    use_paper_style()

    ps = np.array([r["p"] for r in rows])
    g_L = np.array([r["total_fidelity_loss"] for r in rows])
    n_sig = np.array([r["n_significant"] for r in rows])
    n_cols = np.array([r["n_columns"] for r in rows])
    max_attr = np.array([r["max_attribution_pct"] for r in rows])

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(8, 6),
        gridspec_kw={"height_ratios": [3, 2], "hspace": 0.4},
    )

    # Top: total fidelity loss + max attribution share, log x-axis.
    color_loss = "#2563eb"
    color_attr = "#f59e0b"

    ax_top.set_xscale("log")
    ax_top.plot(ps, g_L, "-o", color=color_loss, linewidth=1.5, markersize=4,
                label=r"total fidelity loss $g_L$")
    ax_top.set_ylabel(r"$g_L = 1 - F$", color=color_loss)
    ax_top.tick_params(axis="y", labelcolor=color_loss)
    ax_top.set_ylim(0, max(0.55, max(g_L) * 1.1))

    ax_top_r = ax_top.twinx()
    ax_top_r.plot(ps, max_attr, "-s", color=color_attr,
                  linewidth=1.2, markersize=4,
                  label=r"max $A_i$ (%)")
    ax_top_r.set_ylabel(r"max $A_i$ (%)", color=color_attr)
    ax_top_r.tick_params(axis="y", labelcolor=color_attr)
    ax_top_r.set_ylim(0, 100)
    ax_top_r.spines["right"].set_visible(True)

    ax_top.set_title(
        f"Strength sweep: {circuit_label} under depolarizing noise",
        fontsize=11, fontweight="bold",
    )
    ax_top.grid(True, which="both", linestyle=":", alpha=0.6)

    # Bottom: fraction of columns FDR-significant.
    color_sig = "#10b981"
    ax_bot.set_xscale("log")
    ax_bot.plot(
        ps, n_sig / n_cols, "-D",
        color=color_sig, linewidth=1.5, markersize=4,
    )
    ax_bot.set_xlabel(r"depolarizing $p$ (log scale)")
    ax_bot.set_ylabel("frac. FDR-significant cols")
    ax_bot.set_ylim(0, 1.05)
    ax_bot.grid(True, which="both", linestyle=":", alpha=0.6)
    # Mark the headline default p = 0.05 used in the main figure suite.
    for ax in (ax_top, ax_bot):
        ax.axvline(0.05, color="#64748b", linestyle="--", linewidth=0.8, alpha=0.7)
    ax_bot.text(
        0.05, 0.05, "  paper default",
        rotation=90, va="bottom", ha="left",
        color="#64748b", fontsize=8,
    )

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, str(out_pdf))
    save_figure(fig, str(out_png))
    plt.close(fig)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Sweep depolarizing-noise probability on a benchmark circuit "
            "and chart how attribution responds across two orders of "
            "magnitude of noise strength."
        ),
    )
    p.add_argument("--circuit", default="qaoa_maxcut",
                   choices=sorted(CIRCUITS),
                   help="Benchmark circuit to sweep on (default: QAOA(C_4)).")
    p.add_argument("--n-trials", type=int, default=120,
                   help="Trials per probability setting.")
    p.add_argument("--n-bootstrap", type=int, default=1000,
                   help="Bootstrap resamples per attribution.")
    p.add_argument("--seed", type=int, default=20260427,
                   help="Master seed; per-p seeds derive from this.")
    p.add_argument("--quick", action="store_true",
                   help="Smoke mode: n_trials=30, n_bootstrap=300.")
    p.add_argument("--probabilities", type=float, nargs="+",
                   default=list(DEFAULT_PROBABILITIES),
                   help="Override the default 0.005..0.20 sweep grid.")
    p.add_argument("--output", type=Path,
                   default=_PROJECT_ROOT / "paper" / "summary",
                   help="Output directory for the figure / CSV / JSON.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    n_trials = 30 if args.quick else args.n_trials
    n_bootstrap = 300 if args.quick else args.n_bootstrap

    print(
        f"Strength sweep on {args.circuit} "
        f"(T={n_trials}, B={n_bootstrap})  "
        f"probabilities={list(args.probabilities)}"
    )
    rows = run_sweep(
        args.circuit,
        tuple(args.probabilities),
        n_trials=n_trials,
        n_bootstrap=n_bootstrap,
        seed_base=args.seed,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.output / "strength_sweep.csv")
    write_json(rows, args.output / "strength_sweep.json", meta={
        "circuit": args.circuit,
        "n_trials": n_trials,
        "n_bootstrap": n_bootstrap,
        "seed_base": args.seed,
    })
    render_figure(
        rows,
        circuit_label=CIRCUIT_TITLES.get(args.circuit, args.circuit),
        out_pdf=args.output / "strength_sweep.pdf",
        out_png=args.output / "strength_sweep.png",
    )
    print()
    print(f"Figure : {args.output / 'strength_sweep.pdf'}")
    print(f"CSV    : {args.output / 'strength_sweep.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
