"""Sweep trial budget ``T`` on a representative circuit / noise pair and
chart how attribution stabilises.

Defends the choice of ``T = 120`` for the main figure suite by showing
that increasing ``T`` past the chosen budget does not flip column
significance flags or move the dominant column's q-value beyond noise
floor.

We track three quantities per ``T``:

* number of FDR-significant columns
* dominant column's q-value
* dominant column's attribution percentage with its 95% bootstrap CI

Outputs:
``paper/summary/trial_convergence.{pdf,png,csv,json}``.

Usage::

    python scripts/trial_convergence.py
    python scripts/trial_convergence.py --quick
    python scripts/trial_convergence.py --circuit qaoa_maxcut --noise depolarizing
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
from PRISM.figures import save_figure, use_paper_style  # noqa: E402

sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))
from generate_attribution_figures import (  # noqa: E402
    CIRCUITS, CIRCUIT_TITLES, NOISES, NOISE_TITLES,
)


DEFAULT_TRIAL_GRID = (20, 40, 80, 120, 200, 400)


def run_convergence(
    circuit_name: str,
    noise_name: str,
    trial_grid: tuple[int, ...],
    n_bootstrap: int,
    seed_base: int,
) -> list[dict]:
    if circuit_name not in CIRCUITS:
        raise ValueError(f"Unknown circuit {circuit_name!r}")
    if noise_name not in NOISES:
        raise ValueError(f"Unknown noise {noise_name!r}")

    circuit_factory = CIRCUITS[circuit_name]
    noise_factory = NOISES[noise_name]
    debugger = CircuitDebugger()

    rows: list[dict] = []
    for k, T in enumerate(trial_grid):
        circuit = circuit_factory()
        nm = noise_factory()
        seed = seed_base + 100 * k

        t0 = time.perf_counter()
        attr = debugger.compute_noise_attribution_with_statistics(
            circuit, nm,
            n_trials=T,
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
        elapsed = time.perf_counter() - t0

        stats = attr.statistics
        # Dominant column = largest attribution percentage.
        pct = np.array(attr.column_attribution_pct)
        dom_idx = int(np.argmax(pct))
        rows.append({
            "T": T,
            "n_significant": int(sum(stats.column_significant)),
            "dominant_column_idx": dom_idx,
            "dominant_column_attribution_pct": float(pct[dom_idx]),
            "dominant_column_attribution_ci_lower":
                float(stats.attribution_pct_ci_lower[dom_idx]),
            "dominant_column_attribution_ci_upper":
                float(stats.attribution_pct_ci_upper[dom_idx]),
            "dominant_column_q_value":
                float(stats.delta_fidelity_q_value[dom_idx]),
            "dominant_column_label":
                attr.gate_labels[dom_idx][0]
                if attr.gate_labels[dom_idx] else f"col {dom_idx}",
            "elapsed_seconds": elapsed,
            "seed": seed,
        })
        print(
            f"  T={T:4d}  sig={rows[-1]['n_significant']:2d}  "
            f"dom_pct={rows[-1]['dominant_column_attribution_pct']:5.1f}%  "
            f"q={rows[-1]['dominant_column_q_value']:7.4f}  "
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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"meta": meta, "rows": rows}, indent=2),
        encoding="utf-8",
    )


def render_figure(
    rows: list[dict],
    circuit_label: str,
    noise_label: str,
    out_pdf: Path,
    out_png: Path,
    headline_T: int = 120,
) -> None:
    use_paper_style()

    Ts = np.array([r["T"] for r in rows], dtype=float)
    n_sig = np.array([r["n_significant"] for r in rows])
    pct = np.array([r["dominant_column_attribution_pct"] for r in rows])
    pct_lo = np.array([r["dominant_column_attribution_ci_lower"] for r in rows])
    pct_hi = np.array([r["dominant_column_attribution_ci_upper"] for r in rows])
    q = np.array([r["dominant_column_q_value"] for r in rows])

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(8, 6),
        gridspec_kw={"height_ratios": [3, 2], "hspace": 0.4},
    )

    # Top: dominant column attribution % with bootstrap CI band, plus
    # significant-column count overlay on right axis.
    color_attr = "#2563eb"
    color_sig = "#10b981"

    ax_top.set_xscale("log")
    ax_top.fill_between(
        Ts, pct_lo, pct_hi,
        color=color_attr, alpha=0.18,
        label="95% bootstrap CI",
    )
    ax_top.plot(Ts, pct, "-o", color=color_attr,
                markersize=4, linewidth=1.5,
                label="dominant column $A_i$")
    ax_top.set_ylabel("dominant column $A_i$ (%)", color=color_attr)
    ax_top.tick_params(axis="y", labelcolor=color_attr)
    ax_top.legend(loc="lower right", fontsize=8, frameon=False)

    ax_top_r = ax_top.twinx()
    ax_top_r.plot(Ts, n_sig, "-s", color=color_sig,
                  markersize=4, linewidth=1.2)
    ax_top_r.set_ylabel("# FDR-significant columns", color=color_sig)
    ax_top_r.tick_params(axis="y", labelcolor=color_sig)
    ax_top_r.spines["right"].set_visible(True)

    ax_top.set_title(
        f"Trial-budget convergence: {circuit_label} + {noise_label}",
        fontsize=11, fontweight="bold",
    )
    ax_top.grid(True, which="both", linestyle=":", alpha=0.6)

    # Bottom: dominant column q-value (log scale) -- the cleanest single
    # number for "how confidently is this column flagged".
    color_q = "#dc2626"
    ax_bot.set_xscale("log")
    ax_bot.set_yscale("log")
    ax_bot.plot(Ts, q, "-D", color=color_q,
                markersize=4, linewidth=1.5)
    ax_bot.axhline(0.05, color="#64748b", linestyle="--",
                   linewidth=0.8, alpha=0.7,
                   label="FDR threshold (q = 0.05)")
    ax_bot.set_xlabel("trials $T$ (log scale)")
    ax_bot.set_ylabel("dominant col q-value (log)")
    ax_bot.legend(loc="best", fontsize=8, frameon=False)
    ax_bot.grid(True, which="both", linestyle=":", alpha=0.6)

    for ax in (ax_top, ax_bot):
        ax.axvline(headline_T, color="#64748b", linestyle="--",
                   linewidth=0.8, alpha=0.7)
    ax_bot.text(
        headline_T, ax_bot.get_ylim()[0] * 1.5,
        "  paper default",
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
            "Sweep the trial budget T on a representative attribution "
            "and chart how the dominant column's attribution percentage "
            "and q-value stabilise as T grows."
        ),
    )
    p.add_argument("--circuit", default="qaoa_maxcut",
                   choices=sorted(CIRCUITS))
    p.add_argument("--noise", default="depolarizing",
                   choices=sorted(NOISES))
    p.add_argument("--trial-grid", type=int, nargs="+",
                   default=list(DEFAULT_TRIAL_GRID),
                   help="Trial counts to evaluate (default: %(default)s).")
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=20260428)
    p.add_argument("--quick", action="store_true",
                   help="Smoke mode: trial_grid = (20, 40, 80, 120), B = 200.")
    p.add_argument("--output", type=Path,
                   default=_PROJECT_ROOT / "paper" / "summary")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.quick:
        trial_grid = (20, 40, 80, 120)
        n_bootstrap = 200
    else:
        trial_grid = tuple(args.trial_grid)
        n_bootstrap = args.n_bootstrap

    print(
        f"Trial-budget convergence: {args.circuit} + {args.noise}, "
        f"B={n_bootstrap}, grid={list(trial_grid)}"
    )
    rows = run_convergence(
        args.circuit, args.noise, trial_grid, n_bootstrap, args.seed,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.output / "trial_convergence.csv")
    write_json(rows, args.output / "trial_convergence.json", meta={
        "circuit": args.circuit,
        "noise": args.noise,
        "n_bootstrap": n_bootstrap,
        "seed_base": args.seed,
    })
    render_figure(
        rows,
        circuit_label=CIRCUIT_TITLES.get(args.circuit, args.circuit),
        noise_label=NOISE_TITLES.get(args.noise, args.noise),
        out_pdf=args.output / "trial_convergence.pdf",
        out_png=args.output / "trial_convergence.png",
        headline_T=120,
    )
    print()
    print(f"Figure : {args.output / 'trial_convergence.pdf'}")
    print(f"CSV    : {args.output / 'trial_convergence.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
