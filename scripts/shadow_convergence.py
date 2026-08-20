"""Demonstrate Pauli-classical-shadow estimator convergence on a Bell state.

For the Bell state ``(|00> + |11>) / sqrt(2)`` the analytic Pauli
expectations are

    <X X> = +1,  <Y Y> = -1,  <Z Z> = +1,

with the off-diagonal correlators (XY, XZ, YX, YZ, ZX, ZY) all zero.
This script samples increasing numbers of shadow shots, estimates
the three sign-fixed correlators with their bootstrap standard
errors, and emits a single figure that demonstrates two textbook
properties of the protocol:

* The estimator converges to the analytic value as ``N`` grows.
* The standard error scales as ``1 / sqrt(N)``.

Outputs land under :file:`paper/summary/`:

* ``shadow_convergence.{pdf,png}``  -- two-panel figure
* ``shadow_convergence.csv``        -- ``(observable, N, mean, sem)``
* ``shadow_convergence.json``       -- experiment metadata + headline

Usage::

    python scripts/shadow_convergence.py
    python scripts/shadow_convergence.py --quick
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

from PRISM.engine.circuit import GateInstance, QuantumCircuit  # noqa: E402
from PRISM.engine.shadows import (  # noqa: E402
    estimate_pauli_string,
    take_pauli_shadows,
)
from PRISM.engine.simulator import Simulator  # noqa: E402
from PRISM.figures import save_figure, use_paper_style  # noqa: E402


DEFAULT_SHOT_GRID = (50, 100, 200, 400, 800, 1600, 3200, 6400)

# Three sign-fixed correlators with their analytic Bell-state values.
OBSERVABLES = [
    ("XX", +1.0, "#2563eb"),
    ("YY", -1.0, "#10b981"),
    ("ZZ", +1.0, "#f59e0b"),
]


def bell_state():
    qc = QuantumCircuit(num_qubits=2)
    qc.add_gate(GateInstance("H", [0], [], 0))
    qc.add_gate(GateInstance("CNOT", [0, 1], [], 1))
    return Simulator().run(qc, shots=0, seed=0).final_state


def run_convergence(shot_grid: tuple[int, ...], seed: int) -> list[dict]:
    """Run the convergence sweep, returning a list of one dict per
    ``(observable, N)`` pair."""
    rows: list[dict] = []
    sv = bell_state()

    for k, n_shots in enumerate(shot_grid):
        # One *fresh* shadow set per N so the points are statistically
        # independent (rather than nested subsets of a single big set,
        # which would correlate the points and underestimate variance).
        rng = np.random.default_rng(seed + k)
        t0 = time.perf_counter()
        shadows = take_pauli_shadows(sv, n_shots=n_shots, rng=rng)
        t_sample = time.perf_counter() - t0

        for obs, true_value, _color in OBSERVABLES:
            mean, sem = estimate_pauli_string(shadows, obs)
            rows.append({
                "observable": obs,
                "n_shots": n_shots,
                "mean_estimate": mean,
                "sem": sem,
                "true_value": true_value,
                "abs_error": abs(mean - true_value),
                "elapsed_s": t_sample,
            })
        print(
            f"  N={n_shots:5d}  XX={rows[-3]['mean_estimate']:+.3f}+/-{rows[-3]['sem']:.3f}  "
            f"YY={rows[-2]['mean_estimate']:+.3f}+/-{rows[-2]['sem']:.3f}  "
            f"ZZ={rows[-1]['mean_estimate']:+.3f}+/-{rows[-1]['sem']:.3f}  "
            f"({t_sample:5.2f}s)"
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


def render_figure(rows: list[dict], path_pdf: Path, path_png: Path) -> None:
    use_paper_style()

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(8, 6),
        gridspec_kw={"height_ratios": [3, 2], "hspace": 0.4},
    )

    # Top: estimate +/- SEM vs N.  Horizontal reference lines at the
    # analytic values.
    for obs, true_val, color in OBSERVABLES:
        sub = [r for r in rows if r["observable"] == obs]
        ns = np.array([r["n_shots"] for r in sub], dtype=float)
        means = np.array([r["mean_estimate"] for r in sub])
        sems = np.array([r["sem"] for r in sub])
        ax_top.errorbar(
            ns, means, yerr=sems,
            fmt="-o", color=color, capsize=3,
            label=f"<{obs[0]} {obs[1]}>",
            markersize=4, linewidth=1.5,
        )
        ax_top.axhline(true_val, color=color, alpha=0.3,
                       linestyle="--", linewidth=0.8)

    ax_top.set_xscale("log")
    ax_top.set_xlabel("shadow shots $N$ (log scale)")
    ax_top.set_ylabel("Pauli expectation estimate")
    ax_top.set_title(
        "Pauli classical shadow convergence on the Bell state",
        fontsize=11, fontweight="bold",
    )
    ax_top.legend(loc="best", fontsize=9, frameon=False)
    ax_top.grid(True, which="both", linestyle=":", alpha=0.6)

    # Bottom: SEM vs N on log-log axes; the 1/sqrt(N) line is overlaid.
    color_sem = "#475569"
    obs_for_sem = "XX"
    sub = [r for r in rows if r["observable"] == obs_for_sem]
    ns = np.array([r["n_shots"] for r in sub], dtype=float)
    sems = np.array([r["sem"] for r in sub])

    ax_bot.set_xscale("log")
    ax_bot.set_yscale("log")
    ax_bot.plot(ns, sems, "-o", color=color_sem, markersize=4, linewidth=1.5,
                label=f"SEM on <{obs_for_sem[0]} {obs_for_sem[1]}>")

    # Reference 1/sqrt(N) curve, anchored at the smallest-N SEM.
    ref = sems[0] * np.sqrt(ns[0]) / np.sqrt(ns)
    ax_bot.plot(ns, ref, "--", color="#0f172a", linewidth=0.8, alpha=0.7,
                label=r"$\propto 1 / \sqrt{N}$ reference")

    ax_bot.set_xlabel("shadow shots $N$ (log scale)")
    ax_bot.set_ylabel("SEM (log scale)")
    ax_bot.legend(loc="best", fontsize=9, frameon=False)
    ax_bot.grid(True, which="both", linestyle=":", alpha=0.6)

    path_pdf.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, str(path_pdf))
    save_figure(fig, str(path_png))
    plt.close(fig)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Sample increasing numbers of Pauli classical shadows on a "
            "Bell state and chart estimator convergence."
        ),
    )
    p.add_argument(
        "--shot-grid", type=int, nargs="+",
        default=list(DEFAULT_SHOT_GRID),
    )
    p.add_argument("--seed", type=int, default=20260501)
    p.add_argument("--quick", action="store_true",
                   help="Smoke mode: shot_grid = (50, 200, 800).")
    p.add_argument("--output", type=Path,
                   default=_PROJECT_ROOT / "paper" / "summary")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.quick:
        shot_grid = (50, 200, 800)
    else:
        shot_grid = tuple(args.shot_grid)

    print(f"Pauli shadow convergence: shot grid = {list(shot_grid)}")
    rows = run_convergence(shot_grid, args.seed)

    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.output / "shadow_convergence.csv")
    write_json(rows, args.output / "shadow_convergence.json", meta={
        "kind": "shadow_convergence",
        "state": "bell",
        "shot_grid": list(shot_grid),
        "seed": args.seed,
    })
    render_figure(
        rows,
        args.output / "shadow_convergence.pdf",
        args.output / "shadow_convergence.png",
    )
    print()
    print(f"Figure : {args.output / 'shadow_convergence.pdf'}")
    print(f"CSV    : {args.output / 'shadow_convergence.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
