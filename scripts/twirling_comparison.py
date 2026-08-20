"""Generate the Pauli-twirling comparison figures for paper Section 6.6.

For each ``(circuit, coherent-noise)`` pair listed below we run two
attributions back-to-back -- one untwirled, one twirled -- and emit a
stacked two-row figure where the upper panel is the untwirled
attribution and the lower panel is the twirled.  Both panels share an
x-axis (column labels) and a y-axis range so the visual difference is
immediate: untwirled coherent noise produces zero shot variance and
typically concentrates on the same column every shot, while twirled
coherent noise spreads variance across columns and can shift the
attribution profile entirely.

Outputs land under :file:`paper/summary/`:

* ``twirl_<circuit>_<axis>.pdf``  -- canonical paper artefact
* ``twirl_<circuit>_<axis>.png``  -- inline preview
* ``twirl_<circuit>_<axis>.csv``  -- per-column stats for both runs
* ``twirl_<circuit>_<axis>.json`` -- replay config + headline numbers

Usage::

    python scripts/twirling_comparison.py
    python scripts/twirling_comparison.py --quick
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

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from PRISM.engine.debugger import CircuitDebugger  # noqa: E402
from PRISM.engine.noise import (  # noqa: E402
    CoherentOverRotationNoise,
    NoiseModel,
)
from PRISM.figures import (  # noqa: E402
    attribution_comparison_figure,
    save_figure,
    use_paper_style,
)

sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))
from generate_attribution_figures import (  # noqa: E402
    CIRCUITS, CIRCUIT_TITLES,
)


# ---------------------------------------------------------------------------
# Pair definitions
# ---------------------------------------------------------------------------


# Each pair = (circuit_name, axis, angle, slug).  The slug appears in
# the output filenames; the angle is in radians and intentionally on
# the larger end of "physically realistic" (~10 deg = 0.175 rad) so
# the twirling effect is unambiguous.
PAIRS: list[tuple[str, str, float, str]] = [
    ("qaoa_maxcut", "Z", 0.20, "qaoa_maxcut_z"),
    ("ghz3", "Y", 0.20, "ghz3_y"),
    ("bell", "Z", 0.30, "bell_z"),
    ("qft3", "X", 0.15, "qft3_x"),
]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_pair(
    circuit_name: str,
    axis: str,
    angle: float,
    n_trials: int,
    n_bootstrap: int,
    seed: int,
):
    """Compute both attributions for one ``(circuit, axis, angle)`` triple.

    The same noise model is reconstructed before each call so the
    second run does not inherit RNG state from the first.
    """
    debugger = CircuitDebugger()
    circuit = CIRCUITS[circuit_name]()
    nm_a = NoiseModel()
    nm_a.add_global_noise(CoherentOverRotationNoise(angle, axis=axis))
    untwirled = debugger.compute_noise_attribution_with_statistics(
        circuit, nm_a,
        n_trials=n_trials, n_bootstrap=n_bootstrap,
        seed=seed, twirl=False,
    )

    circuit = CIRCUITS[circuit_name]()
    nm_b = NoiseModel()
    nm_b.add_global_noise(CoherentOverRotationNoise(angle, axis=axis))
    twirled = debugger.compute_noise_attribution_with_statistics(
        circuit, nm_b,
        n_trials=n_trials, n_bootstrap=n_bootstrap,
        seed=seed, twirl=True,
    )
    return untwirled, twirled


def write_csv(path: Path, untwirled, twirled) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow([
            "column", "label",
            "untwirled_pct", "untwirled_q", "untwirled_significant",
            "twirled_pct", "twirled_q", "twirled_significant",
        ])
        for i, label_parts in enumerate(untwirled.gate_labels):
            label = " + ".join(label_parts) or f"col {i}"
            writer.writerow([
                i, label,
                f"{untwirled.column_attribution_pct[i]:.4f}",
                f"{untwirled.statistics.delta_fidelity_q_value[i]:.6e}",
                int(bool(untwirled.statistics.column_significant[i])),
                f"{twirled.column_attribution_pct[i]:.4f}",
                f"{twirled.statistics.delta_fidelity_q_value[i]:.6e}",
                int(bool(twirled.statistics.column_significant[i])),
            ])


def write_json(path: Path, *, slug, circuit, axis, angle,
               n_trials, n_bootstrap, seed,
               untwirled, twirled) -> None:
    payload = {
        "version": "1.0",
        "kind": "twirling_comparison",
        "slug": slug,
        "circuit": circuit,
        "noise": {
            "type": "CoherentOverRotationNoise",
            "axis": axis,
            "angle": angle,
        },
        "params": {
            "n_trials": n_trials,
            "n_bootstrap": n_bootstrap,
            "seed": seed,
        },
        "summary": {
            "untwirled": {
                "n_significant": int(sum(untwirled.statistics.column_significant)),
                "max_attribution_pct":
                    float(max(untwirled.column_attribution_pct)),
                "max_delta_F_std":
                    float(max(untwirled.delta_fidelity_std)),
                "total_fidelity_loss": float(untwirled.total_fidelity_loss),
            },
            "twirled": {
                "n_significant": int(sum(twirled.statistics.column_significant)),
                "max_attribution_pct":
                    float(max(twirled.column_attribution_pct)),
                "max_delta_F_std":
                    float(max(twirled.delta_fidelity_std)),
                "total_fidelity_loss": float(twirled.total_fidelity_loss),
            },
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def render_pair(
    untwirled, twirled,
    circuit_label: str, axis: str, angle: float,
    out_pdf: Path, out_png: Path,
) -> None:
    use_paper_style()
    title_top = (
        f"Untwirled coherent R{axis}({angle:.2f}) on {circuit_label}"
    )
    title_bot = "Pauli-twirled (same circuit, same noise)"
    fig = attribution_comparison_figure(
        untwirled, twirled,
        title_top=title_top,
        title_bottom=title_bot,
    )
    save_figure(fig, str(out_pdf))
    save_figure(fig, str(out_png))
    plt.close(fig)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Generate untwirled-vs-twirled attribution comparison "
            "figures for paper Section 6.6."
        ),
    )
    p.add_argument("--n-trials", type=int, default=120)
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=20260429)
    p.add_argument("--quick", action="store_true",
                   help="Smoke mode: n_trials=30, n_bootstrap=300.")
    p.add_argument("--output", type=Path,
                   default=_PROJECT_ROOT / "paper" / "summary")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    n_trials = 30 if args.quick else args.n_trials
    n_bootstrap = 300 if args.quick else args.n_bootstrap

    args.output.mkdir(parents=True, exist_ok=True)
    print(
        f"Twirling comparison: T={n_trials}, B={n_bootstrap}, "
        f"{len(PAIRS)} (circuit, axis) pairs"
    )

    t_start = time.perf_counter()
    for k, (circuit_name, axis, angle, slug) in enumerate(PAIRS):
        seed = args.seed + 100 * k
        t0 = time.perf_counter()
        untwirled, twirled = run_pair(
            circuit_name, axis, angle,
            n_trials=n_trials, n_bootstrap=n_bootstrap, seed=seed,
        )
        elapsed = time.perf_counter() - t0

        circuit_label = CIRCUIT_TITLES.get(circuit_name, circuit_name)

        out_pdf = args.output / f"twirl_{slug}.pdf"
        out_png = args.output / f"twirl_{slug}.png"
        out_csv = args.output / f"twirl_{slug}.csv"
        out_json = args.output / f"twirl_{slug}.json"

        render_pair(
            untwirled, twirled,
            circuit_label, axis, angle,
            out_pdf, out_png,
        )
        write_csv(out_csv, untwirled, twirled)
        write_json(
            out_json,
            slug=slug, circuit=circuit_name, axis=axis, angle=angle,
            n_trials=n_trials, n_bootstrap=n_bootstrap, seed=seed,
            untwirled=untwirled, twirled=twirled,
        )

        u_sig = sum(untwirled.statistics.column_significant)
        t_sig = sum(twirled.statistics.column_significant)
        u_max_std = max(untwirled.delta_fidelity_std)
        t_max_std = max(twirled.delta_fidelity_std)
        print(
            f"  [{k+1}/{len(PAIRS)}]  twirl_{slug:<22s}  "
            f"sig: {u_sig:2d} -> {t_sig:2d}   "
            f"max(dF_std): {u_max_std:.2e} -> {t_max_std:.2e}   "
            f"({elapsed:5.2f}s)"
        )

    total_elapsed = time.perf_counter() - t_start
    print()
    print(f"Generated {len(PAIRS)} comparison figures in {total_elapsed:.1f}s")
    print(f"  Outputs: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
