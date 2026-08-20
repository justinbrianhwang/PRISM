"""Seed-sensitivity analysis for the attribution suite.

The headline benchmark fixes one deterministic seed per (circuit,
noise) pair.  This script quantifies how much the reported quantities
move under seed variation: for four representative pairs it re-runs the
full bootstrap-aware attribution across ``N_SEEDS`` independent seeds
and reports the distribution of

* the FDR-significant column count,
* the dominant column's identity (modal stability), and
* the dominant column's attribution percentage.

Outputs (under :file:`paper/summary/`):

* ``seed_sensitivity.csv``  -- one row per (pair, seed)
* ``seed_sensitivity.json`` -- aggregate summary per pair
* ``seed_sensitivity.md``   -- Markdown summary table
* ``seed_sensitivity.tex``  -- LaTeX ``tabular`` body for the appendix

Run with::

    python scripts/seed_sensitivity.py
"""

from __future__ import annotations

import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from PRISM.engine.circuit import QuantumCircuit          # noqa: E402
from PRISM.engine.debugger import CircuitDebugger        # noqa: E402
from PRISM.engine.noise import NoiseModel                # noqa: E402
from PRISM.replay import ReplayConfig                    # noqa: E402

N_SEEDS = 50
SEED_BASE = 977000  # disjoint from the 2026xxxx suite seeds

# Representative pairs spanning the qualitative regimes of Table 3:
# dense variational circuit, entangling chain, deep marginal circuit,
# and a weak-noise two-column circuit.
PAIRS = [
    ("attr_qaoa_maxcut_depolarizing", "QAOA($C_4$) + Depol"),
    ("attr_ghz4_depolarizing", "GHZ-4 + Depol"),
    ("attr_qft4_bit_flip", "QFT-4 + BitFlip"),
    ("attr_bell_amp_damping", "Bell + AmpDamp"),
]


def main() -> None:
    experiments = _PROJECT_ROOT / "paper" / "experiments"
    summary_dir = _PROJECT_ROOT / "paper" / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    debugger = CircuitDebugger()
    rows = []
    aggregates = {}

    for stem, label in PAIRS:
        cfg = ReplayConfig.load(experiments / f"{stem}.json")
        circuit = QuantumCircuit.from_dict(cfg.circuit)
        noise = NoiseModel.from_dict(cfg.noise_model)

        sig_counts = []
        dominants = []
        dominant_pcts = []
        t0 = time.perf_counter()
        for k in range(N_SEEDS):
            seed = SEED_BASE + k
            attr = debugger.compute_noise_attribution_with_statistics(
                circuit, noise,
                n_trials=cfg.params.n_trials,
                n_bootstrap=cfg.params.n_bootstrap,
                confidence=cfg.params.confidence,
                fdr_level=cfg.params.fdr_level,
                seed=seed,
            )
            stats = attr.statistics
            n_sig = int(sum(stats.column_significant))
            pcts = np.asarray(attr.column_attribution_pct)
            dom = int(pcts.argmax()) if pcts.max() > 0 else -1
            dom_pct = float(pcts.max())

            sig_counts.append(n_sig)
            dominants.append(dom)
            dominant_pcts.append(dom_pct)
            rows.append({
                "pair": stem, "seed": seed, "n_significant": n_sig,
                "dominant_column": dom,
                "dominant_attribution_pct": round(dom_pct, 4),
            })
        elapsed = time.perf_counter() - t0

        counts = Counter(dominants)
        modal_col, modal_n = counts.most_common(1)[0]
        aggregates[stem] = {
            "label": label,
            "n_seeds": N_SEEDS,
            "sig_min": int(min(sig_counts)),
            "sig_median": float(np.median(sig_counts)),
            "sig_max": int(max(sig_counts)),
            "suite_seed_sig": None,  # filled below from the committed CSV
            "dominant_modal_column": int(modal_col),
            "dominant_modal_fraction": modal_n / N_SEEDS,
            "dominant_pct_mean": float(np.mean(dominant_pcts)),
            "dominant_pct_std": float(np.std(dominant_pcts)),
            "elapsed_seconds": round(elapsed, 1),
        }

        # The suite's committed value for context.
        with open(experiments / f"{stem}.csv", newline="") as fh:
            recs = list(csv.DictReader(fh))
        aggregates[stem]["suite_seed_sig"] = sum(
            int(r["significant"]) for r in recs
        )

        print(f"{label:24s} sig med {np.median(sig_counts):4.1f} "
              f"[{min(sig_counts)}, {max(sig_counts)}]  "
              f"modal dom col {modal_col} ({100 * modal_n / N_SEEDS:.0f}%)  "
              f"({elapsed:.0f}s)")

    # ------------------------------------------------------------------ CSV
    with open(summary_dir / "seed_sensitivity.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()),
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    # ----------------------------------------------------------------- JSON
    with open(summary_dir / "seed_sensitivity.json", "w") as fh:
        json.dump({"n_seeds": N_SEEDS, "seed_base": SEED_BASE,
                   "pairs": aggregates}, fh, indent=2)

    # ------------------------------------------------------------------- MD
    with open(summary_dir / "seed_sensitivity.md", "w") as fh:
        fh.write("| Pair | suite seed sig | sig median [min, max] "
                 "| modal dominant col (stability) | dom. attr % (mean +- std) |\n")
        fh.write("|---|---:|---|---|---|\n")
        for stem, a in aggregates.items():
            fh.write(
                f"| {a['label']} | {a['suite_seed_sig']} "
                f"| {a['sig_median']:.1f} [{a['sig_min']}, {a['sig_max']}] "
                f"| col {a['dominant_modal_column']} "
                f"({100 * a['dominant_modal_fraction']:.0f}%) "
                f"| {a['dominant_pct_mean']:.1f} +- {a['dominant_pct_std']:.1f} |\n"
            )

    # ------------------------------------------------------------------ TeX
    with open(summary_dir / "seed_sensitivity.tex", "w") as fh:
        fh.write("% Auto-generated by scripts/seed_sensitivity.py\n")
        fh.write("\\begin{tabular}{@{}lcccc}\n\\br\n")
        fh.write("Pair & suite sig & sig med [min, max] & "
                 "modal dom.\\ col & dom.\\ $A_i$ (\\%) \\\\\n\\mr\n")
        for stem, a in aggregates.items():
            fh.write(
                f"{a['label']} & {a['suite_seed_sig']} & "
                f"{a['sig_median']:.1f} [{a['sig_min']}, {a['sig_max']}] & "
                f"{a['dominant_modal_column']} "
                f"({100 * a['dominant_modal_fraction']:.0f}\\%) & "
                f"${a['dominant_pct_mean']:.1f} \\pm "
                f"{a['dominant_pct_std']:.1f}$ \\\\\n"
            )
        fh.write("\\br\n\\end{tabular}\n")

    print("wrote paper/summary/seed_sensitivity.{csv,json,md,tex}")


if __name__ == "__main__":
    main()
