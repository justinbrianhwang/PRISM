"""Ground-truth validation: targeted noise injection.

The attribution methodology claims to localise noise sources.  This
script tests that claim against a known ground truth: a five-column
circuit in which every column carries a unique gate type, so that
gate-targeted noise injection puts depolarizing noise on exactly one
known column.  For each (injected column, strength, seed) cell we run
the full bootstrap-aware attribution and score three binary outcomes:

* **hit**       -- the injected column is FDR-significant,
* **top-1**     -- the injected column has the largest attribution %,
* **exact**     -- the FDR-significant set is exactly {injected column}.

Aggregating false discoveries on the non-injected columns also gives an
empirical check of the FDR guarantee under a known null for those
columns.

Outputs (under :file:`paper/summary/`):

* ``ground_truth_injection.csv``  -- one row per run
* ``ground_truth_injection.json`` -- aggregate summary
* ``ground_truth_injection.md``   -- Markdown summary table
* ``ground_truth_injection.tex``  -- LaTeX ``tabular`` body

Run with::

    python scripts/ground_truth_injection.py
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from PRISM.engine.circuit import GateInstance, QuantumCircuit  # noqa: E402
from PRISM.engine.debugger import CircuitDebugger              # noqa: E402
from PRISM.engine.noise import DepolarizingNoise, NoiseModel   # noqa: E402

N_SEEDS = 20
SEED_BASE = 411000
STRENGTHS = [0.05, 0.10, 0.20]
N_TRIALS = 120
N_BOOTSTRAP = 1000
FDR_LEVEL = 0.05


def ground_truth_circuit() -> tuple[QuantumCircuit, list[str]]:
    """A 3-qubit, five-column circuit with one unique gate type per column.

    Unique gate names let ``NoiseModel.add_gate_noise`` target exactly
    one column, which is what turns the run into a ground-truth
    experiment.  The state is kept entangled and non-trivial so that
    depolarizing injection is detectable at every column.
    """
    qc = QuantumCircuit(num_qubits=3)
    qc.add_gate(GateInstance("H", [0], [], 0))
    qc.add_gate(GateInstance("CNOT", [0, 1], [], 1))
    qc.add_gate(GateInstance("Ry", [2], [0.7], 2))
    qc.add_gate(GateInstance("CZ", [1, 2], [], 3))
    qc.add_gate(GateInstance("X", [0], [], 4))
    gate_names = ["H", "CNOT", "Ry", "CZ", "X"]
    return qc, gate_names


def main() -> None:
    summary_dir = _PROJECT_ROOT / "paper" / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    circuit, gate_names = ground_truth_circuit()
    n_cols = len(gate_names)
    debugger = CircuitDebugger()

    rows = []
    t0 = time.perf_counter()
    for strength in STRENGTHS:
        for injected_col, gate_name in enumerate(gate_names):
            nm = NoiseModel()
            nm.add_gate_noise(gate_name, DepolarizingNoise(strength))
            for k in range(N_SEEDS):
                seed = SEED_BASE + 1000 * injected_col + k
                attr = debugger.compute_noise_attribution_with_statistics(
                    circuit, nm,
                    n_trials=N_TRIALS, n_bootstrap=N_BOOTSTRAP,
                    fdr_level=FDR_LEVEL, seed=seed,
                )
                sig = np.asarray(attr.statistics.column_significant, dtype=bool)
                pcts = np.asarray(attr.column_attribution_pct)
                hit = bool(sig[injected_col])
                top1 = bool(pcts.argmax() == injected_col and pcts.max() > 0)
                false_cols = int(sig.sum() - sig[injected_col])
                exact = hit and false_cols == 0
                rows.append({
                    "strength": strength,
                    "injected_column": injected_col,
                    "gate": gate_name,
                    "seed": seed,
                    "hit": int(hit),
                    "top1": int(top1),
                    "exact": int(exact),
                    "false_discoveries": false_cols,
                })
    elapsed = time.perf_counter() - t0

    # ------------------------------------------------------------ aggregate
    agg = {}
    for strength in STRENGTHS:
        sel = [r for r in rows if r["strength"] == strength]
        n = len(sel)
        n_null_cols = (n_cols - 1) * n
        agg[str(strength)] = {
            "runs": n,
            "hit_rate": sum(r["hit"] for r in sel) / n,
            "top1_rate": sum(r["top1"] for r in sel) / n,
            "exact_rate": sum(r["exact"] for r in sel) / n,
            "false_discovery_rate_per_run":
                sum(r["false_discoveries"] for r in sel) / n,
            "null_column_flag_rate":
                sum(r["false_discoveries"] for r in sel) / n_null_cols,
        }

    # ---------------------------------------------------------------- files
    with open(summary_dir / "ground_truth_injection.csv", "w",
              newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()),
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    with open(summary_dir / "ground_truth_injection.json", "w") as fh:
        json.dump({
            "n_seeds": N_SEEDS, "seed_base": SEED_BASE,
            "n_trials": N_TRIALS, "n_bootstrap": N_BOOTSTRAP,
            "fdr_level": FDR_LEVEL,
            "circuit_columns": gate_names,
            "aggregate": agg,
            "elapsed_seconds": round(elapsed, 1),
        }, fh, indent=2)

    with open(summary_dir / "ground_truth_injection.md", "w") as fh:
        fh.write("| p | hit rate | top-1 rate | exact recovery "
                 "| null-column flag rate |\n")
        fh.write("|---:|---:|---:|---:|---:|\n")
        for s, a in agg.items():
            fh.write(f"| {s} | {a['hit_rate']:.2f} | {a['top1_rate']:.2f} "
                     f"| {a['exact_rate']:.2f} "
                     f"| {a['null_column_flag_rate']:.3f} |\n")

    with open(summary_dir / "ground_truth_injection.tex", "w") as fh:
        fh.write("% Auto-generated by scripts/ground_truth_injection.py\n")
        fh.write("\\begin{tabular}{@{}ccccc}\n\\br\n")
        fh.write("$p$ & hit & top-1 & exact & null-column flag rate \\\\\n")
        fh.write("\\mr\n")
        for s, a in agg.items():
            fh.write(f"{s} & {a['hit_rate']:.2f} & {a['top1_rate']:.2f} & "
                     f"{a['exact_rate']:.2f} & "
                     f"{a['null_column_flag_rate']:.3f} \\\\\n")
        fh.write("\\br\n\\end{tabular}\n")

    for s, a in agg.items():
        print(f"p={s}: hit {a['hit_rate']:.2f}  top1 {a['top1_rate']:.2f}  "
              f"exact {a['exact_rate']:.2f}  "
              f"null-flag {a['null_column_flag_rate']:.3f}")
    print(f"total {len(rows)} runs in {elapsed:.0f}s")
    print("wrote paper/summary/ground_truth_injection.{csv,json,md,tex}")


if __name__ == "__main__":
    main()
