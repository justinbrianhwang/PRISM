"""Benjamini-Yekutieli sensitivity check for the attribution suite.

PRISM's per-column p-values are derived from shared trajectory rows and
are therefore mutually dependent.  Benjamini-Hochberg controls the FDR
exactly under independence or positive regression dependence; the
Benjamini-Yekutieli correction is the conservative guarantee under
arbitrary dependence.  This script re-adjusts the *stored* raw p-values
of every ``(circuit, noise)`` pair in :file:`paper/experiments/` under
both procedures and reports how the significant-column counts change.

No simulation is performed -- the raw p-values in the per-figure CSVs
are exactly the ones the paper's BH flags were computed from, so this
is a pure re-analysis of data already on disk.

Outputs (under :file:`paper/summary/`):

* ``by_sensitivity.csv`` -- machine-readable per-pair comparison
* ``by_sensitivity.tex`` -- LaTeX ``tabular`` body for the appendix
* ``by_sensitivity.md``  -- Markdown table for paper.md

Run with::

    python scripts/by_sensitivity.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from PRISM.engine.statistics import benjamini_hochberg, benjamini_yekutieli  # noqa: E402

CIRCUIT_LABELS = {
    "bell": "Bell",
    "ghz3": "GHZ-3",
    "ghz4": "GHZ-4",
    "qft3": "QFT-3",
    "qft4": "QFT-4",
    "qaoa_maxcut": "QAOA(C_4)",
    "bit_flip_encoder": "Bit-flip enc.",
    "bernstein_vazirani_3": "BV-3",
}
NOISE_LABELS = {
    "depolarizing": "Depol",
    "bit_flip": "BitFlip",
    "phase_flip": "PhaseFlip",
    "amp_damping": "AmpDamp",
}
CIRCUIT_ORDER = [
    "bell", "ghz3", "ghz4",
    "qft3", "qft4",
    "qaoa_maxcut",
    "bit_flip_encoder",
    "bernstein_vazirani_3",
]
NOISE_ORDER = ["depolarizing", "bit_flip", "phase_flip", "amp_damping"]

FDR_LEVEL = 0.05


def split_stem(stem: str) -> tuple[str, str]:
    """Split ``attr_<circuit>_<noise>`` into its circuit and noise keys."""
    body = stem[len("attr_"):]
    for circuit in sorted(CIRCUIT_LABELS, key=len, reverse=True):
        if body.startswith(circuit + "_"):
            return circuit, body[len(circuit) + 1:]
    raise ValueError(f"unrecognised experiment stem: {stem}")


def main() -> None:
    experiments = _PROJECT_ROOT / "paper" / "experiments"
    summary_dir = _PROJECT_ROOT / "paper" / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for csv_path in sorted(experiments.glob("attr_*.csv")):
        circuit, noise = split_stem(csv_path.stem)
        with open(csv_path, newline="") as fh:
            records = list(csv.DictReader(fh))

        p_values = np.array([float(r["p_value"]) for r in records])
        stored_sig = np.array([int(r["significant"]) for r in records], dtype=bool)
        stored_q = np.array([float(r["q_value"]) for r in records])

        q_bh, rej_bh = benjamini_hochberg(p_values, fdr=FDR_LEVEL)
        q_by, rej_by = benjamini_yekutieli(p_values, fdr=FDR_LEVEL)

        # Sanity: our BH re-run must reproduce the stored flags exactly,
        # otherwise the stored CSVs and this analysis have diverged.
        if not np.array_equal(rej_bh, stored_sig):
            raise AssertionError(
                f"{csv_path.name}: BH re-run disagrees with stored flags"
            )
        if not np.allclose(q_bh, stored_q, atol=1e-9):
            raise AssertionError(
                f"{csv_path.name}: BH re-run disagrees with stored q-values"
            )

        rows.append({
            "circuit": circuit,
            "noise": noise,
            "n_columns": len(records),
            "sig_bh": int(rej_bh.sum()),
            "sig_by": int(rej_by.sum()),
            "flipped": int((rej_bh & ~rej_by).sum()),
        })

    rows.sort(key=lambda r: (CIRCUIT_ORDER.index(r["circuit"]),
                             NOISE_ORDER.index(r["noise"])))

    total_cols = sum(r["n_columns"] for r in rows)
    total_bh = sum(r["sig_bh"] for r in rows)
    total_by = sum(r["sig_by"] for r in rows)
    total_flipped = sum(r["flipped"] for r in rows)

    # ------------------------------------------------------------------ CSV
    csv_out = summary_dir / "by_sensitivity.csv"
    with open(csv_out, "w", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(
            ["circuit", "noise", "n_columns", "sig_bh", "sig_by", "flipped"]
        )
        for r in rows:
            writer.writerow([
                CIRCUIT_LABELS[r["circuit"]], NOISE_LABELS[r["noise"]],
                r["n_columns"], r["sig_bh"], r["sig_by"], r["flipped"],
            ])
        writer.writerow(["TOTAL", "", total_cols, total_bh, total_by,
                         total_flipped])

    # ------------------------------------------------------------------ MD
    md_out = summary_dir / "by_sensitivity.md"
    with open(md_out, "w") as fh:
        fh.write("| Circuit | Noise | L | sig (BH) | sig (BY) | flipped |\n")
        fh.write("|---|---|---:|---:|---:|---:|\n")
        for r in rows:
            fh.write(
                f"| {CIRCUIT_LABELS[r['circuit']]} | {NOISE_LABELS[r['noise']]} "
                f"| {r['n_columns']} | {r['sig_bh']} | {r['sig_by']} "
                f"| {r['flipped']} |\n"
            )
        fh.write(
            f"| **Total** | | {total_cols} | {total_bh} | {total_by} "
            f"| {total_flipped} |\n"
        )

    # ------------------------------------------------------------------ TeX
    tex_out = summary_dir / "by_sensitivity.tex"
    with open(tex_out, "w") as fh:
        fh.write("% Auto-generated by scripts/by_sensitivity.py\n")
        fh.write("\\begin{tabular}{@{}llrrrr}\n\\br\n")
        fh.write("Circuit & Noise & $L$ & sig (BH) & sig (BY) & flipped \\\\\n")
        fh.write("\\mr\n")
        for r in rows:
            circuit_tex = CIRCUIT_LABELS[r["circuit"]].replace("C_4", "$C_4$")
            fh.write(
                f"{circuit_tex} & {NOISE_LABELS[r['noise']]} & "
                f"{r['n_columns']} & {r['sig_bh']} & {r['sig_by']} & "
                f"{r['flipped']} \\\\\n"
            )
        fh.write("\\mr\n")
        fh.write(
            f"Total & & {total_cols} & {total_bh} & {total_by} & "
            f"{total_flipped} \\\\\n"
        )
        fh.write("\\br\n\\end{tabular}\n")

    print(f"pairs analysed : {len(rows)}")
    print(f"columns        : {total_cols}")
    print(f"sig under BH   : {total_bh}")
    print(f"sig under BY   : {total_by}")
    print(f"flipped BH->BY : {total_flipped}")
    print(f"wrote {csv_out}, {md_out}, {tex_out}")


if __name__ == "__main__":
    main()
