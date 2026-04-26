"""Aggregate the per-figure attribution tables into a paper-ready summary.

Each figure in :file:`paper/figures/` has a matching CSV in
:file:`paper/experiments/` with one row per circuit column.  This
script walks those CSVs and folds each ``(circuit, noise)`` pair down
to a single row of summary statistics that goes straight into the
paper's main table.

Outputs (under :file:`paper/summary/`):

* ``attribution_summary.csv`` -- machine-readable
* ``attribution_summary.tex`` -- LaTeX ``tabular`` body for direct paste
* ``attribution_summary.md``  -- Markdown table for paper.md

Run with::

    python scripts/aggregate_attribution_summary.py

No simulation is performed; this is a fold over data already on disk.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


# Pretty labels for circuits and noise channels.  Mirrors the ones in
# scripts/generate_attribution_figures.py so paper text reads consistently
# whatever angle the reader comes from.
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

# Order in which to render rows so circuit families stay grouped.
CIRCUIT_ORDER = [
    "bell", "ghz3", "ghz4",
    "qft3", "qft4",
    "qaoa_maxcut",
    "bit_flip_encoder",
    "bernstein_vazirani_3",
]
NOISE_ORDER = ["depolarizing", "bit_flip", "phase_flip", "amp_damping"]


@dataclass
class SummaryRow:
    """One ``(circuit, noise)`` pair folded into a single set of headline
    numbers."""

    circuit: str
    noise: str
    n_columns: int
    n_significant: int
    n_recovery: int
    total_fidelity_loss: float
    max_attribution_pct: float
    max_attribution_label: str
    mean_recovery_rate: float

    def as_dict(self) -> dict:
        return {
            "circuit": self.circuit,
            "noise": self.noise,
            "n_columns": self.n_columns,
            "n_significant": self.n_significant,
            "n_recovery": self.n_recovery,
            "total_fidelity_loss": f"{self.total_fidelity_loss:.4f}",
            "max_attribution_pct": f"{self.max_attribution_pct:.2f}",
            "max_attribution_label": self.max_attribution_label,
            "mean_recovery_rate": f"{self.mean_recovery_rate:.3f}",
        }


def _stem_to_pair(stem: str) -> tuple[str, str]:
    """Split ``attr_<circuit>_<noise>`` back into ``(circuit, noise)``.

    The split happens on the *last* underscore-delimited token that
    matches one of the known noise labels, so multi-token circuit names
    like ``bernstein_vazirani_3`` survive intact.
    """
    if not stem.startswith("attr_"):
        raise ValueError(f"Expected stem to start with 'attr_', got {stem!r}")
    body = stem[len("attr_"):]

    # bit_flip and phase_flip and amp_damping each contain underscores,
    # so we have to greedily match the longest noise suffix.
    for noise in sorted(NOISE_LABELS, key=len, reverse=True):
        suffix = "_" + noise
        if body.endswith(suffix):
            return body[: -len(suffix)], noise
    raise ValueError(
        f"Stem {stem!r} does not end with a known noise suffix "
        f"(one of {sorted(NOISE_LABELS)})"
    )


def _summarise_csv(csv_path: Path) -> SummaryRow:
    """Fold a single per-column CSV into a :class:`SummaryRow`."""
    stem = csv_path.stem
    circuit, noise = _stem_to_pair(stem)

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    n_columns = len(rows)
    n_significant = sum(1 for r in rows if r.get("significant") == "1")
    n_recovery = sum(1 for r in rows if r.get("is_recovery") == "1")

    # The table doesn't store the total directly; reconstruct from the
    # mean delta_F per column (which sums to the total fidelity loss).
    total_loss = sum(float(r["delta_F_mean"]) for r in rows)

    pct_floats = [float(r["attribution_pct"]) for r in rows]
    max_idx = max(range(len(pct_floats)), key=pct_floats.__getitem__)
    max_pct = pct_floats[max_idx]
    max_label = rows[max_idx]["label"]

    rec_rates = [float(r["recovery_rate"]) for r in rows if r.get("recovery_rate")]
    mean_rec = sum(rec_rates) / max(len(rec_rates), 1)

    return SummaryRow(
        circuit=circuit,
        noise=noise,
        n_columns=n_columns,
        n_significant=n_significant,
        n_recovery=n_recovery,
        total_fidelity_loss=total_loss,
        max_attribution_pct=max_pct,
        max_attribution_label=max_label,
        mean_recovery_rate=mean_rec,
    )


def _sort_key(row: SummaryRow) -> tuple[int, int]:
    c_idx = CIRCUIT_ORDER.index(row.circuit) if row.circuit in CIRCUIT_ORDER else 99
    n_idx = NOISE_ORDER.index(row.noise) if row.noise in NOISE_ORDER else 99
    return (c_idx, n_idx)


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def write_csv(rows: list[SummaryRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].as_dict().keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(r.as_dict())


def _escape_latex(text: str) -> str:
    """Escape just the characters that matter inside a tabular cell."""
    return (
        text.replace("\\", r"\textbackslash{}")
            .replace("_", r"\_")
            .replace("&", r"\&")
            .replace("%", r"\%")
            .replace("$", r"\$")
            .replace("#", r"\#")
    )


def write_latex(rows: list[SummaryRow], path: Path) -> None:
    r"""Write a LaTeX ``tabular`` *body* (no ``\begin`` / ``\end``) so the user
    can drop it into whatever floating environment they prefer."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("% Auto-generated by scripts/aggregate_attribution_summary.py")
    lines.append("% \\begin{tabular}{llrrrrrl}")
    lines.append(
        "Circuit & Noise & "
        "$L$ & $|\\mathrm{sig}|$ & $|\\mathrm{rec}|$ & "
        "$g_L$ & $\\max A_i$ (\\%) & dominant column \\\\"
    )
    lines.append("\\hline")
    for r in rows:
        cl = CIRCUIT_LABELS.get(r.circuit, r.circuit)
        nl = NOISE_LABELS.get(r.noise, r.noise)
        lines.append(
            "{} & {} & {} & {} & {} & {:.4f} & {:.1f} & {} \\\\".format(
                _escape_latex(cl),
                _escape_latex(nl),
                r.n_columns,
                r.n_significant,
                r.n_recovery,
                r.total_fidelity_loss,
                r.max_attribution_pct,
                _escape_latex(r.max_attribution_label),
            )
        )
    lines.append("% \\end{tabular}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_markdown(rows: list[SummaryRow], path: Path) -> None:
    """Write a GitHub-flavoured markdown table for paper.md."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append(
        "| Circuit | Noise | $L$ | sig | rec | $g_L$ | max $A_i$ (%) | dominant column |"
    )
    lines.append(
        "|---|---|---:|---:|---:|---:|---:|---|"
    )
    for r in rows:
        cl = CIRCUIT_LABELS.get(r.circuit, r.circuit)
        nl = NOISE_LABELS.get(r.noise, r.noise)
        lines.append(
            "| {} | {} | {} | {} | {} | {:.4f} | {:.1f} | `{}` |".format(
                cl, nl,
                r.n_columns,
                r.n_significant,
                r.n_recovery,
                r.total_fidelity_loss,
                r.max_attribution_pct,
                r.max_attribution_label,
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_aggregate_stats(rows: list[SummaryRow], path: Path) -> None:
    """Compact JSON of cross-suite numbers cited in the paper's Section 6
    introduction.

    Stored separately so paper text can reference exact values without
    re-running this script.
    """
    n_total = len(rows)
    n_sig_total = sum(r.n_significant for r in rows)
    n_col_total = sum(r.n_columns for r in rows)
    n_recov_total = sum(r.n_recovery for r in rows)

    pauli_rows = [r for r in rows if r.noise != "amp_damping"]
    nonpauli_rows = [r for r in rows if r.noise == "amp_damping"]

    summary = {
        "n_pairs": n_total,
        "n_columns_total": n_col_total,
        "n_significant_total": n_sig_total,
        "n_recovery_total": n_recov_total,
        "fraction_columns_significant": (
            n_sig_total / n_col_total if n_col_total else 0.0
        ),
        "pauli_subset": {
            "n_pairs": len(pauli_rows),
            "n_columns": sum(r.n_columns for r in pauli_rows),
            "n_significant": sum(r.n_significant for r in pauli_rows),
            "fraction_significant": (
                sum(r.n_significant for r in pauli_rows)
                / max(sum(r.n_columns for r in pauli_rows), 1)
            ),
        },
        "nonpauli_subset": {
            "n_pairs": len(nonpauli_rows),
            "n_columns": sum(r.n_columns for r in nonpauli_rows),
            "n_significant": sum(r.n_significant for r in nonpauli_rows),
            "fraction_significant": (
                sum(r.n_significant for r in nonpauli_rows)
                / max(sum(r.n_columns for r in nonpauli_rows), 1)
            ),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def aggregate(experiments_dir: Path) -> list[SummaryRow]:
    """Walk every CSV under ``experiments_dir`` and produce
    :class:`SummaryRow` objects sorted into paper-table order."""
    rows: list[SummaryRow] = []
    for csv_path in sorted(experiments_dir.glob("attr_*.csv")):
        rows.append(_summarise_csv(csv_path))
    rows.sort(key=_sort_key)
    return rows


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Fold paper/experiments/*.csv into a single attribution "
            "summary table with CSV / LaTeX / Markdown outputs."
        ),
    )
    p.add_argument(
        "--experiments", type=Path,
        default=_PROJECT_ROOT / "paper" / "experiments",
        help="Directory containing the per-figure CSV tables.",
    )
    p.add_argument(
        "--output", type=Path,
        default=_PROJECT_ROOT / "paper" / "summary",
        help="Output directory for the aggregated tables.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    rows = aggregate(args.experiments)
    if not rows:
        print(
            f"error: no attr_*.csv files found under {args.experiments}",
        )
        return 2

    csv_out = args.output / "attribution_summary.csv"
    tex_out = args.output / "attribution_summary.tex"
    md_out = args.output / "attribution_summary.md"
    json_out = args.output / "aggregate_stats.json"

    write_csv(rows, csv_out)
    write_latex(rows, tex_out)
    write_markdown(rows, md_out)
    write_aggregate_stats(rows, json_out)

    print(f"Aggregated {len(rows)} (circuit, noise) pairs.")
    print(f"  CSV  : {csv_out}")
    print(f"  LaTeX: {tex_out}")
    print(f"  MD   : {md_out}")
    print(f"  Stats: {json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
