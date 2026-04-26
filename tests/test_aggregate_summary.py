"""Tests for ``scripts/aggregate_attribution_summary.py``.

The aggregate script is a pure fold over CSV files on disk -- no
simulation, no figure rendering -- so the tests focus on the splitting
logic (which has subtle edge cases for multi-token circuit names) and
on the round-trip behaviour of the LaTeX / Markdown / JSON outputs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# The script imports from ``scripts/`` itself, so make sure that
# directory is importable.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))

import aggregate_attribution_summary as agg  # noqa: E402


# ---------------------------------------------------------------------------
# Stem splitting -- the trickiest part of the script
# ---------------------------------------------------------------------------


class TestStemToPair:

    @pytest.mark.parametrize(
        "stem, circuit, noise",
        [
            ("attr_bell_depolarizing", "bell", "depolarizing"),
            ("attr_ghz3_bit_flip", "ghz3", "bit_flip"),
            ("attr_qft4_phase_flip", "qft4", "phase_flip"),
            ("attr_qaoa_maxcut_amp_damping", "qaoa_maxcut", "amp_damping"),
            (
                "attr_bernstein_vazirani_3_depolarizing",
                "bernstein_vazirani_3",
                "depolarizing",
            ),
            (
                "attr_bit_flip_encoder_phase_flip",
                "bit_flip_encoder",
                "phase_flip",
            ),
        ],
    )
    def test_known_pairs(self, stem, circuit, noise):
        assert agg._stem_to_pair(stem) == (circuit, noise)

    def test_unknown_noise_rejected(self):
        with pytest.raises(ValueError, match="noise suffix"):
            agg._stem_to_pair("attr_bell_unknown")

    def test_missing_attr_prefix_rejected(self):
        with pytest.raises(ValueError, match="attr_"):
            agg._stem_to_pair("bell_depolarizing")


# ---------------------------------------------------------------------------
# Single-CSV summarisation
# ---------------------------------------------------------------------------


class TestSummariseCsv:

    def _write_csv(self, path: Path, rows: list[dict]) -> None:
        import csv as _csv
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            for r in rows:
                writer.writerow(r)

    def _full_row(self, **overrides) -> dict:
        base = {
            "column": 0,
            "label": "H(0)",
            "delta_F_mean": "0.05",
            "delta_F_ci_lower": "0.04",
            "delta_F_ci_upper": "0.06",
            "p_value": "0.001",
            "q_value": "0.002",
            "significant": "1",
            "attribution_pct": "60.0",
            "attribution_pct_ci_lower": "50.0",
            "attribution_pct_ci_upper": "70.0",
            "is_recovery": "0",
            "recovery_rate": "0.10",
            "recovery_rate_ci_lower": "0.05",
            "recovery_rate_ci_upper": "0.15",
        }
        base.update(overrides)
        return base

    def test_count_significance_and_recovery(self, tmp_path):
        rows = [
            self._full_row(column=0, label="H(0)",
                           significant="1", is_recovery="0"),
            self._full_row(column=1, label="CNOT(0,1)",
                           significant="0", is_recovery="1",
                           attribution_pct="40.0"),
            self._full_row(column=2, label="Rz(2)",
                           significant="1", is_recovery="0",
                           attribution_pct="20.0"),
        ]
        path = tmp_path / "attr_bell_depolarizing.csv"
        self._write_csv(path, rows)

        out = agg._summarise_csv(path)
        assert out.circuit == "bell"
        assert out.noise == "depolarizing"
        assert out.n_columns == 3
        assert out.n_significant == 2
        assert out.n_recovery == 1

    def test_max_attribution_is_dominant_column(self, tmp_path):
        rows = [
            self._full_row(column=0, label="A", attribution_pct="10"),
            self._full_row(column=1, label="B", attribution_pct="65"),
            self._full_row(column=2, label="C", attribution_pct="25"),
        ]
        path = tmp_path / "attr_bell_bit_flip.csv"
        self._write_csv(path, rows)
        out = agg._summarise_csv(path)
        assert out.max_attribution_pct == 65.0
        assert out.max_attribution_label == "B"

    def test_total_loss_is_sum_of_means(self, tmp_path):
        rows = [
            self._full_row(column=0, label="A", delta_F_mean="0.10"),
            self._full_row(column=1, label="B", delta_F_mean="0.20"),
        ]
        path = tmp_path / "attr_bell_phase_flip.csv"
        self._write_csv(path, rows)
        out = agg._summarise_csv(path)
        assert out.total_fidelity_loss == pytest.approx(0.30)


# ---------------------------------------------------------------------------
# End-to-end aggregate over the real paper/experiments data
# ---------------------------------------------------------------------------


class TestAggregateRealData:

    def test_aggregate_real_paper_experiments(self):
        rows = agg.aggregate(_PROJECT_ROOT / "paper" / "experiments")
        assert len(rows) == 32, f"expected 32 (circuit,noise) pairs, got {len(rows)}"

        # Sanity: every circuit / noise label is one of the known sets.
        for r in rows:
            assert r.circuit in agg.CIRCUIT_LABELS, f"unknown circuit {r.circuit!r}"
            assert r.noise in agg.NOISE_LABELS, f"unknown noise {r.noise!r}"

        # Every row should report at least one column and a non-negative
        # total fidelity loss.
        for r in rows:
            assert r.n_columns >= 1
            assert r.total_fidelity_loss >= 0
            assert 0 <= r.n_significant <= r.n_columns
            assert 0 <= r.n_recovery <= r.n_columns
            assert 0 <= r.max_attribution_pct <= 100

    def test_writers_produce_non_empty_files(self, tmp_path):
        rows = agg.aggregate(_PROJECT_ROOT / "paper" / "experiments")

        csv_out = tmp_path / "out.csv"
        tex_out = tmp_path / "out.tex"
        md_out = tmp_path / "out.md"
        json_out = tmp_path / "out.json"

        agg.write_csv(rows, csv_out)
        agg.write_latex(rows, tex_out)
        agg.write_markdown(rows, md_out)
        agg.write_aggregate_stats(rows, json_out)

        for path in (csv_out, tex_out, md_out, json_out):
            assert path.exists()
            assert path.stat().st_size > 100, f"{path} is suspiciously small"

        # JSON must be a valid object with the expected top-level keys.
        data = json.loads(json_out.read_text(encoding="utf-8"))
        assert data["n_pairs"] == len(rows)
        assert "fraction_columns_significant" in data
        assert "pauli_subset" in data
        assert "nonpauli_subset" in data
