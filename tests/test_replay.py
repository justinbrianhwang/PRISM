"""Tests for :mod:`PRISM.replay`.

Replay is the reproducibility appendix of the paper, in code form, so
the test suite places weight on three properties:

* **Schema round-trip** -- a :class:`ReplayConfig` survives ``to_json``
  / ``from_json`` and ``save`` / ``load`` without semantic drift.
* **Self-containment** -- a v2 config carries enough information that a
  replay run does not need any in-script registry; the JSON file alone
  suffices.
* **Bit-exact reproducibility** -- running the same config twice
  produces identical CSV tables and significance flags.

A separate test exercises the CLI surface (``python -m PRISM.replay``)
to make sure the entry point keeps working as a script.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

# Headless backend BEFORE pyplot.
import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from PRISM.engine.circuit import GateInstance, QuantumCircuit
from PRISM.engine.noise import (
    BitFlipNoise,
    DepolarizingNoise,
    NoiseModel,
)
from PRISM.replay import (
    ReplayConfig,
    ReplayParams,
    SCHEMA_VERSION,
    replay,
    replay_all,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_circuit() -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits=2)
    qc.add_gate(GateInstance("H", [0], [], 0))
    qc.add_gate(GateInstance("CNOT", [0, 1], [], 1))
    return qc


@pytest.fixture
def depolarizing_noise() -> NoiseModel:
    nm = NoiseModel()
    nm.add_global_noise(DepolarizingNoise(0.05))
    return nm


@pytest.fixture
def small_config(small_circuit, depolarizing_noise) -> ReplayConfig:
    return ReplayConfig.from_current(
        label="bell_depol",
        title="Bell  |  depolarizing 0.05",
        circuit=small_circuit,
        noise_model=depolarizing_noise,
        params=ReplayParams(
            n_trials=40, n_bootstrap=200,
            confidence=0.95, fdr_level=0.05,
        ),
        seed=2026,
    )


# ---------------------------------------------------------------------------
# Schema round-trip
# ---------------------------------------------------------------------------


class TestReplayConfigSerialisation:

    def test_to_dict_includes_all_fields(self, small_config):
        d = small_config.to_dict()
        assert d["version"] == SCHEMA_VERSION
        assert d["kind"] == "noise_attribution"
        assert d["label"] == "bell_depol"
        assert d["title"].startswith("Bell")
        assert d["seed"] == 2026
        assert d["circuit"]["num_qubits"] == 2
        assert d["noise_model"]["global"][0]["type"] == "DepolarizingNoise"
        params = d["params"]
        assert params["n_trials"] == 40
        assert params["n_bootstrap"] == 200
        assert params["confidence"] == 0.95
        assert params["fdr_level"] == 0.05

    def test_json_roundtrip(self, small_config):
        raw = small_config.to_json()
        decoded = ReplayConfig.from_json(raw)
        assert decoded.label == small_config.label
        assert decoded.title == small_config.title
        assert decoded.seed == small_config.seed
        assert decoded.circuit == small_config.circuit
        assert decoded.noise_model == small_config.noise_model
        assert decoded.params == small_config.params

    def test_save_and_load(self, small_config, tmp_path):
        path = tmp_path / "bell_depol.json"
        small_config.save(path)
        assert path.exists() and path.stat().st_size > 100

        loaded = ReplayConfig.load(path)
        assert loaded.to_dict() == small_config.to_dict()

    def test_unsupported_version_rejected(self, tmp_path):
        bogus = tmp_path / "bogus.json"
        bogus.write_text(
            json.dumps({"version": "999.0", "circuit": {}, "params": {}}),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="Unsupported"):
            ReplayConfig.load(bogus)

    def test_unsupported_kind_rejected(self, small_config, tmp_path):
        d = small_config.to_dict()
        d["kind"] = "not_a_real_kind"
        bogus = tmp_path / "k.json"
        bogus.write_text(json.dumps(d), encoding="utf-8")
        with pytest.raises(ValueError, match="Unknown"):
            ReplayConfig.load(bogus)

    def test_noiseless_config_supported(self, small_circuit):
        cfg = ReplayConfig.from_current(
            label="noiseless",
            title="noiseless",
            circuit=small_circuit,
            noise_model=None,
            params=ReplayParams(n_trials=20, n_bootstrap=100),
            seed=1,
        )
        d = cfg.to_dict()
        assert d["noise_model"] is None
        decoded = ReplayConfig.from_json(cfg.to_json())
        assert decoded.noise_model is None


# ---------------------------------------------------------------------------
# Replay end-to-end
# ---------------------------------------------------------------------------


def _read_csv_rows(path: Path) -> list[list[str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.reader(f))


class TestReplay:

    def test_writes_pdf_png_csv(self, small_config, tmp_path):
        res = replay(small_config, tmp_path)
        assert res.output_pdf.exists() and res.output_pdf.stat().st_size > 1000
        assert res.output_png.exists() and res.output_png.stat().st_size > 1000
        assert res.output_csv.exists() and res.output_csv.stat().st_size > 100
        assert res.n_columns == 2
        assert res.elapsed_seconds > 0

    def test_skip_outputs(self, small_config, tmp_path):
        res = replay(
            small_config, tmp_path,
            write_csv=False, write_png=False, write_pdf=True,
        )
        assert res.output_pdf.exists()
        assert not res.output_png.exists()
        assert not res.output_csv.exists()

    def test_replay_is_bit_exact_for_same_config(
        self, small_config, tmp_path,
    ):
        out1 = tmp_path / "run1"
        out2 = tmp_path / "run2"
        replay(small_config, out1)
        replay(small_config, out2)

        rows1 = _read_csv_rows(out1 / f"{small_config.label}.csv")
        rows2 = _read_csv_rows(out2 / f"{small_config.label}.csv")
        assert rows1 == rows2, "CSV diverged between identical replays"

    def test_replay_from_loaded_config_matches_in_memory(
        self, small_config, tmp_path,
    ):
        cfg_path = tmp_path / "cfg.json"
        small_config.save(cfg_path)

        out1 = tmp_path / "from_obj"
        out2 = tmp_path / "from_disk"
        replay(small_config, out1)
        replay(cfg_path, out2)

        rows1 = _read_csv_rows(out1 / f"{small_config.label}.csv")
        rows2 = _read_csv_rows(out2 / f"{small_config.label}.csv")
        assert rows1 == rows2

    def test_replay_self_contained_no_registry(
        self, tmp_path, small_circuit,
    ):
        """A config with a hand-built noise model that the figure-generation
        registry has never seen should still replay correctly."""
        nm = NoiseModel()
        nm.add_gate_noise("CNOT", BitFlipNoise(0.07))
        nm.add_global_noise(DepolarizingNoise(0.01))

        cfg = ReplayConfig.from_current(
            label="custom",
            title="Custom mixed noise",
            circuit=small_circuit,
            noise_model=nm,
            params=ReplayParams(n_trials=30, n_bootstrap=150),
            seed=11,
        )
        cfg_path = tmp_path / "custom.json"
        cfg.save(cfg_path)

        # Wipe memory of the original config, load from disk.
        del cfg
        loaded = ReplayConfig.load(cfg_path)
        res = replay(loaded, tmp_path)
        assert res.output_pdf.exists()
        assert res.n_columns == 2


class TestReplayAll:

    def test_replays_every_json_in_dir(self, small_circuit, tmp_path):
        # Build two tiny configs.
        cfg_dir = tmp_path / "cfgs"
        cfg_dir.mkdir()
        for i, p in enumerate([0.02, 0.05]):
            nm = NoiseModel()
            nm.add_global_noise(DepolarizingNoise(p))
            cfg = ReplayConfig.from_current(
                label=f"bell_p{i}",
                title=f"bell  |  depol p={p}",
                circuit=small_circuit,
                noise_model=nm,
                params=ReplayParams(n_trials=20, n_bootstrap=100),
                seed=42 + i,
            )
            cfg.save(cfg_dir / f"bell_p{i}.json")

        out_dir = tmp_path / "out"
        results = replay_all(cfg_dir, out_dir, csv_dir=cfg_dir)
        assert len(results) == 2
        for res in results:
            assert res.output_pdf.exists()
            assert res.output_png.exists()
            assert res.output_csv.exists()

    def test_empty_directory_raises(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(FileNotFoundError):
            replay_all(empty, tmp_path / "out")


# ---------------------------------------------------------------------------
# CLI subprocess smoke
# ---------------------------------------------------------------------------


class TestReplayCLI:

    def _project_root(self) -> Path:
        return Path(__file__).resolve().parent.parent

    def test_cli_single_config(self, small_config, tmp_path):
        cfg_path = tmp_path / "bell.json"
        small_config.save(cfg_path)
        out_dir = tmp_path / "out"

        result = subprocess.run(
            [
                sys.executable, "-m", "PRISM.replay",
                str(cfg_path),
                "--output", str(out_dir),
                "--csv-dir", str(tmp_path),
                "--no-png",
            ],
            cwd=self._project_root(),
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, (
            f"CLI failed: stderr={result.stderr}"
        )
        assert (out_dir / f"{small_config.label}.pdf").exists()
        assert (tmp_path / f"{small_config.label}.csv").exists()

    def test_cli_missing_config_returns_nonzero(self, tmp_path):
        result = subprocess.run(
            [sys.executable, "-m", "PRISM.replay"],
            cwd=self._project_root(),
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode != 0
        assert "no config" in result.stderr.lower() or "config" in result.stderr.lower()
