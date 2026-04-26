"""Headless experiment replay for PRISM paper figures.

Every figure in :file:`paper/figures/` has a matching JSON config in
:file:`paper/experiments/`.  The config is *self-contained*: it ships
the full serialised circuit, the full serialised noise model, the
random seed, and every numerical knob (``n_trials``, ``n_bootstrap``,
``confidence``, ``fdr_level``).  Running ``replay`` against the config
reconstructs the experiment from these fields alone -- no name lookup
into a script registry, no environment dependence -- and writes the
exact same figure (and per-column CSV) as the original generation.

This module is the reproducibility appendix of the paper, in code
form.  Reviewers who clone the repository and ``pip install -r
requirements.txt`` can do::

    python -m PRISM.replay paper/experiments/attr_bell_depolarizing.json

and rebuild any single figure, or::

    python -m PRISM.replay --all paper/experiments/

to rebuild the whole set.

Schema
------

The on-disk format is JSON with the following top-level keys
(``version`` ``"2.0"``):

* ``kind``           -- ``"noise_attribution"`` (only kind for now).
* ``label``          -- short slug used to derive output filenames.
* ``title``          -- human-readable figure title.
* ``circuit``        -- :pymeth:`QuantumCircuit.to_dict` payload.
* ``noise_model``    -- :pymeth:`NoiseModel.to_dict` payload, or
  ``null`` for noiseless experiments.
* ``params``         -- ``{n_trials, n_bootstrap, confidence, fdr_level}``.
* ``seed``           -- master integer seed.
* ``prism_version``  -- recorded for diagnostic purposes.
* ``generated_at``   -- ISO-8601 UTC timestamp of original generation.

Older v1 configs (which only stored circuit / noise *names* keyed into
the figure-generation script) are supported via
:func:`ReplayConfig.from_v1_dict` -- both forms decode to the same
runtime object.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

# Headless backend BEFORE pyplot.
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from PRISM.engine.circuit import QuantumCircuit  # noqa: E402
from PRISM.engine.debugger import CircuitDebugger, NoiseAttribution  # noqa: E402
from PRISM.engine.noise import NoiseModel  # noqa: E402
from PRISM.figures import (  # noqa: E402
    attribution_summary_figure,
    save_figure,
    use_paper_style,
)

if TYPE_CHECKING:
    from matplotlib.figure import Figure


PRISM_VERSION = "1.0.0"
SCHEMA_VERSION = "2.0"
SUPPORTED_KINDS = {"noise_attribution"}


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------


@dataclass
class ReplayParams:
    """Numerical knobs for the noise-attribution experiment."""

    n_trials: int = 120
    n_bootstrap: int = 1000
    confidence: float = 0.95
    fdr_level: float = 0.05


@dataclass
class ReplayConfig:
    """Self-contained record of one paper-figure experiment.

    Built either via :meth:`from_current` at generation time or
    :meth:`load` / :meth:`from_dict` at replay time.
    """

    label: str
    title: str
    circuit: dict
    noise_model: dict | None
    params: ReplayParams
    seed: int
    kind: str = "noise_attribution"
    version: str = SCHEMA_VERSION
    prism_version: str = PRISM_VERSION
    generated_at: str = ""

    # ---- Construction ----

    @classmethod
    def from_current(
        cls,
        label: str,
        title: str,
        circuit: QuantumCircuit,
        noise_model: NoiseModel | None,
        params: ReplayParams,
        seed: int,
    ) -> ReplayConfig:
        """Build a :class:`ReplayConfig` from in-memory experiment objects."""
        return cls(
            label=label,
            title=title,
            circuit=circuit.to_dict(),
            noise_model=noise_model.to_dict() if noise_model is not None else None,
            params=params,
            seed=int(seed),
            kind="noise_attribution",
            version=SCHEMA_VERSION,
            prism_version=PRISM_VERSION,
            generated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        )

    # ---- Serialisation ----

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "kind": self.kind,
            "label": self.label,
            "title": self.title,
            "circuit": self.circuit,
            "noise_model": self.noise_model,
            "params": asdict(self.params),
            "seed": self.seed,
            "prism_version": self.prism_version,
            "generated_at": self.generated_at,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")

    # ---- Deserialisation ----

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReplayConfig:
        version = data.get("version", "1.0")
        if version.startswith("1."):
            return cls.from_v1_dict(data)
        if not version.startswith("2."):
            raise ValueError(
                f"Unsupported replay config version: {version!r}. "
                "Expected '2.x'."
            )

        kind = data.get("kind", "noise_attribution")
        if kind not in SUPPORTED_KINDS:
            raise ValueError(
                f"Unknown replay config kind: {kind!r}. "
                f"Supported: {sorted(SUPPORTED_KINDS)}."
            )

        params_raw = data.get("params", {})
        params = ReplayParams(
            n_trials=int(params_raw.get("n_trials", 120)),
            n_bootstrap=int(params_raw.get("n_bootstrap", 1000)),
            confidence=float(params_raw.get("confidence", 0.95)),
            fdr_level=float(params_raw.get("fdr_level", 0.05)),
        )

        return cls(
            label=str(data.get("label", "experiment")),
            title=str(data.get("title", data.get("label", ""))),
            circuit=data["circuit"],
            noise_model=data.get("noise_model"),
            params=params,
            seed=int(data.get("seed", 0)),
            kind=kind,
            version=version,
            prism_version=str(data.get("prism_version", "")),
            generated_at=str(data.get("generated_at", "")),
        )

    @classmethod
    def from_v1_dict(cls, data: dict[str, Any]) -> ReplayConfig:
        """Decode a legacy v1 config that stored circuit / noise *names*.

        v1 configs reference :file:`scripts/generate_attribution_figures.py`'s
        in-script registries to look up the actual circuit and noise
        model.  This is brittle (the registries can drift over time) so
        v2 was introduced; we keep v1 support for older experiments
        already on disk.
        """
        from scripts.generate_attribution_figures import (  # noqa: WPS433
            CIRCUIT_TITLES,
            CIRCUITS,
            NOISES,
            NOISE_TITLES,
        )

        circuit_name = data.get("circuit")
        noise_name = data.get("noise")
        if circuit_name not in CIRCUITS:
            raise ValueError(
                f"v1 config references unknown circuit {circuit_name!r}."
            )
        if noise_name not in NOISES:
            raise ValueError(
                f"v1 config references unknown noise {noise_name!r}."
            )

        circuit = CIRCUITS[circuit_name]()
        noise = NOISES[noise_name]()
        params = ReplayParams(
            n_trials=int(data.get("n_trials", 120)),
            n_bootstrap=int(data.get("n_bootstrap", 1000)),
            confidence=float(data.get("confidence", 0.95)),
            fdr_level=float(data.get("fdr_level", 0.05)),
        )
        title = (
            f"{CIRCUIT_TITLES.get(circuit_name, circuit_name)}  |  "
            f"{NOISE_TITLES.get(noise_name, noise_name)}"
        )
        return cls.from_current(
            label=f"attr_{circuit_name}_{noise_name}",
            title=title,
            circuit=circuit,
            noise_model=noise,
            params=params,
            seed=int(data.get("seed", 0)),
        )

    @classmethod
    def from_json(cls, raw: str) -> ReplayConfig:
        return cls.from_dict(json.loads(raw))

    @classmethod
    def load(cls, path: str | Path) -> ReplayConfig:
        return cls.from_json(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Replay engine
# ---------------------------------------------------------------------------


@dataclass
class ReplayResult:
    """Lightweight summary of one replay run (for ``--all`` reporting)."""

    label: str
    output_pdf: Path
    output_png: Path
    output_csv: Path
    n_columns: int
    n_significant: int
    n_recovery: int
    elapsed_seconds: float


def replay(
    config: ReplayConfig | str | Path,
    output_dir: str | Path,
    write_csv: bool = True,
    write_png: bool = True,
    write_pdf: bool = True,
) -> ReplayResult:
    """Reconstruct one experiment from a config and write its outputs.

    Parameters
    ----------
    config : ReplayConfig | str | Path
        Either an in-memory :class:`ReplayConfig` or a path to a JSON
        config on disk.
    output_dir : str | Path
        Directory where ``<label>.pdf`` / ``.png`` / ``.csv`` will be
        written.  Created if missing.
    write_csv, write_png, write_pdf : bool
        Toggle individual outputs.  Useful for fast smoke runs that
        only need the PDF.

    Returns
    -------
    ReplayResult
    """
    if not isinstance(config, ReplayConfig):
        config = ReplayConfig.load(config)

    if config.kind != "noise_attribution":
        raise ValueError(f"replay() does not yet support kind={config.kind!r}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    circuit = QuantumCircuit.from_dict(config.circuit)
    noise = (
        NoiseModel.from_dict(config.noise_model)
        if config.noise_model is not None
        else None
    )

    use_paper_style()

    debugger = CircuitDebugger()
    t0 = time.perf_counter()
    attr = debugger.compute_noise_attribution_with_statistics(
        circuit,
        noise,
        n_trials=config.params.n_trials,
        n_bootstrap=config.params.n_bootstrap,
        confidence=config.params.confidence,
        fdr_level=config.params.fdr_level,
        seed=config.seed,
    )
    elapsed = time.perf_counter() - t0

    pdf_path = output_dir / f"{config.label}.pdf"
    png_path = output_dir / f"{config.label}.png"
    csv_path = output_dir / f"{config.label}.csv"

    fig = attribution_summary_figure(attr, title=config.title)
    if write_pdf:
        save_figure(fig, str(pdf_path))
    if write_png:
        save_figure(fig, str(png_path))
    plt.close(fig)

    if write_csv:
        _write_attribution_csv(csv_path, attr)

    return ReplayResult(
        label=config.label,
        output_pdf=pdf_path,
        output_png=png_path,
        output_csv=csv_path,
        n_columns=len(attr.delta_fidelity),
        n_significant=int(sum(attr.statistics.column_significant)),
        n_recovery=int(sum(attr.is_recovery)),
        elapsed_seconds=elapsed,
    )


def replay_all(
    config_dir: str | Path,
    output_dir: str | Path,
    pattern: str = "*.json",
    csv_dir: str | Path | None = None,
) -> list[ReplayResult]:
    """Replay every JSON config under ``config_dir`` matching ``pattern``.

    Figures (PDF + PNG) go to ``output_dir``.  CSVs go to
    ``csv_dir`` if provided, else next to the configs.
    """
    config_dir = Path(config_dir)
    output_dir = Path(output_dir)
    csv_target = Path(csv_dir) if csv_dir is not None else config_dir
    csv_target.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    configs = sorted(config_dir.glob(pattern))
    if not configs:
        raise FileNotFoundError(
            f"No configs matched {pattern!r} under {config_dir}"
        )

    results: list[ReplayResult] = []
    total = len(configs)
    for i, cfg_path in enumerate(configs, 1):
        cfg = ReplayConfig.load(cfg_path)
        # Render figure into output_dir, but place CSV alongside the config
        # by default so generation and replay share the same layout.
        res = replay(cfg, output_dir, write_csv=False)
        if csv_target != output_dir:
            csv_path = csv_target / f"{cfg.label}.csv"
            _replay_csv_only(cfg, csv_path)
            res = ReplayResult(
                label=res.label,
                output_pdf=res.output_pdf,
                output_png=res.output_png,
                output_csv=csv_path,
                n_columns=res.n_columns,
                n_significant=res.n_significant,
                n_recovery=res.n_recovery,
                elapsed_seconds=res.elapsed_seconds,
            )
        else:
            csv_path = output_dir / f"{cfg.label}.csv"
            _replay_csv_only(cfg, csv_path)
            res = ReplayResult(
                label=res.label,
                output_pdf=res.output_pdf,
                output_png=res.output_png,
                output_csv=csv_path,
                n_columns=res.n_columns,
                n_significant=res.n_significant,
                n_recovery=res.n_recovery,
                elapsed_seconds=res.elapsed_seconds,
            )
        results.append(res)
        print(
            f"  [{i:2d}/{total}]  {res.label:<40s}"
            f"  cols={res.n_columns:2d}"
            f"  sig={res.n_significant:2d}  recov={res.n_recovery:2d}"
            f"  ({res.elapsed_seconds:5.2f}s)"
        )

    return results


def _replay_csv_only(config: ReplayConfig, csv_path: Path) -> None:
    """Recompute the attribution and write its CSV table.

    Used by :func:`replay_all` when CSVs go to a directory distinct
    from the figure directory.  The replay is deterministic so calling
    this on top of :func:`replay` with the same config produces the
    same numbers.
    """
    circuit = QuantumCircuit.from_dict(config.circuit)
    noise = (
        NoiseModel.from_dict(config.noise_model)
        if config.noise_model is not None
        else None
    )
    attr = CircuitDebugger().compute_noise_attribution_with_statistics(
        circuit,
        noise,
        n_trials=config.params.n_trials,
        n_bootstrap=config.params.n_bootstrap,
        confidence=config.params.confidence,
        fdr_level=config.params.fdr_level,
        seed=config.seed,
    )
    _write_attribution_csv(csv_path, attr)


def _write_attribution_csv(path: Path, attr: NoiseAttribution) -> None:
    """Write the per-column statistics table for paper appendix."""
    stats = attr.statistics
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "column", "label",
            "delta_F_mean", "delta_F_ci_lower", "delta_F_ci_upper",
            "p_value", "q_value", "significant",
            "attribution_pct",
            "attribution_pct_ci_lower", "attribution_pct_ci_upper",
            "is_recovery",
            "recovery_rate",
            "recovery_rate_ci_lower", "recovery_rate_ci_upper",
        ])
        for i, mean in enumerate(attr.delta_fidelity):
            label = " + ".join(attr.gate_labels[i]) or f"col {i}"
            row: list[Any] = [i, label, f"{mean:.6e}"]
            if stats is not None:
                row += [
                    f"{stats.delta_fidelity_ci_lower[i]:.6e}",
                    f"{stats.delta_fidelity_ci_upper[i]:.6e}",
                    f"{stats.delta_fidelity_p_value[i]:.6e}",
                    f"{stats.delta_fidelity_q_value[i]:.6e}",
                    int(bool(stats.column_significant[i])),
                ]
            else:
                row += ["", "", "", "", ""]
            row += [f"{attr.column_attribution_pct[i]:.4f}"]
            if stats is not None:
                row += [
                    f"{stats.attribution_pct_ci_lower[i]:.4f}",
                    f"{stats.attribution_pct_ci_upper[i]:.4f}",
                ]
            else:
                row += ["", ""]
            row += [int(bool(attr.is_recovery[i]))]
            if stats is not None:
                row += [
                    f"{stats.recovery_rate[i]:.4f}",
                    f"{stats.recovery_rate_ci_lower[i]:.4f}",
                    f"{stats.recovery_rate_ci_upper[i]:.4f}",
                ]
            else:
                row += ["", "", ""]
            w.writerow(row)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m PRISM.replay",
        description=(
            "Reconstruct a PRISM paper figure from its self-contained "
            "JSON config.  Pass a single config path or use --all to "
            "replay every config in a directory."
        ),
    )
    p.add_argument(
        "config",
        type=Path,
        nargs="?",
        help="Path to one config (or directory when used with --all).",
    )
    p.add_argument(
        "--all", action="store_true",
        help="Treat 'config' as a directory and replay every JSON inside.",
    )
    p.add_argument(
        "--output", type=Path,
        default=None,
        help=(
            "Output directory for figures (and CSVs by default).  "
            "Defaults to the config directory's sibling 'figures/'."
        ),
    )
    p.add_argument(
        "--csv-dir", type=Path,
        default=None,
        help=(
            "Where to write per-column CSV tables; defaults to the "
            "config directory."
        ),
    )
    p.add_argument(
        "--no-csv", action="store_true",
        help="Skip CSV output.",
    )
    p.add_argument(
        "--no-png", action="store_true",
        help="Skip PNG output (PDF only).",
    )
    return p.parse_args(argv)


def _default_output_dir(config_path: Path) -> Path:
    """Default to ``<parent>/figures/`` for a config in ``<parent>``."""
    if config_path.is_dir():
        return config_path.parent / "figures"
    return config_path.parent.parent / "figures"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.config is None:
        print(
            "error: no config supplied. "
            "Pass a JSON path, or pass a directory with --all.",
            file=sys.stderr,
        )
        return 2

    if args.all:
        if not args.config.is_dir():
            print(
                f"error: --all expects a directory, got {args.config}",
                file=sys.stderr,
            )
            return 2
        output_dir = args.output or _default_output_dir(args.config)
        csv_dir = args.csv_dir if args.csv_dir is not None else args.config
        t0 = time.perf_counter()
        results = replay_all(args.config, output_dir, csv_dir=csv_dir)
        elapsed = time.perf_counter() - t0
        print()
        print(f"Replayed {len(results)} configs in {elapsed:.1f}s")
        print(f"  PDFs/PNGs : {output_dir}")
        print(f"  CSVs      : {csv_dir}")
        return 0

    if not args.config.is_file():
        print(f"error: config not found: {args.config}", file=sys.stderr)
        return 2

    output_dir = args.output or _default_output_dir(args.config)
    csv_dir = args.csv_dir if args.csv_dir is not None else args.config.parent

    t0 = time.perf_counter()
    res = replay(
        args.config,
        output_dir,
        write_csv=False,  # write to csv_dir instead
        write_png=not args.no_png,
    )
    if not args.no_csv:
        _replay_csv_only(
            ReplayConfig.load(args.config),
            csv_dir / f"{res.label}.csv",
        )
    elapsed = time.perf_counter() - t0
    print(
        f"  {res.label}  cols={res.n_columns}  sig={res.n_significant}"
        f"  recov={res.n_recovery}  ({elapsed:.2f}s)"
    )
    print(f"  PDF  : {res.output_pdf}")
    if not args.no_png:
        print(f"  PNG  : {res.output_png}")
    if not args.no_csv:
        print(f"  CSV  : {csv_dir / (res.label + '.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
