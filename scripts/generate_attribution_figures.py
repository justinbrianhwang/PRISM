"""Generate the benchmark attribution figures for the PRISM paper.

Runs ``compute_noise_attribution_with_statistics`` on every
(benchmark circuit, noise channel) pair, saves the three-panel summary
figure to ``paper/figures/`` and a per-column statistics table to
``paper/experiments/`` for inclusion in the paper text.

Usage::

    python scripts/generate_attribution_figures.py
    python scripts/generate_attribution_figures.py --quick    # smaller N for smoke
    python scripts/generate_attribution_figures.py --output other_dir/

The default settings (``n_trials=120``, ``n_bootstrap=1000``) give
publication-grade error bars in roughly a minute on a laptop CPU.

Each output figure file is named
``attr_<circuit>_<noise>.pdf`` and ``.png``; the matching CSV
``attr_<circuit>_<noise>.csv`` carries the per-column point estimates,
CI bounds, p-values, and BH q-values for the table appendix.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable

# Headless backend BEFORE pyplot import -- the script must run on CI
# without a display.
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (deferred for backend)

# Project root on path when the script is run directly.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from PRISM.engine.algorithms import AlgorithmTemplate  # noqa: E402
from PRISM.engine.circuit import GateInstance, QuantumCircuit  # noqa: E402
from PRISM.engine.debugger import CircuitDebugger  # noqa: E402
from PRISM.engine.noise import (  # noqa: E402
    AmplitudeDampingNoise,
    BitFlipNoise,
    DepolarizingNoise,
    NoiseModel,
)
from PRISM.figures import (  # noqa: E402
    attribution_summary_figure,
    save_figure,
    use_paper_style,
)


# ---------------------------------------------------------------------------
# Benchmark circuits -- no Measure gates so attribution doesn't see
# phantom zero-weight columns.
# ---------------------------------------------------------------------------


def _bell() -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits=2)
    qc.add_gate(GateInstance("H", [0], [], 0))
    qc.add_gate(GateInstance("CNOT", [0, 1], [], 1))
    return qc


def _ghz3() -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits=3)
    qc.add_gate(GateInstance("H", [0], [], 0))
    qc.add_gate(GateInstance("CNOT", [0, 1], [], 1))
    qc.add_gate(GateInstance("CNOT", [1, 2], [], 2))
    return qc


def _qft3() -> QuantumCircuit:
    """3-qubit QFT without the trailing measurements."""
    qc = AlgorithmTemplate.quantum_fourier_transform(3)
    # Drop any Measure / Barrier gates if the template added them.
    qc.gates = [
        g for g in qc.gates if g.gate_name not in ("Measure", "Barrier")
    ]
    return qc


def _qaoa() -> QuantumCircuit:
    return AlgorithmTemplate.qaoa_maxcut_4cycle(gamma=0.7, beta=0.4)


def _bit_flip_encoder() -> QuantumCircuit:
    return AlgorithmTemplate.bit_flip_encoder()


CIRCUITS: dict[str, Callable[[], QuantumCircuit]] = {
    "bell": _bell,
    "ghz3": _ghz3,
    "qft3": _qft3,
    "qaoa_maxcut": _qaoa,
    "bit_flip_encoder": _bit_flip_encoder,
}


# ---------------------------------------------------------------------------
# Noise channels
# ---------------------------------------------------------------------------


def _depolarizing_noise(p: float = 0.05) -> NoiseModel:
    nm = NoiseModel()
    nm.add_global_noise(DepolarizingNoise(p))
    return nm


def _bit_flip_noise(p: float = 0.05) -> NoiseModel:
    nm = NoiseModel()
    nm.add_global_noise(BitFlipNoise(p))
    return nm


def _amp_damping_noise(gamma: float = 0.05) -> NoiseModel:
    nm = NoiseModel()
    nm.add_global_noise(AmplitudeDampingNoise(gamma))
    return nm


NOISES: dict[str, Callable[[], NoiseModel]] = {
    "depolarizing": _depolarizing_noise,
    "bit_flip": _bit_flip_noise,
    "amp_damping": _amp_damping_noise,
}


# Pretty labels for figure titles
CIRCUIT_TITLES = {
    "bell": "Bell state",
    "ghz3": "GHZ-3",
    "qft3": "QFT (3 qubits)",
    "qaoa_maxcut": "QAOA MaxCut on C_4",
    "bit_flip_encoder": "Bit-flip encoder [3,1,1]",
}
NOISE_TITLES = {
    "depolarizing": "depolarizing p=0.05",
    "bit_flip": "bit-flip p=0.05",
    "amp_damping": "amplitude damping gamma=0.05",
}


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def _write_csv(path: Path, attr) -> None:
    """Write the per-column table that backs the paper appendix."""
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
            row = [
                i, label,
                f"{mean:.6e}",
            ]
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
            row += [
                f"{attr.column_attribution_pct[i]:.4f}",
            ]
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


def _write_config(path: Path, circuit_name: str, noise_name: str,
                  n_trials: int, n_bootstrap: int, seed: int,
                  confidence: float, fdr_level: float) -> None:
    """Write the JSON config that fully describes how to reproduce the figure."""
    payload = {
        "version": "1.0",
        "kind": "noise_attribution",
        "circuit": circuit_name,
        "noise": noise_name,
        "n_trials": n_trials,
        "n_bootstrap": n_bootstrap,
        "confidence": confidence,
        "fdr_level": fdr_level,
        "seed": seed,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run(
    output_dir: Path,
    n_trials: int,
    n_bootstrap: int,
    seed_base: int,
    confidence: float,
    fdr_level: float,
    only_circuits: list[str] | None = None,
    only_noises: list[str] | None = None,
) -> int:
    fig_dir = output_dir / "figures"
    exp_dir = output_dir / "experiments"
    fig_dir.mkdir(parents=True, exist_ok=True)
    exp_dir.mkdir(parents=True, exist_ok=True)

    use_paper_style()

    debugger = CircuitDebugger()

    circuits = only_circuits or list(CIRCUITS.keys())
    noises = only_noises or list(NOISES.keys())

    total = len(circuits) * len(noises)
    done = 0
    t_start = time.perf_counter()

    for ci, circ_name in enumerate(circuits):
        circuit = CIRCUITS[circ_name]()
        for ni, noise_name in enumerate(noises):
            noise = NOISES[noise_name]()
            # Stable seed per (circuit, noise) -- same input always gives
            # the same figure, paper appendix can quote the seed.
            seed = seed_base + 1000 * ci + ni

            t0 = time.perf_counter()
            attr = debugger.compute_noise_attribution_with_statistics(
                circuit,
                noise,
                n_trials=n_trials,
                n_bootstrap=n_bootstrap,
                confidence=confidence,
                fdr_level=fdr_level,
                seed=seed,
            )
            elapsed = time.perf_counter() - t0

            title = (
                f"{CIRCUIT_TITLES.get(circ_name, circ_name)}  |  "
                f"{NOISE_TITLES.get(noise_name, noise_name)}"
            )
            fig = attribution_summary_figure(attr, title=title)

            stem = f"attr_{circ_name}_{noise_name}"
            save_figure(fig, str(fig_dir / f"{stem}.pdf"))
            save_figure(fig, str(fig_dir / f"{stem}.png"))
            plt.close(fig)

            _write_csv(exp_dir / f"{stem}.csv", attr)
            _write_config(
                exp_dir / f"{stem}.json",
                circuit_name=circ_name,
                noise_name=noise_name,
                n_trials=n_trials,
                n_bootstrap=n_bootstrap,
                seed=seed,
                confidence=confidence,
                fdr_level=fdr_level,
            )

            done += 1
            sig_count = sum(attr.statistics.column_significant)
            recov_count = sum(attr.is_recovery)
            print(
                f"  [{done:2d}/{total}]  {stem:<38s}"
                f"  cols={len(attr.delta_fidelity):2d}"
                f"  sig={sig_count:2d}  recov={recov_count:2d}"
                f"  ({elapsed:5.2f}s)"
            )

    total_elapsed = time.perf_counter() - t_start
    print()
    print(f"Generated {done} figures in {total_elapsed:.1f}s")
    print(f"  PDFs/PNGs : {fig_dir}")
    print(f"  CSVs/JSON : {exp_dir}")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate PRISM benchmark attribution figures.",
    )
    p.add_argument(
        "--output", type=Path,
        default=_PROJECT_ROOT / "paper",
        help="Output root containing figures/ and experiments/ subdirs.",
    )
    p.add_argument("--n-trials", type=int, default=120,
                   help="Number of stochastic trials per attribution.")
    p.add_argument("--n-bootstrap", type=int, default=1000,
                   help="Number of bootstrap resamples.")
    p.add_argument("--confidence", type=float, default=0.95,
                   help="Two-sided CI level.")
    p.add_argument("--fdr-level", type=float, default=0.05,
                   help="Benjamini-Hochberg FDR target.")
    p.add_argument("--seed", type=int, default=20260426,
                   help="Master seed; per-pair seeds are derived from this.")
    p.add_argument("--quick", action="store_true",
                   help="Smoke-mode: n_trials=20, n_bootstrap=200.")
    p.add_argument(
        "--only-circuits", nargs="*",
        choices=list(CIRCUITS.keys()),
        help="Restrict to a subset of benchmark circuits.",
    )
    p.add_argument(
        "--only-noises", nargs="*",
        choices=list(NOISES.keys()),
        help="Restrict to a subset of noise channels.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.quick:
        n_trials = 20
        n_bootstrap = 200
    else:
        n_trials = args.n_trials
        n_bootstrap = args.n_bootstrap

    return run(
        output_dir=args.output,
        n_trials=n_trials,
        n_bootstrap=n_bootstrap,
        seed_base=args.seed,
        confidence=args.confidence,
        fdr_level=args.fdr_level,
        only_circuits=args.only_circuits,
        only_noises=args.only_noises,
    )


if __name__ == "__main__":
    raise SystemExit(main())
