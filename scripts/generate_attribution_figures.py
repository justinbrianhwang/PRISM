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
CI bounds, p-values, and BH q-values for the table appendix; the
matching ``attr_<circuit>_<noise>.json`` is a self-contained
:class:`PRISM.replay.ReplayConfig` that any reviewer can re-run via
``python -m PRISM.replay <config.json>`` to bit-exactly reproduce the
figure.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Callable

# Headless backend BEFORE pyplot import -- the script must run on CI
# without a display.
import matplotlib
matplotlib.use("Agg")

# Project root on path when the script is run directly.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from PRISM.engine.algorithms import AlgorithmTemplate  # noqa: E402
from PRISM.engine.circuit import GateInstance, QuantumCircuit  # noqa: E402
from PRISM.engine.noise import (  # noqa: E402
    AmplitudeDampingNoise,
    BitFlipNoise,
    DepolarizingNoise,
    NoiseModel,
    PhaseFlipNoise,
)
from PRISM.figures import use_paper_style  # noqa: E402
from PRISM.replay import (  # noqa: E402
    ReplayConfig,
    ReplayParams,
    replay,
)


# ---------------------------------------------------------------------------
# Benchmark circuits -- no Measure gates so attribution doesn't see
# phantom zero-weight columns.
# ---------------------------------------------------------------------------


def _drop_classical_gates(qc: QuantumCircuit) -> QuantumCircuit:
    """Strip Measure / Barrier instances so attribution does not get
    phantom zero-weight columns at the tail of every plot."""
    qc.gates = [
        g for g in qc.gates if g.gate_name not in ("Measure", "Barrier")
    ]
    return qc


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


def _ghz4() -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits=4)
    qc.add_gate(GateInstance("H", [0], [], 0))
    qc.add_gate(GateInstance("CNOT", [0, 1], [], 1))
    qc.add_gate(GateInstance("CNOT", [1, 2], [], 2))
    qc.add_gate(GateInstance("CNOT", [2, 3], [], 3))
    return qc


def _qft3() -> QuantumCircuit:
    """3-qubit QFT without the trailing measurements."""
    return _drop_classical_gates(AlgorithmTemplate.quantum_fourier_transform(3))


def _qft4() -> QuantumCircuit:
    """4-qubit QFT without the trailing measurements."""
    return _drop_classical_gates(AlgorithmTemplate.quantum_fourier_transform(4))


def _qaoa() -> QuantumCircuit:
    return AlgorithmTemplate.qaoa_maxcut_4cycle(gamma=0.7, beta=0.4)


def _bit_flip_encoder() -> QuantumCircuit:
    return AlgorithmTemplate.bit_flip_encoder()


def _bernstein_vazirani_3() -> QuantumCircuit:
    """Bernstein-Vazirani for the secret string '101' over 3 input qubits.

    The template adds a final Hadamard layer on the inputs followed by
    measurements; we strip the measurements so attribution sees only the
    quantum part.  4 qubits total (3 input + 1 ancilla).
    """
    return _drop_classical_gates(
        AlgorithmTemplate.bernstein_vazirani(secret="101")
    )


CIRCUITS: dict[str, Callable[[], QuantumCircuit]] = {
    "bell": _bell,
    "ghz3": _ghz3,
    "ghz4": _ghz4,
    "qft3": _qft3,
    "qft4": _qft4,
    "qaoa_maxcut": _qaoa,
    "bit_flip_encoder": _bit_flip_encoder,
    "bernstein_vazirani_3": _bernstein_vazirani_3,
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


def _phase_flip_noise(p: float = 0.05) -> NoiseModel:
    nm = NoiseModel()
    nm.add_global_noise(PhaseFlipNoise(p))
    return nm


def _amp_damping_noise(gamma: float = 0.05) -> NoiseModel:
    nm = NoiseModel()
    nm.add_global_noise(AmplitudeDampingNoise(gamma))
    return nm


NOISES: dict[str, Callable[[], NoiseModel]] = {
    "depolarizing": _depolarizing_noise,
    "bit_flip": _bit_flip_noise,
    "phase_flip": _phase_flip_noise,
    "amp_damping": _amp_damping_noise,
}


# Pretty labels for figure titles (kept in the JSON config for posterity
# even though the rendered figures are now title-less).
CIRCUIT_TITLES = {
    "bell": "Bell state",
    "ghz3": "GHZ-3",
    "ghz4": "GHZ-4",
    "qft3": "QFT (3 qubits)",
    "qft4": "QFT (4 qubits)",
    "qaoa_maxcut": "QAOA MaxCut on C_4",
    "bit_flip_encoder": "Bit-flip encoder [3,1,1]",
    "bernstein_vazirani_3": "Bernstein-Vazirani (secret 101)",
}
NOISE_TITLES = {
    "depolarizing": "depolarizing p=0.05",
    "bit_flip": "bit-flip p=0.05",
    "phase_flip": "phase-flip p=0.05",
    "amp_damping": "amplitude damping gamma=0.05",
}


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
    with_png: bool = False,
) -> int:
    fig_dir = output_dir / "figures"
    exp_dir = output_dir / "experiments"
    fig_dir.mkdir(parents=True, exist_ok=True)
    exp_dir.mkdir(parents=True, exist_ok=True)

    use_paper_style()

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
            stem = f"attr_{circ_name}_{noise_name}"
            title = (
                f"{CIRCUIT_TITLES.get(circ_name, circ_name)}  |  "
                f"{NOISE_TITLES.get(noise_name, noise_name)}"
            )

            # Build the self-contained replay config first, then run the
            # experiment via PRISM.replay so the generation path and the
            # `python -m PRISM.replay` path share an implementation.
            config = ReplayConfig.from_current(
                label=stem,
                title=title,
                circuit=circuit,
                noise_model=noise,
                params=ReplayParams(
                    n_trials=n_trials,
                    n_bootstrap=n_bootstrap,
                    confidence=confidence,
                    fdr_level=fdr_level,
                ),
                seed=seed,
            )
            config.save(exp_dir / f"{stem}.json")

            res = replay(
                config, fig_dir, write_csv=False, write_png=with_png,
            )
            # CSV lives next to the JSON config, not in the figure dir,
            # because the appendix tables are organised alongside the
            # configs they correspond to.
            from PRISM.replay import _replay_csv_only
            _replay_csv_only(config, exp_dir / f"{stem}.csv")

            done += 1
            print(
                f"  [{done:2d}/{total}]  {stem:<38s}"
                f"  cols={res.n_columns:2d}"
                f"  sig={res.n_significant:2d}  recov={res.n_recovery:2d}"
                f"  ({res.elapsed_seconds:5.2f}s)"
            )

    total_elapsed = time.perf_counter() - t_start
    print()
    print(f"Generated {done} figures in {total_elapsed:.1f}s")
    print(f"  PDFs      : {fig_dir}")
    if with_png:
        print(f"  PNGs      : {fig_dir}")
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
    p.add_argument(
        "--with-png", action="store_true",
        help=(
            "Also emit a 300-DPI PNG raster alongside each PDF.  Off "
            "by default -- PDF is the canonical paper artefact, PNG is "
            "only useful for chat / web previews."
        ),
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
        with_png=args.with_png,
    )


if __name__ == "__main__":
    raise SystemExit(main())
