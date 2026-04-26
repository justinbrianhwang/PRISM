"""Noise probability sweep -- measure fidelity, entropy, purity vs noise rate.

Usage:
    python scripts/noise_sweep.py --circuit bell --noise depolarizing --seed 42
    python scripts/noise_sweep.py --circuit ghz3 --noise bit_flip --min-p 0.0 --max-p 0.3 --output results.json
"""

from __future__ import annotations

import argparse
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from PRISM.engine.circuit import QuantumCircuit, GateInstance
from PRISM.engine.simulator import Simulator
from PRISM.engine.noise import (
    NoiseModel, BitFlipNoise, PhaseFlipNoise, DepolarizingNoise,
)
from PRISM.engine.analysis import StateAnalysis


# ---- Predefined circuits --------------------------------------------------

def _bell_circuit() -> QuantumCircuit:
    c = QuantumCircuit(2)
    c.add_gate(GateInstance("H", [0], [], column=0))
    c.add_gate(GateInstance("CNOT", [0, 1], [], column=1))
    return c


def _ghz3_circuit() -> QuantumCircuit:
    c = QuantumCircuit(3)
    c.add_gate(GateInstance("H", [0], [], column=0))
    c.add_gate(GateInstance("CNOT", [0, 1], [], column=1))
    c.add_gate(GateInstance("CNOT", [0, 2], [], column=2))
    return c


def _ghz4_circuit() -> QuantumCircuit:
    c = QuantumCircuit(4)
    c.add_gate(GateInstance("H", [0], [], column=0))
    c.add_gate(GateInstance("CNOT", [0, 1], [], column=1))
    c.add_gate(GateInstance("CNOT", [0, 2], [], column=2))
    c.add_gate(GateInstance("CNOT", [0, 3], [], column=3))
    return c


CIRCUITS = {
    "bell": _bell_circuit,
    "ghz3": _ghz3_circuit,
    "ghz4": _ghz4_circuit,
}

NOISE_TYPES = {
    "bit_flip": BitFlipNoise,
    "phase_flip": PhaseFlipNoise,
    "depolarizing": DepolarizingNoise,
}


def run_sweep(
    circuit: QuantumCircuit,
    noise_cls,
    probabilities: np.ndarray,
    n_trials: int,
    seed: int,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    results = []

    # Get ideal state for reference
    sim_ideal = Simulator()
    ideal_result = sim_ideal.run(circuit, shots=0, seed=seed)
    ideal_data = ideal_result.final_state.data

    for p in probabilities:
        fid_acc = 0.0
        ent_acc = 0.0
        pur_acc = 0.0

        for trial in range(n_trials):
            trial_seed = int(rng.integers(0, 2**63))
            nm = NoiseModel()
            nm.add_global_noise(noise_cls(float(p)))
            nm.set_seed(trial_seed)

            sim = Simulator(noise_model=nm)
            result = sim.run(circuit, shots=0, seed=trial_seed)
            state = result.final_state

            fid_acc += StateAnalysis.state_fidelity(ideal_data, state.data)
            ent_acc += StateAnalysis.von_neumann_entropy(state)
            pur_acc += StateAnalysis.purity(state)

        results.append({
            "noise_prob": float(p),
            "mean_fidelity": fid_acc / n_trials,
            "mean_entropy": ent_acc / n_trials,
            "mean_purity": pur_acc / n_trials,
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="Noise probability sweep experiment")
    parser.add_argument("--circuit", choices=list(CIRCUITS.keys()), default="bell")
    parser.add_argument("--noise", choices=list(NOISE_TYPES.keys()), default="depolarizing")
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--max-p", type=float, default=0.3)
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    circuit = CIRCUITS[args.circuit]()
    noise_cls = NOISE_TYPES[args.noise]
    probs = np.linspace(args.min_p, args.max_p, args.steps)

    print(f"Running noise sweep: circuit={args.circuit}, noise={args.noise}, "
          f"p=[{args.min_p:.3f}, {args.max_p:.3f}], "
          f"steps={args.steps}, trials={args.trials}, seed={args.seed}")

    results = run_sweep(circuit, noise_cls, probs, args.trials, args.seed)

    output = {
        "experiment": "noise_sweep",
        "circuit": args.circuit,
        "noise_type": args.noise,
        "n_trials": args.trials,
        "seed": args.seed,
        "results": results,
    }

    if args.output:
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Results saved to {args.output}")
    else:
        print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
