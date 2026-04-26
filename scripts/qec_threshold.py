"""QEC Threshold Analysis -- multi-code threshold sweeps with logical error rate.

Usage:
    python scripts/qec_threshold.py --codes bit_flip,steane --seed 42
    python scripts/qec_threshold.py --codes bit_flip,phase_flip,steane --noise bit_flip --output results.json
"""

from __future__ import annotations

import argparse
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from PRISM.engine.qec import (
    QECSimulator, BitFlipCode, PhaseFlipCode, SteaneCode,
)


CODE_MAP = {
    "bit_flip": BitFlipCode,
    "phase_flip": PhaseFlipCode,
    "steane": SteaneCode,
}


def run_threshold(
    code_name: str,
    noise_type: str,
    n_trials: int,
    seed: int,
) -> dict:
    code_cls = CODE_MAP[code_name]
    code = code_cls()
    sim = QECSimulator(code)

    noise_probs = np.linspace(0.001, 0.3, 15).tolist()

    results = sim.threshold_sweep(
        noise_probs,
        n_trials=n_trials,
        noise_type=noise_type,
        seed=seed,
    )

    sweep_data = []
    for tp in results:
        sweep_data.append({
            "physical_rate": tp.physical_rate,
            "logical_rate": tp.logical_rate,
            "success_rate": tp.success_rate,
            "avg_fidelity": tp.avg_fidelity,
            "logical_z_fidelity": tp.logical_z_fidelity,
            "decoder_success_rate": tp.decoder_success_rate,
        })

    # Find approximate threshold (where logical < physical)
    threshold_p = None
    for tp in results:
        if tp.logical_rate < tp.physical_rate:
            threshold_p = tp.physical_rate

    return {
        "code": code.name,
        "code_key": code_name,
        "noise_type": noise_type,
        "n_trials": n_trials,
        "data_qubits": code.data_qubits,
        "total_qubits": code.total_qubits,
        "code_distance": code.code_distance,
        "estimated_threshold": threshold_p,
        "sweep": sweep_data,
    }


def main():
    parser = argparse.ArgumentParser(description="QEC threshold analysis")
    parser.add_argument(
        "--codes", type=str, default="bit_flip,steane",
        help="Comma-separated code names: bit_flip, phase_flip, steane",
    )
    parser.add_argument("--noise", type=str, default="bit_flip",
                        choices=["bit_flip", "phase_flip", "depolarizing"])
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    code_names = [c.strip() for c in args.codes.split(",")]
    for name in code_names:
        if name not in CODE_MAP:
            print(f"Unknown code: {name}. Available: {list(CODE_MAP.keys())}")
            sys.exit(1)

    print(f"Running QEC threshold: codes={code_names}, noise={args.noise}, "
          f"trials={args.trials}, seed={args.seed}")

    all_results = []
    for code_name in code_names:
        print(f"  Sweeping {code_name}...")
        result = run_threshold(code_name, args.noise, args.trials, args.seed)
        all_results.append(result)
        est = result["estimated_threshold"]
        print(f"    {result['code']}: threshold ~ {est if est else 'N/A'}")

    output = {
        "experiment": "qec_threshold",
        "noise_type": args.noise,
        "seed": args.seed,
        "codes": all_results,
    }

    if args.output:
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Results saved to {args.output}")
    else:
        print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
