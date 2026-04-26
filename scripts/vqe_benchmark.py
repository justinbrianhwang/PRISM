"""VQE Optimization Benchmark -- parameterized circuit optimization performance.

Usage:
    python scripts/vqe_benchmark.py --qubits 2 --iters 50 --seed 42
    python scripts/vqe_benchmark.py --qubits 3 --hamiltonian heisenberg --layers 3 --output results.json
"""

from __future__ import annotations

import argparse
import json
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from PRISM.engine.circuit import QuantumCircuit, GateInstance
from PRISM.engine.optimizer import (
    ParameterizedCircuitConfig,
    CostFunction,
    CircuitOptimizer,
)


def _build_ansatz(n_qubits: int, n_layers: int) -> QuantumCircuit:
    """Build a hardware-efficient ansatz with Ry rotations and CNOT entanglers."""
    c = QuantumCircuit(n_qubits)
    col = 0
    for layer in range(n_layers):
        # Ry rotations on each qubit
        for q in range(n_qubits):
            c.add_gate(GateInstance("Ry", [q], [0.0], column=col))
        col += 1
        # CNOT chain
        for q in range(n_qubits - 1):
            c.add_gate(GateInstance("CNOT", [q, q + 1], [], column=col))
        col += 1
    # Final Ry layer
    for q in range(n_qubits):
        c.add_gate(GateInstance("Ry", [q], [0.0], column=col))
    return c


HAMILTONIANS = {
    "z0": lambda n: CostFunction.z_expectation(0),
    "zz": lambda n: CostFunction.vqe_hamiltonian([
        (-1.0, "ZZ", [i, i + 1]) for i in range(n - 1)
    ]),
    "heisenberg": lambda n: CostFunction.vqe_hamiltonian(
        [(-1.0, "XX", [i, i + 1]) for i in range(n - 1)]
        + [(-1.0, "YY", [i, i + 1]) for i in range(n - 1)]
        + [(-1.0, "ZZ", [i, i + 1]) for i in range(n - 1)]
    ),
}


def run_benchmark(
    n_qubits: int,
    n_layers: int,
    hamiltonian_name: str,
    lr: float,
    max_iters: int,
    seed: int,
) -> dict:
    circuit = _build_ansatz(n_qubits, n_layers)
    config = ParameterizedCircuitConfig.auto_detect(circuit)

    cost_fn = HAMILTONIANS[hamiltonian_name](n_qubits)

    # Randomize initial parameters
    rng = np.random.default_rng(seed)
    init_vals = rng.uniform(-np.pi, np.pi, size=config.num_params)
    for i, b in enumerate(config.bindings):
        circuit.gates[b.gate_index].params[b.param_index] = float(init_vals[i])
    config = ParameterizedCircuitConfig.auto_detect(circuit)

    optimizer = CircuitOptimizer(
        config=config,
        cost_fn=cost_fn,
        learning_rate=lr,
        max_iterations=max_iters,
    )

    t0 = time.perf_counter()
    result = optimizer.run(seed=seed)
    elapsed = time.perf_counter() - t0

    cost_trace = [float(h[1]) for h in result.history]

    return {
        "n_qubits": n_qubits,
        "n_layers": n_layers,
        "n_params": config.num_params,
        "hamiltonian": hamiltonian_name,
        "learning_rate": lr,
        "max_iterations": max_iters,
        "actual_iterations": result.iterations,
        "converged": result.converged,
        "optimal_cost": float(result.optimal_cost),
        "initial_cost": cost_trace[0] if cost_trace else None,
        "cost_improvement": (cost_trace[0] - result.optimal_cost) if cost_trace else 0,
        "elapsed_seconds": round(elapsed, 3),
        "cost_trace": cost_trace,
    }


def main():
    parser = argparse.ArgumentParser(description="VQE optimization benchmark")
    parser.add_argument("--qubits", type=int, default=2)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--hamiltonian", choices=list(HAMILTONIANS.keys()), default="z0")
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    print(f"Running VQE benchmark: qubits={args.qubits}, layers={args.layers}, "
          f"H={args.hamiltonian}, lr={args.lr}, iters={args.iters}, seed={args.seed}")

    result = run_benchmark(
        args.qubits, args.layers, args.hamiltonian,
        args.lr, args.iters, args.seed,
    )

    output = {
        "experiment": "vqe_benchmark",
        "seed": args.seed,
        "result": result,
    }

    if args.output:
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Results saved to {args.output}")
    else:
        print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
