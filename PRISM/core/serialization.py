"""JSON save/load for quantum circuits."""

from __future__ import annotations

import json
from pathlib import Path

from quantum_sim.engine.circuit import QuantumCircuit


class CircuitSerializer:
    """JSON save/load for quantum circuits."""

    FILE_VERSION = "1.0"
    FILE_EXTENSION = ".qsim"

    @staticmethod
    def save(circuit: QuantumCircuit, filepath: Path | str):
        filepath = Path(filepath)
        data = circuit.to_dict()
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @staticmethod
    def load(filepath: Path | str) -> QuantumCircuit:
        filepath = Path(filepath)
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return QuantumCircuit.from_dict(data)
