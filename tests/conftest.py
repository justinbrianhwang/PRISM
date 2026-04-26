"""Shared pytest fixtures for the PRISM test suite.

PRISM (Per-gate Reproducible Inference for Stochastic Mechanics) -- the
quantum circuit simulator with statistical noise attribution.

This file is part of the Phase-1B reproducibility infrastructure: the
existing :mod:`test_validation` script-style tests will be migrated
incrementally into this directory, and every new test from Phase 1A
onwards uses these fixtures so that random seeds, default circuits,
and noise configurations stay consistent across the suite.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Make the project package importable when tests are invoked via plain
# ``pytest`` from the repository root.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from PRISM.engine.circuit import GateInstance, QuantumCircuit
from PRISM.engine.noise import (
    BitFlipNoise,
    DepolarizingNoise,
    NoiseModel,
    PhaseFlipNoise,
)


# ---------------------------------------------------------------------------
# RNG fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rng() -> np.random.Generator:
    """Deterministic RNG seeded at 42 for tests that need randomness."""
    return np.random.default_rng(42)


@pytest.fixture
def make_rng():
    """Factory for seeded RNGs.

    Useful when a test needs *several* independent generators with
    distinct, reproducible seeds::

        def test_x(make_rng):
            r1 = make_rng(1)
            r2 = make_rng(2)
    """

    def _factory(seed: int) -> np.random.Generator:
        return np.random.default_rng(seed)

    return _factory


# ---------------------------------------------------------------------------
# Circuit fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def bell_circuit() -> QuantumCircuit:
    """Two-qubit Bell-state circuit: H on q0, then CNOT(0,1)."""
    qc = QuantumCircuit(num_qubits=2)
    qc.add_gate(GateInstance("H", [0], [], 0))
    qc.add_gate(GateInstance("CNOT", [0, 1], [], 1))
    return qc


@pytest.fixture
def ghz3_circuit() -> QuantumCircuit:
    """Three-qubit GHZ-state circuit."""
    qc = QuantumCircuit(num_qubits=3)
    qc.add_gate(GateInstance("H", [0], [], 0))
    qc.add_gate(GateInstance("CNOT", [0, 1], [], 1))
    qc.add_gate(GateInstance("CNOT", [1, 2], [], 2))
    return qc


@pytest.fixture
def deep_random_circuit() -> QuantumCircuit:
    """A 4-qubit depth-6 circuit with a mix of single- and 2-qubit gates.

    Used as a non-trivial substrate for noise-attribution tests where
    the circuit is rich enough that several columns plausibly differ in
    their noise contributions.
    """
    qc = QuantumCircuit(num_qubits=4)
    qc.add_gate(GateInstance("H", [0], [], 0))
    qc.add_gate(GateInstance("H", [1], [], 0))
    qc.add_gate(GateInstance("CNOT", [0, 2], [], 1))
    qc.add_gate(GateInstance("Ry", [3], [0.7], 1))
    qc.add_gate(GateInstance("CNOT", [1, 3], [], 2))
    qc.add_gate(GateInstance("Rz", [0], [1.1], 2))
    qc.add_gate(GateInstance("CNOT", [2, 3], [], 3))
    qc.add_gate(GateInstance("H", [1], [], 3))
    qc.add_gate(GateInstance("CNOT", [0, 1], [], 4))
    qc.add_gate(GateInstance("Rx", [2], [0.4], 4))
    qc.add_gate(GateInstance("H", [3], [], 5))
    return qc


# ---------------------------------------------------------------------------
# Noise-model fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def light_depolarizing_noise() -> NoiseModel:
    """Global depolarizing noise at p=0.01 -- mild realistic noise."""
    nm = NoiseModel()
    nm.add_global_noise(DepolarizingNoise(0.01))
    return nm


@pytest.fixture
def moderate_depolarizing_noise() -> NoiseModel:
    """Global depolarizing noise at p=0.05 -- strong enough to give a
    measurable, statistically tractable attribution signal."""
    nm = NoiseModel()
    nm.add_global_noise(DepolarizingNoise(0.05))
    return nm


@pytest.fixture
def mixed_noise() -> NoiseModel:
    """Mixed noise: bit-flip(0.02) + phase-flip(0.01) globally.

    Useful for attribution tests where the dominant error channel
    differs by gate type, since bit-flip and phase-flip have different
    effects on superposition states.
    """
    nm = NoiseModel()
    nm.add_global_noise(BitFlipNoise(0.02))
    nm.add_global_noise(PhaseFlipNoise(0.01))
    return nm
