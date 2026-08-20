"""Analytic cross-validation of the trajectory attribution estimator.

For single-qubit channels acting on known states the expected
fidelity-gap contribution has a closed form, so the Monte Carlo
trajectory estimate can be checked against exact quantum-channel
algebra rather than against another simulation:

* Depolarizing(p) after H on |0>:  the state is |+>.  X leaves |+>
  invariant; Y and Z map it to (a phase times) |->, which is
  orthogonal.  Hence E[gap] = (p/3) * 0 + (p/3) * 1 + (p/3) * 1
  = 2p/3.
* BitFlip(p) after X on |0>:  the state is |1>; a bit flip maps it to
  the orthogonal |0>.  Hence E[gap] = p.

These are the strongest available checks that the per-column estimand
targets the CPTP-channel expectation value, complementing the
simulation-level ground-truth injection experiment
(scripts/ground_truth_injection.py).
"""

from __future__ import annotations

import numpy as np

from PRISM.engine.circuit import GateInstance, QuantumCircuit
from PRISM.engine.debugger import CircuitDebugger
from PRISM.engine.noise import BitFlipNoise, DepolarizingNoise, NoiseModel


def _mean_gap(circuit, noise, n_trials=4000, seed=42):
    attr = CircuitDebugger().compute_noise_attribution(
        circuit, noise, n_trials=n_trials, seed=seed,
    )
    return float(attr.delta_fidelity[0])


class TestAnalyticExpectations:

    def test_depolarizing_on_plus_state(self):
        # E[gap] = 2p/3 exactly.
        p = 0.3
        qc = QuantumCircuit(num_qubits=1)
        qc.add_gate(GateInstance("H", [0], [], 0))
        nm = NoiseModel()
        nm.add_global_noise(DepolarizingNoise(p))

        est = _mean_gap(qc, nm)
        expected = 2.0 * p / 3.0
        # Bernoulli(2p/3) mean over 4000 trials: 3 sigma ~ 0.019.
        assert abs(est - expected) < 0.02, (est, expected)

    def test_bit_flip_on_one_state(self):
        # E[gap] = p exactly.
        p = 0.25
        qc = QuantumCircuit(num_qubits=1)
        qc.add_gate(GateInstance("X", [0], [], 0))
        nm = NoiseModel()
        nm.add_global_noise(BitFlipNoise(p))

        est = _mean_gap(qc, nm)
        assert abs(est - p) < 0.021, (est, p)

    def test_depolarizing_strength_sweep_tracks_analytic(self):
        # The estimator must track 2p/3 across strengths, not just at
        # one point.
        qc = QuantumCircuit(num_qubits=1)
        qc.add_gate(GateInstance("H", [0], [], 0))
        for p in (0.05, 0.15, 0.45):
            nm = NoiseModel()
            nm.add_global_noise(DepolarizingNoise(p))
            est = _mean_gap(qc, nm, n_trials=4000, seed=int(1000 * p))
            assert abs(est - 2.0 * p / 3.0) < 0.025, (p, est)
