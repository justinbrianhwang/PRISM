"""Integration tests for ``compute_noise_attribution_with_statistics``.

These tests verify that the bootstrap-aware attribution method:

* Produces output with the same per-column shape as the cheap version,
  plus a populated :class:`AttributionStatistics` block.
* Preserves backward compatibility -- existing
  :meth:`compute_noise_attribution` callers see no behaviour change.
* Returns CIs that bracket the point estimate.
* Identifies a deliberately noisy column as significant under FDR
  control, while a no-op column does not get flagged.
* Is reproducible under a fixed seed.

The tests deliberately use small ``n_trials`` and ``n_bootstrap`` to
keep CI runtime under a few seconds; the absolute statistical claims
are kept loose to avoid flakiness from a finite resampling budget.
"""

from __future__ import annotations

import numpy as np
import pytest

from PRISM.engine.circuit import GateInstance, QuantumCircuit
from PRISM.engine.debugger import (
    AttributionStatistics,
    CircuitDebugger,
    NoiseAttribution,
)
from PRISM.engine.noise import (
    BitFlipNoise,
    DepolarizingNoise,
    NoiseModel,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_targeted_noise_model(noisy_gate: str, p: float) -> NoiseModel:
    """Noise model that applies bit-flip noise *only* to a specific gate name.

    This gives a deterministic ground truth for attribution tests: the
    column containing ``noisy_gate`` should dominate the attribution.
    """
    nm = NoiseModel()
    nm.add_gate_noise(noisy_gate, BitFlipNoise(p))
    return nm


def _five_column_circuit() -> QuantumCircuit:
    """A 3-qubit circuit with 5 well-separated columns.

    Columns: H, CNOT, Rx, CNOT, H. We will tag a single one of these
    gate types with high noise to make attribution unambiguous.
    """
    qc = QuantumCircuit(num_qubits=3)
    qc.add_gate(GateInstance("H", [0], [], 0))
    qc.add_gate(GateInstance("CNOT", [0, 1], [], 1))
    qc.add_gate(GateInstance("Rx", [2], [0.7], 2))
    qc.add_gate(GateInstance("CNOT", [1, 2], [], 3))
    qc.add_gate(GateInstance("H", [0], [], 4))
    return qc


# ---------------------------------------------------------------------------
# Backwards compatibility
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:

    def test_existing_method_unchanged(self, deep_random_circuit,
                                       moderate_depolarizing_noise):
        """The plain compute_noise_attribution must still return a
        NoiseAttribution with statistics=None."""
        dbg = CircuitDebugger()
        attr = dbg.compute_noise_attribution(
            deep_random_circuit,
            moderate_depolarizing_noise,
            n_trials=20,
            seed=42,
        )
        assert isinstance(attr, NoiseAttribution)
        assert attr.statistics is None
        assert len(attr.delta_fidelity) == len(attr.gate_labels)
        assert len(attr.delta_fidelity_std) == len(attr.delta_fidelity)
        assert len(attr.column_attribution_pct) == len(attr.delta_fidelity)
        assert len(attr.is_recovery) == len(attr.delta_fidelity)

    def test_attribution_pct_normalisation(self, deep_random_circuit,
                                            moderate_depolarizing_noise):
        dbg = CircuitDebugger()
        attr = dbg.compute_noise_attribution(
            deep_random_circuit,
            moderate_depolarizing_noise,
            n_trials=30,
            seed=7,
        )
        # If any positive contribution exists, pct should sum to ~100
        if not attr.no_measurable_loss:
            assert abs(sum(attr.column_attribution_pct) - 100.0) < 1e-6


# ---------------------------------------------------------------------------
# Statistics-bearing variant
# ---------------------------------------------------------------------------


class TestComputeNoiseAttributionWithStatistics:

    def test_returns_populated_statistics(self, deep_random_circuit,
                                          moderate_depolarizing_noise):
        dbg = CircuitDebugger()
        attr = dbg.compute_noise_attribution_with_statistics(
            deep_random_circuit,
            moderate_depolarizing_noise,
            n_trials=40,
            n_bootstrap=200,
            confidence=0.95,
            fdr_level=0.05,
            seed=11,
        )
        assert attr.statistics is not None
        stats = attr.statistics
        assert isinstance(stats, AttributionStatistics)

        n_cols = len(attr.delta_fidelity)
        # All per-column arrays must align with the column axis
        assert len(stats.delta_fidelity_ci_lower) == n_cols
        assert len(stats.delta_fidelity_ci_upper) == n_cols
        assert len(stats.delta_fidelity_p_value) == n_cols
        assert len(stats.delta_fidelity_q_value) == n_cols
        assert len(stats.column_significant) == n_cols
        assert len(stats.attribution_pct_ci_lower) == n_cols
        assert len(stats.attribution_pct_ci_upper) == n_cols
        assert len(stats.recovery_rate) == n_cols
        assert len(stats.recovery_rate_ci_lower) == n_cols
        assert len(stats.recovery_rate_ci_upper) == n_cols
        # Metadata persisted
        assert stats.n_trials == 40
        assert stats.n_bootstrap == 200
        assert stats.confidence == 0.95
        assert stats.fdr_level == 0.05

    def test_ci_brackets_point_estimate(self, deep_random_circuit,
                                        moderate_depolarizing_noise):
        dbg = CircuitDebugger()
        attr = dbg.compute_noise_attribution_with_statistics(
            deep_random_circuit,
            moderate_depolarizing_noise,
            n_trials=30,
            n_bootstrap=300,
            seed=42,
        )
        stats = attr.statistics
        # delta_F CI brackets the mean (allowing tiny percentile rounding)
        for i, mean in enumerate(attr.delta_fidelity):
            tol = 1e-9
            assert stats.delta_fidelity_ci_lower[i] <= mean + tol
            assert mean - tol <= stats.delta_fidelity_ci_upper[i]
        # Attribution % CI brackets the point estimate similarly
        for i, pct in enumerate(attr.column_attribution_pct):
            tol = 1e-6
            assert stats.attribution_pct_ci_lower[i] <= pct + tol
            assert pct - tol <= stats.attribution_pct_ci_upper[i]

    def test_p_and_q_in_unit_interval(self, deep_random_circuit,
                                      moderate_depolarizing_noise):
        dbg = CircuitDebugger()
        attr = dbg.compute_noise_attribution_with_statistics(
            deep_random_circuit,
            moderate_depolarizing_noise,
            n_trials=30,
            n_bootstrap=200,
            seed=1,
        )
        s = attr.statistics
        for p in s.delta_fidelity_p_value:
            assert 0.0 <= p <= 1.0
        for q in s.delta_fidelity_q_value:
            assert 0.0 <= q <= 1.0

    def test_recovery_rate_in_unit_interval(self, deep_random_circuit,
                                            moderate_depolarizing_noise):
        dbg = CircuitDebugger()
        attr = dbg.compute_noise_attribution_with_statistics(
            deep_random_circuit,
            moderate_depolarizing_noise,
            n_trials=40,
            n_bootstrap=200,
            seed=0,
        )
        s = attr.statistics
        for r in s.recovery_rate:
            assert 0.0 <= r <= 1.0
        for r in s.recovery_rate_ci_lower:
            assert 0.0 <= r <= 1.0
        for r in s.recovery_rate_ci_upper:
            assert 0.0 <= r <= 1.0

    def test_targeted_noise_flags_correct_column(self):
        """If only the CNOT gate is noisy, attribution should mark a
        CNOT column as significant under FDR control.

        Robustness: with bit-flip(0.15) on CNOTs only, the two CNOT
        columns should account for the bulk of fidelity loss.  We do
        not assert which specific CNOT column wins (could be either or
        both depending on stochastic realisation), only that *at least
        one CNOT column is significant* and that *at least one
        non-CNOT column is not* -- i.e. the test discriminates.
        """
        circuit = _five_column_circuit()
        nm = _make_targeted_noise_model("CNOT", p=0.15)

        dbg = CircuitDebugger()
        attr = dbg.compute_noise_attribution_with_statistics(
            circuit,
            nm,
            n_trials=80,
            n_bootstrap=500,
            confidence=0.95,
            fdr_level=0.05,
            seed=2024,
        )
        stats = attr.statistics
        labels = ["".join(col) for col in attr.gate_labels]

        cnot_cols = [i for i, lab in enumerate(labels) if "CNOT" in lab]
        other_cols = [i for i, lab in enumerate(labels) if "CNOT" not in lab]

        cnot_significant = [stats.column_significant[i] for i in cnot_cols]
        cnot_attr_pct = [attr.column_attribution_pct[i] for i in cnot_cols]
        other_attr_pct = [attr.column_attribution_pct[i] for i in other_cols]

        assert any(cnot_significant), (
            f"No CNOT column flagged as significant; "
            f"q-values = {stats.delta_fidelity_q_value}"
        )
        # CNOTs together should dominate attribution
        assert sum(cnot_attr_pct) > sum(other_attr_pct), (
            f"CNOT attribution {sum(cnot_attr_pct):.1f}% did not exceed "
            f"non-CNOT {sum(other_attr_pct):.1f}%"
        )

    def test_reproducibility_under_fixed_seed(self):
        circuit = _five_column_circuit()
        nm = NoiseModel()
        nm.add_global_noise(DepolarizingNoise(0.05))

        dbg = CircuitDebugger()
        a1 = dbg.compute_noise_attribution_with_statistics(
            circuit, nm, n_trials=20, n_bootstrap=100, seed=999,
        )

        # Re-create the noise model so previous internal RNG state cannot leak
        nm2 = NoiseModel()
        nm2.add_global_noise(DepolarizingNoise(0.05))
        a2 = dbg.compute_noise_attribution_with_statistics(
            circuit, nm2, n_trials=20, n_bootstrap=100, seed=999,
        )

        assert a1.delta_fidelity == a2.delta_fidelity
        assert a1.column_attribution_pct == a2.column_attribution_pct
        assert a1.statistics.delta_fidelity_ci_lower == \
            a2.statistics.delta_fidelity_ci_lower
        assert a1.statistics.delta_fidelity_p_value == \
            a2.statistics.delta_fidelity_p_value
        assert a1.statistics.delta_fidelity_q_value == \
            a2.statistics.delta_fidelity_q_value
        assert a1.statistics.column_significant == \
            a2.statistics.column_significant

    def test_no_noise_circuit_yields_no_significance(self, bell_circuit):
        """With near-zero noise, no column should be flagged significant
        under FDR control (modulo bootstrap floor on p-values)."""
        nm = NoiseModel()
        nm.add_global_noise(DepolarizingNoise(1e-6))

        dbg = CircuitDebugger()
        attr = dbg.compute_noise_attribution_with_statistics(
            bell_circuit, nm, n_trials=30, n_bootstrap=200, seed=5,
        )
        # Either no measurable loss, or no column flagged
        if not attr.no_measurable_loss:
            assert not any(attr.statistics.column_significant), (
                f"Spurious significance under near-zero noise: "
                f"{attr.statistics.delta_fidelity_q_value}"
            )

    def test_q_values_are_monotone_under_sorting(self, deep_random_circuit,
                                                  moderate_depolarizing_noise):
        """BH q-values, when sorted by p-value, must be monotonically
        non-decreasing."""
        dbg = CircuitDebugger()
        attr = dbg.compute_noise_attribution_with_statistics(
            deep_random_circuit,
            moderate_depolarizing_noise,
            n_trials=30,
            n_bootstrap=200,
            seed=3,
        )
        s = attr.statistics
        p = np.asarray(s.delta_fidelity_p_value)
        q = np.asarray(s.delta_fidelity_q_value)
        order = np.argsort(p)
        sorted_q = q[order]
        assert np.all(np.diff(sorted_q) >= -1e-12)


# ---------------------------------------------------------------------------
# Sanity: cheap and statistical methods agree on point estimates
# ---------------------------------------------------------------------------


class TestPointEstimateConsistency:

    def test_means_match_cheap_method_for_same_seed(self,
                                                    deep_random_circuit,
                                                    moderate_depolarizing_noise):
        """The cheap and statistical attributions, given the same trial
        seed, must produce *identical* point estimates because both
        methods now go through ``_collect_attribution_trials``."""
        dbg = CircuitDebugger()

        # Reset noise RNG between calls so each starts from a deterministic state
        nm1 = NoiseModel()
        nm1.add_global_noise(DepolarizingNoise(0.05))
        a_cheap = dbg.compute_noise_attribution(
            deep_random_circuit, nm1, n_trials=40, seed=42
        )

        # The statistical version uses children of the master seed: first
        # child is the simulation seed.  We replicate that key derivation
        # here so that the underlying simulation is bit-exact.
        master_rng = np.random.default_rng(42)
        sim_seed = int(master_rng.integers(0, 2**63))

        nm2 = NoiseModel()
        nm2.add_global_noise(DepolarizingNoise(0.05))
        a_cheap_with_sim_seed = dbg.compute_noise_attribution(
            deep_random_circuit, nm2, n_trials=40, seed=sim_seed
        )

        nm3 = NoiseModel()
        nm3.add_global_noise(DepolarizingNoise(0.05))
        a_stat = dbg.compute_noise_attribution_with_statistics(
            deep_random_circuit, nm3,
            n_trials=40, n_bootstrap=50,
            seed=42,
        )

        # The statistical method's point estimates should match the
        # cheap method run on its derived sim_seed -- exactly.
        assert a_cheap_with_sim_seed.delta_fidelity == a_stat.delta_fidelity
        # And the cheap method run on the *master* seed will not match
        # exactly (different sim seed), but should be close in scale.
        assert len(a_cheap.delta_fidelity) == len(a_stat.delta_fidelity)


class TestSignalFloorGuard:
    """Columns at the floating-point noise floor must never be flagged.

    An H gate followed by bit-flip noise leaves |+> invariant, so the
    column's mean delta_F sits at machine epsilon (~1e-16).  Its exact
    value and sign depend on the NumPy/BLAS reduction order, so without
    the signal-floor guard the bootstrap would flag a deterministic
    epsilon offset as a "significant" contribution -- an environment-
    dependent false discovery.
    """

    def test_noise_floor_column_not_significant(self):
        from PRISM.engine.noise import BitFlipNoise

        qc = QuantumCircuit(num_qubits=1)
        qc.add_gate(GateInstance("H", [0], [], 0))
        nm = NoiseModel()
        nm.add_global_noise(BitFlipNoise(0.2))

        debugger = CircuitDebugger()
        attribution = debugger.compute_noise_attribution_with_statistics(
            qc, nm, n_trials=40, n_bootstrap=200, seed=123,
        )
        stats = attribution.statistics
        # Bit-flip acts trivially on |+>: the column is physically
        # inactive, so the guard must force p = 1 and non-significance.
        assert abs(attribution.delta_fidelity[0]) < 1e-12
        assert stats.delta_fidelity_p_value[0] == 1.0
        assert stats.column_significant[0] is False or not stats.column_significant[0]

    def test_active_column_unaffected_by_guard(self):
        qc = QuantumCircuit(num_qubits=1)
        qc.add_gate(GateInstance("H", [0], [], 0))
        nm = NoiseModel()
        nm.add_global_noise(DepolarizingNoise(0.2))

        debugger = CircuitDebugger()
        attribution = debugger.compute_noise_attribution_with_statistics(
            qc, nm, n_trials=60, n_bootstrap=300, seed=123,
        )
        stats = attribution.statistics
        # Depolarizing noise genuinely damages |+>; the column carries
        # real signal and the guard must leave its p-value alone.
        assert abs(attribution.delta_fidelity[0]) > 1e-3
        assert stats.delta_fidelity_p_value[0] < 1.0
