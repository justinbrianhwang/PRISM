"""Smoke tests for the debugger panel's statistics-aware attribution path.

The panel is the user-facing surface for the bootstrap-aware
attribution introduced in PR #1; PR #7 adds a "Statistics" toggle
that switches the panel from
:pymeth:`CircuitDebugger.compute_noise_attribution` to
:pymeth:`compute_noise_attribution_with_statistics`.  These tests
verify that toggling that checkbox actually populates the
:class:`AttributionStatistics` block on the panel's stored
attribution, and that the heatmap rendering does not crash for
either mode.

Runs against ``QT_QPA_PLATFORM=offscreen`` so no display is needed.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

# Skip the whole module if PyQt6 is not installed (CI minimal env).
pytest.importorskip("PyQt6.QtWidgets")

from PyQt6.QtWidgets import QApplication  # noqa: E402

from PRISM.engine.circuit import GateInstance, QuantumCircuit  # noqa: E402
from PRISM.engine.noise import DepolarizingNoise, NoiseModel  # noqa: E402
from PRISM.gui.panels.debugger_panel import DebuggerPanel  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def bell_circuit() -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits=2)
    qc.add_gate(GateInstance("H", [0], [], 0))
    qc.add_gate(GateInstance("CNOT", [0, 1], [], 1))
    return qc


@pytest.fixture
def depolarizing_noise() -> NoiseModel:
    nm = NoiseModel()
    nm.add_global_noise(DepolarizingNoise(0.10))
    return nm


# ---------------------------------------------------------------------------
# Cheap (default) path
# ---------------------------------------------------------------------------


class TestCheapMode:

    def test_run_without_stats(self, qapp, bell_circuit, depolarizing_noise):
        panel = DebuggerPanel()
        panel.set_circuit(bell_circuit)
        panel.set_noise_model(depolarizing_noise)
        panel._trials_spin.setValue(20)
        panel._chk_show_stats.setChecked(False)

        panel._on_run_debug()
        assert panel._attribution is not None
        assert panel._attribution.statistics is None

    def test_bootstrap_spinbox_hidden_in_cheap_mode(self, qapp):
        panel = DebuggerPanel()
        # Default: unchecked; bootstrap controls hidden.
        assert not panel._lbl_bootstrap.isVisible()
        assert not panel._bootstrap_spin.isVisible()


# ---------------------------------------------------------------------------
# Statistics-aware path
# ---------------------------------------------------------------------------


class TestStatisticsMode:

    def test_toggle_reveals_bootstrap_spinbox(self, qapp):
        panel = DebuggerPanel()
        panel.show()  # widgets need a parent paint event for isVisible()
        qapp.processEvents()

        panel._chk_show_stats.setChecked(True)
        qapp.processEvents()
        assert panel._lbl_bootstrap.isVisible()
        assert panel._bootstrap_spin.isVisible()

        panel._chk_show_stats.setChecked(False)
        qapp.processEvents()
        assert not panel._lbl_bootstrap.isVisible()
        assert not panel._bootstrap_spin.isVisible()

    def test_run_with_stats_populates_statistics_block(
        self, qapp, bell_circuit, depolarizing_noise,
    ):
        panel = DebuggerPanel()
        panel.set_circuit(bell_circuit)
        panel.set_noise_model(depolarizing_noise)
        panel._trials_spin.setValue(20)
        panel._chk_show_stats.setChecked(True)
        panel._bootstrap_spin.setValue(150)

        panel._on_run_debug()
        attr = panel._attribution
        assert attr is not None

        stats = attr.statistics
        assert stats is not None
        assert stats.n_trials == 20
        assert stats.n_bootstrap == 150
        assert len(stats.delta_fidelity_ci_lower) == len(attr.delta_fidelity)
        assert len(stats.delta_fidelity_q_value) == len(attr.delta_fidelity)
        # All p-values and q-values must lie in [0, 1]
        for p in stats.delta_fidelity_p_value:
            assert 0.0 <= p <= 1.0
        for q in stats.delta_fidelity_q_value:
            assert 0.0 <= q <= 1.0

    def test_summary_text_mentions_significance_in_stats_mode(
        self, qapp, bell_circuit, depolarizing_noise,
    ):
        panel = DebuggerPanel()
        panel.set_circuit(bell_circuit)
        panel.set_noise_model(depolarizing_noise)
        panel._trials_spin.setValue(40)
        panel._chk_show_stats.setChecked(True)
        panel._bootstrap_spin.setValue(200)

        panel._on_run_debug()
        text = panel._attr_label.text()
        # Either the bootstrap line or the per-column q-value line
        # should mention significance / FDR in stats mode.
        assert "Bootstrap" in text or "FDR" in text or "q=" in text


# ---------------------------------------------------------------------------
# No-noise sanity
# ---------------------------------------------------------------------------


class TestNoNoise:

    def test_no_noise_clears_attribution(self, qapp, bell_circuit):
        panel = DebuggerPanel()
        panel.set_circuit(bell_circuit)
        panel.set_noise_model(None)
        panel._on_run_debug()
        assert panel._attribution is None
        assert panel._noise_results == []
