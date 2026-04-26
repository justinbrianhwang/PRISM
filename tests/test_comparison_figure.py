"""Smoke tests for :func:`PRISM.figures.attribution_comparison_figure`.

The comparison figure stacks two attribution percentage panels on top
of each other, one for each method (typically untwirled vs twirled).
Tests cover:

* Two axes are produced and shaped consistently with the input data.
* Mismatched column counts are rejected with a clear ValueError.
* Saving to PDF / PNG yields a non-empty file.
* Y-axis range is shared so the visual comparison is honest.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pytest

from PRISM.engine.circuit import GateInstance, QuantumCircuit
from PRISM.engine.debugger import CircuitDebugger
from PRISM.engine.noise import (
    CoherentOverRotationNoise,
    DepolarizingNoise,
    NoiseModel,
)
from PRISM.figures import attribution_comparison_figure, save_figure


def _bell_circuit() -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits=2)
    qc.add_gate(GateInstance("H", [0], [], 0))
    qc.add_gate(GateInstance("CNOT", [0, 1], [], 1))
    return qc


@pytest.fixture
def untwirled_attr():
    nm = NoiseModel()
    nm.add_global_noise(CoherentOverRotationNoise(0.30, axis="Z"))
    return CircuitDebugger().compute_noise_attribution_with_statistics(
        _bell_circuit(), nm,
        n_trials=40, n_bootstrap=200,
        seed=11, twirl=False,
    )


@pytest.fixture
def twirled_attr():
    nm = NoiseModel()
    nm.add_global_noise(CoherentOverRotationNoise(0.30, axis="Z"))
    return CircuitDebugger().compute_noise_attribution_with_statistics(
        _bell_circuit(), nm,
        n_trials=40, n_bootstrap=200,
        seed=11, twirl=True,
    )


class TestAttributionComparisonFigure:

    def test_returns_two_panel_figure(self, untwirled_attr, twirled_attr):
        fig = attribution_comparison_figure(
            untwirled_attr, twirled_attr,
            title_top="Untwirled", title_bottom="Twirled",
        )
        assert len(fig.axes) == 2
        plt.close(fig)

    def test_y_axes_share_range(self, untwirled_attr, twirled_attr):
        fig = attribution_comparison_figure(untwirled_attr, twirled_attr)
        ax_top, ax_bot = fig.axes
        assert ax_top.get_ylim() == ax_bot.get_ylim()
        plt.close(fig)

    def test_mismatched_column_counts_rejected(self, untwirled_attr):
        # Build a different-sized attribution by using a 3-qubit circuit.
        qc3 = QuantumCircuit(num_qubits=3)
        qc3.add_gate(GateInstance("H", [0], [], 0))
        qc3.add_gate(GateInstance("CNOT", [0, 1], [], 1))
        qc3.add_gate(GateInstance("CNOT", [1, 2], [], 2))
        nm = NoiseModel()
        nm.add_global_noise(DepolarizingNoise(0.05))
        other = CircuitDebugger().compute_noise_attribution_with_statistics(
            qc3, nm, n_trials=20, n_bootstrap=100, seed=0,
        )

        with pytest.raises(ValueError, match="same number of columns"):
            attribution_comparison_figure(untwirled_attr, other)

    def test_save_pdf(self, tmp_path, untwirled_attr, twirled_attr):
        fig = attribution_comparison_figure(untwirled_attr, twirled_attr)
        out = tmp_path / "cmp.pdf"
        save_figure(fig, str(out))
        assert out.exists() and out.stat().st_size > 1000
        plt.close(fig)

    def test_titles_render_when_supplied(
        self, untwirled_attr, twirled_attr,
    ):
        fig = attribution_comparison_figure(
            untwirled_attr, twirled_attr,
            title_top="Top label", title_bottom="Bottom label",
        )
        ax_top, ax_bot = fig.axes
        # The helper uses ``set_title(loc="left")``, so query the same
        # location -- the centred title slot stays empty.
        assert ax_top.get_title(loc="left") == "Top label"
        assert ax_bot.get_title(loc="left") == "Bottom label"
        plt.close(fig)
