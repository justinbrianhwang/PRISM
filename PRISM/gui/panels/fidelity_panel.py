"""Fidelity decay visualization panel for noise probability sweeps."""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QDoubleSpinBox,
    QSpinBox, QPushButton, QLabel, QScrollArea,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from quantum_sim.engine.state_vector import StateVector
from quantum_sim.engine.circuit import QuantumCircuit
from quantum_sim.engine.simulator import Simulator
from quantum_sim.engine.noise import (
    NoiseModel, BitFlipNoise, PhaseFlipNoise,
    DepolarizingNoise, AmplitudeDampingNoise,
)
from quantum_sim.engine.analysis import StateAnalysis


_NOISE_TYPES = {
    "Bit Flip": BitFlipNoise,
    "Phase Flip": PhaseFlipNoise,
    "Depolarizing": DepolarizingNoise,
    "Amplitude Damping": AmplitudeDampingNoise,
}


class FidelityPanel(QWidget):
    """Panel displaying fidelity decay curves when sweeping noise probability.

    Shows fidelity and purity as functions of noise probability for the
    currently loaded circuit, compared against the ideal (noiseless) state.
    """

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)

        self._circuit: QuantumCircuit | None = None
        self._ideal_state: StateVector | None = None
        self._dark_theme: bool = True

        self._fidelities: list[float] = []
        self._purities: list[float] = []
        self._probabilities: list[float] = []

        self._setup_ui()

    def _setup_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        outer.addWidget(scroll)
        container = QWidget()
        scroll.setWidget(container)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(4, 4, 4, 4)

        # Title
        title_label = QLabel("Fidelity Decay Analysis")
        title_label.setStyleSheet("font-weight: bold; font-size: 13px;")
        layout.addWidget(title_label)

        # Controls row
        controls_layout = QHBoxLayout()

        controls_layout.addWidget(QLabel("Noise:"))
        self._noise_combo = QComboBox()
        self._noise_combo.addItems(list(_NOISE_TYPES.keys()))
        controls_layout.addWidget(self._noise_combo)

        controls_layout.addWidget(QLabel("Max p:"))
        self._max_p_spin = QDoubleSpinBox()
        self._max_p_spin.setRange(0.01, 1.0)
        self._max_p_spin.setValue(0.3)
        self._max_p_spin.setSingleStep(0.05)
        self._max_p_spin.setDecimals(2)
        self._max_p_spin.setMinimumWidth(70)
        controls_layout.addWidget(self._max_p_spin)

        controls_layout.addWidget(QLabel("Points:"))
        self._num_points_spin = QSpinBox()
        self._num_points_spin.setRange(5, 50)
        self._num_points_spin.setValue(15)
        self._num_points_spin.setMinimumWidth(60)
        controls_layout.addWidget(self._num_points_spin)

        controls_layout.addWidget(QLabel("Trials:"))
        self._trials_spin = QSpinBox()
        self._trials_spin.setRange(1, 500)
        self._trials_spin.setValue(50)
        self._trials_spin.setMinimumWidth(70)
        controls_layout.addWidget(self._trials_spin)

        controls_layout.addStretch()

        self._run_btn = QPushButton("Run Sweep")
        self._run_btn.setMinimumWidth(90)
        self._run_btn.clicked.connect(self._on_run_sweep)
        controls_layout.addWidget(self._run_btn)

        layout.addLayout(controls_layout)

        # Matplotlib figure
        self._figure = Figure(figsize=(6, 3), dpi=100)
        self._canvas = FigureCanvas(self._figure)
        self._canvas.setMinimumHeight(200)
        layout.addWidget(self._canvas, stretch=1)

        # Info label
        self._info_label = QLabel("")
        self._info_label.setStyleSheet("color: gray; font-size: 11px;")
        layout.addWidget(self._info_label)

        self._apply_theme_colors()
        self._redraw()

    def set_circuit(self, circuit: QuantumCircuit) -> None:
        """Store a reference to the current circuit for re-simulation.

        Args:
            circuit: The quantum circuit to simulate with noise.
        """
        self._circuit = circuit

    def update_state(self, state: StateVector) -> None:
        """Store the ideal (noiseless) state for fidelity comparison.

        Args:
            state: The ideal state vector from noiseless simulation.
        """
        self._ideal_state = state

    def _on_run_sweep(self) -> None:
        """Execute the noise probability sweep and update the plot."""
        if self._circuit is None or self._ideal_state is None:
            self._info_label.setText(
                "No circuit or ideal state available. Run a simulation first."
            )
            return

        noise_name = self._noise_combo.currentText()
        noise_cls = _NOISE_TYPES[noise_name]
        max_p = self._max_p_spin.value()
        num_points = self._num_points_spin.value()

        probabilities = np.linspace(0, max_p, num_points).tolist()
        fidelities: list[float] = []
        purities: list[float] = []

        # Average over multiple trials for stochastic noise
        n_trials = self._trials_spin.value()
        base_rng = np.random.default_rng()

        for p in probabilities:
            if np.isclose(p, 0.0):
                # No noise: exact result
                fidelities.append(1.0)
                purities.append(1.0)
                continue

            fid_sum = 0.0
            pur_sum = 0.0
            for _ in range(n_trials):
                model = NoiseModel()
                model.add_global_noise(noise_cls(p))
                model.set_seed(int(base_rng.integers(0, 2**63)))
                sim = Simulator(noise_model=model)
                result = sim.run(self._circuit, shots=0)
                fid_sum += StateAnalysis.state_fidelity(
                    self._ideal_state.data, result.final_state.data
                )
                pur_sum += StateAnalysis.purity(result.final_state)
            fidelities.append(fid_sum / n_trials)
            purities.append(pur_sum / n_trials)

        self._probabilities = probabilities
        self._fidelities = fidelities
        self._purities = purities
        self._redraw()

    def _redraw(self) -> None:
        """Redraw the fidelity decay plot."""
        self._figure.clear()
        ax = self._figure.add_subplot(111)

        if not self._probabilities:
            ax.text(
                0.5, 0.5, "No sweep data\nClick 'Run Sweep' to begin",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=12, color="gray",
            )
            self._apply_axes_theme(ax)
            self._canvas.draw()
            self._info_label.setText("")
            return

        # Plot fidelity and purity curves
        ax.plot(
            self._probabilities, self._fidelities,
            color="#4A90D9", linewidth=2, marker="o",
            markersize=4, label="Fidelity",
        )
        ax.plot(
            self._probabilities, self._purities,
            color="#2ECC71", linewidth=2, marker="s",
            markersize=4, label="Purity",
        )

        # Reference lines
        ax.axhline(
            y=0.99, color="#F39C12", linestyle="--",
            linewidth=1, alpha=0.7, label="High fidelity (0.99)",
        )
        ax.axhline(
            y=2.0 / 3.0, color="#E74C3C", linestyle="--",
            linewidth=1, alpha=0.7, label="Classical limit (2/3)",
        )

        ax.set_xlim(0, self._probabilities[-1])
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Noise Probability")
        ax.set_ylabel("Fidelity / Purity")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)

        self._apply_axes_theme(ax)
        self._figure.tight_layout()
        self._canvas.draw()

        # Info label
        final_fidelity = self._fidelities[-1]
        self._info_label.setText(
            f"Fidelity at max noise: {final_fidelity:.3f} (Trials: {n_trials})"
        )

    def set_theme(self, dark: bool) -> None:
        """Switch between dark and light theme."""
        self._dark_theme = dark
        self._apply_theme_colors()
        self._redraw()

    def _apply_theme_colors(self):
        """Apply theme-specific colors to the figure."""
        if self._dark_theme:
            self._figure.set_facecolor("#2B2B2B")
        else:
            self._figure.set_facecolor("white")

    def _apply_axes_theme(self, ax):
        """Apply theme colors to axes."""
        if self._dark_theme:
            ax.set_facecolor("#363636")
            ax.tick_params(colors="white")
            ax.xaxis.label.set_color("white")
            ax.yaxis.label.set_color("white")
            ax.title.set_color("white")
            for spine in ax.spines.values():
                spine.set_color("#666666")
        else:
            ax.set_facecolor("white")
            ax.tick_params(colors="black")
            ax.xaxis.label.set_color("black")
            ax.yaxis.label.set_color("black")
            for spine in ax.spines.values():
                spine.set_color("#cccccc")

    def clear(self) -> None:
        """Clear all sweep data and reset the display."""
        self._fidelities = []
        self._purities = []
        self._probabilities = []
        self._ideal_state = None
        self._redraw()
