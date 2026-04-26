"""Density matrix heatmap visualization panel."""

from __future__ import annotations

import logging

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QLabel, QScrollArea,
    QSpinBox, QPushButton,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from quantum_sim.engine.state_vector import StateVector

logger = logging.getLogger(__name__)


class DensityMatrixPanel(QWidget):
    """Panel displaying the density matrix as a color-coded heatmap.

    Supports views for Real part, Imaginary part, and Magnitude,
    with appropriate colormaps and colorbar.

    Ensemble mode: computes rho = (1/N) sum |psi_i><psi_i| via multi-trial
    stochastic noise simulation, producing a Monte Carlo estimate of the
    mixed-state density matrix (converges to the exact result as N -> inf).
    """

    MAX_DISPLAY_QUBITS = 8

    # Adaptive trial limits: cap trials for large qubit counts to prevent UI freeze
    _MAX_TRIALS_BY_QUBITS = {
        # n_qubits -> max_trials (each trial is O(2^n) work)
        8: 500, 7: 500, 6: 500, 5: 500, 4: 500, 3: 500, 2: 500, 1: 500,
    }
    _DEFAULT_MAX_TRIALS = 100  # for qubits > 8

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)

        self._state_vector: StateVector | None = None
        self._ensemble_rho: np.ndarray | None = None
        self._ensemble_cache_key: tuple | None = None  # (circuit_hash, noise_hash, trials)
        self._dark_theme: bool = False
        self._circuit = None
        self._noise_model = None

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

        # Controls row
        controls_layout = QHBoxLayout()

        title_label = QLabel("Density Matrix")
        title_label.setStyleSheet("font-weight: bold; font-size: 13px;")
        controls_layout.addWidget(title_label)
        controls_layout.addStretch()

        controls_layout.addWidget(QLabel("View:"))
        self._view_combo = QComboBox()
        self._view_combo.addItems(["Real part", "Imaginary part", "Magnitude"])
        self._view_combo.currentIndexChanged.connect(self._redraw)
        controls_layout.addWidget(self._view_combo)

        layout.addLayout(controls_layout)

        # Ensemble controls row
        ensemble_layout = QHBoxLayout()
        ensemble_layout.addWidget(QLabel("Ensemble:"))
        self._ensemble_spin = QSpinBox()
        self._ensemble_spin.setRange(1, 500)
        self._ensemble_spin.setValue(50)
        self._ensemble_spin.setToolTip("Number of stochastic trials for mixed-state estimation")
        self._ensemble_spin.setFixedWidth(70)
        ensemble_layout.addWidget(self._ensemble_spin)
        ensemble_layout.addWidget(QLabel("trials"))

        self._ensemble_btn = QPushButton("Compute Ensemble")
        self._ensemble_btn.setToolTip(
            "Compute mixed-state rho via multi-trial noise simulation"
        )
        self._ensemble_btn.clicked.connect(self._on_compute_ensemble)
        ensemble_layout.addWidget(self._ensemble_btn)

        self._ensemble_status = QLabel("")
        self._ensemble_status.setStyleSheet("color: #888; font-size: 10px;")
        ensemble_layout.addWidget(self._ensemble_status)
        ensemble_layout.addStretch()
        layout.addLayout(ensemble_layout)

        # Warning label for large qubit counts
        self._warning_label = QLabel("")
        self._warning_label.setStyleSheet(
            "color: #FFA726; font-size: 11px; padding: 2px;"
        )
        self._warning_label.setVisible(False)
        layout.addWidget(self._warning_label)

        # Matplotlib figure
        self._figure = Figure(figsize=(5, 4), dpi=100)
        self._canvas = FigureCanvas(self._figure)
        self._canvas.setMinimumSize(250, 250)
        layout.addWidget(self._canvas, stretch=1)

        # Info label
        self._info_label = QLabel("")
        self._info_label.setStyleSheet("color: gray; font-size: 11px;")
        layout.addWidget(self._info_label)

        self._apply_theme_colors()

    # ---- Public API -------------------------------------------------------

    def set_circuit(self, circuit) -> None:
        """Store the circuit reference for ensemble computation."""
        self._circuit = circuit

    def set_noise_model(self, noise_model) -> None:
        """Store the noise model reference for ensemble computation."""
        self._noise_model = noise_model

    def update_state(self, state_vector: StateVector) -> None:
        """Update the density matrix display.

        Args:
            state_vector: The current quantum state.
        """
        self._state_vector = state_vector
        self._ensemble_rho = None  # Clear ensemble on new state
        self._ensemble_status.setText("")
        num_qubits = state_vector.num_qubits

        if num_qubits > self.MAX_DISPLAY_QUBITS:
            self._warning_label.setText(
                f"Display limited to {self.MAX_DISPLAY_QUBITS} qubits for "
                f"performance (current: {num_qubits} qubits, "
                f"matrix size {2**num_qubits}x{2**num_qubits})"
            )
            self._warning_label.setVisible(True)
        else:
            self._warning_label.setVisible(False)

        self._redraw()

    # ---- Ensemble computation ---------------------------------------------

    def _on_compute_ensemble(self) -> None:
        """Compute ensemble density matrix from multi-trial noise simulation.

        Uses a cache key (circuit_hash, noise_config, trials) to skip
        recomputation when nothing has changed. Applies adaptive trial
        limits for large qubit counts to prevent UI freeze.
        """
        if self._circuit is None:
            self._ensemble_status.setText("No circuit set")
            return
        if self._noise_model is None:
            self._ensemble_status.setText("No noise model (pure state = ensemble)")
            return
        if self._circuit.num_qubits > self.MAX_DISPLAY_QUBITS:
            self._ensemble_status.setText("Too many qubits for ensemble")
            return

        n_qubits = self._circuit.num_qubits
        max_trials = self._MAX_TRIALS_BY_QUBITS.get(
            n_qubits, self._DEFAULT_MAX_TRIALS
        )
        n_trials = min(self._ensemble_spin.value(), max_trials)

        # Cache check: skip if same (circuit, noise, trials)
        try:
            circ_hash = self._circuit.circuit_hash()
            noise_hash = hash(str(self._noise_model.to_dict())) if hasattr(
                self._noise_model, "to_dict"
            ) else id(self._noise_model)
        except Exception:
            circ_hash = 0
            noise_hash = 0
        cache_key = (circ_hash, noise_hash, n_trials)

        if self._ensemble_cache_key == cache_key and self._ensemble_rho is not None:
            self._ensemble_status.setText(
                f"Cached: {n_trials} trials | "
                f"Purity={float(np.real(np.trace(self._ensemble_rho @ self._ensemble_rho))):.4f}"
            )
            return

        if n_trials < self._ensemble_spin.value():
            self._ensemble_status.setText(
                f"Computing ({n_trials} trials, capped from "
                f"{self._ensemble_spin.value()} for {n_qubits}q)..."
            )
        else:
            self._ensemble_status.setText(f"Computing ({n_trials} trials)...")
        from PyQt6.QtWidgets import QApplication
        QApplication.processEvents()

        try:
            from quantum_sim.engine.simulator import Simulator
            sim = Simulator(noise_model=self._noise_model)
            rho = sim.ensemble_density_matrix(
                self._circuit, n_trials=n_trials, seed=42
            )
            self._ensemble_rho = rho
            self._ensemble_cache_key = cache_key
            self._ensemble_status.setText(
                f"Ensemble: {n_trials} trials | "
                f"Purity={float(np.real(np.trace(rho @ rho))):.4f}"
            )
            self._redraw()
        except Exception as e:
            logger.warning("Ensemble computation failed: %s", e, exc_info=True)
            self._ensemble_status.setText(f"Error: {e}")

    # ---- Drawing ----------------------------------------------------------

    def _redraw(self) -> None:
        """Redraw the density matrix heatmap."""
        self._figure.clear()

        if self._state_vector is None and self._ensemble_rho is None:
            ax = self._figure.add_subplot(111)
            ax.text(
                0.5, 0.5, "No state data",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=12, color="gray",
            )
            self._apply_axes_theme(ax)
            self._canvas.draw()
            self._info_label.setText("")
            return

        # Use ensemble rho if available, otherwise pure-state rho
        if self._ensemble_rho is not None:
            rho = self._ensemble_rho
            num_qubits = int(np.log2(rho.shape[0]))
            source = "ensemble"
        elif self._state_vector is not None:
            sv = self._state_vector
            num_qubits = sv.num_qubits
            if num_qubits > self.MAX_DISPLAY_QUBITS:
                self._info_label.setText(
                    f"Density matrix too large to display "
                    f"({2**num_qubits}x{2**num_qubits})"
                )
                ax = self._figure.add_subplot(111)
                ax.text(
                    0.5, 0.5,
                    f"Matrix too large ({2**num_qubits}\u00D7{2**num_qubits})\n"
                    f"Reduce to \u2264{self.MAX_DISPLAY_QUBITS} qubits to display",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=11, color="orange",
                )
                self._apply_axes_theme(ax)
                self._canvas.draw()
                return
            rho = sv.get_density_matrix()
            source = "pure"
        else:
            return

        dim = rho.shape[0]

        # Select view
        view_index = self._view_combo.currentIndex()
        if view_index == 0:
            data = np.real(rho)
            title = "Re(\u03C1)"
            cmap = "RdBu"
            vmin, vmax = -1.0, 1.0
        elif view_index == 1:
            data = np.imag(rho)
            title = "Im(\u03C1)"
            cmap = "RdBu"
            vmin, vmax = -1.0, 1.0
        else:
            data = np.abs(rho)
            title = "|\u03C1|"
            cmap = "viridis"
            vmin, vmax = 0.0, 1.0

        if source == "ensemble":
            title += " [ensemble]"

        ax = self._figure.add_subplot(111)

        # Draw heatmap
        im = ax.imshow(
            data, cmap=cmap, vmin=vmin, vmax=vmax,
            aspect="equal", interpolation="nearest",
            origin="upper",
        )

        # Colorbar
        cbar = self._figure.colorbar(im, ax=ax, shrink=0.8)
        if self._dark_theme:
            cbar.ax.yaxis.set_tick_params(color="white", labelcolor="white")

        # Axis labels as basis states
        if dim <= 32:
            labels = [f"|{format(i, f'0{num_qubits}b')}\u27E9"
                      for i in range(dim)]
            fontsize = max(5, min(9, 120 // max(dim, 1)))
            ax.set_xticks(range(dim))
            ax.set_yticks(range(dim))
            ax.set_xticklabels(labels, rotation=90, fontsize=fontsize)
            ax.set_yticklabels(labels, fontsize=fontsize)
        else:
            # Too many labels, show a subset
            step = max(1, dim // 8)
            ticks = list(range(0, dim, step))
            labels = [f"|{format(i, f'0{num_qubits}b')}\u27E9" for i in ticks]
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            ax.set_xticklabels(labels, rotation=90, fontsize=6)
            ax.set_yticklabels(labels, fontsize=6)

        ax.set_title(title, fontsize=11)

        # Add value annotations for small matrices
        if dim <= 8:
            for i in range(dim):
                for j in range(dim):
                    val = data[i, j]
                    if abs(val) > 0.005:
                        text_color = "white" if abs(val) > 0.5 else "black"
                        if cmap == "RdBu":
                            text_color = (
                                "white" if abs(val) > 0.6 else "black"
                            )
                        ax.text(
                            j, i, f"{val:.2f}",
                            ha="center", va="center",
                            fontsize=max(6, 10 - dim),
                            color=text_color,
                        )

        self._apply_axes_theme(ax)
        self._figure.tight_layout()
        self._canvas.draw()

        # Info: Tr, Purity, and (for ensemble) Von Neumann entropy
        trace = np.real(np.trace(rho))
        purity = np.real(np.trace(rho @ rho))
        src_label = "ensemble" if source == "ensemble" else "pure state"
        info_parts = [
            f"{num_qubits}q",
            f"{dim}\u00D7{dim}",
            f"Tr(\u03C1)={trace:.4f}",
            f"Purity={purity:.4f}",
        ]
        if source == "ensemble":
            # Von Neumann entropy: S = -Tr(rho log2 rho) for mixed states
            eigvals = np.linalg.eigvalsh(rho)
            eigvals = eigvals[eigvals > 1e-15]
            entropy = float(-np.sum(eigvals * np.log2(eigvals)))
            info_parts.append(f"S={entropy:.4f} bits")
        info_parts.append(src_label)
        self._info_label.setText(" | ".join(info_parts))

    def set_theme(self, dark: bool) -> None:
        """Switch between dark and light theme."""
        self._dark_theme = dark
        self._apply_theme_colors()
        self._redraw()

    def _apply_theme_colors(self):
        """Apply theme colors to figure background."""
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
        """Clear the display."""
        self._state_vector = None
        self._ensemble_rho = None
        self._ensemble_status.setText("")
        self._warning_label.setVisible(False)
        self._redraw()
