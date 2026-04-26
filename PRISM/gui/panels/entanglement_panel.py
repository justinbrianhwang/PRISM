"""Entanglement graph visualization panel.

Displays qubits as nodes arranged in a circle with edges whose thickness
is proportional to pairwise entanglement (mutual information or concurrence).
Embeds a matplotlib canvas inside a PyQt6 QWidget.
"""

from __future__ import annotations

import math
from itertools import combinations

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QLabel, QSizePolicy,
    QScrollArea,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.colors as mcolors
import matplotlib.cm as mcm

from quantum_sim.engine.state_vector import StateVector
from quantum_sim.engine.analysis import StateAnalysis

# Distinct per-qubit node colors (same palette used elsewhere in the project)
_NODE_COLORS = [
    "#FF4444", "#44AAFF", "#44DD44", "#FFAA22",
    "#DD44DD", "#22DDDD", "#FFDD44", "#FF66AA",
    "#8866FF", "#66FFAA", "#FF8844", "#44FFDD",
    "#AA88FF", "#AAFF44", "#FF4488", "#4488FF",
]

# Edge-thickness limits
_EDGE_WIDTH_MIN = 0.5
_EDGE_WIDTH_MAX = 5.0

# Qubit count beyond which a performance warning is shown
_WARN_QUBIT_THRESHOLD = 10


class EntanglementPanel(QWidget):
    """Panel displaying a circular entanglement graph between qubits.

    Edges are weighted by a selectable metric (Mutual Information or
    Concurrence).  Edge thickness and colour indicate entanglement
    strength.
    """

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)

        self._state: StateVector | None = None
        self._dark_theme: bool = True
        self._edge_data: list[tuple[int, int, float]] = []  # (qa, qb, value)

        self._setup_ui()

    # ---- UI setup -----------------------------------------------------------

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
        layout.setSpacing(4)

        # Controls row
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(6)

        title_label = QLabel("Entanglement Graph")
        title_label.setStyleSheet("font-weight: bold; font-size: 13px;")
        controls_layout.addWidget(title_label)
        controls_layout.addStretch()

        controls_layout.addWidget(QLabel("Metric:"))
        self._metric_combo = QComboBox()
        self._metric_combo.addItems(["Mutual Information", "Concurrence"])
        self._metric_combo.setMinimumWidth(140)
        self._metric_combo.currentIndexChanged.connect(self._on_metric_changed)
        controls_layout.addWidget(self._metric_combo)

        layout.addLayout(controls_layout)

        # Warning label (hidden by default)
        self._warning_label = QLabel("")
        self._warning_label.setStyleSheet(
            "color: #FFA500; font-size: 11px; font-style: italic;"
        )
        self._warning_label.setWordWrap(True)
        self._warning_label.setVisible(False)
        layout.addWidget(self._warning_label)

        # Matplotlib figure
        self._figure = Figure(dpi=100)
        self._canvas = FigureCanvas(self._figure)
        self._canvas.setMinimumSize(250, 200)
        self._canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding,
        )
        layout.addWidget(self._canvas, stretch=1)

        # Info label at bottom
        self._info_label = QLabel("")
        self._info_label.setStyleSheet("color: gray; font-size: 11px;")
        self._info_label.setWordWrap(True)
        layout.addWidget(self._info_label)

        self._apply_theme_colors()
        self._redraw()

    # ---- Public API ---------------------------------------------------------

    def update_state(self, state: StateVector) -> None:
        """Recompute entanglement metrics and redraw the graph.

        Args:
            state: Current quantum state to visualise.
        """
        self._state = state
        self._compute_entanglement()
        self._redraw()

    def set_theme(self, dark: bool) -> None:
        """Switch between dark and light theme."""
        self._dark_theme = dark
        self._apply_theme_colors()
        self._redraw()

    def clear(self) -> None:
        """Reset the panel to its empty state."""
        self._state = None
        self._edge_data = []
        self._warning_label.setVisible(False)
        self._info_label.setText("")
        self._redraw()

    # ---- Metric computation -------------------------------------------------

    def _on_metric_changed(self, _index: int) -> None:
        """Re-compute and redraw when the user changes the metric."""
        if self._state is not None:
            self._compute_entanglement()
        self._redraw()

    def _current_metric_func(self):
        """Return the analysis function matching the selected metric."""
        if self._metric_combo.currentIndex() == 0:
            return StateAnalysis.mutual_information
        return StateAnalysis.concurrence

    def _compute_entanglement(self) -> None:
        """Compute pairwise entanglement for all qubit pairs."""
        if self._state is None:
            self._edge_data = []
            return

        n = self._state.num_qubits

        # Show or hide the performance warning
        if n > _WARN_QUBIT_THRESHOLD:
            pairs = n * (n - 1) // 2
            self._warning_label.setText(
                f"Warning: {n} qubits \u2192 {pairs} partial traces "
                f"(O(n\u00B2) cost). This may be slow."
            )
            self._warning_label.setVisible(True)
        else:
            self._warning_label.setVisible(False)

        metric_func = self._current_metric_func()
        edges: list[tuple[int, int, float]] = []

        for qa, qb in combinations(range(n), 2):
            value = metric_func(self._state, qa, qb)
            edges.append((qa, qb, float(value)))

        self._edge_data = edges

    # ---- Drawing ------------------------------------------------------------

    def _redraw(self) -> None:
        """Redraw the full entanglement graph."""
        self._figure.clear()
        ax = self._figure.add_subplot(111)

        n = self._state.num_qubits if self._state is not None else 0

        if n == 0:
            ax.text(
                0.5, 0.5, "No quantum state loaded",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=12, color="gray",
            )
            self._apply_axes_theme(ax)
            ax.set_aspect("equal")
            ax.axis("off")
            self._canvas.draw()
            self._info_label.setText("")
            return

        # Compute circular node positions
        positions = self._circle_positions(n)

        # Determine value range for scaling
        values = [v for (_, _, v) in self._edge_data]
        max_val = max(values) if values else 0.0

        # Draw edges first (behind nodes)
        cmap = mcm.get_cmap("plasma")
        if max_val > 1e-9:
            norm = mcolors.Normalize(vmin=0.0, vmax=max_val)
        else:
            norm = mcolors.Normalize(vmin=0.0, vmax=1.0)

        for qa, qb, val in self._edge_data:
            if val < 1e-9:
                continue
            xa, ya = positions[qa]
            xb, yb = positions[qb]
            width = _EDGE_WIDTH_MIN + (
                (_EDGE_WIDTH_MAX - _EDGE_WIDTH_MIN) * (val / max_val)
                if max_val > 1e-9 else 0.0
            )
            color = cmap(norm(val))
            ax.plot(
                [xa, xb], [ya, yb],
                color=color, linewidth=width, alpha=0.8,
                solid_capstyle="round", zorder=1,
            )

        # Draw nodes
        node_size = max(200, 600 - 30 * n)
        text_color = "white" if self._dark_theme else "black"

        for i, (x, y) in enumerate(positions):
            color = _NODE_COLORS[i % len(_NODE_COLORS)]
            ax.scatter(
                x, y, s=node_size, c=color, zorder=3,
                edgecolors="white" if self._dark_theme else "#333333",
                linewidths=1.2,
            )
            ax.text(
                x, y, str(i),
                ha="center", va="center", fontsize=9,
                fontweight="bold", color=text_color, zorder=4,
            )

        # Add a colour bar when there are edges to show
        if max_val > 1e-9:
            sm = mcm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = self._figure.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
            metric_name = self._metric_combo.currentText()
            cbar.set_label(metric_name, color=text_color, fontsize=9)
            cbar.ax.tick_params(colors=text_color, labelsize=8)
            cbar.outline.set_edgecolor(
                "#666666" if self._dark_theme else "#cccccc"
            )

        # Axes housekeeping
        margin = 0.35
        ax.set_xlim(-1.0 - margin, 1.0 + margin)
        ax.set_ylim(-1.0 - margin, 1.0 + margin)
        ax.set_aspect("equal")
        ax.axis("off")
        self._apply_axes_theme(ax)

        self._figure.tight_layout()
        self._canvas.draw()

        # Update info label
        self._update_info_label(n)

    def _update_info_label(self, n: int) -> None:
        """Set the bottom info label to show the most-entangled pair."""
        if n < 2 or not self._edge_data:
            self._info_label.setText(f"{n} qubit(s) | no entanglement pairs")
            return

        best_qa, best_qb, best_val = max(
            self._edge_data, key=lambda t: t[2],
        )
        metric_name = self._metric_combo.currentText()
        self._info_label.setText(
            f"{n} qubits | max {metric_name}: "
            f"q{best_qa}\u2013q{best_qb} = {best_val:.4f}"
        )

    # ---- Geometry helpers ---------------------------------------------------

    @staticmethod
    def _circle_positions(n: int) -> list[tuple[float, float]]:
        """Return *n* evenly-spaced points on the unit circle.

        The first node is placed at the top (12 o'clock).
        """
        positions: list[tuple[float, float]] = []
        for i in range(n):
            angle = math.pi / 2 - 2 * math.pi * i / n
            positions.append((math.cos(angle), math.sin(angle)))
        return positions

    # ---- Theme helpers ------------------------------------------------------

    def _apply_theme_colors(self) -> None:
        """Apply theme-specific colours to the figure background."""
        if self._dark_theme:
            self._figure.set_facecolor("#2B2B2B")
        else:
            self._figure.set_facecolor("white")

    def _apply_axes_theme(self, ax) -> None:
        """Apply theme colours to an axes object."""
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
