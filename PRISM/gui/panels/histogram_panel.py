"""Histogram panel for displaying measurement result distributions."""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QSpinBox,
    QPushButton, QLabel, QScrollArea,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class HistogramPanel(QWidget):
    """Panel displaying measurement results as a bar chart.

    Supports toggling between probability and raw count views.
    Emits run_requested when the user clicks the Run button.
    """

    run_requested = pyqtSignal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)

        self._counts: dict[str, int] = {}
        self._shots: int = 1024
        self._dark_theme: bool = False

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

        title_label = QLabel("Measurement Histogram")
        title_label.setStyleSheet("font-weight: bold; font-size: 13px;")
        controls_layout.addWidget(title_label)
        controls_layout.addStretch()

        # Measurement basis combo
        controls_layout.addWidget(QLabel("Basis:"))
        self._basis_combo = QComboBox()
        self._basis_combo.addItems(["Z", "X", "Y"])
        self._basis_combo.setToolTip(
            "Z: computational basis\nX: Hadamard basis\nY: S-dag + H basis"
        )
        controls_layout.addWidget(self._basis_combo)

        # View mode combo
        controls_layout.addWidget(QLabel("View:"))
        self._view_combo = QComboBox()
        self._view_combo.addItems(["Probability", "Counts"])
        self._view_combo.currentIndexChanged.connect(self._redraw)
        controls_layout.addWidget(self._view_combo)

        # Shots spin box
        controls_layout.addWidget(QLabel("Shots:"))
        self._shots_spin = QSpinBox()
        self._shots_spin.setRange(1, 1_000_000)
        self._shots_spin.setValue(1024)
        self._shots_spin.setSingleStep(256)
        self._shots_spin.setMinimumWidth(100)
        controls_layout.addWidget(self._shots_spin)

        # Run button
        self._run_btn = QPushButton("Run")
        self._run_btn.setMinimumWidth(80)
        self._run_btn.clicked.connect(self._on_run_clicked)
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

    @property
    def shots(self) -> int:
        """Current number of shots selected."""
        return self._shots_spin.value()

    @property
    def measurement_basis(self) -> str:
        """Current measurement basis selection (Z, X, or Y)."""
        return self._basis_combo.currentText()

    def _on_run_clicked(self):
        self._shots = self._shots_spin.value()
        self.run_requested.emit()

    def update_histogram(self, counts: dict[str, int], shots: int) -> None:
        """Update the histogram with new measurement counts.

        Args:
            counts: Mapping of bitstring -> count.
            shots: Total number of shots performed.
        """
        self._counts = dict(counts)
        self._shots = shots
        self._shots_spin.setValue(shots)
        self._redraw()

    def _redraw(self) -> None:
        """Redraw the bar chart."""
        self._figure.clear()
        ax = self._figure.add_subplot(111)

        if not self._counts:
            ax.text(
                0.5, 0.5, "No measurement data",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=12, color="gray",
            )
            self._apply_axes_theme(ax)
            self._canvas.draw()
            self._info_label.setText("")
            return

        # Sort bitstrings
        sorted_states = sorted(self._counts.keys())
        counts = [self._counts[s] for s in sorted_states]

        show_probability = self._view_combo.currentIndex() == 0

        if show_probability:
            total = self._shots if self._shots > 0 else sum(counts)
            values = [c / total for c in counts]
            ylabel = "Probability"
        else:
            values = counts
            ylabel = "Counts"

        # Bar colors
        num_bars = len(sorted_states)
        colors = self._generate_bar_colors(num_bars)

        bars = ax.bar(range(num_bars), values, color=colors, edgecolor="none",
                      width=0.7)

        # Labels
        ax.set_xticks(range(num_bars))
        rotation = 45 if num_bars > 8 else 0
        ha = "right" if rotation > 0 else "center"
        fontsize = max(6, min(10, 120 // max(num_bars, 1)))
        ax.set_xticklabels(
            [f"|{s}\u27E9" for s in sorted_states],
            rotation=rotation, ha=ha, fontsize=fontsize,
        )

        ax.set_ylabel(ylabel)
        ax.set_xlabel("Basis State")

        # Add value labels on bars if not too many
        if num_bars <= 32:
            for bar_obj, val in zip(bars, values):
                height = bar_obj.get_height()
                if height > 0:
                    label = f"{val:.3f}" if show_probability else f"{val}"
                    ax.text(
                        bar_obj.get_x() + bar_obj.get_width() / 2.0,
                        height,
                        label,
                        ha="center", va="bottom",
                        fontsize=max(6, 9 - num_bars // 10),
                        color="white" if self._dark_theme else "black",
                    )

        self._apply_axes_theme(ax)
        self._figure.tight_layout()
        self._canvas.draw()

        # Info
        total_counts = sum(self._counts.values())
        unique = len(self._counts)
        self._info_label.setText(
            f"{total_counts} shots | {unique} unique outcomes"
        )

    def _generate_bar_colors(self, n: int) -> list[str]:
        """Generate a list of colors for bars."""
        if n <= 0:
            return []
        base_colors = [
            "#4A90D9", "#E74C3C", "#2ECC71", "#F39C12",
            "#9B59B6", "#1ABC9C", "#E67E22", "#3498DB",
            "#E91E63", "#00BCD4", "#FF9800", "#673AB7",
        ]
        colors = []
        for i in range(n):
            colors.append(base_colors[i % len(base_colors)])
        return colors

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
        """Clear the histogram."""
        self._counts = {}
        self._redraw()
