"""Bloch sphere visualization widget using matplotlib 3D plot embedded in PyQt6.

Supports two view modes:
- All Qubits: grid of small Bloch spheres showing every qubit at once.
- Single Qubit: detailed view of one selected qubit with projections and state label.
"""

from __future__ import annotations

import math
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QLabel,
    QPushButton, QButtonGroup, QSizePolicy, QCheckBox, QScrollArea,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (required for 3D projection)

from quantum_sim.engine.state_vector import StateVector

# Distinct colors for per-qubit state vectors
_QUBIT_COLORS = [
    "#FF4444",  # red
    "#44AAFF",  # blue
    "#44DD44",  # green
    "#FFAA22",  # orange
    "#DD44DD",  # magenta
    "#22DDDD",  # cyan
    "#FFDD44",  # yellow
    "#FF66AA",  # pink
    "#8866FF",  # purple
    "#66FFAA",  # teal
    "#FF8844",  # dark-orange
    "#44FFDD",  # aqua
    "#AA88FF",  # lavender
    "#AAFF44",  # lime
    "#FF4488",  # hot-pink
    "#4488FF",  # royal-blue
]

# Known quantum states on the Bloch sphere for labelling
_KNOWN_STATES = [
    ((0.0, 0.0, 1.0), "|0\u27E9"),
    ((0.0, 0.0, -1.0), "|1\u27E9"),
    ((1.0, 0.0, 0.0), "|+\u27E9"),
    ((-1.0, 0.0, 0.0), "|-\u27E9"),
    ((0.0, 1.0, 0.0), "|i\u27E9"),
    ((0.0, -1.0, 0.0), "|-i\u27E9"),
]


def _identify_state(x: float, y: float, z: float, threshold: float = 0.12) -> str | None:
    """Return the ket label if (x,y,z) is close to a known Bloch state."""
    for (sx, sy, sz), label in _KNOWN_STATES:
        dist = math.sqrt((x - sx) ** 2 + (y - sy) ** 2 + (z - sz) ** 2)
        if dist < threshold:
            return label
    return None


class BlochSphereWidget(QWidget):
    """Interactive Bloch sphere visualization for individual qubits.

    Supports a multi-qubit grid view (all qubits) and a single-qubit
    detailed view with shadow projections and state identification.
    """

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)

        self._state_vector: StateVector | None = None
        self._num_qubits: int = 1
        self._dark_theme: bool = True  # default to dark
        self._view_all: bool = True  # start with grid view
        self._show_trajectory: bool = False
        # Per-qubit trajectory: list of (x, y, z) tuples per qubit index
        self._trajectory: dict[int, list[tuple[float, float, float]]] = {}

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
        layout.setSpacing(4)

        # Controls row
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(6)

        title_label = QLabel("Bloch Sphere")
        title_label.setStyleSheet("font-weight: bold; font-size: 13px;")
        controls_layout.addWidget(title_label)
        controls_layout.addStretch()

        # View mode toggle
        self._btn_all = QPushButton("All Qubits")
        self._btn_all.setCheckable(True)
        self._btn_all.setChecked(True)
        self._btn_all.setFixedHeight(24)
        self._btn_all.setStyleSheet(self._toggle_btn_style(True))
        self._btn_all.clicked.connect(lambda: self._set_view_mode(True))

        self._btn_single = QPushButton("Single")
        self._btn_single.setCheckable(True)
        self._btn_single.setChecked(False)
        self._btn_single.setFixedHeight(24)
        self._btn_single.setStyleSheet(self._toggle_btn_style(False))
        self._btn_single.clicked.connect(lambda: self._set_view_mode(False))

        controls_layout.addWidget(self._btn_all)
        controls_layout.addWidget(self._btn_single)

        # Qubit selector (visible only in single mode)
        self._qubit_label = QLabel("Qubit:")
        controls_layout.addWidget(self._qubit_label)
        self._qubit_combo = QComboBox()
        self._qubit_combo.setMinimumWidth(60)
        self._qubit_combo.addItem("q0")
        self._qubit_combo.currentIndexChanged.connect(self._on_qubit_changed)
        controls_layout.addWidget(self._qubit_combo)

        self._qubit_label.setVisible(False)
        self._qubit_combo.setVisible(False)

        # Trajectory checkbox
        self._trajectory_cb = QCheckBox("Trajectory")
        self._trajectory_cb.setChecked(False)
        self._trajectory_cb.setToolTip("Show state trajectory during step-by-step simulation")
        self._trajectory_cb.toggled.connect(self._on_trajectory_toggled)
        controls_layout.addWidget(self._trajectory_cb)

        layout.addLayout(controls_layout)

        # Matplotlib figure
        self._figure = Figure(dpi=100)
        self._canvas = FigureCanvas(self._figure)
        self._canvas.setMinimumSize(250, 200)
        self._canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self._canvas, stretch=1)

        # Info label
        self._info_label = QLabel("")
        self._info_label.setStyleSheet("color: gray; font-size: 11px;")
        self._info_label.setWordWrap(True)
        layout.addWidget(self._info_label)

        # Initial draw
        self._update_display()

    # ---- Toggle button style ------------------------------------------------

    def _toggle_btn_style(self, active: bool) -> str:
        if active:
            return (
                "QPushButton { background: #4488CC; color: white; "
                "border: none; border-radius: 3px; padding: 2px 8px; font-size: 11px; }"
            )
        return (
            "QPushButton { background: #444444; color: #AAAAAA; "
            "border: none; border-radius: 3px; padding: 2px 8px; font-size: 11px; }"
        )

    def _set_view_mode(self, view_all: bool):
        self._view_all = view_all
        self._btn_all.setChecked(view_all)
        self._btn_single.setChecked(not view_all)
        self._btn_all.setStyleSheet(self._toggle_btn_style(view_all))
        self._btn_single.setStyleSheet(self._toggle_btn_style(not view_all))
        self._qubit_label.setVisible(not view_all)
        self._qubit_combo.setVisible(not view_all)
        self._update_display()

    # ---- Public API ---------------------------------------------------------

    def update_state(self, state_vector: StateVector, num_qubits: int) -> None:
        """Update the Bloch sphere display for the current quantum state."""
        self._state_vector = state_vector
        self._num_qubits = num_qubits

        # Update qubit selector
        current = self._qubit_combo.currentIndex()
        self._qubit_combo.blockSignals(True)
        self._qubit_combo.clear()
        for i in range(num_qubits):
            self._qubit_combo.addItem(f"q{i}")
        if 0 <= current < num_qubits:
            self._qubit_combo.setCurrentIndex(current)
        else:
            self._qubit_combo.setCurrentIndex(0)
        self._qubit_combo.blockSignals(False)

        self._update_display()

    def set_theme(self, dark: bool) -> None:
        """Switch between dark and light theme."""
        self._dark_theme = dark
        self._update_display()

    def append_trajectory_point(self, state: StateVector) -> None:
        """Record current Bloch coordinates for all qubits into the trajectory."""
        for q in range(state.num_qubits):
            bx, by, bz = state.get_bloch_coordinates(q)
            if q not in self._trajectory:
                self._trajectory[q] = []
            self._trajectory[q].append((bx, by, bz))

    def clear_trajectory(self) -> None:
        """Clear all stored trajectory points."""
        self._trajectory.clear()

    def clear(self) -> None:
        """Reset to default |0> state."""
        self._state_vector = None
        self._trajectory.clear()
        self._update_display()
        self._info_label.setText("")

    # ---- Internal -----------------------------------------------------------

    def _on_trajectory_toggled(self, checked: bool):
        self._show_trajectory = checked
        self._update_display()

    def _on_qubit_changed(self, index: int):
        if index >= 0:
            self._update_display()

    def _get_theme_colors(self) -> dict:
        if self._dark_theme:
            return {
                "bg": "#1E1E2E",
                "wire": "#45475A",
                "text": "#CDD6F4",
                "axis": "#6C7086",
                "equator": "#585B70",
                "shadow": "#313244",
            }
        return {
            "bg": "#FFFFFF",
            "wire": "#CCCCCC",
            "text": "#333333",
            "axis": "#999999",
            "equator": "#BBBBBB",
            "shadow": "#EEEEEE",
        }

    def _update_display(self):
        if self._view_all:
            self._draw_all_qubits()
        else:
            self._draw_single_qubit()

    # ---- All-qubits grid view -----------------------------------------------

    def _draw_all_qubits(self):
        self._figure.clear()
        colors = self._get_theme_colors()
        self._figure.set_facecolor(colors["bg"])

        n = self._num_qubits if self._state_vector else 1

        # Calculate grid layout
        if n <= 1:
            rows, cols = 1, 1
        elif n <= 2:
            rows, cols = 1, 2
        elif n <= 4:
            rows, cols = 2, 2
        elif n <= 6:
            rows, cols = 2, 3
        elif n <= 9:
            rows, cols = 3, 3
        elif n <= 12:
            rows, cols = 3, 4
        else:
            rows, cols = 4, 4

        info_parts = []

        for i in range(min(n, 16)):
            ax = self._figure.add_subplot(rows, cols, i + 1, projection="3d")

            if self._state_vector is not None:
                bx, by, bz = self._state_vector.get_bloch_coordinates(i)
            else:
                bx, by, bz = 0.0, 0.0, 1.0

            qcolor = _QUBIT_COLORS[i % len(_QUBIT_COLORS)]
            self._draw_mini_sphere(ax, bx, by, bz, qcolor, colors, f"q{i}")

            # Draw trajectory if enabled
            if self._show_trajectory and i in self._trajectory:
                self._draw_trajectory(ax, self._trajectory[i], qcolor)

            state_label = _identify_state(bx, by, bz)
            state_str = f" = {state_label}" if state_label else ""
            info_parts.append(f"q{i}{state_str}")

        self._figure.subplots_adjust(
            left=0.02, right=0.98, top=0.95, bottom=0.02,
            wspace=0.05, hspace=0.05,
        )
        self._canvas.draw()

        self._info_label.setText("  |  ".join(info_parts))

    def _draw_mini_sphere(
        self, ax, bx: float, by: float, bz: float,
        vec_color: str, colors: dict, label: str,
    ):
        """Draw a compact Bloch sphere for the grid view."""
        ax.set_facecolor(colors["bg"])

        # Lightweight wireframe
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 12)
        sx = np.outer(np.cos(u), np.sin(v))
        sy = np.outer(np.sin(u), np.sin(v))
        sz = np.outer(np.ones_like(u), np.cos(v))
        ax.plot_wireframe(sx, sy, sz, color=colors["wire"], alpha=0.1,
                          linewidth=0.3)

        # Equator only
        theta = np.linspace(0, 2 * np.pi, 60)
        ax.plot(np.cos(theta), np.sin(theta), np.zeros_like(theta),
                color=colors["equator"], alpha=0.3, linewidth=0.6)

        # Axes
        ax.plot([-1.1, 1.1], [0, 0], [0, 0],
                color=colors["axis"], linewidth=0.4, alpha=0.4)
        ax.plot([0, 0], [-1.1, 1.1], [0, 0],
                color=colors["axis"], linewidth=0.4, alpha=0.4)
        ax.plot([0, 0], [0, 0], [-1.1, 1.1],
                color=colors["axis"], linewidth=0.4, alpha=0.4)

        # Pole labels
        ax.text(0, 0, 1.25, "|0\u27E9", color=colors["text"],
                fontsize=7, ha="center", va="center")
        ax.text(0, 0, -1.25, "|1\u27E9", color=colors["text"],
                fontsize=7, ha="center", va="center")

        # State vector arrow
        ax.quiver(0, 0, 0, bx, by, bz,
                  color=vec_color, arrow_length_ratio=0.15,
                  linewidth=2.0, alpha=0.9)
        ax.scatter([bx], [by], [bz], color=vec_color, s=20, zorder=10)

        # XY shadow (projection onto z=0 plane)
        r_xy = math.sqrt(bx ** 2 + by ** 2)
        if r_xy > 0.05:
            ax.plot([0, bx], [0, by], [0, 0],
                    color=vec_color, linewidth=0.8, alpha=0.25, linestyle="--")
            ax.scatter([bx], [by], [0], color=vec_color, s=8, alpha=0.3)

        # Qubit label (title)
        state_id = _identify_state(bx, by, bz)
        title = f"{label}" + (f" {state_id}" if state_id else "")
        ax.set_title(title, fontsize=8, color=colors["text"], pad=-2)

        # Clean up axes
        ax.set_xlim([-1.4, 1.4])
        ax.set_ylim([-1.4, 1.4])
        ax.set_zlim([-1.4, 1.4])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        try:
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            ax.xaxis.pane.set_edgecolor("none")
            ax.yaxis.pane.set_edgecolor("none")
            ax.zaxis.pane.set_edgecolor("none")
            ax.grid(False)
        except AttributeError:
            pass
        ax.set_box_aspect([1, 1, 1])

    # ---- Single-qubit detailed view ------------------------------------------

    def _draw_single_qubit(self):
        self._figure.clear()
        colors = self._get_theme_colors()
        self._figure.set_facecolor(colors["bg"])

        ax = self._figure.add_subplot(111, projection="3d")
        ax.set_facecolor(colors["bg"])

        qubit = self._qubit_combo.currentIndex()
        if self._state_vector is not None and 0 <= qubit < self._state_vector.num_qubits:
            bx, by, bz = self._state_vector.get_bloch_coordinates(qubit)
        else:
            bx, by, bz = 0.0, 0.0, 1.0

        qcolor = _QUBIT_COLORS[qubit % len(_QUBIT_COLORS)]

        # Wireframe sphere
        u = np.linspace(0, 2 * np.pi, 36)
        v = np.linspace(0, np.pi, 24)
        sx = np.outer(np.cos(u), np.sin(v))
        sy = np.outer(np.sin(u), np.sin(v))
        sz = np.outer(np.ones_like(u), np.cos(v))
        ax.plot_wireframe(sx, sy, sz, color=colors["wire"], alpha=0.12,
                          linewidth=0.4)

        # Equator and meridians
        theta = np.linspace(0, 2 * np.pi, 100)
        for (px, py, pz), alpha in [
            ((np.cos(theta), np.sin(theta), np.zeros_like(theta)), 0.35),
            ((np.cos(theta), np.zeros_like(theta), np.sin(theta)), 0.2),
            ((np.zeros_like(theta), np.cos(theta), np.sin(theta)), 0.2),
        ]:
            ax.plot(px, py, pz, color=colors["equator"], alpha=alpha, linewidth=0.7)

        # Axes
        axis_len = 1.3
        for coords in [
            ([-axis_len, axis_len], [0, 0], [0, 0]),
            ([0, 0], [-axis_len, axis_len], [0, 0]),
            ([0, 0], [0, 0], [-axis_len, axis_len]),
        ]:
            ax.plot(*coords, color=colors["axis"], linewidth=0.7, alpha=0.5)

        # Ket labels at poles and cardinal points
        lo = 1.5
        labels = [
            (0, 0, lo, "|0\u27E9", 11), (0, 0, -lo, "|1\u27E9", 11),
            (lo, 0, 0, "|+\u27E9", 10), (-lo, 0, 0, "|-\u27E9", 10),
            (0, lo, 0, "|i\u27E9", 10), (0, -lo, 0, "|-i\u27E9", 10),
        ]
        for lx, ly, lz, txt, fs in labels:
            ax.text(lx, ly, lz, txt, color=colors["text"],
                    fontsize=fs, ha="center", va="center", fontweight="bold")

        # Axis name labels
        name_offset = lo + 0.22
        for nx, ny, nz, name in [
            (name_offset, 0, 0, "X"),
            (0, name_offset, 0, "Y"),
            (0, 0, name_offset, "Z"),
        ]:
            ax.text(nx, ny, nz, name, color=colors["axis"],
                    fontsize=9, ha="center", va="center")

        # --- State vector arrow ---
        ax.quiver(0, 0, 0, bx, by, bz,
                  color=qcolor, arrow_length_ratio=0.1,
                  linewidth=3.0, alpha=0.95)
        ax.scatter([bx], [by], [bz], color=qcolor, s=50, zorder=10,
                   edgecolors="white", linewidths=0.5)

        # --- Shadow projections ---
        r_xy = math.sqrt(bx ** 2 + by ** 2)
        # XY plane shadow
        if r_xy > 0.03:
            ax.plot([0, bx], [0, by], [0, 0],
                    color=qcolor, linewidth=1.0, alpha=0.3, linestyle="--")
            ax.scatter([bx], [by], [0], color=qcolor, s=15, alpha=0.35)
        # Vertical drop line from tip to XY plane
        if abs(bz) > 0.03 and r_xy > 0.03:
            ax.plot([bx, bx], [by, by], [0, bz],
                    color=qcolor, linewidth=0.6, alpha=0.2, linestyle=":")

        # --- Trajectory ---
        if self._show_trajectory and qubit in self._trajectory:
            self._draw_trajectory(ax, self._trajectory[qubit], qcolor, detailed=True)

        # --- State identification label ---
        state_label = _identify_state(bx, by, bz)
        if state_label:
            ax.text(bx * 0.5, by * 0.5, bz * 0.5 + 0.3, state_label,
                    color=qcolor, fontsize=12, ha="center", va="center",
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.15",
                              facecolor=colors["bg"], alpha=0.7,
                              edgecolor=qcolor, linewidth=0.8))

        # Clean up
        ax.set_xlim([-1.6, 1.6])
        ax.set_ylim([-1.6, 1.6])
        ax.set_zlim([-1.6, 1.6])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        try:
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            ax.xaxis.pane.set_edgecolor("none")
            ax.yaxis.pane.set_edgecolor("none")
            ax.zaxis.pane.set_edgecolor("none")
            ax.grid(False)
        except AttributeError:
            pass
        ax.set_box_aspect([1, 1, 1])

        self._figure.tight_layout()
        self._canvas.draw()

        # Update info label
        r = math.sqrt(bx ** 2 + by ** 2 + bz ** 2)
        # Spherical coordinates
        theta_angle = math.acos(max(-1, min(1, bz / r))) if r > 1e-9 else 0
        phi_angle = math.atan2(by, bx)
        info = (
            f"q{qubit}: "
            f"x={bx:+.4f}  y={by:+.4f}  z={bz:+.4f}  |  "
            f"r={r:.4f}  \u03B8={math.degrees(theta_angle):.1f}\u00B0  "
            f"\u03C6={math.degrees(phi_angle):.1f}\u00B0"
        )
        if state_label:
            info += f"  \u2248 {state_label}"
        if self._show_trajectory and qubit in self._trajectory:
            info += f"  | trajectory: {len(self._trajectory[qubit])} pts"
        self._info_label.setText(info)

    # ---- Trajectory drawing --------------------------------------------------

    def _draw_trajectory(
        self, ax, points: list[tuple[float, float, float]],
        color: str, detailed: bool = False,
    ):
        """Draw trajectory path on a 3D axes with gradual opacity."""
        n = len(points)
        if n < 2:
            return

        lw = 2.0 if detailed else 1.0
        marker_size = 12 if detailed else 5

        # Draw line segments with increasing alpha
        for i in range(n - 1):
            alpha = 0.15 + 0.75 * (i / max(n - 1, 1))
            x0, y0, z0 = points[i]
            x1, y1, z1 = points[i + 1]
            ax.plot([x0, x1], [y0, y1], [z0, z1],
                    color=color, linewidth=lw, alpha=alpha)

        # Draw dots at each point with increasing alpha
        for i, (px, py, pz) in enumerate(points):
            alpha = 0.2 + 0.7 * (i / max(n - 1, 1))
            s = marker_size * (0.4 + 0.6 * (i / max(n - 1, 1)))
            ax.scatter([px], [py], [pz], color=color, s=s, alpha=alpha, zorder=5)

        # Mark start with a ring
        sx, sy, sz = points[0]
        ax.scatter([sx], [sy], [sz], color=color, s=marker_size * 1.5,
                   alpha=0.5, zorder=6, marker="o", facecolors="none",
                   edgecolors=color, linewidths=1.0)
