"""Entropy evolution visualization panel for step-by-step quantum simulation."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QPushButton, QLabel,
    QScrollArea,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from quantum_sim.engine.state_vector import StateVector
from quantum_sim.engine.analysis import (
    StateAnalysis, EntanglementEventDetector, EntanglementEventType,
)

# Distinct colors for per-qubit entropy lines
_QUBIT_COLORS = [
    "#FF4444", "#44AAFF", "#44DD44", "#FFAA22", "#DD44DD",
    "#22DDDD", "#FFDD44", "#FF66AA", "#8866FF", "#66FFAA",
    "#FF8844", "#44FFDD", "#AA88FF", "#AAFF44", "#FF4488", "#4488FF",
]


class EntropyPanel(QWidget):
    """Panel displaying entropy evolution over the course of a step-by-step simulation.

    X-axis represents gate column indices, Y-axis represents entropy in bits.
    Supports three modes via a dropdown:
      - Total System: single line showing von Neumann entropy of the full state.
      - Per-Qubit: one line per qubit showing single-qubit entanglement entropy.
      - Bipartite: entanglement entropy of the first n//2 qubits vs the rest.

    Points are added incrementally via ``update_state(state, column_index)``.
    A Clear button resets the plot and stored history.
    """

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)

        self._dark_theme: bool = True
        # History: list of (step_x, data_dict)
        # data_dict keys: "total", "per_qubit" (list[float]), "bipartite"
        self._history: list[tuple[int, dict]] = []
        self._event_detector = EntanglementEventDetector()

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

        # Controls row
        controls_layout = QHBoxLayout()

        title_label = QLabel("Entropy Evolution")
        title_label.setStyleSheet("font-weight: bold; font-size: 13px;")
        controls_layout.addWidget(title_label)
        controls_layout.addStretch()

        # Mode combo
        controls_layout.addWidget(QLabel("Mode:"))
        self._mode_combo = QComboBox()
        self._mode_combo.addItems([
            "Total System", "Per-Qubit", "Bipartite", "Entanglement Events",
        ])
        self._mode_combo.currentIndexChanged.connect(self._redraw)
        controls_layout.addWidget(self._mode_combo)

        # Clear button
        self._clear_btn = QPushButton("Clear")
        self._clear_btn.setMinimumWidth(60)
        self._clear_btn.clicked.connect(self.clear)
        controls_layout.addWidget(self._clear_btn)

        layout.addLayout(controls_layout)

        # Matplotlib figure
        self._figure = Figure(figsize=(6, 3), dpi=100)
        self._canvas = FigureCanvas(self._figure)
        self._canvas.setMinimumHeight(200)
        layout.addWidget(self._canvas, stretch=1)

        # Info label at bottom
        self._info_label = QLabel("")
        self._info_label.setStyleSheet("color: gray; font-size: 11px;")
        layout.addWidget(self._info_label)

        self._apply_theme_colors()
        self._redraw()

    # ---- Public API ---------------------------------------------------------

    def update_state(self, state: StateVector, column_index: int) -> None:
        """Add a new data point from the current simulation step.

        Args:
            state: The quantum state after applying the gate column.
            column_index: The gate column index (-1 for the initial state,
                          which is displayed as step 0).
        """
        # Map column_index == -1 (initial state) to display step 0
        step_x = column_index if column_index >= 0 else 0

        num_qubits = state.num_qubits

        # Total system entropy
        total_entropy = StateAnalysis.von_neumann_entropy(state)

        # Per-qubit entanglement entropy
        per_qubit: list[float] = []
        for q in range(num_qubits):
            per_qubit.append(StateAnalysis.entanglement_entropy(state, [q]))

        # Bipartite entropy: first n//2 qubits
        half = num_qubits // 2
        if half > 0:
            bipartite_entropy = StateAnalysis.entanglement_entropy(
                state, list(range(half))
            )
        else:
            bipartite_entropy = 0.0

        data_dict = {
            "total": total_entropy,
            "per_qubit": per_qubit,
            "bipartite": bipartite_entropy,
        }

        self._history.append((step_x, data_dict))

        # Feed entanglement event detector
        self._event_detector.process_step(state, step_x)

        self._redraw()

    def clear(self) -> None:
        """Clear all stored history and reset the plot."""
        self._history.clear()
        self._event_detector.reset()
        self._redraw()

    def set_theme(self, dark: bool) -> None:
        """Switch between dark and light theme."""
        self._dark_theme = dark
        self._apply_theme_colors()
        self._redraw()

    # ---- Drawing ------------------------------------------------------------

    def _redraw(self) -> None:
        """Redraw the entropy evolution line plot."""
        self._figure.clear()
        ax = self._figure.add_subplot(111)

        if not self._history:
            ax.text(
                0.5, 0.5, "No entropy data",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=12, color="gray",
            )
            self._apply_axes_theme(ax)
            self._canvas.draw()
            self._info_label.setText("")
            return

        mode = self._mode_combo.currentIndex()  # 0=Total, 1=Per-Qubit, 2=Bipartite
        steps = [entry[0] for entry in self._history]

        if mode == 0:
            # Total System entropy
            values = [entry[1]["total"] for entry in self._history]
            ax.plot(
                steps, values,
                marker="o", markersize=4, linewidth=1.5,
                color="#44AAFF", label="Total S(\u03C1)",
            )
            ax.legend(fontsize=9, loc="upper left",
                      facecolor="none", edgecolor="none",
                      labelcolor="white" if self._dark_theme else "black")

            current_val = values[-1]
            self._info_label.setText(
                f"Total system entropy: {current_val:.6f} bits"
            )

        elif mode == 1:
            # Per-Qubit entanglement entropy
            # Determine the number of qubits from the latest entry
            num_qubits = len(self._history[-1][1]["per_qubit"])
            for q in range(num_qubits):
                q_values = []
                q_steps = []
                for step_x, data in self._history:
                    if q < len(data["per_qubit"]):
                        q_steps.append(step_x)
                        q_values.append(data["per_qubit"][q])
                color = _QUBIT_COLORS[q % len(_QUBIT_COLORS)]
                ax.plot(
                    q_steps, q_values,
                    marker="o", markersize=3, linewidth=1.2,
                    color=color, label=f"q{q}",
                )

            ax.legend(
                fontsize=8, loc="upper left", ncol=max(1, num_qubits // 4),
                facecolor="none", edgecolor="none",
                labelcolor="white" if self._dark_theme else "black",
            )

            # Info: show latest per-qubit values
            latest_pq = self._history[-1][1]["per_qubit"]
            parts = [f"q{i}={v:.4f}" for i, v in enumerate(latest_pq)]
            self._info_label.setText(
                "Per-qubit entropy (bits): " + "  ".join(parts)
            )

        elif mode == 2:
            # Bipartite entropy
            values = [entry[1]["bipartite"] for entry in self._history]
            num_qubits = len(self._history[-1][1]["per_qubit"])
            half = num_qubits // 2
            label = f"S(q0..q{half - 1} | q{half}..q{num_qubits - 1})"

            ax.plot(
                steps, values,
                marker="o", markersize=4, linewidth=1.5,
                color="#44DD44", label=label,
            )
            ax.legend(fontsize=9, loc="upper left",
                      facecolor="none", edgecolor="none",
                      labelcolor="white" if self._dark_theme else "black")

            current_val = values[-1]
            self._info_label.setText(
                f"Bipartite entropy: {current_val:.6f} bits"
            )

        else:
            # Mode 3: Entanglement Events -- pairwise MI lines + event markers
            self._draw_entanglement_events(ax)

        ax.set_xlabel("Gate Column")
        ax.set_ylabel("Entropy (bits)" if mode < 3 else "Mutual Information (bits)")

        # Ensure y-axis starts at 0
        y_min, y_max = ax.get_ylim()
        ax.set_ylim(bottom=0, top=max(y_max, 0.1))

        # Integer x-ticks when few steps
        if len(steps) <= 20:
            ax.set_xticks(sorted(set(steps)))

        self._apply_axes_theme(ax)
        self._figure.tight_layout()
        self._canvas.draw()

    def _draw_entanglement_events(self, ax) -> None:
        """Draw pairwise mutual information lines with event markers."""
        histories = self._event_detector.get_all_pair_histories()
        timeline = self._event_detector.get_timeline()

        if not histories:
            ax.text(
                0.5, 0.5, "No entanglement data yet",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=12, color="gray",
            )
            self._info_label.setText("")
            return

        # Plot MI lines for each pair
        color_idx = 0
        for pair, history in sorted(histories.items()):
            pair_steps = [h[0] for h in history]
            pair_values = [h[1] for h in history]
            color = _QUBIT_COLORS[color_idx % len(_QUBIT_COLORS)]
            ax.plot(
                pair_steps, pair_values,
                marker=".", markersize=3, linewidth=1.2,
                color=color, label=f"q{pair[0]}-q{pair[1]}",
                alpha=0.8,
            )
            color_idx += 1

        # Overlay event markers
        for evt in timeline:
            if evt.event_type == EntanglementEventType.CREATION:
                marker, color = "^", "#00FF00"
            elif evt.event_type == EntanglementEventType.DISENTANGLEMENT:
                marker, color = "v", "#FF4444"
            elif evt.event_type == EntanglementEventType.INCREASE:
                marker, color = "^", "#88FF88"
            else:
                marker, color = "v", "#FF8888"
            ax.plot(
                evt.step, evt.entropy_after,
                marker=marker, markersize=8, color=color,
                markeredgecolor="white" if self._dark_theme else "black",
                markeredgewidth=0.5, zorder=5,
            )

        ax.legend(
            fontsize=7, loc="upper left",
            ncol=max(1, len(histories) // 4),
            facecolor="none", edgecolor="none",
            labelcolor="white" if self._dark_theme else "black",
        )

        # Info: summarize events
        n_events = len(timeline)
        creations = sum(
            1 for e in timeline
            if e.event_type == EntanglementEventType.CREATION
        )
        disent = sum(
            1 for e in timeline
            if e.event_type == EntanglementEventType.DISENTANGLEMENT
        )
        self._info_label.setText(
            f"Events: {n_events} total | "
            f"{creations} creation | {disent} disentanglement"
        )

    # ---- Theme --------------------------------------------------------------

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
