"""Quantum Circuit Debugger panel -- step-through with state inspection.

Provides:
- Timeline slider + Step Forward/Backward/Reset/Run-to-Breakpoint controls
- Tab 1: State Inspector with amplitude table + probability bar chart (ideal vs actual)
- Tab 2: Noise Heatmap showing per-qubit, per-column fidelity drop
- Tab 3: Error Trace with cumulative fidelity loss curve
"""

from __future__ import annotations

import numpy as np

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QSlider, QTabWidget, QTableWidget, QTableWidgetItem,
    QHeaderView, QSpinBox, QGroupBox, QScrollArea,
)
from PyQt6.QtGui import QColor, QFont

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from quantum_sim.engine.state_vector import StateVector
from quantum_sim.engine.circuit import QuantumCircuit
from quantum_sim.engine.debugger import (
    CircuitDebugger, DebugSnapshot, NoiseImpactResult, NoiseAttribution,
)


class DebuggerPanel(QWidget):
    """Circuit debugger panel with step-through execution and state inspection.

    Signals:
        breakpoint_changed(int, bool): (column, is_set) when breakpoints change.
        position_changed(int): current column index when stepping.
    """

    breakpoint_changed = pyqtSignal(int, bool)
    position_changed = pyqtSignal(int)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._dark_theme: bool = True
        self._debugger = CircuitDebugger()
        self._circuit: QuantumCircuit | None = None
        self._noise_model = None
        self._noise_results: list[NoiseImpactResult] = []
        self._attribution: NoiseAttribution | None = None
        self._setup_ui()

    # ---- UI Setup ---------------------------------------------------------

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

        # Title
        title = QLabel("Circuit Debugger")
        title.setStyleSheet("font-weight: bold; font-size: 13px;")
        layout.addWidget(title)

        # Control bar
        ctrl_layout = QHBoxLayout()
        ctrl_layout.setSpacing(4)

        self._btn_reset = QPushButton("Reset")
        self._btn_reset.setToolTip("Reset to initial state")
        self._btn_reset.clicked.connect(self._on_reset)
        ctrl_layout.addWidget(self._btn_reset)

        self._btn_back = QPushButton("<< Back")
        self._btn_back.setToolTip("Step backward one column")
        self._btn_back.clicked.connect(self._on_step_back)
        ctrl_layout.addWidget(self._btn_back)

        self._btn_forward = QPushButton("Fwd >>")
        self._btn_forward.setToolTip("Step forward one column")
        self._btn_forward.clicked.connect(self._on_step_forward)
        ctrl_layout.addWidget(self._btn_forward)

        self._btn_run_bp = QPushButton("Run to BP")
        self._btn_run_bp.setToolTip("Run to next breakpoint")
        self._btn_run_bp.clicked.connect(self._on_run_to_bp)
        ctrl_layout.addWidget(self._btn_run_bp)

        ctrl_layout.addStretch()

        ctrl_layout.addWidget(QLabel("Trials:"))
        self._trials_spin = QSpinBox()
        self._trials_spin.setRange(10, 500)
        self._trials_spin.setValue(50)
        self._trials_spin.setToolTip("Noise impact averaging trials")
        ctrl_layout.addWidget(self._trials_spin)

        self._btn_debug = QPushButton("Run Debug")
        self._btn_debug.setToolTip("Execute full debug analysis")
        self._btn_debug.setStyleSheet("font-weight: bold;")
        self._btn_debug.clicked.connect(self._on_run_debug)
        ctrl_layout.addWidget(self._btn_debug)

        layout.addLayout(ctrl_layout)

        # Timeline slider
        slider_layout = QHBoxLayout()
        self._step_label = QLabel("Step: -- / --")
        self._step_label.setMinimumWidth(120)
        slider_layout.addWidget(self._step_label)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(0)
        self._slider.valueChanged.connect(self._on_slider_changed)
        slider_layout.addWidget(self._slider, stretch=1)

        self._col_label = QLabel("Col: --")
        self._col_label.setMinimumWidth(80)
        slider_layout.addWidget(self._col_label)

        self._fid_label = QLabel("Fidelity: --")
        self._fid_label.setMinimumWidth(120)
        slider_layout.addWidget(self._fid_label)

        layout.addLayout(slider_layout)

        # Tabbed content
        self._tabs = QTabWidget()
        self._tabs.setDocumentMode(True)

        # Tab 1: State Inspector
        self._state_tab = QWidget()
        self._setup_state_inspector()
        self._tabs.addTab(self._state_tab, "State Inspector")

        # Tab 2: Noise Heatmap
        self._heatmap_tab = QWidget()
        self._setup_noise_heatmap()
        self._tabs.addTab(self._heatmap_tab, "Noise Heatmap")

        # Tab 3: Error Trace
        self._trace_tab = QWidget()
        self._setup_error_trace()
        self._tabs.addTab(self._trace_tab, "Error Trace")

        layout.addWidget(self._tabs, stretch=1)

    def _setup_state_inspector(self):
        layout = QVBoxLayout(self._state_tab)
        layout.setContentsMargins(2, 2, 2, 2)

        # Info bar
        info_layout = QHBoxLayout()
        self._info_gates = QLabel("Gates: --")
        info_layout.addWidget(self._info_gates)
        self._info_entropy = QLabel("Entropy: --")
        info_layout.addWidget(self._info_entropy)
        info_layout.addStretch()
        layout.addLayout(info_layout)

        # Split: table + bar chart
        content_layout = QHBoxLayout()

        # Amplitude table
        self._amp_table = QTableWidget()
        self._amp_table.setColumnCount(5)
        self._amp_table.setHorizontalHeaderLabels(
            ["Basis", "Amplitude", "Phase", "Prob", "Ideal Prob"]
        )
        self._amp_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self._amp_table.setAlternatingRowColors(True)
        self._amp_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        content_layout.addWidget(self._amp_table, stretch=1)

        # Probability bar chart
        self._prob_figure = Figure(figsize=(4, 3), dpi=100)
        self._prob_canvas = FigureCanvas(self._prob_figure)
        self._prob_canvas.setMinimumWidth(250)
        content_layout.addWidget(self._prob_canvas, stretch=1)

        layout.addLayout(content_layout, stretch=1)

    def _setup_noise_heatmap(self):
        layout = QVBoxLayout(self._heatmap_tab)
        layout.setContentsMargins(2, 2, 2, 2)

        self._heatmap_info = QLabel("Run debug with noise model to see heatmap.")
        layout.addWidget(self._heatmap_info)

        self._heatmap_figure = Figure(figsize=(6, 3), dpi=100)
        self._heatmap_canvas = FigureCanvas(self._heatmap_figure)
        layout.addWidget(self._heatmap_canvas, stretch=1)

        # Attribution summary text
        self._attr_label = QLabel("")
        self._attr_label.setWordWrap(True)
        self._attr_label.setStyleSheet("font-size: 11px; padding: 2px;")
        layout.addWidget(self._attr_label)

    def _setup_error_trace(self):
        layout = QVBoxLayout(self._trace_tab)
        layout.setContentsMargins(2, 2, 2, 2)

        self._trace_info = QLabel("Run debug to see error trace.")
        layout.addWidget(self._trace_info)

        self._trace_figure = Figure(figsize=(6, 3), dpi=100)
        self._trace_canvas = FigureCanvas(self._trace_figure)
        layout.addWidget(self._trace_canvas, stretch=1)

    # ---- Public API -------------------------------------------------------

    def set_circuit(self, circuit: QuantumCircuit) -> None:
        self._circuit = circuit

    def set_noise_model(self, noise_model) -> None:
        self._noise_model = noise_model

    def update_state(self, state: StateVector) -> None:
        """Called by main window after simulation; shows final state in inspector."""
        pass  # Debugger uses its own execution; this is a no-op

    def clear(self) -> None:
        """Reset debugger state."""
        self._debugger = CircuitDebugger()
        self._noise_results = []
        self._attribution = None
        self._slider.setMaximum(0)
        self._step_label.setText("Step: -- / --")
        self._col_label.setText("Col: --")
        self._fid_label.setText("Fidelity: --")
        self._info_gates.setText("Gates: --")
        self._info_entropy.setText("Entropy: --")
        self._amp_table.setRowCount(0)
        self._prob_figure.clear()
        self._prob_canvas.draw_idle()
        self._heatmap_figure.clear()
        self._heatmap_canvas.draw_idle()
        self._trace_figure.clear()
        self._trace_canvas.draw_idle()
        self._heatmap_info.setText("Run debug with noise model to see heatmap.")
        self._trace_info.setText("Run debug to see error trace.")

    def set_theme(self, dark: bool) -> None:
        self._dark_theme = dark
        self._redraw_all()

    # ---- Button handlers --------------------------------------------------

    def _on_run_debug(self):
        if self._circuit is None:
            return

        seed = 42  # Default seed for reproducibility
        self._debugger.run_full_debug(
            self._circuit,
            noise_model=self._noise_model,
            seed=seed,
        )

        # Compute noise impact and attribution if noise model is set
        if self._noise_model is not None:
            n_trials = self._trials_spin.value()
            self._noise_results = self._debugger.compute_noise_impact(
                self._circuit,
                self._noise_model,
                n_trials=n_trials,
                seed=seed,
            )
            self._attribution = self._debugger.compute_noise_attribution(
                self._circuit,
                self._noise_model,
                n_trials=n_trials,
                seed=seed,
            )
        else:
            self._noise_results = []
            self._attribution = None

        # Update slider
        n = self._debugger.num_steps
        self._slider.blockSignals(True)
        self._slider.setMaximum(n - 1)
        self._slider.setValue(0)
        self._slider.blockSignals(False)

        self._update_display()
        self._draw_heatmap()
        self._draw_error_trace()

    def _on_reset(self):
        if self._debugger.num_steps == 0:
            return
        self._debugger.goto_step(0)
        self._slider.blockSignals(True)
        self._slider.setValue(0)
        self._slider.blockSignals(False)
        self._update_display()

    def _on_step_forward(self):
        snap = self._debugger.step_forward()
        if snap is None:
            return
        self._slider.blockSignals(True)
        self._slider.setValue(self._debugger.position)
        self._slider.blockSignals(False)
        self._update_display()

    def _on_step_back(self):
        snap = self._debugger.step_backward()
        if snap is None:
            return
        self._slider.blockSignals(True)
        self._slider.setValue(self._debugger.position)
        self._slider.blockSignals(False)
        self._update_display()

    def _on_run_to_bp(self):
        snap = self._debugger.run_to_breakpoint()
        if snap is None:
            return
        self._slider.blockSignals(True)
        self._slider.setValue(self._debugger.position)
        self._slider.blockSignals(False)
        self._update_display()

    def _on_slider_changed(self, value: int):
        self._debugger.goto_step(value)
        self._update_display()

    # ---- Breakpoint management (called from scene) ------------------------

    def toggle_breakpoint(self, column: int) -> bool:
        result = self._debugger.toggle_breakpoint(column)
        self.breakpoint_changed.emit(column, result)
        return result

    @property
    def breakpoints(self) -> set[int]:
        return self._debugger.breakpoints

    # ---- Display updates --------------------------------------------------

    def _update_display(self):
        snap = self._debugger.current_snapshot
        if snap is None:
            return

        pos = self._debugger.position
        total = self._debugger.num_steps

        self._step_label.setText(f"Step: {pos} / {total - 1}")

        if snap.column_index < 0:
            self._col_label.setText("Col: init")
        else:
            self._col_label.setText(f"Col: {snap.column_index}")

        self._fid_label.setText(f"Fidelity: {snap.fidelity:.6f}")
        self._info_gates.setText(
            f"Gates: {', '.join(snap.gate_labels) if snap.gate_labels else 'none'}"
        )
        self._info_entropy.setText(f"Entropy: {snap.entropy:.4f} bits")

        self._update_amplitude_table(snap)
        self._draw_probability_chart(snap)

        self.position_changed.emit(snap.column_index)

    def _update_amplitude_table(self, snap: DebugSnapshot):
        state = snap.state
        n = state.num_qubits
        dim = 2 ** n
        data = state.data
        probs = state.probabilities

        ideal_probs = None
        if snap.ideal_state is not None:
            ideal_probs = snap.ideal_state.probabilities

        # Only show non-negligible amplitudes (or all if <= 32)
        if dim <= 32:
            indices = list(range(dim))
        else:
            indices = [i for i in range(dim) if probs[i] > 1e-8]
            if not indices:
                indices = list(range(min(16, dim)))

        self._amp_table.setRowCount(len(indices))
        for row, idx in enumerate(indices):
            # Basis label
            basis = format(idx, f"0{n}b")
            item = QTableWidgetItem(f"|{basis}>")
            item.setTextAlignment(int(Qt.AlignmentFlag.AlignCenter))
            self._amp_table.setItem(row, 0, item)

            # Amplitude
            amp = data[idx]
            amp_str = f"{amp.real:+.4f}{amp.imag:+.4f}i"
            item = QTableWidgetItem(amp_str)
            item.setTextAlignment(int(Qt.AlignmentFlag.AlignCenter))
            self._amp_table.setItem(row, 1, item)

            # Phase (degrees)
            phase = np.angle(amp, deg=True)
            item = QTableWidgetItem(f"{phase:.1f} deg")
            item.setTextAlignment(int(Qt.AlignmentFlag.AlignCenter))
            self._amp_table.setItem(row, 2, item)

            # Probability
            item = QTableWidgetItem(f"{probs[idx]:.6f}")
            item.setTextAlignment(int(Qt.AlignmentFlag.AlignCenter))
            if probs[idx] > 0.01:
                item.setBackground(QColor(70, 130, 180, 60))
            self._amp_table.setItem(row, 3, item)

            # Ideal probability
            if ideal_probs is not None:
                item = QTableWidgetItem(f"{ideal_probs[idx]:.6f}")
            else:
                item = QTableWidgetItem("--")
            item.setTextAlignment(int(Qt.AlignmentFlag.AlignCenter))
            self._amp_table.setItem(row, 4, item)

    def _draw_probability_chart(self, snap: DebugSnapshot):
        fig = self._prob_figure
        fig.clear()
        ax = fig.add_subplot(111)

        state = snap.state
        n = state.num_qubits
        probs = state.probabilities

        # Only show significant states
        threshold = 1e-4
        if 2 ** n <= 16:
            indices = list(range(2 ** n))
        else:
            indices = [i for i in range(2 ** n) if probs[i] > threshold]
            if not indices:
                indices = list(range(min(8, 2 ** n)))

        labels = [f"|{format(i, f'0{n}b')}>" for i in indices]
        vals = [probs[i] for i in indices]

        bar_width = 0.35
        x = np.arange(len(indices))

        if snap.ideal_state is not None:
            ideal_probs = snap.ideal_state.probabilities
            ideal_vals = [ideal_probs[i] for i in indices]
            ax.bar(x - bar_width / 2, ideal_vals, bar_width,
                   label="Ideal", color="#4488FF", alpha=0.7)
            ax.bar(x + bar_width / 2, vals, bar_width,
                   label="Actual", color="#FF6644", alpha=0.7)
            ax.legend(fontsize=8)
        else:
            ax.bar(x, vals, bar_width * 1.5, color="#4488FF", alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Probability", fontsize=9)
        ax.set_title("State Probabilities", fontsize=10)
        ax.set_ylim(0, max(max(vals) * 1.15, 0.1) if vals else 1.0)

        self._apply_fig_theme(ax)
        fig.tight_layout()
        self._prob_canvas.draw_idle()

    def _draw_heatmap(self):
        fig = self._heatmap_figure
        fig.clear()

        if not self._noise_results:
            self._heatmap_info.setText(
                "No noise model configured. Set a noise model and re-run."
            )
            self._attr_label.setText("")
            self._heatmap_canvas.draw_idle()
            return

        self._heatmap_info.setText(
            f"Noise heatmap ({self._trials_spin.value()} trials averaged)"
        )

        ax = fig.add_subplot(111)

        num_cols = len(self._noise_results)
        num_qubits = len(self._noise_results[0].per_qubit_fidelity)

        # Build heatmap data: rows = qubits, cols = gate columns
        # Value = 1 - per_qubit_fidelity (fidelity drop)
        heatmap_data = np.zeros((num_qubits, num_cols))
        for c, nr in enumerate(self._noise_results):
            for q in range(num_qubits):
                heatmap_data[q, c] = 1.0 - nr.per_qubit_fidelity[q]

        im = ax.imshow(
            heatmap_data,
            aspect="auto",
            cmap="YlOrRd",
            interpolation="nearest",
            vmin=0,
        )

        # Labels
        col_labels = []
        for nr in self._noise_results:
            label = f"C{nr.column_index}"
            if nr.gate_labels:
                label += f"\n{nr.gate_labels[0][:8]}"
            col_labels.append(label)

        ax.set_xticks(range(num_cols))
        ax.set_xticklabels(col_labels, fontsize=7, rotation=45, ha="right")
        ax.set_yticks(range(num_qubits))
        ax.set_yticklabels([f"q{i}" for i in range(num_qubits)], fontsize=8)
        ax.set_xlabel("Gate Column", fontsize=9)
        ax.set_ylabel("Qubit", fontsize=9)
        ax.set_title("Per-Qubit Fidelity Drop (1 - F)", fontsize=10)

        # Overlay attribution percentages if available
        if self._attribution is not None:
            for c in range(num_cols):
                pct = self._attribution.column_attribution_pct[c]
                # Place attribution % above the heatmap columns
                ax.text(
                    c, -0.6, f"{pct:.1f}%",
                    ha="center", va="center", fontsize=7,
                    fontweight="bold",
                    color="#FF6644" if pct > 20 else (
                        "#CCCCCC" if self._dark_theme else "#333333"
                    ),
                )
            ax.set_ylim(num_qubits - 0.5, -1.0)

        fig.colorbar(im, ax=ax, shrink=0.8, label="Fidelity Drop")

        self._apply_fig_theme(ax)
        fig.tight_layout()
        self._heatmap_canvas.draw_idle()

        # Update attribution summary text
        self._update_attribution_summary()

    def _draw_error_trace(self):
        fig = self._trace_figure
        fig.clear()

        snapshots = self._debugger.snapshots
        if len(snapshots) < 2:
            self._trace_info.setText("Run debug to see error trace.")
            self._trace_canvas.draw_idle()
            return

        self._trace_info.setText("Cumulative fidelity and entropy over execution")

        ax1 = fig.add_subplot(111)

        cols = []
        fidelities = []
        cum_fidelities = []
        entropies = []

        for snap in snapshots:
            cols.append(snap.column_index if snap.column_index >= 0 else -0.5)
            fidelities.append(snap.fidelity)
            cum_fidelities.append(snap.cumulative_fidelity)
            entropies.append(snap.entropy)

        # Plot fidelity
        x = range(len(cols))
        ax1.plot(x, fidelities, "o-", color="#4488FF", markersize=4,
                 label="Step Fidelity", linewidth=1.5)
        ax1.plot(x, cum_fidelities, "s--", color="#FF6644", markersize=4,
                 label="Cumulative Fidelity", linewidth=1.5)

        # Mark breakpoints
        for bp in self._debugger.breakpoints:
            for i, snap in enumerate(snapshots):
                if snap.column_index == bp:
                    ax1.axvline(x=i, color="#FF4444", linestyle=":",
                                alpha=0.6, linewidth=1.5)

        ax1.set_ylabel("Fidelity", fontsize=9, color="#4488FF")
        ax1.set_ylim(-0.05, 1.05)
        ax1.legend(loc="upper left", fontsize=8)

        # Entropy on secondary axis
        ax2 = ax1.twinx()
        ax2.plot(x, entropies, "^-", color="#44DD44", markersize=4,
                 label="Entropy", linewidth=1.5, alpha=0.8)
        ax2.set_ylabel("Entropy (bits)", fontsize=9, color="#44DD44")
        ax2.legend(loc="upper right", fontsize=8)

        # X-axis labels
        x_labels = []
        for snap in snapshots:
            if snap.column_index < 0:
                x_labels.append("init")
            else:
                x_labels.append(f"C{snap.column_index}")
        ax1.set_xticks(list(x))
        ax1.set_xticklabels(x_labels, fontsize=7, rotation=45, ha="right")
        ax1.set_xlabel("Execution Step", fontsize=9)
        ax1.set_title("Error Trace", fontsize=10)

        self._apply_fig_theme(ax1)
        self._apply_fig_theme(ax2)
        fig.tight_layout()
        self._trace_canvas.draw_idle()

    def _update_attribution_summary(self):
        """Update the attribution summary label below the heatmap."""
        if self._attribution is None:
            self._attr_label.setText("")
            return

        attr = self._attribution
        lines = [f"Total fidelity loss: {attr.total_fidelity_loss:.4f}"]

        if attr.no_measurable_loss:
            lines.append("No measurable fidelity loss (noise too low or noiseless).")
        else:
            # Find top 3 contributors
            indexed = list(enumerate(attr.column_attribution_pct))
            indexed.sort(key=lambda x: x[1], reverse=True)
            top = indexed[:min(3, len(indexed))]

            lines.append("Top contributors:")
            for col_idx, pct in top:
                labels = attr.gate_labels[col_idx]
                gate_str = ", ".join(labels) if labels else f"Col {col_idx}"
                df = attr.delta_fidelity[col_idx]
                std = attr.delta_fidelity_std[col_idx]
                lines.append(
                    f"  Col {col_idx} ({gate_str}): "
                    f"{pct:.1f}% (dF={df:.4f} +/- {std:.4f})"
                )

        self._attr_label.setText("\n".join(lines))

    # ---- Theme helpers ----------------------------------------------------

    def _apply_fig_theme(self, ax):
        if self._dark_theme:
            bg = "#2B2B2B"
            fg = "#CCCCCC"
        else:
            bg = "#FFFFFF"
            fg = "#333333"

        ax.set_facecolor(bg)
        ax.figure.set_facecolor(bg)
        ax.tick_params(colors=fg, labelsize=8)
        ax.xaxis.label.set_color(fg)
        ax.yaxis.label.set_color(fg)
        ax.title.set_color(fg)
        for spine in ax.spines.values():
            spine.set_color(fg)

    def _redraw_all(self):
        snap = self._debugger.current_snapshot
        if snap is not None:
            self._draw_probability_chart(snap)
        self._draw_heatmap()
        self._draw_error_trace()
