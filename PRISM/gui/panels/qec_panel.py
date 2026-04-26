"""QEC Visualizer panel -- quantum error correction codes.

Provides:
- Code selection (Bit-Flip, Phase-Flip, Steane)
- Tab 1: Code Layout (qubit layout diagram)
- Tab 2: Syndrome (syndrome bits + correction + fidelity)
- Tab 3: Threshold (logical vs physical error rate curve)
"""

from __future__ import annotations

import numpy as np

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView,
    QComboBox, QDoubleSpinBox, QSpinBox, QGroupBox, QProgressBar,
    QScrollArea,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from quantum_sim.engine.qec import (
    QECSimulator, QECResult, ThresholdPoint,
    BitFlipCode, PhaseFlipCode, SteaneCode,
    AVAILABLE_CODES,
)


class _ThresholdWorker(QThread):
    """Worker thread for threshold sweep computation."""

    progress = pyqtSignal(int)
    finished = pyqtSignal(list)  # list[ThresholdPoint]

    def __init__(self, simulator: QECSimulator, noise_probs: list[float],
                 n_trials: int, noise_type: str, seed: int):
        super().__init__()
        self._sim = simulator
        self._probs = noise_probs
        self._n_trials = n_trials
        self._noise_type = noise_type
        self._seed = seed

    def run(self):
        results = self._sim.threshold_sweep(
            self._probs,
            n_trials=self._n_trials,
            noise_type=self._noise_type,
            seed=self._seed,
        )
        self.finished.emit(results)


class QECPanel(QWidget):
    """Panel for QEC code visualization and threshold analysis."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._dark_theme: bool = True
        self._code = BitFlipCode()
        self._simulator = QECSimulator(self._code)
        self._last_result: QECResult | None = None
        self._threshold_results: list[ThresholdPoint] = []
        self._worker: _ThresholdWorker | None = None
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

        title = QLabel("QEC Visualizer")
        title.setStyleSheet("font-weight: bold; font-size: 13px;")
        layout.addWidget(title)

        # Controls
        ctrl = QHBoxLayout()

        ctrl.addWidget(QLabel("Code:"))
        self._code_combo = QComboBox()
        for name in AVAILABLE_CODES:
            self._code_combo.addItem(name)
        self._code_combo.currentIndexChanged.connect(self._on_code_changed)
        ctrl.addWidget(self._code_combo)

        ctrl.addWidget(QLabel("Noise:"))
        self._noise_combo = QComboBox()
        self._noise_combo.addItems(["bit_flip", "phase_flip", "depolarizing"])
        ctrl.addWidget(self._noise_combo)

        ctrl.addWidget(QLabel("p:"))
        self._prob_spin = QDoubleSpinBox()
        self._prob_spin.setRange(0.0, 0.5)
        self._prob_spin.setValue(0.1)
        self._prob_spin.setSingleStep(0.01)
        self._prob_spin.setDecimals(3)
        ctrl.addWidget(self._prob_spin)

        ctrl.addWidget(QLabel("Logical:"))
        self._logical_combo = QComboBox()
        self._logical_combo.addItems(["|0>", "|1>"])
        ctrl.addWidget(self._logical_combo)

        ctrl.addStretch()

        self._btn_cycle = QPushButton("Run Cycle")
        self._btn_cycle.setStyleSheet("font-weight: bold;")
        self._btn_cycle.clicked.connect(self._on_run_cycle)
        ctrl.addWidget(self._btn_cycle)

        layout.addLayout(ctrl)

        # Threshold controls
        thresh_ctrl = QHBoxLayout()
        thresh_ctrl.addWidget(QLabel("Trials:"))
        self._trials_spin = QSpinBox()
        self._trials_spin.setRange(10, 1000)
        self._trials_spin.setValue(100)
        thresh_ctrl.addWidget(self._trials_spin)

        self._btn_threshold = QPushButton("Threshold Sweep")
        self._btn_threshold.clicked.connect(self._on_threshold_sweep)
        thresh_ctrl.addWidget(self._btn_threshold)

        self._progress = QProgressBar()
        self._progress.setMaximum(100)
        thresh_ctrl.addWidget(self._progress, stretch=1)

        self._status_label = QLabel("Select code and run cycle.")
        thresh_ctrl.addWidget(self._status_label)

        layout.addLayout(thresh_ctrl)

        # Tabs
        self._tabs = QTabWidget()
        self._tabs.setDocumentMode(True)

        # Tab 1: Code Layout
        self._layout_tab = QWidget()
        self._setup_layout_tab()
        self._tabs.addTab(self._layout_tab, "Code Layout")

        # Tab 2: Syndrome
        self._syndrome_tab = QWidget()
        self._setup_syndrome_tab()
        self._tabs.addTab(self._syndrome_tab, "Syndrome")

        # Tab 3: Threshold
        self._threshold_tab = QWidget()
        self._setup_threshold_tab()
        self._tabs.addTab(self._threshold_tab, "Threshold")

        layout.addWidget(self._tabs, stretch=1)

    def _setup_layout_tab(self):
        layout = QVBoxLayout(self._layout_tab)
        layout.setContentsMargins(2, 2, 2, 2)
        self._layout_figure = Figure(figsize=(6, 4), dpi=100)
        self._layout_canvas = FigureCanvas(self._layout_figure)
        layout.addWidget(self._layout_canvas, stretch=1)
        self._draw_code_layout()

    def _setup_syndrome_tab(self):
        layout = QVBoxLayout(self._syndrome_tab)
        layout.setContentsMargins(2, 2, 2, 2)

        # Info labels
        info = QHBoxLayout()
        self._syn_fid_label = QLabel("Fidelity: before=-- after=--")
        info.addWidget(self._syn_fid_label)
        self._syn_correction_label = QLabel("Correction: --")
        info.addWidget(self._syn_correction_label)
        info.addStretch()
        self._zl_label = QLabel("<Z_L>: --")
        self._zl_label.setMinimumWidth(120)
        info.addWidget(self._zl_label)
        layout.addLayout(info)

        # Syndrome table
        self._syn_table = QTableWidget()
        self._syn_table.setMaximumHeight(120)
        self._syn_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self._syn_table)

        # Fidelity bar chart
        self._syn_figure = Figure(figsize=(6, 2.5), dpi=100)
        self._syn_canvas = FigureCanvas(self._syn_figure)
        layout.addWidget(self._syn_canvas, stretch=1)

    def _setup_threshold_tab(self):
        layout = QVBoxLayout(self._threshold_tab)
        layout.setContentsMargins(2, 2, 2, 2)
        self._thresh_figure = Figure(figsize=(6, 4), dpi=100)
        self._thresh_canvas = FigureCanvas(self._thresh_figure)
        layout.addWidget(self._thresh_canvas, stretch=1)

    # ---- Public API -------------------------------------------------------

    def set_circuit(self, circuit) -> None:
        pass

    def update_state(self, state) -> None:
        pass

    def clear(self) -> None:
        self._last_result = None
        self._threshold_results = []
        self._syn_fid_label.setText("Fidelity: before=-- after=--")
        self._syn_correction_label.setText("Correction: --")
        self._zl_label.setText("<Z_L>: --")
        self._zl_label.setStyleSheet("")
        self._syn_table.setRowCount(0)
        self._syn_figure.clear()
        self._syn_canvas.draw_idle()
        self._thresh_figure.clear()
        self._thresh_canvas.draw_idle()

    def set_theme(self, dark: bool) -> None:
        self._dark_theme = dark
        self._draw_code_layout()
        if self._last_result:
            self._draw_syndrome_chart()
        if self._threshold_results:
            self._draw_threshold()

    # ---- Handlers ---------------------------------------------------------

    def _on_code_changed(self, index: int):
        name = self._code_combo.currentText()
        if name in AVAILABLE_CODES:
            self._code = AVAILABLE_CODES[name]()
            self._simulator = QECSimulator(self._code)
            self._draw_code_layout()
            self._status_label.setText(f"Code: {self._code.name}")

    def _on_run_cycle(self):
        logical = self._logical_combo.currentIndex()
        result = self._simulator.run_cycle(
            logical_state=logical,
            noise_type=self._noise_combo.currentText(),
            noise_prob=self._prob_spin.value(),
            seed=42,
        )
        self._last_result = result
        self._update_syndrome_display(result, logical)
        self._status_label.setText(
            f"Cycle done: fid {result.fidelity_before:.4f} -> {result.fidelity_after:.4f}"
        )

    def _on_threshold_sweep(self):
        probs = np.linspace(0.001, 0.3, 15).tolist()

        self._worker = _ThresholdWorker(
            self._simulator, probs,
            n_trials=self._trials_spin.value(),
            noise_type=self._noise_combo.currentText(),
            seed=42,
        )
        self._worker.finished.connect(self._on_threshold_finished)
        self._btn_threshold.setEnabled(False)
        self._status_label.setText("Running threshold sweep...")
        self._progress.setValue(50)  # indeterminate style
        self._worker.start()

    def _on_threshold_finished(self, results: list):
        self._threshold_results = results
        self._draw_threshold()
        self._btn_threshold.setEnabled(True)
        self._progress.setValue(100)
        self._status_label.setText("Threshold sweep complete.")

    # ---- Drawing ----------------------------------------------------------

    def _draw_code_layout(self):
        fig = self._layout_figure
        fig.clear()
        ax = fig.add_subplot(111)
        ax.set_aspect("equal")

        code = self._code
        n_data = code.data_qubits
        n_ancilla = code.ancilla_qubits
        total = code.total_qubits

        # Draw data qubits
        data_x = np.arange(n_data) * 1.5
        data_y = np.zeros(n_data)

        ax.scatter(data_x, data_y, s=400, c="#4488FF", zorder=5, edgecolors="white", linewidth=2)
        for i in range(n_data):
            ax.text(data_x[i], data_y[i], f"d{i}", ha="center", va="center",
                    fontsize=9, fontweight="bold", color="white")

        # Draw ancilla qubits
        if n_ancilla > 0:
            anc_x = np.linspace(data_x[0], data_x[-1], n_ancilla)
            anc_y = np.ones(n_ancilla) * -1.5

            ax.scatter(anc_x, anc_y, s=400, c="#FF4444", zorder=5,
                       edgecolors="white", linewidth=2, marker="s")
            for i in range(n_ancilla):
                ax.text(anc_x[i], anc_y[i], f"a{i}", ha="center", va="center",
                        fontsize=9, fontweight="bold", color="white")

            # Draw connections (parity checks)
            if isinstance(code, (BitFlipCode, PhaseFlipCode)):
                checks = [(0, 1), (1, 2)]
                for ci, (qa, qb) in enumerate(checks):
                    ax.plot([data_x[qa], anc_x[ci]], [data_y[qa], anc_y[ci]],
                            ":", color="#888888", linewidth=1.5, alpha=0.6)
                    ax.plot([data_x[qb], anc_x[ci]], [data_y[qb], anc_y[ci]],
                            ":", color="#888888", linewidth=1.5, alpha=0.6)

        # Labels
        ax.set_title(f"{code.name}  ({n_data} data + {n_ancilla} ancilla = {total} qubits)",
                     fontsize=11)
        ax.set_xlim(-1, max(data_x[-1] + 1, 2))
        ax.set_ylim(-3, 2)
        ax.axis("off")

        self._apply_fig_theme(ax)
        fig.tight_layout()
        self._layout_canvas.draw_idle()

    def _update_syndrome_display(self, result: QECResult, logical: int):
        # Fidelity labels
        self._syn_fid_label.setText(
            f"Fidelity: before={result.fidelity_before:.6f}  "
            f"after={result.fidelity_after:.6f}"
        )

        corr_str = ", ".join(f"{g}(q{q})" for g, q in result.correction_applied)
        self._syn_correction_label.setText(
            f"Correction: {corr_str if corr_str else 'none needed'}"
        )

        # Z_L indicator
        z_val = result.logical_z_expectation
        if result.logical_error_detected:
            self._zl_label.setText(f"<Z_L>: {z_val:+.4f}")
            self._zl_label.setStyleSheet(
                "color: #FF4444; font-weight: bold; font-size: 12px;"
            )
        else:
            self._zl_label.setText(f"<Z_L>: {z_val:+.4f}")
            self._zl_label.setStyleSheet(
                "color: #44DD44; font-weight: bold; font-size: 12px;"
            )

        # Syndrome table
        n_syn = len(result.syndrome)
        self._syn_table.setColumnCount(n_syn + 1)
        headers = [f"S{i}" for i in range(n_syn)] + ["Correction"]
        self._syn_table.setHorizontalHeaderLabels(headers)
        self._syn_table.setRowCount(1)

        for i, s in enumerate(result.syndrome):
            item = QTableWidgetItem(str(s))
            item.setTextAlignment(int(Qt.AlignmentFlag.AlignCenter))
            if s == 1:
                item.setBackground(Qt.GlobalColor.red)
                item.setForeground(Qt.GlobalColor.white)
            self._syn_table.setItem(0, i, item)

        item = QTableWidgetItem(corr_str if corr_str else "None")
        item.setTextAlignment(int(Qt.AlignmentFlag.AlignCenter))
        self._syn_table.setItem(0, n_syn, item)
        self._syn_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )

        self._draw_syndrome_chart()

    def _draw_syndrome_chart(self):
        fig = self._syn_figure
        fig.clear()
        r = self._last_result
        if r is None:
            self._syn_canvas.draw_idle()
            return

        ax = fig.add_subplot(111)

        labels = ["Before", "After"]
        values = [r.fidelity_before, r.fidelity_after]
        colors = ["#FF6644", "#44DD44"]

        bars = ax.bar(labels, values, color=colors, alpha=0.8, width=0.5)
        ax.set_ylabel("Fidelity", fontsize=9)
        ax.set_title("Error Correction Fidelity", fontsize=10)
        ax.set_ylim(0, 1.05)

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{val:.4f}", ha="center", fontsize=9)

        self._apply_fig_theme(ax)
        fig.tight_layout()
        self._syn_canvas.draw_idle()

    def _draw_threshold(self):
        fig = self._thresh_figure
        fig.clear()

        if not self._threshold_results:
            self._thresh_canvas.draw_idle()
            return

        ax = fig.add_subplot(111)

        phys_rates = [r.physical_rate for r in self._threshold_results]
        log_rates = [r.logical_rate for r in self._threshold_results]
        fidelities = [r.avg_fidelity for r in self._threshold_results]

        ax.plot(phys_rates, log_rates, "o-", color="#FF4444", linewidth=2,
                markersize=5, label="Logical Error Rate")

        # Z_L decoder success rate
        zl_decoder = [r.decoder_success_rate for r in self._threshold_results]
        zl_error = [1.0 - d for d in zl_decoder]
        ax.plot(phys_rates, zl_error, "^-", color="#FFAA22", linewidth=1.5,
                markersize=4, label="Z_L Error Rate")

        # Projection-based logical error rate (1 - mean fidelity)
        proj_rates = [r.projection_logical_rate for r in self._threshold_results]
        ax.plot(phys_rates, proj_rates, "v--", color="#CC44CC", linewidth=1.5,
                markersize=4, alpha=0.8, label="Projection Error")

        # Reference: no correction line (physical = logical)
        ax.plot([0, max(phys_rates)], [0, max(phys_rates)],
                "--", color="#888888", alpha=0.5, label="No Correction")

        ax.set_xlabel("Physical Error Rate", fontsize=9)
        ax.set_ylabel("Logical Error Rate", fontsize=9)
        ax.set_title(f"Threshold Curve: {self._code.name}", fontsize=10)
        ax.legend(fontsize=8, loc="upper left")
        ax.set_xlim(0, max(phys_rates) * 1.05)
        ax.set_ylim(0, 1.05)

        # Fidelity + Z_L fidelity on secondary axis
        ax2 = ax.twinx()
        ax2.plot(phys_rates, fidelities, "s--", color="#44DD44", markersize=4,
                 alpha=0.7, label="Avg Fidelity")
        zl_fid = [r.logical_z_fidelity for r in self._threshold_results]
        ax2.plot(phys_rates, zl_fid, "D--", color="#4488FF", markersize=4,
                 alpha=0.7, label="|<Z_L>|")
        ax2.set_ylabel("Fidelity / |<Z_L>|", fontsize=9)
        ax2.legend(loc="center right", fontsize=8)

        self._apply_fig_theme(ax)
        self._apply_fig_theme(ax2)
        fig.tight_layout()
        self._thresh_canvas.draw_idle()

    # ---- Theme ------------------------------------------------------------

    def _apply_fig_theme(self, ax):
        if self._dark_theme:
            bg, fg = "#2B2B2B", "#CCCCCC"
        else:
            bg, fg = "#FFFFFF", "#333333"
        ax.set_facecolor(bg)
        ax.figure.set_facecolor(bg)
        ax.tick_params(colors=fg, labelsize=8)
        ax.xaxis.label.set_color(fg)
        ax.yaxis.label.set_color(fg)
        ax.title.set_color(fg)
        for spine in ax.spines.values():
            spine.set_color(fg)
