"""Algorithm Comparison panel -- run two circuits side-by-side.

Provides:
- Circuit A (current) / Circuit B (loaded from file or template)
- Tab 1: Histogram Overlay (two distributions overlaid)
- Tab 2: Metrics Table (fidelity, TVD, KL, entropy)
- Tab 3: Resource comparison (gate count, depth bar chart)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView,
    QComboBox, QSpinBox, QFileDialog, QGroupBox, QMessageBox,
    QScrollArea,
)
from PyQt6.QtGui import QColor

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from quantum_sim.engine.circuit import QuantumCircuit, GateInstance
from quantum_sim.engine.comparison import CircuitComparator, ComparisonResult
from quantum_sim.core.serialization import CircuitSerializer


# Built-in template circuits for quick comparison
def _make_bell() -> QuantumCircuit:
    qc = QuantumCircuit(2)
    qc.add_gate(GateInstance("H", [0], [], 0))
    qc.add_gate(GateInstance("CNOT", [0, 1], [], 1))
    return qc


def _make_ghz3() -> QuantumCircuit:
    qc = QuantumCircuit(3)
    qc.add_gate(GateInstance("H", [0], [], 0))
    qc.add_gate(GateInstance("CNOT", [0, 1], [], 1))
    qc.add_gate(GateInstance("CNOT", [1, 2], [], 2))
    return qc


def _make_superposition2() -> QuantumCircuit:
    qc = QuantumCircuit(2)
    qc.add_gate(GateInstance("H", [0], [], 0))
    qc.add_gate(GateInstance("H", [1], [], 0))
    return qc


_TEMPLATES = {
    "Bell State (2q)": _make_bell,
    "GHZ-3 (3q)": _make_ghz3,
    "Full Superposition (2q)": _make_superposition2,
}


class ComparisonPanel(QWidget):
    """Panel for comparing two quantum circuits side-by-side."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._dark_theme: bool = True
        self._circuit_a: QuantumCircuit | None = None
        self._circuit_b: QuantumCircuit | None = None
        self._noise_model = None
        self._comparison_result: ComparisonResult | None = None
        self._comparator = CircuitComparator()
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
        title = QLabel("Algorithm Comparison")
        title.setStyleSheet("font-weight: bold; font-size: 13px;")
        layout.addWidget(title)

        # Control bar
        ctrl_layout = QHBoxLayout()

        ctrl_layout.addWidget(QLabel("Circuit A:"))
        self._label_a = QLabel("(current circuit)")
        self._label_a.setStyleSheet("color: #4488FF; font-weight: bold;")
        ctrl_layout.addWidget(self._label_a)

        ctrl_layout.addStretch()

        ctrl_layout.addWidget(QLabel("Circuit B:"))
        self._template_combo = QComboBox()
        self._template_combo.addItem("-- Select Template --")
        for name in _TEMPLATES:
            self._template_combo.addItem(name)
        self._template_combo.currentIndexChanged.connect(self._on_template_selected)
        ctrl_layout.addWidget(self._template_combo)

        self._btn_load_b = QPushButton("Load File...")
        self._btn_load_b.clicked.connect(self._on_load_circuit_b)
        ctrl_layout.addWidget(self._btn_load_b)

        ctrl_layout.addStretch()

        ctrl_layout.addWidget(QLabel("Shots:"))
        self._shots_spin = QSpinBox()
        self._shots_spin.setRange(100, 100000)
        self._shots_spin.setValue(1024)
        self._shots_spin.setSingleStep(256)
        ctrl_layout.addWidget(self._shots_spin)

        self._btn_compare = QPushButton("Compare")
        self._btn_compare.setStyleSheet("font-weight: bold;")
        self._btn_compare.clicked.connect(self._on_compare)
        ctrl_layout.addWidget(self._btn_compare)

        self._btn_export = QPushButton("Export")
        self._btn_export.clicked.connect(self._on_export)
        ctrl_layout.addWidget(self._btn_export)

        layout.addLayout(ctrl_layout)

        # Status
        self._status_label = QLabel("Select Circuit B and click Compare.")
        layout.addWidget(self._status_label)

        # Tabs
        self._tabs = QTabWidget()
        self._tabs.setDocumentMode(True)

        # Tab 1: Histogram Overlay
        self._hist_tab = QWidget()
        self._setup_histogram_tab()
        self._tabs.addTab(self._hist_tab, "Histogram Overlay")

        # Tab 2: Metrics Table
        self._metrics_tab = QWidget()
        self._setup_metrics_tab()
        self._tabs.addTab(self._metrics_tab, "Metrics")

        # Tab 3: Resource comparison
        self._resource_tab = QWidget()
        self._setup_resource_tab()
        self._tabs.addTab(self._resource_tab, "Resources")

        layout.addWidget(self._tabs, stretch=1)

    def _setup_histogram_tab(self):
        layout = QVBoxLayout(self._hist_tab)
        layout.setContentsMargins(2, 2, 2, 2)
        self._hist_figure = Figure(figsize=(6, 3), dpi=100)
        self._hist_canvas = FigureCanvas(self._hist_figure)
        layout.addWidget(self._hist_canvas, stretch=1)

    def _setup_metrics_tab(self):
        layout = QVBoxLayout(self._metrics_tab)
        layout.setContentsMargins(2, 2, 2, 2)
        self._metrics_table = QTableWidget()
        self._metrics_table.setColumnCount(3)
        self._metrics_table.setHorizontalHeaderLabels(["Metric", "Circuit A", "Circuit B"])
        self._metrics_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self._metrics_table.setAlternatingRowColors(True)
        self._metrics_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self._metrics_table, stretch=1)

    def _setup_resource_tab(self):
        layout = QVBoxLayout(self._resource_tab)
        layout.setContentsMargins(2, 2, 2, 2)
        self._resource_figure = Figure(figsize=(6, 3), dpi=100)
        self._resource_canvas = FigureCanvas(self._resource_figure)
        layout.addWidget(self._resource_canvas, stretch=1)

    # ---- Public API -------------------------------------------------------

    def set_circuit(self, circuit: QuantumCircuit) -> None:
        """Set circuit A (the current circuit)."""
        self._circuit_a = circuit
        self._label_a.setText(f"Current ({circuit.num_qubits}q, {len(circuit.gates)}g)")

    def set_noise_model(self, noise_model) -> None:
        self._noise_model = noise_model

    def update_state(self, state) -> None:
        """Called from main window; updates circuit A reference."""
        pass

    def clear(self) -> None:
        self._comparison_result = None
        self._metrics_table.setRowCount(0)
        self._hist_figure.clear()
        self._hist_canvas.draw_idle()
        self._resource_figure.clear()
        self._resource_canvas.draw_idle()
        self._status_label.setText("Select Circuit B and click Compare.")

    def set_theme(self, dark: bool) -> None:
        self._dark_theme = dark
        if self._comparison_result is not None:
            self._draw_histogram()
            self._draw_resources()

    # ---- Handlers ---------------------------------------------------------

    def _on_template_selected(self, index: int):
        if index <= 0:
            return
        name = self._template_combo.currentText()
        if name in _TEMPLATES:
            self._circuit_b = _TEMPLATES[name]()
            self._status_label.setText(
                f"Circuit B: {name} ({self._circuit_b.num_qubits}q)"
            )

    def _on_load_circuit_b(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Load Circuit B", "", "Quantum Circuit (*.qc *.json);;All Files (*)"
        )
        if not filepath:
            return
        try:
            self._circuit_b = CircuitSerializer.load(filepath)
            name = Path(filepath).stem
            self._status_label.setText(
                f"Circuit B: {name} ({self._circuit_b.num_qubits}q)"
            )
        except Exception as e:
            QMessageBox.warning(self, "Load Error", f"Failed to load: {e}")

    def _on_compare(self):
        if self._circuit_a is None:
            QMessageBox.information(self, "Info", "Run a simulation first to set Circuit A.")
            return
        if self._circuit_b is None:
            QMessageBox.information(self, "Info", "Select a template or load Circuit B first.")
            return

        try:
            self._comparison_result = self._comparator.compare(
                self._circuit_a,
                self._circuit_b,
                shots=self._shots_spin.value(),
                noise_model=self._noise_model,
                seed=42,
            )
            self._update_display()
            self._status_label.setText("Comparison complete.")
        except Exception as e:
            QMessageBox.critical(self, "Comparison Error", f"Failed: {e}")

    def _on_export(self):
        if self._comparison_result is None:
            QMessageBox.information(self, "Info", "Run a comparison first.")
            return
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export Report", "comparison_report.json", "JSON (*.json)"
        )
        if not filepath:
            return
        try:
            CircuitComparator.export_report(self._comparison_result, filepath)
            self._status_label.setText(f"Report exported to {Path(filepath).name}")
        except Exception as e:
            QMessageBox.warning(self, "Export Error", f"Failed: {e}")

    # ---- Display ----------------------------------------------------------

    def _update_display(self):
        self._draw_histogram()
        self._fill_metrics_table()
        self._draw_resources()

    def _draw_histogram(self):
        fig = self._hist_figure
        fig.clear()
        r = self._comparison_result
        if r is None:
            self._hist_canvas.draw_idle()
            return

        ax = fig.add_subplot(111)

        # Collect all bitstrings
        all_keys = sorted(
            set(r.result_a.measurement_counts.keys())
            | set(r.result_b.measurement_counts.keys())
        )
        if not all_keys:
            self._hist_canvas.draw_idle()
            return

        shots_a = r.result_a.num_shots
        shots_b = r.result_b.num_shots
        vals_a = [r.result_a.measurement_counts.get(k, 0) / shots_a for k in all_keys]
        vals_b = [r.result_b.measurement_counts.get(k, 0) / shots_b for k in all_keys]

        x = np.arange(len(all_keys))
        w = 0.35

        ax.bar(x - w / 2, vals_a, w, label="Circuit A", color="#4488FF", alpha=0.75)
        ax.bar(x + w / 2, vals_b, w, label="Circuit B", color="#FF8844", alpha=0.75)

        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"|{k}>" for k in all_keys], rotation=45, ha="right", fontsize=7
        )
        ax.set_ylabel("Probability", fontsize=9)
        ax.set_title(
            f"Distribution Overlay  (TVD={r.distribution_tvd:.4f})", fontsize=10
        )
        ax.legend(fontsize=8)

        self._apply_fig_theme(ax)
        fig.tight_layout()
        self._hist_canvas.draw_idle()

    def _fill_metrics_table(self):
        r = self._comparison_result
        if r is None:
            return

        rows = [
            ("Output Fidelity", f"{r.output_fidelity:.6f}", "--"),
            ("TVD", f"{r.distribution_tvd:.6f}", "--"),
            ("KL(A||B)", f"{r.distribution_kl_ab:.6f}", "--"),
            ("KL(B||A)", f"{r.distribution_kl_ba:.6f}", "--"),
            ("Entropy (bits)", f"{r.entropy_a:.6f}", f"{r.entropy_b:.6f}"),
            ("Purity", f"{r.purity_a:.6f}", f"{r.purity_b:.6f}"),
            ("Gate Count", str(r.metrics_a.gate_count), str(r.metrics_b.gate_count)),
            ("Depth", str(r.metrics_a.depth), str(r.metrics_b.depth)),
            ("1Q Gates", str(r.metrics_a.single_qubit_gates), str(r.metrics_b.single_qubit_gates)),
            ("2Q+ Gates", str(r.metrics_a.multi_qubit_gates), str(r.metrics_b.multi_qubit_gates)),
            ("Param Gates", str(r.metrics_a.parameterized_gates), str(r.metrics_b.parameterized_gates)),
            ("Qubits", str(r.metrics_a.num_qubits), str(r.metrics_b.num_qubits)),
        ]

        self._metrics_table.setRowCount(len(rows))
        for i, (metric, val_a, val_b) in enumerate(rows):
            self._metrics_table.setItem(i, 0, QTableWidgetItem(metric))
            item_a = QTableWidgetItem(val_a)
            item_a.setTextAlignment(int(Qt.AlignmentFlag.AlignCenter))
            self._metrics_table.setItem(i, 1, item_a)
            item_b = QTableWidgetItem(val_b)
            item_b.setTextAlignment(int(Qt.AlignmentFlag.AlignCenter))
            self._metrics_table.setItem(i, 2, item_b)

    def _draw_resources(self):
        fig = self._resource_figure
        fig.clear()
        r = self._comparison_result
        if r is None:
            self._resource_canvas.draw_idle()
            return

        ax = fig.add_subplot(111)

        categories = ["Gate Count", "Depth", "1Q Gates", "2Q+ Gates", "Qubits"]
        vals_a = [
            r.metrics_a.gate_count, r.metrics_a.depth,
            r.metrics_a.single_qubit_gates, r.metrics_a.multi_qubit_gates,
            r.metrics_a.num_qubits,
        ]
        vals_b = [
            r.metrics_b.gate_count, r.metrics_b.depth,
            r.metrics_b.single_qubit_gates, r.metrics_b.multi_qubit_gates,
            r.metrics_b.num_qubits,
        ]

        x = np.arange(len(categories))
        w = 0.35
        ax.bar(x - w / 2, vals_a, w, label="Circuit A", color="#4488FF", alpha=0.8)
        ax.bar(x + w / 2, vals_b, w, label="Circuit B", color="#FF8844", alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontsize=8)
        ax.set_ylabel("Count", fontsize=9)
        ax.set_title("Resource Comparison", fontsize=10)
        ax.legend(fontsize=8)

        self._apply_fig_theme(ax)
        fig.tight_layout()
        self._resource_canvas.draw_idle()

    # ---- Theme helpers ----------------------------------------------------

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
