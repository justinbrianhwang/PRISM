"""Parameterized Circuit Optimizer panel.

Provides:
- Auto-detection of parameterized gates
- Cost function selection (Z-expectation, state fidelity, VQE Hamiltonian)
- Real-time convergence plot, parameter evolution, barren plateau detection
"""

from __future__ import annotations

import numpy as np

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView,
    QComboBox, QSpinBox, QDoubleSpinBox, QGroupBox, QProgressBar,
    QScrollArea,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from quantum_sim.engine.circuit import QuantumCircuit
from quantum_sim.engine.optimizer import (
    ParameterizedCircuitConfig,
    CostFunction,
    CircuitOptimizer,
    OptimizationResult,
    BarrenPlateauAnalysis,
)


class _OptimizerWorker(QThread):
    """Worker thread for running optimization without blocking the GUI."""

    step_completed = pyqtSignal(int, object, float)  # (iteration, values, cost)
    finished = pyqtSignal(object)  # OptimizationResult

    def __init__(self, optimizer: CircuitOptimizer, seed: int | None = None):
        super().__init__()
        self._optimizer = optimizer
        self._seed = seed

    def run(self):
        def _callback(iteration, values, cost):
            self.step_completed.emit(iteration, values.tolist(), cost)

        result = self._optimizer.run(callback=_callback, seed=self._seed)
        self.finished.emit(result)

    def request_stop(self):
        self._optimizer.request_stop()


class OptimizerPanel(QWidget):
    """Panel for parameterized circuit optimization (VQE/QAOA)."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._dark_theme: bool = True
        self._circuit: QuantumCircuit | None = None
        self._config: ParameterizedCircuitConfig | None = None
        self._worker: _OptimizerWorker | None = None
        self._optimizer: CircuitOptimizer | None = None
        self._result: OptimizationResult | None = None
        # For live plots
        self._cost_history: list[float] = []
        self._param_history: list[list[float]] = []
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

        title = QLabel("Circuit Optimizer")
        title.setStyleSheet("font-weight: bold; font-size: 13px;")
        layout.addWidget(title)

        # Controls row 1
        ctrl1 = QHBoxLayout()

        ctrl1.addWidget(QLabel("Cost:"))
        self._cost_combo = QComboBox()
        self._cost_combo.addItems([
            "<Z> on q0",
            "<Z> on q1",
            "Fidelity to |0...0>",
            "Fidelity to |1...1>",
        ])
        ctrl1.addWidget(self._cost_combo)

        ctrl1.addStretch()

        ctrl1.addWidget(QLabel("LR:"))
        self._lr_spin = QDoubleSpinBox()
        self._lr_spin.setRange(0.001, 1.0)
        self._lr_spin.setValue(0.1)
        self._lr_spin.setSingleStep(0.01)
        self._lr_spin.setDecimals(3)
        ctrl1.addWidget(self._lr_spin)

        ctrl1.addWidget(QLabel("Iters:"))
        self._iter_spin = QSpinBox()
        self._iter_spin.setRange(10, 1000)
        self._iter_spin.setValue(100)
        self._iter_spin.setSingleStep(10)
        ctrl1.addWidget(self._iter_spin)

        ctrl1.addWidget(QLabel("Grad:"))
        self._grad_combo = QComboBox()
        self._grad_combo.addItems(["parameter_shift", "finite_difference"])
        ctrl1.addWidget(self._grad_combo)

        layout.addLayout(ctrl1)

        # Controls row 2
        ctrl2 = QHBoxLayout()

        self._btn_detect = QPushButton("Detect Params")
        self._btn_detect.clicked.connect(self._on_detect_params)
        ctrl2.addWidget(self._btn_detect)

        self._param_label = QLabel("Params: --")
        ctrl2.addWidget(self._param_label)

        ctrl2.addStretch()

        self._btn_run = QPushButton("Optimize")
        self._btn_run.setStyleSheet("font-weight: bold;")
        self._btn_run.clicked.connect(self._on_run)
        ctrl2.addWidget(self._btn_run)

        self._btn_stop = QPushButton("Stop")
        self._btn_stop.setEnabled(False)
        self._btn_stop.clicked.connect(self._on_stop)
        ctrl2.addWidget(self._btn_stop)

        self._btn_barren = QPushButton("Check Barren")
        self._btn_barren.setToolTip("Detect barren plateau via gradient variance sampling")
        self._btn_barren.clicked.connect(self._on_check_barren)
        ctrl2.addWidget(self._btn_barren)

        layout.addLayout(ctrl2)

        # Progress and status
        status_layout = QHBoxLayout()
        self._progress = QProgressBar()
        self._progress.setMaximum(100)
        status_layout.addWidget(self._progress, stretch=1)
        self._status_label = QLabel("Detect parameters to begin.")
        status_layout.addWidget(self._status_label)
        layout.addLayout(status_layout)

        # Tabs
        self._tabs = QTabWidget()
        self._tabs.setDocumentMode(True)

        # Tab 1: Convergence
        self._conv_tab = QWidget()
        self._setup_convergence_tab()
        self._tabs.addTab(self._conv_tab, "Convergence")

        # Tab 2: Parameters
        self._params_tab = QWidget()
        self._setup_params_tab()
        self._tabs.addTab(self._params_tab, "Parameters")

        # Tab 3: Barren Plateau
        self._barren_tab = QWidget()
        self._setup_barren_tab()
        self._tabs.addTab(self._barren_tab, "Barren Plateau")

        layout.addWidget(self._tabs, stretch=1)

    def _setup_convergence_tab(self):
        layout = QVBoxLayout(self._conv_tab)
        layout.setContentsMargins(2, 2, 2, 2)
        self._conv_figure = Figure(figsize=(6, 3), dpi=100)
        self._conv_canvas = FigureCanvas(self._conv_figure)
        layout.addWidget(self._conv_canvas, stretch=1)

    def _setup_params_tab(self):
        layout = QVBoxLayout(self._params_tab)
        layout.setContentsMargins(2, 2, 2, 2)
        self._param_figure = Figure(figsize=(6, 3), dpi=100)
        self._param_canvas = FigureCanvas(self._param_figure)
        layout.addWidget(self._param_canvas, stretch=1)

        # Result table
        self._result_table = QTableWidget()
        self._result_table.setColumnCount(3)
        self._result_table.setHorizontalHeaderLabels(["Parameter", "Initial", "Optimal"])
        self._result_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self._result_table.setMaximumHeight(150)
        self._result_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self._result_table)

    def _setup_barren_tab(self):
        layout = QVBoxLayout(self._barren_tab)
        layout.setContentsMargins(2, 2, 2, 2)
        self._barren_info = QLabel("Click 'Check Barren' to analyze gradient variance.")
        layout.addWidget(self._barren_info)
        self._barren_figure = Figure(figsize=(6, 3), dpi=100)
        self._barren_canvas = FigureCanvas(self._barren_figure)
        layout.addWidget(self._barren_canvas, stretch=1)

    # ---- Public API -------------------------------------------------------

    def set_circuit(self, circuit: QuantumCircuit) -> None:
        self._circuit = circuit

    def set_noise_model(self, noise_model) -> None:
        pass  # Optimizer uses noiseless simulation for gradients

    def update_state(self, state) -> None:
        pass

    def clear(self) -> None:
        self._config = None
        self._result = None
        self._cost_history.clear()
        self._param_history.clear()
        self._param_label.setText("Params: --")
        self._status_label.setText("Detect parameters to begin.")
        self._progress.setValue(0)
        self._conv_figure.clear()
        self._conv_canvas.draw_idle()
        self._param_figure.clear()
        self._param_canvas.draw_idle()
        self._barren_figure.clear()
        self._barren_canvas.draw_idle()
        self._result_table.setRowCount(0)

    def set_theme(self, dark: bool) -> None:
        self._dark_theme = dark
        self._redraw_all()

    # ---- Handlers ---------------------------------------------------------

    def _on_detect_params(self):
        if self._circuit is None:
            self._status_label.setText("No circuit loaded.")
            return

        self._config = ParameterizedCircuitConfig.auto_detect(self._circuit)
        n = self._config.num_params
        self._param_label.setText(f"Params: {n}")

        if n == 0:
            self._status_label.setText("No parameterized gates found. Add Rx/Ry/Rz/U3 gates.")
        else:
            names = [b.name for b in self._config.bindings]
            self._status_label.setText(f"Found {n} params: {', '.join(names[:5])}")

    def _build_cost_fn(self):
        """Build cost function from current combo selection."""
        idx = self._cost_combo.currentIndex()
        if idx == 0:
            return CostFunction.z_expectation(0)
        elif idx == 1:
            return CostFunction.z_expectation(1)
        elif idx == 2:
            n = self._circuit.num_qubits
            target = np.zeros(2 ** n, dtype=np.complex128)
            target[0] = 1.0
            return CostFunction.state_fidelity(target)
        elif idx == 3:
            n = self._circuit.num_qubits
            target = np.zeros(2 ** n, dtype=np.complex128)
            target[-1] = 1.0
            return CostFunction.state_fidelity(target)
        return CostFunction.z_expectation(0)

    def _on_run(self):
        if self._config is None or self._config.num_params == 0:
            self._status_label.setText("Detect parameters first.")
            return

        cost_fn = self._build_cost_fn()

        self._optimizer = CircuitOptimizer(
            config=self._config,
            cost_fn=cost_fn,
            learning_rate=self._lr_spin.value(),
            max_iterations=self._iter_spin.value(),
            gradient_method=self._grad_combo.currentText(),
        )

        self._cost_history.clear()
        self._param_history.clear()
        self._progress.setMaximum(self._iter_spin.value())
        self._progress.setValue(0)

        self._btn_run.setEnabled(False)
        self._btn_stop.setEnabled(True)

        self._worker = _OptimizerWorker(self._optimizer, seed=42)
        self._worker.step_completed.connect(self._on_step_completed)
        self._worker.finished.connect(self._on_optimization_finished)
        self._worker.start()

    def _on_stop(self):
        if self._worker is not None:
            self._worker.request_stop()

    def _on_step_completed(self, iteration: int, values: list, cost: float):
        self._cost_history.append(cost)
        self._param_history.append(values)
        self._progress.setValue(iteration + 1)
        self._status_label.setText(f"Iter {iteration + 1}: cost = {cost:.6f}")

        # Update convergence plot every 5 iterations
        if (iteration + 1) % 5 == 0 or iteration < 5:
            self._draw_convergence()

    def _on_optimization_finished(self, result: OptimizationResult):
        self._result = result
        self._btn_run.setEnabled(True)
        self._btn_stop.setEnabled(False)

        status = "converged" if result.converged else "max iterations"
        self._status_label.setText(
            f"Done ({status}): cost = {result.optimal_cost:.6f}, "
            f"iters = {result.iterations}"
        )

        self._draw_convergence()
        self._draw_parameters()
        self._fill_result_table()

    def _on_check_barren(self):
        if self._config is None or self._config.num_params == 0:
            self._status_label.setText("Detect parameters first.")
            return

        cost_fn = self._build_cost_fn()
        optimizer = CircuitOptimizer(
            config=self._config,
            cost_fn=cost_fn,
        )

        self._status_label.setText("Sampling gradient variance (layered)...")
        bp_analysis = optimizer.detect_barren_plateau_layered(n_samples=30, seed=42)
        self._draw_barren_plateau_layered(bp_analysis)

        if bp_analysis.overall_is_barren:
            self._barren_info.setText(
                f"WARNING: Barren plateau detected! "
                f"Mean gradient variance = {bp_analysis.overall_mean_variance:.2e}"
            )
            self._barren_info.setStyleSheet("color: #FF4444; font-weight: bold;")
        else:
            self._barren_info.setText(
                f"No barren plateau detected. "
                f"Mean gradient variance = {bp_analysis.overall_mean_variance:.4f}"
            )
            self._barren_info.setStyleSheet("color: #44DD44; font-weight: bold;")

        self._status_label.setText("Barren plateau check complete.")

    # ---- Drawing ----------------------------------------------------------

    def _draw_convergence(self):
        fig = self._conv_figure
        fig.clear()
        ax = fig.add_subplot(111)

        if self._cost_history:
            ax.plot(
                range(1, len(self._cost_history) + 1),
                self._cost_history,
                "-o", color="#4488FF", markersize=2, linewidth=1.5,
            )
            ax.set_xlabel("Iteration", fontsize=9)
            ax.set_ylabel("Cost", fontsize=9)
            ax.set_title("Optimization Convergence", fontsize=10)

            # Mark minimum
            min_idx = np.argmin(self._cost_history)
            ax.axhline(
                y=self._cost_history[min_idx],
                color="#44DD44", linestyle="--", alpha=0.6,
                label=f"Min: {self._cost_history[min_idx]:.4f}"
            )
            ax.legend(fontsize=8)

        self._apply_fig_theme(ax)
        fig.tight_layout()
        self._conv_canvas.draw_idle()

    def _draw_parameters(self):
        fig = self._param_figure
        fig.clear()

        if not self._param_history or self._config is None:
            self._param_canvas.draw_idle()
            return

        ax = fig.add_subplot(111)
        arr = np.array(self._param_history)
        colors = ["#FF4444", "#4488FF", "#44DD44", "#FFAA22", "#DD44DD",
                  "#22DDDD", "#FFDD44", "#FF66AA"]

        for i in range(arr.shape[1]):
            name = self._config.bindings[i].name if i < len(self._config.bindings) else f"p{i}"
            color = colors[i % len(colors)]
            ax.plot(range(1, len(arr) + 1), arr[:, i],
                    "-", color=color, linewidth=1.5, label=name)

        ax.set_xlabel("Iteration", fontsize=9)
        ax.set_ylabel("Parameter Value", fontsize=9)
        ax.set_title("Parameter Evolution", fontsize=10)
        ax.legend(fontsize=7, ncol=2)

        self._apply_fig_theme(ax)
        fig.tight_layout()
        self._param_canvas.draw_idle()

    def _fill_result_table(self):
        if self._result is None or self._config is None:
            return

        initial = self._config.get_values()
        optimal = self._result.optimal_values
        n = self._config.num_params

        self._result_table.setRowCount(n)
        for i in range(n):
            name = self._config.bindings[i].name
            self._result_table.setItem(i, 0, QTableWidgetItem(name))
            item_init = QTableWidgetItem(f"{initial[i]:.4f}")
            item_init.setTextAlignment(int(Qt.AlignmentFlag.AlignCenter))
            self._result_table.setItem(i, 1, item_init)
            item_opt = QTableWidgetItem(f"{optimal[i]:.4f}")
            item_opt.setTextAlignment(int(Qt.AlignmentFlag.AlignCenter))
            self._result_table.setItem(i, 2, item_opt)

    def _draw_barren_plateau_layered(self, analysis: BarrenPlateauAnalysis):
        fig = self._barren_figure
        fig.clear()

        if not analysis.per_layer_variance:
            self._barren_canvas.draw_idle()
            return

        # Two subplots: heatmap (left) and depth scaling (right)
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        # --- Subplot 1: Variance heatmap (rows=layers, cols=params in layer) ---
        # Build a padded matrix for the heatmap
        n_layers = len(analysis.per_layer_variance)
        max_params = max(len(lv) for lv in analysis.per_layer_variance)

        heatmap_data = np.full((n_layers, max(max_params, 1)), np.nan)
        for li, layer_vars in enumerate(analysis.per_layer_variance):
            for pi, v in enumerate(layer_vars):
                heatmap_data[li, pi] = v

        # Use log scale via safe log
        with np.errstate(divide="ignore"):
            log_data = np.log10(np.where(heatmap_data > 0, heatmap_data, np.nan))

        im = ax1.imshow(
            log_data, aspect="auto", cmap="RdYlGn",
            interpolation="nearest", origin="upper",
        )

        # Annotate cells
        for li in range(n_layers):
            for pi in range(max_params):
                if not np.isnan(heatmap_data[li, pi]):
                    val = heatmap_data[li, pi]
                    ax1.text(pi, li, f"{val:.1e}", ha="center", va="center",
                             fontsize=6, color="black")

        ax1.set_xlabel("Param in Layer", fontsize=8)
        ax1.set_ylabel("Layer (column)", fontsize=8)
        ax1.set_title("Gradient Variance\n(log10 scale)", fontsize=9)

        layer_labels = [f"L{d}" for d, _ in analysis.depth_scaling]
        ax1.set_yticks(range(n_layers))
        ax1.set_yticklabels(layer_labels, fontsize=7)
        ax1.set_xticks(range(max_params))
        ax1.set_xticklabels([f"p{i}" for i in range(max_params)], fontsize=7)

        fig.colorbar(im, ax=ax1, shrink=0.8, label="log10(Var)")

        # --- Subplot 2: Depth scaling plot ---
        depths = [d for d, _ in analysis.depth_scaling]
        mean_vars = [v for _, v in analysis.depth_scaling]

        ax2.semilogy(
            range(len(depths)), mean_vars,
            "o-", color="#4488FF", markersize=5, linewidth=1.5,
        )
        ax2.axhline(
            y=analysis.threshold, color="#FFAA22",
            linestyle="--", linewidth=1.5, label="Barren threshold",
        )

        ax2.set_xticks(range(len(depths)))
        ax2.set_xticklabels([f"L{d}" for d in depths], fontsize=7)
        ax2.set_xlabel("Layer (column)", fontsize=8)
        ax2.set_ylabel("Mean Gradient Variance", fontsize=8)
        ax2.set_title("Variance vs Depth", fontsize=9)
        ax2.legend(fontsize=7)

        self._apply_fig_theme(ax1)
        self._apply_fig_theme(ax2)
        fig.tight_layout()
        self._barren_canvas.draw_idle()

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

    def _redraw_all(self):
        self._draw_convergence()
        if self._result:
            self._draw_parameters()
