"""Main application window for the Quantum Circuit Simulator.

Provides the top-level QMainWindow that coordinates all GUI components:
menu bar, toolbar, circuit editor, dock panels, status bar, and
connections to the engine layer via controllers.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QAction, QKeySequence, QIcon
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QMenuBar, QMenu, QToolBar, QStatusBar, QDockWidget,
    QTabWidget, QSpinBox, QLabel, QFileDialog, QMessageBox,
    QApplication, QInputDialog,
)

# --- Engine / Core imports (always available) ---
from quantum_sim.engine.circuit import QuantumCircuit, GateInstance
from quantum_sim.engine.gate_registry import GateRegistry
from quantum_sim.engine.simulator import Simulator, SimulationResult
from quantum_sim.engine.algorithms import AlgorithmTemplate
from quantum_sim.engine.noise import NoiseModel
from quantum_sim.engine.reference import ReferenceManager
from quantum_sim.core.serialization import CircuitSerializer
from quantum_sim.core.config import AppConfig

# --- Theme manager ---
from quantum_sim.gui.themes.theme_manager import ThemeManager

# --- Panels (always available since we just created them) ---
from quantum_sim.gui.panels.gate_palette import GatePalette
from quantum_sim.gui.panels.properties_panel import PropertiesPanel

# --- Optional imports for components that may not yet exist ---
# Circuit editor view and scene
try:
    from quantum_sim.gui.circuit_editor.view import CircuitView
    from quantum_sim.gui.circuit_editor.scene import CircuitScene
except ImportError:
    CircuitView = None  # type: ignore[assignment, misc]
    CircuitScene = None  # type: ignore[assignment, misc]

# Circuit controller
try:
    from quantum_sim.controller.circuit_controller import CircuitController
except ImportError:
    CircuitController = None  # type: ignore[assignment, misc]

# Simulation controller
try:
    from quantum_sim.controller.simulation_controller import SimulationController
except ImportError:
    SimulationController = None  # type: ignore[assignment, misc]

# Visualization panels
try:
    from quantum_sim.gui.panels.state_vector_panel import StateVectorPanel
except ImportError:
    StateVectorPanel = None  # type: ignore[assignment, misc]

try:
    from quantum_sim.gui.panels.bloch_sphere import BlochSphereWidget
except ImportError:
    BlochSphereWidget = None  # type: ignore[assignment, misc]

try:
    from quantum_sim.gui.panels.histogram_panel import HistogramPanel
except ImportError:
    HistogramPanel = None  # type: ignore[assignment, misc]

try:
    from quantum_sim.gui.panels.density_matrix_panel import DensityMatrixPanel
except ImportError:
    DensityMatrixPanel = None  # type: ignore[assignment, misc]

# New research panels
try:
    from quantum_sim.gui.panels.entanglement_panel import EntanglementPanel
except ImportError:
    EntanglementPanel = None  # type: ignore[assignment, misc]

try:
    from quantum_sim.gui.panels.entropy_panel import EntropyPanel
except ImportError:
    EntropyPanel = None  # type: ignore[assignment, misc]

try:
    from quantum_sim.gui.panels.fidelity_panel import FidelityPanel
except ImportError:
    FidelityPanel = None  # type: ignore[assignment, misc]

try:
    from quantum_sim.gui.panels.analysis_panel import AnalysisPanel
except ImportError:
    AnalysisPanel = None  # type: ignore[assignment, misc]

try:
    from quantum_sim.gui.panels.debugger_panel import DebuggerPanel
except ImportError:
    DebuggerPanel = None  # type: ignore[assignment, misc]

try:
    from quantum_sim.gui.panels.comparison_panel import ComparisonPanel
except ImportError:
    ComparisonPanel = None  # type: ignore[assignment, misc]

try:
    from quantum_sim.gui.panels.optimizer_panel import OptimizerPanel
except ImportError:
    OptimizerPanel = None  # type: ignore[assignment, misc]

try:
    from quantum_sim.gui.panels.qec_panel import QECPanel
except ImportError:
    QECPanel = None  # type: ignore[assignment, misc]

try:
    from quantum_sim.gui.panels.resource_monitor import ResourceMonitorPanel
except ImportError:
    ResourceMonitorPanel = None  # type: ignore[assignment, misc]

# Undo commands
try:
    from quantum_sim.gui.commands.circuit_commands import (
        AddGateCommand, RemoveGateCommand, MoveGateCommand,
        ChangeGateParamsCommand, ClearCircuitCommand,
        SetQubitCountCommand,
    )
    from PyQt6.QtGui import QUndoStack
    HAS_UNDO = True
except ImportError:
    from PyQt6.QtGui import QUndoStack
    HAS_UNDO = False

# Noise configuration dialog
try:
    from quantum_sim.gui.dialogs.noise_config_dialog import NoiseConfigDialog
except ImportError:
    NoiseConfigDialog = None  # type: ignore[assignment, misc]

# Experiment / Seed / Bridge
try:
    from quantum_sim.core.experiment import ExperimentConfig, SeedManager
except ImportError:
    ExperimentConfig = None  # type: ignore[assignment, misc]
    SeedManager = None  # type: ignore[assignment, misc]

try:
    from quantum_sim.engine.benchmarks import BenchmarkSuite
except ImportError:
    BenchmarkSuite = None  # type: ignore[assignment, misc]

try:
    from quantum_sim.bridge.server import BridgeServer, BridgeCommandHandler
except ImportError:
    BridgeServer = None  # type: ignore[assignment, misc]
    BridgeCommandHandler = None  # type: ignore[assignment, misc]


logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """The main application window for Quantum Circuit Simulator.

    Coordinates:
    - Menu bar with File, Edit, Circuit, Simulation, View, Help menus
    - Toolbar with common actions and a qubit count spinner
    - Central circuit editor (CircuitView)
    - Left dock: Gate Palette
    - Right dock: Properties Panel
    - Bottom dock: Tabbed visualization panels (state vector, Bloch
      sphere, histogram, density matrix)
    - Status bar showing qubit count, gate count, simulation status
    - QUndoStack for undo/redo
    - File save/load via CircuitSerializer
    """

    APP_NAME = "Quantum Circuit Simulator"
    FILE_FILTER = "Quantum Circuit Files (*.qsim);;JSON Files (*.json);;All Files (*)"

    def __init__(self, app: QApplication, parent: QWidget | None = None):
        super().__init__(parent)
        self._app = app
        self._config = AppConfig.load()

        # Core state
        self._circuit = QuantumCircuit(num_qubits=self._config.default_qubits)
        self._simulator = Simulator()
        self._noise_model: NoiseModel | None = None
        self._current_file: Path | None = None
        self._simulation_result: SimulationResult | None = None
        self._step_generator = None
        self._step_timer = QTimer(self)
        self._step_timer.setInterval(self._config.step_delay_ms)
        self._step_timer.timeout.connect(self._on_step_tick)

        # Seed manager for reproducibility
        self._seed_manager = SeedManager() if SeedManager is not None else None

        # Bridge server
        self._bridge_handler = None
        self._bridge_server = None
        if BridgeCommandHandler is not None and BridgeServer is not None:
            self._bridge_handler = BridgeCommandHandler()
            self._bridge_server = BridgeServer(self._bridge_handler)

        # Ideal state cache for fidelity comparisons
        self._reference_manager = ReferenceManager()

        # Theme manager
        self._theme_manager = ThemeManager(app)

        # Undo stack
        self._undo_stack = QUndoStack(self)
        self._undo_stack.cleanChanged.connect(self._on_undo_clean_changed)

        # Controllers (optional)
        self._circuit_controller = None
        self._simulation_controller = None

        if CircuitController is not None:
            try:
                self._circuit_controller = CircuitController(
                    self._circuit, parent=self
                )
            except Exception:
                logger.warning(
                    "CircuitController could not be initialized.", exc_info=True
                )

        if SimulationController is not None:
            try:
                self._simulation_controller = SimulationController(parent=self)
            except Exception:
                logger.warning(
                    "SimulationController could not be initialized.",
                    exc_info=True,
                )

        # Build UI
        self._setup_window()
        self._create_actions()
        self._create_menus()
        self._create_toolbar()
        self._create_central_widget()
        self._create_dock_panels()
        self._create_status_bar()
        self._connect_signals()

        # Apply theme
        self._theme_manager.apply_theme(self._config.theme)

        # Initial status bar update
        self._update_status_bar()
        self._update_title()

    # ------------------------------------------------------------------
    # Window setup
    # ------------------------------------------------------------------

    def _setup_window(self):
        """Configure basic window properties."""
        self.setWindowTitle(self.APP_NAME)
        self.resize(self._config.window_width, self._config.window_height)
        self.setMinimumSize(800, 600)
        self.setDockOptions(
            QMainWindow.DockOption.AnimatedDocks
            | QMainWindow.DockOption.AllowNestedDocks
            | QMainWindow.DockOption.AllowTabbedDocks
        )

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _create_actions(self):
        """Create all QActions used in menus and toolbars."""

        # -- File actions --
        self._action_new = QAction("&New", self)
        self._action_new.setShortcut(QKeySequence.StandardKey.New)
        self._action_new.setStatusTip("Create a new circuit")
        self._action_new.triggered.connect(self._on_new)

        self._action_open = QAction("&Open...", self)
        self._action_open.setShortcut(QKeySequence.StandardKey.Open)
        self._action_open.setStatusTip("Open a circuit file")
        self._action_open.triggered.connect(self._on_open)

        self._action_save = QAction("&Save", self)
        self._action_save.setShortcut(QKeySequence.StandardKey.Save)
        self._action_save.setStatusTip("Save the current circuit")
        self._action_save.triggered.connect(self._on_save)

        self._action_save_as = QAction("Save &As...", self)
        self._action_save_as.setShortcut(QKeySequence("Ctrl+Shift+S"))
        self._action_save_as.setStatusTip("Save circuit to a new file")
        self._action_save_as.triggered.connect(self._on_save_as)

        self._action_export_image = QAction("&Export Image...", self)
        self._action_export_image.setShortcut(QKeySequence("Ctrl+E"))
        self._action_export_image.setStatusTip("Export circuit as image")
        self._action_export_image.triggered.connect(self._on_export_image)

        self._action_exit = QAction("E&xit", self)
        self._action_exit.setShortcut(QKeySequence("Alt+F4"))
        self._action_exit.setStatusTip("Exit the application")
        self._action_exit.triggered.connect(self.close)

        # -- Edit actions --
        self._action_undo = self._undo_stack.createUndoAction(self, "&Undo")
        self._action_undo.setShortcut(QKeySequence.StandardKey.Undo)

        self._action_redo = self._undo_stack.createRedoAction(self, "&Redo")
        self._action_redo.setShortcut(QKeySequence.StandardKey.Redo)

        self._action_delete = QAction("&Delete", self)
        self._action_delete.setShortcut(QKeySequence.StandardKey.Delete)
        self._action_delete.setStatusTip("Delete selected gate")
        self._action_delete.triggered.connect(self._on_delete)

        self._action_select_all = QAction("Select &All", self)
        self._action_select_all.setShortcut(QKeySequence.StandardKey.SelectAll)
        self._action_select_all.triggered.connect(self._on_select_all)

        # -- Circuit actions --
        self._action_set_qubits = QAction("Set &Qubit Count...", self)
        self._action_set_qubits.setStatusTip("Change the number of qubits")
        self._action_set_qubits.triggered.connect(self._on_set_qubit_count)

        self._action_clear = QAction("&Clear Circuit", self)
        self._action_clear.setShortcut(QKeySequence("Ctrl+Shift+Delete"))
        self._action_clear.setStatusTip("Remove all gates from the circuit")
        self._action_clear.triggered.connect(self._on_clear_circuit)

        # -- Simulation actions --
        self._action_run = QAction("&Run", self)
        self._action_run.setShortcut(QKeySequence("F5"))
        self._action_run.setStatusTip("Run the full simulation")
        self._action_run.triggered.connect(self._on_run_simulation)

        self._action_step = QAction("&Step-by-Step", self)
        self._action_step.setShortcut(QKeySequence("F6"))
        self._action_step.setStatusTip("Step through the simulation")
        self._action_step.triggered.connect(self._on_step_simulation)

        self._action_reset = QAction("R&eset", self)
        self._action_reset.setShortcut(QKeySequence("F7"))
        self._action_reset.setStatusTip("Reset simulation state")
        self._action_reset.triggered.connect(self._on_reset_simulation)

        self._action_configure_noise = QAction("Configure &Noise...", self)
        self._action_configure_noise.setStatusTip("Configure noise model")
        self._action_configure_noise.triggered.connect(
            self._on_configure_noise
        )

        self._action_set_seed = QAction("Set &Seed...", self)
        self._action_set_seed.setStatusTip("Set random seed for reproducibility")
        self._action_set_seed.triggered.connect(self._on_set_seed)

        self._action_export_experiment = QAction("&Export Experiment...", self)
        self._action_export_experiment.setStatusTip("Export experiment configuration")
        self._action_export_experiment.triggered.connect(self._on_export_experiment)

        self._action_import_experiment = QAction("&Import Experiment...", self)
        self._action_import_experiment.setStatusTip("Import experiment configuration")
        self._action_import_experiment.triggered.connect(self._on_import_experiment)

        self._action_run_benchmarks = QAction("Run &Benchmarks", self)
        self._action_run_benchmarks.setStatusTip("Run benchmark suite")
        self._action_run_benchmarks.triggered.connect(self._on_run_benchmarks)

        self._action_toggle_bridge = QAction("Start &Bridge Server", self)
        self._action_toggle_bridge.setStatusTip("Start/stop the Live Bridge API server")
        self._action_toggle_bridge.triggered.connect(self._on_toggle_bridge)

        # -- View actions --
        self._action_dark_theme = QAction("&Dark Theme", self)
        self._action_dark_theme.setCheckable(True)
        self._action_dark_theme.setChecked(True)
        self._action_dark_theme.triggered.connect(
            lambda: self._on_set_theme("dark")
        )

        self._action_light_theme = QAction("&Light Theme", self)
        self._action_light_theme.setCheckable(True)
        self._action_light_theme.triggered.connect(
            lambda: self._on_set_theme("light")
        )

        # -- Help actions --
        self._action_about = QAction("&About", self)
        self._action_about.setStatusTip("About Quantum Circuit Simulator")
        self._action_about.triggered.connect(self._on_about)

    # ------------------------------------------------------------------
    # Menus
    # ------------------------------------------------------------------

    def _create_menus(self):
        """Build the menu bar."""
        menu_bar = self.menuBar()

        # --- File menu ---
        file_menu = menu_bar.addMenu("&File")
        file_menu.addAction(self._action_new)
        file_menu.addAction(self._action_open)
        file_menu.addAction(self._action_save)
        file_menu.addAction(self._action_save_as)
        file_menu.addAction(self._action_export_image)
        file_menu.addSeparator()
        file_menu.addAction(self._action_exit)

        # --- Edit menu ---
        edit_menu = menu_bar.addMenu("&Edit")
        edit_menu.addAction(self._action_undo)
        edit_menu.addAction(self._action_redo)
        edit_menu.addSeparator()
        edit_menu.addAction(self._action_delete)
        edit_menu.addAction(self._action_select_all)

        # --- Circuit menu ---
        circuit_menu = menu_bar.addMenu("&Circuit")
        circuit_menu.addAction(self._action_set_qubits)
        circuit_menu.addAction(self._action_clear)
        circuit_menu.addSeparator()

        # Load Template submenu
        self._template_menu = circuit_menu.addMenu("Load &Template")
        self._populate_template_menu()

        # --- Simulation menu ---
        sim_menu = menu_bar.addMenu("&Simulation")
        sim_menu.addAction(self._action_run)
        sim_menu.addAction(self._action_step)
        sim_menu.addAction(self._action_reset)
        sim_menu.addSeparator()
        sim_menu.addAction(self._action_configure_noise)
        sim_menu.addAction(self._action_set_seed)
        sim_menu.addSeparator()
        sim_menu.addAction(self._action_export_experiment)
        sim_menu.addAction(self._action_import_experiment)
        sim_menu.addAction(self._action_run_benchmarks)
        sim_menu.addSeparator()
        sim_menu.addAction(self._action_toggle_bridge)

        # --- View menu ---
        self._view_menu = menu_bar.addMenu("&View")
        # Panel toggle actions will be added after dock creation
        self._view_menu.addSeparator()
        self._view_menu.addAction(self._action_dark_theme)
        self._view_menu.addAction(self._action_light_theme)

        # --- Help menu ---
        help_menu = menu_bar.addMenu("&Help")
        help_menu.addAction(self._action_about)

    def _populate_template_menu(self):
        """Add algorithm template actions to the template submenu."""
        self._template_menu.clear()
        templates = AlgorithmTemplate.list_templates()

        for tmpl in templates:
            action = QAction(tmpl["display"], self)
            action.setStatusTip(tmpl["description"])
            name = tmpl["name"]
            action.triggered.connect(
                lambda checked, n=name: self._on_load_template(n)
            )
            self._template_menu.addAction(action)

    # ------------------------------------------------------------------
    # Toolbar
    # ------------------------------------------------------------------

    def _create_toolbar(self):
        """Build the main toolbar."""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setObjectName("MainToolBar")
        toolbar.setMovable(False)
        # Keep the default icon size from the style
        self.addToolBar(toolbar)

        # File actions
        toolbar.addAction(self._action_new)
        toolbar.addAction(self._action_open)
        toolbar.addAction(self._action_save)
        toolbar.addSeparator()

        # Edit actions
        toolbar.addAction(self._action_undo)
        toolbar.addAction(self._action_redo)
        toolbar.addSeparator()

        # Simulation actions
        toolbar.addAction(self._action_run)
        toolbar.addAction(self._action_step)
        toolbar.addAction(self._action_reset)
        toolbar.addSeparator()

        # Qubit count spinner
        qubit_label = QLabel(" Qubits: ")
        toolbar.addWidget(qubit_label)

        self._qubit_spinbox = QSpinBox()
        self._qubit_spinbox.setMinimum(1)
        self._qubit_spinbox.setMaximum(16)
        self._qubit_spinbox.setValue(self._circuit.num_qubits)
        self._qubit_spinbox.setToolTip("Number of qubits (1-16)")
        self._qubit_spinbox.setFixedWidth(60)
        self._qubit_spinbox.valueChanged.connect(self._on_qubit_spinbox_changed)
        toolbar.addWidget(self._qubit_spinbox)

    # ------------------------------------------------------------------
    # Central widget
    # ------------------------------------------------------------------

    def _create_central_widget(self):
        """Create the central circuit editor widget."""
        if CircuitView is not None and CircuitScene is not None:
            try:
                self._circuit_scene = CircuitScene(self._circuit, parent=self)
                self._circuit_view = CircuitView(self._circuit_scene, parent=self)
                self.setCentralWidget(self._circuit_view)
                return
            except Exception:
                logger.warning(
                    "CircuitView could not be initialized, using placeholder.",
                    exc_info=True,
                )

        # Fallback: placeholder widget
        self._circuit_view = None
        placeholder = QWidget()
        layout = QVBoxLayout(placeholder)
        label = QLabel("Circuit Editor\n(CircuitView not yet available)")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet(
            "font-size: 18px; color: #6c7086; font-style: italic;"
        )
        layout.addWidget(label)
        self.setCentralWidget(placeholder)

    # ------------------------------------------------------------------
    # Dock panels
    # ------------------------------------------------------------------

    def _create_dock_panels(self):
        """Create and position all dock widgets."""

        # --- Gate Palette (left) ---
        self._gate_palette = GatePalette(self)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self._gate_palette)

        # --- Properties Panel (right) ---
        self._properties_panel = PropertiesPanel(self)
        self._properties_panel.set_num_qubits(self._circuit.num_qubits)
        self.addDockWidget(
            Qt.DockWidgetArea.RightDockWidgetArea, self._properties_panel
        )

        # --- Bottom panels (tabbed visualization) ---
        self._bottom_dock = QDockWidget("Visualization", self)
        self._bottom_dock.setObjectName("VisualizationDock")
        self._bottom_dock.setAllowedAreas(
            Qt.DockWidgetArea.BottomDockWidgetArea
            | Qt.DockWidgetArea.TopDockWidgetArea
        )

        self._viz_tabs = QTabWidget()

        # State Vector panel
        if StateVectorPanel is not None:
            try:
                self._state_vector_panel = StateVectorPanel()
                self._viz_tabs.addTab(self._state_vector_panel, "State Vector")
            except Exception:
                self._state_vector_panel = None
                self._viz_tabs.addTab(
                    self._create_placeholder_tab("State Vector"), "State Vector"
                )
        else:
            self._state_vector_panel = None
            self._viz_tabs.addTab(
                self._create_placeholder_tab("State Vector"), "State Vector"
            )

        # Bloch Sphere panel
        if BlochSphereWidget is not None:
            try:
                self._bloch_panel = BlochSphereWidget()
                self._viz_tabs.addTab(self._bloch_panel, "Bloch Sphere")
            except Exception:
                self._bloch_panel = None
                self._viz_tabs.addTab(
                    self._create_placeholder_tab("Bloch Sphere"), "Bloch Sphere"
                )
        else:
            self._bloch_panel = None
            self._viz_tabs.addTab(
                self._create_placeholder_tab("Bloch Sphere"), "Bloch Sphere"
            )

        # Histogram panel
        if HistogramPanel is not None:
            try:
                self._histogram_panel = HistogramPanel()
                self._viz_tabs.addTab(self._histogram_panel, "Histogram")
            except Exception:
                self._histogram_panel = None
                self._viz_tabs.addTab(
                    self._create_placeholder_tab("Histogram"), "Histogram"
                )
        else:
            self._histogram_panel = None
            self._viz_tabs.addTab(
                self._create_placeholder_tab("Histogram"), "Histogram"
            )

        # Density Matrix panel
        if DensityMatrixPanel is not None:
            try:
                self._density_matrix_panel = DensityMatrixPanel()
                self._viz_tabs.addTab(
                    self._density_matrix_panel, "Density Matrix"
                )
            except Exception:
                self._density_matrix_panel = None
                self._viz_tabs.addTab(
                    self._create_placeholder_tab("Density Matrix"),
                    "Density Matrix",
                )
        else:
            self._density_matrix_panel = None
            self._viz_tabs.addTab(
                self._create_placeholder_tab("Density Matrix"), "Density Matrix"
            )

        # Entanglement panel
        if EntanglementPanel is not None:
            try:
                self._entanglement_panel = EntanglementPanel()
                self._viz_tabs.addTab(self._entanglement_panel, "Entanglement")
            except Exception:
                self._entanglement_panel = None
                self._viz_tabs.addTab(
                    self._create_placeholder_tab("Entanglement"), "Entanglement"
                )
        else:
            self._entanglement_panel = None
            self._viz_tabs.addTab(
                self._create_placeholder_tab("Entanglement"), "Entanglement"
            )

        # Entropy panel
        if EntropyPanel is not None:
            try:
                self._entropy_panel = EntropyPanel()
                self._viz_tabs.addTab(self._entropy_panel, "Entropy")
            except Exception:
                self._entropy_panel = None
                self._viz_tabs.addTab(
                    self._create_placeholder_tab("Entropy"), "Entropy"
                )
        else:
            self._entropy_panel = None
            self._viz_tabs.addTab(
                self._create_placeholder_tab("Entropy"), "Entropy"
            )

        # Fidelity panel
        if FidelityPanel is not None:
            try:
                self._fidelity_panel = FidelityPanel()
                self._viz_tabs.addTab(self._fidelity_panel, "Fidelity")
            except Exception:
                self._fidelity_panel = None
                self._viz_tabs.addTab(
                    self._create_placeholder_tab("Fidelity"), "Fidelity"
                )
        else:
            self._fidelity_panel = None
            self._viz_tabs.addTab(
                self._create_placeholder_tab("Fidelity"), "Fidelity"
            )

        # Analysis dashboard panel
        if AnalysisPanel is not None:
            try:
                self._analysis_panel = AnalysisPanel()
                self._viz_tabs.addTab(self._analysis_panel, "Analysis")
            except Exception:
                self._analysis_panel = None
                self._viz_tabs.addTab(
                    self._create_placeholder_tab("Analysis"), "Analysis"
                )
        else:
            self._analysis_panel = None
            self._viz_tabs.addTab(
                self._create_placeholder_tab("Analysis"), "Analysis"
            )

        # Debugger panel
        if DebuggerPanel is not None:
            try:
                self._debugger_panel = DebuggerPanel()
                self._viz_tabs.addTab(self._debugger_panel, "Debugger")
            except Exception:
                self._debugger_panel = None
                self._viz_tabs.addTab(
                    self._create_placeholder_tab("Debugger"), "Debugger"
                )
        else:
            self._debugger_panel = None
            self._viz_tabs.addTab(
                self._create_placeholder_tab("Debugger"), "Debugger"
            )

        # Comparison panel
        if ComparisonPanel is not None:
            try:
                self._comparison_panel = ComparisonPanel()
                self._viz_tabs.addTab(self._comparison_panel, "Comparison")
            except Exception:
                self._comparison_panel = None
                self._viz_tabs.addTab(
                    self._create_placeholder_tab("Comparison"), "Comparison"
                )
        else:
            self._comparison_panel = None
            self._viz_tabs.addTab(
                self._create_placeholder_tab("Comparison"), "Comparison"
            )

        # Optimizer panel
        if OptimizerPanel is not None:
            try:
                self._optimizer_panel = OptimizerPanel()
                self._viz_tabs.addTab(self._optimizer_panel, "Optimizer")
            except Exception:
                self._optimizer_panel = None
                self._viz_tabs.addTab(
                    self._create_placeholder_tab("Optimizer"), "Optimizer"
                )
        else:
            self._optimizer_panel = None
            self._viz_tabs.addTab(
                self._create_placeholder_tab("Optimizer"), "Optimizer"
            )

        # QEC panel
        if QECPanel is not None:
            try:
                self._qec_panel = QECPanel()
                self._viz_tabs.addTab(self._qec_panel, "QEC")
            except Exception:
                self._qec_panel = None
                self._viz_tabs.addTab(
                    self._create_placeholder_tab("QEC"), "QEC"
                )
        else:
            self._qec_panel = None
            self._viz_tabs.addTab(
                self._create_placeholder_tab("QEC"), "QEC"
            )

        # Resource Monitor panel
        if ResourceMonitorPanel is not None:
            try:
                self._resource_panel = ResourceMonitorPanel()
                self._viz_tabs.addTab(self._resource_panel, "Resources")
            except Exception:
                self._resource_panel = None
                self._viz_tabs.addTab(
                    self._create_placeholder_tab("Resources"), "Resources"
                )
        else:
            self._resource_panel = None
            self._viz_tabs.addTab(
                self._create_placeholder_tab("Resources"), "Resources"
            )

        self._bottom_dock.setWidget(self._viz_tabs)
        self.addDockWidget(
            Qt.DockWidgetArea.BottomDockWidgetArea, self._bottom_dock
        )

        # Apply initial theme to all themed panels
        is_dark = self._config.theme == "dark"
        for panel in (
            self._bloch_panel, self._histogram_panel, self._density_matrix_panel,
            self._entanglement_panel, self._entropy_panel, self._fidelity_panel,
            self._analysis_panel, self._debugger_panel, self._comparison_panel,
            self._optimizer_panel, self._qec_panel, self._resource_panel,
        ):
            if panel is not None and hasattr(panel, "set_theme"):
                panel.set_theme(is_dark)

        # --- Add toggle actions to View menu ---
        # Insert before the theme separator
        first_separator = None
        for action in self._view_menu.actions():
            if action.isSeparator():
                first_separator = action
                break

        toggle_palette = self._gate_palette.toggleViewAction()
        toggle_palette.setText("Gate &Palette")
        self._view_menu.insertAction(first_separator, toggle_palette)

        toggle_props = self._properties_panel.toggleViewAction()
        toggle_props.setText("P&roperties Panel")
        self._view_menu.insertAction(first_separator, toggle_props)

        toggle_viz = self._bottom_dock.toggleViewAction()
        toggle_viz.setText("&Visualization Panels")
        self._view_menu.insertAction(first_separator, toggle_viz)

    def _create_placeholder_tab(self, name: str) -> QWidget:
        """Create a placeholder widget for visualization tabs not yet available."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        label = QLabel(f"{name}\n(Panel not yet available)")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet(
            "font-size: 14px; color: #6c7086; font-style: italic;"
        )
        layout.addWidget(label)
        return widget

    # ------------------------------------------------------------------
    # Status bar
    # ------------------------------------------------------------------

    def _create_status_bar(self):
        """Build the status bar with qubit count, gate count, sim status, bridge."""
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)

        self._status_qubits = QLabel()
        self._status_gates = QLabel()
        self._status_sim = QLabel()
        self._status_seed = QLabel()
        self._status_bridge = QLabel("Bridge: Off")

        self._status_bar.addPermanentWidget(self._status_qubits)
        self._status_bar.addPermanentWidget(self._status_gates)
        self._status_bar.addPermanentWidget(self._status_sim)
        self._status_bar.addPermanentWidget(self._status_seed)
        self._status_bar.addPermanentWidget(self._status_bridge)

    def _update_status_bar(self):
        """Refresh status bar labels from current circuit state."""
        self._status_qubits.setText(f"Qubits: {self._circuit.num_qubits}")
        self._status_gates.setText(f"Gates: {self._circuit.gate_count()}")

        if self._simulation_result is not None:
            self._status_sim.setText("Simulation: Complete")
        elif self._step_generator is not None:
            self._status_sim.setText("Simulation: Stepping...")
        else:
            self._status_sim.setText("Simulation: Ready")

        # Seed info
        if self._seed_manager is not None and self._seed_manager.seed is not None:
            self._status_seed.setText(f"Seed: {self._seed_manager.seed}")
        else:
            self._status_seed.setText("Seed: Random")

    # ------------------------------------------------------------------
    # Signal connections
    # ------------------------------------------------------------------

    def _connect_signals(self):
        """Connect inter-component signals."""

        # Properties panel signals
        self._properties_panel.params_changed.connect(
            self._on_gate_params_changed
        )
        self._properties_panel.target_qubits_changed.connect(
            self._on_gate_qubits_changed
        )

        # Circuit scene signals
        if hasattr(self, "_circuit_scene") and self._circuit_scene is not None:
            self._circuit_scene.gate_selected.connect(self._on_gate_selected)
            self._circuit_scene.circuit_changed.connect(self._on_circuit_changed)
            self._circuit_scene.gate_double_clicked.connect(
                self._on_gate_double_clicked
            )
            self._circuit_scene.qubit_state_toggled.connect(
                self._on_qubit_state_toggled
            )

        # Circuit view signals
        if self._circuit_view is not None:
            self._circuit_view.delete_selected.connect(self._on_delete)
            self._circuit_view.select_all_requested.connect(self._on_select_all)
            self._circuit_view.undo_requested.connect(self._undo_stack.undo)
            self._circuit_view.redo_requested.connect(self._undo_stack.redo)
            self._circuit_view.fit_view_requested.connect(
                self._circuit_view.fit_circuit_in_view
            )

        # Debugger panel signals
        if self._debugger_panel is not None:
            self._debugger_panel.breakpoint_changed.connect(
                self._on_debugger_breakpoint_changed
            )
            self._debugger_panel.position_changed.connect(
                self._on_debugger_position_changed
            )

        # Controller signals
        if self._circuit_controller is not None:
            self._circuit_controller.circuit_changed.connect(
                self._on_circuit_changed
            )

        if self._simulation_controller is not None:
            self._simulation_controller.simulation_finished.connect(
                self._on_simulation_finished
            )

    # ------------------------------------------------------------------
    # File operations
    # ------------------------------------------------------------------

    def _on_new(self):
        """Create a new empty circuit."""
        if not self._confirm_discard():
            return

        self._circuit = QuantumCircuit(num_qubits=self._config.default_qubits)
        self._current_file = None
        self._simulation_result = None
        self._undo_stack.clear()
        self._qubit_spinbox.setValue(self._circuit.num_qubits)
        self._sync_circuit_to_views()
        self._properties_panel.clear()
        self._update_status_bar()
        self._update_title()
        self.statusBar().showMessage("New circuit created.", 3000)

    def _on_open(self):
        """Open a circuit file."""
        if not self._confirm_discard():
            return

        start_dir = self._config.last_directory or str(Path.home())
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Open Circuit", start_dir, self.FILE_FILTER
        )
        if not filepath:
            return

        try:
            circuit = CircuitSerializer.load(filepath)
            self._circuit = circuit
            self._current_file = Path(filepath)
            self._simulation_result = None
            self._undo_stack.clear()
            self._qubit_spinbox.setValue(self._circuit.num_qubits)
            self._properties_panel.set_num_qubits(self._circuit.num_qubits)
            self._sync_circuit_to_views()
            self._properties_panel.clear()
            self._update_status_bar()
            self._update_title()

            self._config.add_recent_file(filepath)
            self._config.last_directory = str(Path(filepath).parent)
            self._config.save()

            self.statusBar().showMessage(f"Opened: {filepath}", 3000)
        except Exception as e:
            QMessageBox.critical(
                self, "Open Error",
                f"Failed to open circuit file:\n{e}"
            )
            logger.error("Failed to open file: %s", filepath, exc_info=True)

    def _on_save(self):
        """Save the circuit to the current file, or prompt for a new one."""
        if self._current_file is None:
            self._on_save_as()
            return

        self._save_to_file(self._current_file)

    def _on_save_as(self):
        """Save the circuit to a new file."""
        start_dir = self._config.last_directory or str(Path.home())
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Circuit As", start_dir, self.FILE_FILTER
        )
        if not filepath:
            return

        # Ensure .qsim extension
        path = Path(filepath)
        if path.suffix not in (".qsim", ".json"):
            path = path.with_suffix(".qsim")

        self._save_to_file(path)

    def _save_to_file(self, filepath: Path):
        """Save the circuit to the specified file path."""
        try:
            CircuitSerializer.save(self._circuit, filepath)
            self._current_file = filepath
            self._undo_stack.setClean()
            self._update_title()

            self._config.add_recent_file(str(filepath))
            self._config.last_directory = str(filepath.parent)
            self._config.save()

            self.statusBar().showMessage(f"Saved: {filepath}", 3000)
        except Exception as e:
            QMessageBox.critical(
                self, "Save Error",
                f"Failed to save circuit file:\n{e}"
            )
            logger.error("Failed to save file: %s", filepath, exc_info=True)

    def _on_export_image(self):
        """Export the circuit view as an image file."""
        if self._circuit_view is None:
            QMessageBox.information(
                self, "Export Image",
                "Circuit view is not available. Cannot export image."
            )
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export Circuit Image", "",
            "PNG Files (*.png);;SVG Files (*.svg);;All Files (*)"
        )
        if not filepath:
            return

        try:
            if hasattr(self._circuit_view, "export_image"):
                self._circuit_view.export_image(filepath)
                self.statusBar().showMessage(f"Exported: {filepath}", 3000)
            else:
                # Fallback: grab the widget as a pixmap
                pixmap = self._circuit_view.grab()
                pixmap.save(filepath)
                self.statusBar().showMessage(f"Exported: {filepath}", 3000)
        except Exception as e:
            QMessageBox.critical(
                self, "Export Error",
                f"Failed to export image:\n{e}"
            )

    def _confirm_discard(self) -> bool:
        """Ask the user to confirm discarding unsaved changes.

        Returns True if it is safe to proceed (no changes, or user chose
        to discard/save). Returns False if the user cancelled.
        """
        if self._undo_stack.isClean():
            return True

        result = QMessageBox.question(
            self, "Unsaved Changes",
            "The circuit has unsaved changes.\n"
            "Do you want to save before continuing?",
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
        )

        if result == QMessageBox.StandardButton.Save:
            self._on_save()
            return self._undo_stack.isClean()
        elif result == QMessageBox.StandardButton.Discard:
            return True
        else:
            return False

    # ------------------------------------------------------------------
    # Edit operations
    # ------------------------------------------------------------------

    def _on_delete(self):
        """Delete the currently selected gates from the circuit."""
        if not hasattr(self, '_circuit_scene') or self._circuit_scene is None:
            return
        selected = self._circuit_scene.selected_gate_instances()
        if not selected:
            return
        for gate_inst in selected:
            self._circuit.remove_gate(gate_inst)
            self._circuit_scene.remove_gate_visual(gate_inst)
        self._properties_panel.clear()
        self._on_circuit_changed()

    def _on_select_all(self):
        """Select all gates in the circuit editor."""
        if hasattr(self, '_circuit_scene') and self._circuit_scene is not None:
            self._circuit_scene.select_all_gates()

    # ------------------------------------------------------------------
    # Circuit operations
    # ------------------------------------------------------------------

    def _on_set_qubit_count(self):
        """Prompt user for a new qubit count via dialog."""
        count, ok = QInputDialog.getInt(
            self, "Set Qubit Count",
            "Number of qubits (1-16):",
            self._circuit.num_qubits, 1, 16,
        )
        if ok:
            self._set_qubit_count(count)

    def _on_qubit_spinbox_changed(self, value: int):
        """Handle qubit count change from the toolbar spinbox."""
        self._set_qubit_count(value)

    def _set_qubit_count(self, count: int):
        """Set the circuit qubit count, updating all views."""
        if count == self._circuit.num_qubits:
            return

        try:
            self._circuit.set_num_qubits(count)
        except ValueError as e:
            QMessageBox.warning(self, "Invalid Qubit Count", str(e))
            self._qubit_spinbox.setValue(self._circuit.num_qubits)
            return

        # Keep spinbox in sync (avoid recursion via blockSignals)
        self._qubit_spinbox.blockSignals(True)
        self._qubit_spinbox.setValue(count)
        self._qubit_spinbox.blockSignals(False)

        self._properties_panel.set_num_qubits(count)
        self._simulation_result = None
        self._sync_circuit_to_views()
        self._update_status_bar()
        self._update_title()

    def _on_clear_circuit(self):
        """Clear all gates from the circuit after confirmation."""
        if self._circuit.gate_count() == 0:
            return

        result = QMessageBox.question(
            self, "Clear Circuit",
            "Remove all gates from the circuit?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if result != QMessageBox.StandardButton.Yes:
            return

        self._circuit.clear()
        self._simulation_result = None
        self._properties_panel.clear()
        self._sync_circuit_to_views()
        self._update_status_bar()
        self.statusBar().showMessage("Circuit cleared.", 3000)

    def _on_load_template(self, template_name: str):
        """Load an algorithm template circuit, replacing the current one."""
        if not self._confirm_discard():
            return

        try:
            circuit = self._build_template_circuit(template_name)
            if circuit is None:
                return

            self._circuit = circuit
            self._current_file = None
            self._simulation_result = None
            self._undo_stack.clear()
            self._qubit_spinbox.setValue(self._circuit.num_qubits)
            self._properties_panel.set_num_qubits(self._circuit.num_qubits)
            self._sync_circuit_to_views()
            self._properties_panel.clear()
            self._update_status_bar()
            self._update_title()
            self.statusBar().showMessage(
                f"Loaded template: {template_name}", 3000
            )
        except Exception as e:
            QMessageBox.critical(
                self, "Template Error",
                f"Failed to load template '{template_name}':\n{e}"
            )
            logger.error(
                "Failed to load template: %s", template_name, exc_info=True
            )

    def _build_template_circuit(self, name: str) -> QuantumCircuit | None:
        """Build a circuit from a named algorithm template."""
        if name == "bell_state":
            return AlgorithmTemplate.bell_state()
        elif name == "ghz_state":
            count, ok = QInputDialog.getInt(
                self, "GHZ State", "Number of qubits:", 3, 2, 16
            )
            return AlgorithmTemplate.ghz_state(count) if ok else None
        elif name == "qft":
            count, ok = QInputDialog.getInt(
                self, "QFT", "Number of qubits:", 3, 2, 16
            )
            return AlgorithmTemplate.quantum_fourier_transform(count) if ok else None
        elif name == "inverse_qft":
            count, ok = QInputDialog.getInt(
                self, "Inverse QFT", "Number of qubits:", 3, 2, 16
            )
            return AlgorithmTemplate.inverse_qft(count) if ok else None
        elif name == "grover":
            count, ok = QInputDialog.getInt(
                self, "Grover's Search", "Number of qubits:", 3, 2, 8
            )
            if not ok:
                return None
            marked, ok2 = QInputDialog.getInt(
                self, "Grover's Search",
                f"Marked state (0 to {2**count - 1}):", 0, 0, 2**count - 1,
            )
            return AlgorithmTemplate.grover_search(count, marked) if ok2 else None
        elif name == "deutsch_jozsa":
            count, ok = QInputDialog.getInt(
                self, "Deutsch-Jozsa",
                "Total qubits (input + 1 ancilla):", 3, 2, 16,
            )
            return AlgorithmTemplate.deutsch_jozsa(count) if ok else None
        elif name == "teleportation":
            return AlgorithmTemplate.quantum_teleportation()
        elif name == "bernstein_vazirani":
            secret, ok = QInputDialog.getText(
                self, "Bernstein-Vazirani",
                "Secret bitstring (e.g. '101'):",
            )
            if ok and secret and all(c in "01" for c in secret):
                return AlgorithmTemplate.bernstein_vazirani(secret)
            elif ok:
                QMessageBox.warning(
                    self, "Invalid Input",
                    "Secret must be a binary string (e.g. '101')."
                )
            return None
        elif name == "superdense_coding":
            return AlgorithmTemplate.superdense_coding()
        else:
            QMessageBox.warning(
                self, "Unknown Template",
                f"Template '{name}' is not recognized."
            )
            return None

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def _on_run_simulation(self):
        """Run the full simulation and display results."""
        self._step_timer.stop()
        self._step_generator = None

        try:
            t0 = time.perf_counter()

            # Get RNG from seed manager if available
            rng = None
            seed = None
            if self._seed_manager is not None and self._seed_manager.seed is not None:
                rng = self._seed_manager.create_child_rng()
                seed = self._seed_manager.seed

            # First run ideal (noiseless) to cache ideal state
            ideal_sim = Simulator()
            ideal_result = ideal_sim.run(
                self._circuit, shots=0, seed=seed,
                rng=rng,
            )
            circ_hash = self._circuit.circuit_hash()
            self._reference_manager.store(
                ideal_result.final_state, "noiseless",
                circuit_hash=circ_hash,
            )

            # Then run with noise if configured
            simulator = Simulator(noise_model=self._noise_model)
            # Get fresh RNG for the actual run
            actual_rng = None
            if self._seed_manager is not None and self._seed_manager.seed is not None:
                actual_rng = self._seed_manager.create_child_rng()

            if self._noise_model is not None:
                result = simulator.run_with_noise(
                    self._circuit, shots=self._config.default_shots,
                    seed=seed, rng=actual_rng,
                )
                # For noisy sim, keep the ideal state separate
                result.final_state = ideal_result.final_state
            else:
                result = ideal_sim.run(
                    self._circuit,
                    shots=self._config.default_shots,
                    record_steps=False,
                    seed=seed,
                    rng=actual_rng if actual_rng else None,
                )
                self._reference_manager.store(
                    result.final_state, "noiseless",
                    circuit_hash=circ_hash,
                )

            elapsed = time.perf_counter() - t0

            self._simulation_result = result
            self._update_visualization_panels(result)
            self._update_status_bar()

            # Record to resource monitor
            if self._resource_panel is not None:
                self._resource_panel.record_simulation(
                    self._circuit.num_qubits,
                    self._circuit.gate_count(),
                    elapsed,
                )

            self.statusBar().showMessage(
                f"Simulation complete ({elapsed*1000:.1f} ms).", 5000
            )

        except Exception as e:
            QMessageBox.critical(
                self, "Simulation Error",
                f"Simulation failed:\n{e}"
            )
            logger.error("Simulation failed.", exc_info=True)

    def _on_step_simulation(self):
        """Start or continue step-by-step simulation."""
        if self._step_generator is None:
            # Start fresh step-by-step simulation
            simulator = Simulator(noise_model=self._noise_model)
            try:
                self._step_generator = simulator.run_step_by_step(self._circuit)
            except Exception as e:
                QMessageBox.critical(
                    self, "Simulation Error",
                    f"Step-by-step simulation failed:\n{e}"
                )
                return

        self._do_one_step()

    def _do_one_step(self):
        """Advance the step-by-step simulation by one column."""
        if self._step_generator is None:
            return

        try:
            state, col_idx = next(self._step_generator)

            # Build a partial result for visualization
            from quantum_sim.engine.measurement import MeasurementEngine
            counts = MeasurementEngine.sample(state, self._config.default_shots)
            partial_result = SimulationResult(
                final_state=state,
                measurement_counts=counts,
                num_shots=self._config.default_shots,
            )
            self._simulation_result = partial_result

            # Highlight current column in circuit view
            if self._circuit_view is not None and hasattr(
                self._circuit_view, "highlight_column"
            ):
                self._circuit_view.highlight_column(col_idx)

            # Entropy panel: update with column index
            if self._entropy_panel is not None and hasattr(
                self._entropy_panel, "update_state"
            ):
                try:
                    self._entropy_panel.update_state(state, col_idx)
                except Exception:
                    logger.debug("EntropyPanel step update failed.", exc_info=True)

            # Bloch sphere: append trajectory point
            if self._bloch_panel is not None and hasattr(
                self._bloch_panel, "append_trajectory_point"
            ):
                try:
                    self._bloch_panel.append_trajectory_point(state)
                except Exception:
                    logger.debug("Bloch trajectory update failed.", exc_info=True)

            self._update_visualization_panels(partial_result)
            self._update_status_bar()

            step_label = "initial state" if col_idx < 0 else f"column {col_idx}"
            self.statusBar().showMessage(
                f"Step: {step_label}", 2000
            )

        except StopIteration:
            self._step_generator = None
            self._step_timer.stop()
            self._update_status_bar()
            self.statusBar().showMessage(
                "Step-by-step simulation complete.", 3000
            )

    def _on_step_tick(self):
        """Timer tick for auto-stepping."""
        self._do_one_step()

    def _on_reset_simulation(self):
        """Reset all simulation state."""
        self._step_timer.stop()
        self._step_generator = None
        self._simulation_result = None
        self._reference_manager.clear()

        # Clear Bloch trajectory
        if self._bloch_panel is not None and hasattr(
            self._bloch_panel, "clear_trajectory"
        ):
            self._bloch_panel.clear_trajectory()

        # Clear entropy history
        if self._entropy_panel is not None and hasattr(
            self._entropy_panel, "clear"
        ):
            self._entropy_panel.clear()

        # Reset seed manager for fresh sequence
        if self._seed_manager is not None and self._seed_manager.seed is not None:
            self._seed_manager.reset()

        # Clear visualization panels
        self._clear_visualization_panels()

        # Remove column highlight
        if self._circuit_view is not None and hasattr(
            self._circuit_view, "highlight_column"
        ):
            self._circuit_view.highlight_column(-1)

        self._update_status_bar()
        self.statusBar().showMessage("Simulation reset.", 3000)

    def _on_configure_noise(self):
        """Open the noise configuration dialog."""
        if NoiseConfigDialog is not None:
            try:
                dialog = NoiseConfigDialog(self._noise_model, self)
                if dialog.exec():
                    self._noise_model = dialog.get_noise_model()
                    noise_status = (
                        "Noise model configured."
                        if self._noise_model is not None
                        else "Noise model cleared."
                    )
                    self.statusBar().showMessage(noise_status, 3000)
                return
            except Exception:
                logger.warning(
                    "NoiseConfigDialog failed.", exc_info=True
                )

        # Fallback: simple dialog to enable/disable depolarizing noise
        from quantum_sim.engine.noise import DepolarizingNoise

        if self._noise_model is not None:
            result = QMessageBox.question(
                self, "Noise Model",
                "A noise model is currently active.\n"
                "Do you want to disable it?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if result == QMessageBox.StandardButton.Yes:
                self._noise_model = None
                self.statusBar().showMessage("Noise model disabled.", 3000)
        else:
            prob, ok = QInputDialog.getDouble(
                self, "Configure Noise",
                "Depolarizing noise probability (0.0 - 1.0):",
                0.01, 0.0, 1.0, 4,
            )
            if ok and prob > 0:
                model = NoiseModel()
                model.add_global_noise(DepolarizingNoise(prob))
                self._noise_model = model
                self.statusBar().showMessage(
                    f"Depolarizing noise enabled (p={prob}).", 3000
                )

    # ------------------------------------------------------------------
    # Visualization updates
    # ------------------------------------------------------------------

    def _update_visualization_panels(self, result: SimulationResult):
        """Update all visualization tabs with simulation results."""
        state = result.final_state
        counts = result.measurement_counts

        # State Vector panel
        if self._state_vector_panel is not None and hasattr(
            self._state_vector_panel, "update_state"
        ):
            try:
                self._state_vector_panel.update_state(state)
            except Exception:
                logger.debug("StateVectorPanel update failed.", exc_info=True)

        # Bloch Sphere panel (BlochSphereWidget.update_state takes state + num_qubits)
        if self._bloch_panel is not None and hasattr(
            self._bloch_panel, "update_state"
        ):
            try:
                self._bloch_panel.update_state(state, state.num_qubits)
            except Exception:
                logger.debug("BlochSphereWidget update failed.", exc_info=True)

        # Histogram panel (HistogramPanel.update_histogram takes counts + shots)
        if self._histogram_panel is not None and hasattr(
            self._histogram_panel, "update_histogram"
        ):
            try:
                self._histogram_panel.update_histogram(counts, result.num_shots)
            except Exception:
                logger.debug("HistogramPanel update failed.", exc_info=True)

        # Density Matrix panel
        if self._density_matrix_panel is not None:
            try:
                if hasattr(self._density_matrix_panel, "set_circuit"):
                    self._density_matrix_panel.set_circuit(self._circuit)
                if hasattr(self._density_matrix_panel, "set_noise_model"):
                    self._density_matrix_panel.set_noise_model(self._noise_model)
                if hasattr(self._density_matrix_panel, "update_state"):
                    self._density_matrix_panel.update_state(state)
            except Exception:
                logger.debug(
                    "DensityMatrixPanel update failed.", exc_info=True
                )

        # Entanglement panel
        if self._entanglement_panel is not None and hasattr(
            self._entanglement_panel, "update_state"
        ):
            try:
                self._entanglement_panel.update_state(state)
            except Exception:
                logger.debug("EntanglementPanel update failed.", exc_info=True)

        # Fidelity panel - set circuit and ideal state
        if self._fidelity_panel is not None:
            try:
                if hasattr(self._fidelity_panel, "set_circuit"):
                    self._fidelity_panel.set_circuit(self._circuit)
                if hasattr(self._fidelity_panel, "update_state"):
                    self._fidelity_panel.update_state(state)
            except Exception:
                logger.debug("FidelityPanel update failed.", exc_info=True)

        # Analysis dashboard panel
        if self._analysis_panel is not None and hasattr(
            self._analysis_panel, "update_state"
        ):
            try:
                if self._reference_manager.has_reference and hasattr(
                    self._analysis_panel, "set_reference_state"
                ):
                    self._analysis_panel.set_reference_state(
                        self._reference_manager.reference.state
                    )
                self._analysis_panel.update_state(state)
            except Exception:
                logger.debug("AnalysisPanel update failed.", exc_info=True)

        # Optimizer panel - set circuit
        if self._optimizer_panel is not None:
            try:
                if hasattr(self._optimizer_panel, "set_circuit"):
                    self._optimizer_panel.set_circuit(self._circuit)
            except Exception:
                logger.debug("OptimizerPanel update failed.", exc_info=True)

        # Comparison panel - set circuit and noise model
        if self._comparison_panel is not None:
            try:
                if hasattr(self._comparison_panel, "set_circuit"):
                    self._comparison_panel.set_circuit(self._circuit)
                if hasattr(self._comparison_panel, "set_noise_model"):
                    self._comparison_panel.set_noise_model(self._noise_model)
            except Exception:
                logger.debug("ComparisonPanel update failed.", exc_info=True)

        # Debugger panel - set circuit and noise model
        if self._debugger_panel is not None:
            try:
                if hasattr(self._debugger_panel, "set_circuit"):
                    self._debugger_panel.set_circuit(self._circuit)
                if hasattr(self._debugger_panel, "set_noise_model"):
                    self._debugger_panel.set_noise_model(self._noise_model)
            except Exception:
                logger.debug("DebuggerPanel update failed.", exc_info=True)

        # Update bridge handler state
        if self._bridge_handler is not None:
            self._bridge_handler.set_circuit(self._circuit)
            self._bridge_handler.set_noise_model(self._noise_model)
            self._bridge_handler.set_last_result(result)
            if self._reference_manager.has_reference:
                self._bridge_handler.set_ideal_state(
                    self._reference_manager.reference.state
                )

    def _clear_visualization_panels(self):
        """Clear all visualization panels to their default state."""
        for panel_attr in (
            "_state_vector_panel",
            "_bloch_panel",
            "_histogram_panel",
            "_density_matrix_panel",
            "_entanglement_panel",
            "_entropy_panel",
            "_fidelity_panel",
            "_analysis_panel",
            "_debugger_panel",
            "_comparison_panel",
            "_optimizer_panel",
            "_qec_panel",
            "_resource_panel",
        ):
            panel = getattr(self, panel_attr, None)
            if panel is not None and hasattr(panel, "clear"):
                try:
                    panel.clear()
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # View / Theme
    # ------------------------------------------------------------------

    def _on_set_theme(self, theme_name: str):
        """Apply a theme and update the action check states."""
        self._theme_manager.apply_theme(theme_name)
        self._action_dark_theme.setChecked(theme_name == "dark")
        self._action_light_theme.setChecked(theme_name == "light")
        self._config.theme = theme_name
        self._config.save()

        # Propagate theme to all themed panels
        is_dark = theme_name == "dark"
        for panel in (
            self._bloch_panel, self._histogram_panel, self._density_matrix_panel,
            self._entanglement_panel, self._entropy_panel, self._fidelity_panel,
            self._analysis_panel, self._debugger_panel, self._comparison_panel,
            self._optimizer_panel, self._qec_panel, self._resource_panel,
        ):
            if panel is not None and hasattr(panel, "set_theme"):
                panel.set_theme(is_dark)

    # ------------------------------------------------------------------
    # Signal handlers
    # ------------------------------------------------------------------

    def _on_gate_selected(self, gate: GateInstance | None):
        """Handle gate selection in the circuit view."""
        self._properties_panel.set_gate(gate)

    def _on_circuit_changed(self):
        """Handle circuit modification from any source."""
        self._simulation_result = None
        # Invalidate reference if circuit structure changed
        self._reference_manager.check_invalidation(self._circuit.circuit_hash())
        self._update_status_bar()
        self._update_title()

    def _on_gate_double_clicked(self, gate: GateInstance | None):
        """Handle gate double-click to open parameter editor."""
        if gate is None:
            return
        try:
            gate_def = GateRegistry.instance().get(gate.gate_name)
        except KeyError:
            return
        if gate_def.num_params == 0:
            return
        from quantum_sim.gui.dialogs.gate_param_dialog import GateParamDialog
        dialog = GateParamDialog(gate_def, gate.params, self)
        if dialog.exec():
            gate.params = dialog.get_params()
            if hasattr(self, '_circuit_scene') and self._circuit_scene is not None:
                self._circuit_scene.refresh_gate_visual(gate)
            self._on_circuit_changed()

    def _on_gate_params_changed(self, gate: GateInstance, new_params: list):
        """Handle parameter changes from the properties panel."""
        if gate is None:
            return

        gate.params = list(new_params)
        self._on_circuit_changed()
        self.statusBar().showMessage("Gate parameters updated.", 2000)

    def _on_gate_qubits_changed(self, gate: GateInstance, new_qubits: list):
        """Handle qubit assignment changes from the properties panel."""
        if gate is None:
            return

        # Validate: no duplicate qubits
        if len(set(new_qubits)) != len(new_qubits):
            QMessageBox.warning(
                self, "Invalid Qubits",
                "Each qubit in a gate must be unique."
            )
            # Reset the properties panel to current values
            self._properties_panel.set_gate(gate)
            return

        gate.target_qubits = list(new_qubits)
        self._on_circuit_changed()
        self.statusBar().showMessage("Gate qubits updated.", 2000)

    def _on_qubit_state_toggled(self, qubit_index: int, new_state: int):
        """Handle qubit initial state toggle from the circuit editor."""
        state_label = f"|{new_state}>"
        self.statusBar().showMessage(
            f"q{qubit_index} initial state set to {state_label}", 2000
        )

    def _on_debugger_breakpoint_changed(self, column: int, is_set: bool):
        """Handle breakpoint toggle from debugger panel."""
        if hasattr(self, "_circuit_scene") and self._circuit_scene is not None:
            self._circuit_scene.set_breakpoint(column, is_set)
        status = "set" if is_set else "removed"
        self.statusBar().showMessage(f"Breakpoint {status} at column {column}", 2000)

    def _on_debugger_position_changed(self, column_index: int):
        """Handle debugger step position change."""
        if hasattr(self, "_circuit_scene") and self._circuit_scene is not None:
            self._circuit_scene.set_debug_highlight(
                column_index if column_index >= 0 else None
            )

    def _on_undo_clean_changed(self, clean: bool):
        """Update title bar asterisk when undo stack clean state changes."""
        self._update_title()

    def _on_simulation_finished(self, result: SimulationResult):
        """Handle simulation completion from the simulation controller."""
        self._simulation_result = result
        self._update_visualization_panels(result)
        self._update_status_bar()
        self.statusBar().showMessage("Simulation complete.", 3000)

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _sync_circuit_to_views(self):
        """Push the current circuit to all views that need it."""
        # Update the circuit scene (which owns the visual representation)
        if hasattr(self, "_circuit_scene") and self._circuit_scene is not None:
            try:
                if hasattr(self._circuit_scene, "set_circuit"):
                    self._circuit_scene.set_circuit(self._circuit)
                elif hasattr(self._circuit_scene, "circuit"):
                    self._circuit_scene.circuit = self._circuit
            except Exception:
                pass

        # Update the circuit controller
        if self._circuit_controller is not None:
            try:
                if hasattr(self._circuit_controller, "circuit"):
                    self._circuit_controller.circuit = self._circuit
                elif hasattr(self._circuit_controller, "set_circuit"):
                    self._circuit_controller.set_circuit(self._circuit)
            except Exception:
                pass

        # Update the simulation controller
        if self._simulation_controller is not None:
            try:
                if hasattr(self._simulation_controller, "set_circuit"):
                    self._simulation_controller.set_circuit(self._circuit)
            except Exception:
                pass

    def _update_title(self):
        """Update the window title to reflect current file and modified state."""
        title = self.APP_NAME
        if self._current_file is not None:
            title = f"{self._current_file.name} - {title}"
        else:
            title = f"Untitled - {title}"

        if not self._undo_stack.isClean():
            title = f"* {title}"

        self.setWindowTitle(title)

    # ------------------------------------------------------------------
    # Seed / Experiment / Benchmarks / Bridge
    # ------------------------------------------------------------------

    def _on_set_seed(self):
        """Prompt user for a random seed."""
        current = ""
        if self._seed_manager is not None and self._seed_manager.seed is not None:
            current = str(self._seed_manager.seed)

        text, ok = QInputDialog.getText(
            self, "Set Random Seed",
            "Enter seed (integer, or leave empty for random):",
            text=current,
        )
        if not ok:
            return

        if text.strip() == "":
            if self._seed_manager is not None:
                self._seed_manager.set_seed(None)
            self.statusBar().showMessage("Seed cleared (random mode).", 3000)
        else:
            try:
                seed = int(text.strip())
                if self._seed_manager is not None:
                    self._seed_manager.set_seed(seed)
                self.statusBar().showMessage(f"Seed set to {seed}.", 3000)
            except ValueError:
                QMessageBox.warning(
                    self, "Invalid Seed", "Seed must be an integer."
                )

    def _on_export_experiment(self):
        """Export current experiment configuration to JSON."""
        if ExperimentConfig is None:
            QMessageBox.information(
                self, "Export Experiment",
                "Experiment module not available."
            )
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export Experiment", "",
            "JSON Files (*.json);;All Files (*)"
        )
        if not filepath:
            return

        try:
            seed = self._seed_manager.seed if self._seed_manager else None
            exp = ExperimentConfig.from_current(
                circuit=self._circuit,
                noise_model=self._noise_model,
                seed=seed,
                shots=self._config.default_shots,
                result=self._simulation_result,
            )
            exp.save(filepath)
            self.statusBar().showMessage(f"Experiment exported: {filepath}", 3000)
        except Exception as e:
            QMessageBox.critical(
                self, "Export Error", f"Failed to export experiment:\n{e}"
            )

    def _on_import_experiment(self):
        """Import experiment configuration from JSON."""
        if ExperimentConfig is None:
            QMessageBox.information(
                self, "Import Experiment",
                "Experiment module not available."
            )
            return

        filepath, _ = QFileDialog.getOpenFileName(
            self, "Import Experiment", "",
            "JSON Files (*.json);;All Files (*)"
        )
        if not filepath:
            return

        try:
            exp = ExperimentConfig.load(filepath)

            # Restore circuit
            if exp.circuit is not None:
                from quantum_sim.engine.circuit import QuantumCircuit
                self._circuit = QuantumCircuit.from_dict(exp.circuit)
                self._qubit_spinbox.setValue(self._circuit.num_qubits)
                self._properties_panel.set_num_qubits(self._circuit.num_qubits)
                self._sync_circuit_to_views()

            # Restore seed
            if exp.seed is not None and self._seed_manager is not None:
                self._seed_manager.set_seed(exp.seed)

            # Restore noise model
            if exp.noise_model is not None:
                self._noise_model = NoiseModel.from_dict(exp.noise_model)
            else:
                self._noise_model = None

            self._update_status_bar()
            self._update_title()
            self.statusBar().showMessage(f"Experiment imported: {filepath}", 3000)
        except Exception as e:
            QMessageBox.critical(
                self, "Import Error", f"Failed to import experiment:\n{e}"
            )

    def _on_run_benchmarks(self):
        """Run the benchmark suite and display results."""
        if BenchmarkSuite is None:
            QMessageBox.information(
                self, "Benchmarks", "Benchmark module not available."
            )
            return

        try:
            self.statusBar().showMessage("Running benchmarks...", 0)
            QApplication.processEvents()

            seed = self._seed_manager.seed if self._seed_manager else None
            results = BenchmarkSuite.run_all(
                noise_model=self._noise_model, seed=seed
            )

            # Build results message
            lines = ["Benchmark Results:\n"]
            all_passed = True
            for r in results:
                status = "PASS" if r.passed else "FAIL"
                if not r.passed:
                    all_passed = False
                lines.append(
                    f"  {r.name}: {status}  "
                    f"(F={r.fidelity:.4f}, TVD={r.tvd:.4f}, "
                    f"{r.runtime_ms:.1f}ms)"
                )

            summary = "All benchmarks passed!" if all_passed else "Some benchmarks failed."
            lines.append(f"\n{summary}")

            QMessageBox.information(
                self, "Benchmark Results", "\n".join(lines)
            )
            self.statusBar().showMessage(summary, 5000)

        except Exception as e:
            QMessageBox.critical(
                self, "Benchmark Error", f"Benchmarks failed:\n{e}"
            )

    def _on_toggle_bridge(self):
        """Start or stop the Live Bridge API server."""
        if self._bridge_server is None:
            QMessageBox.information(
                self, "Bridge", "Bridge module not available."
            )
            return

        if self._bridge_server.is_running:
            self._bridge_server.stop()
            self._action_toggle_bridge.setText("Start &Bridge Server")
            self._status_bridge.setText("Bridge: Off")
            self.statusBar().showMessage("Bridge server stopped.", 3000)
        else:
            # Sync current state to handler
            if self._bridge_handler is not None:
                self._bridge_handler.set_circuit(self._circuit)
                self._bridge_handler.set_noise_model(self._noise_model)
                if self._simulation_result is not None:
                    self._bridge_handler.set_last_result(self._simulation_result)
                if self._reference_manager.has_reference:
                    self._bridge_handler.set_ideal_state(
                        self._reference_manager.reference.state
                    )

            self._bridge_server.start()
            self._action_toggle_bridge.setText("Stop &Bridge Server")
            self._status_bridge.setText("Bridge: Listening")

            # Connect status signal
            if self._bridge_server.worker is not None:
                self._bridge_server.worker.status_changed.connect(
                    lambda s: self._status_bridge.setText(f"Bridge: {s}")
                )

            self.statusBar().showMessage("Bridge server started on port 9876.", 3000)

    # ------------------------------------------------------------------
    # About dialog
    # ------------------------------------------------------------------

    def _on_about(self):
        """Show the About dialog."""
        QMessageBox.about(
            self, "About Quantum Circuit Simulator",
            "<h2>Quantum Circuit Simulator</h2>"
            "<p>A research-grade visual quantum circuit simulator built with PyQt6.</p>"
            "<p><b>Core Features:</b></p>"
            "<ul>"
            "<li>Drag-and-drop circuit editor (1-16 qubits)</li>"
            "<li>State vector, Bloch sphere, histogram, density matrix</li>"
            "<li>Step-by-step simulation with trajectory tracking</li>"
            "<li>Noise models (bit-flip, phase-flip, depolarizing, amplitude damping)</li>"
            "<li>Built-in quantum algorithm templates</li>"
            "</ul>"
            "<p><b>Research Features:</b></p>"
            "<ul>"
            "<li>Quantitative analysis: fidelity, entropy, purity, concurrence</li>"
            "<li>Entanglement graph visualization</li>"
            "<li>Entropy evolution tracking</li>"
            "<li>Fidelity decay sweep analysis</li>"
            "<li>Reproducible experiments with seed management</li>"
            "<li>Benchmark suite for validation</li>"
            "<li>Live Bridge API for external script connectivity</li>"
            "</ul>"
        )

    # ------------------------------------------------------------------
    # Close event
    # ------------------------------------------------------------------

    def closeEvent(self, event):
        """Handle window close: prompt to save, persist config."""
        if not self._confirm_discard():
            event.ignore()
            return

        # Stop bridge server
        if self._bridge_server is not None and self._bridge_server.is_running:
            self._bridge_server.stop()

        # Stop any running simulation
        self._step_timer.stop()
        self._step_generator = None

        # Save window geometry to config
        self._config.window_width = self.width()
        self._config.window_height = self.height()
        self._config.save()

        event.accept()
