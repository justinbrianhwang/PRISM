"""Properties panel dock widget for viewing and editing gate properties.

Displays the selected gate's name, type, target qubits, and parameters.
Parameters can be edited for parameterized gates (Rx, Ry, Rz, Phase, U3).
Changes are emitted via the params_changed signal.
"""

from __future__ import annotations

import math
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QColor
from PyQt6.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QSpinBox, QDoubleSpinBox, QPushButton, QFrame,
    QGroupBox, QScrollArea, QSizePolicy,
)

from quantum_sim.engine.gate_registry import GateRegistry
from quantum_sim.engine.gates import GateType, GateDefinition
from quantum_sim.engine.circuit import GateInstance


class PropertiesPanel(QDockWidget):
    """Dock widget displaying and editing properties of the selected gate.

    Signals:
        params_changed(GateInstance, list[float]):
            Emitted when the user clicks Apply after editing parameters.
            The first argument is the gate instance, the second is the
            new parameter values.
        target_qubits_changed(GateInstance, list[int]):
            Emitted when the user changes the target qubit assignments.
    """

    params_changed = pyqtSignal(object, list)
    target_qubits_changed = pyqtSignal(object, list)

    def __init__(self, parent: QWidget | None = None):
        super().__init__("Properties", parent)
        self.setObjectName("PropertiesPanelDock")
        self.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea
            | Qt.DockWidgetArea.RightDockWidgetArea
        )
        self.setMinimumWidth(220)

        self._current_gate: GateInstance | None = None
        self._current_gate_def: GateDefinition | None = None
        self._num_qubits: int = 4
        self._param_spinboxes: list[QDoubleSpinBox] = []
        self._qubit_spinboxes: list[QSpinBox] = []

        self._setup_ui()
        self._show_empty_state()

    def _setup_ui(self):
        """Build the properties panel UI."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        self._container = QWidget()
        self._main_layout = QVBoxLayout(self._container)
        self._main_layout.setContentsMargins(8, 8, 8, 8)
        self._main_layout.setSpacing(8)

        # --- Gate Info Section ---
        self._info_group = QGroupBox("Gate Information")
        info_layout = QFormLayout(self._info_group)
        info_layout.setSpacing(6)

        self._name_label = QLabel("--")
        self._name_label.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))

        self._type_label = QLabel("--")
        self._type_label.setStyleSheet("color: #a6adc8;")

        self._color_indicator = QFrame()
        self._color_indicator.setFixedSize(20, 20)
        self._color_indicator.setStyleSheet(
            "background-color: transparent; border-radius: 4px; border: 1px solid #45475a;"
        )

        name_row = QHBoxLayout()
        name_row.addWidget(self._color_indicator)
        name_row.addWidget(self._name_label)
        name_row.addStretch()

        info_layout.addRow(name_row)
        info_layout.addRow("Type:", self._type_label)

        self._main_layout.addWidget(self._info_group)

        # --- Target Qubits Section ---
        self._qubits_group = QGroupBox("Target Qubits")
        self._qubits_layout = QFormLayout(self._qubits_group)
        self._qubits_layout.setSpacing(6)
        self._main_layout.addWidget(self._qubits_group)

        # --- Parameters Section ---
        self._params_group = QGroupBox("Parameters")
        self._params_layout = QFormLayout(self._params_group)
        self._params_layout.setSpacing(6)
        self._main_layout.addWidget(self._params_group)

        # --- Apply Button ---
        self._apply_btn = QPushButton("Apply Changes")
        self._apply_btn.setObjectName("primaryButton")
        self._apply_btn.setMinimumHeight(32)
        self._apply_btn.clicked.connect(self._on_apply)
        self._apply_btn.setEnabled(False)
        self._main_layout.addWidget(self._apply_btn)

        # --- Spacer ---
        self._main_layout.addStretch()

        # --- Empty state label ---
        self._empty_label = QLabel("Select a gate in the circuit\nto view its properties.")
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_label.setStyleSheet("color: #6c7086; font-style: italic;")
        self._empty_label.setWordWrap(True)
        self._main_layout.addWidget(self._empty_label)

        scroll.setWidget(self._container)
        self.setWidget(scroll)

    def _show_empty_state(self):
        """Show the empty state (no gate selected)."""
        self._info_group.hide()
        self._qubits_group.hide()
        self._params_group.hide()
        self._apply_btn.hide()
        self._empty_label.show()

    def _show_gate_state(self):
        """Show the gate editing state."""
        self._info_group.show()
        self._qubits_group.show()
        self._empty_label.hide()

    def set_num_qubits(self, n: int):
        """Update the maximum qubit index for spin boxes."""
        self._num_qubits = n
        for sb in self._qubit_spinboxes:
            sb.setMaximum(n - 1)

    def set_gate(self, gate: GateInstance | None):
        """Display properties for the given gate instance, or clear if None.

        Args:
            gate: The GateInstance to display, or None to clear the panel.
        """
        self._current_gate = gate

        if gate is None:
            self._current_gate_def = None
            self._show_empty_state()
            return

        try:
            registry = GateRegistry.instance()
            gate_def = registry.get(gate.gate_name)
        except KeyError:
            self._current_gate_def = None
            self._show_empty_state()
            return

        self._current_gate_def = gate_def
        self._show_gate_state()

        # Update info section
        self._name_label.setText(gate_def.display_name)
        self._type_label.setText(self._format_gate_type(gate_def.gate_type))
        self._color_indicator.setStyleSheet(
            f"background-color: {gate_def.color}; border-radius: 4px; "
            f"border: 1px solid {gate_def.color};"
        )

        # Update target qubits section
        self._rebuild_qubit_spinboxes(gate, gate_def)

        # Update parameters section
        self._rebuild_param_spinboxes(gate, gate_def)

    def _format_gate_type(self, gate_type: GateType) -> str:
        """Return a human-readable string for the gate type."""
        type_labels = {
            GateType.SINGLE: "Single Qubit",
            GateType.CONTROLLED: "Controlled",
            GateType.MULTI: "Multi-Qubit",
            GateType.MEASUREMENT: "Measurement",
            GateType.BARRIER: "Barrier",
        }
        return type_labels.get(gate_type, str(gate_type.value))

    def _rebuild_qubit_spinboxes(
        self, gate: GateInstance, gate_def: GateDefinition
    ):
        """Rebuild the target qubit spin boxes for the current gate."""
        # Clear existing qubit widgets
        self._qubit_spinboxes.clear()
        while self._qubits_layout.count() > 0:
            item = self._qubits_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if gate_def.gate_type == GateType.CONTROLLED:
            # Show control and target qubits separately
            for i in range(gate_def.num_controls):
                sb = QSpinBox()
                sb.setMinimum(0)
                sb.setMaximum(self._num_qubits - 1)
                if i < len(gate.target_qubits):
                    sb.setValue(gate.target_qubits[i])
                sb.valueChanged.connect(self._on_value_changed)
                self._qubit_spinboxes.append(sb)
                label = f"Control {i}:" if gate_def.num_controls > 1 else "Control:"
                self._qubits_layout.addRow(label, sb)

            for i in range(gate_def.num_targets):
                sb = QSpinBox()
                sb.setMinimum(0)
                sb.setMaximum(self._num_qubits - 1)
                idx = gate_def.num_controls + i
                if idx < len(gate.target_qubits):
                    sb.setValue(gate.target_qubits[idx])
                sb.valueChanged.connect(self._on_value_changed)
                self._qubit_spinboxes.append(sb)
                label = f"Target {i}:" if gate_def.num_targets > 1 else "Target:"
                self._qubits_layout.addRow(label, sb)
        else:
            # Single qubit or other gate types
            for i, qubit_idx in enumerate(gate.target_qubits):
                sb = QSpinBox()
                sb.setMinimum(0)
                sb.setMaximum(self._num_qubits - 1)
                sb.setValue(qubit_idx)
                sb.valueChanged.connect(self._on_value_changed)
                self._qubit_spinboxes.append(sb)
                label = f"Qubit {i}:" if len(gate.target_qubits) > 1 else "Qubit:"
                self._qubits_layout.addRow(label, sb)

        self._qubits_group.show()

    def _rebuild_param_spinboxes(
        self, gate: GateInstance, gate_def: GateDefinition
    ):
        """Rebuild the parameter spin boxes for the current gate."""
        self._param_spinboxes.clear()
        while self._params_layout.count() > 0:
            item = self._params_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if gate_def.num_params == 0:
            self._params_group.hide()
            self._apply_btn.hide()
            return

        self._params_group.show()
        self._apply_btn.show()
        self._apply_btn.setEnabled(True)

        for i in range(gate_def.num_params):
            sb = QDoubleSpinBox()
            sb.setMinimum(-100.0)
            sb.setMaximum(100.0)
            sb.setDecimals(6)
            sb.setSingleStep(0.1)
            sb.setSuffix(" rad")

            # Set current value
            if i < len(gate.params):
                sb.setValue(gate.params[i])
            else:
                sb.setValue(0.0)

            sb.valueChanged.connect(self._on_value_changed)
            self._param_spinboxes.append(sb)

            # Use the parameter name from the gate definition
            param_name = (
                gate_def.param_names[i] if i < len(gate_def.param_names)
                else f"p{i}"
            )
            self._params_layout.addRow(f"{param_name}:", sb)

        # Add preset buttons for common angles
        presets_widget = QWidget()
        presets_layout = QHBoxLayout(presets_widget)
        presets_layout.setContentsMargins(0, 4, 0, 0)
        presets_layout.setSpacing(4)

        preset_label = QLabel("Presets:")
        preset_label.setStyleSheet("font-size: 11px; color: #a6adc8;")
        presets_layout.addWidget(preset_label)

        presets = [
            ("\u03c0/4", math.pi / 4),
            ("\u03c0/2", math.pi / 2),
            ("\u03c0", math.pi),
            ("2\u03c0", 2 * math.pi),
        ]

        for label_text, value in presets:
            btn = QPushButton(label_text)
            btn.setFixedSize(40, 24)
            btn.setStyleSheet("font-size: 11px; padding: 2px;")
            btn.setToolTip(f"Set first parameter to {value:.4f} rad")
            btn.clicked.connect(
                lambda checked, v=value: self._set_first_param(v)
            )
            presets_layout.addWidget(btn)

        presets_layout.addStretch()
        self._params_layout.addRow(presets_widget)

    def _set_first_param(self, value: float):
        """Set the first parameter spin box to the given value."""
        if self._param_spinboxes:
            self._param_spinboxes[0].setValue(value)

    def _on_value_changed(self):
        """Called when any spin box value changes. Enables the apply button."""
        self._apply_btn.setEnabled(True)

    def _on_apply(self):
        """Emit signals with the updated parameter and qubit values."""
        if self._current_gate is None:
            return

        # Collect new qubit assignments
        new_qubits = [sb.value() for sb in self._qubit_spinboxes]
        if new_qubits and new_qubits != self._current_gate.target_qubits:
            self.target_qubits_changed.emit(self._current_gate, new_qubits)

        # Collect new parameter values
        if self._param_spinboxes:
            new_params = [sb.value() for sb in self._param_spinboxes]
            if new_params != self._current_gate.params:
                self.params_changed.emit(self._current_gate, new_params)

        self._apply_btn.setEnabled(False)

    def clear(self):
        """Clear the panel, showing the empty state."""
        self.set_gate(None)
