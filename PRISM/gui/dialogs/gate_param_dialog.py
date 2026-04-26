"""Dialog for editing quantum gate parameters with quick-value buttons."""

from __future__ import annotations

import math

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QDoubleSpinBox, QPushButton, QDialogButtonBox, QLabel,
    QWidget, QGroupBox,
)

from quantum_sim.engine.gates import GateDefinition


class GateParamDialog(QDialog):
    """Dialog for editing parameters of a parameterized quantum gate.

    Displays a labeled QDoubleSpinBox for each parameter, along with
    quick-value buttons for common angles (0, pi/4, pi/2, pi, 2*pi).
    """

    # Common angle values with display labels
    QUICK_VALUES: list[tuple[str, float]] = [
        ("0", 0.0),
        ("\u03C0/4", math.pi / 4),
        ("\u03C0/2", math.pi / 2),
        ("\u03C0", math.pi),
        ("2\u03C0", 2.0 * math.pi),
    ]

    def __init__(
        self,
        gate_def: GateDefinition,
        current_params: list[float] | None = None,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)

        self._gate_def = gate_def
        self._param_spinboxes: list[QDoubleSpinBox] = []

        if current_params is None:
            current_params = [0.0] * gate_def.num_params

        self._current_params = list(current_params)

        self.setWindowTitle(f"{gate_def.display_name} Parameters")
        self.setMinimumWidth(380)

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Title
        title_label = QLabel(
            f"<b>{self._gate_def.display_name}</b> "
            f"({self._gate_def.symbol})"
        )
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)

        # Parameter inputs
        form_layout = QFormLayout()

        for i, param_name in enumerate(self._gate_def.param_names):
            param_widget = QWidget()
            param_layout = QHBoxLayout(param_widget)
            param_layout.setContentsMargins(0, 0, 0, 0)

            # Spin box for the parameter value
            spinbox = QDoubleSpinBox()
            spinbox.setRange(-100.0, 100.0)
            spinbox.setDecimals(6)
            spinbox.setSingleStep(0.01)
            spinbox.setMinimumWidth(120)

            # Set current value
            if i < len(self._current_params):
                spinbox.setValue(self._current_params[i])
            else:
                spinbox.setValue(0.0)

            param_layout.addWidget(spinbox)
            self._param_spinboxes.append(spinbox)

            # Quick value buttons
            for label_text, value in self.QUICK_VALUES:
                btn = QPushButton(label_text)
                btn.setMaximumWidth(45)
                btn.setToolTip(f"Set to {value:.6f}")
                # Use default argument to capture value in closure
                btn.clicked.connect(
                    lambda checked, sb=spinbox, v=value: sb.setValue(v)
                )
                param_layout.addWidget(btn)

            # Negative pi button
            neg_pi_btn = QPushButton("-\u03C0")
            neg_pi_btn.setMaximumWidth(45)
            neg_pi_btn.setToolTip(f"Set to {-math.pi:.6f}")
            neg_pi_btn.clicked.connect(
                lambda checked, sb=spinbox: sb.setValue(-math.pi)
            )
            param_layout.addWidget(neg_pi_btn)

            form_layout.addRow(f"{param_name}:", param_widget)

        layout.addLayout(form_layout)

        # Current value display
        if self._gate_def.num_params > 0:
            info_group = QGroupBox("Current Values (radians)")
            info_layout = QVBoxLayout(info_group)
            self._value_label = QLabel("")
            self._value_label.setStyleSheet("color: gray; font-size: 11px;")
            info_layout.addWidget(self._value_label)
            layout.addWidget(info_group)

            # Update the value display when spinboxes change
            for sb in self._param_spinboxes:
                sb.valueChanged.connect(self._update_value_display)
            self._update_value_display()

        # OK / Cancel buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _update_value_display(self):
        """Update the informational display of current parameter values."""
        parts = []
        for i, sb in enumerate(self._param_spinboxes):
            name = (
                self._gate_def.param_names[i]
                if i < len(self._gate_def.param_names)
                else f"p{i}"
            )
            value = sb.value()
            # Show value in terms of pi if close
            pi_ratio = value / math.pi if abs(value) > 1e-10 else 0.0
            if abs(value) < 1e-10:
                parts.append(f"{name} = 0")
            elif abs(pi_ratio - round(pi_ratio)) < 1e-6:
                r = round(pi_ratio)
                if r == 1:
                    parts.append(f"{name} = \u03C0")
                elif r == -1:
                    parts.append(f"{name} = -\u03C0")
                else:
                    parts.append(f"{name} = {r}\u03C0")
            else:
                parts.append(f"{name} = {value:.6f} rad")
        self._value_label.setText("  |  ".join(parts))

    def get_params(self) -> list[float]:
        """Return the current parameter values from the spinboxes."""
        return [sb.value() for sb in self._param_spinboxes]

    @staticmethod
    def get_gate_params(
        gate_def: GateDefinition,
        current_params: list[float] | None = None,
        parent: QWidget | None = None,
    ) -> tuple[list[float] | None, bool]:
        """Static convenience method to show the dialog and return results.

        Returns:
            Tuple of (params, accepted). params is None if cancelled.
        """
        dialog = GateParamDialog(gate_def, current_params, parent)
        result = dialog.exec()
        if result == QDialog.DialogCode.Accepted:
            return dialog.get_params(), True
        return None, False
