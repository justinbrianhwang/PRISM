"""Dialog for configuring quantum noise models."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QCheckBox, QDoubleSpinBox, QDialogButtonBox, QLabel,
    QWidget, QGroupBox, QTextEdit, QPushButton,
)

from quantum_sim.engine.noise import (
    NoiseModel,
    BitFlipNoise,
    PhaseFlipNoise,
    DepolarizingNoise,
    AmplitudeDampingNoise,
    ReadoutError,
)


class _NoiseChannelWidget(QWidget):
    """Widget for configuring a single noise channel: checkbox + probability slider."""

    def __init__(
        self,
        name: str,
        description: str,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self._name = name

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)

        # Enable checkbox
        self._checkbox = QCheckBox(name)
        self._checkbox.setToolTip(description)
        self._checkbox.toggled.connect(self._on_toggled)
        layout.addWidget(self._checkbox)

        layout.addStretch()

        # Probability label
        layout.addWidget(QLabel("p ="))

        # Probability spinbox
        self._spinbox = QDoubleSpinBox()
        self._spinbox.setRange(0.0, 1.0)
        self._spinbox.setDecimals(4)
        self._spinbox.setSingleStep(0.01)
        self._spinbox.setValue(0.01)
        self._spinbox.setMinimumWidth(90)
        self._spinbox.setEnabled(False)
        layout.addWidget(self._spinbox)

    def _on_toggled(self, checked: bool):
        self._spinbox.setEnabled(checked)

    @property
    def is_enabled(self) -> bool:
        return self._checkbox.isChecked()

    @property
    def probability(self) -> float:
        return self._spinbox.value()

    def set_enabled(self, enabled: bool) -> None:
        self._checkbox.setChecked(enabled)

    def set_probability(self, p: float) -> None:
        self._spinbox.setValue(p)


class NoiseConfigDialog(QDialog):
    """Dialog for configuring quantum noise channels.

    Provides checkboxes and probability spinboxes for each supported
    noise type: BitFlip, PhaseFlip, Depolarizing, AmplitudeDamping.
    Returns a NoiseModel or None.
    """

    def __init__(
        self,
        current_model: NoiseModel | None = None,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Noise Model Configuration")
        self.setMinimumWidth(450)
        self.setMinimumHeight(350)

        self._current_model = current_model
        self._channel_widgets: dict[str, _NoiseChannelWidget] = {}

        self._setup_ui()

        if current_model is not None:
            self._load_from_model(current_model)

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Title
        title_label = QLabel("<b>Configure Noise Model</b>")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)

        description = QLabel(
            "Enable noise channels that will be applied globally "
            "after each gate operation during simulation."
        )
        description.setWordWrap(True)
        description.setStyleSheet("color: gray; font-size: 11px; padding: 4px;")
        layout.addWidget(description)

        # Noise channels group
        channels_group = QGroupBox("Noise Channels")
        channels_layout = QVBoxLayout(channels_group)

        # Bit Flip
        self._channel_widgets["BitFlip"] = _NoiseChannelWidget(
            "Bit Flip",
            "Applies X gate with probability p (classical bit error)",
        )
        channels_layout.addWidget(self._channel_widgets["BitFlip"])

        # Phase Flip
        self._channel_widgets["PhaseFlip"] = _NoiseChannelWidget(
            "Phase Flip",
            "Applies Z gate with probability p (phase error)",
        )
        channels_layout.addWidget(self._channel_widgets["PhaseFlip"])

        # Depolarizing
        self._channel_widgets["Depolarizing"] = _NoiseChannelWidget(
            "Depolarizing",
            "Applies random Pauli (X, Y, or Z) with total probability p",
        )
        channels_layout.addWidget(self._channel_widgets["Depolarizing"])

        # Amplitude Damping
        self._channel_widgets["AmplitudeDamping"] = _NoiseChannelWidget(
            "Amplitude Damping",
            "Energy relaxation / T1 decay with damping rate gamma",
        )
        channels_layout.addWidget(self._channel_widgets["AmplitudeDamping"])

        layout.addWidget(channels_group)

        # Readout error group
        readout_group = QGroupBox("Readout Error")
        readout_layout = QFormLayout(readout_group)
        readout_layout.setHorizontalSpacing(12)

        self._readout_enable = QCheckBox("Enable readout error")
        self._readout_enable.toggled.connect(self._on_readout_toggled)
        readout_layout.addRow(self._readout_enable)

        self._p01_spin = QDoubleSpinBox()
        self._p01_spin.setRange(0.0, 0.5)
        self._p01_spin.setDecimals(4)
        self._p01_spin.setSingleStep(0.01)
        self._p01_spin.setValue(0.01)
        self._p01_spin.setEnabled(False)
        readout_layout.addRow("P(1|0):", self._p01_spin)

        self._p10_spin = QDoubleSpinBox()
        self._p10_spin.setRange(0.0, 0.5)
        self._p10_spin.setDecimals(4)
        self._p10_spin.setSingleStep(0.01)
        self._p10_spin.setValue(0.01)
        self._p10_spin.setEnabled(False)
        readout_layout.addRow("P(0|1):", self._p10_spin)

        layout.addWidget(readout_group)

        # Preview
        preview_group = QGroupBox("Preview")
        preview_layout = QVBoxLayout(preview_group)

        self._preview_text = QTextEdit()
        self._preview_text.setReadOnly(True)
        self._preview_text.setMaximumHeight(100)
        self._preview_text.setStyleSheet("font-family: monospace; font-size: 11px;")
        preview_layout.addWidget(self._preview_text)

        layout.addWidget(preview_group)

        # Update preview when any channel changes
        for widget in self._channel_widgets.values():
            widget._checkbox.toggled.connect(self._update_preview)
            widget._spinbox.valueChanged.connect(self._update_preview)
        self._readout_enable.toggled.connect(self._update_preview)
        self._p01_spin.valueChanged.connect(self._update_preview)
        self._p10_spin.valueChanged.connect(self._update_preview)
        self._update_preview()

        # Buttons
        button_layout = QHBoxLayout()

        clear_btn = QPushButton("Clear All")
        clear_btn.clicked.connect(self._clear_all)
        button_layout.addWidget(clear_btn)

        button_layout.addStretch()

        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        button_layout.addWidget(button_box)

        layout.addLayout(button_layout)

    def _on_readout_toggled(self, checked: bool):
        self._p01_spin.setEnabled(checked)
        self._p10_spin.setEnabled(checked)

    def _load_from_model(self, model: NoiseModel) -> None:
        """Load channel states from an existing NoiseModel."""
        data = model.to_dict()
        channel_type_map = {
            "BitFlipNoise": "BitFlip",
            "PhaseFlipNoise": "PhaseFlip",
            "DepolarizingNoise": "Depolarizing",
            "AmplitudeDampingNoise": "AmplitudeDamping",
        }
        for ch_data in data.get("global", []):
            ch_type = ch_data.get("type", "")
            key = channel_type_map.get(ch_type)
            if key and key in self._channel_widgets:
                self._channel_widgets[key].set_enabled(True)
                self._channel_widgets[key].set_probability(
                    ch_data.get("probability", 0.01)
                )
        if model.readout_error is not None:
            self._readout_enable.setChecked(True)
            self._p01_spin.setValue(model.readout_error.p01)
            self._p10_spin.setValue(model.readout_error.p10)

    def _update_preview(self):
        """Update the noise model preview text."""
        lines = []
        any_enabled = False
        for key, widget in self._channel_widgets.items():
            if widget.is_enabled:
                any_enabled = True
                lines.append(f"  {key}: p = {widget.probability:.4f}")
        if self._readout_enable.isChecked():
            any_enabled = True
            lines.append(
                f"  Readout: P(1|0)={self._p01_spin.value():.4f}, "
                f"P(0|1)={self._p10_spin.value():.4f}"
            )

        if any_enabled:
            self._preview_text.setPlainText(
                "Active noise channels:\n" + "\n".join(lines)
            )
        else:
            self._preview_text.setPlainText("No noise (ideal simulation)")

    def _clear_all(self):
        """Disable all noise channels."""
        for widget in self._channel_widgets.values():
            widget.set_enabled(False)
        self._readout_enable.setChecked(False)

    def get_noise_model(self) -> NoiseModel | None:
        """Build and return a NoiseModel from the current settings.

        Returns None if no channels are enabled.
        """
        channel_classes = {
            "BitFlip": BitFlipNoise,
            "PhaseFlip": PhaseFlipNoise,
            "Depolarizing": DepolarizingNoise,
            "AmplitudeDamping": AmplitudeDampingNoise,
        }

        any_enabled = False
        model = NoiseModel()

        for key, widget in self._channel_widgets.items():
            if widget.is_enabled:
                any_enabled = True
                cls = channel_classes[key]
                model.add_global_noise(cls(widget.probability))

        if self._readout_enable.isChecked():
            any_enabled = True
            model.set_readout_error(
                ReadoutError(p01=self._p01_spin.value(), p10=self._p10_spin.value())
            )

        return model if any_enabled else None

    @staticmethod
    def get_noise_config(
        current_model: NoiseModel | None = None,
        parent: QWidget | None = None,
    ) -> tuple[NoiseModel | None, bool]:
        """Static convenience method to show the dialog and return results.

        Returns:
            Tuple of (noise_model, accepted).
            noise_model is None if cancelled or no channels enabled.
        """
        dialog = NoiseConfigDialog(current_model, parent)
        result = dialog.exec()
        if result == QDialog.DialogCode.Accepted:
            return dialog.get_noise_model(), True
        return None, False

