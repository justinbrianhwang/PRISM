"""Dialog for selecting and configuring algorithm templates."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QListWidget, QListWidgetItem, QPushButton, QDialogButtonBox,
    QLabel, QSpinBox, QLineEdit, QComboBox, QWidget,
    QGroupBox, QStackedWidget, QTextEdit,
)

from quantum_sim.engine.algorithms import AlgorithmTemplate


class AlgorithmDialog(QDialog):
    """Dialog for selecting a quantum algorithm template and its parameters.

    Displays a list of available algorithm templates with descriptions
    and parameter inputs specific to each algorithm.
    """

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Algorithm Templates")
        self.setMinimumWidth(600)
        self.setMinimumHeight(450)

        self._selected_template: str | None = None
        self._template_params: dict = {}

        self._setup_ui()

    def _setup_ui(self):
        layout = QHBoxLayout(self)

        # Left panel: template list
        left_layout = QVBoxLayout()
        left_layout.addWidget(QLabel("<b>Algorithms</b>"))

        self._template_list = QListWidget()
        self._template_list.setMinimumWidth(200)
        self._template_list.currentRowChanged.connect(self._on_template_selected)
        left_layout.addWidget(self._template_list)

        layout.addLayout(left_layout)

        # Right panel: description + parameters
        right_layout = QVBoxLayout()

        # Description
        desc_group = QGroupBox("Description")
        desc_layout = QVBoxLayout(desc_group)
        self._description_label = QLabel("Select an algorithm to see its description.")
        self._description_label.setWordWrap(True)
        self._description_label.setStyleSheet("font-size: 12px; padding: 4px;")
        desc_layout.addWidget(self._description_label)
        right_layout.addWidget(desc_group)

        # Parameters (stacked widget for each algorithm)
        params_group = QGroupBox("Parameters")
        params_layout = QVBoxLayout(params_group)
        self._params_stack = QStackedWidget()
        params_layout.addWidget(self._params_stack)
        right_layout.addWidget(params_group)

        # Buttons
        button_layout = QHBoxLayout()

        self._preview_btn = QPushButton("Preview")
        self._preview_btn.setEnabled(False)
        self._preview_btn.clicked.connect(self._on_preview)
        button_layout.addWidget(self._preview_btn)

        button_layout.addStretch()

        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self._on_accept)
        button_box.rejected.connect(self.reject)
        button_layout.addWidget(button_box)

        right_layout.addLayout(button_layout)

        # Preview area
        self._preview_text = QTextEdit()
        self._preview_text.setReadOnly(True)
        self._preview_text.setMaximumHeight(100)
        self._preview_text.setStyleSheet("font-family: monospace; font-size: 10px;")
        self._preview_text.setVisible(False)
        right_layout.addWidget(self._preview_text)

        layout.addLayout(right_layout, stretch=1)

        # Populate templates
        self._populate_templates()

    def _populate_templates(self):
        """Fill the template list from AlgorithmTemplate.list_templates()."""
        templates = AlgorithmTemplate.list_templates()

        for tmpl in templates:
            item = QListWidgetItem(tmpl["display"])
            item.setData(Qt.ItemDataRole.UserRole, tmpl)
            self._template_list.addItem(item)

        # Create parameter widgets for each template
        self._param_widgets: dict[str, dict[str, QWidget]] = {}

        self._create_params_page("bell_state", [])  # No user params needed
        self._create_params_page("ghz_state", [
            ("num_qubits", "Number of qubits:", "spin", 3, 2, 16),
        ])
        self._create_params_page("qft", [
            ("num_qubits", "Number of qubits:", "spin", 3, 2, 10),
        ])
        self._create_params_page("inverse_qft", [
            ("num_qubits", "Number of qubits:", "spin", 3, 2, 10),
        ])
        self._create_params_page("grover", [
            ("num_qubits", "Number of qubits:", "spin", 3, 2, 8),
            ("marked_state", "Marked state (integer):", "spin", 0, 0, 255),
        ])
        self._create_params_page("deutsch_jozsa", [
            ("num_qubits", "Number of qubits:", "spin", 3, 2, 10),
            ("oracle_type", "Oracle type:", "combo", ["balanced", "constant"]),
        ])
        self._create_params_page("teleportation", [])  # Fixed 3 qubits
        self._create_params_page("bernstein_vazirani", [
            ("secret", "Secret bitstring:", "text", "101"),
        ])
        self._create_params_page("superdense_coding", [])  # Fixed 2 qubits

        # Select first item
        if self._template_list.count() > 0:
            self._template_list.setCurrentRow(0)

    def _create_params_page(
        self,
        template_name: str,
        params: list,
    ) -> None:
        """Create a parameter page for a specific template.

        params is a list of tuples:
            ("key", "Label:", "spin", default, min, max)
            ("key", "Label:", "text", default)
            ("key", "Label:", "combo", [options])
        """
        page = QWidget()
        form = QFormLayout(page)
        widgets: dict[str, QWidget] = {}

        if not params:
            no_params_label = QLabel("No configurable parameters.")
            no_params_label.setStyleSheet("color: gray;")
            form.addRow(no_params_label)
        else:
            for param_spec in params:
                key = param_spec[0]
                label = param_spec[1]
                widget_type = param_spec[2]

                if widget_type == "spin":
                    default, vmin, vmax = param_spec[3], param_spec[4], param_spec[5]
                    spin = QSpinBox()
                    spin.setRange(vmin, vmax)
                    spin.setValue(default)
                    spin.setMinimumWidth(80)
                    form.addRow(label, spin)
                    widgets[key] = spin

                elif widget_type == "text":
                    default = param_spec[3]
                    line_edit = QLineEdit(default)
                    line_edit.setMinimumWidth(100)
                    form.addRow(label, line_edit)
                    widgets[key] = line_edit

                elif widget_type == "combo":
                    options = param_spec[3]
                    combo = QComboBox()
                    combo.addItems(options)
                    combo.setMinimumWidth(100)
                    form.addRow(label, combo)
                    widgets[key] = combo

        self._param_widgets[template_name] = widgets
        self._params_stack.addWidget(page)

    def _on_template_selected(self, row: int):
        """Handle template selection."""
        if row < 0:
            return

        item = self._template_list.item(row)
        tmpl = item.data(Qt.ItemDataRole.UserRole)

        self._description_label.setText(tmpl["description"])
        self._preview_btn.setEnabled(True)
        self._preview_text.setVisible(False)

        # Map template display names to internal names
        template_names = [
            "bell_state", "ghz_state", "qft", "inverse_qft",
            "grover", "deutsch_jozsa", "teleportation",
            "bernstein_vazirani", "superdense_coding",
        ]
        if row < len(template_names):
            self._selected_template = template_names[row]
            # Switch to the corresponding params page
            self._params_stack.setCurrentIndex(row)

    def _get_template_kwargs(self) -> dict:
        """Collect parameter values from the current parameter page widgets."""
        if self._selected_template is None:
            return {}

        widgets = self._param_widgets.get(self._selected_template, {})
        kwargs = {}

        for key, widget in widgets.items():
            if isinstance(widget, QSpinBox):
                kwargs[key] = widget.value()
            elif isinstance(widget, QLineEdit):
                kwargs[key] = widget.text().strip()
            elif isinstance(widget, QComboBox):
                kwargs[key] = widget.currentText()

        return kwargs

    def _on_preview(self):
        """Show a preview of the circuit that would be generated."""
        if self._selected_template is None:
            return

        kwargs = self._get_template_kwargs()

        try:
            circuit = self._build_preview_circuit(self._selected_template, kwargs)
            if circuit is None:
                self._preview_text.setPlainText("Unable to generate preview.")
            else:
                lines = [
                    f"Qubits: {circuit.num_qubits}",
                    f"Gates: {circuit.gate_count()}",
                    f"Columns: {circuit.get_column_count()}",
                    "",
                    "Gate sequence:",
                ]
                for col_gates in circuit.get_ordered_gates():
                    for gate in col_gates:
                        params_str = ""
                        if gate.params:
                            params_str = f"({', '.join(f'{p:.3f}' for p in gate.params)})"
                        targets_str = ", ".join(f"q{q}" for q in gate.target_qubits)
                        lines.append(
                            f"  col {gate.column}: {gate.gate_name}{params_str} "
                            f"[{targets_str}]"
                        )
                self._preview_text.setPlainText("\n".join(lines))

        except Exception as exc:
            self._preview_text.setPlainText(f"Error: {exc}")

        self._preview_text.setVisible(True)

    def _build_preview_circuit(self, template_name: str, kwargs: dict):
        """Build a circuit from the template for preview."""
        builders = {
            "bell_state": lambda: AlgorithmTemplate.bell_state(),
            "ghz_state": lambda: AlgorithmTemplate.ghz_state(
                kwargs.get("num_qubits", 3)
            ),
            "qft": lambda: AlgorithmTemplate.quantum_fourier_transform(
                kwargs.get("num_qubits", 3)
            ),
            "inverse_qft": lambda: AlgorithmTemplate.inverse_qft(
                kwargs.get("num_qubits", 3)
            ),
            "grover": lambda: AlgorithmTemplate.grover_search(
                kwargs.get("num_qubits", 3),
                kwargs.get("marked_state", 0),
            ),
            "deutsch_jozsa": lambda: AlgorithmTemplate.deutsch_jozsa(
                kwargs.get("num_qubits", 3),
                kwargs.get("oracle_type", "balanced"),
            ),
            "teleportation": lambda: AlgorithmTemplate.quantum_teleportation(),
            "bernstein_vazirani": lambda: AlgorithmTemplate.bernstein_vazirani(
                kwargs.get("secret", "101")
            ),
            "superdense_coding": lambda: AlgorithmTemplate.superdense_coding(),
        }

        builder = builders.get(template_name)
        if builder is None:
            return None
        return builder()

    def _on_accept(self):
        """Accept and store the selected template info."""
        if self._selected_template is None:
            return
        self._template_params = self._get_template_kwargs()
        self.accept()

    def get_selected_template(self) -> str | None:
        """Return the name of the selected template."""
        return self._selected_template

    def get_template_params(self) -> dict:
        """Return the parameter values for the selected template."""
        return dict(self._template_params)

    @staticmethod
    def select_algorithm(
        parent: QWidget | None = None,
    ) -> tuple[str | None, dict, bool]:
        """Static convenience method.

        Returns:
            Tuple of (template_name, params_dict, accepted).
        """
        dialog = AlgorithmDialog(parent)
        result = dialog.exec()
        if result == QDialog.DialogCode.Accepted:
            return (
                dialog.get_selected_template(),
                dialog.get_template_params(),
                True,
            )
        return None, {}, False
