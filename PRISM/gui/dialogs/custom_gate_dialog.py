"""Dialog for defining custom quantum gates via matrix entry."""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QTableWidget, QTableWidgetItem, QPushButton, QDialogButtonBox,
    QLabel, QLineEdit, QComboBox, QWidget, QMessageBox,
    QHeaderView,
)

from quantum_sim.engine.gate_registry import GateRegistry
from quantum_sim.engine.gates import GateDefinition, GateType, _const


class CustomGateDialog(QDialog):
    """Dialog for creating a custom quantum gate by entering a unitary matrix.

    Supports 2x2 (single-qubit) and 4x4 (two-qubit) matrices.
    Validates unitarity before accepting.
    """

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)

        self.setWindowTitle("Define Custom Gate")
        self.setMinimumWidth(500)
        self.setMinimumHeight(400)

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Gate name
        name_layout = QFormLayout()
        self._name_input = QLineEdit()
        self._name_input.setPlaceholderText("e.g. MyGate")
        self._name_input.setMaxLength(20)
        name_layout.addRow("Gate name:", self._name_input)

        # Display name
        self._display_name_input = QLineEdit()
        self._display_name_input.setPlaceholderText("e.g. My Custom Gate")
        name_layout.addRow("Display name:", self._display_name_input)

        # Symbol (1-3 characters)
        self._symbol_input = QLineEdit()
        self._symbol_input.setPlaceholderText("e.g. MG")
        self._symbol_input.setMaxLength(4)
        name_layout.addRow("Symbol:", self._symbol_input)

        layout.addLayout(name_layout)

        # Matrix size selector
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("Matrix size:"))
        self._size_combo = QComboBox()
        self._size_combo.addItems(["2\u00D72 (1 qubit)", "4\u00D74 (2 qubits)"])
        self._size_combo.currentIndexChanged.connect(self._on_size_changed)
        size_layout.addWidget(self._size_combo)

        # Preset buttons
        size_layout.addStretch()
        identity_btn = QPushButton("Identity")
        identity_btn.clicked.connect(self._fill_identity)
        size_layout.addWidget(identity_btn)

        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self._clear_matrix)
        size_layout.addWidget(clear_btn)

        layout.addLayout(size_layout)

        # Matrix entry instructions
        instruction_label = QLabel(
            "Enter complex numbers as <i>a+bj</i> or <i>a</i> "
            "(e.g., <code>0.707+0.707j</code>, <code>1</code>, <code>1j</code>)"
        )
        instruction_label.setStyleSheet("color: gray; font-size: 11px;")
        layout.addWidget(instruction_label)

        # Matrix table
        self._matrix_table = QTableWidget()
        self._matrix_table.setMinimumHeight(150)
        self._setup_matrix_table(2)
        layout.addWidget(self._matrix_table)

        # Validation result
        self._validation_label = QLabel("")
        self._validation_label.setStyleSheet("font-size: 11px;")
        layout.addWidget(self._validation_label)

        # Validate button
        validate_btn = QPushButton("Validate Unitarity")
        validate_btn.clicked.connect(self._validate_matrix)
        layout.addWidget(validate_btn)

        # OK / Cancel
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self._on_accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _setup_matrix_table(self, size: int):
        """Set up the matrix table for the given size."""
        self._matrix_table.setRowCount(size)
        self._matrix_table.setColumnCount(size)

        labels = [str(i) for i in range(size)]
        self._matrix_table.setHorizontalHeaderLabels(labels)
        self._matrix_table.setVerticalHeaderLabels(labels)

        header = self._matrix_table.horizontalHeader()
        for col in range(size):
            header.setSectionResizeMode(col, QHeaderView.ResizeMode.Stretch)

        # Fill with zeros
        for row in range(size):
            for col in range(size):
                item = QTableWidgetItem("0")
                item.setTextAlignment(
                    Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter
                )
                self._matrix_table.setItem(row, col, item)

    def _on_size_changed(self, index: int):
        """Handle matrix size change."""
        size = 2 if index == 0 else 4
        self._setup_matrix_table(size)
        self._validation_label.setText("")

    def _fill_identity(self):
        """Fill the matrix with the identity."""
        size = self._matrix_table.rowCount()
        for row in range(size):
            for col in range(size):
                val = "1" if row == col else "0"
                item = self._matrix_table.item(row, col)
                if item is None:
                    item = QTableWidgetItem(val)
                    item.setTextAlignment(
                        Qt.AlignmentFlag.AlignCenter
                        | Qt.AlignmentFlag.AlignVCenter
                    )
                    self._matrix_table.setItem(row, col, item)
                else:
                    item.setText(val)

    def _clear_matrix(self):
        """Clear all matrix entries to zero."""
        size = self._matrix_table.rowCount()
        for row in range(size):
            for col in range(size):
                item = self._matrix_table.item(row, col)
                if item is None:
                    item = QTableWidgetItem("0")
                    item.setTextAlignment(
                        Qt.AlignmentFlag.AlignCenter
                        | Qt.AlignmentFlag.AlignVCenter
                    )
                    self._matrix_table.setItem(row, col, item)
                else:
                    item.setText("0")

    def _parse_matrix(self) -> np.ndarray | None:
        """Parse the table contents into a numpy complex matrix.

        Returns None if any cell cannot be parsed.
        """
        size = self._matrix_table.rowCount()
        matrix = np.zeros((size, size), dtype=np.complex128)

        for row in range(size):
            for col in range(size):
                item = self._matrix_table.item(row, col)
                text = item.text().strip() if item else "0"
                if not text:
                    text = "0"
                try:
                    value = complex(text.replace(" ", ""))
                    matrix[row, col] = value
                except ValueError:
                    self._validation_label.setText(
                        f"<span style='color: red;'>Invalid entry at "
                        f"row {row}, col {col}: '{text}'</span>"
                    )
                    return None

        return matrix

    def _is_unitary(self, matrix: np.ndarray, tolerance: float = 1e-6) -> bool:
        """Check if a matrix is unitary: U * U^dagger = I."""
        product = matrix @ matrix.conj().T
        identity = np.eye(matrix.shape[0], dtype=np.complex128)
        return np.allclose(product, identity, atol=tolerance)

    def _validate_matrix(self) -> bool:
        """Validate that the entered matrix is unitary."""
        matrix = self._parse_matrix()
        if matrix is None:
            return False

        if self._is_unitary(matrix):
            self._validation_label.setText(
                "<span style='color: green;'>"
                "\u2713 Matrix is unitary</span>"
            )
            return True
        else:
            # Show how far from unitary it is
            product = matrix @ matrix.conj().T
            identity = np.eye(matrix.shape[0], dtype=np.complex128)
            error = np.max(np.abs(product - identity))
            self._validation_label.setText(
                f"<span style='color: red;'>"
                f"\u2717 Matrix is NOT unitary "
                f"(max error: {error:.6e})</span>"
            )
            return False

    def _on_accept(self):
        """Validate inputs and accept the dialog."""
        # Validate name
        name = self._name_input.text().strip()
        if not name:
            QMessageBox.warning(self, "Validation Error", "Gate name is required.")
            return

        if not name.isidentifier():
            QMessageBox.warning(
                self, "Validation Error",
                "Gate name must be a valid identifier "
                "(letters, digits, underscores; cannot start with a digit)."
            )
            return

        # Check if name already exists
        registry = GateRegistry.instance()
        if name in registry.gate_names():
            QMessageBox.warning(
                self, "Validation Error",
                f"A gate named '{name}' already exists in the registry."
            )
            return

        # Validate matrix
        matrix = self._parse_matrix()
        if matrix is None:
            return

        if not self._is_unitary(matrix):
            result = QMessageBox.question(
                self, "Non-Unitary Matrix",
                "The matrix is not unitary. Quantum gates must be unitary.\n\n"
                "Do you want to register it anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if result != QMessageBox.StandardButton.Yes:
                return

        # Register the gate
        display_name = self._display_name_input.text().strip() or name
        symbol = self._symbol_input.text().strip() or name[:3]
        size = self._matrix_table.rowCount()
        num_qubits = 1 if size == 2 else 2

        gate_def = GateDefinition(
            name=name,
            display_name=display_name,
            gate_type=GateType.SINGLE if num_qubits == 1 else GateType.MULTI,
            num_qubits=num_qubits,
            num_params=0,
            param_names=(),
            matrix_func=_const(matrix.copy()),
            symbol=symbol,
            color="#FF6B6B",  # Custom gate color
            num_controls=0,
            num_targets=num_qubits,
        )

        registry.register(gate_def)
        self.accept()

    def get_gate_name(self) -> str:
        """Return the name of the registered gate."""
        return self._name_input.text().strip()
