"""State vector visualization panel with amplitude, phase, and probability display."""

from __future__ import annotations

import math

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QPainter, QBrush
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QCheckBox, QHeaderView, QLabel, QStyledItemDelegate, QStyleOptionViewItem,
    QStyle,
)

from quantum_sim.engine.state_vector import StateVector


class ProbabilityBarDelegate(QStyledItemDelegate):
    """Custom delegate that draws a color-coded probability bar in a table cell."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._bar_color = QColor(76, 175, 80)  # Green

    def set_bar_color(self, color: QColor):
        self._bar_color = color

    def paint(self, painter: QPainter, option: QStyleOptionViewItem,
              index) -> None:
        # Draw the default background
        self.initStyleOption(option, index)
        painter.save()

        # Draw selection highlight if selected
        if option.state & QStyle.StateFlag.State_Selected:
            painter.fillRect(option.rect, option.palette.highlight())

        # Get probability value from the item data
        value = index.data(Qt.ItemDataRole.UserRole)
        if value is not None and isinstance(value, (int, float)):
            probability = float(value)
            text = index.data(Qt.ItemDataRole.DisplayRole) or ""

            # Draw the bar
            rect = option.rect.adjusted(4, 3, -4, -3)
            bar_width = int(rect.width() * min(probability, 1.0))
            if bar_width > 0:
                bar_rect = rect.adjusted(0, 0, -(rect.width() - bar_width), 0)
                bar_color = QColor(self._bar_color)
                bar_color.setAlpha(120)
                painter.fillRect(bar_rect, QBrush(bar_color))

            # Draw text on top
            painter.setPen(option.palette.text().color())
            painter.drawText(
                option.rect.adjusted(6, 0, -4, 0),
                Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                text,
            )
        else:
            # Fallback to default painting
            super().paint(painter, option, index)

        painter.restore()


class StateVectorPanel(QWidget):
    """Panel displaying the quantum state vector in a table format.

    Shows each basis state with its amplitude (a + bi), phase in degrees,
    and probability as a percentage with color-coded bars.
    """

    ZERO_THRESHOLD = 1e-10

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)

        self._state_vector: StateVector | None = None

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Header
        header_layout = QHBoxLayout()
        title_label = QLabel("State Vector")
        title_label.setStyleSheet("font-weight: bold; font-size: 13px;")
        header_layout.addWidget(title_label)
        header_layout.addStretch()

        self._hide_zero_cb = QCheckBox("Hide zero-amplitude states")
        self._hide_zero_cb.setChecked(False)
        self._hide_zero_cb.toggled.connect(self._refresh_table)
        header_layout.addWidget(self._hide_zero_cb)

        layout.addLayout(header_layout)

        # Table
        self._table = QTableWidget()
        self._table.setColumnCount(4)
        self._table.setHorizontalHeaderLabels(
            ["State", "Amplitude", "Phase (deg)", "Probability"]
        )
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setAlternatingRowColors(True)
        self._table.verticalHeader().setVisible(False)

        header = self._table.horizontalHeader()
        header.setStretchLastSection(True)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)

        # Set probability bar delegate on column 3
        self._prob_delegate = ProbabilityBarDelegate(self._table)
        self._table.setItemDelegateForColumn(3, self._prob_delegate)

        layout.addWidget(self._table)

        # Info label
        self._info_label = QLabel("")
        self._info_label.setStyleSheet("color: gray; font-size: 11px;")
        layout.addWidget(self._info_label)

    def update_state(self, state_vector: StateVector) -> None:
        """Update the displayed state vector."""
        self._state_vector = state_vector
        self._refresh_table()

    def _refresh_table(self) -> None:
        """Rebuild the table from the current state vector."""
        if self._state_vector is None:
            self._table.setRowCount(0)
            self._info_label.setText("")
            return

        sv = self._state_vector
        num_qubits = sv.num_qubits
        data = sv.data
        probabilities = sv.probabilities
        hide_zero = self._hide_zero_cb.isChecked()

        # Collect visible rows
        rows: list[tuple[int, complex, float, float]] = []
        for idx in range(len(data)):
            amplitude = data[idx]
            prob = probabilities[idx]
            if hide_zero and abs(amplitude) < self.ZERO_THRESHOLD:
                continue
            phase_rad = math.atan2(amplitude.imag, amplitude.real)
            phase_deg = math.degrees(phase_rad)
            rows.append((idx, amplitude, phase_deg, prob))

        self._table.setRowCount(len(rows))

        for row_idx, (basis_idx, amplitude, phase_deg, prob) in enumerate(rows):
            # State column: ket notation
            bitstring = format(basis_idx, f"0{num_qubits}b")
            state_item = QTableWidgetItem(f"|{bitstring}\u27E9")
            state_item.setTextAlignment(
                Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter
            )
            state_item.setFont(self._table.font())
            self._table.setItem(row_idx, 0, state_item)

            # Amplitude column: a + bi with 4 decimal places
            amp_str = self._format_amplitude(amplitude)
            amp_item = QTableWidgetItem(amp_str)
            amp_item.setTextAlignment(
                Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter
            )
            self._table.setItem(row_idx, 1, amp_item)

            # Phase column
            if abs(amplitude) < self.ZERO_THRESHOLD:
                phase_str = "-"
            else:
                phase_str = f"{phase_deg:.1f}\u00B0"
            phase_item = QTableWidgetItem(phase_str)
            phase_item.setTextAlignment(
                Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter
            )
            self._table.setItem(row_idx, 2, phase_item)

            # Probability column (with custom delegate bar)
            prob_pct = prob * 100.0
            prob_str = f"{prob_pct:.2f}%"
            prob_item = QTableWidgetItem(prob_str)
            prob_item.setData(Qt.ItemDataRole.UserRole, prob)
            prob_item.setTextAlignment(
                Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft
            )
            self._table.setItem(row_idx, 3, prob_item)

        total_states = 2 ** num_qubits
        visible = len(rows)
        self._info_label.setText(
            f"{num_qubits} qubits | {visible}/{total_states} states shown"
        )

    @staticmethod
    def _format_amplitude(c: complex) -> str:
        """Format a complex number as 'a + bi' with 4 decimal places."""
        real = c.real
        imag = c.imag

        # Handle near-zero values
        if abs(real) < 1e-10:
            real = 0.0
        if abs(imag) < 1e-10:
            imag = 0.0

        if imag == 0.0 and real == 0.0:
            return "0.0000"
        elif imag == 0.0:
            return f"{real:.4f}"
        elif real == 0.0:
            return f"{imag:.4f}i"
        else:
            if imag >= 0:
                return f"{real:.4f} + {imag:.4f}i"
            else:
                return f"{real:.4f} - {abs(imag):.4f}i"

    def clear(self) -> None:
        """Clear the panel."""
        self._state_vector = None
        self._table.setRowCount(0)
        self._info_label.setText("")
