"""Simple about dialog for the Quantum Circuit Simulator."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QPushButton, QWidget,
)


class AboutDialog(QDialog):
    """Simple About dialog showing application name, version, and description."""

    APP_NAME = "Quantum Circuit Simulator"
    APP_VERSION = "1.0.0"
    APP_DESCRIPTION = (
        "An interactive quantum circuit simulator with a visual circuit editor, "
        "state vector visualization, Bloch sphere display, measurement histograms, "
        "and support for common quantum algorithms.\n\n"
        "Built with PyQt6 and NumPy."
    )

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("About")
        self.setFixedSize(420, 300)

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(12)

        # Application name
        name_label = QLabel(self.APP_NAME)
        name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        name_label.setStyleSheet("font-size: 20px; font-weight: bold;")
        layout.addWidget(name_label)

        # Version
        version_label = QLabel(f"Version {self.APP_VERSION}")
        version_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        version_label.setStyleSheet("font-size: 13px; color: gray;")
        layout.addWidget(version_label)

        # Separator
        separator = QLabel()
        separator.setFixedHeight(1)
        separator.setStyleSheet("background-color: #555555;")
        layout.addWidget(separator)

        # Description
        desc_label = QLabel(self.APP_DESCRIPTION)
        desc_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("font-size: 12px; padding: 8px;")
        layout.addWidget(desc_label)

        layout.addStretch()

        # Close button
        close_btn = QPushButton("Close")
        close_btn.setFixedWidth(100)
        close_btn.clicked.connect(self.accept)

        btn_layout = QVBoxLayout()
        btn_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)
