"""QGraphicsItem subclass for qubit wires in the circuit editor."""

from __future__ import annotations

from PyQt6.QtWidgets import QGraphicsItem, QGraphicsTextItem, QStyleOptionGraphicsItem, QWidget
from PyQt6.QtGui import QPen, QColor, QFont, QPainter
from PyQt6.QtCore import QRectF, Qt

# Layout constants
GRID_SIZE = 60
WIRE_Y_SPACING = 60
WIRE_START_X = 120


class WireItem(QGraphicsItem):
    """A horizontal qubit wire with label 'q{index}: |state>' on the left.

    The wire extends from WIRE_START_X to WIRE_START_X + wire_length.
    The label is drawn to the left of the wire start.
    Clicking the label toggles the initial qubit state between |0> and |1>.
    """

    def __init__(
        self,
        qubit_index: int,
        initial_state: int = 0,
        wire_length: float = 600.0,
        parent: QGraphicsItem | None = None,
    ):
        super().__init__(parent)
        self._qubit_index = qubit_index
        self._initial_state = initial_state
        self._wire_length = wire_length

        # Wire appearance
        self._wire_color = QColor("#9E9E9E")  # light gray for dark theme
        self._label_color = QColor("#CCCCCC")
        self._pen = QPen(self._wire_color, 1.5, Qt.PenStyle.SolidLine)

        # Build the label text
        self._label = f"q{qubit_index}: |{initial_state}\u27E9"
        self._font = QFont("Segoe UI", 11)

        # Position: the wire sits at y = qubit_index * WIRE_Y_SPACING
        self.setPos(0.0, qubit_index * WIRE_Y_SPACING)

        # Label area is clickable for state toggle
        self.setAcceptedMouseButtons(Qt.MouseButton.LeftButton)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, False)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, False)

    # ---- Properties -------------------------------------------------------

    @property
    def qubit_index(self) -> int:
        return self._qubit_index

    @property
    def initial_state(self) -> int:
        return self._initial_state

    @initial_state.setter
    def initial_state(self, value: int) -> None:
        self._initial_state = value
        self._label = f"q{self._qubit_index}: |{value}\u27E9"
        self.update()

    @property
    def wire_length(self) -> float:
        return self._wire_length

    @wire_length.setter
    def wire_length(self, value: float) -> None:
        self.prepareGeometryChange()
        self._wire_length = value
        self.update()

    # ---- QGraphicsItem interface ------------------------------------------

    def boundingRect(self) -> QRectF:  # noqa: N802
        # Include the label area to the left and some padding
        label_width = WIRE_START_X - 10
        return QRectF(
            0,
            -15,
            WIRE_START_X + self._wire_length + 10,
            30,
        )

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: QWidget | None = None,
    ) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        # --- Draw the label ---------------------------------------------------
        painter.setFont(self._font)
        painter.setPen(QPen(self._label_color))
        label_rect = QRectF(5, -12, WIRE_START_X - 15, 24)
        painter.drawText(
            label_rect,
            int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter),
            self._label,
        )

        # --- Draw the horizontal wire line ------------------------------------
        painter.setPen(self._pen)
        y = 0.0
        painter.drawLine(
            int(WIRE_START_X), int(y),
            int(WIRE_START_X + self._wire_length), int(y),
        )

    def mousePressEvent(self, event) -> None:
        """Toggle initial state when clicking on the label area."""
        if event.pos().x() < WIRE_START_X:
            self._initial_state = 1 - self._initial_state
            self._label = f"q{self._qubit_index}: |{self._initial_state}\u27E9"
            self.update()
            # Scene handles the signal via its mousePressEvent
        else:
            super().mousePressEvent(event)

    def set_wire_color(self, color: QColor) -> None:
        """Update wire color (useful for theme changes)."""
        self._wire_color = color
        self._pen = QPen(self._wire_color, 1.5, Qt.PenStyle.SolidLine)
        self.update()

    def set_label_color(self, color: QColor) -> None:
        """Update label text color (useful for theme changes)."""
        self._label_color = color
        self.update()

    def __repr__(self) -> str:
        return f"WireItem(qubit={self._qubit_index})"
