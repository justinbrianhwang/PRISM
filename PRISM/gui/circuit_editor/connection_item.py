"""Simple QGraphicsLineItem for connecting multi-qubit gate parts.

Used to draw auxiliary vertical or horizontal lines between the separate
visual components of a multi-qubit gate (e.g. the control dot and the
target symbol of a CNOT placed far apart).
"""

from __future__ import annotations

from PyQt6.QtWidgets import QGraphicsLineItem, QGraphicsItem
from PyQt6.QtGui import QPen, QColor
from PyQt6.QtCore import QLineF, QPointF


class ConnectionItem(QGraphicsLineItem):
    """A cosmetic line that visually connects parts of a multi-qubit gate.

    The line is not selectable or interactive -- it simply tracks the
    positions of the two endpoints it connects.

    Parameters
    ----------
    x1, y1 : float
        Start point in scene coordinates.
    x2, y2 : float
        End point in scene coordinates.
    color : QColor | str
        Line colour (default light grey).
    width : float
        Pen width (default 2.0).
    parent : QGraphicsItem | None
        Optional parent item.
    """

    def __init__(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        color: QColor | str = "#CCCCCC",
        width: float = 2.0,
        parent: QGraphicsItem | None = None,
    ):
        super().__init__(QLineF(x1, y1, x2, y2), parent)

        if isinstance(color, str):
            color = QColor(color)

        pen = QPen(color, width)
        self.setPen(pen)

        # Not interactive
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, False)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, False)
        self.setAcceptedMouseButtons(0)  # type: ignore[arg-type]

    # --- Convenience methods -----------------------------------------------

    def update_endpoints(
        self,
        x1: float, y1: float,
        x2: float, y2: float,
    ) -> None:
        """Move both endpoints (e.g. after a gate has been repositioned)."""
        self.setLine(QLineF(x1, y1, x2, y2))

    def set_color(self, color: QColor | str) -> None:
        if isinstance(color, str):
            color = QColor(color)
        pen = self.pen()
        pen.setColor(color)
        self.setPen(pen)

    def set_width(self, width: float) -> None:
        pen = self.pen()
        pen.setWidthF(width)
        self.setPen(pen)

    @classmethod
    def between_points(
        cls,
        p1: QPointF,
        p2: QPointF,
        color: QColor | str = "#CCCCCC",
        width: float = 2.0,
        parent: QGraphicsItem | None = None,
    ) -> ConnectionItem:
        """Create a ConnectionItem from two QPointF objects."""
        return cls(p1.x(), p1.y(), p2.x(), p2.y(), color, width, parent)

    def __repr__(self) -> str:
        line = self.line()
        return (
            f"ConnectionItem(({line.x1():.0f},{line.y1():.0f}) -> "
            f"({line.x2():.0f},{line.y2():.0f}))"
        )
