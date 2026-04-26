"""QGraphicsItem subclasses for quantum gates in the circuit editor.

Provides visual representations for:
- Single-qubit gates (H, X, Y, Z, S, T, Rx, Ry, Rz, etc.)
- Controlled gates (CNOT, CZ, Toffoli, Fredkin)
- SWAP gates
- Measurement symbols
- Barrier lines
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from PyQt6.QtWidgets import (
    QGraphicsItem,
    QGraphicsSceneMouseEvent,
    QStyleOptionGraphicsItem,
    QWidget,
)
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QPainter,
    QPainterPath,
    QPen,
)
from PyQt6.QtCore import QPointF, QRectF, Qt

from quantum_sim.engine.circuit import GateInstance
from quantum_sim.engine.gates import GateDefinition, GateType
from quantum_sim.engine.gate_registry import GateRegistry

if TYPE_CHECKING:
    from quantum_sim.gui.circuit_editor.scene import CircuitScene

# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------
GRID_SIZE = 60
WIRE_Y_SPACING = 60
GATE_WIDTH = 50
WIRE_START_X = 120

# Gate box dimensions (centred inside GRID_SIZE cell)
GATE_BOX = 40  # width & height of the painted rectangle


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cell_x(column: int) -> float:
    """Return the scene x-coordinate for a circuit column."""
    return WIRE_START_X + column * GRID_SIZE + (GRID_SIZE - GATE_WIDTH) / 2


def _cell_y(qubit_index: int) -> float:
    """Return the scene y-coordinate for a qubit wire row."""
    return qubit_index * WIRE_Y_SPACING


def _snap_to_grid(pos: QPointF) -> tuple[int, int]:
    """Convert a scene position to the nearest (column, qubit_index)."""
    col = max(0, round((pos.x() - WIRE_START_X - (GRID_SIZE - GATE_WIDTH) / 2) / GRID_SIZE))
    qubit = max(0, round(pos.y() / WIRE_Y_SPACING))
    return col, qubit


# =========================================================================
# Base class
# =========================================================================

class GateGraphicsItem(QGraphicsItem):
    """Base visual item for a quantum gate placed on the circuit.

    Stores the associated *GateInstance* (circuit-model object) and the
    *GateDefinition* from the registry.  Supports selection, movement with
    grid-snapping, and double-click for parameter editing.
    """

    def __init__(
        self,
        gate_instance: GateInstance,
        gate_def: GateDefinition,
        parent: QGraphicsItem | None = None,
    ):
        super().__init__(parent)
        self._gate_instance = gate_instance
        self._gate_def = gate_def
        self._scene_ref: CircuitScene | None = None

        # Make selectable and movable
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)

        # Position in scene based on column / first target qubit
        self._sync_position()

    # ---- Properties -------------------------------------------------------

    @property
    def gate_instance(self) -> GateInstance:
        return self._gate_instance

    @property
    def gate_def(self) -> GateDefinition:
        return self._gate_def

    @property
    def circuit_scene(self) -> CircuitScene | None:
        return self._scene_ref

    @circuit_scene.setter
    def circuit_scene(self, scene: CircuitScene | None) -> None:
        self._scene_ref = scene

    # ---- Positioning ------------------------------------------------------

    def _sync_position(self) -> None:
        """Move item to the scene position derived from gate_instance."""
        col = self._gate_instance.column
        qubit = self._gate_instance.target_qubits[0] if self._gate_instance.target_qubits else 0
        # Center the gate vertically on the wire (wire is at _cell_y, gate height is GATE_WIDTH)
        self.setPos(_cell_x(col), _cell_y(qubit) - GATE_WIDTH / 2)

    def refresh_position(self) -> None:
        """Public method to re-sync after model changes."""
        self._sync_position()

    # ---- QGraphicsItem interface ------------------------------------------

    def boundingRect(self) -> QRectF:  # noqa: N802
        return QRectF(0, 0, GATE_WIDTH, GATE_WIDTH)

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: QWidget | None = None,
    ) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        # Background rounded rectangle
        color = QColor(self._gate_def.color)
        painter.setBrush(QBrush(color))
        pen = QPen(QColor("#FFFFFF"), 1.5)
        if self.isSelected():
            pen = QPen(QColor("#FFD700"), 2.5)  # gold highlight
        painter.setPen(pen)

        margin = (GATE_WIDTH - GATE_BOX) / 2
        painter.drawRoundedRect(
            QRectF(margin, margin, GATE_BOX, GATE_BOX), 6, 6,
        )

        # Symbol text
        painter.setPen(QPen(QColor("#FFFFFF")))
        font = QFont("Segoe UI", 13, QFont.Weight.Bold)
        painter.setFont(font)
        painter.drawText(
            QRectF(margin, margin, GATE_BOX, GATE_BOX),
            int(Qt.AlignmentFlag.AlignCenter),
            self._gate_def.symbol,
        )

    # ---- Snapping & interaction -------------------------------------------

    def itemChange(self, change, value):  # noqa: N802
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange and self.scene():
            new_pos: QPointF = value
            # Compensate for the -GATE_WIDTH/2 offset to find the true wire position
            adjusted = QPointF(new_pos.x(), new_pos.y() + GATE_WIDTH / 2)
            col, qubit = _snap_to_grid(adjusted)
            snapped = QPointF(_cell_x(col), _cell_y(qubit) - GATE_WIDTH / 2)

            # Update the underlying model when dragged
            old_col = self._gate_instance.column
            old_qubits = list(self._gate_instance.target_qubits)
            if col != old_col or (old_qubits and qubit != old_qubits[0]):
                offset = qubit - old_qubits[0] if old_qubits else 0
                self._gate_instance.column = col
                self._gate_instance.target_qubits = [q + offset for q in old_qubits]
                if self._scene_ref is not None:
                    self._scene_ref.circuit_changed.emit()
            return snapped
        if change == QGraphicsItem.GraphicsItemChange.ItemSelectedHasChanged:
            if self._scene_ref is not None and value:
                self._scene_ref.gate_selected.emit(self._gate_instance)
        return super().itemChange(change, value)

    def mouseDoubleClickEvent(self, event: QGraphicsSceneMouseEvent) -> None:  # noqa: N802
        """Emit a signal so the application can open a parameter editor."""
        if self._scene_ref is not None:
            self._scene_ref.gate_double_clicked.emit(self._gate_instance)
        super().mouseDoubleClickEvent(event)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"{self._gate_def.name}, col={self._gate_instance.column}, "
            f"qubits={self._gate_instance.target_qubits})"
        )


# =========================================================================
# SingleQubitGateItem
# =========================================================================

class SingleQubitGateItem(GateGraphicsItem):
    """Visual for any single-qubit gate (H, X, Y, Z, S, T, Rx, ...)."""

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: QWidget | None = None,
    ) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        color = QColor(self._gate_def.color)
        margin = (GATE_WIDTH - GATE_BOX) / 2

        # Fill
        painter.setBrush(QBrush(color))
        pen = QPen(QColor("#FFFFFF"), 1.5)
        if self.isSelected():
            pen = QPen(QColor("#FFD700"), 2.5)
        painter.setPen(pen)
        painter.drawRoundedRect(QRectF(margin, margin, GATE_BOX, GATE_BOX), 6, 6)

        # Symbol
        painter.setPen(QPen(QColor("#FFFFFF")))
        font = QFont("Segoe UI", 13, QFont.Weight.Bold)
        painter.setFont(font)
        painter.drawText(
            QRectF(margin, margin, GATE_BOX, GATE_BOX),
            int(Qt.AlignmentFlag.AlignCenter),
            self._gate_def.symbol,
        )


# =========================================================================
# ControlledGateItem
# =========================================================================

class ControlledGateItem(GateGraphicsItem):
    """Visual for controlled gates: CNOT, CZ, Toffoli, Fredkin.

    Draws vertical connecting lines, control dots, and target symbols
    spanning the appropriate qubit rows.
    """

    def boundingRect(self) -> QRectF:  # noqa: N802
        qubits = self._gate_instance.target_qubits
        if len(qubits) < 2:
            return QRectF(0, 0, GATE_WIDTH, GATE_WIDTH)
        min_q = min(qubits)
        max_q = max(qubits)
        first_q = qubits[0]
        # Item is at _cell_y(first_q) - GATE_WIDTH/2, so local y=GATE_WIDTH/2 is the wire
        top = (min_q - first_q) * WIRE_Y_SPACING + GATE_WIDTH / 2 - 10
        bottom = (max_q - first_q) * WIRE_Y_SPACING + GATE_WIDTH / 2 + 10
        return QRectF(-5, top, GATE_WIDTH + 10, bottom - top + 20)

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: QWidget | None = None,
    ) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        qubits = self._gate_instance.target_qubits
        first_q = qubits[0]
        name = self._gate_def.name
        num_controls = self._gate_def.num_controls

        # Centre x within the cell
        cx = GATE_WIDTH / 2

        # Helper: relative y for a qubit index
        # Item positioned at _cell_y(first_q) - GATE_WIDTH/2, so wire of first_q is at local y=GATE_WIDTH/2
        def rel_y(q: int) -> float:
            return (q - first_q) * WIRE_Y_SPACING + GATE_WIDTH / 2

        min_q = min(qubits)
        max_q = max(qubits)

        # --- Vertical connecting line ---
        line_pen = QPen(QColor("#CCCCCC"), 2.0)
        if self.isSelected():
            line_pen = QPen(QColor("#FFD700"), 2.5)
        painter.setPen(line_pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawLine(
            QPointF(cx, rel_y(min_q)),
            QPointF(cx, rel_y(max_q)),
        )

        # Separate control and target qubits
        control_qubits = qubits[:num_controls]
        target_qubits = qubits[num_controls:]

        # --- Draw control dots ---
        control_radius = 6
        painter.setBrush(QBrush(QColor("#CCCCCC")))
        painter.setPen(QPen(QColor("#CCCCCC"), 1.5))
        for cq in control_qubits:
            painter.drawEllipse(
                QPointF(cx, rel_y(cq)),
                control_radius, control_radius,
            )

        # --- Draw targets ---
        if name == "CNOT":
            # Target: circled plus
            tq = target_qubits[0]
            ty = rel_y(tq)
            r = 14
            painter.setBrush(Qt.BrushStyle.NoBrush)
            pen = QPen(QColor("#CCCCCC"), 2.0)
            if self.isSelected():
                pen = QPen(QColor("#FFD700"), 2.5)
            painter.setPen(pen)
            painter.drawEllipse(QPointF(cx, ty), r, r)
            # Plus sign inside circle
            painter.drawLine(QPointF(cx - r, ty), QPointF(cx + r, ty))
            painter.drawLine(QPointF(cx, ty - r), QPointF(cx, ty + r))

        elif name == "CZ":
            # Target: Z label in a small box
            tq = target_qubits[0]
            ty = rel_y(tq)
            box_size = 28
            margin = (GATE_WIDTH - box_size) / 2
            color = QColor(self._gate_def.color)
            painter.setBrush(QBrush(color))
            pen = QPen(QColor("#FFFFFF"), 1.5)
            if self.isSelected():
                pen = QPen(QColor("#FFD700"), 2.5)
            painter.setPen(pen)
            painter.drawRoundedRect(
                QRectF(margin, ty - box_size / 2, box_size, box_size), 4, 4,
            )
            painter.setPen(QPen(QColor("#FFFFFF")))
            font = QFont("Segoe UI", 12, QFont.Weight.Bold)
            painter.setFont(font)
            painter.drawText(
                QRectF(margin, ty - box_size / 2, box_size, box_size),
                int(Qt.AlignmentFlag.AlignCenter),
                "Z",
            )

        elif name == "Toffoli":
            # Two control dots already drawn; target: circled plus
            tq = target_qubits[0]
            ty = rel_y(tq)
            r = 14
            painter.setBrush(Qt.BrushStyle.NoBrush)
            pen = QPen(QColor("#CCCCCC"), 2.0)
            if self.isSelected():
                pen = QPen(QColor("#FFD700"), 2.5)
            painter.setPen(pen)
            painter.drawEllipse(QPointF(cx, ty), r, r)
            painter.drawLine(QPointF(cx - r, ty), QPointF(cx + r, ty))
            painter.drawLine(QPointF(cx, ty - r), QPointF(cx, ty + r))

        elif name == "Fredkin":
            # One control dot already drawn; two X marks at targets
            x_size = 8
            pen = QPen(QColor("#CCCCCC"), 2.5)
            if self.isSelected():
                pen = QPen(QColor("#FFD700"), 2.5)
            painter.setPen(pen)
            for tq in target_qubits:
                ty = rel_y(tq)
                painter.drawLine(
                    QPointF(cx - x_size, ty - x_size),
                    QPointF(cx + x_size, ty + x_size),
                )
                painter.drawLine(
                    QPointF(cx - x_size, ty + x_size),
                    QPointF(cx + x_size, ty - x_size),
                )

    def itemChange(self, change, value):  # noqa: N802
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange and self.scene():
            new_pos: QPointF = value
            adjusted = QPointF(new_pos.x(), new_pos.y() + GATE_WIDTH / 2)
            col, qubit = _snap_to_grid(adjusted)
            snapped = QPointF(_cell_x(col), _cell_y(qubit) - GATE_WIDTH / 2)

            old_col = self._gate_instance.column
            old_qubits = list(self._gate_instance.target_qubits)
            if col != old_col or (old_qubits and qubit != old_qubits[0]):
                offset = qubit - old_qubits[0] if old_qubits else 0
                self._gate_instance.column = col
                self._gate_instance.target_qubits = [q + offset for q in old_qubits]
                if self._scene_ref is not None:
                    self._scene_ref.circuit_changed.emit()
            return snapped
        if change == QGraphicsItem.GraphicsItemChange.ItemSelectedHasChanged:
            if self._scene_ref is not None and value:
                self._scene_ref.gate_selected.emit(self._gate_instance)
        return super(GateGraphicsItem, self).itemChange(change, value)


# =========================================================================
# SwapGateItem
# =========================================================================

class SwapGateItem(GateGraphicsItem):
    """Visual for SWAP gates: two X marks connected by a vertical line."""

    def boundingRect(self) -> QRectF:  # noqa: N802
        qubits = self._gate_instance.target_qubits
        if len(qubits) < 2:
            return QRectF(0, 0, GATE_WIDTH, GATE_WIDTH)
        first_q = qubits[0]
        min_q = min(qubits)
        max_q = max(qubits)
        top = (min_q - first_q) * WIRE_Y_SPACING + GATE_WIDTH / 2 - 10
        bottom = (max_q - first_q) * WIRE_Y_SPACING + GATE_WIDTH / 2 + 10
        return QRectF(-5, top, GATE_WIDTH + 10, bottom - top + 20)

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: QWidget | None = None,
    ) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        qubits = self._gate_instance.target_qubits
        first_q = qubits[0]
        cx = GATE_WIDTH / 2

        def rel_y(q: int) -> float:
            return (q - first_q) * WIRE_Y_SPACING + GATE_WIDTH / 2

        min_q = min(qubits)
        max_q = max(qubits)

        # Vertical line
        line_pen = QPen(QColor("#CCCCCC"), 2.0)
        if self.isSelected():
            line_pen = QPen(QColor("#FFD700"), 2.5)
        painter.setPen(line_pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawLine(
            QPointF(cx, rel_y(min_q)),
            QPointF(cx, rel_y(max_q)),
        )

        # X marks at each qubit
        x_size = 8
        pen = QPen(QColor("#CCCCCC"), 2.5)
        if self.isSelected():
            pen = QPen(QColor("#FFD700"), 2.5)
        painter.setPen(pen)
        for q in qubits:
            qy = rel_y(q)
            painter.drawLine(
                QPointF(cx - x_size, qy - x_size),
                QPointF(cx + x_size, qy + x_size),
            )
            painter.drawLine(
                QPointF(cx - x_size, qy + x_size),
                QPointF(cx + x_size, qy - x_size),
            )


# =========================================================================
# MeasurementGateItem
# =========================================================================

class MeasurementGateItem(GateGraphicsItem):
    """Visual for measurement: a box with arc and arrow."""

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: QWidget | None = None,
    ) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        margin = (GATE_WIDTH - GATE_BOX) / 2
        box_rect = QRectF(margin, margin, GATE_BOX, GATE_BOX)

        # Background box
        color = QColor(self._gate_def.color)
        painter.setBrush(QBrush(color))
        pen = QPen(QColor("#FFFFFF"), 1.5)
        if self.isSelected():
            pen = QPen(QColor("#FFD700"), 2.5)
        painter.setPen(pen)
        painter.drawRoundedRect(box_rect, 6, 6)

        # Measurement arc (semicircle in lower portion)
        arc_pen = QPen(QColor("#FFFFFF"), 2.0)
        painter.setPen(arc_pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        arc_cx = GATE_WIDTH / 2
        arc_cy = margin + GATE_BOX * 0.6
        arc_rx = GATE_BOX * 0.32
        arc_ry = GATE_BOX * 0.28

        arc_rect = QRectF(
            arc_cx - arc_rx, arc_cy - arc_ry,
            arc_rx * 2, arc_ry * 2,
        )
        # Draw arc from 0 to 180 degrees (Qt uses 16ths of a degree)
        painter.drawArc(arc_rect, 0 * 16, 180 * 16)

        # Arrow from arc centre going upper-right
        arrow_end_x = arc_cx + GATE_BOX * 0.28
        arrow_end_y = margin + GATE_BOX * 0.2
        painter.drawLine(
            QPointF(arc_cx, arc_cy),
            QPointF(arrow_end_x, arrow_end_y),
        )

        # Small arrowhead
        angle = math.atan2(arc_cy - arrow_end_y, arc_cx - arrow_end_x)
        arrow_len = 6
        a1x = arrow_end_x + arrow_len * math.cos(angle + 0.5)
        a1y = arrow_end_y + arrow_len * math.sin(angle + 0.5)
        a2x = arrow_end_x + arrow_len * math.cos(angle - 0.5)
        a2y = arrow_end_y + arrow_len * math.sin(angle - 0.5)
        path = QPainterPath()
        path.moveTo(arrow_end_x, arrow_end_y)
        path.lineTo(a1x, a1y)
        path.lineTo(a2x, a2y)
        path.closeSubpath()
        painter.setBrush(QBrush(QColor("#FFFFFF")))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawPath(path)


# =========================================================================
# BarrierItem
# =========================================================================

class BarrierItem(QGraphicsItem):
    """Vertical dashed line representing a barrier across qubits.

    Unlike gates, a barrier is not selectable/movable in the usual sense;
    it serves as a visual separator.
    """

    def __init__(
        self,
        column: int,
        qubit_start: int,
        qubit_end: int,
        parent: QGraphicsItem | None = None,
    ):
        super().__init__(parent)
        self._column = column
        self._qubit_start = qubit_start
        self._qubit_end = qubit_end

        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, False)

        # Position at the column centre
        self.setPos(
            WIRE_START_X + column * GRID_SIZE + GRID_SIZE / 2,
            qubit_start * WIRE_Y_SPACING,
        )

    @property
    def column(self) -> int:
        return self._column

    def boundingRect(self) -> QRectF:  # noqa: N802
        height = (self._qubit_end - self._qubit_start) * WIRE_Y_SPACING
        return QRectF(-5, -10, 10, height + 20)

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: QWidget | None = None,
    ) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        pen = QPen(QColor("#BDBDBD"), 2.0, Qt.PenStyle.DashLine)
        if self.isSelected():
            pen = QPen(QColor("#FFD700"), 2.5, Qt.PenStyle.DashLine)
        painter.setPen(pen)

        height = (self._qubit_end - self._qubit_start) * WIRE_Y_SPACING
        painter.drawLine(QPointF(0, 0), QPointF(0, height))


# =========================================================================
# Factory
# =========================================================================

def create_gate_item(
    gate_instance: GateInstance,
    gate_def: GateDefinition | None = None,
) -> GateGraphicsItem | BarrierItem:
    """Create the appropriate visual item for a given GateInstance.

    If *gate_def* is not supplied it is looked up from the singleton
    GateRegistry.
    """
    if gate_def is None:
        gate_def = GateRegistry.instance().get(gate_instance.gate_name)

    if gate_def.gate_type == GateType.BARRIER:
        qubits = gate_instance.target_qubits
        q_start = min(qubits) if qubits else 0
        q_end = max(qubits) if qubits else 0
        return BarrierItem(gate_instance.column, q_start, q_end)

    if gate_def.gate_type == GateType.MEASUREMENT:
        return MeasurementGateItem(gate_instance, gate_def)

    if gate_def.name == "SWAP":
        return SwapGateItem(gate_instance, gate_def)

    if gate_def.gate_type == GateType.CONTROLLED:
        return ControlledGateItem(gate_instance, gate_def)

    # Default: single-qubit gate
    return SingleQubitGateItem(gate_instance, gate_def)
