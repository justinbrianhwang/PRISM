"""CircuitScene -- QGraphicsScene that manages the visual circuit editor.

Responsibilities:
- Maintains qubit wire items and gate visual items.
- Rebuilds the entire scene from the circuit model.
- Handles drag-and-drop of gates from the palette.
- Snap-to-grid positioning.
- Emits signals for gate selection and circuit changes.
- Provides add/remove gate visual helpers for undo/redo integration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtWidgets import (
    QGraphicsScene,
    QGraphicsRectItem,
    QGraphicsLineItem,
    QGraphicsSceneDragDropEvent,
    QInputDialog,
)
from PyQt6.QtGui import QBrush, QColor, QPen
from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal

from quantum_sim.engine.circuit import GateInstance, QuantumCircuit
from quantum_sim.engine.gates import GateType
from quantum_sim.engine.gate_registry import GateRegistry

from quantum_sim.gui.circuit_editor.wire_item import WireItem
from quantum_sim.gui.circuit_editor.gate_items import (
    GRID_SIZE,
    WIRE_Y_SPACING,
    GATE_WIDTH,
    WIRE_START_X,
    GateGraphicsItem,
    BarrierItem,
    create_gate_item,
    _cell_x,
    _cell_y,
    _snap_to_grid,
)

if TYPE_CHECKING:
    pass

# Re-export the layout constants so other modules can import from scene
__all__ = [
    "CircuitScene",
    "GRID_SIZE",
    "WIRE_Y_SPACING",
    "GATE_WIDTH",
    "WIRE_START_X",
]


class CircuitScene(QGraphicsScene):
    """Graphics scene that renders a QuantumCircuit visually.

    Signals
    -------
    gate_selected(object)
        Emitted when a gate item is selected (carries the GateInstance).
    gate_double_clicked(object)
        Emitted when a gate item is double-clicked (for parameter editing).
    circuit_changed()
        Emitted whenever the visual circuit is modified.
    """

    gate_selected = pyqtSignal(object)
    gate_double_clicked = pyqtSignal(object)
    circuit_changed = pyqtSignal()
    qubit_state_toggled = pyqtSignal(int, int)  # (qubit_index, new_state)

    def __init__(self, circuit: QuantumCircuit, parent=None):
        super().__init__(parent)
        self._circuit = circuit
        self._wire_items: list[WireItem] = []
        self._gate_items: list[GateGraphicsItem | BarrierItem] = []
        self._drop_indicator: QGraphicsRectItem | None = None

        # Debugger overlay items
        self._breakpoint_lines: dict[int, QGraphicsLineItem] = {}
        self._debug_highlight: QGraphicsRectItem | None = None

        # Initial background for dark theme
        self.setBackgroundBrush(QBrush(QColor("#1E1E1E")))

        self.rebuild()

    # ---- Properties -------------------------------------------------------

    @property
    def circuit(self) -> QuantumCircuit:
        return self._circuit

    @circuit.setter
    def circuit(self, value: QuantumCircuit) -> None:
        self._circuit = value
        self.rebuild()

    # ---- Rebuild ----------------------------------------------------------

    def rebuild(self) -> None:
        """Clear the scene and redraw everything from the circuit model."""
        self.clear()
        self._wire_items.clear()
        self._gate_items.clear()
        self._drop_indicator = None

        num_cols = max(self._circuit.get_column_count() + 4, 10)
        wire_length = num_cols * GRID_SIZE + 40

        # Draw qubit wires
        for qi in range(self._circuit.num_qubits):
            init_state = self._circuit.initial_states[qi] if qi < len(self._circuit.initial_states) else 0
            wire = WireItem(qi, initial_state=init_state, wire_length=wire_length)
            self.addItem(wire)
            self._wire_items.append(wire)

        # Draw gates
        registry = GateRegistry.instance()
        for gate_inst in self._circuit.gates:
            try:
                gate_def = registry.get(gate_inst.gate_name)
            except KeyError:
                continue
            item = create_gate_item(gate_inst, gate_def)
            if isinstance(item, GateGraphicsItem):
                item.circuit_scene = self
            self.addItem(item)
            self._gate_items.append(item)

        # Set scene rect with some padding
        total_width = WIRE_START_X + wire_length + 40
        total_height = max(self._circuit.num_qubits, 1) * WIRE_Y_SPACING + 40
        self.setSceneRect(QRectF(-10, -30, total_width, total_height))

    # ---- Wire label click handling -----------------------------------------

    def _on_wire_label_clicked(self, wire_item: WireItem) -> None:
        """Handle a qubit label click - toggle initial state."""
        qi = wire_item.qubit_index
        new_state = wire_item.initial_state  # already toggled by WireItem
        if qi < len(self._circuit.initial_states):
            self._circuit.initial_states[qi] = new_state
        self.qubit_state_toggled.emit(qi, new_state)
        self.circuit_changed.emit()

    def mousePressEvent(self, event) -> None:
        """Override to detect clicks on wire label areas."""
        pos = event.scenePos()
        # Check if click is in the label area (x < WIRE_START_X)
        if pos.x() < WIRE_START_X:
            for wire in self._wire_items:
                wire_y = wire.pos().y()
                if abs(pos.y() - wire_y) < WIRE_Y_SPACING / 2:
                    # Toggle happens in WireItem.mousePressEvent
                    super().mousePressEvent(event)
                    self._on_wire_label_clicked(wire)
                    return
        super().mousePressEvent(event)

    # ---- Grid helpers -----------------------------------------------------

    def snap_to_grid(self, pos: QPointF) -> tuple[int, int]:
        """Convert a scene position to (column, qubit_index), clamped."""
        col, qubit = _snap_to_grid(pos)
        qubit = max(0, min(qubit, self._circuit.num_qubits - 1))
        return col, qubit

    def column_qubit_to_pos(self, col: int, qubit: int) -> QPointF:
        """Return scene centre for a given (column, qubit) cell."""
        return QPointF(
            _cell_x(col) + GATE_WIDTH / 2,
            _cell_y(qubit),
        )

    # ---- Gate visual management (for undo/redo) ---------------------------

    def add_gate_visual(self, gate_instance: GateInstance) -> None:
        """Create and add a visual item for an existing GateInstance."""
        registry = GateRegistry.instance()
        try:
            gate_def = registry.get(gate_instance.gate_name)
        except KeyError:
            return
        item = create_gate_item(gate_instance, gate_def)
        if isinstance(item, GateGraphicsItem):
            item.circuit_scene = self
        self.addItem(item)
        self._gate_items.append(item)
        self._update_wire_lengths()
        self.circuit_changed.emit()

    def remove_gate_visual(self, gate_instance: GateInstance) -> None:
        """Remove the visual item associated with a GateInstance."""
        for item in list(self._gate_items):
            if isinstance(item, GateGraphicsItem) and item.gate_instance is gate_instance:
                self.removeItem(item)
                self._gate_items.remove(item)
                break
        self._update_wire_lengths()
        self.circuit_changed.emit()

    def refresh_gate_visual(self, gate_instance: GateInstance) -> None:
        """Re-sync the visual for a gate (position or appearance change)."""
        for item in self._gate_items:
            if isinstance(item, GateGraphicsItem) and item.gate_instance is gate_instance:
                item.refresh_position()
                item.update()
                break
        self.circuit_changed.emit()

    def _update_wire_lengths(self) -> None:
        """Extend wires to accommodate new columns."""
        num_cols = max(self._circuit.get_column_count() + 4, 10)
        wire_length = num_cols * GRID_SIZE + 40
        for wire in self._wire_items:
            wire.wire_length = wire_length

    # ---- Drag-and-drop from palette ---------------------------------------

    def dragEnterEvent(self, event: QGraphicsSceneDragDropEvent) -> None:  # noqa: N802
        if event.mimeData().hasText():
            event.acceptProposedAction()
            self._show_drop_indicator(event.scenePos())
        else:
            event.ignore()

    def dragMoveEvent(self, event: QGraphicsSceneDragDropEvent) -> None:  # noqa: N802
        if event.mimeData().hasText():
            event.acceptProposedAction()
            self._show_drop_indicator(event.scenePos())
        else:
            event.ignore()

    def dragLeaveEvent(self, event: QGraphicsSceneDragDropEvent) -> None:  # noqa: N802
        self._hide_drop_indicator()
        super().dragLeaveEvent(event)

    def dropEvent(self, event: QGraphicsSceneDragDropEvent) -> None:  # noqa: N802
        self._hide_drop_indicator()

        if not event.mimeData().hasText():
            event.ignore()
            return

        gate_name = event.mimeData().text().strip()
        registry = GateRegistry.instance()
        try:
            gate_def = registry.get(gate_name)
        except KeyError:
            event.ignore()
            return

        col, qubit = self.snap_to_grid(event.scenePos())

        # Build target qubit list
        target_qubits = self._resolve_target_qubits(gate_def, qubit)
        if target_qubits is None:
            # User cancelled target selection
            event.ignore()
            return

        # Create default params (zeroes for parameterized gates)
        params = [0.0] * gate_def.num_params

        gate_inst = GateInstance(
            gate_name=gate_name,
            target_qubits=target_qubits,
            params=params,
            column=col,
        )

        # Add to model and scene
        self._circuit.add_gate(gate_inst)
        self.add_gate_visual(gate_inst)

        event.acceptProposedAction()

    def _resolve_target_qubits(self, gate_def, primary_qubit: int) -> list[int] | None:
        """Determine the full target-qubit list for a gate drop.

        For single-qubit / measurement gates, just use the drop qubit.
        For multi-qubit gates, show a dialog to pick additional qubits.
        """
        num_q = gate_def.num_qubits

        if num_q <= 1:
            return [primary_qubit]

        # For multi-qubit gates, the first qubit comes from the drop position.
        # Additional qubits are chosen via an input dialog.
        target_qubits = [primary_qubit]
        remaining = num_q - 1

        for i in range(remaining):
            if gate_def.gate_type == GateType.CONTROLLED and i < gate_def.num_controls:
                role = f"control qubit {i + 1}" if i > 0 or gate_def.num_controls > 1 else "control qubit"
                # The primary drop is the first control; ask for target(s)
                # Actually, let's treat the drop position as control[0]
                # and ask for the rest.
                pass  # role already set

            if gate_def.gate_type == GateType.CONTROLLED and i >= gate_def.num_controls:
                role = "target qubit"
            elif gate_def.gate_type == GateType.MULTI:
                role = f"qubit {i + 2}"
            elif gate_def.gate_type == GateType.CONTROLLED:
                role = f"control qubit {i + 2}" if gate_def.num_controls > 1 else "target qubit"
            else:
                role = f"qubit {i + 2}"

            # Get available qubits (not yet selected)
            available = [q for q in range(self._circuit.num_qubits) if q not in target_qubits]
            if not available:
                return None

            items = [str(q) for q in available]
            # Default to the next sequential qubit if available
            default_idx = 0
            next_q = primary_qubit + len(target_qubits)
            if next_q in available:
                default_idx = available.index(next_q)

            chosen, ok = QInputDialog.getItem(
                None,
                f"Select {role}",
                f"Select {role} for {gate_def.display_name}:",
                items,
                default_idx,
                False,
            )
            if not ok:
                return None
            target_qubits.append(int(chosen))

        return target_qubits

    # ---- Drop indicator ---------------------------------------------------

    def _show_drop_indicator(self, scene_pos: QPointF) -> None:
        """Draw a translucent rectangle at the snapped cell."""
        col, qubit = self.snap_to_grid(scene_pos)
        x = _cell_x(col)
        y = _cell_y(qubit) - GATE_WIDTH / 2

        if self._drop_indicator is None:
            self._drop_indicator = QGraphicsRectItem()
            self._drop_indicator.setBrush(QBrush(QColor(100, 149, 237, 60)))
            self._drop_indicator.setPen(QPen(QColor(100, 149, 237, 160), 1.5, Qt.PenStyle.DashLine))
            self._drop_indicator.setZValue(1000)
            self.addItem(self._drop_indicator)

        self._drop_indicator.setRect(QRectF(x, y, GATE_WIDTH, GATE_WIDTH))
        self._drop_indicator.setVisible(True)

    def _hide_drop_indicator(self) -> None:
        if self._drop_indicator is not None:
            self._drop_indicator.setVisible(False)

    # ---- Selection helpers ------------------------------------------------

    def selected_gate_items(self) -> list[GateGraphicsItem]:
        """Return all currently selected gate items."""
        return [
            item for item in self.selectedItems()
            if isinstance(item, GateGraphicsItem)
        ]

    def selected_gate_instances(self) -> list[GateInstance]:
        """Return the GateInstance objects for selected gate visuals."""
        return [item.gate_instance for item in self.selected_gate_items()]

    def select_all_gates(self) -> None:
        """Select every gate item in the scene."""
        for item in self._gate_items:
            item.setSelected(True)

    def clear_gate_selection(self) -> None:
        """Deselect all items."""
        self.clearSelection()

    # ---- Debugger overlay -------------------------------------------------

    def set_breakpoint(self, column: int, active: bool) -> None:
        """Show or hide a breakpoint marker at the given column."""
        if active:
            if column in self._breakpoint_lines:
                return  # already shown
            x = _cell_x(column) + GATE_WIDTH / 2
            y_top = -20
            y_bot = max(self._circuit.num_qubits, 1) * WIRE_Y_SPACING + 10
            line = QGraphicsLineItem(x, y_top, x, y_bot)
            line.setPen(QPen(QColor(220, 50, 50, 180), 2.0, Qt.PenStyle.DashLine))
            line.setZValue(900)
            self.addItem(line)
            self._breakpoint_lines[column] = line
        else:
            line = self._breakpoint_lines.pop(column, None)
            if line is not None:
                self.removeItem(line)

    def clear_breakpoints_visual(self) -> None:
        """Remove all breakpoint visuals."""
        for line in self._breakpoint_lines.values():
            self.removeItem(line)
        self._breakpoint_lines.clear()

    def set_debug_highlight(self, column: int | None) -> None:
        """Highlight the column currently being inspected in the debugger.

        Pass None to clear the highlight.
        """
        if column is None:
            if self._debug_highlight is not None:
                self._debug_highlight.setVisible(False)
            return

        x = _cell_x(column)
        y_top = -20
        height = max(self._circuit.num_qubits, 1) * WIRE_Y_SPACING + 30

        if self._debug_highlight is None:
            self._debug_highlight = QGraphicsRectItem()
            self._debug_highlight.setBrush(QBrush(QColor(255, 200, 50, 30)))
            self._debug_highlight.setPen(QPen(QColor(255, 200, 50, 100), 1.0, Qt.PenStyle.DashLine))
            self._debug_highlight.setZValue(800)
            self.addItem(self._debug_highlight)

        self._debug_highlight.setRect(QRectF(x, y_top, GATE_WIDTH, height))
        self._debug_highlight.setVisible(True)
