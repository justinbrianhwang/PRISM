"""Gate palette dock widget with drag-and-drop gate buttons.

Provides a categorized palette of quantum gates that users can drag
onto the circuit editor. Gates are organized into collapsible sections
using a QToolBox: Single Qubit, Rotations, Multi-Qubit, and
Measurement & Other.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, QMimeData, QSize, QPoint
from PyQt6.QtGui import QDrag, QColor, QPainter, QFont, QPixmap, QPen
from PyQt6.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QToolBox,
    QScrollArea, QSizePolicy,
)

from quantum_sim.engine.gate_registry import GateRegistry
from quantum_sim.engine.gates import GateType


class FlowLayout(QVBoxLayout):
    """A simple layout that wraps widgets into rows that fit the
    available width, emulating a CSS flex-wrap: wrap layout.

    Since QLayout subclassing in PyQt6 can be complex, this uses a
    container widget approach: it places a widget grid manually.
    We use a simpler approach with a wrapping widget.
    """
    pass


class FlowWidget(QWidget):
    """A container widget that arranges children in a flow (wrapping) layout.

    Children are placed left-to-right and wrap to the next row when the
    available width is exceeded. This is used to display gate buttons in
    a compact grid-like arrangement that adapts to the panel width.
    """

    SPACING = 6

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._widgets: list[QWidget] = []

    def add_widget(self, widget: QWidget):
        """Add a child widget to the flow layout."""
        widget.setParent(self)
        self._widgets.append(widget)
        self._relayout()

    def clear_widgets(self):
        """Remove and delete all child widgets."""
        for w in self._widgets:
            w.setParent(None)
            w.deleteLater()
        self._widgets.clear()
        self._relayout()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._relayout()

    def _relayout(self):
        """Recompute positions of all child widgets to wrap within width."""
        if not self._widgets:
            self.setMinimumHeight(0)
            return

        x = self.SPACING
        y = self.SPACING
        row_height = 0
        available_width = max(self.width(), 100)

        for widget in self._widgets:
            hint = widget.sizeHint()
            w = hint.width()
            h = hint.height()

            if x + w + self.SPACING > available_width and x > self.SPACING:
                # Wrap to next row
                x = self.SPACING
                y += row_height + self.SPACING
                row_height = 0

            widget.setGeometry(x, y, w, h)
            widget.show()
            x += w + self.SPACING
            row_height = max(row_height, h)

        total_height = y + row_height + self.SPACING
        self.setMinimumHeight(total_height)

    def sizeHint(self):
        return QSize(200, self.minimumHeight() if self.minimumHeight() > 0 else 60)


class GateButton(QWidget):
    """A draggable button representing a quantum gate.

    Displays the gate symbol centered on a colored square. Supports
    drag-and-drop via QDrag with the gate name stored in QMimeData.

    Attributes:
        gate_name: The registry name of the gate.
        gate_symbol: The display symbol shown on the button.
        gate_color: The hex color string for the button background.
    """

    BUTTON_SIZE = 50

    def __init__(
        self,
        gate_name: str,
        gate_symbol: str,
        gate_color: str,
        display_name: str = "",
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.gate_name = gate_name
        self.gate_symbol = gate_symbol
        self.gate_color = gate_color
        self.display_name = display_name or gate_name
        self._hovered = False
        self._pressed = False
        self._drag_start_pos: QPoint | None = None

        self.setFixedSize(self.BUTTON_SIZE, self.BUTTON_SIZE)
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        self.setToolTip(f"{self.display_name} ({self.gate_name})")
        self.setMouseTracking(True)

    def sizeHint(self):
        return QSize(self.BUTTON_SIZE, self.BUTTON_SIZE)

    def enterEvent(self, event):
        self._hovered = True
        self.update()

    def leaveEvent(self, event):
        self._hovered = False
        self._pressed = False
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._pressed = True
            self._drag_start_pos = event.pos()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            self.update()

    def mouseReleaseEvent(self, event):
        self._pressed = False
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        self.update()

    def mouseMoveEvent(self, event):
        if not (event.buttons() & Qt.MouseButton.LeftButton):
            return
        if self._drag_start_pos is None:
            return

        # Check minimum drag distance
        distance = (event.pos() - self._drag_start_pos).manhattanLength()
        if distance < 10:
            return

        # Create drag
        drag = QDrag(self)
        mime_data = QMimeData()
        mime_data.setText(self.gate_name)
        mime_data.setData(
            "application/x-quantum-gate", self.gate_name.encode("utf-8")
        )
        drag.setMimeData(mime_data)

        # Create drag pixmap
        pixmap = self._create_drag_pixmap()
        drag.setPixmap(pixmap)
        drag.setHotSpot(QPoint(pixmap.width() // 2, pixmap.height() // 2))

        self._pressed = False
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        self.update()

        drag.exec(Qt.DropAction.CopyAction)

    def _create_drag_pixmap(self) -> QPixmap:
        """Create a semi-transparent pixmap for the drag operation."""
        size = self.BUTTON_SIZE
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.GlobalColor.transparent)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setOpacity(0.8)

        # Draw background
        color = QColor(self.gate_color)
        painter.setBrush(color)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(2, 2, size - 4, size - 4, 6, 6)

        # Draw symbol
        painter.setPen(QPen(QColor("#ffffff")))
        font = QFont("Segoe UI", 12, QFont.Weight.Bold)
        painter.setFont(font)
        painter.drawText(
            2, 2, size - 4, size - 4,
            Qt.AlignmentFlag.AlignCenter, self.gate_symbol,
        )

        painter.end()
        return pixmap

    def paintEvent(self, event):
        """Draw the gate button with color, symbol, and hover/press effects."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        size = self.BUTTON_SIZE
        margin = 2
        rect_size = size - 2 * margin

        # Determine colors based on state
        base_color = QColor(self.gate_color)
        if self._pressed:
            base_color = base_color.darker(130)
        elif self._hovered:
            base_color = base_color.lighter(120)

        # Draw rounded rect background
        painter.setBrush(base_color)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(margin, margin, rect_size, rect_size, 6, 6)

        # Draw border on hover
        if self._hovered:
            border_color = QColor(base_color)
            border_color = border_color.lighter(150)
            painter.setPen(QPen(border_color, 1.5))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(margin, margin, rect_size, rect_size, 6, 6)

        # Draw symbol text
        painter.setPen(QPen(QColor("#ffffff")))
        font = QFont("Segoe UI", 11, QFont.Weight.Bold)
        painter.setFont(font)
        painter.drawText(
            margin, margin, rect_size, rect_size,
            Qt.AlignmentFlag.AlignCenter, self.gate_symbol,
        )

        painter.end()


class GatePalette(QDockWidget):
    """Dock widget containing the gate palette organized by category.

    Gates are organized into collapsible sections:
    - Single Qubit: I, H, X, Y, Z, S, S-dagger, T, T-dagger
    - Rotations: Rx, Ry, Rz, Phase, U3
    - Multi-Qubit: CNOT, CZ, SWAP, Toffoli, Fredkin
    - Measurement & Other: Measure, Barrier

    Each gate is represented by a GateButton that can be dragged
    onto the circuit editor.
    """

    def __init__(self, parent: QWidget | None = None):
        super().__init__("Gate Palette", parent)
        self.setObjectName("GatePaletteDock")
        self.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea
            | Qt.DockWidgetArea.RightDockWidgetArea
        )
        self.setMinimumWidth(180)

        self._setup_ui()

    def _setup_ui(self):
        """Build the palette UI with a QToolBox for collapsible sections."""
        # Main container
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(0)

        # Scrollable toolbox
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        scroll.setFrameShape(scroll.Shape.NoFrame)

        self._toolbox = QToolBox()
        self._toolbox.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.MinimumExpanding
        )

        # Populate sections from the gate registry
        self._populate_sections()

        scroll.setWidget(self._toolbox)
        layout.addWidget(scroll)
        self.setWidget(container)

    def _populate_sections(self):
        """Create gate button sections from the GateRegistry."""
        registry = GateRegistry.instance()

        # Define section categories with gate name lists
        sections = [
            (
                "Single Qubit",
                [g for g in registry.all_gates()
                 if g.gate_type == GateType.SINGLE and g.num_params == 0],
            ),
            (
                "Rotations",
                [g for g in registry.all_gates()
                 if g.gate_type == GateType.SINGLE and g.num_params > 0],
            ),
            (
                "Multi-Qubit",
                [g for g in registry.all_gates()
                 if g.gate_type in (GateType.CONTROLLED, GateType.MULTI)],
            ),
            (
                "Measurement & Other",
                [g for g in registry.all_gates()
                 if g.gate_type in (GateType.MEASUREMENT, GateType.BARRIER)],
            ),
        ]

        for section_name, gate_defs in sections:
            if not gate_defs:
                continue

            flow = FlowWidget()
            for gate_def in gate_defs:
                btn = GateButton(
                    gate_name=gate_def.name,
                    gate_symbol=gate_def.symbol,
                    gate_color=gate_def.color,
                    display_name=gate_def.display_name,
                )
                flow.add_widget(btn)

            # Wrap flow widget in scroll area for each section
            section_scroll = QScrollArea()
            section_scroll.setWidgetResizable(True)
            section_scroll.setHorizontalScrollBarPolicy(
                Qt.ScrollBarPolicy.ScrollBarAlwaysOff
            )
            section_scroll.setFrameShape(section_scroll.Shape.NoFrame)
            section_scroll.setWidget(flow)
            section_scroll.setMinimumHeight(60)
            section_scroll.setSizePolicy(
                QSizePolicy.Policy.Preferred,
                QSizePolicy.Policy.MinimumExpanding,
            )

            self._toolbox.addItem(section_scroll, section_name)

    def refresh(self):
        """Rebuild the palette from the current gate registry state.

        Call this if custom gates have been registered after initial creation.
        """
        # Remove existing items
        while self._toolbox.count() > 0:
            widget = self._toolbox.widget(0)
            self._toolbox.removeItem(0)
            if widget is not None:
                widget.deleteLater()

        self._populate_sections()
