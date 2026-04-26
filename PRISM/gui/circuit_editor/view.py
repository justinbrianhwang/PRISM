"""CircuitView -- QGraphicsView for the quantum circuit editor.

Features:
- Antialiased rendering.
- Ctrl+Scroll wheel for zoom (0.25x -- 4.0x).
- Plain scroll for vertical pan.
- Rubber-band selection.
- Accepts drops from the gate palette.
- Keyboard shortcuts: Delete (remove selected), Ctrl+Z (undo),
  Ctrl+Y (redo), Ctrl+A (select all), Ctrl+0 (fit in view).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtWidgets import QGraphicsView
from PyQt6.QtGui import QPainter, QKeyEvent, QWheelEvent
from PyQt6.QtCore import Qt, pyqtSignal

if TYPE_CHECKING:
    from quantum_sim.gui.circuit_editor.scene import CircuitScene

# Zoom limits
MIN_ZOOM = 0.25
MAX_ZOOM = 4.0
ZOOM_FACTOR = 1.15  # per wheel step


class CircuitView(QGraphicsView):
    """Interactive graphics view for the quantum circuit editor.

    Signals
    -------
    delete_selected()
        Emitted when the Delete key is pressed with selected items.
    undo_requested()
        Emitted when Ctrl+Z is pressed.
    redo_requested()
        Emitted when Ctrl+Y is pressed.
    select_all_requested()
        Emitted when Ctrl+A is pressed.
    fit_view_requested()
        Emitted when Ctrl+0 is pressed.
    """

    delete_selected = pyqtSignal()
    undo_requested = pyqtSignal()
    redo_requested = pyqtSignal()
    select_all_requested = pyqtSignal()
    fit_view_requested = pyqtSignal()

    def __init__(self, scene: CircuitScene | None = None, parent=None):
        super().__init__(parent)

        if scene is not None:
            self.setScene(scene)

        self._current_zoom = 1.0

        # --- Rendering --------------------------------------------------------
        self.setRenderHints(
            QPainter.RenderHint.Antialiasing
            | QPainter.RenderHint.TextAntialiasing
            | QPainter.RenderHint.SmoothPixmapTransform
        )

        # --- Interaction ------------------------------------------------------
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)

        # --- Drag-and-drop from palette (gate buttons) ------------------------
        self.setAcceptDrops(True)

        # --- Scrollbar policy (always show to keep layout predictable) --------
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        # --- Visual styling ---------------------------------------------------
        self.setStyleSheet(
            "QGraphicsView { border: 1px solid #3C3C3C; background: #1E1E1E; }"
        )

    # ---- Zoom -------------------------------------------------------------

    @property
    def current_zoom(self) -> float:
        return self._current_zoom

    def zoom_in(self) -> None:
        """Zoom in by one step."""
        if self._current_zoom * ZOOM_FACTOR <= MAX_ZOOM:
            self.scale(ZOOM_FACTOR, ZOOM_FACTOR)
            self._current_zoom *= ZOOM_FACTOR

    def zoom_out(self) -> None:
        """Zoom out by one step."""
        if self._current_zoom / ZOOM_FACTOR >= MIN_ZOOM:
            inv = 1.0 / ZOOM_FACTOR
            self.scale(inv, inv)
            self._current_zoom /= ZOOM_FACTOR

    def reset_zoom(self) -> None:
        """Reset zoom to 1.0x."""
        self.resetTransform()
        self._current_zoom = 1.0

    def fit_circuit_in_view(self) -> None:
        """Scale and scroll so the entire circuit fits the viewport."""
        scene = self.scene()
        if scene is not None:
            self.fitInView(scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
            # Approximate the new zoom level from the transform
            t = self.transform()
            self._current_zoom = t.m11()

    # ---- Event overrides --------------------------------------------------

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        """Ctrl+Scroll = zoom, plain scroll = pan."""
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            angle = event.angleDelta().y()
            if angle > 0:
                self.zoom_in()
            elif angle < 0:
                self.zoom_out()
            event.accept()
        else:
            # Default behaviour: scroll / pan
            super().wheelEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        key = event.key()
        modifiers = event.modifiers()
        ctrl = bool(modifiers & Qt.KeyboardModifier.ControlModifier)

        if key == Qt.Key.Key_Delete or key == Qt.Key.Key_Backspace:
            self.delete_selected.emit()
            event.accept()
            return

        if ctrl and key == Qt.Key.Key_Z:
            self.undo_requested.emit()
            event.accept()
            return

        if ctrl and key == Qt.Key.Key_Y:
            self.redo_requested.emit()
            event.accept()
            return

        if ctrl and key == Qt.Key.Key_A:
            self.select_all_requested.emit()
            event.accept()
            return

        if ctrl and key == Qt.Key.Key_0:
            self.fit_view_requested.emit()
            event.accept()
            return

        # Plus / Minus for zoom without Ctrl (convenience)
        if key == Qt.Key.Key_Plus or key == Qt.Key.Key_Equal:
            self.zoom_in()
            event.accept()
            return
        if key == Qt.Key.Key_Minus:
            self.zoom_out()
            event.accept()
            return

        super().keyPressEvent(event)

    # ---- Drag enter (delegate to scene) -----------------------------------

    def dragEnterEvent(self, event) -> None:  # noqa: N802
        if event.mimeData().hasText():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event) -> None:  # noqa: N802
        if event.mimeData().hasText():
            event.acceptProposedAction()
        # Let the scene also handle the event for the indicator
        super().dragMoveEvent(event)

    def dropEvent(self, event) -> None:  # noqa: N802
        # Delegate to the scene's dropEvent
        super().dropEvent(event)
