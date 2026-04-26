"""Circuit export utilities for PNG and SVG output."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt, QRectF, QSizeF, QMarginsF
from PyQt6.QtGui import QImage, QPainter, QColor
from PyQt6.QtWidgets import QGraphicsScene


class CircuitExporter:
    """Exports a QGraphicsScene (circuit view) to PNG or SVG files."""

    @staticmethod
    def export_png(
        scene: QGraphicsScene,
        filepath: str | Path,
        scale: float = 2.0,
    ) -> None:
        """Export the circuit scene as a PNG image.

        Args:
            scene: The QGraphicsScene containing the circuit diagram.
            filepath: Output file path (should end in .png).
            scale: Scale factor for the output resolution (default 2x for
                   high-DPI / retina quality).
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Get the bounding rect of all items with a small margin
        scene_rect = scene.itemsBoundingRect()
        margin = 20.0
        scene_rect = scene_rect.adjusted(-margin, -margin, margin, margin)

        # Create a high-resolution image
        width = int(scene_rect.width() * scale)
        height = int(scene_rect.height() * scale)

        if width <= 0 or height <= 0:
            # Empty scene, create a minimal image
            width = max(width, 100)
            height = max(height, 100)

        image = QImage(width, height, QImage.Format.Format_ARGB32_Premultiplied)
        image.fill(QColor(255, 255, 255, 255))  # White background

        painter = QPainter(image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        # Scale the painter
        painter.scale(scale, scale)

        # Render the scene
        scene.render(
            painter,
            QRectF(0, 0, scene_rect.width(), scene_rect.height()),
            scene_rect,
        )

        painter.end()
        image.save(str(filepath))

    @staticmethod
    def export_svg(
        scene: QGraphicsScene,
        filepath: str | Path,
    ) -> None:
        """Export the circuit scene as an SVG file.

        Uses QSvgGenerator if available (requires PyQt6-QSvgWidgets).
        Falls back gracefully if the SVG module is not installed.

        Args:
            scene: The QGraphicsScene containing the circuit diagram.
            filepath: Output file path (should end in .svg).
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        try:
            from PyQt6.QtSvg import QSvgGenerator
        except ImportError:
            raise ImportError(
                "SVG export requires PyQt6-QSvgWidgets. "
                "Install it with: pip install PyQt6-QSvgWidgets"
            )

        scene_rect = scene.itemsBoundingRect()
        margin = 20.0
        scene_rect = scene_rect.adjusted(-margin, -margin, margin, margin)

        generator = QSvgGenerator()
        generator.setFileName(str(filepath))
        generator.setSize(
            QSizeF(scene_rect.width(), scene_rect.height()).toSize()
        )
        generator.setViewBox(
            QRectF(0, 0, scene_rect.width(), scene_rect.height())
        )
        generator.setTitle("Quantum Circuit")
        generator.setDescription("Exported from Quantum Circuit Simulator")

        painter = QPainter(generator)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)

        scene.render(
            painter,
            QRectF(0, 0, scene_rect.width(), scene_rect.height()),
            scene_rect,
        )

        painter.end()
