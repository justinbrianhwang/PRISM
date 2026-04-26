"""Export utilities for PRISM.

Two flavours of exporter live here:

* :class:`CircuitExporter` -- saves a single ``QGraphicsScene`` (the
  circuit diagram) as PNG or SVG.
* :class:`WindowExporter` -- saves the *entire* main window (or any
  ``QWidget``) as a high-DPI PNG and / or vector-rich PDF, using
  ``QWidget.render`` so the output is independent of the user's
  display DPI and is always sharp regardless of the OS-level zoom
  setting.  Designed for paper screenshots: matches what the user
  sees on screen but rendered at 3x by default so that the print
  result stays crisp.
"""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QMarginsF, QRectF, QSize, QSizeF, Qt
from PyQt6.QtGui import (
    QColor,
    QImage,
    QPageLayout,
    QPageSize,
    QPainter,
    QPdfWriter,
)
from PyQt6.QtWidgets import QGraphicsScene, QWidget


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
        generator.setDescription("Exported from PRISM")

        painter = QPainter(generator)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)

        scene.render(
            painter,
            QRectF(0, 0, scene_rect.width(), scene_rect.height()),
            scene_rect,
        )

        painter.end()


# ---------------------------------------------------------------------------
# Whole-window exporter
# ---------------------------------------------------------------------------


_DEFAULT_WINDOW_SCALE = 3.0
"""Default supersampling factor for window screenshots.

3x is chosen so a 1920x1080 window produces a 5760x3240 image: well
above the 300 DPI threshold even when printed across a full A4 page,
and visibly sharper than a 1x grab even on a regular monitor.
"""


class WindowExporter:
    """Capture an arbitrary ``QWidget`` (typically the main window) at
    arbitrary supersample factors, saving as PNG and / or PDF.

    Unlike a manual screenshot, this rasterises through Qt's painter
    pipeline so the output is independent of the user's screen DPI:
    the same window will look identical (and equally sharp) whether
    captured on a 1920x1080 monitor or a high-DPI laptop.

    PDF output uses :class:`QPdfWriter`, which keeps Qt-rendered text
    and shapes as vectors where possible -- so menu labels, panel
    titles and matplotlib axis text print sharply at any zoom.
    Embedded raster content (images, OpenGL views) stays raster but is
    sized to ``scale * widget.size()`` so resolution stays high.
    """

    @staticmethod
    def export_png(
        widget: QWidget,
        filepath: str | Path,
        scale: float = _DEFAULT_WINDOW_SCALE,
    ) -> None:
        """Render ``widget`` into a PNG at ``scale`` x its on-screen size.

        Args:
            widget: Any QWidget; the main ``QMainWindow`` is the
                typical caller.
            filepath: Output path (must end in ``.png``).
            scale: Supersampling factor.  Default is 3x: a 1920x1080
                window becomes a 5760x3240 PNG.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        size = widget.size()
        if size.isEmpty():
            raise ValueError("Cannot export a widget with empty size")

        target_w = int(size.width() * scale)
        target_h = int(size.height() * scale)
        image = QImage(target_w, target_h, QImage.Format.Format_ARGB32_Premultiplied)
        image.fill(QColor(255, 255, 255, 255))

        painter = QPainter(image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.scale(scale, scale)
        widget.render(painter)
        painter.end()

        image.save(str(filepath), "PNG")

    @staticmethod
    def export_pdf(
        widget: QWidget,
        filepath: str | Path,
        resolution: int = 300,
    ) -> None:
        """Render ``widget`` into a PDF page.

        The page is sized to match the widget's on-screen aspect ratio
        at the requested ``resolution`` (DPI), so the resulting PDF is
        a 1:1 reproduction of the on-screen window with sharp vector
        text.  Margins are zero -- the page IS the window.

        Args:
            widget: Any QWidget.
            filepath: Output path (must end in ``.pdf``).
            resolution: PDF page resolution in DPI (default 300, which
                meets typical journal print requirements).
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        size = widget.size()
        if size.isEmpty():
            raise ValueError("Cannot export a widget with empty size")

        # Page size in millimetres so the PDF aspect matches the widget
        # exactly.  We map "1 widget pixel == 1/96 inch" (the de-facto
        # CSS reference) so a 1920x1080 widget becomes a 20"x11.25" page.
        widget_inch_w = size.width() / 96.0
        widget_inch_h = size.height() / 96.0
        widget_mm_w = widget_inch_w * 25.4
        widget_mm_h = widget_inch_h * 25.4

        writer = QPdfWriter(str(filepath))
        writer.setResolution(resolution)
        writer.setPageSize(
            QPageSize(QSizeF(widget_mm_w, widget_mm_h),
                      QPageSize.Unit.Millimeter)
        )
        writer.setPageMargins(QMarginsF(0, 0, 0, 0))

        # Match the painter's coordinate system to widget coordinates so
        # that ``widget.render(painter)`` paints into the full page.
        writer_w = writer.width()
        writer_h = writer.height()
        scale_x = writer_w / size.width()
        scale_y = writer_h / size.height()

        painter = QPainter(writer)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.scale(scale_x, scale_y)
        widget.render(painter)
        painter.end()

    @staticmethod
    def export_both(
        widget: QWidget,
        directory: str | Path,
        stem: str,
        scale: float = _DEFAULT_WINDOW_SCALE,
        resolution: int = 300,
    ) -> tuple[Path, Path]:
        """Convenience: emit ``<stem>.png`` and ``<stem>.pdf`` together.

        Returns
        -------
        (png_path, pdf_path)
        """
        directory = Path(directory)
        png_path = directory / f"{stem}.png"
        pdf_path = directory / f"{stem}.pdf"
        WindowExporter.export_png(widget, png_path, scale=scale)
        WindowExporter.export_pdf(widget, pdf_path, resolution=resolution)
        return png_path, pdf_path
