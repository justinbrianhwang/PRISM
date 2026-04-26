"""Smoke tests for :class:`PRISM.core.export.WindowExporter`.

Runs against a Qt offscreen platform so the suite does not need a
display.  We render a minimal QWidget hierarchy (a labelled, coloured
box) and verify that:

* :pymeth:`WindowExporter.export_png` writes a non-empty PNG whose
  raster dimensions equal the supersample factor times the widget
  size.
* :pymeth:`WindowExporter.export_pdf` writes a non-empty PDF whose
  first bytes are the PDF magic ``%PDF``.
* :pymeth:`WindowExporter.export_both` writes both files.
* Empty-size widgets are rejected with a clear error.
"""

from __future__ import annotations

import os

# ``QT_QPA_PLATFORM=offscreen`` must be set before importing any Qt
# module, otherwise PyQt6 will try to connect to a display.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import pytest

# Skip the whole module if PyQt6 is not installed (CI minimal env).
pytest.importorskip("PyQt6.QtWidgets")

from PyQt6.QtCore import QSize  # noqa: E402
from PyQt6.QtGui import QImage  # noqa: E402
from PyQt6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget  # noqa: E402

from PRISM.core.export import WindowExporter  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    """Module-scoped QApplication; PyQt only allows one instance."""
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def small_widget(qapp) -> QWidget:
    """Visible-style widget with a fixed 320x180 size."""
    w = QWidget()
    w.setStyleSheet("background-color: #2563eb;")
    layout = QVBoxLayout(w)
    layout.addWidget(QLabel("PRISM window export smoke test"))
    w.resize(QSize(320, 180))
    # Adjust must run for layout to be applied prior to render().
    w.adjustSize()
    w.resize(QSize(320, 180))
    return w


# ---------------------------------------------------------------------------
# PNG
# ---------------------------------------------------------------------------


class TestExportPng:

    def test_writes_non_empty_file(self, small_widget, tmp_path):
        out = tmp_path / "smoke.png"
        WindowExporter.export_png(small_widget, out, scale=2.0)
        assert out.exists()
        assert out.stat().st_size > 500

    def test_png_dimensions_match_scale(self, small_widget, tmp_path):
        scale = 2.0
        out = tmp_path / "smoke.png"
        WindowExporter.export_png(small_widget, out, scale=scale)

        img = QImage(str(out))
        assert not img.isNull(), "Output PNG could not be reopened"
        size = small_widget.size()
        assert img.width() == int(size.width() * scale)
        assert img.height() == int(size.height() * scale)

    def test_higher_scale_means_larger_image(self, small_widget, tmp_path):
        a = tmp_path / "1x.png"
        b = tmp_path / "3x.png"
        WindowExporter.export_png(small_widget, a, scale=1.0)
        WindowExporter.export_png(small_widget, b, scale=3.0)
        assert b.stat().st_size > a.stat().st_size

    def test_creates_parent_directory(self, small_widget, tmp_path):
        out = tmp_path / "deep" / "nested" / "smoke.png"
        WindowExporter.export_png(small_widget, out)
        assert out.exists()

    def test_rejects_empty_widget(self, qapp, tmp_path):
        empty = QWidget()
        empty.resize(QSize(0, 0))
        with pytest.raises(ValueError, match="empty size"):
            WindowExporter.export_png(empty, tmp_path / "x.png")


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------


class TestExportPdf:

    def test_writes_non_empty_file(self, small_widget, tmp_path):
        out = tmp_path / "smoke.pdf"
        WindowExporter.export_pdf(small_widget, out)
        assert out.exists()
        assert out.stat().st_size > 500

    def test_pdf_magic_header(self, small_widget, tmp_path):
        out = tmp_path / "smoke.pdf"
        WindowExporter.export_pdf(small_widget, out)
        with out.open("rb") as f:
            head = f.read(4)
        assert head == b"%PDF", f"unexpected header: {head!r}"

    def test_rejects_empty_widget(self, qapp, tmp_path):
        empty = QWidget()
        empty.resize(QSize(0, 0))
        with pytest.raises(ValueError, match="empty size"):
            WindowExporter.export_pdf(empty, tmp_path / "x.pdf")

    def test_creates_parent_directory(self, small_widget, tmp_path):
        out = tmp_path / "deep" / "nested" / "smoke.pdf"
        WindowExporter.export_pdf(small_widget, out)
        assert out.exists()


# ---------------------------------------------------------------------------
# export_both
# ---------------------------------------------------------------------------


class TestExportBoth:

    def test_writes_png_and_pdf(self, small_widget, tmp_path):
        png_path, pdf_path = WindowExporter.export_both(
            small_widget, tmp_path, "smoke",
        )
        assert png_path == tmp_path / "smoke.png"
        assert pdf_path == tmp_path / "smoke.pdf"
        assert png_path.exists() and pdf_path.exists()
        assert png_path.stat().st_size > 500
        assert pdf_path.stat().st_size > 500
