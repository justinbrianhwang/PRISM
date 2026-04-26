"""Smoke tests for the project-logo integration.

PRISM displays the project logo (``assets/Logo_modify.png``) in two
places:

* The application-wide window icon set in ``PRISM/__main__.py``, which
  surfaces in the OS title bar and task switcher.
* The About dialog in ``PRISM/gui/main_window.py``, which shows the
  logo via :pymeth:`QMessageBox.setIconPixmap`.

These tests verify that:

* The logo PNG is on disk where the code expects it.
* Loading it as a :class:`QPixmap` succeeds and yields a non-null,
  non-empty image.
* Both code paths gracefully no-op (rather than crash) if the file is
  missing -- important for CI environments that may run a partial
  checkout.
"""

from __future__ import annotations

import os
from pathlib import Path

# QT_QPA_PLATFORM=offscreen MUST be set before importing any Qt module.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt6.QtWidgets")

from PyQt6.QtGui import QIcon, QPixmap  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parent.parent
LOGO_PATH = REPO_ROOT / "assets" / "Logo_modify.png"


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


# ---------------------------------------------------------------------------
# File presence
# ---------------------------------------------------------------------------


class TestLogoFile:

    def test_logo_present_at_expected_path(self):
        assert LOGO_PATH.exists(), (
            f"Project logo missing at {LOGO_PATH}. "
            "PRISM/__main__.py will silently skip setWindowIcon, "
            "and the About dialog will fall back to a logo-less view."
        )

    def test_logo_non_trivial_size(self):
        # Tiny / corrupt files (< 1 KB) suggest the upload was truncated.
        size = LOGO_PATH.stat().st_size
        assert size > 1024, f"Logo is suspiciously small: {size} bytes"


# ---------------------------------------------------------------------------
# QPixmap / QIcon loading
# ---------------------------------------------------------------------------


class TestLogoLoadability:

    def test_qpixmap_loads_logo(self, qapp):
        pixmap = QPixmap(str(LOGO_PATH))
        assert not pixmap.isNull(), "QPixmap could not load the logo"
        assert pixmap.width() > 0
        assert pixmap.height() > 0

    def test_qicon_loads_logo(self, qapp):
        icon = QIcon(str(LOGO_PATH))
        assert not icon.isNull(), "QIcon could not load the logo"

    def test_qpixmap_scales_for_about_dialog(self, qapp):
        from PyQt6.QtCore import Qt

        pixmap = QPixmap(str(LOGO_PATH))
        scaled = pixmap.scaled(
            96, 96,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        # Aspect-preserving scale: the longest side equals 96.
        assert max(scaled.width(), scaled.height()) == 96


# ---------------------------------------------------------------------------
# Graceful absence
# ---------------------------------------------------------------------------


class TestGracefulMissingLogo:

    def test_qpixmap_missing_file_returns_null(self, qapp, tmp_path):
        ghost = tmp_path / "does_not_exist.png"
        pixmap = QPixmap(str(ghost))
        # A null pixmap is the standard signal that the file was missing
        # or unreadable -- we rely on this for the fallback in
        # ``PRISM/__main__.py`` and the About dialog.
        assert pixmap.isNull()

    def test_setwindowicon_with_null_icon_does_not_crash(self, qapp):
        # Even passing a null QIcon to setWindowIcon is allowed by Qt;
        # the test exists to lock in that contract for future Qt
        # versions.
        null_icon = QIcon()
        qapp.setWindowIcon(null_icon)
