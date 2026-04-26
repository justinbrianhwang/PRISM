"""PRISM application entry point.

Run with::

    python -m PRISM
    ./PRISM.sh           # cross-platform launcher (preferred)

Both routes execute :func:`main` below, which boots the Qt application,
applies the configured theme, and shows the main window.
"""

from __future__ import annotations

import sys
import warnings

from PyQt6.QtCore import QtMsgType, qInstallMessageHandler
from PyQt6.QtWidgets import QApplication

from PRISM.core.config import AppConfig
from PRISM.gui.main_window import MainWindow
from PRISM.gui.themes.theme_manager import ThemeManager


# Qt warnings we know to be benign (font sizing on high-DPI Windows).
_SUPPRESSED_QT_PATTERNS = (
    "setPointSize",
    "QFont::",
)


def _qt_message_handler(msg_type, _context, message):
    """Qt message handler that filters known harmless warnings."""
    if msg_type == QtMsgType.QtWarningMsg:
        for pattern in _SUPPRESSED_QT_PATTERNS:
            if pattern in message:
                return
    if msg_type in (QtMsgType.QtCriticalMsg, QtMsgType.QtFatalMsg):
        print(message, file=sys.stderr)


# Matplotlib emits set_ticklabels / tight_layout UserWarnings that we
# cannot fix at the call site without breaking other panels.
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")


def main() -> int:
    qInstallMessageHandler(_qt_message_handler)
    app = QApplication(sys.argv)

    app.setApplicationName("PRISM")
    app.setOrganizationName("PRISM")
    app.setApplicationVersion("1.0.0")

    config = AppConfig.load()
    theme_mgr = ThemeManager(app)
    theme_mgr.apply_theme(config.theme)

    window = MainWindow(app=app)
    window.setWindowTitle("PRISM")
    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
