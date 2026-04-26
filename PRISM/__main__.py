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
from pathlib import Path

from PyQt6.QtCore import QtMsgType, qInstallMessageHandler
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QApplication

from PRISM.core.config import AppConfig
from PRISM.gui.main_window import MainWindow
from PRISM.gui.themes.theme_manager import ThemeManager


# Repo-root ``assets/Logo_modify.png`` -- bundled with the source tree.
# Used as the QApplication-wide window icon so it appears in the title
# bar and OS task switcher.
_LOGO_PATH = Path(__file__).resolve().parent.parent / "assets" / "Logo_modify.png"


# Application User Model ID (Windows-only).  Without this, Windows groups
# PRISM under the host ``python.exe`` AUMID and the taskbar shows
# Python's icon instead of ours.  Setting it explicitly *before* the
# QApplication starts gives the running process its own taskbar slot
# with the correct icon.  No-op on non-Windows platforms.
_APP_USER_MODEL_ID = "PRISM.Simulator.QuantumCircuit.1.0"


def _set_windows_app_user_model_id(aumid: str) -> None:
    """Tag the current Windows process with an explicit AppUserModelID.

    Falls back silently when:

    * we're not on Windows (any non-``win32`` platform);
    * ``ctypes`` cannot reach ``shell32`` (e.g. an unusual Wine layer);
    * the API call returns a failure HRESULT.

    The icon may still appear correctly without this on Windows 10/11
    when running from a shortcut, but interactive launches via
    ``python -m PRISM`` need the explicit AUMID to override Python's
    inherited identity.
    """
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(aumid)
    except (AttributeError, OSError, ImportError):
        # Older Windows / restricted environments may lack the symbol.
        # Failing here is harmless -- the icon just falls back.
        pass


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
    # Tag the Windows process *before* the QApplication exists so the
    # taskbar groups our window under the PRISM AUMID instead of the
    # parent ``python.exe`` AUMID.  Harmless on Linux / macOS.
    _set_windows_app_user_model_id(_APP_USER_MODEL_ID)

    qInstallMessageHandler(_qt_message_handler)
    app = QApplication(sys.argv)

    app.setApplicationName("PRISM")
    app.setOrganizationName("PRISM")
    app.setApplicationVersion("1.0.0")

    # Build the icon once and apply it both app-wide *and* per-window.
    # Some Windows builds only honour the per-window icon for the
    # taskbar entry; setting both gives the highest chance of the logo
    # surfacing in every OS chrome surface (title bar, taskbar, alt-tab,
    # task switcher).  Falls back to a default icon if the PNG is
    # missing.
    if _LOGO_PATH.exists():
        icon = QIcon(str(_LOGO_PATH))
        app.setWindowIcon(icon)
    else:
        icon = None

    config = AppConfig.load()
    theme_mgr = ThemeManager(app)
    theme_mgr.apply_theme(config.theme)

    window = MainWindow(app=app)
    window.setWindowTitle("PRISM")
    if icon is not None:
        window.setWindowIcon(icon)
    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
