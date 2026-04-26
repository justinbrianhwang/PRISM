"""Theme manager for switching between dark and light QSS stylesheets."""

from __future__ import annotations

from pathlib import Path
from PyQt6.QtWidgets import QApplication


class ThemeManager:
    """Manages application themes by loading and applying QSS stylesheets.

    Supports switching between 'dark' (Catppuccin Mocha) and 'light'
    (Catppuccin Latte) themes at runtime.
    """

    THEMES = {'dark': 'dark.qss', 'light': 'light.qss'}

    def __init__(self, app: QApplication):
        self._app = app
        self._current_theme = 'dark'

    @property
    def current_theme(self) -> str:
        """Returns the name of the currently applied theme."""
        return self._current_theme

    def apply_theme(self, theme_name: str):
        """Apply the specified theme by loading its QSS file.

        Args:
            theme_name: Either 'dark' or 'light'.

        Raises:
            KeyError: If theme_name is not a recognized theme.
            FileNotFoundError: If the QSS file does not exist.
        """
        if theme_name not in self.THEMES:
            raise KeyError(
                f"Unknown theme '{theme_name}'. "
                f"Available themes: {list(self.THEMES.keys())}"
            )
        qss_path = Path(__file__).parent / self.THEMES[theme_name]
        with open(qss_path, 'r', encoding='utf-8') as f:
            self._app.setStyleSheet(f.read())
        self._current_theme = theme_name

    def toggle_theme(self):
        """Toggle between dark and light themes."""
        new_theme = 'light' if self._current_theme == 'dark' else 'dark'
        self.apply_theme(new_theme)
