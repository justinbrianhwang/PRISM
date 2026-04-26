"""Application configuration management."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AppConfig:
    """Persistent application configuration."""
    theme: str = "dark"
    default_qubits: int = 4
    default_shots: int = 1024
    step_delay_ms: int = 500
    max_qubits: int = 16
    window_width: int = 1400
    window_height: int = 900
    recent_files: list[str] = field(default_factory=list)
    last_directory: str = ""

    _config_dir: Path = field(
        default_factory=lambda: Path.home() / ".quantum_sim",
        repr=False)

    @property
    def config_path(self) -> Path:
        return self._config_dir / "config.json"

    def save(self):
        self._config_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "theme": self.theme,
            "default_qubits": self.default_qubits,
            "default_shots": self.default_shots,
            "step_delay_ms": self.step_delay_ms,
            "max_qubits": self.max_qubits,
            "window_width": self.window_width,
            "window_height": self.window_height,
            "recent_files": self.recent_files[-10:],  # Keep last 10
            "last_directory": self.last_directory,
        }
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls) -> AppConfig:
        config = cls()
        if config.config_path.exists():
            try:
                with open(config.config_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                for key, value in data.items():
                    if hasattr(config, key) and not key.startswith('_'):
                        setattr(config, key, value)
            except (json.JSONDecodeError, OSError):
                pass  # Use defaults
        return config

    def add_recent_file(self, filepath: str):
        if filepath in self.recent_files:
            self.recent_files.remove(filepath)
        self.recent_files.insert(0, filepath)
        self.recent_files = self.recent_files[:10]
