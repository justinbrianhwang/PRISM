"""Real-time resource monitor panel.

Shows CPU/memory usage, simulation timing, and resource efficiency
comparison with other quantum simulators.

Uses Windows ctypes APIs directly for zero extra dependencies.
Optional: install ``psutil`` for more accurate CPU % readings.
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes as wintypes
import os
import sys
import time

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QGroupBox,
    QGridLayout, QProgressBar, QScrollArea,
)

import numpy as np

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.figure import Figure
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


# ---- Windows ctypes helpers (no extra deps) --------------------------------

class _PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


class _MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", wintypes.DWORD),
        ("dwMemoryLoad", wintypes.DWORD),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def _win_process_memory() -> float:
    """Return RSS (Working Set) in bytes using Win32 API."""
    try:
        pmc = _PROCESS_MEMORY_COUNTERS()
        pmc.cb = ctypes.sizeof(pmc)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        ctypes.windll.psapi.GetProcessMemoryInfo(
            handle, ctypes.byref(pmc), pmc.cb
        )
        return float(pmc.WorkingSetSize)
    except Exception:
        return 0.0


def _win_system_memory() -> tuple[float, float, float]:
    """Return (total_bytes, available_bytes, used_pct) via GlobalMemoryStatusEx."""
    try:
        stat = _MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(stat)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        return float(stat.ullTotalPhys), float(stat.ullAvailPhys), float(stat.dwMemoryLoad)
    except Exception:
        return 8.0 * 1024**3, 4.0 * 1024**3, 50.0


# ---- Unified resource getters ----------------------------------------------

def _get_process_info() -> dict:
    """Get current process resource usage."""
    if HAS_PSUTIL:
        proc = psutil.Process(os.getpid())
        mem_info = proc.memory_info()
        try:
            cpu_pct = proc.cpu_percent(interval=0)
        except Exception:
            cpu_pct = 0.0
        return {
            "rss_mb": mem_info.rss / (1024 * 1024),
            "cpu_pct": cpu_pct,
            "threads": proc.num_threads(),
        }

    # Fallback: Windows ctypes
    rss = _win_process_memory()
    return {
        "rss_mb": rss / (1024 * 1024),
        "cpu_pct": 0.0,  # CPU% not available without psutil
        "threads": 0,
    }


def _get_system_info() -> dict:
    """Get system-wide resource info."""
    if HAS_PSUTIL:
        vm = psutil.virtual_memory()
        return {
            "total_mb": vm.total / (1024 * 1024),
            "available_mb": vm.available / (1024 * 1024),
            "used_pct": vm.percent,
            "cpu_count": psutil.cpu_count(logical=True),
            "cpu_pct": psutil.cpu_percent(interval=0),
        }

    total, avail, used_pct = _win_system_memory()
    return {
        "total_mb": total / (1024 * 1024),
        "available_mb": avail / (1024 * 1024),
        "used_pct": used_pct,
        "cpu_count": os.cpu_count() or 1,
        "cpu_pct": 0.0,
    }


# ---- Panel -----------------------------------------------------------------

class ResourceMonitorPanel(QWidget):
    """Real-time resource monitoring panel with simulator comparison."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._sim_elapsed: float = 0.0
        self._last_sim_qubits: int = 0
        self._last_sim_gates: int = 0
        self._peak_mem_mb: float = 0.0

        # History for plots
        self._mem_history: list[float] = []
        self._cpu_history: list[float] = []
        self._time_history: list[float] = []
        self._history_start: float = time.time()
        self._max_history = 120  # 2 minutes at 1 Hz

        self._build_ui()

        # Update timer (1 second interval)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update_stats)
        self._timer.start(1000)

        # Seed psutil CPU baseline
        if HAS_PSUTIL:
            try:
                psutil.Process(os.getpid()).cpu_percent(interval=0)
            except Exception:
                pass

    # ---- UI construction ---------------------------------------------------

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        outer.addWidget(scroll)

        container = QWidget()
        scroll.setWidget(container)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # --- Process Resources ---
        proc_group = QGroupBox("Process Resources")
        proc_layout = QGridLayout(proc_group)
        proc_layout.setSpacing(4)

        self._lbl_cpu = QLabel("CPU: --")
        self._lbl_cpu.setStyleSheet("font-weight: bold; font-size: 13px;")
        self._bar_cpu = QProgressBar()
        self._bar_cpu.setRange(0, 100)
        self._bar_cpu.setTextVisible(True)
        self._bar_cpu.setFixedHeight(18)

        self._lbl_mem = QLabel("Memory: --")
        self._lbl_mem.setStyleSheet("font-weight: bold; font-size: 13px;")
        self._bar_mem = QProgressBar()
        self._bar_mem.setRange(0, 100)
        self._bar_mem.setTextVisible(True)
        self._bar_mem.setFixedHeight(18)

        self._lbl_threads = QLabel("Threads: --")
        self._lbl_peak_mem = QLabel("Peak: --")

        proc_layout.addWidget(self._lbl_cpu, 0, 0)
        proc_layout.addWidget(self._bar_cpu, 0, 1)
        proc_layout.addWidget(self._lbl_mem, 1, 0)
        proc_layout.addWidget(self._bar_mem, 1, 1)
        proc_layout.addWidget(self._lbl_threads, 2, 0)
        proc_layout.addWidget(self._lbl_peak_mem, 2, 1)

        if not HAS_PSUTIL:
            hint = QLabel("(Install psutil for CPU % monitoring)")
            hint.setStyleSheet("font-size: 10px; color: #888;")
            proc_layout.addWidget(hint, 3, 0, 1, 2)

        layout.addWidget(proc_group)

        # --- Simulation Timing ---
        sim_group = QGroupBox("Simulation")
        sim_layout = QGridLayout(sim_group)
        sim_layout.setSpacing(4)

        self._lbl_sim_time = QLabel("Last Sim: --")
        self._lbl_sim_time.setStyleSheet("font-weight: bold; font-size: 13px;")
        self._lbl_sim_qubits = QLabel("Qubits: --")
        self._lbl_sim_gates = QLabel("Gates: --")
        self._lbl_state_size = QLabel("State Vector: --")

        sim_layout.addWidget(self._lbl_sim_time, 0, 0)
        sim_layout.addWidget(self._lbl_sim_qubits, 0, 1)
        sim_layout.addWidget(self._lbl_sim_gates, 1, 0)
        sim_layout.addWidget(self._lbl_state_size, 1, 1)
        layout.addWidget(sim_group)

        # --- Resource Usage Plot ---
        if HAS_MPL:
            plot_group = QGroupBox("Resource History (2 min)")
            plot_layout = QVBoxLayout(plot_group)
            plot_layout.setContentsMargins(2, 2, 2, 2)

            self._fig = Figure(figsize=(5, 1.8), dpi=80)
            self._fig.set_facecolor("none")
            self._canvas = FigureCanvasQTAgg(self._fig)
            self._canvas.setFixedHeight(140)
            self._ax_mem = self._fig.add_subplot(121)
            self._ax_cpu = self._fig.add_subplot(122)
            plot_layout.addWidget(self._canvas)
            layout.addWidget(plot_group)

        # --- Simulator Comparison ---
        cmp_group = QGroupBox("Resource Advantage vs Other Simulators")
        cmp_layout = QVBoxLayout(cmp_group)
        cmp_layout.setSpacing(2)
        cmp_layout.setContentsMargins(6, 6, 6, 6)

        self._cmp_label = QLabel()
        self._cmp_label.setWordWrap(True)
        self._cmp_label.setStyleSheet("font-size: 11px;")
        cmp_layout.addWidget(self._cmp_label)
        layout.addWidget(cmp_group)

        layout.addStretch(1)

        # Set initial comparison text
        self._update_comparison_text(0)

    # ---- Real-time update --------------------------------------------------

    def _update_stats(self):
        """Called every second to refresh resource stats."""
        try:
            proc = _get_process_info()
            sys_info = _get_system_info()
        except Exception:
            return

        cpu = proc["cpu_pct"]
        mem_mb = proc["rss_mb"]
        total_mb = sys_info["total_mb"]
        mem_pct = (mem_mb / total_mb) * 100 if total_mb > 0 else 0

        if HAS_PSUTIL:
            self._lbl_cpu.setText(f"CPU: {cpu:.1f}%")
            self._bar_cpu.setValue(min(int(cpu), 100))
            self._bar_cpu.setFormat(f"{cpu:.1f}%")
        else:
            self._lbl_cpu.setText("CPU: N/A")
            self._bar_cpu.setValue(0)
            self._bar_cpu.setFormat("N/A")

        self._lbl_mem.setText(f"Memory: {mem_mb:.1f} MB")
        self._bar_mem.setValue(min(int(mem_pct), 100))
        self._bar_mem.setFormat(f"{mem_mb:.0f} MB ({mem_pct:.1f}%)")

        if proc["threads"] > 0:
            self._lbl_threads.setText(f"Threads: {proc['threads']}")
        else:
            self._lbl_threads.setText("Threads: --")

        if mem_mb > self._peak_mem_mb:
            self._peak_mem_mb = mem_mb
        self._lbl_peak_mem.setText(f"Peak: {self._peak_mem_mb:.1f} MB")

        # Color bars
        if cpu > 80:
            self._bar_cpu.setStyleSheet("QProgressBar::chunk { background: #e74c3c; }")
        elif cpu > 50:
            self._bar_cpu.setStyleSheet("QProgressBar::chunk { background: #f39c12; }")
        else:
            self._bar_cpu.setStyleSheet("QProgressBar::chunk { background: #2ecc71; }")

        if mem_pct > 80:
            self._bar_mem.setStyleSheet("QProgressBar::chunk { background: #e74c3c; }")
        elif mem_pct > 50:
            self._bar_mem.setStyleSheet("QProgressBar::chunk { background: #f39c12; }")
        else:
            self._bar_mem.setStyleSheet("QProgressBar::chunk { background: #2ecc71; }")

        # Record history
        t = time.time() - self._history_start
        self._time_history.append(t)
        self._mem_history.append(mem_mb)
        self._cpu_history.append(cpu)

        if len(self._time_history) > self._max_history:
            self._time_history = self._time_history[-self._max_history:]
            self._mem_history = self._mem_history[-self._max_history:]
            self._cpu_history = self._cpu_history[-self._max_history:]

        if HAS_MPL and len(self._time_history) > 1:
            self._draw_history()

    def _draw_history(self):
        """Redraw the resource history mini-plots."""
        times = np.array(self._time_history)
        t_rel = times - times[0]

        for ax in (self._ax_mem, self._ax_cpu):
            ax.clear()
            ax.tick_params(labelsize=7)

        self._ax_mem.fill_between(t_rel, self._mem_history, alpha=0.3, color="#3498db")
        self._ax_mem.plot(t_rel, self._mem_history, color="#3498db", linewidth=1)
        self._ax_mem.set_ylabel("MB", fontsize=7)
        self._ax_mem.set_title("Memory", fontsize=8)

        self._ax_cpu.fill_between(t_rel, self._cpu_history, alpha=0.3, color="#e74c3c")
        self._ax_cpu.plot(t_rel, self._cpu_history, color="#e74c3c", linewidth=1)
        self._ax_cpu.set_ylabel("%", fontsize=7)
        self._ax_cpu.set_title("CPU", fontsize=8)

        self._fig.tight_layout(pad=0.5)
        self._canvas.draw_idle()

    # ---- Simulation recording ----------------------------------------------

    def record_simulation(self, n_qubits: int, n_gates: int, elapsed: float):
        """Record a completed simulation's resource usage."""
        self._sim_elapsed = elapsed
        self._last_sim_qubits = n_qubits
        self._last_sim_gates = n_gates

        if elapsed < 1.0:
            time_str = f"{elapsed * 1000:.1f} ms"
        else:
            time_str = f"{elapsed:.3f} s"

        self._lbl_sim_time.setText(f"Last Sim: {time_str}")
        self._lbl_sim_qubits.setText(f"Qubits: {n_qubits}")
        self._lbl_sim_gates.setText(f"Gates: {n_gates}")

        sv_size = 2 ** n_qubits
        sv_bytes = sv_size * 16  # complex128 = 16 bytes
        self._lbl_state_size.setText(
            f"State: 2^{n_qubits} = {sv_size} ({self._fmt_bytes(sv_bytes)})"
        )
        self._update_comparison_text(n_qubits)

    # ---- Simulator comparison ----------------------------------------------

    def _update_comparison_text(self, n_qubits: int):
        """Show resource advantage vs other simulators."""
        if n_qubits <= 0:
            self._cmp_label.setText(
                "<b>This Simulator (State Vector + Stochastic Noise)</b><br>"
                "Run a simulation to see resource comparison."
            )
            return

        sv_bytes = (2 ** n_qubits) * 16
        dm_bytes = (2 ** n_qubits) ** 2 * 16

        lines: list[str] = []
        lines.append("<b>This Simulator (State Vector + Stochastic Noise)</b>")
        lines.append(f"  State vector: {self._fmt_bytes(sv_bytes)}")
        lines.append(f"  Tensor contraction: O(2^n * 4^k) per gate")
        lines.append("")

        ratio = dm_bytes / sv_bytes if sv_bytes > 0 else 0
        lines.append("<b>vs Density Matrix Simulators</b> (Qiskit Aer DM, QuTiP)")
        lines.append(f"  Would need: {self._fmt_bytes(dm_bytes)}")
        lines.append(
            f"  Memory saving: <span style='color:#2ecc71;font-weight:bold'>"
            f"{ratio:.0f}x less memory</span>"
        )
        lines.append("")

        our_max = self._max_qubits_for_ram(8 * 1024**3, mode="sv")
        dm_max = self._max_qubits_for_ram(8 * 1024**3, mode="dm")

        lines.append("<b>Scaling Comparison (max qubits in 8 GB RAM):</b>")
        lines.append("<table cellpadding='2' style='font-size:11px'>")
        lines.append(
            "<tr><td><b>Simulator</b></td><td><b>Method</b></td>"
            "<td><b>Max Qubits</b></td><td><b>Memory</b></td></tr>"
        )
        lines.append(
            f"<tr><td>This Sim</td><td>State Vector</td>"
            f"<td style='color:#2ecc71;font-weight:bold'>{our_max}</td>"
            f"<td>{self._fmt_bytes(2**our_max * 16)}</td></tr>"
        )
        lines.append(
            f"<tr><td>Qiskit SV</td><td>State Vector</td>"
            f"<td>{our_max}</td>"
            f"<td>{self._fmt_bytes(2**our_max * 16)}</td></tr>"
        )
        lines.append(
            f"<tr><td>QuTiP / DM</td><td>Density Matrix</td>"
            f"<td>{dm_max}</td>"
            f"<td>{self._fmt_bytes(2**(2*dm_max) * 16)}</td></tr>"
        )
        lines.append(
            "<tr><td>Qiskit MPS</td><td>Tensor Network</td>"
            "<td>50+</td><td>Depends on entanglement</td></tr>"
        )
        lines.append("</table>")
        lines.append("")
        lines.append("<b>Our advantages:</b>")
        lines.append("- Pure NumPy: no C++ compilation, instant setup")
        lines.append("- Tensor contraction: efficient multi-qubit gates")
        lines.append("- Stochastic noise: memory = state vector only")
        lines.append("- 3 dependencies only (PyQt6 + NumPy + Matplotlib)")

        self._cmp_label.setText("<br>".join(lines))

    # ---- Helpers -----------------------------------------------------------

    @staticmethod
    def _max_qubits_for_ram(ram_bytes: int, mode: str = "sv") -> int:
        if mode == "dm":
            n = 1
            while (2 ** (2 * n)) * 16 < ram_bytes:
                n += 1
            return n - 1
        n = 1
        while (2 ** n) * 16 < ram_bytes:
            n += 1
        return n - 1

    @staticmethod
    def _fmt_bytes(b: float) -> str:
        if b < 1024:
            return f"{b:.0f} B"
        elif b < 1024 ** 2:
            return f"{b / 1024:.1f} KB"
        elif b < 1024 ** 3:
            return f"{b / (1024**2):.1f} MB"
        return f"{b / (1024**3):.2f} GB"

    def set_theme(self, theme_name: str):
        """Apply theme to the panel."""
        if HAS_MPL:
            bg = "#1e1e2e" if theme_name == "dark" else "#ffffff"
            fg = "#cdd6f4" if theme_name == "dark" else "#000000"
            self._fig.set_facecolor(bg)
            for ax in (self._ax_mem, self._ax_cpu):
                ax.set_facecolor(bg)
                ax.tick_params(colors=fg)
                ax.xaxis.label.set_color(fg)
                ax.yaxis.label.set_color(fg)
                ax.title.set_color(fg)
                for spine in ax.spines.values():
                    spine.set_color(fg)

    def clear(self):
        """Reset simulation stats (keep resource monitoring running)."""
        self._lbl_sim_time.setText("Last Sim: --")
        self._lbl_sim_qubits.setText("Qubits: --")
        self._lbl_sim_gates.setText("Gates: --")
        self._lbl_state_size.setText("State Vector: --")
        self._update_comparison_text(0)
