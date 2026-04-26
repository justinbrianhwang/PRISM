"""Analysis dashboard panel -- pure PyQt6 widget showing quantum state metrics.

Displays four grouped sections:
1. State Properties  (purity, entropy, amplitude count, qubit count)
2. Fidelity          (vs a stored reference state)
3. Pauli Expectations (per-qubit <X>, <Y>, <Z>)
4. Entanglement      (bipartite entropy, concurrence, separability)

No matplotlib dependency -- all rendering is done with QLabel and QFormLayout.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QFormLayout,
    QGroupBox,
    QLabel,
    QScrollArea,
)

from quantum_sim.engine.state_vector import StateVector
from quantum_sim.engine.analysis import StateAnalysis


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _quality_color(value: float) -> str:
    """Return a hex colour string based on a quality metric in [0, 1]."""
    if value > 0.99:
        return "#2ECC71"  # green
    elif value > 0.9:
        return "#F39C12"  # yellow / orange
    else:
        return "#E74C3C"  # red


def _colored_label(text: str, color: str) -> str:
    """Wrap *text* in an inline style span with the given colour."""
    return f'<span style="color:{color};">{text}</span>'


# ---------------------------------------------------------------------------
# AnalysisPanel
# ---------------------------------------------------------------------------

class AnalysisPanel(QWidget):
    """Comprehensive quantum-state analysis dashboard.

    Intended to be embedded as a tab inside the main window.
    Call :meth:`update_state` after every simulation step to refresh
    all displayed metrics.
    """

    _MAX_PAULI_QUBITS = 8  # display cap for per-qubit Pauli rows

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)

        self._state: StateVector | None = None
        self._reference_state: StateVector | None = None
        self._dark_theme: bool = True

        # Value labels that get updated dynamically
        self._lbl_purity: QLabel | None = None
        self._lbl_entropy: QLabel | None = None
        self._lbl_nonzero: QLabel | None = None
        self._lbl_num_qubits: QLabel | None = None

        self._lbl_fidelity: QLabel | None = None

        self._pauli_labels: list[QLabel] = []
        self._pauli_layout: QFormLayout | None = None

        self._lbl_bipartite: QLabel | None = None
        self._lbl_concurrence: QLabel | None = None
        self._lbl_separability: QLabel | None = None

        self._setup_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)

        # Scroll area wrapping all content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        root_layout.addWidget(scroll)

        container = QWidget()
        self._container_layout = QVBoxLayout(container)
        self._container_layout.setContentsMargins(6, 6, 6, 6)
        self._container_layout.setSpacing(8)

        # -- 1. State Properties --
        self._grp_state = QGroupBox("State Properties")
        form_state = QFormLayout()
        form_state.setHorizontalSpacing(12)
        form_state.setVerticalSpacing(4)

        self._lbl_purity = self._make_value_label("--")
        form_state.addRow("Purity:", self._lbl_purity)

        self._lbl_entropy = self._make_value_label("--")
        form_state.addRow("Von Neumann Entropy:", self._lbl_entropy)

        self._lbl_nonzero = self._make_value_label("--")
        form_state.addRow("Non-zero amplitudes:", self._lbl_nonzero)

        self._lbl_num_qubits = self._make_value_label("--")
        form_state.addRow("Num qubits:", self._lbl_num_qubits)

        self._grp_state.setLayout(form_state)
        self._container_layout.addWidget(self._grp_state)

        # -- 2. Fidelity --
        self._grp_fidelity = QGroupBox("Fidelity")
        form_fidelity = QFormLayout()
        form_fidelity.setHorizontalSpacing(12)
        form_fidelity.setVerticalSpacing(4)

        self._lbl_fidelity = self._make_value_label("No reference state set")
        form_fidelity.addRow("Fidelity vs reference:", self._lbl_fidelity)

        self._grp_fidelity.setLayout(form_fidelity)
        self._container_layout.addWidget(self._grp_fidelity)

        # -- 3. Pauli Expectations --
        self._grp_pauli = QGroupBox("Pauli Expectations")
        self._pauli_layout = QFormLayout()
        self._pauli_layout.setHorizontalSpacing(12)
        self._pauli_layout.setVerticalSpacing(2)
        self._grp_pauli.setLayout(self._pauli_layout)
        self._container_layout.addWidget(self._grp_pauli)

        # -- 4. Entanglement --
        self._grp_entanglement = QGroupBox("Entanglement")
        form_ent = QFormLayout()
        form_ent.setHorizontalSpacing(12)
        form_ent.setVerticalSpacing(4)

        self._lbl_bipartite = self._make_value_label("--")
        form_ent.addRow("Bipartite entropy:", self._lbl_bipartite)

        self._lbl_concurrence = self._make_value_label("--")
        form_ent.addRow("Concurrence (q0-q1):", self._lbl_concurrence)

        self._lbl_separability = self._make_value_label("--")
        form_ent.addRow("Status:", self._lbl_separability)

        self._grp_entanglement.setLayout(form_ent)
        self._container_layout.addWidget(self._grp_entanglement)

        self._container_layout.addStretch()

        scroll.setWidget(container)

        # Apply initial theme
        self._apply_theme()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_state(self, state: StateVector) -> None:
        """Recompute every metric and refresh all labels."""
        self._state = state
        self._update_state_properties(state)
        self._update_fidelity(state)
        self._update_pauli(state)
        self._update_entanglement(state)

    def set_reference_state(self, state: StateVector) -> None:
        """Store a reference state for fidelity comparison."""
        self._reference_state = state
        # If we already have a current state, refresh fidelity immediately
        if self._state is not None:
            self._update_fidelity(self._state)

    def clear(self) -> None:
        """Reset all displayed values to placeholder dashes."""
        self._state = None

        self._lbl_purity.setText("--")
        self._lbl_entropy.setText("--")
        self._lbl_nonzero.setText("--")
        self._lbl_num_qubits.setText("--")

        self._lbl_fidelity.setText(
            "No reference state set"
            if self._reference_state is None
            else "--"
        )

        self._clear_pauli_rows()

        self._lbl_bipartite.setText("--")
        self._lbl_concurrence.setText("--")
        self._lbl_separability.setText("--")

    def set_theme(self, dark: bool) -> None:
        """Toggle between dark and light colour scheme."""
        self._dark_theme = dark
        self._apply_theme()

    # ------------------------------------------------------------------
    # Section updaters
    # ------------------------------------------------------------------

    def _update_state_properties(self, state: StateVector) -> None:
        # Purity
        purity = StateAnalysis.purity(state)
        color = _quality_color(purity)
        self._lbl_purity.setText(
            _colored_label(f"{purity:.6f}", color)
        )
        self._lbl_purity.setTextFormat(Qt.TextFormat.RichText)

        # Von Neumann entropy
        entropy = StateAnalysis.von_neumann_entropy(state)
        self._lbl_entropy.setText(f"{entropy:.6f} bits")

        # Non-zero amplitudes
        import numpy as np
        nonzero = int(np.count_nonzero(np.abs(state.data) > 1e-10))
        total = 2 ** state.num_qubits
        self._lbl_nonzero.setText(f"{nonzero} / {total}")

        # Num qubits
        self._lbl_num_qubits.setText(str(state.num_qubits))

    def _update_fidelity(self, state: StateVector) -> None:
        if self._reference_state is None:
            self._lbl_fidelity.setText("No reference state set")
            return

        if state.num_qubits != self._reference_state.num_qubits:
            self._lbl_fidelity.setText(
                _colored_label("Qubit count mismatch", "#E74C3C")
            )
            self._lbl_fidelity.setTextFormat(Qt.TextFormat.RichText)
            return

        fidelity = StateAnalysis.state_fidelity(
            state.data, self._reference_state.data
        )
        color = _quality_color(fidelity)
        self._lbl_fidelity.setText(
            _colored_label(f"{fidelity:.6f}", color)
        )
        self._lbl_fidelity.setTextFormat(Qt.TextFormat.RichText)

    def _update_pauli(self, state: StateVector) -> None:
        self._clear_pauli_rows()

        n = state.num_qubits
        display_n = min(n, self._MAX_PAULI_QUBITS)

        for q in range(display_n):
            ex = StateAnalysis.pauli_expectation(state, "X", q)
            ey = StateAnalysis.pauli_expectation(state, "Y", q)
            ez = StateAnalysis.pauli_expectation(state, "Z", q)

            # Determine the dominant axis for colour highlighting
            vals = {"X": abs(ex), "Y": abs(ey), "Z": abs(ez)}
            dominant = max(vals, key=vals.get)

            parts: list[str] = []
            for axis, val in [("X", ex), ("Y", ey), ("Z", ez)]:
                formatted = f"{axis}={val:+.2f}"
                if axis == dominant and vals[dominant] > 0.01:
                    parts.append(
                        f'<b style="color:#56B6C2;">{formatted}</b>'
                    )
                else:
                    parts.append(formatted)

            lbl = self._make_value_label("")
            lbl.setTextFormat(Qt.TextFormat.RichText)
            lbl.setText("  ".join(parts))
            self._pauli_layout.addRow(f"q{q}:", lbl)
            self._pauli_labels.append(lbl)

        if n > self._MAX_PAULI_QUBITS:
            ellipsis_lbl = self._make_value_label(
                f"... ({n - self._MAX_PAULI_QUBITS} more qubits)"
            )
            self._pauli_layout.addRow("", ellipsis_lbl)
            self._pauli_labels.append(ellipsis_lbl)

    def _update_entanglement(self, state: StateVector) -> None:
        n = state.num_qubits

        if n < 2:
            self._lbl_bipartite.setText("N/A (single qubit)")
            self._lbl_concurrence.setText("N/A (single qubit)")
            self._lbl_separability.setText("Separable")
            return

        # Bipartite entropy: first n//2 qubits vs last n//2 qubits
        half = n // 2
        subsystem_a = list(range(half))
        bipartite = StateAnalysis.entanglement_entropy(state, subsystem_a)
        self._lbl_bipartite.setText(
            f"{bipartite:.6f} bits  (q0..q{half - 1} | q{half}..q{n - 1})"
        )

        # Concurrence for q0-q1
        conc = StateAnalysis.concurrence(state, 0, 1)
        self._lbl_concurrence.setText(f"{conc:.6f}")

        # Separability label
        if bipartite > 0.01:
            self._lbl_separability.setText(
                _colored_label("Entangled", "#56B6C2")
            )
        else:
            self._lbl_separability.setText(
                _colored_label("Separable", "#2ECC71")
            )
        self._lbl_separability.setTextFormat(Qt.TextFormat.RichText)

    # ------------------------------------------------------------------
    # Theme
    # ------------------------------------------------------------------

    def _apply_theme(self) -> None:
        """Apply dark or light colour scheme to all group boxes and labels."""
        if self._dark_theme:
            title_color = "#ffffff"
            value_color = "#cdd6f4"
            border_color = "#45475a"
            bg = "transparent"
        else:
            title_color = "#000000"
            value_color = "#333333"
            border_color = "#cccccc"
            bg = "transparent"

        group_style = (
            f"QGroupBox {{"
            f"  font-weight: bold;"
            f"  font-size: 13px;"
            f"  color: {title_color};"
            f"  border: 1px solid {border_color};"
            f"  border-radius: 4px;"
            f"  margin-top: 10px;"
            f"  padding-top: 14px;"
            f"  background: {bg};"
            f"}}"
            f"QGroupBox::title {{"
            f"  subcontrol-origin: margin;"
            f"  left: 8px;"
            f"  padding: 0 4px;"
            f"  color: {title_color};"
            f"}}"
        )

        for grp in (
            self._grp_state,
            self._grp_fidelity,
            self._grp_pauli,
            self._grp_entanglement,
        ):
            grp.setStyleSheet(group_style)

        # Update all existing value labels
        self._value_label_color = value_color
        self._update_value_label_colors()

    def _update_value_label_colors(self) -> None:
        """Refresh the base text colour of every value label."""
        color = getattr(self, "_value_label_color", "#cdd6f4")
        base_style = f"font-size: 12px; color: {color};"

        for lbl in (
            self._lbl_purity,
            self._lbl_entropy,
            self._lbl_nonzero,
            self._lbl_num_qubits,
            self._lbl_fidelity,
            self._lbl_bipartite,
            self._lbl_concurrence,
            self._lbl_separability,
        ):
            if lbl is not None:
                lbl.setStyleSheet(base_style)

        for lbl in self._pauli_labels:
            lbl.setStyleSheet(base_style)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_value_label(self, text: str) -> QLabel:
        """Create a value QLabel with consistent styling."""
        lbl = QLabel(text)
        lbl.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        color = getattr(self, "_value_label_color", "#cdd6f4")
        lbl.setStyleSheet(f"font-size: 12px; color: {color};")
        return lbl

    def _clear_pauli_rows(self) -> None:
        """Remove all dynamically created Pauli expectation rows."""
        while self._pauli_layout.rowCount() > 0:
            self._pauli_layout.removeRow(0)
        self._pauli_labels.clear()
