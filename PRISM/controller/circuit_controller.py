"""Circuit controller connecting the circuit model to the circuit scene view.

Routes all circuit modifications through a QUndoStack for undo/redo support.
"""

from __future__ import annotations

from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtGui import QUndoStack, QUndoCommand

from quantum_sim.engine.circuit import QuantumCircuit, GateInstance
from quantum_sim.engine.gate_registry import GateRegistry
from quantum_sim.engine.algorithms import AlgorithmTemplate


# ---------------------------------------------------------------------------
# Undo / Redo command classes
# ---------------------------------------------------------------------------

class AddGateCommand(QUndoCommand):
    """Undoable command for adding a gate to the circuit."""

    def __init__(
        self,
        circuit: QuantumCircuit,
        gate: GateInstance,
        description: str = "",
    ):
        super().__init__(description or f"Add {gate.gate_name}")
        self._circuit = circuit
        self._gate = gate

    def redo(self) -> None:
        self._circuit.add_gate(self._gate)

    def undo(self) -> None:
        self._circuit.remove_gate(self._gate)


class RemoveGateCommand(QUndoCommand):
    """Undoable command for removing a gate from the circuit."""

    def __init__(
        self,
        circuit: QuantumCircuit,
        gate: GateInstance,
        description: str = "",
    ):
        super().__init__(description or f"Remove {gate.gate_name}")
        self._circuit = circuit
        self._gate = gate

    def redo(self) -> None:
        self._circuit.remove_gate(self._gate)

    def undo(self) -> None:
        self._circuit.add_gate(self._gate)


class MoveGateCommand(QUndoCommand):
    """Undoable command for moving a gate to a new position."""

    def __init__(
        self,
        circuit: QuantumCircuit,
        gate: GateInstance,
        new_col: int,
        new_targets: list[int],
        description: str = "",
    ):
        super().__init__(description or f"Move {gate.gate_name}")
        self._circuit = circuit
        self._gate = gate
        self._new_col = new_col
        self._new_targets = list(new_targets)
        self._old_col = gate.column
        self._old_targets = list(gate.target_qubits)

    def redo(self) -> None:
        self._circuit.move_gate(self._gate, self._new_col, self._new_targets)

    def undo(self) -> None:
        self._circuit.move_gate(self._gate, self._old_col, self._old_targets)


class UpdateGateParamsCommand(QUndoCommand):
    """Undoable command for updating gate parameters."""

    def __init__(
        self,
        gate: GateInstance,
        new_params: list[float],
        description: str = "",
    ):
        super().__init__(description or f"Update {gate.gate_name} params")
        self._gate = gate
        self._new_params = list(new_params)
        self._old_params = list(gate.params)

    def redo(self) -> None:
        self._gate.params = list(self._new_params)

    def undo(self) -> None:
        self._gate.params = list(self._old_params)


class SetQubitCountCommand(QUndoCommand):
    """Undoable command for changing the number of qubits."""

    def __init__(
        self,
        circuit: QuantumCircuit,
        new_count: int,
        description: str = "",
    ):
        super().__init__(description or f"Set {new_count} qubits")
        self._circuit = circuit
        self._new_count = new_count
        self._old_count = circuit.num_qubits
        # Save gates that would be removed
        self._removed_gates = [
            g for g in circuit.gates
            if any(q >= new_count for q in g.target_qubits)
        ]

    def redo(self) -> None:
        self._circuit.set_num_qubits(self._new_count)

    def undo(self) -> None:
        self._circuit.num_qubits = self._old_count
        for gate in self._removed_gates:
            if gate not in self._circuit.gates:
                self._circuit.gates.append(gate)


class ClearCircuitCommand(QUndoCommand):
    """Undoable command for clearing all gates from the circuit."""

    def __init__(
        self,
        circuit: QuantumCircuit,
        description: str = "Clear circuit",
    ):
        super().__init__(description)
        self._circuit = circuit
        self._saved_gates: list[GateInstance] = []

    def redo(self) -> None:
        self._saved_gates = list(self._circuit.gates)
        self._circuit.clear()

    def undo(self) -> None:
        for gate in self._saved_gates:
            self._circuit.add_gate(gate)


class LoadTemplateCommand(QUndoCommand):
    """Undoable command for loading an algorithm template."""

    def __init__(
        self,
        circuit: QuantumCircuit,
        template_circuit: QuantumCircuit,
        description: str = "Load template",
    ):
        super().__init__(description)
        self._circuit = circuit
        self._template_circuit = template_circuit
        self._old_gates: list[GateInstance] = []
        self._old_num_qubits: int = 0

    def redo(self) -> None:
        self._old_gates = list(self._circuit.gates)
        self._old_num_qubits = self._circuit.num_qubits
        self._circuit.gates.clear()
        self._circuit.num_qubits = self._template_circuit.num_qubits
        for gate in self._template_circuit.gates:
            self._circuit.add_gate(GateInstance(
                gate_name=gate.gate_name,
                target_qubits=list(gate.target_qubits),
                params=list(gate.params),
                column=gate.column,
            ))

    def undo(self) -> None:
        self._circuit.gates.clear()
        self._circuit.num_qubits = self._old_num_qubits
        for gate in self._old_gates:
            self._circuit.add_gate(gate)


# ---------------------------------------------------------------------------
# Circuit Controller
# ---------------------------------------------------------------------------

class CircuitController(QObject):
    """Controller that connects the QuantumCircuit model to the GUI scene.

    All modifications are routed through a QUndoStack to enable undo/redo.
    Emits circuit_changed whenever the circuit is modified.
    """

    circuit_changed = pyqtSignal()

    def __init__(
        self,
        circuit: QuantumCircuit | None = None,
        parent: QObject | None = None,
    ):
        super().__init__(parent)

        self._circuit = circuit or QuantumCircuit()
        self._undo_stack = QUndoStack(self)
        self._gate_registry = GateRegistry.instance()

        # Re-emit circuit_changed when the undo stack index changes
        self._undo_stack.indexChanged.connect(self._on_stack_changed)

    @property
    def circuit(self) -> QuantumCircuit:
        """The underlying circuit model."""
        return self._circuit

    @circuit.setter
    def circuit(self, new_circuit: QuantumCircuit) -> None:
        """Replace the circuit model."""
        self._circuit = new_circuit
        self._undo_stack.clear()
        self.circuit_changed.emit()

    @property
    def undo_stack(self) -> QUndoStack:
        """The undo stack for this controller."""
        return self._undo_stack

    def _on_stack_changed(self, _index: int) -> None:
        self.circuit_changed.emit()

    # ------------------------------------------------------------------
    # Public modification methods
    # ------------------------------------------------------------------

    def add_gate(
        self,
        gate_name: str,
        col: int,
        target_qubits: list[int],
        params: list[float] | None = None,
    ) -> None:
        """Add a gate to the circuit.

        Args:
            gate_name: Name of the gate (must exist in GateRegistry).
            col: Column index in the circuit.
            target_qubits: List of target qubit indices.
            params: Optional parameters for parameterized gates.
        """
        # Validate that the gate exists
        self._gate_registry.get(gate_name)  # Raises if not found

        gate = GateInstance(
            gate_name=gate_name,
            target_qubits=list(target_qubits),
            params=list(params) if params else [],
            column=col,
        )
        cmd = AddGateCommand(self._circuit, gate)
        self._undo_stack.push(cmd)

    def remove_gate(self, gate: GateInstance) -> None:
        """Remove a specific gate from the circuit."""
        if gate in self._circuit.gates:
            cmd = RemoveGateCommand(self._circuit, gate)
            self._undo_stack.push(cmd)

    def remove_selected_gates(self, gates: list[GateInstance]) -> None:
        """Remove multiple selected gates as a single undoable action.

        Args:
            gates: List of gate instances to remove.
        """
        if not gates:
            return

        self._undo_stack.beginMacro("Remove selected gates")
        for gate in gates:
            if gate in self._circuit.gates:
                cmd = RemoveGateCommand(self._circuit, gate)
                self._undo_stack.push(cmd)
        self._undo_stack.endMacro()

    def move_gate(
        self,
        gate: GateInstance,
        new_col: int,
        new_targets: list[int],
    ) -> None:
        """Move a gate to a new position.

        Args:
            gate: The gate instance to move.
            new_col: New column index.
            new_targets: New target qubit indices.
        """
        if gate not in self._circuit.gates:
            return
        cmd = MoveGateCommand(self._circuit, gate, new_col, new_targets)
        self._undo_stack.push(cmd)

    def update_gate_params(
        self,
        gate: GateInstance,
        new_params: list[float],
    ) -> None:
        """Update the parameters of a parameterized gate.

        Args:
            gate: The gate instance to update.
            new_params: New parameter values.
        """
        cmd = UpdateGateParamsCommand(gate, new_params)
        self._undo_stack.push(cmd)

    def set_qubit_count(self, count: int) -> None:
        """Set the number of qubits in the circuit.

        Gates targeting qubits beyond the new count will be removed.

        Args:
            count: New number of qubits (1-16).
        """
        if count == self._circuit.num_qubits:
            return
        cmd = SetQubitCountCommand(self._circuit, count)
        self._undo_stack.push(cmd)

    def clear_circuit(self) -> None:
        """Remove all gates from the circuit."""
        if not self._circuit.gates:
            return
        cmd = ClearCircuitCommand(self._circuit)
        self._undo_stack.push(cmd)

    def load_template(self, template_name: str, **kwargs) -> None:
        """Load a predefined algorithm template into the circuit.

        Replaces the current circuit contents.

        Args:
            template_name: Name of the algorithm template.
            **kwargs: Parameters for the template (e.g., num_qubits, marked_state).
        """
        template_circuit = self._build_template(template_name, **kwargs)
        if template_circuit is None:
            return

        cmd = LoadTemplateCommand(
            self._circuit,
            template_circuit,
            description=f"Load {template_name}",
        )
        self._undo_stack.push(cmd)

    def _build_template(
        self, template_name: str, **kwargs
    ) -> QuantumCircuit | None:
        """Build a QuantumCircuit from a template name and parameters."""
        builders = {
            "bell_state": lambda: AlgorithmTemplate.bell_state(
                kwargs.get("qubit0", 0), kwargs.get("qubit1", 1)
            ),
            "ghz_state": lambda: AlgorithmTemplate.ghz_state(
                kwargs.get("num_qubits", 3)
            ),
            "qft": lambda: AlgorithmTemplate.quantum_fourier_transform(
                kwargs.get("num_qubits", 3)
            ),
            "inverse_qft": lambda: AlgorithmTemplate.inverse_qft(
                kwargs.get("num_qubits", 3)
            ),
            "grover": lambda: AlgorithmTemplate.grover_search(
                kwargs.get("num_qubits", 3),
                kwargs.get("marked_state", 0),
            ),
            "deutsch_jozsa": lambda: AlgorithmTemplate.deutsch_jozsa(
                kwargs.get("num_qubits", 3),
                kwargs.get("oracle_type", "balanced"),
            ),
            "teleportation": lambda: AlgorithmTemplate.quantum_teleportation(),
            "bernstein_vazirani": lambda: AlgorithmTemplate.bernstein_vazirani(
                kwargs.get("secret", "101")
            ),
            "superdense_coding": lambda: AlgorithmTemplate.superdense_coding(),
        }

        builder = builders.get(template_name)
        if builder is None:
            return None
        return builder()

    # ------------------------------------------------------------------
    # Undo / Redo
    # ------------------------------------------------------------------

    def undo(self) -> None:
        """Undo the last circuit modification."""
        self._undo_stack.undo()

    def redo(self) -> None:
        """Redo the last undone circuit modification."""
        self._undo_stack.redo()

    def can_undo(self) -> bool:
        return self._undo_stack.canUndo()

    def can_redo(self) -> bool:
        return self._undo_stack.canRedo()
