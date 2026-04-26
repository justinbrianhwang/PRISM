"""QUndoCommand subclasses for the quantum circuit editor.

Each command modifies both the circuit data model (QuantumCircuit) and the
visual scene (CircuitScene) so that undo / redo keeps them in sync.

Commands
--------
- AddGateCommand
- RemoveGateCommand
- MoveGateCommand
- ChangeGateParamsCommand
- SetQubitCountCommand
- ClearCircuitCommand
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtGui import QUndoCommand

from quantum_sim.engine.circuit import GateInstance, QuantumCircuit

if TYPE_CHECKING:
    from quantum_sim.gui.circuit_editor.scene import CircuitScene


# =========================================================================
# AddGateCommand
# =========================================================================

class AddGateCommand(QUndoCommand):
    """Add a gate to the circuit model and create its visual item."""

    def __init__(
        self,
        circuit: QuantumCircuit,
        scene: CircuitScene,
        gate_instance: GateInstance,
    ):
        super().__init__(f"Add {gate_instance.gate_name}")
        self._circuit = circuit
        self._scene = scene
        self._gate = gate_instance

    def redo(self) -> None:
        self._circuit.add_gate(self._gate)
        self._scene.add_gate_visual(self._gate)

    def undo(self) -> None:
        self._circuit.remove_gate(self._gate)
        self._scene.remove_gate_visual(self._gate)


# =========================================================================
# RemoveGateCommand
# =========================================================================

class RemoveGateCommand(QUndoCommand):
    """Remove a gate from the circuit model and its visual item."""

    def __init__(
        self,
        circuit: QuantumCircuit,
        scene: CircuitScene,
        gate_instance: GateInstance,
    ):
        super().__init__(f"Remove {gate_instance.gate_name}")
        self._circuit = circuit
        self._scene = scene
        self._gate = gate_instance

    def redo(self) -> None:
        self._circuit.remove_gate(self._gate)
        self._scene.remove_gate_visual(self._gate)

    def undo(self) -> None:
        self._circuit.add_gate(self._gate)
        self._scene.add_gate_visual(self._gate)


# =========================================================================
# MoveGateCommand
# =========================================================================

class MoveGateCommand(QUndoCommand):
    """Move a gate to a different column / target qubit(s).

    Stores both old and new positions so the move is fully reversible.
    """

    def __init__(
        self,
        circuit: QuantumCircuit,
        scene: CircuitScene,
        gate_instance: GateInstance,
        old_col: int,
        old_targets: list[int],
        new_col: int,
        new_targets: list[int],
    ):
        super().__init__(f"Move {gate_instance.gate_name}")
        self._circuit = circuit
        self._scene = scene
        self._gate = gate_instance
        self._old_col = old_col
        self._old_targets = list(old_targets)
        self._new_col = new_col
        self._new_targets = list(new_targets)

    def redo(self) -> None:
        self._circuit.move_gate(self._gate, self._new_col, self._new_targets)
        self._scene.refresh_gate_visual(self._gate)

    def undo(self) -> None:
        self._circuit.move_gate(self._gate, self._old_col, self._old_targets)
        self._scene.refresh_gate_visual(self._gate)


# =========================================================================
# ChangeGateParamsCommand
# =========================================================================

class ChangeGateParamsCommand(QUndoCommand):
    """Modify the parameters of a parameterized gate.

    This is used for gates like Rx, Ry, Rz, Phase, U3 where the user
    edits the angle / parameter values.
    """

    def __init__(
        self,
        circuit: QuantumCircuit,
        scene: CircuitScene,
        gate_instance: GateInstance,
        old_params: list[float],
        new_params: list[float],
    ):
        super().__init__(f"Edit {gate_instance.gate_name} params")
        self._circuit = circuit
        self._scene = scene
        self._gate = gate_instance
        self._old_params = list(old_params)
        self._new_params = list(new_params)

    def redo(self) -> None:
        self._gate.params = list(self._new_params)
        self._scene.refresh_gate_visual(self._gate)

    def undo(self) -> None:
        self._gate.params = list(self._old_params)
        self._scene.refresh_gate_visual(self._gate)


# =========================================================================
# SetQubitCountCommand
# =========================================================================

class SetQubitCountCommand(QUndoCommand):
    """Change the number of qubits in the circuit.

    When reducing the count, gates on removed qubits are deleted by
    ``QuantumCircuit.set_num_qubits``.  We snapshot the removed gates so
    they can be restored on undo.
    """

    def __init__(
        self,
        circuit: QuantumCircuit,
        scene: CircuitScene,
        old_count: int,
        new_count: int,
    ):
        super().__init__(f"Set qubits {old_count} -> {new_count}")
        self._circuit = circuit
        self._scene = scene
        self._old_count = old_count
        self._new_count = new_count
        # Snapshot gates that will be removed when shrinking
        self._removed_gates: list[GateInstance] = []
        if new_count < old_count:
            self._removed_gates = [
                g for g in circuit.gates
                if any(q >= new_count for q in g.target_qubits)
            ]

    def redo(self) -> None:
        self._circuit.set_num_qubits(self._new_count)
        self._scene.rebuild()

    def undo(self) -> None:
        self._circuit.num_qubits = self._old_count
        # Restore gates that were removed
        for g in self._removed_gates:
            if g not in self._circuit.gates:
                self._circuit.gates.append(g)
        self._scene.rebuild()


# =========================================================================
# ClearCircuitCommand
# =========================================================================

class ClearCircuitCommand(QUndoCommand):
    """Remove all gates from the circuit.

    The previous gate list is saved for undo.
    """

    def __init__(
        self,
        circuit: QuantumCircuit,
        scene: CircuitScene,
        old_gates: list[GateInstance],
    ):
        super().__init__("Clear circuit")
        self._circuit = circuit
        self._scene = scene
        # Deep-copy the list (references to the same GateInstance objects)
        self._old_gates = list(old_gates)

    def redo(self) -> None:
        self._circuit.clear()
        self._scene.rebuild()

    def undo(self) -> None:
        for g in self._old_gates:
            self._circuit.add_gate(g)
        self._scene.rebuild()
