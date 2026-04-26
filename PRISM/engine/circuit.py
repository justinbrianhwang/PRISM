"""Quantum circuit data model."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GateInstance:
    """A specific gate placed in the circuit."""
    gate_name: str
    target_qubits: list[int]
    params: list[float] = field(default_factory=list)
    column: int = 0

    def to_dict(self) -> dict:
        return {
            "name": self.gate_name,
            "targets": self.target_qubits,
            "params": self.params,
            "column": self.column,
        }

    @classmethod
    def from_dict(cls, data: dict) -> GateInstance:
        return cls(
            gate_name=data["name"],
            target_qubits=data["targets"],
            params=data.get("params", []),
            column=data.get("column", 0),
        )


@dataclass
class QuantumCircuit:
    """The full circuit model - a list of gate instances on n qubits."""
    num_qubits: int = 4
    gates: list[GateInstance] = field(default_factory=list)
    initial_states: list[int] = field(default_factory=list)

    def __post_init__(self):
        if not self.initial_states:
            self.initial_states = [0] * self.num_qubits
        # Ensure length matches num_qubits
        while len(self.initial_states) < self.num_qubits:
            self.initial_states.append(0)
        self.initial_states = self.initial_states[:self.num_qubits]

    def add_gate(self, gate: GateInstance):
        self.gates.append(gate)

    def remove_gate(self, gate: GateInstance):
        if gate in self.gates:
            self.gates.remove(gate)

    def move_gate(self, gate: GateInstance, new_col: int, new_targets: list[int]):
        if gate in self.gates:
            gate.column = new_col
            gate.target_qubits = new_targets

    def get_column_count(self) -> int:
        if not self.gates:
            return 0
        return max(g.column for g in self.gates) + 1

    def get_gates_at_column(self, col: int) -> list[GateInstance]:
        return [g for g in self.gates if g.column == col]

    def get_ordered_gates(self) -> list[list[GateInstance]]:
        """Returns gates grouped by column, sorted by column index."""
        if not self.gates:
            return []
        max_col = max(g.column for g in self.gates)
        result = []
        for col in range(max_col + 1):
            col_gates = self.get_gates_at_column(col)
            if col_gates:
                result.append(sorted(col_gates, key=lambda g: g.target_qubits[0]))
        return result

    def compute_layers(self) -> list[list[int]]:
        """Compute layer structure: group gate indices by column.

        A 'layer' is a set of gates at the same column index. This definition
        is shared across optimizer (barren plateau), debugger (noise attribution),
        and entropy (entanglement events) for consistency.

        Returns:
            List of layers, where each layer is a list of gate indices
            into ``self.gates``. Sorted by column ascending.
        """
        if not self.gates:
            return []
        col_to_indices: dict[int, list[int]] = {}
        for gi, gate in enumerate(self.gates):
            col = gate.column
            if col not in col_to_indices:
                col_to_indices[col] = []
            col_to_indices[col].append(gi)
        return [col_to_indices[c] for c in sorted(col_to_indices)]

    def gate_to_layer_map(self) -> list[int]:
        """Map each gate index to its layer index.

        Returns:
            List where index i gives the layer index of ``self.gates[i]``.
        """
        layers = self.compute_layers()
        mapping = [0] * len(self.gates)
        for layer_idx, gate_indices in enumerate(layers):
            for gi in gate_indices:
                mapping[gi] = layer_idx
        return mapping

    def circuit_hash(self) -> int:
        """Compute a hash of the circuit structure for invalidation checks.

        Captures: num_qubits, initial_states, and all gate definitions.
        """
        parts: list = [self.num_qubits, tuple(self.initial_states)]
        for g in self.gates:
            parts.append((g.gate_name, tuple(g.target_qubits),
                          tuple(g.params), g.column))
        return hash(tuple(parts))

    def clear(self):
        self.gates.clear()

    def set_num_qubits(self, n: int):
        if n < 1 or n > 16:
            raise ValueError(f"num_qubits must be 1-16, got {n}")
        # Remove gates that reference qubits >= n
        self.gates = [g for g in self.gates
                      if all(q < n for q in g.target_qubits)]
        self.num_qubits = n
        # Resize initial_states
        while len(self.initial_states) < n:
            self.initial_states.append(0)
        self.initial_states = self.initial_states[:n]

    def toggle_qubit_initial_state(self, qubit: int) -> None:
        """Toggle qubit initial state between |0> and |1>."""
        if 0 <= qubit < self.num_qubits:
            self.initial_states[qubit] = 1 - self.initial_states[qubit]

    def set_qubit_initial_state(self, qubit: int, state: int) -> None:
        """Set a specific qubit's initial state (0 or 1)."""
        if 0 <= qubit < self.num_qubits and state in (0, 1):
            self.initial_states[qubit] = state

    def gate_count(self) -> int:
        return len(self.gates)

    def to_dict(self) -> dict:
        d = {
            "version": "1.0",
            "num_qubits": self.num_qubits,
            "gates": [g.to_dict() for g in self.gates],
        }
        # Only include initial_states if any are non-zero (backward compat)
        if any(s != 0 for s in self.initial_states):
            d["initial_states"] = self.initial_states
        return d

    @classmethod
    def from_dict(cls, data: dict) -> QuantumCircuit:
        circuit = cls(
            num_qubits=data["num_qubits"],
            initial_states=data.get("initial_states", []),
        )
        for g_data in data["gates"]:
            circuit.add_gate(GateInstance.from_dict(g_data))
        return circuit
