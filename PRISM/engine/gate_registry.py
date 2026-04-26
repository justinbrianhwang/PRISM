"""Extensible gate registry using the Singleton pattern."""

from __future__ import annotations

from .gates import (
    GateDefinition, GateType, _const,
    I_MATRIX, X_MATRIX, Y_MATRIX, Z_MATRIX, H_MATRIX,
    S_MATRIX, S_DAG_MATRIX, T_MATRIX, T_DAG_MATRIX,
    CNOT_MATRIX, CZ_MATRIX, SWAP_MATRIX, TOFFOLI_MATRIX, FREDKIN_MATRIX,
    rx_matrix, ry_matrix, rz_matrix, phase_matrix, u3_matrix,
)


class GateRegistry:
    """Singleton registry mapping gate names to GateDefinition objects."""

    _instance: GateRegistry | None = None

    def __init__(self):
        self._gates: dict[str, GateDefinition] = {}

    @classmethod
    def instance(cls) -> GateRegistry:
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._register_builtins()
        return cls._instance

    @classmethod
    def reset(cls):
        """Reset the singleton (for testing)."""
        cls._instance = None

    def _register_builtins(self):
        # Single-qubit fixed gates
        self.register(GateDefinition(
            name="I", display_name="Identity", gate_type=GateType.SINGLE,
            num_qubits=1, num_params=0, param_names=(),
            matrix_func=_const(I_MATRIX), symbol="I", color="#888888"))

        self.register(GateDefinition(
            name="H", display_name="Hadamard", gate_type=GateType.SINGLE,
            num_qubits=1, num_params=0, param_names=(),
            matrix_func=_const(H_MATRIX), symbol="H", color="#4A90D9"))

        self.register(GateDefinition(
            name="X", display_name="Pauli-X", gate_type=GateType.SINGLE,
            num_qubits=1, num_params=0, param_names=(),
            matrix_func=_const(X_MATRIX), symbol="X", color="#E74C3C"))

        self.register(GateDefinition(
            name="Y", display_name="Pauli-Y", gate_type=GateType.SINGLE,
            num_qubits=1, num_params=0, param_names=(),
            matrix_func=_const(Y_MATRIX), symbol="Y", color="#2ECC71"))

        self.register(GateDefinition(
            name="Z", display_name="Pauli-Z", gate_type=GateType.SINGLE,
            num_qubits=1, num_params=0, param_names=(),
            matrix_func=_const(Z_MATRIX), symbol="Z", color="#3498DB"))

        self.register(GateDefinition(
            name="S", display_name="S Gate", gate_type=GateType.SINGLE,
            num_qubits=1, num_params=0, param_names=(),
            matrix_func=_const(S_MATRIX), symbol="S", color="#9B59B6"))

        self.register(GateDefinition(
            name="S_DAG", display_name="S\u2020 Gate", gate_type=GateType.SINGLE,
            num_qubits=1, num_params=0, param_names=(),
            matrix_func=_const(S_DAG_MATRIX), symbol="S\u2020", color="#8E44AD"))

        self.register(GateDefinition(
            name="T", display_name="T Gate", gate_type=GateType.SINGLE,
            num_qubits=1, num_params=0, param_names=(),
            matrix_func=_const(T_MATRIX), symbol="T", color="#E67E22"))

        self.register(GateDefinition(
            name="T_DAG", display_name="T\u2020 Gate", gate_type=GateType.SINGLE,
            num_qubits=1, num_params=0, param_names=(),
            matrix_func=_const(T_DAG_MATRIX), symbol="T\u2020", color="#D35400"))

        # Single-qubit parameterized gates
        self.register(GateDefinition(
            name="Rx", display_name="Rotation-X", gate_type=GateType.SINGLE,
            num_qubits=1, num_params=1, param_names=("\u03b8",),
            matrix_func=rx_matrix, symbol="Rx", color="#E91E63"))

        self.register(GateDefinition(
            name="Ry", display_name="Rotation-Y", gate_type=GateType.SINGLE,
            num_qubits=1, num_params=1, param_names=("\u03b8",),
            matrix_func=ry_matrix, symbol="Ry", color="#00BCD4"))

        self.register(GateDefinition(
            name="Rz", display_name="Rotation-Z", gate_type=GateType.SINGLE,
            num_qubits=1, num_params=1, param_names=("\u03b8",),
            matrix_func=rz_matrix, symbol="Rz", color="#FF9800"))

        self.register(GateDefinition(
            name="Phase", display_name="Phase Gate", gate_type=GateType.SINGLE,
            num_qubits=1, num_params=1, param_names=("\u03c6",),
            matrix_func=phase_matrix, symbol="P", color="#795548"))

        self.register(GateDefinition(
            name="U3", display_name="Universal U3", gate_type=GateType.SINGLE,
            num_qubits=1, num_params=3, param_names=("\u03b8", "\u03c6", "\u03bb"),
            matrix_func=u3_matrix, symbol="U3", color="#607D8B"))

        # Multi-qubit gates
        self.register(GateDefinition(
            name="CNOT", display_name="Controlled-NOT", gate_type=GateType.CONTROLLED,
            num_qubits=2, num_params=0, param_names=(),
            matrix_func=_const(CNOT_MATRIX), symbol="CX", color="#FF5722",
            num_controls=1, num_targets=1))

        self.register(GateDefinition(
            name="CZ", display_name="Controlled-Z", gate_type=GateType.CONTROLLED,
            num_qubits=2, num_params=0, param_names=(),
            matrix_func=_const(CZ_MATRIX), symbol="CZ", color="#673AB7",
            num_controls=1, num_targets=1))

        self.register(GateDefinition(
            name="SWAP", display_name="SWAP", gate_type=GateType.MULTI,
            num_qubits=2, num_params=0, param_names=(),
            matrix_func=_const(SWAP_MATRIX), symbol="SW", color="#009688",
            num_controls=0, num_targets=2))

        self.register(GateDefinition(
            name="Toffoli", display_name="Toffoli (CCX)", gate_type=GateType.CONTROLLED,
            num_qubits=3, num_params=0, param_names=(),
            matrix_func=_const(TOFFOLI_MATRIX), symbol="CCX", color="#F44336",
            num_controls=2, num_targets=1))

        self.register(GateDefinition(
            name="Fredkin", display_name="Fredkin (CSWAP)", gate_type=GateType.CONTROLLED,
            num_qubits=3, num_params=0, param_names=(),
            matrix_func=_const(FREDKIN_MATRIX), symbol="CSW", color="#4CAF50",
            num_controls=1, num_targets=2))

        # Measurement
        self.register(GateDefinition(
            name="Measure", display_name="Measurement", gate_type=GateType.MEASUREMENT,
            num_qubits=1, num_params=0, param_names=(),
            matrix_func=_const(I_MATRIX), symbol="M", color="#FFC107"))

        # Barrier
        self.register(GateDefinition(
            name="Barrier", display_name="Barrier", gate_type=GateType.BARRIER,
            num_qubits=1, num_params=0, param_names=(),
            matrix_func=_const(I_MATRIX), symbol="||", color="#BDBDBD"))

    def register(self, gate_def: GateDefinition):
        self._gates[gate_def.name] = gate_def

    def get(self, name: str) -> GateDefinition:
        if name not in self._gates:
            raise KeyError(f"Gate '{name}' not found in registry")
        return self._gates[name]

    def all_gates(self) -> list[GateDefinition]:
        return list(self._gates.values())

    def single_qubit_gates(self) -> list[GateDefinition]:
        return [g for g in self._gates.values()
                if g.gate_type == GateType.SINGLE]

    def multi_qubit_gates(self) -> list[GateDefinition]:
        return [g for g in self._gates.values()
                if g.gate_type in (GateType.CONTROLLED, GateType.MULTI)]

    def parameterized_gates(self) -> list[GateDefinition]:
        return [g for g in self._gates.values() if g.num_params > 0]

    def gate_names(self) -> list[str]:
        return list(self._gates.keys())
