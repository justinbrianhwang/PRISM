"""Built-in quantum algorithm circuit templates."""

from __future__ import annotations

import math
from .circuit import QuantumCircuit, GateInstance


class AlgorithmTemplate:
    """Factory for common quantum algorithm circuits."""

    @staticmethod
    def bell_state(qubit0: int = 0, qubit1: int = 1) -> QuantumCircuit:
        """Bell state |Phi+> = (|00> + |11>) / sqrt(2)."""
        n = max(qubit0, qubit1) + 1
        circuit = QuantumCircuit(num_qubits=n)
        circuit.add_gate(GateInstance("H", [qubit0], [], 0))
        circuit.add_gate(GateInstance("CNOT", [qubit0, qubit1], [], 1))
        circuit.add_gate(GateInstance("Measure", [qubit0], [], 2))
        circuit.add_gate(GateInstance("Measure", [qubit1], [], 2))
        return circuit

    @staticmethod
    def ghz_state(num_qubits: int) -> QuantumCircuit:
        """GHZ state (|00...0> + |11...1>) / sqrt(2)."""
        circuit = QuantumCircuit(num_qubits=num_qubits)
        circuit.add_gate(GateInstance("H", [0], [], 0))
        for i in range(1, num_qubits):
            circuit.add_gate(GateInstance("CNOT", [0, i], [], i))
        for i in range(num_qubits):
            circuit.add_gate(GateInstance("Measure", [i], [], num_qubits))
        return circuit

    @staticmethod
    def quantum_fourier_transform(num_qubits: int) -> QuantumCircuit:
        """Quantum Fourier Transform circuit."""
        circuit = QuantumCircuit(num_qubits=num_qubits)
        col = 0
        for i in range(num_qubits):
            circuit.add_gate(GateInstance("H", [i], [], col))
            col += 1
            for j in range(i + 1, num_qubits):
                k = j - i + 1
                angle = math.pi / (2 ** (k - 1))
                # Controlled phase rotation: use Phase on target controlled by j
                # Simplified: apply controlled-phase as a sequence
                circuit.add_gate(GateInstance("Phase", [j], [angle], col))
                col += 1

        # SWAP qubits to reverse order
        for i in range(num_qubits // 2):
            circuit.add_gate(GateInstance("SWAP", [i, num_qubits - 1 - i], [], col))
            col += 1

        return circuit

    @staticmethod
    def inverse_qft(num_qubits: int) -> QuantumCircuit:
        """Inverse Quantum Fourier Transform."""
        circuit = QuantumCircuit(num_qubits=num_qubits)
        col = 0

        # SWAP qubits first
        for i in range(num_qubits // 2):
            circuit.add_gate(GateInstance("SWAP", [i, num_qubits - 1 - i], [], col))
            col += 1

        for i in range(num_qubits - 1, -1, -1):
            for j in range(num_qubits - 1, i, -1):
                k = j - i + 1
                angle = -math.pi / (2 ** (k - 1))
                circuit.add_gate(GateInstance("Phase", [j], [angle], col))
                col += 1
            circuit.add_gate(GateInstance("H", [i], [], col))
            col += 1

        return circuit

    @staticmethod
    def grover_search(num_qubits: int, marked_state: int = 0) -> QuantumCircuit:
        """Grover's search algorithm.

        Creates a circuit with oracle for the marked state
        and diffusion operator.
        """
        circuit = QuantumCircuit(num_qubits=num_qubits)
        col = 0

        # Number of Grover iterations
        num_iterations = max(1, int(round(math.pi / 4 * math.sqrt(2 ** num_qubits))))

        # Initial superposition
        for i in range(num_qubits):
            circuit.add_gate(GateInstance("H", [i], [], col))
        col += 1

        for _ in range(num_iterations):
            # Oracle: flip the phase of the marked state
            # Apply X to qubits where marked_state bit is 0
            for i in range(num_qubits):
                bit = (marked_state >> (num_qubits - 1 - i)) & 1
                if bit == 0:
                    circuit.add_gate(GateInstance("X", [i], [], col))
            col += 1

            # Multi-controlled Z (implemented as H-Toffoli-H for small cases)
            if num_qubits == 2:
                circuit.add_gate(GateInstance("CZ", [0, 1], [], col))
            elif num_qubits >= 3:
                # Phase kickback using ancilla-free approach
                circuit.add_gate(GateInstance("H", [num_qubits - 1], [], col))
                col += 1
                if num_qubits == 3:
                    circuit.add_gate(GateInstance("Toffoli", [0, 1, 2], [], col))
                else:
                    # For >3 qubits, use multi-CNOT decomposition
                    for i in range(num_qubits - 2):
                        circuit.add_gate(GateInstance("CNOT", [i, num_qubits - 1], [], col))
                        col += 1
                col += 1
                circuit.add_gate(GateInstance("H", [num_qubits - 1], [], col))
            col += 1

            # Undo oracle X gates
            for i in range(num_qubits):
                bit = (marked_state >> (num_qubits - 1 - i)) & 1
                if bit == 0:
                    circuit.add_gate(GateInstance("X", [i], [], col))
            col += 1

            # Diffusion operator
            for i in range(num_qubits):
                circuit.add_gate(GateInstance("H", [i], [], col))
            col += 1
            for i in range(num_qubits):
                circuit.add_gate(GateInstance("X", [i], [], col))
            col += 1

            if num_qubits == 2:
                circuit.add_gate(GateInstance("CZ", [0, 1], [], col))
            elif num_qubits >= 3:
                circuit.add_gate(GateInstance("H", [num_qubits - 1], [], col))
                col += 1
                if num_qubits == 3:
                    circuit.add_gate(GateInstance("Toffoli", [0, 1, 2], [], col))
                else:
                    for i in range(num_qubits - 2):
                        circuit.add_gate(GateInstance("CNOT", [i, num_qubits - 1], [], col))
                        col += 1
                col += 1
                circuit.add_gate(GateInstance("H", [num_qubits - 1], [], col))
            col += 1

            for i in range(num_qubits):
                circuit.add_gate(GateInstance("X", [i], [], col))
            col += 1
            for i in range(num_qubits):
                circuit.add_gate(GateInstance("H", [i], [], col))
            col += 1

        # Measurements
        for i in range(num_qubits):
            circuit.add_gate(GateInstance("Measure", [i], [], col))

        return circuit

    @staticmethod
    def deutsch_jozsa(num_qubits: int, oracle_type: str = "balanced") -> QuantumCircuit:
        """Deutsch-Jozsa algorithm.

        oracle_type: 'constant' or 'balanced'
        """
        # Uses num_qubits - 1 input qubits + 1 ancilla
        circuit = QuantumCircuit(num_qubits=num_qubits)
        n = num_qubits - 1  # Input qubits
        ancilla = num_qubits - 1
        col = 0

        # Initialize ancilla to |1>
        circuit.add_gate(GateInstance("X", [ancilla], [], col))
        col += 1

        # Hadamard all qubits
        for i in range(num_qubits):
            circuit.add_gate(GateInstance("H", [i], [], col))
        col += 1

        # Oracle
        if oracle_type == "balanced":
            # Balanced oracle: CNOT from each input to ancilla
            for i in range(n):
                circuit.add_gate(GateInstance("CNOT", [i, ancilla], [], col))
                col += 1
        elif oracle_type == "constant":
            # Constant oracle: do nothing (f(x) = 0) or X on ancilla (f(x) = 1)
            pass  # f(x) = 0, no gates needed
        col += 1

        # Hadamard on input qubits
        for i in range(n):
            circuit.add_gate(GateInstance("H", [i], [], col))
        col += 1

        # Measure input qubits
        for i in range(n):
            circuit.add_gate(GateInstance("Measure", [i], [], col))

        return circuit

    @staticmethod
    def quantum_teleportation() -> QuantumCircuit:
        """Quantum teleportation protocol (3 qubits)."""
        circuit = QuantumCircuit(num_qubits=3)

        # q0: state to teleport (prepare |+> for demo)
        circuit.add_gate(GateInstance("H", [0], [], 0))

        # Create Bell pair between q1 and q2
        circuit.add_gate(GateInstance("H", [1], [], 1))
        circuit.add_gate(GateInstance("CNOT", [1, 2], [], 2))

        # Bell measurement on q0 and q1
        circuit.add_gate(GateInstance("CNOT", [0, 1], [], 3))
        circuit.add_gate(GateInstance("H", [0], [], 4))

        # Measurements
        circuit.add_gate(GateInstance("Measure", [0], [], 5))
        circuit.add_gate(GateInstance("Measure", [1], [], 5))

        # Corrections (classical-conditional, here applied unconditionally for demo)
        circuit.add_gate(GateInstance("CNOT", [1, 2], [], 6))
        circuit.add_gate(GateInstance("CZ", [0, 2], [], 7))

        return circuit

    @staticmethod
    def bernstein_vazirani(secret: str) -> QuantumCircuit:
        """Bernstein-Vazirani algorithm to find a secret bitstring."""
        n = len(secret)
        circuit = QuantumCircuit(num_qubits=n + 1)
        ancilla = n
        col = 0

        # Initialize ancilla to |1>
        circuit.add_gate(GateInstance("X", [ancilla], [], col))
        col += 1

        # Hadamard all
        for i in range(n + 1):
            circuit.add_gate(GateInstance("H", [i], [], col))
        col += 1

        # Oracle: CNOT from qubit i to ancilla if secret[i] == '1'
        for i, bit in enumerate(secret):
            if bit == '1':
                circuit.add_gate(GateInstance("CNOT", [i, ancilla], [], col))
                col += 1

        # Hadamard on input qubits
        for i in range(n):
            circuit.add_gate(GateInstance("H", [i], [], col))
        col += 1

        # Measure input qubits
        for i in range(n):
            circuit.add_gate(GateInstance("Measure", [i], [], col))

        return circuit

    @staticmethod
    def superdense_coding() -> QuantumCircuit:
        """Superdense coding protocol (2 qubits)."""
        circuit = QuantumCircuit(num_qubits=2)

        # Create Bell pair
        circuit.add_gate(GateInstance("H", [0], [], 0))
        circuit.add_gate(GateInstance("CNOT", [0, 1], [], 1))

        # Encode 2 classical bits (example: encode '11')
        circuit.add_gate(GateInstance("X", [0], [], 2))
        circuit.add_gate(GateInstance("Z", [0], [], 3))

        # Decode
        circuit.add_gate(GateInstance("CNOT", [0, 1], [], 4))
        circuit.add_gate(GateInstance("H", [0], [], 5))

        # Measure
        circuit.add_gate(GateInstance("Measure", [0], [], 6))
        circuit.add_gate(GateInstance("Measure", [1], [], 6))

        return circuit

    @staticmethod
    def list_templates() -> list[dict[str, str]]:
        """Returns list of available algorithm templates."""
        return [
            {"name": "bell_state", "display": "Bell State",
             "description": "Creates a Bell state |Phi+> = (|00> + |11>) / sqrt(2)"},
            {"name": "ghz_state", "display": "GHZ State",
             "description": "Creates a GHZ state (|00...0> + |11...1>) / sqrt(2)"},
            {"name": "qft", "display": "Quantum Fourier Transform",
             "description": "Quantum Fourier Transform circuit"},
            {"name": "inverse_qft", "display": "Inverse QFT",
             "description": "Inverse Quantum Fourier Transform"},
            {"name": "grover", "display": "Grover's Search",
             "description": "Grover's quantum search algorithm"},
            {"name": "deutsch_jozsa", "display": "Deutsch-Jozsa",
             "description": "Deutsch-Jozsa algorithm for function classification"},
            {"name": "teleportation", "display": "Quantum Teleportation",
             "description": "Quantum teleportation protocol"},
            {"name": "bernstein_vazirani", "display": "Bernstein-Vazirani",
             "description": "Bernstein-Vazirani algorithm for finding secret strings"},
            {"name": "superdense_coding", "display": "Superdense Coding",
             "description": "Superdense coding protocol"},
        ]
