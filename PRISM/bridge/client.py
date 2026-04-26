"""Client library for connecting to the Quantum Simulator Bridge.

Usage example::

    from quantum_sim.bridge.client import SimulatorClient

    with SimulatorClient() as sim:
        sim.set_circuit(circuit_dict)
        result = sim.run(shots=1024, seed=42)
        analysis = sim.get_analysis(["fidelity", "entropy", "purity"])
        sweep = sim.sweep_parameter("noise_p", [0.01, 0.05, 0.1])
"""

from __future__ import annotations

import json
import socket
import uuid
from typing import Any

from .protocol import BridgeMessage

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 9876


class SimulatorClient:
    """Synchronous TCP client for the Quantum Simulator Bridge.

    Connects to a running simulator instance and sends JSON commands.
    Can be used as a context manager.
    """

    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT,
                 timeout: float = 30.0):
        self._host = host
        self._port = port
        self._timeout = timeout
        self._socket: socket.socket | None = None
        self._buffer = b""

    def connect(self) -> None:
        """Establish TCP connection to the bridge server."""
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.settimeout(self._timeout)
        self._socket.connect((self._host, self._port))

    def close(self) -> None:
        """Close the TCP connection."""
        if self._socket is not None:
            try:
                self._socket.close()
            except OSError:
                pass
            self._socket = None

    def __enter__(self) -> SimulatorClient:
        self.connect()
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def _send_request(self, action: str, params: dict | None = None) -> dict:
        """Send a request and wait for the response.

        Returns the response data dict on success; raises RuntimeError on error.
        """
        if self._socket is None:
            raise RuntimeError("Not connected. Call connect() first.")

        msg = BridgeMessage(
            type="request",
            id=str(uuid.uuid4()),
            action=action,
            params=params or {},
        )
        self._socket.sendall(msg.to_bytes())

        # Read response (newline-delimited)
        while b"\n" not in self._buffer:
            chunk = self._socket.recv(65536)
            if not chunk:
                raise ConnectionError("Server closed the connection.")
            self._buffer += chunk

        line, self._buffer = self._buffer.split(b"\n", 1)
        response = BridgeMessage.from_json(line.decode("utf-8"))

        if response.status == "error":
            raise RuntimeError(f"Bridge error: {response.error}")

        return response.data

    # -- High-level API methods --

    def ping(self) -> bool:
        """Check if the bridge server is responding."""
        data = self._send_request("ping")
        return data.get("pong", False)

    def set_circuit(self, circuit_dict: dict) -> dict:
        """Set the simulator circuit from a dictionary representation."""
        return self._send_request("set_circuit", {"circuit": circuit_dict})

    def get_circuit(self) -> dict:
        """Get the current circuit as a dictionary."""
        return self._send_request("get_circuit")

    def add_gate(self, gate_name: str, target_qubits: list[int],
                 params: list[float] | None = None,
                 column: int = 0) -> dict:
        """Add a gate to the current circuit."""
        return self._send_request("add_gate", {
            "gate_name": gate_name,
            "target_qubits": target_qubits,
            "params": params or [],
            "column": column,
        })

    def clear_circuit(self) -> dict:
        """Remove all gates from the circuit."""
        return self._send_request("clear_circuit")

    def run(self, shots: int = 1024, seed: int | None = None) -> dict:
        """Run the simulation and return measurement results."""
        params: dict[str, Any] = {"shots": shots}
        if seed is not None:
            params["seed"] = seed
        return self._send_request("run", params)

    def get_state(self) -> dict:
        """Get the state vector of the last simulation."""
        return self._send_request("get_state")

    def get_result(self) -> dict:
        """Get the last simulation result (counts + shots)."""
        return self._send_request("get_result")

    def set_noise(self, noise_dict: dict) -> dict:
        """Set the noise model from a dictionary representation."""
        return self._send_request("set_noise", {"noise_model": noise_dict})

    def clear_noise(self) -> dict:
        """Remove the noise model."""
        return self._send_request("clear_noise")

    def get_analysis(self, metrics: list[str] | None = None) -> dict:
        """Get analysis metrics for the last simulation result.

        Args:
            metrics: List of metric names. Supported:
                "fidelity", "entropy", "purity", "pauli".
                Defaults to ["fidelity", "entropy", "purity"].
        """
        return self._send_request("get_analysis", {
            "metrics": metrics or ["fidelity", "entropy", "purity"],
        })

    def sweep_parameter(self, param: str, values: list[float],
                        shots: int = 0, seed: int | None = None,
                        trials: int | None = None) -> dict:
        """Sweep a parameter (e.g., noise probability) and collect results.

        Args:
            param: Parameter name (e.g., "noise_p").
            values: List of parameter values to sweep.
            shots: Number of measurement shots per point (0 for statevector).
            seed: Optional seed for reproducibility.
            trials: Number of Monte Carlo trials per point.
        """
        params: dict[str, Any] = {
            "param": param,
            "values": values,
            "shots": shots,
        }
        if seed is not None:
            params["seed"] = seed
        if trials is not None:
            params["trials"] = trials
        return self._send_request("sweep_parameter", params)
