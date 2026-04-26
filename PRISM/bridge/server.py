"""TCP server for the Live Bridge API.

Runs in a QThread so the GUI remains responsive. Accepts connections
on localhost and dispatches JSON commands to BridgeCommandHandler.
"""

from __future__ import annotations

import json
import logging
import select
import socket
import threading
from typing import TYPE_CHECKING

import numpy as np
from PyQt6.QtCore import QObject, QThread, pyqtSignal

from .protocol import BridgeMessage

if TYPE_CHECKING:
    from quantum_sim.engine.circuit import QuantumCircuit
    from quantum_sim.engine.simulator import SimulationResult

logger = logging.getLogger(__name__)

DEFAULT_PORT = 9876


class BridgeCommandHandler:
    """Processes incoming bridge commands and produces responses.

    The handler holds references to the simulator state managed by
    MainWindow.  All public methods return BridgeMessage responses.
    """

    def __init__(self):
        # These are set by MainWindow after construction
        self._circuit: QuantumCircuit | None = None
        self._noise_model: object | None = None
        self._last_result: SimulationResult | None = None
        self._ideal_state = None  # StateVector

    # -- state setters (called from MainWindow) --

    def set_circuit(self, circuit):
        self._circuit = circuit

    def set_noise_model(self, noise_model):
        self._noise_model = noise_model

    def set_last_result(self, result):
        self._last_result = result

    def set_ideal_state(self, state):
        self._ideal_state = state

    # -- command dispatch --

    def handle(self, msg: BridgeMessage) -> BridgeMessage:
        """Route a request message to the appropriate handler."""
        action = msg.action
        handler = getattr(self, f"_cmd_{action}", None)
        if handler is None:
            return BridgeMessage.error_response(
                msg.id, f"Unknown action: {action}"
            )
        try:
            return handler(msg)
        except Exception as e:
            logger.error("Bridge command '%s' failed: %s", action, e, exc_info=True)
            return BridgeMessage.error_response(msg.id, str(e))

    # -- individual commands --

    def _cmd_ping(self, msg: BridgeMessage) -> BridgeMessage:
        return BridgeMessage.ok_response(msg.id, {"pong": True})

    def _cmd_get_circuit(self, msg: BridgeMessage) -> BridgeMessage:
        if self._circuit is None:
            return BridgeMessage.error_response(msg.id, "No circuit loaded")
        return BridgeMessage.ok_response(msg.id, self._circuit.to_dict())

    def _cmd_set_circuit(self, msg: BridgeMessage) -> BridgeMessage:
        from quantum_sim.engine.circuit import QuantumCircuit
        circuit_dict = msg.params.get("circuit")
        if circuit_dict is None:
            return BridgeMessage.error_response(msg.id, "Missing 'circuit' param")
        self._circuit = QuantumCircuit.from_dict(circuit_dict)
        return BridgeMessage.ok_response(msg.id, {
            "num_qubits": self._circuit.num_qubits,
            "gate_count": self._circuit.gate_count(),
        })

    def _cmd_add_gate(self, msg: BridgeMessage) -> BridgeMessage:
        from quantum_sim.engine.circuit import GateInstance
        if self._circuit is None:
            return BridgeMessage.error_response(msg.id, "No circuit loaded")
        p = msg.params
        gate = GateInstance(
            gate_name=p.get("gate_name", "H"),
            target_qubits=p.get("target_qubits", [0]),
            params=p.get("params", []),
            column=p.get("column", 0),
        )
        self._circuit.add_gate(gate)
        return BridgeMessage.ok_response(msg.id, {
            "gate_count": self._circuit.gate_count()
        })

    def _cmd_clear_circuit(self, msg: BridgeMessage) -> BridgeMessage:
        if self._circuit is None:
            return BridgeMessage.error_response(msg.id, "No circuit loaded")
        self._circuit.clear()
        return BridgeMessage.ok_response(msg.id)

    def _cmd_run(self, msg: BridgeMessage) -> BridgeMessage:
        from quantum_sim.engine.simulator import Simulator
        if self._circuit is None:
            return BridgeMessage.error_response(msg.id, "No circuit loaded")

        shots = msg.params.get("shots", 1024)
        seed = msg.params.get("seed")

        sim = Simulator(noise_model=self._noise_model)
        if self._noise_model is not None and shots > 0:
            result = sim.run_with_noise(self._circuit, shots=shots, seed=seed)
        else:
            result = sim.run(self._circuit, shots=shots, seed=seed)

        self._last_result = result
        # Store ideal state for analysis commands
        if self._noise_model is None:
            self._ideal_state = result.final_state

        data = {
            "measurement_counts": result.measurement_counts,
            "num_shots": result.num_shots,
            "seed": result.seed,
        }
        return BridgeMessage.ok_response(msg.id, data)

    def _cmd_get_state(self, msg: BridgeMessage) -> BridgeMessage:
        if self._last_result is None:
            return BridgeMessage.error_response(msg.id, "No simulation result")
        sv = self._last_result.final_state
        # Convert state vector to serializable format
        amplitudes = [
            {"re": float(a.real), "im": float(a.imag)}
            for a in sv.data
        ]
        return BridgeMessage.ok_response(msg.id, {
            "num_qubits": sv.num_qubits,
            "amplitudes": amplitudes,
            "probabilities": sv.probabilities.tolist(),
        })

    def _cmd_get_result(self, msg: BridgeMessage) -> BridgeMessage:
        if self._last_result is None:
            return BridgeMessage.error_response(msg.id, "No simulation result")
        r = self._last_result
        return BridgeMessage.ok_response(msg.id, {
            "measurement_counts": r.measurement_counts,
            "num_shots": r.num_shots,
            "seed": r.seed,
        })

    def _cmd_set_noise(self, msg: BridgeMessage) -> BridgeMessage:
        from quantum_sim.engine.noise import NoiseModel
        noise_dict = msg.params.get("noise_model")
        if noise_dict is None:
            return BridgeMessage.error_response(msg.id, "Missing 'noise_model' param")
        self._noise_model = NoiseModel.from_dict(noise_dict)
        return BridgeMessage.ok_response(msg.id)

    def _cmd_clear_noise(self, msg: BridgeMessage) -> BridgeMessage:
        self._noise_model = None
        return BridgeMessage.ok_response(msg.id)

    def _cmd_get_analysis(self, msg: BridgeMessage) -> BridgeMessage:
        from quantum_sim.engine.analysis import StateAnalysis
        if self._last_result is None:
            return BridgeMessage.error_response(msg.id, "No simulation result")

        state = self._last_result.final_state
        metrics = msg.params.get("metrics", ["fidelity", "entropy", "purity"])
        data = {}

        for m in metrics:
            if m == "fidelity" and self._ideal_state is not None:
                data["fidelity"] = StateAnalysis.state_fidelity(
                    self._ideal_state.data, state.data
                )
            elif m == "entropy":
                data["entropy"] = StateAnalysis.von_neumann_entropy(state)
            elif m == "purity":
                data["purity"] = StateAnalysis.purity(state)
            elif m == "pauli":
                pauli_data = {}
                for q in range(state.num_qubits):
                    pauli_data[f"q{q}"] = {
                        "X": StateAnalysis.pauli_expectation(state, "X", q),
                        "Y": StateAnalysis.pauli_expectation(state, "Y", q),
                        "Z": StateAnalysis.pauli_expectation(state, "Z", q),
                    }
                data["pauli"] = pauli_data

        return BridgeMessage.ok_response(msg.id, data)

    def _cmd_sweep_parameter(self, msg: BridgeMessage) -> BridgeMessage:
        """Sweep a noise parameter and collect results."""
        from quantum_sim.engine.simulator import Simulator
        from quantum_sim.engine.noise import NoiseModel, DepolarizingNoise
        from quantum_sim.engine.analysis import StateAnalysis

        if self._circuit is None:
            return BridgeMessage.error_response(msg.id, "No circuit loaded")

        param_name = msg.params.get("param", "noise_p")
        values = msg.params.get("values", [0.01, 0.05, 0.1])
        shots = msg.params.get("shots", 0)
        seed = msg.params.get("seed")
        trials = msg.params.get("trials", 50)
        try:
            n_trials = max(1, int(trials))
        except (TypeError, ValueError):
            n_trials = 50

        # Get ideal state
        sim_ideal = Simulator()
        rng = np.random.default_rng(seed)
        result_ideal = sim_ideal.run(
            self._circuit,
            shots=0,
            rng=np.random.default_rng(rng.integers(0, 2**63)),
        )
        ideal_state = result_ideal.final_state

        sweep_results = []
        for val in values:
            if float(val) == 0.0:
                sweep_results.append({
                    "value": val, "fidelity": 1.0, "purity": 1.0,
                })
                continue

            fid_sum = 0.0
            pur_sum = 0.0
            for _ in range(n_trials):
                model = NoiseModel()
                model.add_global_noise(DepolarizingNoise(float(val)))
                model.set_seed(int(rng.integers(0, 2**63)))
                sim = Simulator(noise_model=model)
                child_rng = np.random.default_rng(rng.integers(0, 2**63))
                res = sim.run(self._circuit, shots=shots, rng=child_rng)
                fid_sum += StateAnalysis.state_fidelity(
                    ideal_state.data, res.final_state.data
                )
                pur_sum += StateAnalysis.purity(res.final_state)
            sweep_results.append({
                "value": val,
                "fidelity": fid_sum / n_trials,
                "purity": pur_sum / n_trials,
                "trials": n_trials,
            })

        return BridgeMessage.ok_response(msg.id, {"sweep": sweep_results})


class BridgeWorker(QObject):
    """Non-blocking TCP server running inside a QThread.

    Uses select() to avoid blocking, checking a stop flag periodically.
    Signals are emitted for GUI integration.
    """

    client_connected = pyqtSignal(str)
    client_disconnected = pyqtSignal(str)
    command_received = pyqtSignal(str)  # action name
    status_changed = pyqtSignal(str)    # status text

    def __init__(self, handler: BridgeCommandHandler, port: int = DEFAULT_PORT):
        super().__init__()
        self._handler = handler
        self._port = port
        self._running = False
        self._server_socket: socket.socket | None = None

    def start_server(self):
        """Start accepting connections (called from thread)."""
        self._running = True
        self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_socket.setblocking(False)

        try:
            self._server_socket.bind(("127.0.0.1", self._port))
            self._server_socket.listen(5)
            self.status_changed.emit(f"Listening on port {self._port}")
            logger.info("Bridge server listening on 127.0.0.1:%d", self._port)
        except OSError as e:
            self.status_changed.emit(f"Failed: {e}")
            logger.error("Failed to bind bridge server: %s", e)
            self._running = False
            return

        clients: list[socket.socket] = []
        buffers: dict[int, bytes] = {}  # fd -> accumulated bytes

        while self._running:
            try:
                readable = [self._server_socket] + clients
                ready, _, _ = select.select(readable, [], [], 0.2)
            except (ValueError, OSError):
                break

            for sock in ready:
                if sock is self._server_socket:
                    # New connection
                    try:
                        client, addr = self._server_socket.accept()
                        client.setblocking(False)
                        clients.append(client)
                        buffers[client.fileno()] = b""
                        addr_str = f"{addr[0]}:{addr[1]}"
                        self.client_connected.emit(addr_str)
                        self.status_changed.emit(f"Connected: {addr_str}")
                        logger.info("Bridge client connected: %s", addr_str)
                    except OSError:
                        pass
                else:
                    # Data from client
                    fd = sock.fileno()
                    try:
                        data = sock.recv(65536)
                    except (OSError, ConnectionError):
                        data = b""

                    if not data:
                        # Client disconnected
                        clients.remove(sock)
                        buffers.pop(fd, None)
                        try:
                            addr_str = f"{sock.getpeername()[0]}:{sock.getpeername()[1]}"
                        except OSError:
                            addr_str = "unknown"
                        self.client_disconnected.emit(addr_str)
                        self.status_changed.emit("Listening")
                        sock.close()
                        continue

                    buffers[fd] = buffers.get(fd, b"") + data

                    # Process complete messages (newline-delimited)
                    while b"\n" in buffers[fd]:
                        line, buffers[fd] = buffers[fd].split(b"\n", 1)
                        try:
                            msg = BridgeMessage.from_json(line.decode("utf-8"))
                            self.command_received.emit(msg.action)
                            response = self._handler.handle(msg)
                            sock.sendall(response.to_bytes())
                        except json.JSONDecodeError as e:
                            err = BridgeMessage.error_response(
                                "", f"Invalid JSON: {e}"
                            )
                            sock.sendall(err.to_bytes())
                        except Exception as e:
                            err = BridgeMessage.error_response("", str(e))
                            try:
                                sock.sendall(err.to_bytes())
                            except OSError:
                                pass

        # Cleanup
        for client in clients:
            try:
                client.close()
            except OSError:
                pass
        if self._server_socket:
            try:
                self._server_socket.close()
            except OSError:
                pass
        self.status_changed.emit("Stopped")
        logger.info("Bridge server stopped.")

    def stop_server(self):
        """Signal the server loop to stop."""
        self._running = False


class BridgeServer:
    """Manages the bridge server lifecycle with QThread."""

    def __init__(self, handler: BridgeCommandHandler, port: int = DEFAULT_PORT):
        self._handler = handler
        self._port = port
        self._thread: QThread | None = None
        self._worker: BridgeWorker | None = None

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.isRunning()

    @property
    def worker(self) -> BridgeWorker | None:
        return self._worker

    def start(self):
        """Start the bridge server in a background thread."""
        if self.is_running:
            return

        self._thread = QThread()
        self._worker = BridgeWorker(self._handler, self._port)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.start_server)
        self._thread.start()

    def stop(self):
        """Stop the bridge server and wait for thread to finish."""
        if self._worker is not None:
            self._worker.stop_server()
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(3000)
            self._thread = None
            self._worker = None
