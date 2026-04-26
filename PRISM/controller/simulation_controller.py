"""Simulation controller using QThread worker pattern for non-blocking simulation."""

from __future__ import annotations

import time
from typing import Any

from PyQt6.QtCore import QObject, QThread, pyqtSignal, pyqtSlot, QMutex, QWaitCondition

from quantum_sim.engine.circuit import QuantumCircuit
from quantum_sim.engine.simulator import Simulator, SimulationResult
from quantum_sim.engine.state_vector import StateVector
from quantum_sim.engine.noise import NoiseModel


class SimulationWorker(QObject):
    """Worker object that performs simulation in a background thread.

    This runs on a QThread and communicates results back to the main thread
    via signals.
    """

    # Signals emitted by the worker
    finished = pyqtSignal(object)          # SimulationResult
    step_updated = pyqtSignal(object, int) # (StateVector, column_index)
    error = pyqtSignal(str)
    progress = pyqtSignal(int)             # percentage 0-100

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self._circuit: QuantumCircuit | None = None
        self._shots: int = 1024
        self._step_mode: bool = False
        self._step_delay_ms: int = 500
        self._noise_model: NoiseModel | None = None
        self._stop_requested: bool = False
        self._mutex = QMutex()

    def configure(
        self,
        circuit: QuantumCircuit,
        shots: int = 1024,
        step_mode: bool = False,
        step_delay_ms: int = 500,
        noise_model: NoiseModel | None = None,
    ) -> None:
        """Configure the worker before starting.

        Must be called before the thread starts.
        """
        self._circuit = circuit
        self._shots = shots
        self._step_mode = step_mode
        self._step_delay_ms = step_delay_ms
        self._noise_model = noise_model
        self._stop_requested = False

    def request_stop(self) -> None:
        """Request the worker to stop at the next safe point."""
        self._mutex.lock()
        self._stop_requested = True
        self._mutex.unlock()

    def _is_stopped(self) -> bool:
        self._mutex.lock()
        stopped = self._stop_requested
        self._mutex.unlock()
        return stopped

    @pyqtSlot()
    def run(self) -> None:
        """Execute the simulation. Called when the thread starts."""
        try:
            if self._circuit is None:
                self.error.emit("No circuit configured")
                return

            if self._step_mode:
                self._run_step_by_step()
            else:
                self._run_full()
        except Exception as exc:
            self.error.emit(f"Simulation error: {exc}")

    def _run_full(self) -> None:
        """Run the full simulation."""
        simulator = Simulator(noise_model=self._noise_model)

        if self._is_stopped():
            return

        if self._noise_model is not None:
            result = simulator.run_with_noise(
                self._circuit, shots=self._shots
            )
        else:
            result = simulator.run(
                self._circuit, shots=self._shots, record_steps=False
            )

        if not self._is_stopped():
            self.progress.emit(100)
            self.finished.emit(result)

    def _run_step_by_step(self) -> None:
        """Run simulation step by step, emitting state after each column."""
        simulator = Simulator(noise_model=self._noise_model)

        step_generator = simulator.run_step_by_step(self._circuit)
        ordered_gates = self._circuit.get_ordered_gates()
        total_steps = len(ordered_gates) + 1  # +1 for initial state

        for step_idx, (state, col_idx) in enumerate(step_generator):
            if self._is_stopped():
                return

            self.step_updated.emit(state, col_idx)

            # Report progress
            if total_steps > 0:
                pct = int((step_idx + 1) / total_steps * 100)
                self.progress.emit(min(pct, 99))

            # Delay between steps (except for the last one)
            if self._step_delay_ms > 0 and step_idx < total_steps - 1:
                # Sleep in small intervals so we can check for stop
                elapsed = 0
                interval = 50  # ms
                while elapsed < self._step_delay_ms:
                    if self._is_stopped():
                        return
                    sleep_time = min(interval, self._step_delay_ms - elapsed)
                    time.sleep(sleep_time / 1000.0)
                    elapsed += interval

        if self._is_stopped():
            return

        # After step-by-step, run measurement sampling
        from quantum_sim.engine.measurement import MeasurementEngine

        last_state = StateVector(self._circuit.num_qubits)
        # Re-run to get the final state for sampling
        sim_result = simulator.run(
            self._circuit, shots=self._shots, record_steps=False
        )

        self.progress.emit(100)
        self.finished.emit(sim_result)


class SimulationController(QObject):
    """Controller that manages simulation execution on a background thread.

    Uses the QThread + worker pattern: the SimulationWorker is moved to a
    QThread, ensuring the UI remains responsive during long simulations.
    """

    # Public signals
    simulation_started = pyqtSignal()
    simulation_finished = pyqtSignal(object)      # SimulationResult
    step_state_updated = pyqtSignal(object, int)   # (StateVector, column_index)
    error_occurred = pyqtSignal(str)
    progress_updated = pyqtSignal(int)             # percentage 0-100

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)

        self._thread: QThread | None = None
        self._worker: SimulationWorker | None = None
        self._noise_model: NoiseModel | None = None
        self._step_delay_ms: int = 500
        self._running: bool = False

    @property
    def is_running(self) -> bool:
        """Whether a simulation is currently running."""
        return self._running

    def set_noise_model(self, noise_model: NoiseModel | None) -> None:
        """Set or clear the noise model for future simulations."""
        self._noise_model = noise_model

    def set_step_delay(self, delay_ms: int) -> None:
        """Set the delay between steps in step-by-step mode."""
        self._step_delay_ms = max(0, delay_ms)

    def run_simulation(self, circuit: QuantumCircuit, shots: int = 1024) -> None:
        """Run a full simulation in a background thread.

        Args:
            circuit: The quantum circuit to simulate.
            shots: Number of measurement shots.
        """
        if self._running:
            self.error_occurred.emit("A simulation is already running")
            return

        self._start_worker(circuit, shots, step_mode=False)

    def run_step_by_step(self, circuit: QuantumCircuit, shots: int = 1024) -> None:
        """Run simulation step by step in a background thread.

        Emits step_state_updated after each circuit column.

        Args:
            circuit: The quantum circuit to simulate.
            shots: Number of measurement shots (used after stepping completes).
        """
        if self._running:
            self.error_occurred.emit("A simulation is already running")
            return

        self._start_worker(circuit, shots, step_mode=True)

    def stop_simulation(self) -> None:
        """Stop the currently running simulation."""
        if self._worker is not None:
            self._worker.request_stop()

        self._cleanup_thread()

    def _start_worker(self, circuit: QuantumCircuit, shots: int,
                      step_mode: bool) -> None:
        """Create and start the worker thread."""
        self._cleanup_thread()

        self._thread = QThread()
        self._worker = SimulationWorker()

        self._worker.configure(
            circuit=circuit,
            shots=shots,
            step_mode=step_mode,
            step_delay_ms=self._step_delay_ms,
            noise_model=self._noise_model,
        )

        # Move worker to thread
        self._worker.moveToThread(self._thread)

        # Connect signals
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_finished)
        self._worker.step_updated.connect(self._on_step_updated)
        self._worker.error.connect(self._on_error)
        self._worker.progress.connect(self._on_progress)

        # Cleanup connections
        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        self._thread.finished.connect(self._on_thread_finished)

        self._running = True
        self.simulation_started.emit()
        self._thread.start()

    def _on_finished(self, result: SimulationResult) -> None:
        """Handle simulation completion."""
        self._running = False
        self.simulation_finished.emit(result)

    def _on_step_updated(self, state: StateVector, col_idx: int) -> None:
        """Forward step update to external listeners."""
        self.step_state_updated.emit(state, col_idx)

    def _on_error(self, message: str) -> None:
        """Handle simulation error."""
        self._running = False
        self.error_occurred.emit(message)

    def _on_progress(self, pct: int) -> None:
        """Forward progress updates."""
        self.progress_updated.emit(pct)

    def _on_thread_finished(self) -> None:
        """Handle thread completion."""
        self._running = False

    def _cleanup_thread(self) -> None:
        """Clean up the previous thread and worker."""
        if self._thread is not None:
            if self._worker is not None:
                self._worker.request_stop()
            if self._thread.isRunning():
                self._thread.quit()
                self._thread.wait(3000)  # Wait up to 3 seconds
                if self._thread.isRunning():
                    self._thread.terminate()
                    self._thread.wait()

        self._worker = None
        self._thread = None
        self._running = False
