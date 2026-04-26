"""Experiment configuration management and seed management for reproducibility.

Provides :class:`ExperimentConfig` for serialising / deserialising full
experiment snapshots (circuit, noise model, results, analysis) and
:class:`SeedManager` for deterministic, fork-safe random number generation.
"""

from __future__ import annotations

import json
import datetime
from dataclasses import dataclass, field, asdict, is_dataclass
from pathlib import Path

import numpy as np

from quantum_sim.engine.circuit import QuantumCircuit
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from quantum_sim.engine.simulator import SimulationResult


# ---------------------------------------------------------------------------
# ExperimentConfig
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    """Immutable snapshot of an experiment's full configuration and results.

    This dataclass captures every detail needed to reproduce or review a
    simulation run: the circuit definition, noise model, random seed, shot
    count, raw results, derived analysis, and arbitrary user metadata.

    Parameters
    ----------
    seed : int | None
        Master random seed used for the simulation.  ``None`` means the
        simulation was not seeded (non-deterministic).
    circuit : dict | None
        Serialised circuit representation produced by
        :pymeth:`QuantumCircuit.to_dict`.
    noise_model : dict | None
        Serialised noise-model representation produced by
        :pymeth:`NoiseModel.to_dict`.
    num_shots : int
        Number of measurement shots executed.
    timestamp : str
        ISO-8601 formatted timestamp of when the experiment was created.
    simulator_version : str
        Version string of the simulator that produced the results.
    results : dict | None
        Raw simulation output such as measurement counts.
    analysis : dict | None
        Computed metrics derived from the results (e.g. expectation values,
        fidelity estimates).
    metadata : dict | None
        Free-form dictionary for user notes, tags, or other annotations.
    """

    seed: int | None = None
    circuit: dict | None = None
    noise_model: dict | None = None
    num_shots: int = 1024
    timestamp: str = ""
    simulator_version: str = "1.0.0"
    results: dict | None = None
    analysis: dict | None = None
    metadata: dict | None = None

    # -- Serialisation helpers ------------------------------------------------

    @staticmethod
    def _json_default(obj):
        """Best-effort JSON conversion for numpy and dataclass objects."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, complex):
            return {"re": float(obj.real), "im": float(obj.imag)}
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        if is_dataclass(obj):
            return asdict(obj)
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    def to_json(self) -> str:
        """Serialise the experiment configuration to a JSON string.

        Returns
        -------
        str
            A JSON-formatted string representing the full experiment config.
        """
        return json.dumps(asdict(self), indent=2, default=self._json_default)

    def save(self, filepath: str | Path) -> None:
        """Write the experiment configuration to a JSON file.

        Parameters
        ----------
        filepath : str | Path
            Destination file path.  Parent directories are created
            automatically if they do not exist.
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def from_json(cls, json_str: str) -> ExperimentConfig:
        """Deserialise an :class:`ExperimentConfig` from a JSON string.

        Parameters
        ----------
        json_str : str
            JSON string previously produced by :pymeth:`to_json`.

        Returns
        -------
        ExperimentConfig
            Reconstructed experiment configuration.
        """
        data = json.loads(json_str)
        return cls(**data)

    @classmethod
    def load(cls, filepath: str | Path) -> ExperimentConfig:
        """Load an :class:`ExperimentConfig` from a JSON file on disk.

        Parameters
        ----------
        filepath : str | Path
            Path to the JSON file to load.

        Returns
        -------
        ExperimentConfig
            Reconstructed experiment configuration.
        """
        path = Path(filepath)
        return cls.from_json(path.read_text(encoding="utf-8"))

    @classmethod
    def from_current(
        cls,
        circuit: QuantumCircuit,
        noise_model=None,
        seed: int | None = None,
        shots: int = 1024,
        result: dict | SimulationResult | None = None,
    ) -> ExperimentConfig:
        """Build an :class:`ExperimentConfig` from the current simulator state.

        This is the primary factory method used right after a simulation run
        to capture the full experiment context for later review or replay.

        Parameters
        ----------
        circuit : QuantumCircuit
            The circuit that was simulated.  Serialised via
            :pymeth:`QuantumCircuit.to_dict`.
        noise_model : NoiseModel | None, optional
            The noise model applied during simulation.  Serialised via
            ``noise_model.to_dict()`` when not ``None``.
        seed : int | None, optional
            The master random seed used for the run.
        shots : int, optional
            Number of measurement shots (default ``1024``).
        result : dict | SimulationResult | None, optional
            Raw measurement results (e.g. ``{"counts": {...}, "num_shots": N}``).

        Returns
        -------
        ExperimentConfig
            A fully populated experiment configuration snapshot.
        """
        result_payload = result
        try:
            from quantum_sim.engine.simulator import SimulationResult
            if isinstance(result, SimulationResult):
                result_payload = {
                    "measurement_counts": {
                        str(k): int(v)
                        for k, v in result.measurement_counts.items()
                    },
                    "num_shots": int(result.num_shots),
                    "seed": result.seed,
                }
        except Exception:
            result_payload = result

        return cls(
            seed=seed,
            circuit=circuit.to_dict(),
            noise_model=noise_model.to_dict() if noise_model is not None else None,
            num_shots=shots,
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            results=result_payload,
        )


# ---------------------------------------------------------------------------
# SeedManager
# ---------------------------------------------------------------------------

class SeedManager:
    """Deterministic seed manager that supports fork-safe child RNG creation.

    Wraps NumPy's :class:`numpy.random.Generator` to provide a single point
    of control for all randomness in a simulation run.  A fixed master seed
    guarantees that successive calls to :pymeth:`create_child_rng` always
    produce the same sequence of child generators, enabling full
    reproducibility.

    Parameters
    ----------
    seed : int | None
        Master seed.  When ``None`` the underlying generator is initialised
        from OS entropy (non-deterministic).

    Examples
    --------
    >>> mgr = SeedManager(42)
    >>> rng1 = mgr.create_child_rng()
    >>> rng2 = mgr.create_child_rng()
    >>> mgr.reset()
    >>> rng1b = mgr.create_child_rng()
    >>> # rng1 and rng1b will produce identical sequences
    """

    def __init__(self, seed: int | None = None):
        self._master_seed = seed
        self._rng = np.random.default_rng(seed)

    @property
    def seed(self) -> int | None:
        """Return the master seed, or ``None`` if unseeded."""
        return self._master_seed

    def set_seed(self, seed: int | None) -> None:
        """Replace the master seed and reinitialise the internal generator.

        Parameters
        ----------
        seed : int | None
            New master seed.  ``None`` reverts to OS-entropy seeding.
        """
        self._master_seed = seed
        self._rng = np.random.default_rng(seed)

    def create_child_rng(self) -> np.random.Generator:
        """Fork a new independent child RNG from the master sequence.

        Each call advances the master generator's state, so the *n*-th child
        RNG is fully determined by the master seed.

        Returns
        -------
        numpy.random.Generator
            A new, independent random number generator.
        """
        child_seed = self._rng.integers(0, 2**63)
        return np.random.default_rng(child_seed)

    def reset(self) -> None:
        """Reset the internal generator back to the master seed.

        After calling this method the next :pymeth:`create_child_rng` call
        will return the same generator as the very first call after
        construction (assuming the master seed has not been changed).
        """
        self._rng = np.random.default_rng(self._master_seed)
