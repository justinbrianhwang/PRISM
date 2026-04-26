"""Noise models for quantum simulation using Kraus operators."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .state_vector import StateVector
from .circuit import GateInstance
from .gates import X_MATRIX, Y_MATRIX, Z_MATRIX, I_MATRIX


class NoiseChannel(ABC):
    """Abstract base for a single noise channel."""

    @abstractmethod
    def get_kraus_operators(self) -> list[np.ndarray]:
        ...

    @property
    @abstractmethod
    def probability(self) -> float:
        ...


class BitFlipNoise(NoiseChannel):
    """Bit-flip noise: X applied with probability p."""

    def __init__(self, p: float):
        if not 0 <= p <= 1:
            raise ValueError(f"Probability must be in [0, 1], got {p}")
        self._p = p

    @property
    def probability(self) -> float:
        return self._p

    def get_kraus_operators(self) -> list[np.ndarray]:
        return [
            np.sqrt(1 - self._p) * I_MATRIX,
            np.sqrt(self._p) * X_MATRIX,
        ]


class PhaseFlipNoise(NoiseChannel):
    """Phase-flip noise: Z applied with probability p."""

    def __init__(self, p: float):
        if not 0 <= p <= 1:
            raise ValueError(f"Probability must be in [0, 1], got {p}")
        self._p = p

    @property
    def probability(self) -> float:
        return self._p

    def get_kraus_operators(self) -> list[np.ndarray]:
        return [
            np.sqrt(1 - self._p) * I_MATRIX,
            np.sqrt(self._p) * Z_MATRIX,
        ]


class DepolarizingNoise(NoiseChannel):
    """Depolarizing noise channel."""

    def __init__(self, p: float):
        if not 0 <= p <= 1:
            raise ValueError(f"Probability must be in [0, 1], got {p}")
        self._p = p

    @property
    def probability(self) -> float:
        return self._p

    def get_kraus_operators(self) -> list[np.ndarray]:
        return [
            np.sqrt(1 - self._p) * I_MATRIX,
            np.sqrt(self._p / 3) * X_MATRIX,
            np.sqrt(self._p / 3) * Y_MATRIX,
            np.sqrt(self._p / 3) * Z_MATRIX,
        ]


class AmplitudeDampingNoise(NoiseChannel):
    """Amplitude damping (energy relaxation) noise."""

    def __init__(self, gamma: float):
        if not 0 <= gamma <= 1:
            raise ValueError(f"Gamma must be in [0, 1], got {gamma}")
        self._gamma = gamma

    @property
    def probability(self) -> float:
        return self._gamma

    def get_kraus_operators(self) -> list[np.ndarray]:
        K0 = np.array([[1, 0],
                        [0, np.sqrt(1 - self._gamma)]], dtype=np.complex128)
        K1 = np.array([[0, np.sqrt(self._gamma)],
                        [0, 0]], dtype=np.complex128)
        return [K0, K1]


class ReadoutError:
    """Classical readout error model.

    Applies a confusion matrix to measurement outcomes.
    p01: probability of reading 1 when true state is 0
    p10: probability of reading 0 when true state is 1
    """

    def __init__(self, p01: float = 0.0, p10: float = 0.0):
        if not (0 <= p01 <= 1 and 0 <= p10 <= 1):
            raise ValueError("Readout error probabilities must be in [0, 1]")
        self.p01 = p01
        self.p10 = p10

    @property
    def confusion_matrix(self) -> np.ndarray:
        """2x2 confusion matrix: C[measured][true]."""
        return np.array([
            [1 - self.p01, self.p10],
            [self.p01, 1 - self.p10],
        ])

    def apply_to_bitstring(self, bitstring: str,
                           rng: np.random.Generator) -> str:
        """Apply readout error to a single bitstring stochastically."""
        result = []
        for bit in bitstring:
            true_val = int(bit)
            if true_val == 0:
                measured = 1 if rng.random() < self.p01 else 0
            else:
                measured = 0 if rng.random() < self.p10 else 1
            result.append(str(measured))
        return "".join(result)

    def apply_to_distribution(self, probs: np.ndarray, num_qubits: int) -> np.ndarray:
        """Apply readout error via confusion matrix to a probability distribution.

        Distribution-transform mode: applies the per-bit 2x2 confusion matrix
        independently to each qubit axis using tensor contraction. This avoids
        building the full 2^n x 2^n Kronecker product matrix, keeping memory
        at O(2^n) instead of O(4^n).

        The probability vector is reshaped to (2, 2, ..., 2) with n axes,
        then each axis is contracted with the 2x2 confusion matrix C[measured][true].

        Args:
            probs: Probability distribution over 2^n basis states.
            num_qubits: Number of qubits.

        Returns:
            Transformed probability distribution.
        """
        c1 = self.confusion_matrix  # 2x2: C[measured][true]

        # Reshape flat probs to tensor with one axis per qubit
        p_tensor = probs.reshape([2] * num_qubits)

        # Apply 2x2 confusion matrix to each qubit axis independently
        # C[measured][true] contracted along the 'true' axis
        for axis in range(num_qubits):
            p_tensor = np.tensordot(c1, p_tensor, axes=([1], [axis]))
            # tensordot puts the new axis first; move it back to position
            p_tensor = np.moveaxis(p_tensor, 0, axis)

        result = p_tensor.reshape(-1)
        total = result.sum()
        if total > 1e-15:
            result /= total
        return result

    def to_dict(self) -> dict:
        return {"p01": self.p01, "p10": self.p10}

    @classmethod
    def from_dict(cls, data: dict) -> ReadoutError:
        return cls(p01=data.get("p01", 0.0), p10=data.get("p10", 0.0))


class NoiseModel:
    """Configures which noise channels to apply after gates."""

    def __init__(self):
        self._global_noise: list[NoiseChannel] = []
        self._gate_noise: dict[str, list[NoiseChannel]] = {}
        self._readout_error: ReadoutError | None = None
        self._rng = np.random.default_rng()

    @property
    def readout_error(self) -> ReadoutError | None:
        return self._readout_error

    def set_readout_error(self, error: ReadoutError) -> None:
        self._readout_error = error

    def add_global_noise(self, channel: NoiseChannel):
        self._global_noise.append(channel)

    def add_gate_noise(self, gate_name: str, channel: NoiseChannel):
        if gate_name not in self._gate_noise:
            self._gate_noise[gate_name] = []
        self._gate_noise[gate_name].append(channel)

    def set_seed(self, seed: int):
        self._rng = np.random.default_rng(seed)

    def apply(self, state: StateVector, gate: GateInstance):
        """Apply noise to state after a gate operation.

        Uses stochastic Kraus operator selection for state vector simulation.
        """
        channels = list(self._global_noise)
        if gate.gate_name in self._gate_noise:
            channels.extend(self._gate_noise[gate.gate_name])

        for channel in channels:
            self._apply_channel(state, channel, gate.target_qubits)

    def _apply_channel(self, state: StateVector, channel: NoiseChannel,
                       target_qubits: list[int]):
        """Apply a single noise channel stochastically.

        For each Kraus operator K_i, compute p_i = ||K_i|psi>||^2,
        then randomly select one operator and apply it.
        """
        kraus_ops = channel.get_kraus_operators()

        # For single-qubit noise, apply to each target qubit
        for qubit in target_qubits:
            if qubit >= state.num_qubits:
                continue

            # Compute probability for each Kraus operator
            probs = []
            results = []
            for K in kraus_ops:
                temp = state.copy()
                temp.apply_gate(K, [qubit])
                p = np.sum(np.abs(temp.data) ** 2)
                probs.append(p)
                results.append(temp)

            probs = np.array(probs)
            total = probs.sum()
            if total > 1e-15:
                probs /= total

            # Select one Kraus operator
            idx = self._rng.choice(len(kraus_ops), p=probs)
            state.data = results[idx].data

            # Renormalize
            norm = np.sqrt(np.sum(np.abs(state.data) ** 2))
            if norm > 1e-15:
                state._data /= norm

    def to_dict(self) -> dict:
        """Serialize noise model configuration."""
        result = {"global": [], "gate_specific": {}}
        for ch in self._global_noise:
            result["global"].append({
                "type": type(ch).__name__,
                "probability": ch.probability,
            })
        for gate_name, channels in self._gate_noise.items():
            result["gate_specific"][gate_name] = [
                {"type": type(ch).__name__, "probability": ch.probability}
                for ch in channels
            ]
        if self._readout_error is not None:
            result["readout_error"] = self._readout_error.to_dict()
        return result

    @classmethod
    def from_dict(cls, data: dict) -> NoiseModel:
        """Deserialize noise model from dict."""
        channel_types = {
            "BitFlipNoise": BitFlipNoise,
            "PhaseFlipNoise": PhaseFlipNoise,
            "DepolarizingNoise": DepolarizingNoise,
            "AmplitudeDampingNoise": AmplitudeDampingNoise,
        }
        model = cls()
        for ch_data in data.get("global", []):
            ch_cls = channel_types[ch_data["type"]]
            model.add_global_noise(ch_cls(ch_data["probability"]))
        for gate_name, channels in data.get("gate_specific", {}).items():
            for ch_data in channels:
                ch_cls = channel_types[ch_data["type"]]
                model.add_gate_noise(gate_name, ch_cls(ch_data["probability"]))
        if "readout_error" in data:
            model.set_readout_error(ReadoutError.from_dict(data["readout_error"]))
        return model
