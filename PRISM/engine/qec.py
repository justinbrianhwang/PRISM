"""Quantum Error Correction -- encode, detect, correct, and threshold analysis.

Provides:
- QECCode (abstract base): encode, syndrome extraction, decoding, correction
- BitFlipCode: 3-qubit bit-flip repetition code [3,1,1]
- PhaseFlipCode: 3-qubit phase-flip repetition code [3,1,1]
- SteaneCode: [[7,1,3]] CSS code correcting arbitrary single-qubit errors
- QECSimulator: run full QEC cycles and threshold sweeps
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from .circuit import QuantumCircuit, GateInstance
from .simulator import Simulator
from .state_vector import StateVector
from .noise import NoiseModel, BitFlipNoise, PhaseFlipNoise, DepolarizingNoise
from .analysis import StateAnalysis


@dataclass
class QECResult:
    """Result of a single QEC cycle."""

    encoded_state: StateVector
    noisy_state: StateVector
    syndrome: list[int]
    corrected_state: StateVector
    fidelity_before: float  # fidelity of noisy state vs ideal
    fidelity_after: float  # fidelity of corrected state vs ideal
    correction_applied: list[tuple[str, int]]  # [(gate_name, qubit)]
    logical_z_expectation: float = 0.0  # <Z_L> after correction
    logical_error_detected: bool = False  # True if Z_L sign is wrong


@dataclass
class ThresholdPoint:
    """Result at one physical error rate in a threshold sweep."""

    physical_rate: float
    logical_rate: float
    success_rate: float
    avg_fidelity: float
    logical_z_fidelity: float = 0.0     # mean |<Z_L>|
    decoder_success_rate: float = 0.0   # fraction with correct Z_L sign
    projection_logical_rate: float = 0.0  # 1 - F(corrected, ideal_codeword)


class QECCode(ABC):
    """Abstract base for quantum error correcting codes."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def data_qubits(self) -> int:
        """Number of data qubits."""
        ...

    @property
    @abstractmethod
    def ancilla_qubits(self) -> int:
        """Number of ancilla qubits."""
        ...

    @property
    def total_qubits(self) -> int:
        return self.data_qubits + self.ancilla_qubits

    @property
    @abstractmethod
    def code_distance(self) -> int:
        ...

    @abstractmethod
    def encode(self, logical_state: int) -> StateVector:
        """Encode a logical |0> or |1> into the code space.

        Args:
            logical_state: 0 or 1.

        Returns:
            StateVector of the full (data + ancilla) system.
        """
        ...

    @abstractmethod
    def extract_syndrome(self, state: StateVector, rng: np.random.Generator) -> list[int]:
        """Measure syndrome bits (non-destructively).

        Returns list of syndrome measurement outcomes.
        """
        ...

    @abstractmethod
    def decode_syndrome(self, syndrome: list[int]) -> list[tuple[str, int]]:
        """Map syndrome to correction operations.

        Returns list of (gate_name, qubit_index) corrections.
        """
        ...

    def apply_correction(self, state: StateVector, corrections: list[tuple[str, int]]) -> None:
        """Apply correction gates to state in-place."""
        from .gates import X_MATRIX, Z_MATRIX
        gate_map = {"X": X_MATRIX, "Z": Z_MATRIX}
        for gate_name, qubit in corrections:
            if gate_name in gate_map and qubit < state.num_qubits:
                state.apply_gate(gate_map[gate_name], [qubit])

    @abstractmethod
    def logical_fidelity(self, state: StateVector, logical_state: int) -> float:
        """Compute fidelity of the physical state with the ideal logical state."""
        ...

    @abstractmethod
    def logical_z_operators(self) -> list[int]:
        """Qubit indices for the logical Z operator.

        Z_L = Z_{q0} (tensor) Z_{q1} (tensor) ... for the listed qubits.
        """
        ...

    def logical_z_expectation(self, state: StateVector) -> float:
        """Compute <psi|Z_L|psi>.

        For |0>_L this should be +1, for |1>_L this should be -1.
        """
        z_qubits = self.logical_z_operators()
        n = state.num_qubits
        probs = state.probabilities
        expectation = 0.0

        for idx in range(len(probs)):
            # Compute parity of Z_L qubits
            parity = 0
            for q in z_qubits:
                bit_pos = n - 1 - q
                parity ^= (idx >> bit_pos) & 1
            # Z eigenvalue: +1 for |0>, -1 for |1>
            sign = 1.0 if parity == 0 else -1.0
            expectation += sign * probs[idx]

        return float(expectation)


# ---- 3-Qubit Bit-Flip Code -----------------------------------------------

class BitFlipCode(QECCode):
    """3-qubit bit-flip repetition code.

    Encodes |0>_L = |000>, |1>_L = |111>.
    Detects and corrects single X (bit-flip) errors.

    Layout: 3 data qubits (0,1,2) + 2 ancilla qubits (3,4).
    Syndrome:
        ancilla 3 measures Z0*Z1 (parity of q0,q1)
        ancilla 4 measures Z1*Z2 (parity of q1,q2)
    """

    @property
    def name(self) -> str:
        return "Bit-Flip [3,1,1]"

    @property
    def data_qubits(self) -> int:
        return 3

    @property
    def ancilla_qubits(self) -> int:
        return 2

    @property
    def code_distance(self) -> int:
        return 1

    def encode(self, logical_state: int) -> StateVector:
        qc = QuantumCircuit(5)
        if logical_state == 1:
            qc.add_gate(GateInstance("X", [0], [], 0))
        # Fan out: |psi>|00> -> CNOT(0,1), CNOT(0,2)
        qc.add_gate(GateInstance("CNOT", [0, 1], [], 1))
        qc.add_gate(GateInstance("CNOT", [0, 2], [], 2))

        sim = Simulator()
        result = sim.run(qc, shots=0)
        return result.final_state

    def extract_syndrome(self, state: StateVector, rng: np.random.Generator) -> list[int]:
        """Extract syndrome by projective measurement simulation.

        We compute the expected parity outcomes from the state amplitudes.
        """
        return _extract_parity_syndrome(
            state,
            parity_checks=[(0, 1), (1, 2)],
            ancilla_start=3,
            rng=rng,
        )

    def decode_syndrome(self, syndrome: list[int]) -> list[tuple[str, int]]:
        s0, s1 = syndrome[0], syndrome[1]
        if s0 == 0 and s1 == 0:
            return []  # no error
        if s0 == 1 and s1 == 0:
            return [("X", 0)]
        if s0 == 1 and s1 == 1:
            return [("X", 1)]
        if s0 == 0 and s1 == 1:
            return [("X", 2)]
        return []

    def logical_fidelity(self, state: StateVector, logical_state: int) -> float:
        ideal = self.encode(logical_state)
        return StateAnalysis.state_fidelity(ideal.data, state.data)

    def logical_z_operators(self) -> list[int]:
        # Z_L = Z0 Z1 Z2 (all three data qubits)
        return [0, 1, 2]


# ---- 3-Qubit Phase-Flip Code ---------------------------------------------

class PhaseFlipCode(QECCode):
    """3-qubit phase-flip repetition code.

    Encodes |0>_L = |+++>, |1>_L = |--->.
    Detects and corrects single Z (phase-flip) errors.

    Layout: 3 data + 2 ancilla.
    Uses Hadamard basis: H on all data qubits, then same parity checks as bit-flip.
    """

    @property
    def name(self) -> str:
        return "Phase-Flip [3,1,1]"

    @property
    def data_qubits(self) -> int:
        return 3

    @property
    def ancilla_qubits(self) -> int:
        return 2

    @property
    def code_distance(self) -> int:
        return 1

    def encode(self, logical_state: int) -> StateVector:
        qc = QuantumCircuit(5)
        if logical_state == 1:
            qc.add_gate(GateInstance("X", [0], [], 0))
        qc.add_gate(GateInstance("CNOT", [0, 1], [], 1))
        qc.add_gate(GateInstance("CNOT", [0, 2], [], 2))
        # Apply H to all data qubits to move to phase basis
        qc.add_gate(GateInstance("H", [0], [], 3))
        qc.add_gate(GateInstance("H", [1], [], 3))
        qc.add_gate(GateInstance("H", [2], [], 3))

        sim = Simulator()
        result = sim.run(qc, shots=0)
        return result.final_state

    def extract_syndrome(self, state: StateVector, rng: np.random.Generator) -> list[int]:
        # Transform to computational basis, extract, transform back
        from .gates import H_MATRIX
        temp = state.copy()
        for q in range(3):
            temp.apply_gate(H_MATRIX, [q])

        syndrome = _extract_parity_syndrome(
            temp,
            parity_checks=[(0, 1), (1, 2)],
            ancilla_start=3,
            rng=rng,
        )
        return syndrome

    def decode_syndrome(self, syndrome: list[int]) -> list[tuple[str, int]]:
        s0, s1 = syndrome[0], syndrome[1]
        if s0 == 0 and s1 == 0:
            return []
        if s0 == 1 and s1 == 0:
            return [("Z", 0)]
        if s0 == 1 and s1 == 1:
            return [("Z", 1)]
        if s0 == 0 and s1 == 1:
            return [("Z", 2)]
        return []

    def logical_fidelity(self, state: StateVector, logical_state: int) -> float:
        ideal = self.encode(logical_state)
        return StateAnalysis.state_fidelity(ideal.data, state.data)

    def logical_z_operators(self) -> list[int]:
        # X_L = X0 X1 X2 (phase-flip code operates in X basis)
        return [0, 1, 2]

    def logical_z_expectation(self, state: StateVector) -> float:
        """For phase-flip code, logical Z is X_L = X0 X1 X2."""
        from .gates import H_MATRIX
        # Transform to X basis, then measure Z parity
        temp = state.copy()
        for q in range(3):
            temp.apply_gate(H_MATRIX, [q])
        # Now use parent's Z-parity computation
        return super().logical_z_expectation(temp)


# ---- Steane [[7,1,3]] Code -----------------------------------------------

class SteaneCode(QECCode):
    """Steane [[7,1,3]] CSS code.

    7 data qubits + 6 ancilla qubits = 13 total (within 16-qubit limit).
    Corrects any single-qubit error (X, Y, or Z).

    Data qubits: 0-6
    X-syndrome ancilla: 7, 8, 9 (measure Z-type stabilizers)
    Z-syndrome ancilla: 10, 11, 12 (measure X-type stabilizers)

    H matrix (classical [7,4,3] Hamming):
        H = [[1,0,1,0,1,0,1],
             [0,1,1,0,0,1,1],
             [0,0,0,1,1,1,1]]
    """

    # Parity check matrix rows (0-indexed qubit positions where each check has support)
    _HX = [[0, 2, 4, 6], [1, 2, 5, 6], [3, 4, 5, 6]]  # X stabilizer generators
    _HZ = [[0, 2, 4, 6], [1, 2, 5, 6], [3, 4, 5, 6]]  # Z stabilizer generators

    @property
    def name(self) -> str:
        return "Steane [[7,1,3]]"

    @property
    def data_qubits(self) -> int:
        return 7

    @property
    def ancilla_qubits(self) -> int:
        return 6

    @property
    def code_distance(self) -> int:
        return 3

    def encode(self, logical_state: int) -> StateVector:
        """Encode logical qubit into Steane code.

        |0>_L = (1/sqrt(8)) * sum of all even-weight codewords of [7,4,3] Hamming code
        |1>_L = (1/sqrt(8)) * sum of all odd-weight codewords
        """
        # The 16 codewords of the [7,4,3] Hamming code
        # Generator matrix: rows generate the code space
        gen = np.array([
            [1, 0, 0, 0, 1, 1, 0],
            [0, 1, 0, 0, 1, 0, 1],
            [0, 0, 1, 0, 0, 1, 1],
            [0, 0, 0, 1, 1, 1, 1],
        ], dtype=int)

        codewords = []
        for i in range(16):
            bits = [(i >> b) & 1 for b in range(4)]
            cw = np.array(bits) @ gen % 2
            codewords.append(tuple(cw.tolist()))

        # |0>_L: even weight codewords, |1>_L: odd weight
        even_cw = [cw for cw in codewords if sum(cw) % 2 == 0]
        odd_cw = [cw for cw in codewords if sum(cw) % 2 == 1]

        n_total = 13
        state_data = np.zeros(2 ** n_total, dtype=np.complex128)

        target_cw = even_cw if logical_state == 0 else odd_cw
        amp = 1.0 / np.sqrt(len(target_cw))

        for cw in target_cw:
            # Data qubits 0-6 have codeword bits, ancilla 7-12 are |0>
            idx = 0
            for qi, bit in enumerate(cw):
                if bit:
                    idx |= (1 << (n_total - 1 - qi))
            state_data[idx] = amp

        sv = StateVector(n_total)
        sv.data = state_data
        return sv

    def extract_syndrome(self, state: StateVector, rng: np.random.Generator) -> list[int]:
        """Extract 6 syndrome bits (3 X-syndrome + 3 Z-syndrome)."""
        # X-syndrome: measure Z-type stabilizers on data qubits
        x_syndrome = []
        for check in self._HX:
            parity = _compute_z_parity(state, check)
            x_syndrome.append(parity)

        # Z-syndrome: measure X-type stabilizers on data qubits
        z_syndrome = []
        from .gates import H_MATRIX
        temp = state.copy()
        for q in range(7):
            temp.apply_gate(H_MATRIX, [q])
        for check in self._HZ:
            parity = _compute_z_parity(temp, check)
            z_syndrome.append(parity)

        return x_syndrome + z_syndrome

    def decode_syndrome(self, syndrome: list[int]) -> list[tuple[str, int]]:
        """Decode 6-bit syndrome into correction.

        First 3 bits: X-syndrome (detect Z errors)
        Last 3 bits: Z-syndrome (detect X errors)
        """
        corrections = []

        # Z-syndrome → X correction
        z_syn = syndrome[3:6]
        z_idx = z_syn[0] + 2 * z_syn[1] + 4 * z_syn[2]
        if z_idx > 0 and z_idx <= 7:
            corrections.append(("X", z_idx - 1))

        # X-syndrome → Z correction
        x_syn = syndrome[0:3]
        x_idx = x_syn[0] + 2 * x_syn[1] + 4 * x_syn[2]
        if x_idx > 0 and x_idx <= 7:
            corrections.append(("Z", x_idx - 1))

        return corrections

    def logical_fidelity(self, state: StateVector, logical_state: int) -> float:
        ideal = self.encode(logical_state)
        return StateAnalysis.state_fidelity(ideal.data, state.data)

    def logical_z_operators(self) -> list[int]:
        # Z_L = Z0 Z1 Z2 Z3 Z4 Z5 Z6 (all 7 data qubits)
        return [0, 1, 2, 3, 4, 5, 6]


# ---- Helper functions -----------------------------------------------------

def _extract_parity_syndrome(
    state: StateVector,
    parity_checks: list[tuple[int, int]],
    ancilla_start: int,
    rng: np.random.Generator,
) -> list[int]:
    """Extract Z-parity syndrome for pairs of qubits."""
    syndrome = []
    for qa, qb in parity_checks:
        parity = _compute_z_parity(state, [qa, qb])
        syndrome.append(parity)
    return syndrome


def _compute_z_parity(state: StateVector, qubits: list[int]) -> int:
    """Compute the most likely parity of Z measurements on given qubits.

    Returns 0 if even parity is more likely, 1 if odd.
    """
    n = state.num_qubits
    probs = state.probabilities
    p_even = 0.0
    p_odd = 0.0

    for idx in range(len(probs)):
        parity = 0
        for q in qubits:
            bit_pos = n - 1 - q
            parity ^= (idx >> bit_pos) & 1
        if parity == 0:
            p_even += probs[idx]
        else:
            p_odd += probs[idx]

    return 0 if p_even >= p_odd else 1


# ---- QEC Simulator --------------------------------------------------------

class QECSimulator:
    """Run QEC cycles with noise injection and threshold analysis."""

    def __init__(self, code: QECCode):
        self._code = code

    def run_cycle(
        self,
        logical_state: int = 0,
        noise_type: str = "bit_flip",
        noise_prob: float = 0.1,
        seed: int | None = None,
    ) -> QECResult:
        """Run a single QEC encode-noise-syndrome-correct cycle.

        Args:
            logical_state: 0 or 1 to encode.
            noise_type: "bit_flip", "phase_flip", or "depolarizing".
            noise_prob: Error probability.
            seed: Reproducibility seed.
        """
        rng = np.random.default_rng(seed)

        # Encode: generate ideal codeword once and reuse for all fidelity computations
        ideal_codeword = self._code.encode(logical_state)

        # Apply noise to data qubits only
        noisy = ideal_codeword.copy()
        self._apply_noise(noisy, noise_type, noise_prob, rng)

        # Extract syndrome
        syndrome = self._code.extract_syndrome(noisy, rng)

        # Decode and correct
        corrections = self._code.decode_syndrome(syndrome)
        corrected = noisy.copy()
        self._code.apply_correction(corrected, corrections)

        # Compute fidelities against the single ideal codeword (no redundant encode)
        fid_before = StateAnalysis.state_fidelity(ideal_codeword.data, noisy.data)
        fid_after = StateAnalysis.state_fidelity(ideal_codeword.data, corrected.data)

        # Compute logical Z expectation
        z_exp = self._code.logical_z_expectation(corrected)
        # Expected sign: +1 for |0>_L, -1 for |1>_L
        expected_sign = 1.0 if logical_state == 0 else -1.0
        logical_error = (z_exp * expected_sign) < 0

        return QECResult(
            encoded_state=ideal_codeword,
            noisy_state=noisy,
            syndrome=syndrome,
            corrected_state=corrected,
            fidelity_before=fid_before,
            fidelity_after=fid_after,
            correction_applied=corrections,
            logical_z_expectation=z_exp,
            logical_error_detected=logical_error,
        )

    def threshold_sweep(
        self,
        noise_probs: list[float],
        n_trials: int = 100,
        noise_type: str = "bit_flip",
        seed: int | None = None,
    ) -> list[ThresholdPoint]:
        """Sweep physical error rate and compute logical error rate.

        Computes three independent logical error metrics:
        1. Fidelity-based: success if F(corrected, ideal) > 0.5
        2. Z_L sign-based: success if <Z_L> has correct sign
        3. Projection-based: 1 - mean fidelity to ideal codeword

        Args:
            noise_probs: List of physical error probabilities.
            n_trials: Number of trials per probability.
            noise_type: Type of noise channel.
            seed: Base seed.

        Returns:
            List of ThresholdPoint with physical/logical error rates.
        """
        rng = np.random.default_rng(seed)
        results = []

        for p in noise_probs:
            successes = 0
            total_fid = 0.0
            z_fid_sum = 0.0
            z_sign_correct = 0
            proj_fid_sum = 0.0

            for trial in range(n_trials):
                trial_seed = int(rng.integers(0, 2**63))
                logical = trial % 2  # alternate |0> and |1>
                qec_result = self.run_cycle(
                    logical_state=logical,
                    noise_type=noise_type,
                    noise_prob=p,
                    seed=trial_seed,
                )

                # Success if fidelity improved or stayed high
                if qec_result.fidelity_after > 0.5:
                    successes += 1
                total_fid += qec_result.fidelity_after

                # Z_L-based metrics
                z_fid_sum += abs(qec_result.logical_z_expectation)
                if not qec_result.logical_error_detected:
                    z_sign_correct += 1

                # Projection-based: fidelity to ideal codeword
                proj_fid_sum += qec_result.fidelity_after

            logical_rate = 1.0 - successes / n_trials
            success_rate = successes / n_trials
            avg_fidelity = total_fid / n_trials
            proj_logical_rate = 1.0 - proj_fid_sum / n_trials

            results.append(ThresholdPoint(
                physical_rate=p,
                logical_rate=logical_rate,
                success_rate=success_rate,
                avg_fidelity=avg_fidelity,
                logical_z_fidelity=z_fid_sum / n_trials,
                decoder_success_rate=z_sign_correct / n_trials,
                projection_logical_rate=proj_logical_rate,
            ))

        return results

    def projection_logical_error(
        self,
        logical_state: int,
        noise_type: str,
        noise_prob: float,
        n_trials: int = 100,
        seed: int | None = None,
    ) -> dict:
        """Compute projection-based logical error rate.

        Logical error = 1 - F(corrected_state, ideal_codeword), averaged
        over n_trials. This is complementary to the sign-based Z_L metric.

        Returns:
            {
                "mean_fidelity": float,
                "logical_error_rate": float,  # 1 - mean_fidelity
                "z_sign_error_rate": float,
                "n_trials": int,
            }
        """
        rng = np.random.default_rng(seed)
        fid_sum = 0.0
        z_errors = 0

        for trial in range(n_trials):
            trial_seed = int(rng.integers(0, 2**63))
            result = self.run_cycle(
                logical_state=logical_state,
                noise_type=noise_type,
                noise_prob=noise_prob,
                seed=trial_seed,
            )
            fid_sum += result.fidelity_after
            if result.logical_error_detected:
                z_errors += 1

        mean_fid = fid_sum / n_trials
        return {
            "mean_fidelity": mean_fid,
            "logical_error_rate": 1.0 - mean_fid,
            "z_sign_error_rate": z_errors / n_trials,
            "n_trials": n_trials,
        }

    def _apply_noise(
        self,
        state: StateVector,
        noise_type: str,
        prob: float,
        rng: np.random.Generator,
    ) -> None:
        """Apply noise to data qubits stochastically."""
        from .gates import X_MATRIX, Z_MATRIX, Y_MATRIX

        for q in range(self._code.data_qubits):
            if noise_type == "bit_flip":
                if rng.random() < prob:
                    state.apply_gate(X_MATRIX, [q])
            elif noise_type == "phase_flip":
                if rng.random() < prob:
                    state.apply_gate(Z_MATRIX, [q])
            elif noise_type == "depolarizing":
                r = rng.random()
                if r < prob / 3:
                    state.apply_gate(X_MATRIX, [q])
                elif r < 2 * prob / 3:
                    state.apply_gate(Y_MATRIX, [q])
                elif r < prob:
                    state.apply_gate(Z_MATRIX, [q])


# Convenience: available codes
AVAILABLE_CODES = {
    "Bit-Flip [3,1,1]": BitFlipCode,
    "Phase-Flip [3,1,1]": PhaseFlipCode,
    "Steane [[7,1,3]]": SteaneCode,
}
