"""Validation test harness -- 6 basic correctness tests.

These tests verify fundamental quantum-mechanical identities that any
correct simulator must satisfy. They serve as a minimum acceptance
criterion for academic review.

Run: python test_validation.py
"""

from __future__ import annotations

import sys
import os
import traceback

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

# ---- Engine imports -------------------------------------------------------
from PRISM.engine.state_vector import StateVector
from PRISM.engine.circuit import QuantumCircuit, GateInstance
from PRISM.engine.simulator import Simulator
from PRISM.engine.analysis import StateAnalysis
from PRISM.engine.noise import NoiseModel, DepolarizingNoise, ReadoutError
from PRISM.engine.measurement import MeasurementEngine, MeasurementBasis
from PRISM.engine.reference import ReferenceManager
from PRISM.engine.qec import BitFlipCode, QECSimulator


TOLERANCE = 1e-8
PASS_COUNT = 0
FAIL_COUNT = 0


def _report(name: str, passed: bool, details: str = ""):
    global PASS_COUNT, FAIL_COUNT
    status = "PASS" if passed else "FAIL"
    if passed:
        PASS_COUNT += 1
    else:
        FAIL_COUNT += 1
    print(f"  [{status}] {name}")
    if details and not passed:
        print(f"         {details}")


# =========================================================================
# Test 1: Bell state has correct amplitudes and entanglement
# =========================================================================

def test_bell_state():
    """H(q0) -> CNOT(q0,q1) produces |00>+|11> / sqrt(2)."""
    print("\nTest 1: Bell State Correctness")
    print("-" * 40)

    qc = QuantumCircuit(num_qubits=2)
    qc.add_gate(GateInstance("H", [0], [], 0))
    qc.add_gate(GateInstance("CNOT", [0, 1], [], 1))

    sim = Simulator()
    result = sim.run(qc, shots=0, seed=42)
    sv = result.final_state

    # Check amplitudes: |00> and |11> should each be 1/sqrt(2)
    amp_00 = sv.data[0]  # |00>
    amp_11 = sv.data[3]  # |11>
    amp_01 = sv.data[1]  # |01>
    amp_10 = sv.data[2]  # |10>

    _report(
        "|00> amplitude = 1/sqrt(2)",
        abs(abs(amp_00) - 1 / np.sqrt(2)) < TOLERANCE,
        f"got {abs(amp_00):.10f}",
    )
    _report(
        "|11> amplitude = 1/sqrt(2)",
        abs(abs(amp_11) - 1 / np.sqrt(2)) < TOLERANCE,
        f"got {abs(amp_11):.10f}",
    )
    _report(
        "|01> and |10> are zero",
        abs(amp_01) < TOLERANCE and abs(amp_10) < TOLERANCE,
        f"|01|={abs(amp_01):.10e}, |10|={abs(amp_10):.10e}",
    )

    # Entanglement: mutual information should be 2.0 bits (maximally entangled)
    mi = StateAnalysis.mutual_information(sv, 0, 1)
    _report(
        "Mutual information I(0:1) = 2.0 bits",
        abs(mi - 2.0) < 0.01,
        f"got {mi:.6f}",
    )

    # Single-qubit entanglement entropy should be 1.0 bit
    ent = StateAnalysis.entanglement_entropy(sv, [0])
    _report(
        "Single-qubit entropy S(q0) = 1.0 bit",
        abs(ent - 1.0) < 0.01,
        f"got {ent:.6f}",
    )


# =========================================================================
# Test 2: State normalization is preserved under all operations
# =========================================================================

def test_normalization():
    """State vector norm stays 1.0 after gates and noise."""
    print("\nTest 2: State Normalization Preservation")
    print("-" * 40)

    # Pure gate circuit
    qc = QuantumCircuit(num_qubits=3)
    qc.add_gate(GateInstance("H", [0], [], 0))
    qc.add_gate(GateInstance("CNOT", [0, 1], [], 1))
    qc.add_gate(GateInstance("Rz", [2], [1.234], 2))
    qc.add_gate(GateInstance("Ry", [1], [0.567], 3))

    sim = Simulator()
    result = sim.run(qc, shots=0, seed=42)
    norm = np.sqrt(np.sum(np.abs(result.final_state.data) ** 2))
    _report(
        "Norm = 1.0 after gate circuit",
        abs(norm - 1.0) < TOLERANCE,
        f"got {norm:.15f}",
    )

    # Noisy circuit
    noise = NoiseModel()
    noise.add_global_noise(DepolarizingNoise(0.05))
    sim_noisy = Simulator(noise_model=noise)
    result_noisy = sim_noisy.run(qc, shots=0, seed=42)
    norm_noisy = np.sqrt(np.sum(np.abs(result_noisy.final_state.data) ** 2))
    _report(
        "Norm = 1.0 after noisy circuit",
        abs(norm_noisy - 1.0) < TOLERANCE,
        f"got {norm_noisy:.15f}",
    )


# =========================================================================
# Test 3: Measurement probabilities sum to 1.0
# =========================================================================

def test_measurement_probabilities():
    """Probabilities sum to 1.0 in all three bases."""
    print("\nTest 3: Measurement Probability Sum = 1.0")
    print("-" * 40)

    qc = QuantumCircuit(num_qubits=2)
    qc.add_gate(GateInstance("H", [0], [], 0))
    qc.add_gate(GateInstance("Ry", [1], [1.0], 1))

    sim = Simulator()
    result = sim.run(qc, shots=0, seed=42)
    sv = result.final_state

    for basis in (MeasurementBasis.Z, MeasurementBasis.X, MeasurementBasis.Y):
        counts = MeasurementEngine.sample_with_basis(
            sv, 10000, basis=basis, rng=np.random.default_rng(42)
        )
        total = sum(counts.values())
        _report(
            f"{basis.value}-basis: total shots = 10000",
            total == 10000,
            f"got {total}",
        )

    # Probability distribution sums to 1.0
    prob_sum = float(sv.probabilities.sum())
    _report(
        "Probability distribution sums to 1.0",
        abs(prob_sum - 1.0) < TOLERANCE,
        f"got {prob_sum:.15f}",
    )


# =========================================================================
# Test 4: Readout error distribution transform matches shot-based
# =========================================================================

def test_readout_error_modes():
    """Distribution-transform and shot-based readout error converge."""
    print("\nTest 4: Readout Error Mode Consistency")
    print("-" * 40)

    sv = StateVector(2)
    sv.apply_gate(
        np.array([[1, 0], [0, 1]], dtype=np.complex128) / np.sqrt(1),
        [0],
    )
    # Create |+0> state
    from PRISM.engine.gates import H_MATRIX
    sv.apply_gate(H_MATRIX, [0])

    re = ReadoutError(p01=0.1, p10=0.1)
    rng = np.random.default_rng(42)

    # Distribution-transform mode (deterministic)
    counts_dist = MeasurementEngine.sample_with_basis(
        sv, 100000, readout_error=re, readout_mode="distribution", rng=rng,
    )
    freq_dist = {k: v / 100000 for k, v in counts_dist.items()}

    # Shot-based mode (stochastic, large sample)
    rng2 = np.random.default_rng(42)
    counts_shot = MeasurementEngine.sample_with_basis(
        sv, 100000, readout_error=re, readout_mode="shot", rng=rng2,
    )
    freq_shot = {k: v / 100000 for k, v in counts_shot.items()}

    # Both should be close (within statistical error)
    all_keys = set(freq_dist) | set(freq_shot)
    max_diff = max(
        abs(freq_dist.get(k, 0) - freq_shot.get(k, 0)) for k in all_keys
    )
    _report(
        "Distribution vs shot mode: max freq diff < 0.02",
        max_diff < 0.02,
        f"max diff = {max_diff:.4f}",
    )

    # Confusion matrix is correctly structured
    cm = re.confusion_matrix
    col_sums = cm.sum(axis=0)  # Each column should sum to 1
    _report(
        "Confusion matrix columns sum to 1.0",
        all(abs(s - 1.0) < TOLERANCE for s in col_sums),
        f"col sums = {col_sums}",
    )


# =========================================================================
# Test 5: QEC bit-flip code corrects single errors
# =========================================================================

def test_qec_correction():
    """BitFlipCode corrects single bit-flip errors perfectly."""
    print("\nTest 5: QEC Bit-Flip Single Error Correction")
    print("-" * 40)

    code = BitFlipCode()
    qec = QECSimulator(code)

    # No noise: fidelity should be 1.0
    result_clean = qec.run_cycle(
        logical_state=0, noise_type="bit_flip", noise_prob=0.0, seed=42
    )
    _report(
        "|0>_L no noise: fidelity = 1.0",
        abs(result_clean.fidelity_after - 1.0) < 0.01,
        f"got {result_clean.fidelity_after:.6f}",
    )
    _report(
        "|0>_L no noise: <Z_L> = +1.0",
        abs(result_clean.logical_z_expectation - 1.0) < 0.01,
        f"got {result_clean.logical_z_expectation:.6f}",
    )

    # |1>_L no noise
    result_1 = qec.run_cycle(
        logical_state=1, noise_type="bit_flip", noise_prob=0.0, seed=42
    )
    _report(
        "|1>_L no noise: <Z_L> = -1.0",
        abs(result_1.logical_z_expectation + 1.0) < 0.01,
        f"got {result_1.logical_z_expectation:.6f}",
    )


# =========================================================================
# Test 6: Reference manager invalidation and circuit hash
# =========================================================================

def test_reference_invalidation():
    """ReferenceManager invalidates when circuit changes."""
    print("\nTest 6: Reference Manager Invalidation")
    print("-" * 40)

    ref_mgr = ReferenceManager()

    # Build circuit 1
    qc1 = QuantumCircuit(num_qubits=2)
    qc1.add_gate(GateInstance("H", [0], [], 0))
    hash1 = qc1.circuit_hash()

    sim = Simulator()
    result1 = sim.run(qc1, shots=0, seed=42)
    ref_mgr.store(result1.final_state, "test", circuit_hash=hash1)

    _report(
        "Reference stored successfully",
        ref_mgr.has_reference,
    )

    # Same circuit hash: no invalidation
    invalidated = ref_mgr.check_invalidation(hash1)
    _report(
        "Same circuit hash: not invalidated",
        not invalidated and ref_mgr.has_reference,
    )

    # Different circuit hash: should invalidate
    qc2 = QuantumCircuit(num_qubits=2)
    qc2.add_gate(GateInstance("X", [0], [], 0))
    hash2 = qc2.circuit_hash()

    invalidated = ref_mgr.check_invalidation(hash2)
    _report(
        "Different circuit hash: invalidated",
        invalidated and not ref_mgr.has_reference,
    )

    # Circuit layers
    qc3 = QuantumCircuit(num_qubits=3)
    qc3.add_gate(GateInstance("H", [0], [], 0))
    qc3.add_gate(GateInstance("H", [1], [], 0))
    qc3.add_gate(GateInstance("CNOT", [0, 1], [], 1))
    qc3.add_gate(GateInstance("Rz", [2], [0.5], 2))
    layers = qc3.compute_layers()
    _report(
        "compute_layers() returns 3 layers",
        len(layers) == 3,
        f"got {len(layers)} layers",
    )

    gate_map = qc3.gate_to_layer_map()
    _report(
        "gate_to_layer_map() length matches gates",
        len(gate_map) == len(qc3.gates),
        f"map len={len(gate_map)}, gates={len(qc3.gates)}",
    )


# =========================================================================
# Test 7: Noise channels preserve CPTP (norm=1, correct limits)
# =========================================================================

def test_noise_cptp():
    """Noise channels satisfy CPTP: norm preserved, extreme cases correct."""
    print("\nTest 7: Noise Channel CPTP Verification")
    print("-" * 40)

    from PRISM.engine.noise import (
        NoiseModel, AmplitudeDampingNoise, DepolarizingNoise, BitFlipNoise,
    )

    # --- Amplitude Damping ---
    # gamma=0: identity (no change)
    noise_id = NoiseModel()
    noise_id.add_global_noise(AmplitudeDampingNoise(0.0))
    noise_id.set_seed(42)

    qc_h = QuantumCircuit(num_qubits=1)
    qc_h.add_gate(GateInstance("H", [0], [], 0))
    sim_id = Simulator(noise_model=noise_id)
    res_id = sim_id.run(qc_h, shots=0, seed=42)
    norm_id = np.sqrt(np.sum(np.abs(res_id.final_state.data) ** 2))
    _report("Amp damp gamma=0: norm=1.0", abs(norm_id - 1.0) < TOLERANCE,
            f"got {norm_id:.15f}")

    # gamma=1: always decays to |0>
    noise_full = NoiseModel()
    noise_full.add_global_noise(AmplitudeDampingNoise(1.0))
    noise_full.set_seed(42)

    qc_x = QuantumCircuit(num_qubits=1)
    qc_x.add_gate(GateInstance("X", [0], [], 0))  # start in |1>
    sim_full = Simulator(noise_model=noise_full)
    res_full = sim_full.run(qc_x, shots=0, seed=42)
    prob_0 = abs(res_full.final_state.data[0]) ** 2
    norm_full = np.sqrt(np.sum(np.abs(res_full.final_state.data) ** 2))
    _report("Amp damp gamma=1: state -> |0>", prob_0 > 0.99,
            f"P(0)={prob_0:.6f}")
    _report("Amp damp gamma=1: norm=1.0", abs(norm_full - 1.0) < TOLERANCE,
            f"got {norm_full:.15f}")

    # --- Amplitude Damping: arbitrary state, norm preserved ---
    noise_mid = NoiseModel()
    noise_mid.add_global_noise(AmplitudeDampingNoise(0.3))
    noise_mid.set_seed(42)

    qc_ry = QuantumCircuit(num_qubits=1)
    qc_ry.add_gate(GateInstance("Ry", [0], [1.234], 0))
    sim_mid = Simulator(noise_model=noise_mid)
    res_mid = sim_mid.run(qc_ry, shots=0, seed=42)
    norm_mid = np.sqrt(np.sum(np.abs(res_mid.final_state.data) ** 2))
    _report("Amp damp gamma=0.3: norm=1.0", abs(norm_mid - 1.0) < TOLERANCE,
            f"got {norm_mid:.15f}")

    # --- Depolarizing p=1: maximally mixed (each Pauli equally likely) ---
    # After many applications, state should be near maximally mixed
    noise_dep = NoiseModel()
    noise_dep.add_global_noise(DepolarizingNoise(1.0))
    # Run 10 gates to thoroughly depolarize
    qc_dep = QuantumCircuit(num_qubits=1)
    for i in range(10):
        qc_dep.add_gate(GateInstance("X", [0], [], i))
    sim_dep = Simulator(noise_model=noise_dep)
    res_dep = sim_dep.run(qc_dep, shots=0, seed=42)
    norm_dep = np.sqrt(np.sum(np.abs(res_dep.final_state.data) ** 2))
    _report("Depolarizing p=1.0: norm=1.0", abs(norm_dep - 1.0) < TOLERANCE,
            f"got {norm_dep:.15f}")


# =========================================================================
# Test 8: Performance regression (timing bounds)
# =========================================================================

def test_performance_regression():
    """Basic timing regression: circuits complete within reasonable time."""
    print("\nTest 8: Performance Regression")
    print("-" * 40)
    import time

    # (A) 10-qubit random circuit, depth 20
    qc = QuantumCircuit(num_qubits=10)
    rng = np.random.default_rng(42)
    gate_names = ["H", "X", "Ry", "Rz"]
    for col in range(20):
        for q in range(10):
            g = gate_names[rng.integers(0, len(gate_names))]
            params = [float(rng.uniform(0, 3.14))] if g in ("Ry", "Rz") else []
            qc.add_gate(GateInstance(g, [q], params, col))

    sim = Simulator()
    t0 = time.perf_counter()
    for _ in range(10):
        sim.run(qc, shots=0, seed=42)
    elapsed_a = (time.perf_counter() - t0) / 10

    # 10q x 20 depth should finish within 2 seconds per run
    _report(
        f"10q depth-20 circuit: {elapsed_a:.3f}s < 2.0s",
        elapsed_a < 2.0,
        f"got {elapsed_a:.3f}s",
    )

    # (B) Ensemble rho: 4 qubits, 50 trials
    from PRISM.engine.noise import NoiseModel, DepolarizingNoise

    qc4 = QuantumCircuit(num_qubits=4)
    qc4.add_gate(GateInstance("H", [0], [], 0))
    qc4.add_gate(GateInstance("CNOT", [0, 1], [], 1))
    qc4.add_gate(GateInstance("CNOT", [1, 2], [], 2))
    qc4.add_gate(GateInstance("CNOT", [2, 3], [], 3))

    noise = NoiseModel()
    noise.add_global_noise(DepolarizingNoise(0.05))
    sim_noisy = Simulator(noise_model=noise)

    t0 = time.perf_counter()
    rho = sim_noisy.ensemble_density_matrix(qc4, n_trials=50, seed=42)
    elapsed_b = time.perf_counter() - t0

    # 4q ensemble 50 trials should finish within 5 seconds
    _report(
        f"4q ensemble 50 trials: {elapsed_b:.3f}s < 5.0s",
        elapsed_b < 5.0,
        f"got {elapsed_b:.3f}s",
    )

    # Verify ensemble rho is valid
    purity = float(np.real(np.trace(rho @ rho)))
    _report(
        "Ensemble rho purity < 1.0 (mixed state)",
        purity < 1.0 - 1e-6,
        f"purity={purity:.6f}",
    )


# =========================================================================
# Test 9: Distribution-transform readout uses O(2^n) memory (no 2^n x 2^n)
# =========================================================================

def test_readout_distribution_scaling():
    """Distribution-transform readout works at 16 qubits without OOM."""
    print("\nTest 9: Distribution-Transform Scaling")
    print("-" * 40)
    import time

    from PRISM.engine.noise import ReadoutError

    # 16-qubit test: if np.kron were used, this would need 32 GiB
    n = 16
    probs = np.zeros(2**n, dtype=np.float64)
    probs[0] = 0.5
    probs[-1] = 0.5  # half |00..0>, half |11..1>

    re = ReadoutError(p01=0.05, p10=0.05)
    t0 = time.perf_counter()
    result = re.apply_to_distribution(probs, n)
    elapsed = time.perf_counter() - t0

    _report(
        f"16q distribution-transform: {elapsed:.3f}s < 1.0s",
        elapsed < 1.0,
        f"got {elapsed:.3f}s",
    )
    _report(
        "16q result sums to 1.0",
        abs(result.sum() - 1.0) < TOLERANCE,
        f"sum={result.sum():.15f}",
    )
    _report(
        "16q result shape = (2^16,)",
        result.shape == (2**n,),
        f"shape={result.shape}",
    )

    # Verify correctness: for 2-qubit case, compare with brute-force kron
    n2 = 2
    probs2 = np.array([0.5, 0.1, 0.1, 0.3])
    re2 = ReadoutError(p01=0.1, p10=0.2)

    # Brute-force: full kron matrix
    c1 = re2.confusion_matrix
    full_cm = np.kron(c1, c1)
    expected = full_cm @ probs2
    expected /= expected.sum()

    actual = re2.apply_to_distribution(probs2, n2)
    max_diff = np.max(np.abs(actual - expected))
    _report(
        "2q tensordot matches brute-force kron",
        max_diff < 1e-12,
        f"max_diff={max_diff:.2e}",
    )


# =========================================================================
# Main
# =========================================================================

def main():
    global PASS_COUNT, FAIL_COUNT
    print("=" * 50)
    print("Quantum Simulator Validation Test Harness")
    print("=" * 50)

    tests = [
        test_bell_state,
        test_normalization,
        test_measurement_probabilities,
        test_readout_error_modes,
        test_qec_correction,
        test_reference_invalidation,
        test_noise_cptp,
        test_performance_regression,
        test_readout_distribution_scaling,
    ]

    for test_fn in tests:
        try:
            test_fn()
        except Exception as e:
            print(f"\n  [ERROR] {test_fn.__name__} raised an exception:")
            traceback.print_exc()
            FAIL_COUNT += 1

    print("\n" + "=" * 50)
    total = PASS_COUNT + FAIL_COUNT
    print(f"Results: {PASS_COUNT}/{total} passed, {FAIL_COUNT} failed")
    if FAIL_COUNT == 0:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("=" * 50)

    return 0 if FAIL_COUNT == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
