# PRISM

> **P**er-gate **R**eproducible **I**nference for **S**tochastic **M**echanics

A research-grade quantum circuit simulator with **statistically rigorous noise attribution**, an interactive 13-panel GUI, and a fully reproducible experiment-replay pipeline.

Built entirely with **PyQt6 + NumPy + Matplotlib** -- no Qiskit, no Cirq, no additional dependencies.

```bash
./PRISM.sh                  # cross-platform launcher (preferred)
python -m PRISM             # equivalent direct invocation
```

The launcher activates a project-local virtualenv (`.venv`, `venv`, or `.env`) if one is present and resolves a Python interpreter that has PyQt6 installed.

---

## What This Simulator Does

Build quantum circuits by dragging and dropping gates, run simulations with or without noise, and instantly visualize results across **13 interactive panels**. Everything from basic state vectors to advanced quantum error correction is handled within a single application.

**Core capabilities:**

- **1-16 qubits** with efficient tensor contraction (not matrix multiplication)
- **Noiseless and noisy simulation** with 4 noise channels + readout error
- **13 visualization panels** covering state analysis, debugging, optimization, and QEC
- **Statistical noise attribution** with bootstrap CIs, two-sided p-values, and Benjamini-Hochberg FDR correction (the PRISM contribution)
- **Experiment automation** via CLI scripts with full seed-based reproducibility
- **Live Bridge API** for external program control via TCP

---

## Requirements

```bash
pip install PyQt6 numpy matplotlib
```

- Python 3.10+
- Optional: `psutil` for CPU % monitoring in the Resource Monitor panel

---

## GUI Overview
<img width="1919" height="1006" alt="image" src="https://github.com/user-attachments/assets/a47f16c9-fea4-4494-b7de-68ae5ce8cd3a" />

---

## Building a Circuit

1. **Drag gates** from the palette on the left and **drop** onto the circuit editor.
2. **Qubit count**: Adjust with the spinbox in the toolbar (1-16).
3. **Multi-qubit gates** (CNOT, CZ, SWAP, Toffoli): A dialog prompts target/control qubits.
4. **Edit parameters**: Double-click Rx, Ry, Rz, or U3 to adjust rotation angles.
5. **Initial states**: Click qubit labels to toggle between |0> and |1>.
6. **File > Save / Load**: Circuits save as `.qsim` or `.json` files.

---

## Implemented Features

### Simulation Engine

| Feature | Description |
|---------|-------------|
| **State vector simulation** | Pure state evolution via `np.tensordot` tensor contraction. O(2^n * 4^k) per gate, not O(4^n). |
| **4 noise channels** | Bit-flip, Phase-flip, Depolarizing, Amplitude Damping. Applied per-gate via stochastic Kraus operator selection. |
| **Readout error** | Per-bit confusion matrix. Two modes: **shot-based** (stochastic bitstring corruption) and **distribution-transform** (reshapes the probability vector to a (2,2,...,2) tensor and applies the 2x2 confusion matrix per qubit axis via `np.tensordot` -- O(2^n) memory, no explicit 2^n x 2^n matrix construction). |
| **Z / X / Y basis measurement** | Z (computational), X (apply H then measure), Y (apply S-dagger + H then measure). |
| **Ensemble density matrix** | For noisy circuits, a single run always produces a pure state. The simulator computes a Monte Carlo estimate of the mixed-state density matrix by averaging over N stochastic trials: rho = (1/N) sum \|psi_i><psi_i\| (converges in expectation to the CPTP channel output as trials increase). Configurable trial count (1-500) in the Density Matrix panel. |
| **Reference state management** | All fidelity calculations use a centralized ReferenceManager. The noiseless simulation result is automatically stored as the baseline. **State reference** (noiseless \|psi>) is basis-independent and invalidated only when the circuit structure changes (via circuit hash). **Measurement references** (probability distributions) are lazily computed and cached per basis (Z/X/Y), so a basis change triggers a one-time recomputation, not a full invalidation. State fidelity (\|<psi\|phi>\|^2) is used for pure-state comparisons; Uhlmann fidelity is used when comparing reduced density matrices. |
| **Seed management** | All operations (simulation, optimization, QEC, scripts) accept a seed for full reproducibility. |

### Noise Attribution (PRISM core contribution)

| Feature | Description |
|---------|-------------|
| **Per-gate fidelity gap tracking** | For each gate column i: gap_i = 1 - F(ideal_i, noisy_i), contribution_i = gap_i - gap_{i-1}. |
| **Attribution percentage** | Each column's share of total fidelity loss. Negative contributions (where noise coincidentally recovers fidelity) are flagged as "recovery" and clamped to 0% for normalization. If total positive loss < epsilon, attribution is reported as 0% with a "no measurable loss" indicator. |
| **Bootstrap CIs (Phase 1A)** | `compute_noise_attribution_with_statistics()` returns row-resampling bootstrap confidence intervals for both per-column delta_F and attribution percentages. Joint resampling preserves the percentage's denominator coupling so per-column CIs are not biased toward independence. |
| **Two-sided p-values** | Each column's contribution is tested against H0: mean(delta_F) = 0. p-values are floored at 1 / n_bootstrap to avoid the meaningless "p = 0" artifact of finite resampling. |
| **Benjamini-Hochberg FDR correction** | Family-wise q-values across all columns. `column_significant` flags columns whose contribution survives multiple-comparison correction at the configured FDR level. |
| **Recovery rate analysis** | Per-column empirical P(delta_F < 0) with bootstrap CI. Distinguishes columns with truly small contribution from those whose mean is biased by occasional recovery events. |
| **Per-qubit breakdown** | Reduced density matrix fidelity per qubit per column. |

### Entanglement Event Detection

| Feature | Description |
|---------|-------------|
| **Pairwise mutual information tracking** | I(A:B) = S(A) + S(B) - S(AB) computed for every qubit pair at every simulation step. |
| **Hysteresis thresholds** | Separate thresholds for creation (epsilon_on) and destruction (epsilon_off < epsilon_on) to prevent spurious events from numerical noise. |
| **Persistence filter** | An event must persist for N consecutive steps before being emitted. Prevents flickering from stochastic fluctuations. |
| **Event types** | Creation, Disentanglement, Increase, Decrease -- with green/red triangle markers on the plot. |

### Circuit Layer Definition

All modules (Optimizer, Debugger, Entropy) share a single layer definition:

- A **layer** = all gates at the same column index in the circuit editor
- `circuit.compute_layers()` returns gate indices grouped by column
- `circuit.gate_to_layer_map()` maps each gate to its layer index
- `circuit.circuit_hash()` produces a structure hash for invalidation checks

### Quantum Error Correction

| Feature | Description |
|---------|-------------|
| **3 codes** | Bit-Flip [3,1,1], Phase-Flip [3,1,1], Steane [[7,1,3]] |
| **Logical Z operator** | Z_L = tensor product of Z on data qubits. \|0>_L gives <Z_L> = +1, \|1>_L gives <Z_L> = -1. |
| **3 logical error metrics** | (1) Fidelity-based: success if F > 0.5. (2) Z_L sign-based: success if <Z_L> has correct sign. (3) Projection-based: logical error = 1 - F(corrected, ideal codeword). |
| **Threshold sweep** | Plots all 3 metrics vs physical error rate on the same chart. |

### Parameterized Optimization

| Feature | Description |
|---------|-------------|
| **Auto parameter detection** | Finds all tunable parameters (Rx, Ry, Rz, U3 angles) in the circuit. |
| **Gradient methods** | Parameter-shift rule (exact for Rx/Ry/Rz) and finite difference (general). |
| **Adam optimizer** | Pure NumPy implementation with convergence detection. |
| **Barren plateau analysis** | Layer-wise gradient variance heatmap + depth scaling plot. Uses the shared layer definition from `circuit.gate_to_layer_map()`. |

---

## Visualization Panels (13 Panels)

### Panels 1-8: State Analysis

| # | Panel | What It Shows |
|---|-------|---------------|
| 1 | **State Vector** | Complex amplitude, probability, and phase for every basis state. |
| 2 | **Bloch Sphere** | 3D Bloch sphere per qubit. Step mode shows trajectory tracking. |
| 3 | **Histogram** | Measurement distribution bar chart. Z/X/Y basis selector. |
| 4 | **Density Matrix** | Heatmap of rho (Real, Imaginary, Magnitude views). **Ensemble mode** button computes a Monte Carlo estimate of mixed-state rho under noise via multi-trial averaging. Shows purity and Von Neumann entropy to characterize mixed states. |
| 5 | **Entanglement** | Pairwise mutual information, concurrence, separability indicator. |
| 6 | **Entropy** | 4 modes: Total system, Per-qubit, Bipartite, **Entanglement Events** (pairwise MI timeline with creation/destruction markers and hysteresis). |
| 7 | **Fidelity** | Noise sweep: fidelity vs noise probability. Configurable trials (1-500). Uses centralized reference state. |
| 8 | **Analysis** | Dashboard: purity, entropy, Pauli expectations, entanglement metrics, fidelity vs reference. |

### Panels 9-13: Advanced Tools

| # | Panel | What It Shows |
|---|-------|---------------|
| 9 | **Debugger** | Step-through execution with 3 sub-tabs: **State Inspector** (amplitude table + ideal vs actual bar chart), **Noise Heatmap** (per-qubit fidelity drop + attribution % overlay + recovery flags), **Error Trace** (fidelity + entropy curves with breakpoints). |
| 10 | **Comparison** | Side-by-side circuit comparison: histogram overlay, metrics (fidelity, TVD, KL divergence), resource bar chart, JSON export. |
| 11 | **Optimizer** | VQE/QAOA optimization with 3 sub-tabs: **Convergence** (cost curve), **Parameters** (value evolution), **Barren Plateau** (layer-wise variance heatmap + depth scaling plot). |
| 12 | **QEC** | Quantum error correction with 3 sub-tabs: **Code Layout** (qubit diagram), **Syndrome** (detection + correction + Z_L indicator), **Threshold** (4 curves: fidelity-based, Z_L sign-based, projection-based error rate, and average fidelity). |
| 13 | **Resources** | Real-time CPU/memory monitoring, simulation timing, peak memory, thread count, 2-min rolling history graph, simulator comparison table. |

---

## Noise Configuration

Access via **Simulation > Noise Configuration**.

### Gate Noise Channels

| Channel | Description | Parameter |
|---------|-------------|-----------|
| Bit Flip | X applied with probability p | p: [0, 1] |
| Phase Flip | Z applied with probability p | p: [0, 1] |
| Depolarizing | Random X/Y/Z with probability p/3 each | p: [0, 1] |
| Amplitude Damping | Energy relaxation (T1 decay) | gamma: [0, 1] |

### Readout Error

| Parameter | Description |
|-----------|-------------|
| P(1\|0) | Probability of reading 1 when true state is 0 |
| P(0\|1) | Probability of reading 0 when true state is 1 |

Two readout error modes:

- **Shot-based** (default): Each measurement bitstring is independently corrupted per bit. Realistic for hardware simulation.
- **Distribution-transform**: The probability vector is reshaped to (2,2,...,2) and each qubit axis is contracted with the per-bit 2x2 confusion matrix via `np.tensordot`. No explicit 2^n x 2^n matrix is ever constructed (O(2^n) memory). Deterministic, no sampling noise. Useful for benchmarks and paper figures.

---

## Live Bridge API

External programs can control the simulator via TCP (port 9876).

**Start**: Simulation > Start Bridge Server

**Protocol**: JSON over TCP, newline-delimited.

```json
{"command": "run", "shots": 1024}
{"command": "get_state"}
{"command": "add_gate", "gate": "H", "qubits": [0], "column": 0}
{"command": "set_noise", "type": "depolarizing", "probability": 0.01}
{"command": "sweep_parameter", "gate_index": 0, "param_index": 0, "values": [0, 0.5, 1.0]}
```

---

## CLI Experiment Scripts

Headless experiment automation for reproducible research (no GUI required):

```bash
# Noise sweep: fidelity vs noise probability
python scripts/noise_sweep.py --circuit bell --noise depolarizing --seed 42 --output results.json

# VQE optimization benchmark
python scripts/vqe_benchmark.py --qubits 2 --hamiltonian heisenberg --iters 50 --seed 42

# QEC threshold analysis
python scripts/qec_threshold.py --codes bit_flip,steane --trials 100 --seed 42
```

All scripts output JSON and support `--seed` for full reproducibility.

---

## Resource Efficiency

| Simulator | Method | Max Qubits (8 GiB) | Notes |
|-----------|--------|---------------------|-------|
| **This Sim** | State Vector | **28** | Pure NumPy, instant setup |
| Qiskit SV | State Vector | 28 | Requires C++ Aer backend |
| QuTiP / DM | Density Matrix | 14 | 2^2n scaling |
| Qiskit MPS | Tensor Network | 50+ | Depends on entanglement |

Memory formula: 2^n * 16 bytes (complex128). At 28 qubits, the state vector alone is 4 GiB (2^28 * 16 = 4,294,967,296 bytes). Our advantage over density matrix simulators: **2^n times less memory**.

---

## Research Applications

| Experiment | Panels / Scripts | Output |
|------------|-----------------|--------|
| Noise resilience | Fidelity + Debugger + noise_sweep.py | Fidelity curves, per-gate attribution heatmap |
| VQE/QAOA | Optimizer + vqe_benchmark.py | Cost convergence, parameter evolution |
| Barren plateau | Optimizer (Barren tab) | Layer-wise variance heatmap + depth scaling |
| QEC threshold | QEC + qec_threshold.py | 3 logical error metrics vs physical rate |
| Entanglement dynamics | Entropy (Events mode) | MI timeline with creation/destruction detection |
| Mixed-state analysis | Density Matrix (Ensemble mode) | Monte Carlo mixed-state rho under noise |
| Readout characterization | Histogram + Noise config | Basis-dependent measurement with readout error |
| Algorithm comparison | Comparison panel | Side-by-side fidelity, TVD, KL, resources |

---

## Architecture

```
PRISM/
  __main__.py             # Entry point (run via `python -m PRISM`)
  engine/                 # Pure NumPy engine (no GUI dependencies)
    circuit.py              # QuantumCircuit, GateInstance, compute_layers(), circuit_hash()
    simulator.py            # Simulator (run, step-by-step, ensemble_density_matrix)
    state_vector.py         # StateVector with tensor contraction
    gates.py                # Gate matrices (H, X, Y, Z, Rx, Ry, Rz, U3, CNOT, ...)
    noise.py                # NoiseModel, 4 channels, ReadoutError (shot + distribution)
    measurement.py          # MeasurementEngine, MeasurementBasis (Z/X/Y)
    analysis.py             # StateAnalysis, EntanglementEventDetector (hysteresis)
    debugger.py             # CircuitDebugger, NoiseAttribution + AttributionStatistics
    statistics.py           # Bootstrap CI / p-values / Benjamini-Hochberg FDR
    comparison.py           # CircuitComparator, ComparisonResult
    optimizer.py            # CircuitOptimizer, BarrenPlateauAnalysis (layer-wise)
    qec.py                  # 3 QEC codes, QECSimulator (3 logical error metrics)
    reference.py            # ReferenceManager (auto-invalidation via circuit hash)
  gui/
    main_window.py          # Main window (13 tabs, menus, toolbar)
    circuit_editor/         # QGraphicsScene-based circuit editor
    panels/                 # 13 visualization panels
    dialogs/                # Configuration dialogs
    themes/                 # Dark / Light theme
    commands/               # Undo/redo
  controller/               # MVC controllers
  core/                     # Serialization, config, SeedManager
  bridge/                   # Live Bridge API server (TCP, port 9876)
scripts/                    # CLI experiment automation
  noise_sweep.py            # Noise probability sweep
  vqe_benchmark.py          # VQE optimization benchmark
  qec_threshold.py          # Multi-code QEC threshold analysis
tests/                      # pytest suite (Phase-1B reproducibility infra)
  conftest.py               # Shared fixtures (RNGs, circuits, noise models)
  test_statistics.py        # 29 unit tests for engine.statistics
  test_attribution_statistics.py  # 13 integration tests for the bootstrap-aware attribution
PRISM.sh                    # Cross-platform shell launcher
test_validation.py          # Legacy 33-assertion harness (kept for now; will migrate to tests/)
```

---

## Validation

PRISM ships with two complementary test layers.

### Legacy validation harness

`test_validation.py` (33 assertions) verifies fundamental quantum-mechanical identities that any correct simulator must satisfy:

1. **Bell state correctness** -- amplitudes, mutual information, entanglement entropy
2. **State normalization** -- norm = 1.0 after gates and noise
3. **Measurement probabilities** -- sum = 1.0 in Z/X/Y bases
4. **Readout error consistency** -- distribution-transform and shot-based modes converge
5. **QEC correction** -- BitFlipCode corrects single errors, Z_L expectation correct
6. **Reference invalidation** -- circuit hash change triggers auto-clear, layer API consistency
7. **Noise channel CPTP** -- amplitude damping (gamma=0/0.3/1.0 norm preservation, gamma=1 decay to |0>), depolarizing (p=1.0 norm)
8. **Performance regression** -- 10q depth-20 circuit < 2s, 4q ensemble 50 trials < 5s, ensemble purity < 1.0
9. **Distribution-transform scaling** -- 16q readout in O(2^n) memory, correctness verified against brute-force kron for 2q

```bash
python test_validation.py
# Results: 33/33 passed, ALL TESTS PASSED
```

### Pytest suite (Phase-1B)

The `tests/` directory hosts a pytest suite covering the statistical attribution layer (42 tests). Install dev dependencies and run:

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
# Results: 42 passed in ~1.3s
```

Coverage of the bootstrap statistics module is **95%**; the legacy harness is gradually being migrated into this directory.

---

## Roadmap

PRISM is being developed toward an arXiv preprint. The work is organized into three phases:

- **Phase 1 -- Statistical foundation + reproducibility** (in progress)
  - 1A: Bootstrap CI / p-values / FDR for noise attribution -- **shipped** (`engine/statistics.py`, `compute_noise_attribution_with_statistics()`)
  - 1B: pytest migration, GitHub Actions CI, headless replay CLI, paper directory
- **Phase 2 -- Pauli twirling + QEC three-metric agreement analysis**
- **Phase 3 (optional)** -- Classical shadows / Numba hot-path / Mirror RB (one of)

The North-Star deliverable is a paper that publishes (i) the statistically rigorous per-column noise attribution methodology, (ii) the quantitative effect of Pauli twirling on attribution profiles, and (iii) PRISM itself as an integrated, fully reproducible quantum-circuit research workbench.

---

## License

This project is developed for academic research purposes.
