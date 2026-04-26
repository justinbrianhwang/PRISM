"""Quantum Circuit Debugger -- step-through execution with state inspection.

Provides breakpoint support, forward/backward stepping, noise impact analysis,
and state diff between any two execution points.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .state_vector import StateVector
from .circuit import QuantumCircuit, GateInstance
from .gate_registry import GateRegistry
from .gates import GateType
from .analysis import StateAnalysis


@dataclass
class DebugSnapshot:
    """State captured at a single execution point."""

    column_index: int  # -1 for initial state
    state: StateVector  # deep copy of state at this point
    ideal_state: StateVector | None  # noiseless state (None if no noise)
    gate_labels: list[str]  # gates applied in this column
    fidelity: float  # fidelity vs ideal (1.0 if no noise)
    cumulative_fidelity: float  # fidelity from initial state to here
    entropy: float  # von Neumann entropy at this point


@dataclass
class NoiseImpactResult:
    """Noise impact for a single gate column."""

    column_index: int
    gate_labels: list[str]
    fidelity_before: float
    fidelity_after: float
    fidelity_drop: float
    entropy_before: float
    entropy_after: float
    entropy_change: float
    per_qubit_fidelity: list[float]  # reduced density matrix fidelity per qubit
    mean_delta_fidelity: float = 0.0  # mean fidelity drop across trials
    std_delta_fidelity: float = 0.0   # std of fidelity drop across trials


@dataclass
class NoiseAttribution:
    """Per-gate noise attribution analysis.

    Quantifies how much each gate column contributes to total fidelity loss:
        delta_F_i = F(ref, psi_{i-1}) - F(ref, psi_i)

    Negative contributions (fidelity recovery) are preserved in raw values
    but clamped to zero for percentage attribution. Recovery columns are
    labeled with is_recovery flags.
    """

    delta_fidelity: list[float]           # per-column mean delta_F (may be negative)
    delta_fidelity_std: list[float]       # per-column std delta_F
    total_fidelity_loss: float            # F(ref, initial) - F(ref, final)
    column_attribution_pct: list[float]   # per-column % of total loss (clamped >= 0)
    per_qubit_attribution: list[list[float]]  # [col][qubit] fidelity contribution
    gate_labels: list[list[str]]          # [col] -> list of gate labels
    is_recovery: list[bool] = field(default_factory=list)  # True if delta_F < 0
    no_measurable_loss: bool = False      # True if total positive loss < epsilon


class CircuitDebugger:
    """Debugger that caches per-column states for forward/backward stepping.

    Usage::

        dbg = CircuitDebugger()
        dbg.run_full_debug(circuit, noise_model, seed=42)
        snap = dbg.current_snapshot        # initial state
        snap = dbg.step_forward()           # after column 0
        snap = dbg.step_backward()          # back to initial
        dbg.add_breakpoint(3)
        snap = dbg.run_to_breakpoint()      # jump to column 3
    """

    def __init__(self):
        self._snapshots: list[DebugSnapshot] = []
        self._position: int = 0  # index into _snapshots
        self._breakpoints: set[int] = set()  # column indices
        self._registry = GateRegistry.instance()

    # ---- Public API -------------------------------------------------------

    def run_full_debug(
        self,
        circuit: QuantumCircuit,
        noise_model=None,
        seed: int | None = None,
    ) -> list[DebugSnapshot]:
        """Execute the circuit and cache state after every column.

        Args:
            circuit: The quantum circuit to debug.
            noise_model: Optional NoiseModel for noisy simulation.
            seed: Reproducibility seed.

        Returns:
            List of DebugSnapshot, starting with the initial state.
        """
        rng = np.random.default_rng(seed)
        self._snapshots.clear()
        self._position = 0

        # Create initial states
        state = StateVector.from_initial_states(circuit.initial_states)
        ideal_state = StateVector.from_initial_states(circuit.initial_states)

        # Snapshot for initial state
        self._snapshots.append(DebugSnapshot(
            column_index=-1,
            state=state.copy(),
            ideal_state=ideal_state.copy() if noise_model else None,
            gate_labels=[],
            fidelity=1.0,
            cumulative_fidelity=1.0,
            entropy=StateAnalysis.von_neumann_entropy(state),
        ))

        # Execute column by column
        ordered = circuit.get_ordered_gates()
        for col_idx, column_gates in enumerate(ordered):
            labels = []
            for gate_inst in column_gates:
                gate_def = self._registry.get(gate_inst.gate_name)
                if gate_def.gate_type in (GateType.MEASUREMENT, GateType.BARRIER):
                    continue

                # Apply gate to both ideal and actual
                matrix = gate_def.matrix_func(*gate_inst.params)
                ideal_state.apply_gate(matrix, gate_inst.target_qubits)
                state.apply_gate(matrix, gate_inst.target_qubits)

                # Apply noise to actual only
                if noise_model is not None:
                    noise_model.apply(state, gate_inst)

                qubits_str = ",".join(str(q) for q in gate_inst.target_qubits)
                labels.append(f"{gate_inst.gate_name}({qubits_str})")

            # Compute fidelity
            if noise_model is not None:
                fid = StateAnalysis.state_fidelity(
                    ideal_state.data, state.data
                )
            else:
                fid = 1.0

            # Cumulative fidelity from initial
            initial_ideal = self._snapshots[0].state
            cum_fid = StateAnalysis.state_fidelity(
                initial_ideal.data, state.data
            ) if noise_model else 1.0

            self._snapshots.append(DebugSnapshot(
                column_index=col_idx,
                state=state.copy(),
                ideal_state=ideal_state.copy() if noise_model else None,
                gate_labels=labels,
                fidelity=fid,
                cumulative_fidelity=cum_fid,
                entropy=StateAnalysis.von_neumann_entropy(state),
            ))

        return self._snapshots

    @property
    def snapshots(self) -> list[DebugSnapshot]:
        return self._snapshots

    @property
    def position(self) -> int:
        return self._position

    @position.setter
    def position(self, value: int) -> None:
        if self._snapshots:
            self._position = max(0, min(value, len(self._snapshots) - 1))

    @property
    def current_snapshot(self) -> DebugSnapshot | None:
        if not self._snapshots:
            return None
        return self._snapshots[self._position]

    @property
    def num_steps(self) -> int:
        return len(self._snapshots)

    def step_forward(self) -> DebugSnapshot | None:
        """Advance one step. Returns new snapshot or None if at end."""
        if not self._snapshots or self._position >= len(self._snapshots) - 1:
            return None
        self._position += 1
        return self._snapshots[self._position]

    def step_backward(self) -> DebugSnapshot | None:
        """Go back one step. Returns new snapshot or None if at start."""
        if not self._snapshots or self._position <= 0:
            return None
        self._position -= 1
        return self._snapshots[self._position]

    def goto_step(self, step: int) -> DebugSnapshot | None:
        """Jump to a specific step index."""
        if not self._snapshots:
            return None
        self._position = max(0, min(step, len(self._snapshots) - 1))
        return self._snapshots[self._position]

    # ---- Breakpoints ------------------------------------------------------

    def add_breakpoint(self, column: int) -> None:
        self._breakpoints.add(column)

    def remove_breakpoint(self, column: int) -> None:
        self._breakpoints.discard(column)

    def toggle_breakpoint(self, column: int) -> bool:
        """Toggle breakpoint at column. Returns True if now set."""
        if column in self._breakpoints:
            self._breakpoints.discard(column)
            return False
        self._breakpoints.add(column)
        return True

    @property
    def breakpoints(self) -> set[int]:
        return self._breakpoints

    def clear_breakpoints(self) -> None:
        self._breakpoints.clear()

    def run_to_breakpoint(self) -> DebugSnapshot | None:
        """Run forward until a breakpoint is hit or end is reached."""
        if not self._snapshots:
            return None

        start = self._position + 1
        for i in range(start, len(self._snapshots)):
            snap = self._snapshots[i]
            if snap.column_index in self._breakpoints:
                self._position = i
                return snap

        # No breakpoint found, go to end
        self._position = len(self._snapshots) - 1
        return self._snapshots[self._position]

    # ---- Noise impact analysis --------------------------------------------

    def compute_noise_impact(
        self,
        circuit: QuantumCircuit,
        noise_model,
        n_trials: int = 50,
        seed: int | None = None,
    ) -> list[NoiseImpactResult]:
        """Compute per-column fidelity drop due to noise.

        Runs multiple trials and averages the fidelity to get reliable results.

        Args:
            circuit: Circuit to analyze.
            noise_model: NoiseModel to use.
            n_trials: Number of stochastic trials to average.
            seed: Base seed for reproducibility.

        Returns:
            List of NoiseImpactResult, one per gate column.
        """
        if noise_model is None:
            return []

        base_rng = np.random.default_rng(seed)
        ordered = circuit.get_ordered_gates()
        num_cols = len(ordered)

        # Accumulate per-column metrics across trials
        fid_before_acc = np.zeros(num_cols)
        fid_after_acc = np.zeros(num_cols)
        ent_before_acc = np.zeros(num_cols)
        ent_after_acc = np.zeros(num_cols)
        per_qubit_fid_acc = [np.zeros(circuit.num_qubits) for _ in range(num_cols)]
        # Per-trial fidelity drops for std computation
        fid_drop_trials = np.zeros((n_trials, num_cols))

        for trial in range(n_trials):
            trial_seed = int(base_rng.integers(0, 2**63))
            noise_model.set_seed(trial_seed)

            ideal = StateVector.from_initial_states(circuit.initial_states)
            noisy = StateVector.from_initial_states(circuit.initial_states)

            for col_idx, column_gates in enumerate(ordered):
                # Fidelity before this column's gates
                fb = StateAnalysis.state_fidelity(ideal.data, noisy.data)
                fid_before_acc[col_idx] += fb
                ent_before_acc[col_idx] += StateAnalysis.von_neumann_entropy(noisy)

                for gate_inst in column_gates:
                    gate_def = self._registry.get(gate_inst.gate_name)
                    if gate_def.gate_type in (GateType.MEASUREMENT, GateType.BARRIER):
                        continue
                    matrix = gate_def.matrix_func(*gate_inst.params)
                    ideal.apply_gate(matrix, gate_inst.target_qubits)
                    noisy.apply_gate(matrix, gate_inst.target_qubits)
                    noise_model.apply(noisy, gate_inst)

                # Fidelity after this column's gates
                fa = StateAnalysis.state_fidelity(ideal.data, noisy.data)
                fid_after_acc[col_idx] += fa
                ent_after_acc[col_idx] += StateAnalysis.von_neumann_entropy(noisy)
                fid_drop_trials[trial, col_idx] = fb - fa

                # Per-qubit fidelity (reduced density matrix)
                for q in range(circuit.num_qubits):
                    rho_ideal = ideal.get_reduced_density_matrix(q)
                    rho_noisy = noisy.get_reduced_density_matrix(q)
                    pq_fid = StateAnalysis.density_fidelity(rho_ideal, rho_noisy)
                    per_qubit_fid_acc[col_idx][q] += pq_fid

        # Average
        results = []
        for col_idx, column_gates in enumerate(ordered):
            labels = []
            for g in column_gates:
                gd = self._registry.get(g.gate_name)
                if gd.gate_type not in (GateType.MEASUREMENT, GateType.BARRIER):
                    qstr = ",".join(str(q) for q in g.target_qubits)
                    labels.append(f"{g.gate_name}({qstr})")

            fb = fid_before_acc[col_idx] / n_trials
            fa = fid_after_acc[col_idx] / n_trials
            eb = ent_before_acc[col_idx] / n_trials
            ea = ent_after_acc[col_idx] / n_trials
            pqf = (per_qubit_fid_acc[col_idx] / n_trials).tolist()

            results.append(NoiseImpactResult(
                column_index=col_idx,
                gate_labels=labels,
                fidelity_before=fb,
                fidelity_after=fa,
                fidelity_drop=fb - fa,
                entropy_before=eb,
                entropy_after=ea,
                entropy_change=ea - eb,
                per_qubit_fidelity=pqf,
                mean_delta_fidelity=float(np.mean(fid_drop_trials[:, col_idx])),
                std_delta_fidelity=float(np.std(fid_drop_trials[:, col_idx])),
            ))

        return results

    # ---- Noise attribution ------------------------------------------------

    def compute_noise_attribution(
        self,
        circuit: QuantumCircuit,
        noise_model,
        reference_state: StateVector | None = None,
        n_trials: int = 50,
        seed: int | None = None,
    ) -> NoiseAttribution:
        """Compute per-gate noise attribution by tracking the fidelity gap
        between ideal and noisy trajectories at each column.

        noise_contrib_i = gap_i - gap_{i-1}
        where gap_i = 1 - F(ideal_i, noisy_i)

        This isolates each column's noise contribution from gate progress.

        Args:
            circuit: Circuit to analyze.
            noise_model: NoiseModel to use.
            reference_state: Optional external reference. If None, the ideal
                trajectory is used as the reference at each step.
            n_trials: Number of stochastic trials to average.
            seed: Base seed for reproducibility.

        Returns:
            NoiseAttribution with per-column fidelity attribution.
        """
        base_rng = np.random.default_rng(seed)
        ordered = circuit.get_ordered_gates()
        num_cols = len(ordered)
        n_qubits = circuit.num_qubits

        # Collect per-trial, per-column noise contribution
        noise_contrib_trials = np.zeros((n_trials, num_cols))
        # Per-qubit attribution accumulator
        pq_attr_acc = np.zeros((num_cols, n_qubits))

        # Build gate labels
        all_labels: list[list[str]] = []
        for column_gates in ordered:
            labels = []
            for g in column_gates:
                gd = self._registry.get(g.gate_name)
                if gd.gate_type not in (GateType.MEASUREMENT, GateType.BARRIER):
                    qstr = ",".join(str(q) for q in g.target_qubits)
                    labels.append(f"{g.gate_name}({qstr})")
            all_labels.append(labels)

        for trial in range(n_trials):
            trial_seed = int(base_rng.integers(0, 2**63))
            noise_model.set_seed(trial_seed)

            ideal = StateVector.from_initial_states(circuit.initial_states)
            noisy = StateVector.from_initial_states(circuit.initial_states)
            prev_gap = 0.0  # gap before any gates (ideal == noisy)

            for col_idx, column_gates in enumerate(ordered):
                for gate_inst in column_gates:
                    gate_def = self._registry.get(gate_inst.gate_name)
                    if gate_def.gate_type in (GateType.MEASUREMENT, GateType.BARRIER):
                        continue
                    matrix = gate_def.matrix_func(*gate_inst.params)
                    ideal.apply_gate(matrix, gate_inst.target_qubits)
                    noisy.apply_gate(matrix, gate_inst.target_qubits)
                    noise_model.apply(noisy, gate_inst)

                # Fidelity gap after this column
                fid = StateAnalysis.state_fidelity(ideal.data, noisy.data)
                gap = 1.0 - fid
                noise_contrib_trials[trial, col_idx] = gap - prev_gap
                prev_gap = gap

                # Per-qubit attribution via reduced density matrices
                for q in range(n_qubits):
                    rho_ideal = ideal.get_reduced_density_matrix(q)
                    rho_noisy = noisy.get_reduced_density_matrix(q)
                    pq_attr_acc[col_idx, q] += (
                        1.0 - StateAnalysis.density_fidelity(rho_ideal, rho_noisy)
                    )

        # Statistics
        mean_contrib = np.mean(noise_contrib_trials, axis=0).tolist()
        std_contrib = np.std(noise_contrib_trials, axis=0).tolist()

        # Total fidelity loss = final gap = sum of contributions
        total_loss = float(np.sum(mean_contrib))

        # Identify recovery columns (negative delta_F = fidelity improvement)
        is_recovery = [d < -1e-12 for d in mean_contrib]

        # Attribution percentage: clamp negatives to 0, normalize positive-only
        positive_sum = sum(max(0.0, d) for d in mean_contrib)
        no_loss = positive_sum <= 1e-12
        if not no_loss:
            attr_pct = [max(0.0, d) / positive_sum * 100.0 for d in mean_contrib]
        else:
            attr_pct = [0.0] * num_cols

        # Per-qubit attribution (averaged over trials)
        pq_attr = (pq_attr_acc / n_trials).tolist()

        return NoiseAttribution(
            delta_fidelity=mean_contrib,
            delta_fidelity_std=std_contrib,
            total_fidelity_loss=total_loss,
            column_attribution_pct=attr_pct,
            per_qubit_attribution=pq_attr,
            gate_labels=all_labels,
            is_recovery=is_recovery,
            no_measurable_loss=no_loss,
        )

    # ---- State diff -------------------------------------------------------

    @staticmethod
    def compute_state_diff(
        snap_a: DebugSnapshot,
        snap_b: DebugSnapshot,
    ) -> dict:
        """Compare two debug snapshots.

        Returns a dict with:
            fidelity: float - |<a|b>|^2
            tvd: float - total variation distance of probability distributions
            amplitude_diffs: list of (index, bitstring, amp_a, amp_b, |diff|)
                for the top differing amplitudes
            entropy_diff: float - entropy(b) - entropy(a)
            prob_diffs: np.ndarray - |P(a) - P(b)| per basis state
        """
        data_a = snap_a.state.data
        data_b = snap_b.state.data
        n = snap_a.state.num_qubits

        fid = StateAnalysis.state_fidelity(data_a, data_b)

        prob_a = np.abs(data_a) ** 2
        prob_b = np.abs(data_b) ** 2
        tvd = 0.5 * np.sum(np.abs(prob_a - prob_b))

        # Find top amplitude differences
        amp_diffs = np.abs(data_a - data_b)
        top_indices = np.argsort(amp_diffs)[::-1][:min(10, len(amp_diffs))]

        amplitude_diffs = []
        for idx in top_indices:
            if amp_diffs[idx] < 1e-10:
                break
            bitstring = format(idx, f"0{n}b")
            amplitude_diffs.append((
                int(idx),
                bitstring,
                complex(data_a[idx]),
                complex(data_b[idx]),
                float(amp_diffs[idx]),
            ))

        return {
            "fidelity": float(fid),
            "tvd": float(tvd),
            "amplitude_diffs": amplitude_diffs,
            "entropy_diff": snap_b.entropy - snap_a.entropy,
            "prob_diffs": np.abs(prob_a - prob_b),
        }
