"""Quantum Circuit Debugger -- step-through execution with state inspection.

Provides breakpoint support, forward/backward stepping, noise impact analysis,
and state diff between any two execution points.

Noise attribution comes in two flavours:

* :meth:`CircuitDebugger.compute_noise_attribution` -- mean / std only.
  Cheap, suitable for the live GUI panel.
* :meth:`CircuitDebugger.compute_noise_attribution_with_statistics` --
  bootstrap CIs, two-sided p-values per column, Benjamini-Hochberg FDR
  correction, and recovery-rate analysis.  Designed for publication-
  quality figures and headless replay scripts.

Both routines share a single trial-collection pass to avoid simulating
the circuit twice.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .state_vector import StateVector
from .circuit import QuantumCircuit, GateInstance
from .gate_registry import GateRegistry
from .gates import GateType
from .analysis import StateAnalysis
from .statistics import (
    attribution_percentage,
    benjamini_hochberg,
    bootstrap_matrix_statistics,
    column_pvalues_from_bootstrap,
    recovery_rate,
)


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
class AttributionStatistics:
    """Bootstrap statistics attached to a :class:`NoiseAttribution`.

    All arrays are aligned with the gate-column axis of the parent
    attribution: index ``i`` refers to the ``i``-th column of the
    circuit's :pymeth:`compute_layers` output.

    Attributes
    ----------
    delta_fidelity_ci_lower, delta_fidelity_ci_upper : list[float]
        Two-sided percentile bootstrap CI bounds for each column's
        per-trial mean fidelity contribution.
    delta_fidelity_p_value : list[float]
        Two-sided bootstrap p-value for ``H0: mean(delta_F_i) == 0``.
    delta_fidelity_q_value : list[float]
        BH-FDR corrected q-values across the column family.
    column_significant : list[bool]
        ``True`` where ``q_value <= fdr_level`` -- i.e. the column's
        contribution is significantly different from zero after
        multiple-comparison correction.
    attribution_pct_ci_lower, attribution_pct_ci_upper : list[float]
        Bootstrap CI for the per-column attribution percentage.  These
        use the *same* row-resampled bootstrap as ``delta_fidelity``,
        so the joint dependence introduced by the percentage's
        denominator is preserved.
    recovery_rate : list[float]
        Empirical fraction of trials in which a column's contribution
        was negative (fidelity recovered).
    recovery_rate_ci_lower, recovery_rate_ci_upper : list[float]
        Bootstrap CI for the recovery rate.
    n_trials : int
        Number of stochastic trials averaged in the parent attribution.
    n_bootstrap : int
        Number of bootstrap resamples used to compute the CIs.
    confidence : float
        Two-sided CI level (typically ``0.95``).
    fdr_level : float
        FDR target used to derive ``column_significant``.
    """

    delta_fidelity_ci_lower: list[float]
    delta_fidelity_ci_upper: list[float]
    delta_fidelity_p_value: list[float]
    delta_fidelity_q_value: list[float]
    column_significant: list[bool]
    attribution_pct_ci_lower: list[float]
    attribution_pct_ci_upper: list[float]
    recovery_rate: list[float]
    recovery_rate_ci_lower: list[float]
    recovery_rate_ci_upper: list[float]
    n_trials: int
    n_bootstrap: int
    confidence: float
    fdr_level: float


@dataclass
class NoiseAttribution:
    """Per-gate noise attribution analysis.

    Quantifies how much each gate column contributes to total fidelity loss:
        delta_F_i = F(ref, psi_{i-1}) - F(ref, psi_i)

    Negative contributions (fidelity recovery) are preserved in raw values
    but clamped to zero for percentage attribution. Recovery columns are
    labeled with is_recovery flags.

    The optional :attr:`statistics` field carries bootstrap CIs and
    multiple-comparison-corrected p-values when the attribution was
    computed via
    :meth:`CircuitDebugger.compute_noise_attribution_with_statistics`.
    """

    delta_fidelity: list[float]           # per-column mean delta_F (may be negative)
    delta_fidelity_std: list[float]       # per-column std delta_F
    total_fidelity_loss: float            # F(ref, initial) - F(ref, final)
    column_attribution_pct: list[float]   # per-column % of total loss (clamped >= 0)
    per_qubit_attribution: list[list[float]]  # [col][qubit] fidelity contribution
    gate_labels: list[list[str]]          # [col] -> list of gate labels
    is_recovery: list[bool] = field(default_factory=list)  # True if delta_F < 0
    no_measurable_loss: bool = False      # True if total positive loss < epsilon
    statistics: AttributionStatistics | None = None  # bootstrap CIs / p-values


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

    def _collect_attribution_trials(
        self,
        circuit: QuantumCircuit,
        noise_model,
        n_trials: int,
        seed: int | None,
        twirl: bool = False,
        noise_columns: set[int] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, list[list[str]]]:
        """Run ``n_trials`` stochastic simulations and return raw per-column data.

        This is the shared kernel behind
        :meth:`compute_noise_attribution`,
        :meth:`compute_noise_attribution_with_statistics`, and the
        Pauli-twirled variants.  It performs the expensive part of
        attribution -- running the circuit ``n_trials`` times -- exactly
        once.

        When ``twirl`` is ``True`` each gate's noise application is
        Pauli-twirled per shot (see :class:`PRISM.engine.twirling.PauliTwirler`).
        Twirling is a no-op for Pauli channels and converts coherent
        noise into a stochastic Pauli channel, so attribution figures
        rendered with and without twirling diverge precisely where the
        underlying noise is non-Pauli.

        When ``noise_columns`` is given, noise is applied only at the
        listed column indices; all other columns evolve noiselessly.
        This enables interventional analyses such as leave-one-out
        attribution (run with all columns except ``i`` active) and
        targeted ground-truth injection, without touching the noise
        model itself.

        Returns
        -------
        noise_contrib_trials : np.ndarray, shape ``(n_trials, num_cols)``
            Per-trial per-column noise contribution
            ``delta_F_i = gap_i - gap_{i-1}`` where ``gap_i = 1 - F``.
        per_qubit_attr_acc : np.ndarray, shape ``(num_cols, n_qubits)``
            Sum (over trials) of per-qubit reduced-density-matrix
            fidelity drop.  Caller is responsible for dividing by
            ``n_trials``.
        gate_labels : list[list[str]]
            Human-readable labels per column.
        """
        base_rng = np.random.default_rng(seed)
        ordered = circuit.get_ordered_gates()
        num_cols = len(ordered)
        n_qubits = circuit.num_qubits

        noise_contrib_trials = np.zeros((n_trials, num_cols))
        pq_attr_acc = np.zeros((num_cols, n_qubits))

        all_labels: list[list[str]] = []
        for column_gates in ordered:
            labels = []
            for g in column_gates:
                gd = self._registry.get(g.gate_name)
                if gd.gate_type not in (GateType.MEASUREMENT, GateType.BARRIER):
                    qstr = ",".join(str(q) for q in g.target_qubits)
                    labels.append(f"{g.gate_name}({qstr})")
            all_labels.append(labels)

        # Lazy import: twirling is only needed when the caller asks for it.
        if twirl:
            from .twirling import PauliTwirler  # noqa: WPS433

        for trial in range(n_trials):
            trial_seed = int(base_rng.integers(0, 2**63))
            if noise_model is not None:
                noise_model.set_seed(trial_seed)
            twirl_rng = (
                np.random.default_rng(int(base_rng.integers(0, 2**63)))
                if twirl else None
            )

            ideal = StateVector.from_initial_states(circuit.initial_states)
            noisy = StateVector.from_initial_states(circuit.initial_states)
            prev_gap = 0.0

            for col_idx, column_gates in enumerate(ordered):
                column_noisy = (
                    noise_columns is None or col_idx in noise_columns
                )
                for gate_inst in column_gates:
                    gate_def = self._registry.get(gate_inst.gate_name)
                    if gate_def.gate_type in (GateType.MEASUREMENT, GateType.BARRIER):
                        continue
                    matrix = gate_def.matrix_func(*gate_inst.params)
                    ideal.apply_gate(matrix, gate_inst.target_qubits)
                    noisy.apply_gate(matrix, gate_inst.target_qubits)
                    if not column_noisy:
                        continue
                    if twirl:
                        PauliTwirler.apply_twirled_noise(
                            noisy, noise_model, gate_inst, twirl_rng,
                        )
                    elif noise_model is not None:
                        noise_model.apply(noisy, gate_inst)

                fid = StateAnalysis.state_fidelity(ideal.data, noisy.data)
                gap = 1.0 - fid
                noise_contrib_trials[trial, col_idx] = gap - prev_gap
                prev_gap = gap

                for q in range(n_qubits):
                    rho_ideal = ideal.get_reduced_density_matrix(q)
                    rho_noisy = noisy.get_reduced_density_matrix(q)
                    pq_attr_acc[col_idx, q] += (
                        1.0 - StateAnalysis.density_fidelity(rho_ideal, rho_noisy)
                    )

        return noise_contrib_trials, pq_attr_acc, all_labels

    @staticmethod
    def _aggregate_attribution(
        noise_contrib_trials: np.ndarray,
        pq_attr_acc: np.ndarray,
        all_labels: list[list[str]],
        statistics: AttributionStatistics | None = None,
    ) -> NoiseAttribution:
        """Aggregate trial-level data into a :class:`NoiseAttribution`.

        Mean / std / attribution-% logic is identical to the original
        single-pass implementation, so existing callers see no change.
        Optional bootstrap statistics are attached when supplied.
        """
        n_trials = noise_contrib_trials.shape[0]
        num_cols = noise_contrib_trials.shape[1]

        mean_contrib = np.mean(noise_contrib_trials, axis=0).tolist()
        std_contrib = np.std(noise_contrib_trials, axis=0).tolist()
        total_loss = float(np.sum(mean_contrib))

        is_recovery = [d < -1e-12 for d in mean_contrib]

        positive_sum = sum(max(0.0, d) for d in mean_contrib)
        no_loss = positive_sum <= 1e-12
        if not no_loss:
            attr_pct = [max(0.0, d) / positive_sum * 100.0 for d in mean_contrib]
        else:
            attr_pct = [0.0] * num_cols

        if n_trials > 0:
            pq_attr = (pq_attr_acc / n_trials).tolist()
        else:
            pq_attr = pq_attr_acc.tolist()

        return NoiseAttribution(
            delta_fidelity=mean_contrib,
            delta_fidelity_std=std_contrib,
            total_fidelity_loss=total_loss,
            column_attribution_pct=attr_pct,
            per_qubit_attribution=pq_attr,
            gate_labels=all_labels,
            is_recovery=is_recovery,
            no_measurable_loss=no_loss,
            statistics=statistics,
        )

    def compute_noise_attribution(
        self,
        circuit: QuantumCircuit,
        noise_model,
        reference_state: StateVector | None = None,
        n_trials: int = 50,
        seed: int | None = None,
        twirl: bool = False,
        noise_columns: set[int] | None = None,
    ) -> NoiseAttribution:
        """Compute per-gate noise attribution by tracking the fidelity gap
        between ideal and noisy trajectories at each column.

        ``noise_contrib_i = gap_i - gap_{i-1}`` where
        ``gap_i = 1 - F(ideal_i, noisy_i)``.  This isolates each column's
        noise contribution from gate progress.

        Args:
            circuit: Circuit to analyze.
            noise_model: NoiseModel to use.
            reference_state: Reserved for future external-reference support.
                Currently the ideal trajectory is always used.
            n_trials: Number of stochastic trials to average.
            seed: Base seed for reproducibility.
            twirl: When ``True`` each gate's noise application is Pauli-
                twirled per shot via
                :class:`PRISM.engine.twirling.PauliTwirler`.  Twirling is
                a no-op for Pauli channels (BitFlip / PhaseFlip /
                Depolarizing) and converts coherent / non-Pauli noise
                into a stochastic Pauli channel.  Defaults to ``False``.

        Returns:
            NoiseAttribution with per-column fidelity attribution and no
            attached statistics (use
            :meth:`compute_noise_attribution_with_statistics` for that).
        """
        trials, pq_acc, labels = self._collect_attribution_trials(
            circuit, noise_model, n_trials, seed, twirl=twirl,
            noise_columns=noise_columns,
        )
        return self._aggregate_attribution(trials, pq_acc, labels, statistics=None)

    def compute_noise_attribution_with_statistics(
        self,
        circuit: QuantumCircuit,
        noise_model,
        reference_state: StateVector | None = None,
        n_trials: int = 100,
        n_bootstrap: int = 1000,
        confidence: float = 0.95,
        fdr_level: float = 0.05,
        seed: int | None = None,
        twirl: bool = False,
        signal_floor: float = 1e-12,
    ) -> NoiseAttribution:
        """Per-gate attribution with bootstrap CIs and FDR-corrected p-values.

        Same trial-collection logic as :meth:`compute_noise_attribution`,
        but the per-trial matrix is then passed through a row-resampling
        bootstrap to produce:

        * Per-column 95% (default) CIs for ``mean(delta_F_i)``.
        * Per-column 95% CIs for the attribution percentage, jointly
          bootstrapped to preserve the percentage's denominator coupling.
        * Two-sided bootstrap p-values for ``H0: mean(delta_F_i) == 0``.
        * Benjamini-Hochberg FDR-corrected q-values across the column
          family, plus a boolean ``column_significant`` mask.
        * Per-column recovery rate ``P(delta_F_i < 0)`` with bootstrap
          CI -- useful for spotting columns that never truly contribute
          but whose mean is biased by occasional recovery events.

        ``n_trials`` defaults to 100 (vs 50 for the cheap version) so
        that the bootstrap has enough rows to resample meaningfully.

        Args:
            circuit: Circuit to analyse.
            noise_model: NoiseModel to use.
            reference_state: Reserved for future external-reference support.
            n_trials: Number of stochastic simulations.
            n_bootstrap: Number of bootstrap resamples.
            confidence: Two-sided CI level for all returned intervals.
            fdr_level: Target false discovery rate for column significance.
            seed: Base seed for reproducibility.  Both the simulation
                trials and the bootstrap use children of this seed.
            twirl: When ``True`` each gate's noise application is Pauli-
                twirled per shot via
                :class:`PRISM.engine.twirling.PauliTwirler`.  Defaults
                to ``False``.
            signal_floor: Columns whose ``|mean(delta_F)|`` falls below
                this floor are treated as physically inactive: their
                p-value is forced to ``1.0`` before FDR correction.
                Machine-epsilon offsets (~1e-16) on inactive columns are
                artefacts of floating-point reduction order -- their sign
                and magnitude are not stable across NumPy/BLAS versions,
                and a bootstrap over an (almost) constant sample would
                otherwise flag them as spuriously significant.  The
                default (``1e-12``) matches the epsilon used by
                :func:`PRISM.engine.statistics.attribution_percentage`
                and sits several orders of magnitude below any physical
                contribution at realistic trial budgets.

        Returns:
            :class:`NoiseAttribution` with :attr:`AttributionStatistics`
            attached.
        """
        master_rng = np.random.default_rng(seed)
        sim_seed = int(master_rng.integers(0, 2**63))
        boot_seed = int(master_rng.integers(0, 2**63))

        trials, pq_acc, labels = self._collect_attribution_trials(
            circuit, noise_model, n_trials, sim_seed, twirl=twirl,
        )

        boot_rng = np.random.default_rng(boot_seed)

        # Joint bootstrap on the trials matrix.  A single resampled matrix
        # produces a column vector via ``statistic_fn``; we run two
        # independent calls (one for mean, one for percentage) so they
        # share the *bootstrap concept* but use independent resamplings.
        # Sharing the same resampling indices would also be valid; we use
        # independent ones for simplicity, which is conservative.
        mean_estimates, mean_lo, mean_hi, mean_boot_dist = bootstrap_matrix_statistics(
            trials,
            statistic_fn=lambda m: m.mean(axis=0),
            confidence=confidence,
            n_bootstrap=n_bootstrap,
            rng=boot_rng,
        )

        pct_estimates, pct_lo, pct_hi, _ = bootstrap_matrix_statistics(
            trials,
            statistic_fn=attribution_percentage,
            confidence=confidence,
            n_bootstrap=n_bootstrap,
            rng=boot_rng,
        )

        # Per-column p-values from the mean bootstrap distribution.
        p_values = column_pvalues_from_bootstrap(
            mean_boot_dist, mean_estimates, null_value=0.0
        )

        # Noise-floor guard: a column whose mean contribution is at the
        # floating-point noise floor carries no physical signal.  Its
        # bootstrap p-value reflects a deterministic machine-epsilon
        # offset whose sign flips with the reduction order of the
        # underlying BLAS, so testing it would produce environment-
        # dependent "discoveries".  Force p = 1 so the FDR family only
        # contains physically meaningful hypotheses.
        inactive = np.abs(mean_estimates) < signal_floor
        p_values[inactive] = 1.0

        q_values, significant = benjamini_hochberg(p_values, fdr=fdr_level)

        # Recovery rate with bootstrap CI.
        rec_estimates, rec_lo, rec_hi, _ = bootstrap_matrix_statistics(
            trials,
            statistic_fn=recovery_rate,
            confidence=confidence,
            n_bootstrap=n_bootstrap,
            rng=boot_rng,
        )

        stats = AttributionStatistics(
            delta_fidelity_ci_lower=mean_lo.tolist(),
            delta_fidelity_ci_upper=mean_hi.tolist(),
            delta_fidelity_p_value=p_values.tolist(),
            delta_fidelity_q_value=q_values.tolist(),
            column_significant=significant.tolist(),
            attribution_pct_ci_lower=pct_lo.tolist(),
            attribution_pct_ci_upper=pct_hi.tolist(),
            recovery_rate=rec_estimates.tolist(),
            recovery_rate_ci_lower=rec_lo.tolist(),
            recovery_rate_ci_upper=rec_hi.tolist(),
            n_trials=n_trials,
            n_bootstrap=n_bootstrap,
            confidence=confidence,
            fdr_level=fdr_level,
        )

        return self._aggregate_attribution(trials, pq_acc, labels, statistics=stats)

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
