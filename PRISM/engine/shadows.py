"""Classical shadows for state-vector simulation.

Implements the *Pauli classical shadow* protocol of Huang, Kueng, and
Preskill (2020) -- the cheapest variant of classical shadows, in
which each shot picks a uniformly random single-qubit Pauli basis per
qubit and reads out one bit.  Compared to random Clifford shadows,
the Pauli variant has more variance on high-weight observables but
is dramatically simpler to sample from a state-vector simulator.

What you can ask of it
----------------------

Given a state $\\rho$, after $N$ shadow shots one can estimate the
expectation value of any Pauli string $P = P_1 \\otimes \\cdots
\\otimes P_n$ via a *single-shot estimator*

.. math::

    \\hat{o}_t = \\prod_{i \\in \\mathrm{supp}(P)}
        3 \\cdot \\delta_{b_i^{(t)},\\, P_i}
        \\cdot \\langle s_i^{(t)} | \\sigma_{P_i} | s_i^{(t)} \\rangle,

where shot $t$ picked basis string $b^{(t)}$ and observed outcome
string $s^{(t)}$, $\\delta_{b, P}$ is the indicator that the random
basis matched the queried Pauli on that qubit, and the inner product
contributes $+1$ / $-1$ depending on the measurement outcome.  The
average $\\bar{o} = N^{-1} \\sum_t \\hat{o}_t$ converges to
$\\mathrm{Tr}(P \\rho)$.  Variance scales as $3^k$ where $k$ is the
support size of $P$, so single- and two-qubit Pauli observables are
estimable in $O(10)$ shots, and observables of weight 4-5 still in
$O(10^3)$ shots.  Higher-weight observables are off-limits without
switching to Clifford shadows -- a planned extension.

Why ship this in PRISM
----------------------

PRISM's headline attribution methodology is computed from the *full*
state vector at every column.  Hardware experiments do not have the
state vector; they have measurement shots.  Pauli classical shadows
let users cross-check that PRISM's attribution claims are recoverable
from the same measurement primitive that hardware actually exposes,
which is the standard way of validating a simulator's claims against
the experimental loop.

The interface in this module matches that role:

* :func:`take_pauli_shadows` -- run ``n_shots`` shadow shots on a
  :class:`StateVector` and return the snapshot record.
* :func:`estimate_pauli_string` -- on-the-fly estimator with Standard
  Error of the Mean.
* :func:`shadow_pauli_observables` -- batch helper that estimates a
  list of Pauli strings from one shadow set, useful for paper figures.

Sanity properties exercised in :file:`tests/test_shadows.py`:

* Bell-state shadows recover ``<XX> = +1``, ``<YY> = -1``, ``<ZZ> =
  +1`` within Monte Carlo error at $N = 2000$ shots.
* The standard error scales as $1/\\sqrt{N}$ (within constant
  factors).
* The empty Pauli string ``"III...I"`` always estimates to $1$
  exactly with zero variance (the trivial observable $\\mathbb{1}$).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .gates import H_MATRIX, S_DAG_MATRIX
from .state_vector import StateVector


PAULI_LABELS_NONIDENTITY: tuple[str, ...] = ("X", "Y", "Z")
"""Labels sampled per qubit per shot.  ``'I'`` is never sampled --
including it would produce shots that throw away their own outcome on
that qubit, which is slower and statistically equivalent."""


# ---------------------------------------------------------------------------
# Snapshot containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ShadowSnapshot:
    """One classical-shadow shot.

    Attributes
    ----------
    basis : tuple[str, ...]
        The random Pauli basis chosen on each qubit, in qubit-index
        order.  Each entry is one of ``'X'``, ``'Y'``, ``'Z'``.
    outcomes : tuple[int, ...]
        The measurement outcomes ``b_i in {0, 1}`` for each qubit, in
        the *measurement* basis (post-rotation).  ``b_i = 0``
        corresponds to the $+1$ eigenstate of the chosen Pauli on
        qubit $i$.
    """

    basis: tuple[str, ...]
    outcomes: tuple[int, ...]


@dataclass
class ShadowSet:
    """Collection of shadow snapshots from a single state."""

    n_qubits: int
    snapshots: list[ShadowSnapshot]

    @property
    def n_shots(self) -> int:
        return len(self.snapshots)


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------


def take_pauli_shadows(
    state: StateVector,
    n_shots: int,
    rng: np.random.Generator | None = None,
) -> ShadowSet:
    """Sample ``n_shots`` Pauli classical shadows from ``state``.

    Each shot does:

    1. Sample a uniform random basis label in ``{X, Y, Z}`` per qubit.
    2. Apply the rotation that brings that Pauli's $+1$ eigenstate to
       the computational $|0\\rangle$ state: ``H`` for ``X``,
       ``S^dagger H`` for ``Y``, identity for ``Z``.
    3. Measure all qubits in the computational basis.

    The state is *copied* before each shot so the caller's state is
    not consumed by the sampler.

    Parameters
    ----------
    state : StateVector
        The state to take shadows of.  Untouched by this function.
    n_shots : int
        Number of shadow shots to record.
    rng : numpy.random.Generator, optional
        Source of randomness.  ``None`` uses OS entropy.

    Returns
    -------
    ShadowSet
    """
    if n_shots < 0:
        raise ValueError(f"n_shots must be non-negative, got {n_shots}")
    rng = rng or np.random.default_rng()

    snapshots: list[ShadowSnapshot] = []
    n = state.num_qubits

    for _ in range(n_shots):
        basis = tuple(
            PAULI_LABELS_NONIDENTITY[int(rng.integers(0, 3))]
            for _ in range(n)
        )

        rotated = state.copy()
        for q, p in enumerate(basis):
            if p == "X":
                rotated.apply_gate(H_MATRIX, [q])
            elif p == "Y":
                # S^dagger followed by H rotates the Y +1 eigenstate
                # to the computational |0> state.
                rotated.apply_gate(S_DAG_MATRIX, [q])
                rotated.apply_gate(H_MATRIX, [q])
            # 'Z' -> no rotation needed; outcome already in Z basis

        bitstring = rotated.measure_all(rng)
        outcomes = tuple(int(b) for b in bitstring)
        snapshots.append(ShadowSnapshot(basis=basis, outcomes=outcomes))

    return ShadowSet(n_qubits=n, snapshots=snapshots)


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------


def estimate_pauli_string(
    shadow_set: ShadowSet,
    pauli_string: str,
) -> tuple[float, float]:
    """Estimate ``Tr(sigma_P rho)`` from a shadow set.

    Parameters
    ----------
    shadow_set : ShadowSet
        Snapshots taken from the state of interest.
    pauli_string : str
        Length-``n_qubits`` string drawn from
        ``{'I', 'X', 'Y', 'Z'}`` -- e.g. ``"XYIZ"`` for
        ``X_0 Y_1 I_2 Z_3``.  Identity entries reduce the support.

    Returns
    -------
    (mean_estimate, sem) : tuple[float, float]
        ``mean_estimate`` is the empirical estimator and ``sem`` is
        the standard error of the mean.

    Notes
    -----
    The trivial observable ``"III...I"`` (identity on all qubits) is
    handled as a special case: it equals $1$ exactly with zero
    variance, regardless of the shadow set.
    """
    n = shadow_set.n_qubits
    if len(pauli_string) != n:
        raise ValueError(
            f"pauli_string must have length {n}, got {len(pauli_string)}"
        )
    for ch in pauli_string:
        if ch not in ("I", "X", "Y", "Z"):
            raise ValueError(
                f"pauli_string entries must be in {{I, X, Y, Z}}, "
                f"got {ch!r}"
            )

    support = [i for i, p in enumerate(pauli_string) if p != "I"]
    queried_paulis = [pauli_string[i] for i in support]

    # The all-identity observable is the trace and equals 1 by
    # definition; report exactly with zero variance.
    if not support:
        return 1.0, 0.0

    n_shots = shadow_set.n_shots
    if n_shots == 0:
        return 0.0, 0.0

    estimates = np.empty(n_shots, dtype=float)
    for t, snap in enumerate(shadow_set.snapshots):
        contribution = 1.0
        for q, p_query in zip(support, queried_paulis):
            if snap.basis[q] != p_query:
                contribution = 0.0
                break
            # Outcome 0 -> +1 eigenvalue (in measurement basis after
            # rotation); outcome 1 -> -1.  The factor of 3 is the
            # inverse-measurement-channel coefficient for Pauli
            # shadows on a single qubit.
            sign = 1.0 if snap.outcomes[q] == 0 else -1.0
            contribution *= 3.0 * sign
        estimates[t] = contribution

    mean = float(estimates.mean())
    if n_shots > 1:
        sem = float(estimates.std(ddof=1) / np.sqrt(n_shots))
    else:
        sem = 0.0
    return mean, sem


def shadow_pauli_observables(
    shadow_set: ShadowSet,
    pauli_strings: list[str],
) -> dict[str, tuple[float, float]]:
    """Batch estimator over a list of Pauli strings.

    Convenience for paper figures that plot many observables at once.
    Returns a dict mapping each input string to its
    ``(mean, sem)`` estimate.
    """
    return {
        ps: estimate_pauli_string(shadow_set, ps)
        for ps in pauli_strings
    }
