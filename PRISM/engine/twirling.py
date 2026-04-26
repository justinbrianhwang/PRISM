"""Pauli twirling for state-vector simulation.

For an arbitrary CPTP channel ``N`` representing the noise after a
gate ``G``, the Pauli-twirled channel is

    N_T(rho) = (1 / 4^k) * sum_{P in Pauli^k}  P  N(P rho P)  P,

where ``k`` is the number of qubits in ``N``'s support and the sum
runs over the ``4^k`` k-qubit Pauli operators.  The twirled channel
``N_T`` is *always* a stochastic Pauli channel (diagonal in the Pauli
basis), regardless of whether the original channel was coherent or
non-Pauli.

Pauli twirling is the foundation of randomised compilation
(Wallman & Emerson 2016): it converts coherent gate errors -- which
do not average out across shots -- into stochastic Pauli noise
that does, dramatically simplifying the effective noise structure
seen by error-correction protocols and benchmarking experiments.

This module provides a Monte Carlo implementation suitable for
state-vector simulation: per shot, sample a random Pauli on the
gate's target qubits and conjugate the noise application by it.
Averaging across shots converges to ``N_T`` applied to ``G(rho)``.

Sanity properties (verified in :file:`tests/test_twirling.py`):

* For Pauli channels (BitFlip, PhaseFlip, Depolarizing) the
  twirling is the identity transformation -- shot statistics match
  the untwirled run within Monte Carlo error.
* For :class:`PRISM.engine.noise.CoherentOverRotationNoise` the
  shot-to-shot fidelity variance drops sharply under twirling, and
  the mean fidelity converges to the twirled-channel result.
* Everything is deterministic under a fixed
  :class:`numpy.random.Generator`.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .gates import I_MATRIX, X_MATRIX, Y_MATRIX, Z_MATRIX
from .state_vector import StateVector


# ---------------------------------------------------------------------------
# Pauli string primitives
# ---------------------------------------------------------------------------


PAULI_LABELS: tuple[str, ...] = ("I", "X", "Y", "Z")
"""Labels for the four single-qubit Pauli operators."""

PAULI_MATRICES: dict[str, np.ndarray] = {
    "I": I_MATRIX,
    "X": X_MATRIX,
    "Y": Y_MATRIX,
    "Z": Z_MATRIX,
}
"""Mapping from Pauli label to its 2x2 unitary matrix."""


def random_pauli_string(
    n_qubits: int,
    rng: np.random.Generator,
) -> tuple[str, ...]:
    """Sample a uniform random ``n_qubits``-fold Pauli string.

    Each qubit independently gets one of ``I, X, Y, Z`` with
    probability ``1/4``.

    Parameters
    ----------
    n_qubits : int
        Number of qubits the Pauli string acts on.  ``0`` returns the
        empty tuple.
    rng : numpy.random.Generator
        Generator used to sample the Pauli labels.

    Returns
    -------
    tuple[str, ...]
        Length-``n_qubits`` tuple of single-qubit Pauli labels.
    """
    if n_qubits < 0:
        raise ValueError(f"n_qubits must be >= 0, got {n_qubits}")
    if n_qubits == 0:
        return ()
    indices = rng.integers(0, 4, size=n_qubits)
    return tuple(PAULI_LABELS[int(i)] for i in indices)


def apply_pauli_string(
    state: StateVector,
    paulis: Sequence[str],
    qubits: Sequence[int],
) -> None:
    """Apply a Pauli string to ``state`` in-place.

    Each qubit ``qubits[i]`` is acted on by the single-qubit Pauli
    matrix ``paulis[i]``.  Identity entries are skipped (no-op) for
    efficiency: a 4-qubit Pauli string averages 1 non-identity factor,
    so skipping ``'I'`` cuts gate calls by ~3/4 for typical twirling
    workloads.

    Parameters
    ----------
    state : StateVector
        Mutated in place.
    paulis : Sequence[str]
        Same length as ``qubits``.  Each entry must be one of
        ``'I', 'X', 'Y', 'Z'``.
    qubits : Sequence[int]
        Target qubit indices, length-matched with ``paulis``.

    Raises
    ------
    ValueError
        If ``paulis`` and ``qubits`` have different lengths or if any
        Pauli label is not one of the four valid options.
    """
    if len(paulis) != len(qubits):
        raise ValueError(
            f"paulis and qubits must have the same length: "
            f"{len(paulis)} vs {len(qubits)}"
        )
    for p, q in zip(paulis, qubits):
        if p == "I":
            continue
        try:
            matrix = PAULI_MATRICES[p]
        except KeyError as exc:
            raise ValueError(
                f"Unknown Pauli label {p!r}; expected one of "
                f"{PAULI_LABELS}"
            ) from exc
        state.apply_gate(matrix, [q])


# ---------------------------------------------------------------------------
# Twirler
# ---------------------------------------------------------------------------


class PauliTwirler:
    """Per-gate Pauli twirling of a noise channel.

    Designed to slot into the existing
    :pymeth:`PRISM.engine.debugger.CircuitDebugger._collect_attribution_trials`
    loop without changing the surrounding control flow.  The caller
    invokes :meth:`apply_twirled_noise` instead of
    ``noise_model.apply(state, gate)`` after each gate; the rest of
    the simulation -- ideal-trajectory tracking, fidelity
    accumulation, per-qubit attribution -- is identical.

    The twirler does not own the noise model: the user passes one in
    per call so a single :class:`PauliTwirler` instance can be reused
    across circuits and shots.
    """

    @staticmethod
    def apply_twirled_noise(
        state: StateVector,
        noise_model,
        gate,
        rng: np.random.Generator,
    ) -> tuple[str, ...]:
        """One Monte Carlo sample of the Pauli-twirled noise channel.

        Implements the textbook twirl by sandwiching the stochastic
        noise application between two copies of the same random Pauli
        string ``P`` on the gate's target qubits:

            state -> P state -> N(P state) -> P N(P state),

        which equals ``P N(P rho P) P`` on the density matrix and
        averages to ``N_T(rho)`` over shots.  Pauli matrices are
        Hermitian and self-inverse (``P^2 = I``), so the same ``P``
        is applied before and after noise.

        If ``noise_model`` is ``None`` this is a no-op and returns the
        all-identity Pauli string.

        Parameters
        ----------
        state : StateVector
            Mutated in place.
        noise_model : NoiseModel or None
            The noise model to twirl.  ``None`` short-circuits the
            twirl (no noise to apply, no Pauli to sample).
        gate : GateInstance
            The gate whose noise application is being twirled.  Used
            for its ``target_qubits``.
        rng : numpy.random.Generator
            Source of randomness for the Pauli sample.

        Returns
        -------
        tuple[str, ...]
            The Pauli string sampled for this twirling event.  Useful
            for diagnostic logging and tests.
        """
        if noise_model is None:
            return ()

        qubits = gate.target_qubits
        paulis = random_pauli_string(len(qubits), rng)

        # P rho P (apply Pauli before noise).  P is self-inverse so we
        # can use the same matrix for the post-noise application.
        apply_pauli_string(state, paulis, qubits)
        noise_model.apply(state, gate)
        apply_pauli_string(state, paulis, qubits)

        return paulis
