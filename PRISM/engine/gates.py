"""Quantum gate matrix definitions and GateDefinition dataclass."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Callable
from enum import Enum


class GateType(Enum):
    SINGLE = "single"
    CONTROLLED = "controlled"
    MULTI = "multi"
    MEASUREMENT = "measurement"
    BARRIER = "barrier"


@dataclass(frozen=True)
class GateDefinition:
    """Immutable definition of a quantum gate."""
    name: str
    display_name: str
    gate_type: GateType
    num_qubits: int
    num_params: int
    param_names: tuple[str, ...]
    matrix_func: Callable[..., np.ndarray]
    symbol: str
    color: str
    num_controls: int = 0
    num_targets: int = 1


# --- Fixed single-qubit gate matrices ---

I_MATRIX = np.eye(2, dtype=np.complex128)

X_MATRIX = np.array([[0, 1],
                      [1, 0]], dtype=np.complex128)

Y_MATRIX = np.array([[0, -1j],
                      [1j, 0]], dtype=np.complex128)

Z_MATRIX = np.array([[1, 0],
                      [0, -1]], dtype=np.complex128)

H_MATRIX = np.array([[1, 1],
                      [1, -1]], dtype=np.complex128) / np.sqrt(2)

S_MATRIX = np.array([[1, 0],
                      [0, 1j]], dtype=np.complex128)

S_DAG_MATRIX = np.array([[1, 0],
                          [0, -1j]], dtype=np.complex128)

T_MATRIX = np.array([[1, 0],
                      [0, np.exp(1j * np.pi / 4)]], dtype=np.complex128)

T_DAG_MATRIX = np.array([[1, 0],
                          [0, np.exp(-1j * np.pi / 4)]], dtype=np.complex128)


# --- Parameterized single-qubit gate functions ---

def rx_matrix(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -1j * s],
                      [-1j * s, c]], dtype=np.complex128)


def ry_matrix(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s],
                      [s, c]], dtype=np.complex128)


def rz_matrix(theta: float) -> np.ndarray:
    return np.array([[np.exp(-1j * theta / 2), 0],
                      [0, np.exp(1j * theta / 2)]], dtype=np.complex128)


def phase_matrix(phi: float) -> np.ndarray:
    return np.array([[1, 0],
                      [0, np.exp(1j * phi)]], dtype=np.complex128)


def u3_matrix(theta: float, phi: float, lam: float) -> np.ndarray:
    """General single-qubit unitary U3(theta, phi, lambda)."""
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([
        [c, -np.exp(1j * lam) * s],
        [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]
    ], dtype=np.complex128)


# --- Fixed multi-qubit gate matrices ---

CNOT_MATRIX = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0]], dtype=np.complex128)

CZ_MATRIX = np.diag([1, 1, 1, -1]).astype(np.complex128)

SWAP_MATRIX = np.array([
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1]], dtype=np.complex128)

# Toffoli (CCX) - 8x8
TOFFOLI_MATRIX = np.eye(8, dtype=np.complex128)
TOFFOLI_MATRIX[6, 6] = 0
TOFFOLI_MATRIX[7, 7] = 0
TOFFOLI_MATRIX[6, 7] = 1
TOFFOLI_MATRIX[7, 6] = 1

# Fredkin (CSWAP) - 8x8
FREDKIN_MATRIX = np.eye(8, dtype=np.complex128)
FREDKIN_MATRIX[5, 5] = 0
FREDKIN_MATRIX[6, 6] = 0
FREDKIN_MATRIX[5, 6] = 1
FREDKIN_MATRIX[6, 5] = 1


# --- Lambda wrappers for fixed matrices ---

def _const(matrix: np.ndarray) -> Callable[[], np.ndarray]:
    """Returns a no-arg callable that returns the given matrix."""
    def _fn() -> np.ndarray:
        return matrix
    return _fn
