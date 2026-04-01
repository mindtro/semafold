"""Default CPU backend using NumPy.

This backend delegates to existing TurboQuant functions so they
conform to the :class:`ComputeBackend` protocol.  It is always
available and incurs zero overhead.

Delegation avoids divergence: if the canonical implementations in
``semafold.turboquant.quantizer`` or ``semafold.turboquant.rotation``
evolve, this backend automatically picks up the changes.

Note: delegating to canonical functions means this backend inherits their
input validation guards (shape checks, finite checks).  GPU backends skip
those guards intentionally — validation is the codec's responsibility, not
the backend's.
"""

from __future__ import annotations

import numpy as np

from semafold.turboquant.quantizer import normalize_rows as _normalize_rows
from semafold.turboquant.quantizer import restore_rows as _restore_rows
from semafold.turboquant.rotation import apply_rotation as _apply_rotation
from semafold.turboquant.rotation import invert_rotation as _invert_rotation

__all__ = ["NumPyBackend"]


class NumPyBackend:
    """Pure NumPy compute backend (CPU).

    This is the fallback backend.  It is always importable and does
    not require any optional dependency beyond ``numpy``.
    """

    @property
    def name(self) -> str:
        return "numpy"

    @property
    def device_description(self) -> str:
        return "CPU (NumPy)"

    @property
    def is_accelerated(self) -> bool:
        return False

    def rotate(
        self,
        unit_rows: np.ndarray,
        rotation: np.ndarray,
    ) -> np.ndarray:
        return _apply_rotation(unit_rows, rotation)

    def rotate_inverse(
        self,
        rotated_rows: np.ndarray,
        rotation: np.ndarray,
    ) -> np.ndarray:
        return _invert_rotation(rotated_rows, rotation)

    def project(
        self,
        rows: np.ndarray,
        projection: np.ndarray,
    ) -> np.ndarray:
        return np.asarray(rows @ projection.T, dtype=np.float32)

    def normalize_rows(
        self,
        rows: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        return _normalize_rows(rows)

    def restore_norms(
        self,
        unit_rows: np.ndarray,
        norms: np.ndarray,
    ) -> np.ndarray:
        return _restore_rows(unit_rows, norms)
