"""MLX compute backend for Apple Silicon.

Accelerates GEMM operations using Metal GPU on M-series chips.

MLX uses unified memory — there is no explicit CPU/GPU transfer cost.
The matrix cache avoids repeated numpy → MLX conversion overhead for
rotation and projection matrices that are reused across many calls.

The upstream ``lru_cache(maxsize=32)`` on both rotation and projection
limits live arrays to ≤ 64 entries, so ``_matrix_cache`` is naturally
bounded.

Requires: ``pip install semafold[mlx]``
"""

from __future__ import annotations

import numpy as np
import mlx.core as mx

__all__ = ["MLXBackend"]


class MLXBackend:
    """MLX/Metal compute backend for Apple Silicon."""

    def __init__(self) -> None:
        # Cache converted MLX arrays keyed by (data pointer, shape).
        # Avoids repeated np → mx conversion for read-only matrices.
        self._matrix_cache: dict[tuple[int, tuple[int, ...]], mx.array] = {}

    @property
    def name(self) -> str:
        return "mlx"

    @property
    def device_description(self) -> str:
        return "Metal (Apple Silicon)"

    @property
    def is_accelerated(self) -> bool:
        return True

    @staticmethod
    def _to_mlx(array: np.ndarray) -> mx.array:
        return mx.array(np.ascontiguousarray(array), dtype=mx.float32)

    @staticmethod
    def _to_numpy(tensor: mx.array) -> np.ndarray:
        mx.eval(tensor)
        return np.array(tensor, dtype=np.float32)

    def _pin_matrix(self, matrix: np.ndarray) -> mx.array:
        """Cache a converted MLX array keyed by (data pointer, shape).

        MLX uses unified memory — no PCIe transfer cost.  The cache
        avoids repeated np → mx conversion for frequently reused matrices
        such as rotation (D × D, float32, read-only from LRU cache).

        Keying by ``(ctypes.data, shape)`` is collision-safe: same pointer
        with a different shape is a distinct allocation; same shape with a
        different pointer is a different matrix.
        """
        key = (matrix.ctypes.data, matrix.shape)
        if key not in self._matrix_cache:
            self._matrix_cache[key] = mx.array(
                np.ascontiguousarray(matrix), dtype=mx.float32
            )
        return self._matrix_cache[key]

    def rotate(
        self,
        unit_rows: np.ndarray,
        rotation: np.ndarray,
    ) -> np.ndarray:
        rows_m = self._to_mlx(unit_rows)
        rot_m = self._pin_matrix(rotation)
        return self._to_numpy(rows_m @ rot_m.T)

    def rotate_inverse(
        self,
        rotated_rows: np.ndarray,
        rotation: np.ndarray,
    ) -> np.ndarray:
        rows_m = self._to_mlx(rotated_rows)
        rot_m = self._pin_matrix(rotation)
        return self._to_numpy(rows_m @ rot_m)

    def project(
        self,
        rows: np.ndarray,
        projection: np.ndarray,
    ) -> np.ndarray:
        rows_m = self._to_mlx(rows)
        proj_m = self._pin_matrix(projection)
        return self._to_numpy(rows_m @ proj_m.T)

    def normalize_rows(
        self,
        rows: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        rows_m = self._to_mlx(rows)
        norms_m = mx.linalg.norm(rows_m, axis=1)
        safe = mx.where(norms_m > 0, norms_m, mx.ones_like(norms_m))
        unit_m = rows_m / mx.expand_dims(safe, 1)
        return self._to_numpy(unit_m), self._to_numpy(norms_m)

    def restore_norms(
        self,
        unit_rows: np.ndarray,
        norms: np.ndarray,
    ) -> np.ndarray:
        unit_m = self._to_mlx(unit_rows)
        norms_m = self._to_mlx(norms)
        return self._to_numpy(unit_m * mx.expand_dims(norms_m, 1))
