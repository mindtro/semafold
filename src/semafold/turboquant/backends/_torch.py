"""PyTorch/CUDA compute backend.

Accelerates GEMM operations (rotation, projection) on NVIDIA GPUs
or Apple Silicon via MPS.  Transfer overhead is amortized for large
batch sizes (V > 1000).

Rotation and projection matrices are cached on-device after the
first transfer to avoid repeated PCIe/USB uploads.  For D=4096 a
rotation matrix is 64 MB — re-uploading it every call would negate
GPU gains entirely.

Requires: ``pip install semafold[torch]``
"""

from __future__ import annotations

import numpy as np
import torch

__all__ = ["TorchBackend"]


class TorchBackend:
    """PyTorch compute backend (CUDA / MPS / CPU).

    Parameters
    ----------
    device : str, default ``"cuda"``
        PyTorch device string.  Falls back to ``"cpu"`` if the
        requested accelerator is not available.
    """

    def __init__(self, device: str = "cuda") -> None:
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        if device == "mps" and (
            not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available()
        ):
            device = "cpu"
        self._device = torch.device(device)
        self._matrix_cache: dict[tuple[int, tuple[int, ...]], torch.Tensor] = {}

    @property
    def name(self) -> str:
        return "torch"

    @property
    def device_description(self) -> str:
        if self._device.type == "cuda":
            idx = self._device.index or 0
            return f"CUDA ({torch.cuda.get_device_name(idx)})"
        if self._device.type == "mps":
            return "MPS (Apple Silicon)"
        return f"PyTorch ({self._device.type})"

    @property
    def is_accelerated(self) -> bool:
        return self._device.type in ("cuda", "mps")

    # ── Device transfer ──────────────────────────────────────────

    def _to_device(self, array: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(array)).to(
            device=self._device, dtype=torch.float32,
        )

    @staticmethod
    def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
        return tensor.detach().cpu().numpy().astype(np.float32)

    def _pin_matrix(self, matrix: np.ndarray) -> torch.Tensor:
        """Cache a read-only matrix on-device keyed by (data pointer, shape).

        ``id(matrix)`` alone is unsafe: if the upstream ``lru_cache`` evicts
        an entry and the array is GC'd, a new array may reuse the same address
        producing a stale cache hit with a wrong matrix.  The (ctypes.data,
        shape) pair is collision-safe — same pointer with a different shape
        produces a distinct key, and same shape with a different pointer is
        a different allocation.

        The upstream ``lru_cache(maxsize=32)`` on both rotation and projection
        limits live arrays to ≤ 64 entries, so this cache is naturally bounded.
        """
        key = (matrix.ctypes.data, matrix.shape)
        if key not in self._matrix_cache:
            self._matrix_cache[key] = torch.from_numpy(
                np.array(matrix, dtype=np.float32, copy=True)
            ).to(device=self._device, dtype=torch.float32)
        return self._matrix_cache[key]

    # ── Protocol methods ─────────────────────────────────────────

    def rotate(
        self,
        unit_rows: np.ndarray,
        rotation: np.ndarray,
    ) -> np.ndarray:
        rows_t = self._to_device(unit_rows)
        rot_t = self._pin_matrix(rotation)
        return self._to_numpy(rows_t @ rot_t.T)

    def rotate_inverse(
        self,
        rotated_rows: np.ndarray,
        rotation: np.ndarray,
    ) -> np.ndarray:
        rows_t = self._to_device(rotated_rows)
        rot_t = self._pin_matrix(rotation)
        return self._to_numpy(rows_t @ rot_t)

    def project(
        self,
        rows: np.ndarray,
        projection: np.ndarray,
    ) -> np.ndarray:
        rows_t = self._to_device(rows)
        proj_t = self._pin_matrix(projection)
        return self._to_numpy(rows_t @ proj_t.T)

    def normalize_rows(
        self,
        rows: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        # Computed in float32 throughout.  The canonical NumPy backend uses a
        # float64 intermediate for numerical stability; the difference is within
        # float32 epsilon for well-conditioned inputs (norms >> 0).
        rows_t = self._to_device(rows)
        norms_t = torch.linalg.norm(rows_t, dim=1)
        safe = torch.where(norms_t > 0, norms_t, torch.ones_like(norms_t))
        unit_t = rows_t / safe.unsqueeze(1)
        return self._to_numpy(unit_t), self._to_numpy(norms_t)

    def restore_norms(
        self,
        unit_rows: np.ndarray,
        norms: np.ndarray,
    ) -> np.ndarray:
        unit_t = self._to_device(unit_rows)
        norms_t = self._to_device(norms)
        return self._to_numpy(unit_t * norms_t.unsqueeze(1))
