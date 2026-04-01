"""Formal protocol for TurboQuant compute backends.

A backend provides device-accelerated implementations of the linear
algebra primitives used by TurboQuant codecs.  All other operations
(bit packing, serialization, metadata, codebook quantize/dequantize)
remain in NumPy regardless of backend.

Inputs and outputs at the protocol boundary are always ``numpy.ndarray``.
Device transfer is the backend's responsibility.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

__all__ = ["ComputeBackend"]


@runtime_checkable
class ComputeBackend(Protocol):
    """Contract for TurboQuant compute acceleration backends.

    Implementors provide linear algebra primitives on a specific
    device (CPU, CUDA, Metal).  The protocol is intentionally narrow:
    only operations where GPU acceleration yields measurable gains
    are included.

    ==========  ==========  ==============================
    Operation   Complexity  GPU benefit
    ==========  ==========  ==============================
    rotate       O(V × D²)  Large — GEMM
    project      O(V × D²)  Large — GEMM
    normalize    O(V × D)   Medium — vectorized
    ==========  ==========  ==============================
    """

    @property
    def name(self) -> str:
        """Backend identifier: ``'numpy'``, ``'torch'``, or ``'mlx'``."""
        ...

    @property
    def device_description(self) -> str:
        """Human-readable device string, e.g. ``'CUDA (NVIDIA A100)'``."""
        ...

    @property
    def is_accelerated(self) -> bool:
        """``True`` if this backend uses hardware acceleration."""
        ...

    def rotate(
        self,
        unit_rows: np.ndarray,
        rotation: np.ndarray,
    ) -> np.ndarray:
        """Apply orthogonal rotation: ``unit_rows @ rotation.T``.

        Parameters
        ----------
        unit_rows : ndarray, shape ``(V, D)``, float32
        rotation  : ndarray, shape ``(D, D)``, float32

        Returns
        -------
        ndarray, shape ``(V, D)``, float32
        """
        ...

    def rotate_inverse(
        self,
        rotated_rows: np.ndarray,
        rotation: np.ndarray,
    ) -> np.ndarray:
        """Invert rotation: ``rotated_rows @ rotation``.

        Parameters
        ----------
        rotated_rows : ndarray, shape ``(V, D)``, float32
        rotation     : ndarray, shape ``(D, D)``, float32

        Returns
        -------
        ndarray, shape ``(V, D)``, float32
        """
        ...

    def project(
        self,
        rows: np.ndarray,
        projection: np.ndarray,
    ) -> np.ndarray:
        """Random projection: ``rows @ projection.T``.

        Used by QJL sign-bit encoding.

        Parameters
        ----------
        rows       : ndarray, shape ``(V, D)``, float32
        projection : ndarray, shape ``(P, D)``, float32

        Returns
        -------
        ndarray, shape ``(V, P)``, float32
        """
        ...

    def normalize_rows(
        self,
        rows: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """L2-normalize rows and return norms separately.

        Parameters
        ----------
        rows : ndarray, shape ``(V, D)``, float32

        Returns
        -------
        unit_rows : ndarray, shape ``(V, D)``, float32
        norms     : ndarray, shape ``(V,)``, float32
        """
        ...

    def restore_norms(
        self,
        unit_rows: np.ndarray,
        norms: np.ndarray,
    ) -> np.ndarray:
        """Scale unit rows by norms: ``unit_rows * norms[:, None]``.

        Parameters
        ----------
        unit_rows : ndarray, shape ``(V, D)``, float32
        norms     : ndarray, shape ``(V,)``, float32

        Returns
        -------
        ndarray, shape ``(V, D)``, float32
        """
        ...

    # ── v0.3.0 extension point ───────────────────────────────────
    #
    # When backends are wired into codecs, calling normalize → rotate
    # → project as separate methods causes V×D numpy round-trips
    # between each step.  For V=4M (KV cache) this is 512 MB per
    # transfer.  The fused pipeline keeps data on-device:
    #
    #   def encode_pipeline(
    #       self,
    #       rows: np.ndarray,
    #       rotation: np.ndarray,
    #       projection: np.ndarray | None,
    #   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    #       """Fused: normalize → rotate (→ project) in one device
    #       round-trip.  Returns (rotated_unit_rows, norms, projected).
    #       """
    #
    # Backends that do NOT override this will get a default
    # implementation that chains the individual methods.  GPU
    # backends override it for zero-copy pipelining.
