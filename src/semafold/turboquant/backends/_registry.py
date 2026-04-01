"""Backend registry with lazy loading and graceful fallback.

Auto-detection priority: torch (CUDA) → torch (MPS) → mlx (Metal) → numpy (CPU).

The registry never imports optional dependencies at module level.
All backend loading is deferred until :func:`get_backend` is called.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from semafold.turboquant.backends._protocol import ComputeBackend

__all__ = ["get_backend", "list_backends"]

logger = logging.getLogger("semafold.backends")

_BACKEND_PRIORITY: list[str] = ["torch", "mlx", "numpy"]

_lock = threading.Lock()
_cached_backend: ComputeBackend | None = None


def _try_load_torch() -> ComputeBackend | None:
    try:
        import torch  # noqa: F811
    except ImportError:
        return None

    from semafold.turboquant.backends._torch import TorchBackend

    if torch.cuda.is_available():
        return TorchBackend(device="cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return TorchBackend(device="mps")

    logger.debug("PyTorch available but no GPU backend detected, skipping")
    return None


def _try_load_mlx() -> ComputeBackend | None:
    try:
        import mlx.core  # noqa: F811
    except ImportError:
        return None
    from semafold.turboquant.backends._mlx import MLXBackend

    return MLXBackend()


def _load_numpy() -> ComputeBackend:
    from semafold.turboquant.backends._numpy import NumPyBackend

    return NumPyBackend()


_LOADERS: dict[str, Callable[[], ComputeBackend | None]] = {
    "torch": _try_load_torch,
    "mlx": _try_load_mlx,
    "numpy": _load_numpy,
}


def _detect() -> ComputeBackend:
    """Walk the priority chain and return the first available backend."""
    for candidate in _BACKEND_PRIORITY:
        loader = _LOADERS[candidate]
        backend = loader()
        if backend is not None:
            logger.info(
                "Auto-detected backend: %s (%s)",
                backend.name,
                backend.device_description,
            )
            return backend
    # NumPy is always available, so this should never happen.
    raise RuntimeError("No compute backend available")  # pragma: no cover


def get_backend(name: str = "auto") -> ComputeBackend:
    """Return a compute backend by name, or auto-detect the best available.

    Parameters
    ----------
    name : str, default ``"auto"``
        ``"auto"`` selects the first available backend in priority order.
        Explicit names: ``"numpy"``, ``"torch"``, ``"mlx"``.

    Returns
    -------
    ComputeBackend
        A backend instance ready for use.

    Raises
    ------
    ValueError
        If *name* is not a recognised backend identifier.
    RuntimeError
        If an explicitly requested backend is not installed.
    """
    global _cached_backend  # noqa: PLW0603

    if name == "auto":
        with _lock:
            if _cached_backend is None:
                _cached_backend = _detect()
        return _cached_backend

    if name not in _LOADERS:
        available = sorted(_LOADERS)
        raise ValueError(f"Unknown backend: {name!r}. Available: {available}")

    backend = _LOADERS[name]()
    if backend is None:
        raise RuntimeError(
            f"Backend {name!r} is not available. "
            f"Install with: pip install semafold[{name}]"
        )
    return backend


def _reset_backend_cache() -> None:
    """Clear the cached auto-detected backend.  Test-only hook."""
    global _cached_backend  # noqa: PLW0603
    with _lock:
        _cached_backend = None


def _is_available(name: str) -> bool:
    """Return True if *name* backend can be used on this machine.

    Checks import availability and hardware presence without instantiating
    a backend object.  Safe to call at module import time.
    """
    if name == "numpy":
        return True
    if name == "torch":
        try:
            import torch  # noqa: F401

            return torch.cuda.is_available() or (
                hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
            )
        except ImportError:
            return False
    if name == "mlx":
        try:
            import mlx.core  # noqa: F401

            return True
        except ImportError:
            return False
    return False


def list_backends() -> dict[str, bool]:
    """Return availability of all registered backends.

    Returns
    -------
    dict[str, bool]
        Mapping of backend name to whether it can be used right now.
        Does not instantiate any backend — safe to call at any time.
    """
    return {name: _is_available(name) for name in _BACKEND_PRIORITY}
