"""Compute backend selection for TurboQuant acceleration.

Usage::

    from semafold.turboquant.backends import get_backend, list_backends

    backend = get_backend()        # auto-detect best available
    backend = get_backend("numpy") # explicit CPU selection
    backend = get_backend("torch") # requires: pip install semafold[torch]
    backend = get_backend("mlx")   # requires: pip install semafold[mlx]

    available = list_backends()    # {"numpy": True, "torch": False, ...}
"""

from semafold.turboquant.backends._registry import get_backend, list_backends

__all__ = ["get_backend", "list_backends"]
