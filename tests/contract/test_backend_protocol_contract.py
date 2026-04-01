"""Contract tests for the compute backend protocol and registry."""

from __future__ import annotations

import threading

import numpy as np
import pytest

from semafold.turboquant.backends import get_backend, list_backends
from semafold.turboquant.backends._protocol import ComputeBackend
from semafold.turboquant.backends._registry import _reset_backend_cache


class TestBackendRegistry:
    """Registry semantics: auto-detection, explicit selection, errors."""

    def setup_method(self) -> None:
        _reset_backend_cache()

    def teardown_method(self) -> None:
        _reset_backend_cache()

    def test_auto_detect_returns_a_backend(self) -> None:
        backend = get_backend("auto")
        assert isinstance(backend, ComputeBackend)

    def test_auto_detect_never_raises(self) -> None:
        # NumPy is always available, so auto should always succeed.
        backend = get_backend()
        assert backend.name in ("numpy", "torch", "mlx")

    def test_numpy_backend_always_available(self) -> None:
        backend = get_backend("numpy")
        assert backend.name == "numpy"
        assert backend.is_accelerated is False

    def test_list_backends_includes_numpy(self) -> None:
        available = list_backends()
        assert "numpy" in available
        assert available["numpy"] is True

    def test_list_backends_returns_all_registered_names(self) -> None:
        available = list_backends()
        assert set(available) == {"torch", "mlx", "numpy"}

    def test_unknown_backend_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown backend"):
            get_backend("nonexistent")

    def test_auto_detect_is_cached(self) -> None:
        a = get_backend("auto")
        b = get_backend("auto")
        assert a is b

    def test_reset_clears_cache(self) -> None:
        a = get_backend("auto")
        _reset_backend_cache()
        b = get_backend("auto")
        # After reset, a new instance is created.
        assert a is not b

    def test_auto_detect_is_thread_safe(self) -> None:
        results: list[ComputeBackend] = []
        errors: list[Exception] = []

        def worker() -> None:
            try:
                results.append(get_backend("auto"))
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        _reset_backend_cache()
        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(results) == 8
        # All threads must receive the same cached instance.
        assert all(r is results[0] for r in results)


class TestBackendErrorMessages:
    """Backend error messages must be actionable."""

    def test_unknown_backend_raises_value_error_with_name(self) -> None:
        with pytest.raises(ValueError, match="Unknown backend"):
            get_backend("nonexistent")

    def test_unavailable_explicit_backend_raises_with_install_hint(self) -> None:
        available = list_backends()
        for name in ("torch", "mlx"):
            if not available[name]:
                with pytest.raises(RuntimeError, match=rf"pip install semafold\[{name}\]"):
                    get_backend(name)


class TestBackendProtocolConformance:
    """Every backend must implement the ComputeBackend protocol."""

    def test_numpy_backend_is_protocol_compliant(self) -> None:
        backend = get_backend("numpy")
        assert isinstance(backend, ComputeBackend)
        assert backend.name == "numpy"
        assert isinstance(backend.device_description, str)
        assert backend.is_accelerated is False

    def test_numpy_backend_has_all_protocol_methods(self) -> None:
        backend = get_backend("numpy")
        for method_name in ("rotate", "rotate_inverse", "project", "normalize_rows", "restore_norms"):
            assert callable(getattr(backend, method_name, None)), f"missing: {method_name}"
