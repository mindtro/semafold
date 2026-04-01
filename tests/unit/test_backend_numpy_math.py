"""Unit tests for the NumPy compute backend math correctness."""

from __future__ import annotations

import numpy as np
import pytest

from semafold.turboquant.backends import get_backend


@pytest.fixture()
def backend():  # type: ignore[no-untyped-def]
    return get_backend("numpy")


@pytest.fixture()
def rng():  # type: ignore[no-untyped-def]
    return np.random.default_rng(42)


@pytest.fixture()
def rotation(rng: np.random.Generator) -> np.ndarray:
    """Deterministic orthogonal rotation matrix."""
    from semafold.turboquant.rotation import seeded_haar_rotation

    return seeded_haar_rotation(dimension=128, seed=99)


class TestRotateInverseIdentity:
    """rotate → rotate_inverse must recover the original rows."""

    def test_roundtrip_is_identity(self, backend, rng, rotation) -> None:  # type: ignore[no-untyped-def]
        rows = rng.standard_normal((64, 128)).astype(np.float32)
        rotated = backend.rotate(rows, rotation)
        recovered = backend.rotate_inverse(rotated, rotation)
        np.testing.assert_allclose(recovered, rows, atol=1e-4)

    def test_output_shape_matches_input(self, backend, rng, rotation) -> None:  # type: ignore[no-untyped-def]
        rows = rng.standard_normal((32, 128)).astype(np.float32)
        rotated = backend.rotate(rows, rotation)
        assert rotated.shape == rows.shape
        assert rotated.dtype == np.float32


class TestNormalizeRestoreIdentity:
    """normalize_rows → restore_norms must recover the original rows."""

    def test_roundtrip_is_identity(self, backend, rng) -> None:  # type: ignore[no-untyped-def]
        rows = rng.standard_normal((64, 128)).astype(np.float32)
        unit, norms = backend.normalize_rows(rows)
        recovered = backend.restore_norms(unit, norms)
        np.testing.assert_allclose(recovered, rows, atol=1e-5)

    def test_unit_rows_have_unit_norm(self, backend, rng) -> None:  # type: ignore[no-untyped-def]
        rows = rng.standard_normal((32, 128)).astype(np.float32)
        unit, _ = backend.normalize_rows(rows)
        norms = np.linalg.norm(unit, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_zero_norm_rows_handled_safely(self, backend) -> None:  # type: ignore[no-untyped-def]
        rows = np.zeros((4, 128), dtype=np.float32)
        unit, norms = backend.normalize_rows(rows)
        assert np.all(np.isfinite(unit))
        np.testing.assert_allclose(norms, 0.0, atol=1e-8)

    def test_mixed_zero_and_nonzero_rows(self, backend, rng) -> None:  # type: ignore[no-untyped-def]
        """Zero-norm rows must produce zero unit vectors; nonzero rows must be unit."""
        nonzero = rng.standard_normal((4, 128)).astype(np.float32)
        rows = np.zeros((8, 128), dtype=np.float32)
        rows[::2] = nonzero  # alternating: zero, nonzero, zero, nonzero...
        unit, norms = backend.normalize_rows(rows)
        assert np.all(np.isfinite(unit))
        # Zero rows → zero unit vectors
        np.testing.assert_allclose(unit[1::2], 0.0, atol=1e-8)
        # Nonzero rows → unit norm
        nonzero_unit_norms = np.linalg.norm(unit[::2], axis=1)
        np.testing.assert_allclose(nonzero_unit_norms, 1.0, atol=1e-5)

    def test_single_row_input(self, backend, rng) -> None:  # type: ignore[no-untyped-def]
        """Backend must handle V=1 (single row) without broadcasting errors."""
        rows = rng.standard_normal((1, 128)).astype(np.float32)
        unit, norms = backend.normalize_rows(rows)
        assert unit.shape == (1, 128)
        assert norms.shape == (1,)
        np.testing.assert_allclose(np.linalg.norm(unit, axis=1), 1.0, atol=1e-5)

    def test_large_dimension(self, backend, rng) -> None:  # type: ignore[no-untyped-def]
        """Backend must handle D=4096 (KV cache head_dim scenario)."""
        rows = rng.standard_normal((16, 4096)).astype(np.float32)
        rotation_4096 = np.eye(4096, dtype=np.float32)  # identity for shape validation
        unit, norms = backend.normalize_rows(rows)
        rotated = backend.rotate(unit, rotation_4096)
        assert rotated.shape == (16, 4096)
        assert rotated.dtype == np.float32


class TestProject:
    """project output shape and dtype."""

    def test_output_shape_is_correct(self, backend, rng) -> None:  # type: ignore[no-untyped-def]
        rows = rng.standard_normal((64, 128)).astype(np.float32)
        projection = rng.standard_normal((32, 128)).astype(np.float32)
        result = backend.project(rows, projection)
        assert result.shape == (64, 32)
        assert result.dtype == np.float32

    def test_output_is_finite(self, backend, rng) -> None:  # type: ignore[no-untyped-def]
        rows = rng.standard_normal((16, 64)).astype(np.float32)
        projection = rng.standard_normal((8, 64)).astype(np.float32)
        result = backend.project(rows, projection)
        assert np.all(np.isfinite(result))


class TestNumPyBackendDelegation:
    """NumPy backend delegates to canonical implementations."""

    def test_rotate_matches_canonical(self, rng, rotation) -> None:  # type: ignore[no-untyped-def]
        from semafold.turboquant.rotation import apply_rotation

        rows = rng.standard_normal((32, 128)).astype(np.float32)
        backend = get_backend("numpy")
        backend_result = backend.rotate(rows, rotation)
        canonical_result = apply_rotation(rows, rotation)
        np.testing.assert_array_equal(backend_result, canonical_result)

    def test_normalize_matches_canonical(self, rng) -> None:  # type: ignore[no-untyped-def]
        from semafold.turboquant.quantizer import normalize_rows

        rows = rng.standard_normal((32, 128)).astype(np.float32)
        backend = get_backend("numpy")
        b_unit, b_norms = backend.normalize_rows(rows)
        c_unit, c_norms = normalize_rows(rows)
        np.testing.assert_array_equal(b_unit, c_unit)
        np.testing.assert_array_equal(b_norms, c_norms)
