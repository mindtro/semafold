"""Numerical equivalence tests across compute backends.

For each backend that is actually installed, we verify that:
- rotate / rotate_inverse produce results within float32 tolerance of NumPy
- normalize_rows / restore_norms produce results within float32 tolerance
- project produces results within float32 tolerance

GPU backends compute in float32 throughout; the NumPy backend uses float64
intermediates for normalization.  We therefore test with atol=1e-5, not
exact equality.

GPU-only tests are gated by ``pytest.importorskip`` so the suite degrades
gracefully on CPU-only machines.
"""

from __future__ import annotations

import numpy as np
import pytest

from semafold.turboquant.backends import get_backend, list_backends
from semafold.turboquant.rotation import seeded_haar_rotation


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def rng() -> np.random.Generator:
    return np.random.default_rng(2024)


@pytest.fixture(scope="module")
def rotation_128() -> np.ndarray:
    return seeded_haar_rotation(dimension=128, seed=7)


@pytest.fixture(scope="module")
def rows_64x128(rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal((64, 128)).astype(np.float32)


@pytest.fixture(scope="module")
def projection_128(rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal((128, 128)).astype(np.float32)


@pytest.fixture(scope="module")
def numpy_backend():  # type: ignore[no-untyped-def]
    return get_backend("numpy")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _skip_if_unavailable(name: str) -> None:
    """Skip the test if the backend is not installed on this machine."""
    if not list_backends().get(name, False):
        pytest.skip(f"{name!r} backend not available on this machine")


# ---------------------------------------------------------------------------
# PyTorch equivalence
# ---------------------------------------------------------------------------

class TestTorchNumericalEquivalence:
    """NumPy vs Torch results must be within float32 tolerance."""

    def setup_method(self) -> None:
        _skip_if_unavailable("torch")

    @pytest.fixture()
    def torch_backend(self):  # type: ignore[no-untyped-def]
        return get_backend("torch")

    def test_rotate_matches_numpy(
        self,
        numpy_backend,
        torch_backend,
        rows_64x128: np.ndarray,
        rotation_128: np.ndarray,
    ) -> None:
        ref = numpy_backend.rotate(rows_64x128, rotation_128)
        got = torch_backend.rotate(rows_64x128, rotation_128)
        np.testing.assert_allclose(got, ref, atol=1e-5, rtol=0)

    def test_rotate_inverse_matches_numpy(
        self,
        numpy_backend,
        torch_backend,
        rows_64x128: np.ndarray,
        rotation_128: np.ndarray,
    ) -> None:
        rotated = numpy_backend.rotate(rows_64x128, rotation_128)
        ref = numpy_backend.rotate_inverse(rotated, rotation_128)
        got = torch_backend.rotate_inverse(rotated, rotation_128)
        np.testing.assert_allclose(got, ref, atol=1e-5, rtol=0)

    def test_normalize_rows_matches_numpy(
        self,
        numpy_backend,
        torch_backend,
        rows_64x128: np.ndarray,
    ) -> None:
        ref_unit, ref_norms = numpy_backend.normalize_rows(rows_64x128)
        got_unit, got_norms = torch_backend.normalize_rows(rows_64x128)
        # Torch uses float32 throughout; NumPy uses float64 intermediate.
        # Tolerance is relaxed accordingly.
        np.testing.assert_allclose(got_unit, ref_unit, atol=1e-5, rtol=0)
        np.testing.assert_allclose(got_norms, ref_norms, atol=1e-5, rtol=0)

    def test_restore_norms_matches_numpy(
        self,
        numpy_backend,
        torch_backend,
        rows_64x128: np.ndarray,
    ) -> None:
        unit, norms = numpy_backend.normalize_rows(rows_64x128)
        ref = numpy_backend.restore_norms(unit, norms)
        got = torch_backend.restore_norms(unit, norms)
        np.testing.assert_allclose(got, ref, atol=1e-5, rtol=0)

    def test_project_matches_numpy(
        self,
        numpy_backend,
        torch_backend,
        rows_64x128: np.ndarray,
        projection_128: np.ndarray,
    ) -> None:
        ref = numpy_backend.project(rows_64x128, projection_128)
        got = torch_backend.project(rows_64x128, projection_128)
        np.testing.assert_allclose(got, ref, atol=1e-5, rtol=0)

    def test_full_pipeline_roundtrip_matches_numpy(
        self,
        numpy_backend,
        torch_backend,
        rows_64x128: np.ndarray,
        rotation_128: np.ndarray,
    ) -> None:
        """normalize → rotate → rotate_inverse → restore: same result."""
        def pipeline(b):  # type: ignore[no-untyped-def]
            unit, norms = b.normalize_rows(rows_64x128)
            rotated = b.rotate(unit, rotation_128)
            recovered = b.rotate_inverse(rotated, rotation_128)
            return b.restore_norms(recovered, norms)

        ref = pipeline(numpy_backend)
        got = pipeline(torch_backend)
        np.testing.assert_allclose(got, ref, atol=1e-5, rtol=0)

    def test_matrix_cache_returns_consistent_results(
        self,
        torch_backend,
        rows_64x128: np.ndarray,
        rotation_128: np.ndarray,
    ) -> None:
        """Calling rotate twice with the same matrix must yield identical output."""
        first = torch_backend.rotate(rows_64x128, rotation_128)
        second = torch_backend.rotate(rows_64x128, rotation_128)
        np.testing.assert_array_equal(first, second)


# ---------------------------------------------------------------------------
# MLX equivalence
# ---------------------------------------------------------------------------

class TestMLXNumericalEquivalence:
    """NumPy vs MLX results must be within float32 tolerance."""

    def setup_method(self) -> None:
        _skip_if_unavailable("mlx")

    @pytest.fixture()
    def mlx_backend(self):  # type: ignore[no-untyped-def]
        return get_backend("mlx")

    def test_rotate_matches_numpy(
        self,
        numpy_backend,
        mlx_backend,
        rows_64x128: np.ndarray,
        rotation_128: np.ndarray,
    ) -> None:
        ref = numpy_backend.rotate(rows_64x128, rotation_128)
        got = mlx_backend.rotate(rows_64x128, rotation_128)
        np.testing.assert_allclose(got, ref, atol=1e-5, rtol=0)

    def test_rotate_inverse_matches_numpy(
        self,
        numpy_backend,
        mlx_backend,
        rows_64x128: np.ndarray,
        rotation_128: np.ndarray,
    ) -> None:
        rotated = numpy_backend.rotate(rows_64x128, rotation_128)
        ref = numpy_backend.rotate_inverse(rotated, rotation_128)
        got = mlx_backend.rotate_inverse(rotated, rotation_128)
        np.testing.assert_allclose(got, ref, atol=1e-5, rtol=0)

    def test_normalize_rows_matches_numpy(
        self,
        numpy_backend,
        mlx_backend,
        rows_64x128: np.ndarray,
    ) -> None:
        ref_unit, ref_norms = numpy_backend.normalize_rows(rows_64x128)
        got_unit, got_norms = mlx_backend.normalize_rows(rows_64x128)
        np.testing.assert_allclose(got_unit, ref_unit, atol=1e-5, rtol=0)
        np.testing.assert_allclose(got_norms, ref_norms, atol=1e-5, rtol=0)

    def test_restore_norms_matches_numpy(
        self,
        numpy_backend,
        mlx_backend,
        rows_64x128: np.ndarray,
    ) -> None:
        unit, norms = numpy_backend.normalize_rows(rows_64x128)
        ref = numpy_backend.restore_norms(unit, norms)
        got = mlx_backend.restore_norms(unit, norms)
        np.testing.assert_allclose(got, ref, atol=1e-5, rtol=0)

    def test_project_matches_numpy(
        self,
        numpy_backend,
        mlx_backend,
        rows_64x128: np.ndarray,
        projection_128: np.ndarray,
    ) -> None:
        ref = numpy_backend.project(rows_64x128, projection_128)
        got = mlx_backend.project(rows_64x128, projection_128)
        # MLX Metal FMA (fused multiply-add) ordering differs from NumPy.
        # For non-orthogonal 128×128 projection, max abs diff ≈ 3e-5.
        np.testing.assert_allclose(got, ref, atol=1e-4, rtol=0)

    def test_matrix_cache_returns_consistent_results(
        self,
        mlx_backend,
        rows_64x128: np.ndarray,
        rotation_128: np.ndarray,
    ) -> None:
        """Calling rotate twice with the same matrix must yield identical output."""
        first = mlx_backend.rotate(rows_64x128, rotation_128)
        second = mlx_backend.rotate(rows_64x128, rotation_128)
        np.testing.assert_array_equal(first, second)
