from __future__ import annotations

import numpy as np

from semafold import VectorDecodeRequest
from semafold import VectorEncodeRequest
from semafold.turboquant import TurboQuantProdConfig
from semafold.turboquant import TurboQuantProdVectorCodec
from semafold.vector.models import EncodeMetric, EncodeObjective


def _unit_rows(*, seed: int, shape: tuple[int, int]) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rows = rng.normal(size=shape).astype(np.float32)
    norms = np.linalg.norm(rows.astype(np.float64), axis=1, keepdims=True).astype(np.float32)
    norms = np.where(norms == 0.0, np.float32(1.0), norms)
    return np.asarray(rows / norms, dtype=np.float32)


def _mean_inner_product_error(*, queries: np.ndarray, data: np.ndarray, encoding) -> float:  # type: ignore[no-untyped-def]
    decoded = TurboQuantProdVectorCodec().decode(VectorDecodeRequest(encoding=encoding)).data.astype(np.float64)
    exact_scores = queries.astype(np.float64) @ data.astype(np.float64).T
    approx_scores = queries.astype(np.float64) @ decoded.T
    return float(np.mean(np.abs(approx_scores - exact_scores)))


def _theory_proxy(encoding) -> float:  # type: ignore[no-untyped-def]
    evidence = next(item for item in encoding.evidence if item.scope == "theory_proxy")
    value = evidence.metrics["mean_query_free_variance_factor"]
    assert isinstance(value, float)
    return value


def test_turboquant_prod_rate_distortion_tradeoff_is_visible_in_artifact_size_and_inner_product_quality() -> None:
    data = _unit_rows(seed=123, shape=(12, 64))
    queries = _unit_rows(seed=456, shape=(7, 64))
    request = VectorEncodeRequest(
        data=data,
        objective=EncodeObjective.INNER_PRODUCT_ESTIMATION,
        metric=EncodeMetric.DOT_PRODUCT_ERROR,
        role="embedding",
        seed=11,
    )

    low = TurboQuantProdVectorCodec(
        config=TurboQuantProdConfig(total_bits_per_scalar=2, default_rotation_seed=5, default_qjl_seed=17)
    ).encode(request)
    high = TurboQuantProdVectorCodec(
        config=TurboQuantProdConfig(total_bits_per_scalar=5, default_rotation_seed=5, default_qjl_seed=17)
    ).encode(request)

    assert low.footprint.total_bytes < high.footprint.total_bytes
    assert low.footprint.payload_bytes < high.footprint.payload_bytes
    assert low.footprint.compression_ratio > high.footprint.compression_ratio

    assert _theory_proxy(high) < _theory_proxy(low)
    assert _mean_inner_product_error(queries=queries, data=data, encoding=high) < _mean_inner_product_error(
        queries=queries,
        data=data,
        encoding=low,
    )
