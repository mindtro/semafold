from __future__ import annotations

import numpy as np
import pytest

from semafold import VectorEncodeRequest
from semafold.turboquant import (
    TurboQuantMSEConfig,
    TurboQuantMSEVectorCodec,
    TurboQuantProdConfig,
    TurboQuantProdVectorCodec,
)
from semafold.vector.models import EncodeMetric, EncodeObjective


def _normalized_data(*, seed: int, shape: tuple[int, ...], dtype: type[np.generic]) -> np.ndarray:
    rng = np.random.default_rng(seed)
    data = rng.normal(size=shape).astype(np.float32)
    if len(shape) == 2:
        norms = np.linalg.norm(data.astype(np.float64), axis=1, keepdims=True).astype(np.float32)
        norms = np.where(norms == 0.0, np.float32(1.0), norms)
        data = np.asarray(data / norms, dtype=np.float32)
    return data.astype(dtype)


@pytest.mark.parametrize(
    ("codec_factory", "request_factory", "seed"),
    [
        (
            lambda: TurboQuantMSEVectorCodec(
                config=TurboQuantMSEConfig(default_bits_per_scalar=1, default_rotation_seed=5)
            ),
            lambda seed: VectorEncodeRequest(
                data=_normalized_data(seed=seed, shape=(16,), dtype=np.float32),
                objective=EncodeObjective.RECONSTRUCTION,
                metric=EncodeMetric.MSE,
                role="embedding",
                seed=11,
            ),
            101,
        ),
        (
            lambda: TurboQuantMSEVectorCodec(
                config=TurboQuantMSEConfig(default_bits_per_scalar=4, default_rotation_seed=5)
            ),
            lambda seed: VectorEncodeRequest(
                data=_normalized_data(seed=seed, shape=(6, 32), dtype=np.float64),
                objective=EncodeObjective.RECONSTRUCTION,
                metric=EncodeMetric.MSE,
                role="embedding",
                seed=13,
            ),
            202,
        ),
        (
            lambda: TurboQuantProdVectorCodec(
                config=TurboQuantProdConfig(total_bits_per_scalar=2, default_rotation_seed=7, default_qjl_seed=17)
            ),
            lambda seed: VectorEncodeRequest(
                data=_normalized_data(seed=seed, shape=(8, 32), dtype=np.float32),
                objective=EncodeObjective.INNER_PRODUCT_ESTIMATION,
                metric=EncodeMetric.DOT_PRODUCT_ERROR,
                role="embedding",
                seed=19,
            ),
            303,
        ),
        (
            lambda: TurboQuantProdVectorCodec(
                config=TurboQuantProdConfig(total_bits_per_scalar=5, default_rotation_seed=7, default_qjl_seed=17)
            ),
            lambda seed: VectorEncodeRequest(
                data=_normalized_data(seed=seed, shape=(4, 64), dtype=np.float16),
                objective=EncodeObjective.INNER_PRODUCT_ESTIMATION,
                metric=EncodeMetric.DOT_PRODUCT_ERROR,
                role="embedding",
                seed=23,
            ),
            404,
        ),
    ],
)
def test_turboquant_estimate_matches_encode_across_supported_shapes_and_precisions(
    codec_factory,
    request_factory,
    seed: int,
) -> None:
    codec = codec_factory()
    encode_request = request_factory(seed)

    estimate = codec.estimate(encode_request)
    encoding = codec.encode(encode_request)

    assert estimate.estimated_total_bytes is not None
    assert estimate.estimated_payload_bytes is not None
    assert estimate.estimated_metadata_bytes is not None
    assert estimate.estimated_sidecar_bytes is not None
    assert estimate.estimated_compression_ratio is not None

    assert encoding.footprint.total_bytes == estimate.estimated_total_bytes
    assert encoding.footprint.payload_bytes == estimate.estimated_payload_bytes
    assert encoding.footprint.metadata_bytes == estimate.estimated_metadata_bytes
    assert encoding.footprint.sidecar_bytes == estimate.estimated_sidecar_bytes
    assert encoding.footprint.compression_ratio == pytest.approx(estimate.estimated_compression_ratio)
    assert encoding.footprint.baseline_bytes == estimate.baseline_bytes
