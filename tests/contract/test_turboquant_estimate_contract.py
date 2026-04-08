from __future__ import annotations

import numpy as np
import pytest

from semafold import EncodeMetric, EncodeObjective, VectorEncodeRequest
from semafold.turboquant import (
    TurboQuantMSEConfig,
    TurboQuantMSEVectorCodec,
    TurboQuantProdConfig,
    TurboQuantProdVectorCodec,
)


def _normalized_rows(*, seed: int, shape: tuple[int, int], dtype: type[np.generic]) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rows = rng.normal(size=shape).astype(np.float32)
    norms = np.linalg.norm(rows.astype(np.float64), axis=1, keepdims=True).astype(np.float32)
    norms = np.where(norms == 0.0, np.float32(1.0), norms)
    return np.asarray(rows / norms, dtype=dtype)


@pytest.mark.parametrize(
    ("codec", "encode_request"),
    [
        (
            TurboQuantMSEVectorCodec(
                config=TurboQuantMSEConfig(default_bits_per_scalar=3, default_rotation_seed=7)
            ),
            VectorEncodeRequest(
                data=np.random.default_rng(17).normal(size=(8, 32)).astype(np.float32),
                objective=EncodeObjective.RECONSTRUCTION,
                metric=EncodeMetric.MSE,
                role="embedding",
                seed=19,
            ),
        ),
        (
            TurboQuantProdVectorCodec(
                config=TurboQuantProdConfig(total_bits_per_scalar=4, default_rotation_seed=7, default_qjl_seed=11)
            ),
            VectorEncodeRequest(
                data=_normalized_rows(seed=23, shape=(8, 32), dtype=np.float32),
                objective=EncodeObjective.INNER_PRODUCT_ESTIMATION,
                metric=EncodeMetric.DOT_PRODUCT_ERROR,
                role="embedding",
                seed=29,
            ),
        ),
    ],
)
def test_turboquant_estimate_contract_exposes_exact_accounting_fields(
    codec,
    encode_request: VectorEncodeRequest,
) -> None:
    estimate = codec.estimate(encode_request)
    encoding = codec.encode(encode_request)

    assert estimate.baseline_bytes == int(encode_request.data.nbytes)
    assert estimate.estimated_payload_bytes is not None
    assert estimate.estimated_metadata_bytes is not None
    assert estimate.estimated_sidecar_bytes is not None
    assert estimate.estimated_protected_passthrough_bytes == 0
    assert estimate.estimated_decoder_state_bytes == 0
    assert estimate.estimated_total_bytes is not None
    assert estimate.estimated_compression_ratio is not None

    assert estimate.estimated_total_bytes == (
        estimate.estimated_payload_bytes
        + estimate.estimated_metadata_bytes
        + estimate.estimated_sidecar_bytes
        + estimate.estimated_protected_passthrough_bytes
        + estimate.estimated_decoder_state_bytes
    )
    assert estimate.estimated_compression_ratio == pytest.approx(
        float(estimate.baseline_bytes) / float(estimate.estimated_total_bytes)
    )

    assert encoding.footprint.payload_bytes == estimate.estimated_payload_bytes
    assert encoding.footprint.metadata_bytes == estimate.estimated_metadata_bytes
    assert encoding.footprint.sidecar_bytes == estimate.estimated_sidecar_bytes
    assert encoding.footprint.protected_passthrough_bytes == estimate.estimated_protected_passthrough_bytes
    assert encoding.footprint.decoder_state_bytes == estimate.estimated_decoder_state_bytes
    assert encoding.footprint.total_bytes == estimate.estimated_total_bytes
    assert encoding.footprint.compression_ratio == pytest.approx(estimate.estimated_compression_ratio)
