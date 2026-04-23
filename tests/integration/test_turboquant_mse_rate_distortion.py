from __future__ import annotations

import numpy as np

from semafold import VectorDecodeRequest
from semafold import VectorEncodeRequest
from semafold.turboquant import TurboQuantMSEConfig
from semafold.turboquant import TurboQuantMSEVectorCodec
from semafold.vector.models import EncodeMetric, EncodeObjective


def _observed_mse(encoding) -> float:  # type: ignore[no-untyped-def]
    value = next(guarantee.value for guarantee in encoding.guarantees if guarantee.metric == "observed_mse")
    assert isinstance(value, float)
    return value


def _decode_mse(*, data: np.ndarray, encoding) -> float:  # type: ignore[no-untyped-def]
    decoded = TurboQuantMSEVectorCodec().decode(VectorDecodeRequest(encoding=encoding)).data
    return float(np.mean((data.astype(np.float64) - decoded.astype(np.float64)) ** 2))


def test_turboquant_mse_rate_distortion_tradeoff_is_visible_in_artifact_size_and_decode_error() -> None:
    rng = np.random.default_rng(7)
    data = rng.normal(size=(12, 64)).astype(np.float32)
    request = VectorEncodeRequest(
        data=data,
        objective=EncodeObjective.RECONSTRUCTION,
        metric=EncodeMetric.MSE,
        role="embedding",
        seed=11,
    )

    low = TurboQuantMSEVectorCodec(
        config=TurboQuantMSEConfig(default_bits_per_scalar=1, default_rotation_seed=5)
    ).encode(request)
    high = TurboQuantMSEVectorCodec(
        config=TurboQuantMSEConfig(default_bits_per_scalar=4, default_rotation_seed=5)
    ).encode(request)

    assert low.footprint.total_bytes < high.footprint.total_bytes
    assert low.footprint.payload_bytes < high.footprint.payload_bytes
    assert low.footprint.compression_ratio > high.footprint.compression_ratio

    assert _observed_mse(high) < _observed_mse(low)
    assert _decode_mse(data=data, encoding=high) < _decode_mse(data=data, encoding=low)
