from __future__ import annotations

import numpy as np

from semafold.turboquant.kv import TurboQuantKVConfig, TurboQuantKVPreviewCodec


def _normalize_last_axis(array: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(array.astype(np.float64), axis=-1, keepdims=True).astype(np.float32)
    norms = np.where(norms == 0.0, np.float32(1.0), norms)
    return np.asarray(array / norms, dtype=np.float32)


def test_turboquant_kv_memory_stats_golden_snapshot() -> None:
    rng = np.random.default_rng(31)
    keys = _normalize_last_axis(rng.standard_normal((2, 2, 6, 16), dtype=np.float32))
    values = rng.standard_normal((2, 2, 6, 16), dtype=np.float32)
    codec = TurboQuantKVPreviewCodec(
        config=TurboQuantKVConfig(
            key_total_bits_per_scalar=3,
            value_bits_per_scalar=3,
            default_key_rotation_seed=7,
            default_key_qjl_seed=11,
            default_value_rotation_seed=17,
        )
    )

    artifact = codec.compress(keys, values)

    assert codec.memory_stats(artifact) == {
        "baseline_bytes": 3072,
        "baseline_fp16_bytes": 1536,
        "baseline_bf16_bytes": 1536,
        "key_bytes": 829,
        "value_bytes": 722,
        "combined_bytes": 1551,
        "combined_compression_ratio": 1.9806576402321083,
        "combined_compression_ratio_vs_fp16": 0.9903288201160542,
        "combined_compression_ratio_vs_bf16": 0.9903288201160542,
    }
