from __future__ import annotations

import numpy as np

from semafold.turboquant.kv import TurboQuantKVConfig, TurboQuantKVPreviewCodec


def _normalize_last_axis(array: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(array.astype(np.float64), axis=-1, keepdims=True).astype(np.float32)
    norms = np.where(norms == 0.0, np.float32(1.0), norms)
    return np.asarray(array / norms, dtype=np.float32)


def _softmax(array: np.ndarray, *, axis: int = -1) -> np.ndarray:
    shifted = array - np.max(array, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def _attention_output(queries: np.ndarray, keys: np.ndarray, values: np.ndarray) -> np.ndarray:
    scale = float(np.sqrt(keys.shape[-1], dtype=np.float32))
    scores = np.einsum("bhqd,bhkd->bhqk", queries.astype(np.float64), keys.astype(np.float64)) / scale
    weights = _softmax(scores, axis=-1)
    return np.einsum("bhqk,bhkd->bhqd", weights, values.astype(np.float64))


def _sample_attention_inputs(*, seed: int = 123) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    queries = _normalize_last_axis(rng.standard_normal((2, 2, 5, 16), dtype=np.float32))
    keys = _normalize_last_axis(rng.standard_normal((2, 2, 7, 16), dtype=np.float32))
    values = rng.standard_normal((2, 2, 7, 16), dtype=np.float32)
    return queries, keys, values


def _attention_quality(
    *,
    queries: np.ndarray,
    keys: np.ndarray,
    values: np.ndarray,
    codec: TurboQuantKVPreviewCodec,
) -> tuple[dict[str, float | int], float, float]:
    artifact = codec.compress(keys, values)
    restored_keys, restored_values = codec.decompress(artifact)
    exact_output = _attention_output(queries, keys, values)
    approx_output = _attention_output(queries, restored_keys, restored_values)
    mse = float(np.mean(np.square(exact_output - approx_output)))
    cosine_similarity = float(
        np.sum(exact_output * approx_output)
        / ((np.linalg.norm(exact_output) + 1e-12) * (np.linalg.norm(approx_output) + 1e-12))
    )
    return codec.memory_stats(artifact), mse, cosine_similarity


def test_turboquant_kv_rate_distortion_tradeoff_is_visible_in_memory_stats_and_attention_quality() -> None:
    queries, keys, values = _sample_attention_inputs()

    low_codec = TurboQuantKVPreviewCodec(
        config=TurboQuantKVConfig(
            key_total_bits_per_scalar=2,
            value_bits_per_scalar=1,
            default_key_rotation_seed=7,
            default_key_qjl_seed=11,
            default_value_rotation_seed=17,
        )
    )
    high_codec = TurboQuantKVPreviewCodec(
        config=TurboQuantKVConfig(
            key_total_bits_per_scalar=5,
            value_bits_per_scalar=4,
            default_key_rotation_seed=7,
            default_key_qjl_seed=11,
            default_value_rotation_seed=17,
        )
    )

    low_stats, low_mse, low_cosine = _attention_quality(
        queries=queries,
        keys=keys,
        values=values,
        codec=low_codec,
    )
    high_stats, high_mse, high_cosine = _attention_quality(
        queries=queries,
        keys=keys,
        values=values,
        codec=high_codec,
    )

    assert int(low_stats["combined_bytes"]) < int(high_stats["combined_bytes"])
    assert int(low_stats["key_bytes"]) < int(high_stats["key_bytes"])
    assert int(low_stats["value_bytes"]) < int(high_stats["value_bytes"])
    assert float(low_stats["combined_compression_ratio"]) > float(high_stats["combined_compression_ratio"])

    assert high_mse < low_mse
    assert high_cosine > low_cosine


def test_turboquant_kv_key_bits_mainly_move_key_memory_and_attention_quality() -> None:
    queries, keys, values = _sample_attention_inputs()

    low_key_codec = TurboQuantKVPreviewCodec(
        config=TurboQuantKVConfig(
            key_total_bits_per_scalar=2,
            value_bits_per_scalar=3,
            default_key_rotation_seed=7,
            default_key_qjl_seed=11,
            default_value_rotation_seed=17,
        )
    )
    high_key_codec = TurboQuantKVPreviewCodec(
        config=TurboQuantKVConfig(
            key_total_bits_per_scalar=5,
            value_bits_per_scalar=3,
            default_key_rotation_seed=7,
            default_key_qjl_seed=11,
            default_value_rotation_seed=17,
        )
    )

    low_stats, low_mse, low_cosine = _attention_quality(
        queries=queries,
        keys=keys,
        values=values,
        codec=low_key_codec,
    )
    high_stats, high_mse, high_cosine = _attention_quality(
        queries=queries,
        keys=keys,
        values=values,
        codec=high_key_codec,
    )

    assert int(low_stats["key_bytes"]) < int(high_stats["key_bytes"])
    assert int(low_stats["combined_bytes"]) < int(high_stats["combined_bytes"])
    assert abs(int(low_stats["value_bytes"]) - int(high_stats["value_bytes"])) <= 16

    assert high_mse < low_mse
    assert high_cosine > low_cosine


def test_turboquant_kv_value_bits_mainly_move_value_memory_and_attention_quality() -> None:
    queries, keys, values = _sample_attention_inputs()

    low_value_codec = TurboQuantKVPreviewCodec(
        config=TurboQuantKVConfig(
            key_total_bits_per_scalar=3,
            value_bits_per_scalar=1,
            default_key_rotation_seed=7,
            default_key_qjl_seed=11,
            default_value_rotation_seed=17,
        )
    )
    high_value_codec = TurboQuantKVPreviewCodec(
        config=TurboQuantKVConfig(
            key_total_bits_per_scalar=3,
            value_bits_per_scalar=4,
            default_key_rotation_seed=7,
            default_key_qjl_seed=11,
            default_value_rotation_seed=17,
        )
    )

    low_stats, low_mse, low_cosine = _attention_quality(
        queries=queries,
        keys=keys,
        values=values,
        codec=low_value_codec,
    )
    high_stats, high_mse, high_cosine = _attention_quality(
        queries=queries,
        keys=keys,
        values=values,
        codec=high_value_codec,
    )

    assert int(low_stats["value_bytes"]) < int(high_stats["value_bytes"])
    assert int(low_stats["combined_bytes"]) < int(high_stats["combined_bytes"])
    assert abs(int(low_stats["key_bytes"]) - int(high_stats["key_bytes"])) <= 16

    assert high_mse < low_mse
    assert high_cosine > low_cosine
