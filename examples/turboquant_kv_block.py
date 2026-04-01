"""TurboQuant KV block example with cache accounting output."""

from __future__ import annotations

import numpy as np

from semafold.turboquant.kv import TurboQuantKVConfig
from semafold.turboquant.kv import TurboQuantKVPreviewCodec
from semafold.turboquant.backends import get_backend


def _format_bytes(value: int) -> str:
    return f"{value:,} B"


def _format_summary(*, baseline_bytes: int, fp16_bytes: int, bf16_bytes: int, stats: dict[str, float | int]) -> str:
    combined_bytes = int(stats["combined_bytes"])
    key_bytes = int(stats["key_bytes"])
    value_bytes = int(stats["value_bytes"])
    smaller_vs_float32 = 100.0 * (baseline_bytes - combined_bytes) / baseline_bytes if baseline_bytes > 0 else 0.0
    smaller_vs_fp16 = 100.0 * (fp16_bytes - combined_bytes) / fp16_bytes if fp16_bytes > 0 else 0.0
    smaller_vs_bf16 = 100.0 * (bf16_bytes - combined_bytes) / bf16_bytes if bf16_bytes > 0 else 0.0
    backend = get_backend()
    return "\n".join(
        [
            "Semafold TurboQuant KV block example",
            f"Active compute backend: {backend.name.upper()} ({backend.device_description})",
            f"baseline float32 bytes: {_format_bytes(baseline_bytes)}",
            f"baseline fp16/bf16 bytes: {_format_bytes(fp16_bytes)}",
            f"key artifact bytes: {_format_bytes(key_bytes)}",
            f"value artifact bytes: {_format_bytes(value_bytes)}",
            f"combined artifact bytes: {_format_bytes(combined_bytes)}",
            f"smaller vs float32: {smaller_vs_float32:.2f}%",
            f"smaller vs fp16: {smaller_vs_fp16:.2f}%",
            f"smaller vs bf16: {smaller_vs_bf16:.2f}%",
            f"combined compression ratio: {float(stats['combined_compression_ratio']):.2f}x",
            f"combined ratio vs fp16/bf16: {float(stats['combined_compression_ratio_vs_fp16']):.2f}x",
        ]
    )


def main() -> None:
    """Compress and reconstruct a synthetic cache-shaped K/V tensor block."""

    keys = np.random.default_rng(7).normal(size=(4, 8, 256, 128)).astype(np.float32)
    values = np.random.default_rng(11).normal(size=(4, 8, 256, 128)).astype(np.float32)
    codec = TurboQuantKVPreviewCodec(
        config=TurboQuantKVConfig(
            key_total_bits_per_scalar=3,
            value_bits_per_scalar=3,
            default_key_rotation_seed=7,
            default_key_qjl_seed=11,
            default_value_rotation_seed=7,
        )
    )
    artifact = codec.compress(keys, values)
    keys_hat, values_hat = codec.decompress(artifact)
    stats = codec.memory_stats(artifact)
    if keys_hat.shape != keys.shape or values_hat.shape != values.shape:
        raise SystemExit("KV shape mismatch after round-trip")
    print(
        _format_summary(
            baseline_bytes=int(keys.nbytes + values.nbytes),
            fp16_bytes=int((keys.nbytes + values.nbytes) // 2),
            bf16_bytes=int((keys.nbytes + values.nbytes) // 2),
            stats=stats,
        )
    )


if __name__ == "__main__":
    main()
