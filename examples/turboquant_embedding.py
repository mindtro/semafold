"""TurboQuant embedding example with compression output."""

from __future__ import annotations

import numpy as np

from semafold import VectorDecodeRequest
from semafold import VectorEncodeRequest
from semafold.turboquant import TurboQuantMSEConfig
from semafold.turboquant import TurboQuantMSEVectorCodec
from semafold.turboquant.backends import get_backend


def _row_cosine_similarity(lhs: np.ndarray, rhs: np.ndarray) -> float:
    lhs64 = lhs.astype(np.float64, copy=False)
    rhs64 = rhs.astype(np.float64, copy=False)
    numerator = np.sum(lhs64 * rhs64, axis=1)
    denominator = np.linalg.norm(lhs64, axis=1) * np.linalg.norm(rhs64, axis=1)
    safe = np.where(denominator > 0.0, denominator, 1.0)
    return float(np.mean(numerator / safe))


def _format_bytes(value: int) -> str:
    return f"{value:,} B"


def _format_summary(*, baseline_bytes: int, artifact_bytes: int, mse: float, cosine_similarity: float) -> str:
    bytes_saved = baseline_bytes - artifact_bytes
    smaller_pct = (100.0 * bytes_saved / baseline_bytes) if baseline_bytes > 0 else 0.0
    ratio = (baseline_bytes / artifact_bytes) if artifact_bytes > 0 else 0.0
    backend = get_backend()
    return "\n".join(
        [
            "Semafold TurboQuant embedding example",
            f"Active compute backend: {backend.name.upper()} ({backend.device_description})",
            f"baseline bytes: {_format_bytes(baseline_bytes)}",
            f"artifact bytes: {_format_bytes(artifact_bytes)}",
            f"bytes saved: {_format_bytes(bytes_saved)}",
            f"smaller: {smaller_pct:.2f}%",
            f"compression ratio: {ratio:.2f}x",
            f"mse: {mse:.6f}",
            f"mean row cosine similarity: {cosine_similarity:.6f}",
        ]
    )


def main() -> None:
    """Compress and reconstruct a synthetic embedding batch with TurboQuant."""

    rows = np.random.default_rng(7).normal(size=(128, 1536)).astype(np.float32)
    codec = TurboQuantMSEVectorCodec(
        config=TurboQuantMSEConfig(default_bits_per_scalar=3, default_rotation_seed=7)
    )
    encoding = codec.encode(
        VectorEncodeRequest(
            data=rows,
            objective="reconstruction",
            metric="mse",
            role="embedding",
            seed=11,
        )
    )
    decoded = codec.decode(VectorDecodeRequest(encoding=encoding))
    mse = float(np.mean((rows.astype(np.float64) - decoded.data.astype(np.float64)) ** 2))
    cosine_similarity = _row_cosine_similarity(rows, decoded.data)
    if decoded.data.shape != rows.shape:
        raise SystemExit("embedding shape mismatch after round-trip")
    print(
        _format_summary(
            baseline_bytes=int(rows.nbytes),
            artifact_bytes=int(encoding.footprint.total_bytes),
            mse=mse,
            cosine_similarity=cosine_similarity,
        )
    )
    print(f"segments: {[segment.segment_kind for segment in encoding.segments]}")


if __name__ == "__main__":
    main()
