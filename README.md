# Semafold

[![CI](https://github.com/mindtro/semafold/actions/workflows/ci.yml/badge.svg)](https://github.com/mindtro/semafold/actions/workflows/ci.yml)
[![tests](https://img.shields.io/badge/tests-189%20passed-brightgreen)](https://github.com/mindtro/semafold/actions)
[![python](https://img.shields.io/badge/python-3.10%2B-blue)](https://github.com/mindtro/semafold)
[![license](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)

**Vector compression with TurboQuant codecs for embeddings, retrieval, and KV-cache. 10x compression, pure NumPy core — no GPU required by default, but professionally accelerated on NVIDIA (CUDA) and Apple Silicon (Metal) when available.**

Semafold is a vector-first compression toolkit for AI workloads that compresses embeddings, retrieval representations, and cache-shaped KV tensors with explicit byte accounting, typed encode/decode contracts, and validation evidence. It is designed for teams building AI infrastructure that need measurable storage reduction without losing visibility into distortion, artifact size, or integration boundaries.

Today it is strongest at two jobs:
- compressing embedding / vector workloads
- compressing cache-shaped K/V tensors with TurboQuant-based codecs

It gives you:
- typed encode/decode contracts
- measured byte accounting
- explicit guarantees and validation evidence
- deterministic synthetic validation and benchmarks
- pure NumPy core — no GPU required, runs anywhere
- enterprise GPU acceleration — zero-config, automatic offloading to PyTorch (CUDA/MPS) or MLX (Apple Metal) when installed

## Compression Results

| Workload | Baseline | Setting | Artifact Size | Smaller | Ratio |
|---|---:|---|---:|---:|---:|
| Embedding `128 x 1536` | `float32` `786,432 B` | `TurboQuantMSE 3-bit` | `74,738 B` | `90.50%` | `10.52x` |
| Embedding `128 x 1536` | `fp16/bf16` `393,216 B` | `TurboQuantMSE 3-bit` | `74,738 B` | `80.99%` | `5.26x` |
| KV tensor `(4,8,256,128)` | `float32` `8,388,608 B` | `K=Prod 3b, V=MSE 3b` | `885,734 B` | `89.44%` | `9.47x` |
| KV tensor `(4,8,256,128)` | `fp16/bf16` `4,194,304 B` | `K=Prod 3b, V=MSE 3b` | `885,734 B` | `78.88%` | `4.74x` |

Full benchmark details: [turboquant_benchmark_report.md](benchmarks/turboquant_benchmark_report.md)

Distribution / import names today:
- distribution: `semafold`
- import: `semafold`

## Architecture

```text
semafold
├─ Stable root API
│  ├─ core
│  │  ├─ CompressionBudget
│  │  ├─ CompressionEstimate
│  │  ├─ CompressionFootprint
│  │  ├─ CompressionGuarantee
│  │  └─ ValidationEvidence
│  └─ vector
│     ├─ VectorEncodeRequest
│     ├─ VectorEncoding
│     ├─ VectorDecodeRequest
│     └─ VectorCodec
├─ Codec layer
│  ├─ PassthroughVectorCodec
│  ├─ ScalarReferenceVectorCodec
│  └─ TurboQuant family
│     ├─ TurboQuantMSEVectorCodec
│     ├─ TurboQuantProdVectorCodec
│     └─ kv
│        ├─ TurboQuantKVConfig
│        └─ TurboQuantKVPreviewCodec
├─ Compute backend layer  (v0.2.0)
│  ├─ ComputeBackend protocol
│  ├─ NumPyBackend   — always available (default)
│  ├─ TorchBackend   — CUDA / MPS  (pip install semafold[torch])
│  └─ MLXBackend     — Metal       (pip install semafold[mlx])
└─ Validation and benchmarking
   ├─ contract / unit / integration tests
   ├─ paper-shaped vector validation
   └─ synthetic KV benchmark and benchmark report
```

Read it as:
- the stable root gives you the generic Semafold contract surface
- the codec layer provides concrete compression implementations
- the TurboQuant family is the current high-performance path for vector and KV-tensor workloads
- the validation layer keeps storage, distortion, and behavioral checks measurable

## Where It Fits

Semafold is a good fit when you want to reduce the storage footprint of numeric AI representations:

- embedding stores
- vector databases and retrieval pipelines
- long-term vector memory in AI orchestrators
- cache-shaped K/V tensor compression in custom inference stacks

Semafold is **not** a text summarizer. It does not shorten prompts or reduce token counts by rewriting text. Its current strength is compression of vectors and tensors.

## Current Capability Surface

Stable today:
- root imports from `semafold`
- `CompressionBudget`
- `CompressionEstimate`
- `CompressionFootprint`
- `CompressionGuarantee`
- `ValidationEvidence`
- `VectorEncodeRequest`
- `VectorEncodingSegment`
- `VectorEncoding`
- `VectorDecodeRequest`
- `VectorDecodeResult`
- `VectorCodec`
- `PassthroughVectorCodec`

Available today, but intentionally outside the stable root surface:
- `semafold.turboquant`
- `semafold.turboquant.kv`
- `ScalarReferenceVectorCodec`

That means TurboQuant already works, but it is currently a deep-import surface rather than a root export.

## Install

```bash
pip install semafold              # NumPy core — no GPU required
pip install semafold[torch]       # + NVIDIA CUDA / Apple MPS acceleration
pip install semafold[mlx]         # + Apple Silicon Metal acceleration
pip install "semafold[torch,mlx]" # both
```

## Quickstart

Install locally from the package directory:

```bash
python3 -m pip install -e ".[dev]"
```

Runnable versions of the examples below live in [examples/](examples/).

### Stable Root Quickstart

Run the exact file here: [examples/wire_roundtrip.py](examples/wire_roundtrip.py)

```python
import numpy as np

from semafold import PassthroughVectorCodec
from semafold import VectorDecodeRequest
from semafold import VectorEncodeRequest

codec = PassthroughVectorCodec()
request = VectorEncodeRequest(
    data=np.linspace(-1.0, 1.0, 1024, dtype=np.float32),
    objective="reconstruction",
)

encoding = codec.encode(request)
decoded = codec.decode(VectorDecodeRequest(encoding=encoding))

assert decoded.data.shape == request.data.shape
```

### TurboQuant Embedding Example

Run the exact file here: [examples/turboquant_embedding.py](examples/turboquant_embedding.py)

```python
import numpy as np

from semafold import VectorDecodeRequest
from semafold import VectorEncodeRequest
from semafold.turboquant import TurboQuantMSEConfig
from semafold.turboquant import TurboQuantMSEVectorCodec

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

print(encoding.footprint.total_bytes, encoding.footprint.compression_ratio)
assert decoded.data.shape == rows.shape
```

### TurboQuant KV Tensor Example

Run the exact file here: [examples/turboquant_kv_block.py](examples/turboquant_kv_block.py)

These examples use the current TurboQuant deep-import surface rather than stable root exports.

```python
import numpy as np

from semafold.turboquant.kv import TurboQuantKVConfig
from semafold.turboquant.kv import TurboQuantKVPreviewCodec

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

print(stats["combined_bytes"], stats["combined_compression_ratio"])
assert keys_hat.shape == keys.shape
assert values_hat.shape == values.shape
```

Runnable versions of these examples live here:

- [examples/README.md](examples/README.md)
- [examples/wire_roundtrip.py](examples/wire_roundtrip.py)
- [examples/turboquant_embedding.py](examples/turboquant_embedding.py)
- [examples/turboquant_kv_block.py](examples/turboquant_kv_block.py)

## Benchmark Details

Benchmark runners and detailed report:

- [turboquant_paper_validation.py](benchmarks/turboquant_paper_validation.py)
- [turboquant_synthetic_kv_benchmark.py](benchmarks/turboquant_synthetic_kv_benchmark.py)
- [turboquant_benchmark_report.md](benchmarks/turboquant_benchmark_report.md)

## Benchmarks

Run the synthetic benchmark runners from the package directory:

```bash
PYTHONPATH=src python benchmarks/turboquant_paper_validation.py --output /tmp/turboquant-paper.json
PYTHONPATH=src python benchmarks/turboquant_synthetic_kv_benchmark.py --output /tmp/turboquant-kv.json
```

Benchmark documentation lives here:
- [benchmarks/README.md](benchmarks/README.md)

## Validation and Quality Gates

Current local closeout commands:

```bash
PYTHONPATH=src pytest tests -q
PYTHONPATH=src pyright --project pyproject.toml src tests examples benchmarks
python3 -m build
```

## Repository Notes

- stability policy: [STABILITY.md](STABILITY.md)
- change log: [CHANGELOG.md](CHANGELOG.md)

## License

Semafold is currently packaged here under:
- [LICENSE](LICENSE)
- [NOTICE](NOTICE)

For the current package directory, the intended license is Apache-2.0.

## Current Maturity

Semafold already supports:
- vector / embedding compression
- cache-shaped K/V tensor compression
- measured compression accounting
- synthetic attention-proxy validation for compressed K/V tensors

The next layer after this is runtime/backend integration, not core compression math.

## References

- TurboQuant paper: [TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate](https://arxiv.org/abs/2504.19874)
