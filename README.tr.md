# Semafold

[![CI](https://github.com/mindtro/semafold/actions/workflows/ci.yml/badge.svg)](https://github.com/mindtro/semafold/actions/workflows/ci.yml)
[![tests](https://img.shields.io/badge/tests-189%20passed-brightgreen)](https://github.com/mindtro/semafold/actions)
[![python](https://img.shields.io/badge/python-3.10%2B-blue)](https://github.com/mindtro/semafold)
[![license](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)

**Embedding, retrieval ve KV-cache is yukleri icin TurboQuant codec'leriyle vektor sıkıştırma. Varsayilan olarak saf NumPy cekirdegiyle calisir; uygun oldugunda NVIDIA (CUDA) ve Apple Silicon (Metal) uzerinde hizlandirma kullanabilir.**

Semafold, AI is yukleri icin embedding'leri, retrieval temsillerini ve cache bicimindeki KV tensorlerini; acik byte muhasebesi, tiplenmis encode/decode sozlesmeleri ve dogrulama kanitlariyla sıkıştıran, vektor odakli bir sıkıştırma arac kutusudur. Olculebilir depolama kazanci isterken bozulma, artifact boyutu ve entegrasyon sinirlari uzerindeki gorunurlugu kaybetmek istemeyen ekipler icin tasarlanmistir.

Bugun iki ana alanda en gucludur:
- embedding / vektor is yuklerini sıkıştırmak.
- TurboQuant tabanli codec'lerle cache bicimindeki K/V tensorlerini sıkıtırmak.

Sana sunduklari:
- tiplenmis encode/decode sozlesmeleri
- olculmus byte muhasebesi
- acik garanti ve dogrulama kanitlari
- deterministik sentetik dogrulama ve benchmark'lar
- saf NumPy cekirdegi, GPU zorunlulugu olmadan her yerde calisma
- kurulum yapildiginda PyTorch (CUDA/MPS) veya MLX (Apple Metal) ile otomatik GPU hizlandirma

## Sıkıştırma Sonuclari

| Is Yuku | Baslangic | Ayar | Artifact Boyutu | Kuculme | Oran |
|---|---:|---|---:|---:|---:|
| Embedding `128 x 1536` | `float32` `786,432 B` | `TurboQuantMSE 3-bit` | `74,738 B` | `90.50%` | `10.52x` |
| Embedding `128 x 1536` | `fp16/bf16` `393,216 B` | `TurboQuantMSE 3-bit` | `74,738 B` | `80.99%` | `5.26x` |
| KV tensor `(4,8,256,128)` | `float32` `8,388,608 B` | `K=Prod 3b, V=MSE 3b` | `885,734 B` | `89.44%` | `9.47x` |
| KV tensor `(4,8,256,128)` | `fp16/bf16` `4,194,304 B` | `K=Prod 3b, V=MSE 3b` | `885,734 B` | `78.88%` | `4.74x` |

Tum benchmark ayrintilari: [turboquant_benchmark_report.md](benchmarks/turboquant_benchmark_report.md)

Dagitim / import adlari:
- dagitim: `semafold`
- import: `semafold`

## Mimari

```text
semafold
|- Kararli kok API
|  |- core
|  |  |- CompressionBudget
|  |  |- CompressionEstimate
|  |  |- CompressionFootprint
|  |  |- CompressionGuarantee
|  |  '- ValidationEvidence
|  '- vector
|     |- VectorEncodeRequest
|     |- VectorEncoding
|     |- VectorDecodeRequest
|     '- VectorCodec
|- Codec katmani
|  |- PassthroughVectorCodec
|  |- ScalarReferenceVectorCodec
|  '- TurboQuant ailesi
|     |- TurboQuantMSEVectorCodec
|     |- TurboQuantProdVectorCodec
|     '- kv
|        |- TurboQuantKVConfig
|        '- TurboQuantKVPreviewCodec
|- Hesaplama backend katmani (v0.2.0)
|  |- ComputeBackend protocol
|  |- NumPyBackend   - her zaman mevcut (varsayilan)
|  |- TorchBackend   - CUDA / MPS  (pip install semafold[torch])
|  '- MLXBackend     - Metal       (pip install semafold[mlx])
'- Dogrulama ve benchmark
   |- contract / unit / integration testleri
   |- makale bicimli vektor dogrulamasi
   '- sentetik KV benchmark ve benchmark raporu
```

Bunu soyle okuyabilirsin:
- kararli kok katman, genel Semafold sozlesme yuzeyini verir
- codec katmani, somut sıkıştırma uygulamalarini sunar
- TurboQuant ailesi, su an vektor ve KV-tensor is yukleri icin yuksek performansli yoldur
- dogrulama katmani; depolama, bozulma ve davranissal kontrolleri olculebilir tutar

## Nerede Kullanilir

Semafold, sayisal AI temsillerinin depolama ayak izini azaltmak istediginde iyi bir secenektir:

- embedding depolari
- vektor veritabanlari ve retrieval pipeline'lari
- AI orchestrator'larinda uzun sureli vektor bellegi
- ozel inference stack'lerinde cache bicimindeki K/V tensor sıkıştırma

Semafold bir **metin ozetleme** araci degildir. Prompt'lari yeniden yazarak kisaltmaz veya token sayisini dusurmez. Mevcut gucu vektor ve tensor sıkıştırmadadır.

## Guncel Yetenek Yuzeyi

Bugun kararli olanlar:
- `semafold` kok import'lari
- `CompressionBudget`
- `CompressionEstimate`
- `CompressionFootprint`
- `CompressionGuarantee`
- `ValidationEvidence`
- `EncodingBoundType`
- `WorkloadSuitability`
- `VectorEncodeRequest`
- `VectorEncodingSegment`
- `VectorEncoding`
- `VectorDecodeRequest`
- `VectorDecodeResult`
- `VectorCodec`
- `PassthroughVectorCodec`
- `EncodeObjective`
- `EncodeMetric`
- `EncodingSegmentKind`

Bugun mevcut olup bilerek kararli kok yuzeyin disinda tutulanlar:
- `semafold.turboquant`
- `semafold.turboquant.kv`
- `ScalarReferenceVectorCodec`

Bu, TurboQuant'in hali hazirda calistigi ancak simdilik kok export yerine derin import yuzeyi olarak sunuldugu anlamina gelir.

## Kurulum

```bash
pip install semafold              # NumPy core - GPU gerekmez
pip install semafold[torch]       # + NVIDIA CUDA / Apple MPS hizlandirma
pip install semafold[mlx]         # + Apple Silicon Metal hizlandirma
pip install "semafold[torch,mlx]" # ikisi birden
```

## Hizli Baslangic

Paket dizininden yerel kurulum:

```bash
python3 -m pip install -e ".[dev]"
```

Asagidaki orneklerin calisabilir halleri [examples/](examples/) altindadir.

### Kararli Kok Hizli Baslangic

Buradaki dosyayi birebir calistir: [examples/wire_roundtrip.py](examples/wire_roundtrip.py)

```python
import numpy as np

from semafold import EncodeObjective
from semafold import PassthroughVectorCodec
from semafold import VectorDecodeRequest
from semafold import VectorEncodeRequest

codec = PassthroughVectorCodec()
request = VectorEncodeRequest(
    data=np.linspace(-1.0, 1.0, 1024, dtype=np.float32),
    objective=EncodeObjective.RECONSTRUCTION,
)

encoding = codec.encode(request)
decoded = codec.decode(VectorDecodeRequest(encoding=encoding))

assert decoded.data.shape == request.data.shape
```

### TurboQuant Embedding Ornegi

Buradaki dosyayi birebir calistir: [examples/turboquant_embedding.py](examples/turboquant_embedding.py)

```python
import numpy as np

from semafold import EncodeMetric
from semafold import EncodeObjective
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
        objective=EncodeObjective.RECONSTRUCTION,
        metric=EncodeMetric.MSE,
        role="embedding",
        seed=11,
    )
)
decoded = codec.decode(VectorDecodeRequest(encoding=encoding))

print(encoding.footprint.total_bytes, encoding.footprint.compression_ratio)
assert decoded.data.shape == rows.shape
```

### TurboQuant KV Tensor Ornegi

Buradaki dosyayi birebir calistir: [examples/turboquant_kv_block.py](examples/turboquant_kv_block.py)

Bu ornekler, kararli kok export'lari yerine mevcut TurboQuant derin import yuzeyini kullanir.

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

Bu orneklerin calisabilir halleri burada:

- [examples/README.md](examples/README.md)
- [examples/wire_roundtrip.py](examples/wire_roundtrip.py)
- [examples/turboquant_embedding.py](examples/turboquant_embedding.py)
- [examples/turboquant_kv_block.py](examples/turboquant_kv_block.py)

## Benchmark Ayrintilari

Benchmark calistiricilari ve detayli rapor:

- [turboquant_paper_validation.py](benchmarks/turboquant_paper_validation.py)
- [turboquant_synthetic_kv_benchmark.py](benchmarks/turboquant_synthetic_kv_benchmark.py)
- [turboquant_benchmark_report.md](benchmarks/turboquant_benchmark_report.md)

## Benchmark'lar

Sentetik benchmark calistiricilarini paket dizininden calistir:

```bash
PYTHONPATH=src python benchmarks/turboquant_paper_validation.py --output /tmp/turboquant-paper.json
PYTHONPATH=src python benchmarks/turboquant_synthetic_kv_benchmark.py --output /tmp/turboquant-kv.json
```

Benchmark dokumantasyonu burada:
- [benchmarks/README.md](benchmarks/README.md)

## Dogrulama ve Kalite Kapilari

Guncel yerel kapanis komutlari:

```bash
PYTHONPATH=src pytest tests -q
PYTHONPATH=src pyright --project pyproject.toml src tests examples benchmarks
python3 -m build
```

## Repo Notlari

- kararlilik politikasi: [STABILITY.md](STABILITY.md)
- degisiklik gunlugu: [CHANGELOG.md](CHANGELOG.md)

## Lisans

Semafold su anda bu paket dizininde sunulur:
- [LICENSE](LICENSE)
- [NOTICE](NOTICE)

Bu paket dizini icin hedeflenen lisans Apache-2.0'dir.

## Guncel Olgunluk Seviyesi

Semafold su anda sunlari destekler:
- vektor / embedding sıkıştırma
- cache bicimindeki K/V tensor sıkıştırma
- olculmus sıkıştırma muhasebesi
- sıkıştırılmış K/V tensorleri icin sentetik attention-proxy dogrulamasi

Bir sonraki katman, cekirdek sıkıştırma matematiginden cok runtime/backend entegrasyonudur.

## Referanslar

- TurboQuant makalesi: [TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate](https://arxiv.org/abs/2504.19874)
