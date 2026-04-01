# Changelog

## 0.2.0

- add optional GPU acceleration layer for TurboQuant hot-path linear algebra
- new `ComputeBackend` protocol covers `rotate`, `rotate_inverse`, `project`, `normalize_rows`, `restore_norms` — the two O(V×D²) GEMMs that dominate encode/decode cost
- `NumPyBackend`: always available, delegates to canonical `quantizer.py` and `rotation.py` to prevent drift
- `TorchBackend`: CUDA (NVIDIA) and MPS (Apple Silicon via PyTorch); install with `pip install semafold[torch]`
- `MLXBackend`: Metal (Apple Silicon via MLX); install with `pip install semafold[mlx]`
- thread-safe auto-detection registry with priority chain: CUDA → MPS → MLX → NumPy
- on-device matrix cache keyed by `(ctypes.data, shape)` — safe against `id()` reuse after LRU eviction, bounded to ≤ 64 entries
- `list_backends()` probes availability without instantiating backend objects
- v0.3.0 extension point stubbed in protocol: fused `encode_pipeline()` for zero-copy KV cache workloads
- plain `pip install semafold` unchanged — NumPy remains the default, no new required dependencies
- 189 tests passing; Pyright 0 errors

## 0.1.0

- initial Phase 0 and Phase 1 Semafold scaffold
- stable vector contract, measured accounting, and explicit evidence records
- root-exported `PassthroughVectorCodec` and non-root scalar reference codec
- package-facing closeout docs aligned to the current Phase 1 surface and verified commands
- preview-only TurboQuant KV config/codec surface under deep import `semafold.turboquant.kv`; stable root unchanged
- synthetic KV attention-proxy, multi-shape smoke, and fp16/bf16-aware accounting validation for the KV codec surface
