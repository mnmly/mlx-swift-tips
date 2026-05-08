# TIPS image-encoder benchmarks

Two benchmarks share the same protocol (iteration count, image, model variant,
dtype) so the numbers are directly comparable:

| Bench | Backend | Source |
|---|---|---|
| `Benchmarks/torch_tips_bench.py` | PyTorch (MPS / CUDA / CPU) | upstream `tips.pytorch` |
| `Tools/tips-bench/main.swift` | mlx-swift (Metal) | this repo's `TIPSPipeline` |

Both report identical fields: `inference-only` vs `end-to-end` mode, cold model
load time, mean / median / min / max over `iters` runs after `warmup` warmup
iterations.

## Modes

- **`inference-only` (default).** The image is decoded, resized, and uploaded
  to the device once. The timed loop runs only the model forward pass.
- **`end-to-end` (`--include-load`).** The timed loop also re-decodes the
  image from disk and uploads it, modeling a per-frame pipeline.

Synchronization:

- PyTorch: explicit `torch.mps.synchronize()` (or `cuda.synchronize()`)
  inside the timing loop.
- mlx-swift: lazy graphs are forced via `MLX.eval(...)` inside
  `TIPSPipeline.predict(_:)`, which is the inferred sync point.

## Reproduction

### Setup
- Get the upstream `google-deepmind/tips` repo cloned at a known path. You
  don't need to install it — both benches add its parent directory to
  `sys.path` directly (PEP 420 namespace package, no `__init__.py`).
- Have the HF snapshot cached (or pass `--snapshot <dir>`). The first run
  will download via `huggingface_hub.snapshot_download` when given
  `--hf-repo`.

### PyTorch reference

```bash
uv run --with torch --with safetensors --with huggingface_hub \
       --with pillow --with numpy \
    python Benchmarks/torch_tips_bench.py \
    --variant B --hf-repo \
    --input <path/to/image.jpg> \
    --iters 50 --warmup 5
```

Add `--include-load` for end-to-end mode. Use `--device cpu` to compare
against single-thread CPU. Defaults to MPS on Apple Silicon.

### mlx-swift port

Build with `xcodebuild` (not `swift run` — the latter doesn't ship the
Metal-capable toolchain MLX requires):

```bash
xcodebuild -scheme tips-bench -destination 'platform=macOS' \
    -configuration release -derivedDataPath .xcdd build

.xcdd/Build/Products/release/tips-bench \
    --variant B --hf-repo \
    --input <path/to/image.jpg> \
    --iters 50 --warmup 5
```

## Results

### B14, 448 px input

Hardware: Apple M5 Max, 128 GB unified memory, macOS 26.5.
100 timed iterations after 10 warmup. Image:
`python/tips/scenic/images/example_image.jpg` (448-resized).

#### fp32

| Backend | Mode | median | mean | min | max | cold load |
|---|---|---:|---:|---:|---:|---:|
| PyTorch (MPS) | inference-only | 24.0 ms | 24.1 ms | 23.6 ms | 24.8 ms | 0.93 s |
| mlx-swift (Metal) | inference-only | 9.0 ms | 9.1 ms | 8.8 ms | 9.4 ms | 0.11 s |
| PyTorch (MPS) | end-to-end | 29.6 ms | 29.6 ms | 28.3 ms | 31.9 ms | 0.89 s |
| mlx-swift (Metal) | end-to-end | 9.4 ms | 9.5 ms | 9.2 ms | 10.0 ms | 0.05 s |

#### fp16

| Backend | Mode | median | mean | min | max | cold load |
|---|---|---:|---:|---:|---:|---:|
| PyTorch (MPS) | inference-only | 8.3 ms | 8.3 ms | 8.1 ms | 8.7 ms | 1.31 s |
| mlx-swift (Metal) | inference-only | 7.1 ms | 7.1 ms | 6.9 ms | 7.5 ms | 0.20 s |
| PyTorch (MPS) | end-to-end | 14.4 ms | 14.4 ms | 13.2 ms | 15.7 ms | 1.35 s |
| mlx-swift (Metal) | end-to-end | 7.6 ms | 7.6 ms | 7.3 ms | 8.0 ms | 0.06 s |

### Reading these numbers

- **fp32 inference (model only):** mlx-swift ~2.7× faster. Most of this is
  PyTorch MPS's general-purpose op dispatch and the lack of a fused
  fp32-SDPA path on MPS in the version measured.
- **fp16 inference (model only):** the gap collapses to ~17%. PyTorch MPS
  with fp16 hits its faster attention kernel and is competitive.
- **End-to-end gap is bigger than inference-only.** Swift's `CGImage`
  decode + tight float conversion is faster than PIL→`Image.resize`→
  `ToTensor`→`.to(device, fp16)`. End-to-end fp16 still shows ~2× because
  the Torch-side preprocessing is heavier even after the model overhead
  shrinks.
- **Cold load is mostly disk + state-dict overhead, not "model
  initialization."** mlx-swift's safetensors loader is memory-mapped and
  lazy. The 10× gap shrinks when you amortize over many calls — the first
  call is what you actually pay.

Rerun on your hardware to get numbers meaningful for *your* machine.
Different chips, macOS versions, and PyTorch releases move these
numbers — don't quote out-of-context speedups.

## Notes

- **Iteration count:** the headline table uses 10 iterations for quick
  reproduction. For lower variance, use `--iters 50 --warmup 5` or higher.
- **Real image vs random.** The bench requires `--input` because real
  images exercise realistic data flow (decode + resize + colorspace
  conversion) which a random tensor would skip in `--include-load` mode.
  The model itself doesn't care — TIPS has no value-conditioned post-
  processing — but `end-to-end` numbers do.
- **fp16.** Both benches accept `--dtype fp16`. The current `TIPSPipeline`
  default is `fp32`; fp16 is plumbed but not exhaustively measured here.
