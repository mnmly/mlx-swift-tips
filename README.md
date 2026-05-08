# mlx-swift-tipsv2

A Swift port of [**TIPS v2**](https://github.com/google-deepmind/tips) — Google DeepMind's contrastive vision-language model with spatially-aware patch features — built on [mlx-swift](https://github.com/ml-explore/mlx-swift) for Apple Silicon.

Supports zero-shot classification, zero-shot segmentation, PCA patch-feature visualisation, intermediate-layer extraction, and DPT dense-prediction heads (depth, surface normals, semantic segmentation).

| Variant | Vision params | Text params | FFN |
|---|---:|---:|---|
| `B` | 85.7 M | 109.6 M | MLP |
| `L` | 303.2 M | 183.9 M | MLP |
| `So400m` | 412.4 M | 448.3 M | MLP |
| `g` | 1.1 B | 389.1 M | SwiGLU |

Weights are loaded from either:
- **Hugging Face** (`google/tipsv2-b14`) — combined safetensors
- **Google Storage** — official split `.npz` files for TIPS v1 (ICLR 2025) and TIPS v2 (pre-converted to safetensors)

---

## Requirements

- macOS 14 or later
- Swift 5.9+
- [swift-sentencepiece](https://github.com/jkrukowski/swift-sentencepiece) — resolved automatically via SPM

---

## Package structure

```
Sources/
  TIPS/                ← library target (MLXTIPS)
    VisionTransformer.swift   PatchEmbed, VisionBlock, VisionTransformer
    TextEncoder.swift         TextAttention, TextResBlock, TextEncoder
    Decoders.swift            DPT heads (depth / normals / segmentation)
    TIPSModel.swift           TIPSModel wrapper, l2Normalize
    Pipeline.swift            TIPSPipeline / TIPSDPTPipeline + MLXTIPS namespace
    WeightLoading.swift       TIPSWeightLoader, DPTVariantConfig
    Tokenizer.swift           TIPSTokenizer (SentencePiece, no BOS/EOS)
Tools/
  tips-bench/          ← Swift benchmark CLI (xcodebuild scheme `tips-bench`)
Benchmarks/
  torch_tips_bench.py  ← PyTorch reference benchmark (same protocol)
  README.md            ← reproduction commands + tolerances
Tests/
  TIPSTests/
    ShapeTests.swift
    ParityTests.swift          ← per-stage parity (debugging aid)
    ParityFixtureTests.swift   ← end-to-end pipeline parity (release gate)
  generate_parity_fixtures.py  ← writes Fixtures/parity_fixtures.safetensors
                                  from the upstream PyTorch reference
```

---

## Building

```bash
swift build -c release
```

Run tests (use `xcodebuild` — `swift test` cannot load MLX's Metal library):

```bash
xcodebuild -scheme mlx-swift-tipsv2 -destination 'platform=macOS' test
```

---

## Quick start

The high-level API is `TIPSPipeline` (and `TIPSDPTPipeline` for the DPT heads).
It hides preprocessing, dtype handling, and tokenization behind a `CGImage`-in,
typed-prediction-out surface. The bare `TIPSModel` / `TIPSDPTModel` types are
still exposed for callers that need direct access to the forward pass.

### 1. Zero-shot classification (high-level)

```swift
import CoreGraphics
import MLXTIPS

// Load a pipeline from a HF snapshot directory (model.safetensors + tokenizer.model)
let pipeline = try MLXTIPS.fromPretrained(
    directory: URL(fileURLWithPath: "path/to/google-tipsv2-b14"),
    variant: .B
)

let image = try TIPSPipeline.loadCGImage(
    at: URL(fileURLWithPath: "cat.jpg")
)

// Ranked (label, cosine score) pairs
let ranking = try pipeline.zeroShotClassify(
    image, labels: ["a cat", "a dog", "a car"]
)
for (label, score) in ranking {
    print(String(format: "%+.3f  %@", score, label))
}
```

For lower-level access — for example, batching pre-decoded tensors or wiring
the model into a custom render loop — use the bare `TIPSWeightLoader` /
`TIPSModel` API:

```swift
import MLX
import MLXTIPS

let model = try TIPSWeightLoader.load(
    directory: URL(fileURLWithPath: "path/to/google-tipsv2-b14"),
    variant: .B
)

let pixelValues: MLXArray = ...              // (1, H, W, 3), [0,1], NHWC
let imgOut = model.encodeImage(pixelValues)  // .clsToken / .registerTokens / .patchTokens
let txtEmb = try model.encodeText(["a cat", "a dog"])

let imgFeat = l2Normalize(imgOut.clsToken.squeezed(axis: 1))
let sim = matmul(imgFeat, l2Normalize(txtEmb).transposed(1, 0))
eval(sim)
```

### 1b. Zero-shot segmentation (pipeline)

```swift
let seg = try pipeline.zeroShotSegment(image, labels: ["sky", "ground", "tree"])
print("\(seg.gridSide)x\(seg.gridSide) patches, \(seg.labels.count) labels")
```

### 2. Intermediate layers

Mirrors `model.vision_encoder.get_intermediate_layers(...)` from the PyTorch notebooks.

```swift
// Last 4 blocks, reshaped to (B, h, w, D), with CLS tokens
let layers = model.visionEncoder.getIntermediateLayers(
    pixelValues,
    n: 4,
    reshape: true,
    returnClassToken: true
)
for layer in layers {
    print(layer.patchTokens.shape)   // [1, 32, 32, 1024]  (NHWC)
    print(layer.classToken!.shape)   // [1, 1024]
}

// Or by explicit block indices
let layers = model.visionEncoder.getIntermediateLayers(
    pixelValues,
    indices: [5, 11, 17, 23]
)
```

> **Note:** `reshape=true` returns NHWC `(B, h, w, D)` to match MLX's channels-last convention, whereas the PyTorch original returns NCHW.

### 3. DPT dense-prediction heads (high-level)

```swift
import MLXTIPS

let dpt = try MLXTIPS.dptFromPretrained(
    dptDirectory:      URL(fileURLWithPath: "path/to/google-tipsv2-b14-dpt"),
    backboneDirectory: URL(fileURLWithPath: "path/to/google-tipsv2-b14")
)

let image = try TIPSPipeline.loadCGImage(at: URL(fileURLWithPath: "scene.jpg"))
let pred = try dpt(image)
// pred.depth        (1, 1, H, W) metres
// pred.normals      (1, 3, H, W) unit-length
// pred.segmentation (1, 150, H, W) logits
```

For the lower-level model:

```swift
let dptModel = try TIPSWeightLoader.loadDPT(
    dptDirectory:      URL(fileURLWithPath: "path/to/google-tipsv2-b14-dpt"),
    backboneDirectory: URL(fileURLWithPath: "path/to/google-tipsv2-b14")
)
let out = dptModel(pixelValues)              // TIPSDPTOutput
eval(out.depth, out.normals, out.segmentation)
```

---

## Benchmarks

Both this port and the upstream PyTorch reference are benchmarked through
`Benchmarks/torch_tips_bench.py` and `Tools/tips-bench/main.swift` with the
same iteration count, image, model variant, and dtype. See
[Benchmarks/README.md](Benchmarks/README.md) for reproduction commands.

**B14, 448 px input** — Apple M5 Max, 128 GB unified memory, macOS 26.5.
100 iterations after 10 warmup.

#### fp32

| Backend | Mode | median | mean | cold load |
|---|---|---:|---:|---:|
| PyTorch (MPS) | inference-only | 24.0 ms | 24.1 ms | 0.93 s |
| mlx-swift (Metal) | inference-only | 9.0 ms | 9.1 ms | 0.11 s |
| PyTorch (MPS) | end-to-end | 29.6 ms | 29.6 ms | — |
| mlx-swift (Metal) | end-to-end | 9.4 ms | 9.5 ms | — |

#### fp16

| Backend | Mode | median | mean | cold load |
|---|---|---:|---:|---:|
| PyTorch (MPS) | inference-only | 8.3 ms | 8.3 ms | 1.31 s |
| mlx-swift (Metal) | inference-only | 7.1 ms | 7.1 ms | 0.20 s |
| PyTorch (MPS) | end-to-end | 14.4 ms | 14.4 ms | — |
| mlx-swift (Metal) | end-to-end | 7.6 ms | 7.6 ms | — |

The fp32 gap is real but mostly reflects PyTorch MPS's slower fp32 path
(no fused SDPA in this version). At fp16 — the dtype most production
inference uses — the inference-only gap is only ~17%. End-to-end stays
wider because Swift's `CGImage` preprocessing is faster than PIL+ToTensor.
See [Benchmarks/README.md](Benchmarks/README.md) for full discussion and
reproduction commands. Rerun on your hardware before quoting speedups.

---

## CLI example (`tipsv2-example`)

The executable mirrors the three PyTorch demo notebooks:

| Notebook | `--task` | Output |
|---|---|---|
| `TIPS_Demo.ipynb` — zero-shot cell | `zeroshot` | ranked cosine scores to stdout |
| `TIPS_Demo.ipynb` — PCA cell | `pca` | `pca.png` (SVD of patch features → RGB) |
| `TIPS_Demo.ipynb` / `TIPS_zeroshot_segmentation.ipynb` | `seg` | `seg.png` (argmax label map per patch) |
| `mlx_tipsv2_dpt.py` | `dpt` | `dpt_depth.png`, `dpt_normals.png`, `dpt_seg.png` |
| Value-attention zero-shot segmentation | `vseg` | `vseg.png` (pixel-resolution cosine similarity map) |
| DINOv2-style foreground segmentation | `fg` | `fg.png` (binary foreground with median-filtered scores) |

### Build

```bash
swift build -c release
# binary at .build/release/tipsv2-example
```

### Zero-shot classification

```bash
tipsv2-example \
  --model-dir path/to/google-tipsv2-b14 \
  --task zeroshot \
  image.jpg "a cat" "a dog" "a car" "a building"
```

```
zero-shot ranking:
  +0.312  a cat
  +0.198  a building
  +0.141  a dog
  -0.021  a car
```

### PCA patch-feature heatmap

```bash
tipsv2-example \
  --model-dir path/to/google-tipsv2-b14 \
  --task pca \
  --out-dir ./out \
  image.jpg
# → out/pca.png
```

The three principal components of the patch feature matrix are mapped to RGB channels, reproducing the `dense_feature_pca` demo from `TIPS_Demo.ipynb`.

### Zero-shot dense segmentation

```bash
tipsv2-example \
  --model-dir path/to/google-tipsv2-b14 \
  --task seg \
  --out-dir ./out \
  image.jpg sky building road tree person car water
# → out/seg.png  (patch-resolution label map, coloured by class)
```

### All three tasks at once

```bash
tipsv2-example \
  --model-dir path/to/google-tipsv2-b14 \
  --task all \
  --out-dir ./out \
  image.jpg sky building road tree
```

### DPT dense prediction

```bash
tipsv2-example \
  --dpt-dir    path/to/google-tipsv2-b14-dpt \
  --backbone-dir path/to/google-tipsv2-b14 \
  --task dpt \
  --out-dir ./out \
  image.jpg
# → out/dpt_depth.png   (greyscale depth map, normalised)
# → out/dpt_normals.png (RGB surface normals, mapped from [-1,1] to [0,255])
# → out/dpt_seg.png     (150-class ADE20K segmentation, coloured)
```

### Value-attention zero-shot segmentation (`vseg`)

Encodes each label through 9 TCL (text-conditioned later) prompts, normalizes and mean-pools, then computes cosine similarity against L2-normalized patch features for the full image.

```bash
tipsv2-example \
  --model-dir path/to/google-tipsv2-b14 \
  --task vseg \
  --out-dir ./out \
  image.jpg sky person building
# → out/vseg.png
```

Sliding-window mode with configurable stride for higher resolution:

```bash
tipsv2-example \
  --model-dir path/to/google-tipsv2-b14 \
  --task vseg \
  --slide \
  --stride 384 \
  --templates \
  --out-dir ./out \
  image.jpg sky person building
```

The `--templates` flag uses the 9 TCL prompt templates for richer text encoding.

### Foreground segmentation (`fg`)

Mirrors the DINOv2 foreground segmentation notebook. Loads training images with masks, extracts last-block normalized features, trains an L2-regularized logistic regression via MLX gradient descent, then runs inference with a 3×3 median filter.

```bash
tipsv2-example \
  --model-dir path/to/google-tipsv2-b14 \
  --task fg \
  --train-dir ./train_imgs \
  --mask-dir ./masks \
  --fg-c 0.1 \
  --out-dir ./out \
  image.jpg
# → out/fg.png
```

### Split safetensors files

```bash
tipsv2-example \
  --split vision.safetensors text.safetensors tokenizer.model \
  --variant L \
  --task zeroshot \
  image.jpg "a landscape" "a portrait"
```

### All CLI options

```
tipsv2-example [options] <image> [<label> ...]

--model-dir   <dir>          Directory with model.safetensors + tokenizer.model
--split       <v> <t> <tok>  Three separate files (vision ST, text ST, tokenizer)
--dpt-dir     <dir>          DPT head directory (requires --backbone-dir)
--backbone-dir <dir>         Backbone directory (used with --dpt-dir)
--variant     B|L|So400m|g   Model size (default: B)
--task        zeroshot|pca|seg|dpt|all|vseg|fg  (default: zeroshot)
--out-dir     <dir>          Where to write output images (default: .)
--img-size    <n>            Resize input to n×n pixels (default: 448)
--stride      <n>           Sliding window stride for vseg (default: 384)
--slide                         Enable sliding-window mode for vseg
--templates                     Use TCL prompt templates for vseg text encoding
--train-dir   <dir>          Training images directory for fg task
--mask-dir    <dir>          Mask images directory for fg task
--fg-c        <f>            L2 regularization strength for fg logistic regression (default: 0.1)
```

---

## Weight loading details

### Vision encoder

```swift
// Combined HF safetensors (keys prefixed with vision_encoder.)
let vision = try TIPSWeightLoader.loadVisionEncoder(
    safetensorsURL: URL(fileURLWithPath: "model.safetensors"),
    variant: .L,
    imgSize: 448
)

// Pre-converted split file (no prefix)
let raw = try MLX.loadArrays(url: splitURL)
let params = TIPSWeightLoader.buildVisionWeights(raw)
try vision.update(parameters: ModuleParameters.unflattened(params), verify: [.noUnusedKeys])
```

`Conv2d` kernels are transposed NCHW → NHWC automatically.

### DPT heads

```swift
let raw = try MLX.loadArrays(url: dptSafetensorsURL)
let params = TIPSWeightLoader.buildDPTWeights(raw)
// ConvTranspose2d kernels: (in, out, kH, kW) → (out, kH, kW, in)
// Conv2d kernels:          (out, in, kH, kW) → (out, kH, kW, in)
```


### Check parity against mlx-tips

Numerical parity against [mlx-tips](https://github.com/mnmly/mlx-tips) (the Python MLX reference implementation).

**1. Generate fixtures** (run from the Python package root):

```sh
cd /path/to/mlx-tips
uv run python /path/to/mlx-swift-tipsv2/tests/generate_parity_fixtures.py
```

**2. Run parity tests:**

```sh
TIPS_SNAPSHOT_DIR=/path/to/snapshots/245de45... swift test --filter ParityTests
```

Expected output (all cosine similarities should be ≥ 0.9999, max_err ≤ 0.02):

```
[cls_token]        max_err=4.20e-04  cosine=1.000000
[register_tokens]  max_err=2.70e-04  cosine=1.000000
[patch_tokens]     max_err=6.91e-03  cosine=1.000000
[text_embeddings]  max_err=4.39e-04  cosine=1.000000
```

---

## Implementation notes

- **Vision tower**: DINOv2-style ViT with 1 register token, LayerScale (`initValues=1.0`), MLP or SwiGLU FFN (fused, with DINOv2 hidden-size rounding `(int(h·2/3)+7)//8·8`).
- **Text tower**: Transformer with sinusoidal positional embeddings, ReLU MLP with padding-mask-gated FFN, mean-pooling over unpadded tokens, SentencePiece tokenizer (lowercased, no BOS/EOS).
- **DPT heads**: `ReassembleBlocks` → per-level `Conv2d` → 4× `FeatureFusionBlock` (2× bilinear upsample per block) → task-specific linear head. Bilinear resize uses MLX `Upsample` with explicit per-axis scale, matching `align_corners=True` (fusion upsample) and `align_corners=False` (head resize to input resolution).
- **Attention**: `MLXFast.scaledDotProductAttention` throughout.
- **Weight layout**: all convolutions stored NHWC (channels-last). `ConvTranspose2d` kernels follow the same convention `(out, kH, kW, in)`.
- **Lazy evaluation**: both model initialisation and forward passes are lazy; call `eval(...)` only when you need materialised values.

---

## License

This Swift port is released under the **Apache License 2.0** — see [`LICENSE`](LICENSE) below — matching the upstream model's license.

The upstream TIPS weights and reference PyTorch code are Copyright 2025 Google DeepMind / Google LLC, also licensed under Apache-2.0. This repository does **not** redistribute the weights; they are downloaded from Hugging Face or Google Storage on first use.

See the [upstream repository](https://github.com/google-deepmind/tips) and the [HF model card](https://huggingface.co/google/tipsv2-b14) for full terms and intended use.
