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
    WeightLoading.swift       TIPSWeightLoader, DPTVariantConfig
    Tokenizer.swift           TIPSTokenizer (SentencePiece, no BOS/EOS)
Tests/
  TIPSTests/
    ShapeTests.swift
    ParityTests.swift
tests/
  generate_parity_fixtures.py  ← generates Tests/TIPSTests/Fixtures/parity_fixtures.safetensors
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

### 1. Zero-shot classification

Mirrors the zero-shot cell from `TIPS_Demo.ipynb`.

```swift
import MLX
import MLXTIPS

// Load from a directory containing model.safetensors + tokenizer.model
let model = try TIPSWeightLoader.load(
    directory: URL(fileURLWithPath: "path/to/google-tipsv2-b14"),
    variant: .B
)

// Or from split safetensors files
let model = try TIPSWeightLoader.load(
    visionSafetensorsURL: URL(fileURLWithPath: "vision.safetensors"),
    textSafetensorsURL:   URL(fileURLWithPath: "text.safetensors"),
    tokenizerURL:         URL(fileURLWithPath: "tokenizer.model"),
    variant: .B
)

// Encode image — pixel values in [0, 1], NHWC (1, H, W, 3)
let pixelValues: MLXArray = ...            // load your image here
let imgOut = model.encodeImage(pixelValues)
// imgOut.clsToken       (1, 1, D)  — global image feature
// imgOut.registerTokens (1, 1, D)  — register token
// imgOut.patchTokens    (1, N, D)  — spatially-aware patch features

// Encode text
let labels = ["a cat", "a dog", "a car"]
let txtEmb = try model.encodeText(labels)  // (3, D)

// Cosine similarity
let imgFeat = l2Normalize(imgOut.clsToken.squeezed(axis: 1))  // (1, D)
let txtFeat = l2Normalize(txtEmb)                              // (3, D)
let sim = matmul(imgFeat, txtFeat.transposed(1, 0))            // (1, 3)
eval(sim)
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

### 3. DPT dense-prediction heads

Mirrors `load_tipsv2_dpt` and `model.predict_*` from `mlx_tipsv2_dpt.py`.

```swift
import MLXTIPS

// config.json + model.safetensors come from google/tipsv2-b14-dpt
let config = try DPTVariantConfig(
    directory: URL(fileURLWithPath: "path/to/google-tipsv2-b14-dpt")
)
let dptModel = try TIPSWeightLoader.loadDPT(
    dptDirectory:      URL(fileURLWithPath: "path/to/google-tipsv2-b14-dpt"),
    backboneDirectory: URL(fileURLWithPath: "path/to/google-tipsv2-b14")
)

let pixelValues: MLXArray = ...   // (1, H, W, 3), [0, 1], NHWC

// Individual heads — all return NCHW for PyTorch API parity
let depth  = dptModel.predictDepth(pixelValues)        // (1, 1, H, W)  metres
let norms  = dptModel.predictNormals(pixelValues)      // (1, 3, H, W)  unit normals
let seg    = dptModel.predictSegmentation(pixelValues) // (1, 150, H, W) logits

// Or all three at once
let out = dptModel(pixelValues)
eval(out.depth, out.normals, out.segmentation)
```

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
