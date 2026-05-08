# TIPSExplorer — macOS demo app

A SwiftUI macOS app that wraps the `TIPSPipeline` / `TIPSDPTPipeline`
high-level API of the parent [`mlx-swift-tipsv2`](../..) package. Use it to
visualize TIPSv2 image features and DPT dense predictions interactively.

## Features

| Tab | Pipeline | What it shows |
|---|---|---|
| PCA & Features | `TIPSPipeline` | SVD-based 3-component RGB visualization, 1st-component depth-style heatmap, and K-means clusters of patch features. |
| Zero-shot Seg | `TIPSPipeline` (value-attention) | Per-patch text-conditioned segmentation using TCL templates and the value-attention features. |
| Depth & Normals | `TIPSDPTPipeline` | Metric depth (turbo colormap) and surface normals from the DPT heads. |
| Segmentation | `TIPSDPTPipeline` | ADE20K (150-class) supervised segmentation from the DPT seg head. |

## Build & run

This example is a separate SPM package that depends on the parent package
via a relative path:

```bash
cd Examples/macOSExample
swift build -c release
.build/release/TIPSExplorer
```

You can also open it as a Swift package in Xcode and hit run. The toolbar
has buttons to load model directories at runtime — point them at your
local copy of:

- `google/tipsv2-b14` (or `-l14`, `-so400m14`, `-g14`) — the backbone +
  tokenizer (`model.safetensors` + `tokenizer.model`)
- `google/tipsv2-b14-dpt` (or matching variant) — DPT head
  (`model.safetensors` + `config.json`)

The "Variant" picker has to match whichever directory you select for TIPS;
"Resolution" controls the tensor size passed to the model (positional
embeddings interpolate at non-default sizes).

## Layout

```
Examples/macOSExample/
  Package.swift              SPM package, depends on `../..` (this repo)
  TIPSExplorerApp.swift      @main entry point
  ContentView.swift          Tabbed shell + toolbar (model load / variant)
  ModelManager.swift         @MainActor wrapper around the pipelines
  ImageUtils.swift           PCA / K-means / colormap rendering
  Views/
    PCAView.swift
    ZeroSegView.swift
    DepthNormalsView.swift
    SegmentationView.swift
```

## How the example uses the library

Loading is one line per pipeline:

```swift
let pipeline = try TIPSPipeline.fromPretrained(directory: tipsDir, variant: .B)
let dpt      = try TIPSDPTPipeline.fromPretrained(
    dptDirectory: dptDir, backboneDirectory: backboneDir
)
```

Per-image inference goes through the pipeline's `preprocess` + bare-model
hook for paths that need direct features (e.g., value-attention for
zero-shot segmentation):

```swift
let cg     = try TIPSPipeline.loadCGImage(at: imageURL)
let pixels = try pipeline.preprocess(cg, size: resolution)
let feats  = pipeline.model.visionEncoder.encodeValueAttention(pixels)
```

DPT predictions:

```swift
let cg     = try TIPSPipeline.loadCGImage(at: imageURL)
let pixels = try dpt.preprocess(cg, size: resolution)
let depth  = dpt.model.predictDepth(pixels)
```

The CGImage byte path that used to live inline in `ImageUtils.swift` was
removed — `TIPSPipeline.preprocess` now owns it, so the app and the
library share one definition.
