# ``MLXTIPS``

An MLX-Swift port of Google DeepMind's TIPSv2 image–text model, with dense
prediction (depth, normals, ADE20K segmentation) heads.

## Overview

`MLXTIPS` runs the TIPSv2 contrastive backbone and its DPT dense-prediction
heads on Apple silicon via [mlx-swift](https://github.com/ml-explore/mlx-swift).

The recommended entry point is ``TIPSSession`` — a single loaded-model handle
that owns the backbone (and optional DPT heads) and exposes every task as a
method returning a `CGImage` or a small value type. Both the `tips-cli`
executable and the SwiftUI demo app drive the *same* `TIPSSession`, so their
outputs are identical.

```swift
import MLXTIPS

// Download (or reuse a cached) checkpoint, then load a session.
let backbone = try await TIPSHub.download(repo: .b14)
let session = try TIPSSession.load(.init(backboneDirectory: backbone, variant: .B))

// Run a task; results are CGImages ready to display or write to PNG.
let cg = try TIPSSession.loadImage(at: URL(fileURLWithPath: "photo.jpg"))
let (pca, pcaDepth, _) = try session.pca(cg)
let seg = try session.zeroShotSegment(cg, labels: ["cat", "sofa", "wall"])
print(seg.detected)   // e.g. "sofa (41.2%), wall (33.0%)"
```

For depth / normals / semantic segmentation, point the session at a DPT
checkpoint too:

```swift
let session = try TIPSSession.load(.init(
    backboneDirectory: backbone, dptDirectory: dpt, variant: .B))
let (depth, normals) = try session.depthAndNormals(cg)
```

> Threading: ``TIPSSession`` is `@unchecked Sendable` with a single-writer
> contract — drive it from the CLI's main thread or one detached `Task` at a
> time, not from several concurrent callers.

## Topics

### Loading a model

- ``TIPSSession``
- ``TIPSSessionConfig``
- ``TIPSHub``

### Running tasks

- ``TIPSSession/pca(_:size:)``
- ``TIPSSession/kmeans(_:nClusters:)``
- ``TIPSSession/zeroShotSegment(_:labels:size:)``
- ``TIPSSession/depthAndNormals(_:size:)``
- ``TIPSSession/segmentation(_:size:)``
- ``ZeroSegResult``
- ``SpatialFeatures``

### Rendering results

- ``TIPSRender``

### Lower-level pipelines

- ``TIPSPipeline``
- ``TIPSDPTPipeline``
- ``TIPSPrediction``
- ``TIPSDPTPrediction``
- ``MLXTIPS``

### Model components

- ``TIPSModel``
- ``VisionTransformer``
- ``TextEncoder``
- ``TIPSDPTModel``
- ``TIPSWeightLoader``
- ``TIPSTokenizer``
