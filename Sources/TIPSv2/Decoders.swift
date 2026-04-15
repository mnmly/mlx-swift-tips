import MLX
import MLXNN
import Foundation

// MARK: - Identity Layer

/// A no-op layer that satisfies UnaryLayer, used as a placeholder in heterogeneous
/// layer arrays (e.g. ReassembleBlocks.resizeLayers[2]).
public final class IdentityLayer: Module, UnaryLayer {
    public override init() { super.init() }
    public func callAsFunction(_ x: MLXArray) -> MLXArray { x }
}

// MARK: - Bilinear Resize

/// Bilinear resize an NHWC tensor to (targetH, targetW).
///
/// Matches `torch.nn.functional.interpolate(mode='bilinear')` for both
/// `alignCorners=true` and `false`. Mirrors `bilinear_resize` in
/// `python/mlx-tipsv2/mlx_tipsv2_dpt.py`.
public func bilinearResize(_ x: MLXArray, targetH: Int, targetW: Int, alignCorners: Bool) -> MLXArray {
    let h = x.dim(1)
    let w = x.dim(2)
    if h == targetH && w == targetW { return x }
    let sh = Float(targetH) / Float(h)
    let sw = Float(targetW) / Float(w)
    let up = Upsample(scaleFactor: .array([sh, sw]), mode: .linear(alignCorners: alignCorners))
    return up(x)
}

// MARK: - PreActResidualConvUnit

/// Pre-activation residual unit: ReLU → Conv3x3 → ReLU → Conv3x3 + skip.
///
/// Matches `PreActResidualConvUnit` in `mlx_tipsv2_dpt.py`.
public class PreActResidualConvUnit: Module, UnaryLayer {
    @ModuleInfo(key: "conv1") var conv1: Conv2d
    @ModuleInfo(key: "conv2") var conv2: Conv2d

    public init(features: Int) {
        self._conv1.wrappedValue = Conv2d(
            inputChannels: features, outputChannels: features,
            kernelSize: .init(3), padding: .init(1), bias: false
        )
        self._conv2.wrappedValue = Conv2d(
            inputChannels: features, outputChannels: features,
            kernelSize: .init(3), padding: .init(1), bias: false
        )
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = relu(x)
        out = conv1(out)
        out = relu(out)
        out = conv2(out)
        return out + x
    }
}

// MARK: - FeatureFusionBlock

/// Feature-fusion block used in the DPT decoder.
///
/// When `hasResidual=true` (blocks 1–3), an incoming residual feature map is
/// optionally resized, processed by a `PreActResidualConvUnit`, and added to
/// the main path before the main unit. The output is upsampled 2× with
/// `alignCorners=true`, then projected via a 1×1 conv.
///
/// Note: even when `hasResidual=false` (block 0), the `residualUnit` module is
/// always allocated so weight loading remains uniform. It is never called.
public class FeatureFusionBlock: Module {
    public let hasResidual: Bool

    // Always allocated; only called when hasResidual == true.
    @ModuleInfo(key: "residual_unit") var residualUnit: PreActResidualConvUnit
    @ModuleInfo(key: "main_unit") var mainUnit: PreActResidualConvUnit
    @ModuleInfo(key: "out_conv") var outConv: Conv2d

    public init(features: Int, hasResidual: Bool = false, expand: Bool = false) {
        self.hasResidual = hasResidual
        self._residualUnit.wrappedValue = PreActResidualConvUnit(features: features)
        self._mainUnit.wrappedValue = PreActResidualConvUnit(features: features)
        let outFeatures = expand ? features / 2 : features
        self._outConv.wrappedValue = Conv2d(
            inputChannels: features, outputChannels: outFeatures,
            kernelSize: .init(1)
        )
    }

    public func callAsFunction(_ x: MLXArray, residual: MLXArray? = nil) -> MLXArray {
        var out = x
        if hasResidual, let res = residual {
            var r = res
            if r.dim(1) != out.dim(1) || r.dim(2) != out.dim(2) {
                r = bilinearResize(r, targetH: out.dim(1), targetW: out.dim(2), alignCorners: false)
            }
            r = residualUnit(r)
            out = out + r
        }
        out = mainUnit(out)
        // 2× upsample with alignCorners=true (matches Scenic reference)
        out = bilinearResize(out, targetH: out.dim(1) * 2, targetW: out.dim(2) * 2, alignCorners: true)
        out = outConv(out)
        return out
    }
}

// MARK: - ReassembleBlocks

/// Reassemble ViT intermediate features into a multi-scale feature pyramid.
///
/// Matches `ReassembleBlocks` in `mlx_tipsv2_dpt.py`:
/// - `readout_projects`: optional CLS-token projection (when `readoutType == "project"`)
/// - `out_projections`: 1×1 conv to per-level output channels
/// - `resize_layers`: [ConvTranspose k=4 s=4, ConvTranspose k=2 s=2, Identity, Conv k=3 s=2 p=1]
public class ReassembleBlocks: Module {
    public let readoutType: String

    @ModuleInfo(key: "out_projections") var outProjections: [Conv2d]
    @ModuleInfo(key: "resize_layers") var resizeLayers: [UnaryLayer]
    @ModuleInfo(key: "readout_projects") var readoutProjects: [Linear]

    public init(
        inputEmbedDim: Int = 1024,
        outChannels: [Int] = [128, 256, 512, 1024],
        readoutType: String = "project"
    ) {
        self.readoutType = readoutType

        self._outProjections.wrappedValue = outChannels.map { ch in
            Conv2d(inputChannels: inputEmbedDim, outputChannels: ch, kernelSize: .init(1))
        }

        let layers: [UnaryLayer] = [
            ConvTransposed2d(
                inputChannels: outChannels[0], outputChannels: outChannels[0],
                kernelSize: .init(4), stride: .init(4)
            ),
            ConvTransposed2d(
                inputChannels: outChannels[1], outputChannels: outChannels[1],
                kernelSize: .init(2), stride: .init(2)
            ),
            IdentityLayer(),
            Conv2d(
                inputChannels: outChannels[3], outputChannels: outChannels[3],
                kernelSize: .init(3), stride: .init(2), padding: .init(1)
            ),
        ]
        self._resizeLayers.wrappedValue = layers

        if readoutType == "project" {
            self._readoutProjects.wrappedValue = outChannels.map { _ in
                Linear(2 * inputEmbedDim, inputEmbedDim)
            }
        } else {
            self._readoutProjects.wrappedValue = []
        }
    }

    /// Process a list of `(clsToken, patchFeatures)` tuples in NHWC layout.
    ///
    /// - Parameter features: one entry per block index:
    ///   - `clsToken`:  `(B, D)` or `(B, 1, D)`
    ///   - `patches`:   `(B, H', W', D)` — already reshaped by `getIntermediateLayers`
    /// - Returns: list of per-level feature maps `(B, H'', W'', outChannels[i])`.
    public func callAsFunction(_ features: [(cls: MLXArray, patches: MLXArray)]) -> [MLXArray] {
        var out: [MLXArray] = []
        for (i, (cls, x)) in features.enumerated() {
            let b = x.dim(0), h = x.dim(1), w = x.dim(2), d = x.dim(3)
            var feat = x

            if readoutType == "project" {
                // Flatten cls to (B, D)
                let clsFlat = cls.ndim == 3 ? cls[0..., 0] : cls
                let xFlat = feat.reshaped(b, h * w, d)
                let readout = broadcast(clsFlat[0..., .newAxis, 0...], to: [b, h * w, d])
                let xCat = concatenated([xFlat, readout], axis: -1)
                let xProj = gelu(readoutProjects[i](xCat))
                feat = xProj.reshaped(b, h, w, d)
            }

            feat = outProjections[i](feat)
            feat = resizeLayers[i](feat)
            out.append(feat)
        }
        return out
    }
}

// MARK: - Shared DPT trunk helpers

// The three task heads (depth / normals / segmentation) all share the same
// trunk topology: reassemble + per-level convs + 4 fusion blocks + project conv.
// Python inlines these fields in each head class (no shared base), so we do the
// same to preserve weight-key alignment.

private func makeFusionBlocks(channels: Int) -> [FeatureFusionBlock] {
    [
        FeatureFusionBlock(features: channels, hasResidual: false),
        FeatureFusionBlock(features: channels, hasResidual: true),
        FeatureFusionBlock(features: channels, hasResidual: true),
        FeatureFusionBlock(features: channels, hasResidual: true),
    ]
}

private func runTrunk(
    features: [(cls: MLXArray, patches: MLXArray)],
    reassemble: ReassembleBlocks,
    convs: [Conv2d],
    fusionBlocks: [FeatureFusionBlock],
    project: Conv2d
) -> MLXArray {
    let multiscale = reassemble(features)
    let projected = multiscale.enumerated().map { i, feat in convs[i](feat) }

    var out = fusionBlocks[0](projected[projected.count - 1])
    for i in 1..<4 {
        out = fusionBlocks[i](out, residual: projected[projected.count - 1 - i])
    }
    return project(out)
}

// MARK: - DPTDepthHead

/// Depth estimation head.
///
/// Produces `(B, 1, H, W)` depth maps (NCHW, matching PyTorch API).
/// Internally uses a soft-binning approach over `numDepthBins` log-linear bins.
public class DPTDepthHead: Module {
    public let numDepthBins: Int
    public let minDepth: Float
    public let maxDepth: Float

    @ModuleInfo(key: "reassemble") var reassemble: ReassembleBlocks
    @ModuleInfo(key: "convs") var convs: [Conv2d]
    @ModuleInfo(key: "fusion_blocks") var fusionBlocks: [FeatureFusionBlock]
    @ModuleInfo(key: "project") var project: Conv2d
    @ModuleInfo(key: "depth_head") var depthHead: Linear

    public init(
        inputEmbedDim: Int = 1024,
        channels: Int = 256,
        postProcessChannels: [Int] = [128, 256, 512, 1024],
        readoutType: String = "project",
        numDepthBins: Int = 256,
        minDepth: Float = 1e-3,
        maxDepth: Float = 10.0
    ) {
        self.numDepthBins = numDepthBins
        self.minDepth = minDepth
        self.maxDepth = maxDepth

        self._reassemble.wrappedValue = ReassembleBlocks(
            inputEmbedDim: inputEmbedDim,
            outChannels: postProcessChannels,
            readoutType: readoutType
        )
        self._convs.wrappedValue = postProcessChannels.map { ch in
            Conv2d(inputChannels: ch, outputChannels: channels, kernelSize: .init(3), padding: .init(1), bias: false)
        }
        self._fusionBlocks.wrappedValue = makeFusionBlocks(channels: channels)
        self._project.wrappedValue = Conv2d(
            inputChannels: channels, outputChannels: channels,
            kernelSize: .init(3), padding: .init(1)
        )
        self._depthHead.wrappedValue = Linear(channels, numDepthBins)
    }

    public func callAsFunction(
        _ features: [(cls: MLXArray, patches: MLXArray)],
        imageSize: (h: Int, w: Int)? = nil
    ) -> MLXArray {
        var out = runTrunk(
            features: features, reassemble: reassemble,
            convs: convs, fusionBlocks: fusionBlocks, project: project
        )
        out = relu(out)
        out = depthHead(out)  // (B, H', W', numBins)

        let binCenters = MLXArray(
            stride(from: minDepth, through: maxDepth, by: (maxDepth - minDepth) / Float(numDepthBins - 1))
                .map { $0 },
            [numDepthBins]
        ).asType(out.dtype)

        out = relu(out) + minDepth
        let outNorm = out / sum(out, axes: [-1], keepDims: true)
        // Weighted sum over bins -> (B, H', W')
        var depth = sum(outNorm * binCenters, axes: [-1], keepDims: true)  // (B, H', W', 1)

        if let sz = imageSize {
            depth = bilinearResize(depth, targetH: sz.h, targetW: sz.w, alignCorners: false)
        }
        // Return NCHW: (B, 1, H, W)
        return depth.transposed(0, 3, 1, 2)
    }
}

// MARK: - DPTNormalsHead

/// Surface normals estimation head.
///
/// Produces `(B, 3, H, W)` unit-norm normal maps (NCHW).
public class DPTNormalsHead: Module {
    @ModuleInfo(key: "reassemble") var reassemble: ReassembleBlocks
    @ModuleInfo(key: "convs") var convs: [Conv2d]
    @ModuleInfo(key: "fusion_blocks") var fusionBlocks: [FeatureFusionBlock]
    @ModuleInfo(key: "project") var project: Conv2d
    @ModuleInfo(key: "normals_head") var normalsHead: Linear

    public init(
        inputEmbedDim: Int = 1024,
        channels: Int = 256,
        postProcessChannels: [Int] = [128, 256, 512, 1024],
        readoutType: String = "project"
    ) {
        self._reassemble.wrappedValue = ReassembleBlocks(
            inputEmbedDim: inputEmbedDim,
            outChannels: postProcessChannels,
            readoutType: readoutType
        )
        self._convs.wrappedValue = postProcessChannels.map { ch in
            Conv2d(inputChannels: ch, outputChannels: channels, kernelSize: .init(3), padding: .init(1), bias: false)
        }
        self._fusionBlocks.wrappedValue = makeFusionBlocks(channels: channels)
        self._project.wrappedValue = Conv2d(
            inputChannels: channels, outputChannels: channels,
            kernelSize: .init(3), padding: .init(1)
        )
        self._normalsHead.wrappedValue = Linear(channels, 3)
    }

    public func callAsFunction(
        _ features: [(cls: MLXArray, patches: MLXArray)],
        imageSize: (h: Int, w: Int)? = nil
    ) -> MLXArray {
        var out = runTrunk(
            features: features, reassemble: reassemble,
            convs: convs, fusionBlocks: fusionBlocks, project: project
        )
        out = normalsHead(out)  // (B, H', W', 3)
        // L2-normalize along channel axis
        out = out / (MLX.norm(out, axis: [-1], keepDims: true) + 1e-12)

        if let sz = imageSize {
            out = bilinearResize(out, targetH: sz.h, targetW: sz.w, alignCorners: false)
        }
        return out.transposed(0, 3, 1, 2)  // (B, 3, H, W)
    }
}

// MARK: - DPTSegmentationHead

/// Semantic segmentation head.
///
/// Produces `(B, numClasses, H, W)` logit maps (NCHW).
public class DPTSegmentationHead: Module {
    @ModuleInfo(key: "reassemble") var reassemble: ReassembleBlocks
    @ModuleInfo(key: "convs") var convs: [Conv2d]
    @ModuleInfo(key: "fusion_blocks") var fusionBlocks: [FeatureFusionBlock]
    @ModuleInfo(key: "project") var project: Conv2d
    @ModuleInfo(key: "segmentation_head") var segmentationHead: Linear

    public init(
        inputEmbedDim: Int = 1024,
        channels: Int = 256,
        postProcessChannels: [Int] = [128, 256, 512, 1024],
        readoutType: String = "project",
        numClasses: Int = 150
    ) {
        self._reassemble.wrappedValue = ReassembleBlocks(
            inputEmbedDim: inputEmbedDim,
            outChannels: postProcessChannels,
            readoutType: readoutType
        )
        self._convs.wrappedValue = postProcessChannels.map { ch in
            Conv2d(inputChannels: ch, outputChannels: channels, kernelSize: .init(3), padding: .init(1), bias: false)
        }
        self._fusionBlocks.wrappedValue = makeFusionBlocks(channels: channels)
        self._project.wrappedValue = Conv2d(
            inputChannels: channels, outputChannels: channels,
            kernelSize: .init(3), padding: .init(1)
        )
        self._segmentationHead.wrappedValue = Linear(channels, numClasses)
    }

    public func callAsFunction(
        _ features: [(cls: MLXArray, patches: MLXArray)],
        imageSize: (h: Int, w: Int)? = nil
    ) -> MLXArray {
        var out = runTrunk(
            features: features, reassemble: reassemble,
            convs: convs, fusionBlocks: fusionBlocks, project: project
        )
        out = segmentationHead(out)  // (B, H', W', numClasses)

        if let sz = imageSize {
            out = bilinearResize(out, targetH: sz.h, targetW: sz.w, alignCorners: false)
        }
        return out.transposed(0, 3, 1, 2)  // (B, numClasses, H, W)
    }
}

// MARK: - TIPSv2DPTOutput

public struct TIPSv2DPTOutput {
    /// `(B, 1, H, W)` — metric depth in metres
    public let depth: MLXArray
    /// `(B, 3, H, W)` — unit-length surface normals
    public let normals: MLXArray
    /// `(B, numClasses, H, W)` — per-class logits
    public let segmentation: MLXArray
}

// MARK: - TIPSv2DPTModel

/// Full TIPSv2-DPT model: a frozen `VisionTransformer` backbone plus three
/// task heads for depth, normals, and segmentation.
///
/// The backbone is **not** part of the module parameter tree (excluded from
/// `update(parameters:)`) so that DPT head weights can be loaded independently.
///
/// Mirrors `TIPSv2DPTModel` in `mlx_tipsv2_dpt.py`.
public class TIPSv2DPTModel: Module {
    /// ViT backbone — not tracked by the module system; load separately.
    public let backbone: VisionTransformer
    public let blockIndices: [Int]

    @ModuleInfo(key: "depth_head") var depthHead: DPTDepthHead
    @ModuleInfo(key: "normals_head") var normalsHead: DPTNormalsHead
    @ModuleInfo(key: "segmentation_head") var segmentationHead: DPTSegmentationHead

    public init(
        backbone: VisionTransformer,
        embedDim: Int,
        channels: Int = 256,
        postProcessChannels: [Int] = [128, 256, 512, 1024],
        readoutType: String = "project",
        numDepthBins: Int = 256,
        minDepth: Float = 1e-3,
        maxDepth: Float = 10.0,
        numSegClasses: Int = 150,
        blockIndices: [Int] = [5, 11, 17, 23]
    ) {
        self.backbone = backbone
        self.blockIndices = blockIndices

        self._depthHead.wrappedValue = DPTDepthHead(
            inputEmbedDim: embedDim, channels: channels,
            postProcessChannels: postProcessChannels, readoutType: readoutType,
            numDepthBins: numDepthBins, minDepth: minDepth, maxDepth: maxDepth
        )
        self._normalsHead.wrappedValue = DPTNormalsHead(
            inputEmbedDim: embedDim, channels: channels,
            postProcessChannels: postProcessChannels, readoutType: readoutType
        )
        self._segmentationHead.wrappedValue = DPTSegmentationHead(
            inputEmbedDim: embedDim, channels: channels,
            postProcessChannels: postProcessChannels, readoutType: readoutType,
            numClasses: numSegClasses
        )
    }

    // MARK: - Feature extraction

    private func extractIntermediate(_ pixelValues: MLXArray) -> [(cls: MLXArray, patches: MLXArray)] {
        // get_intermediate_layers with reshape=true returns (patches, cls) pairs.
        let layers = backbone.getIntermediateLayers(
            pixelValues,
            indices: blockIndices,
            reshape: true,
            returnClassToken: true,
            applyNorm: true
        )
        // Reorder to (cls, patches) to match Python's convention.
        return layers.map { layer in
            (cls: layer.classToken!, patches: layer.patchTokens)
        }
    }

    // MARK: - Predict

    public func predictDepth(_ pixelValues: MLXArray) -> MLXArray {
        let h = pixelValues.dim(1), w = pixelValues.dim(2)
        let feats = extractIntermediate(pixelValues)
        return depthHead(feats, imageSize: (h, w))
    }

    public func predictNormals(_ pixelValues: MLXArray) -> MLXArray {
        let h = pixelValues.dim(1), w = pixelValues.dim(2)
        let feats = extractIntermediate(pixelValues)
        return normalsHead(feats, imageSize: (h, w))
    }

    public func predictSegmentation(_ pixelValues: MLXArray) -> MLXArray {
        let h = pixelValues.dim(1), w = pixelValues.dim(2)
        let feats = extractIntermediate(pixelValues)
        return segmentationHead(feats, imageSize: (h, w))
    }

    public func callAsFunction(_ pixelValues: MLXArray) -> TIPSv2DPTOutput {
        let h = pixelValues.dim(1), w = pixelValues.dim(2)
        let feats = extractIntermediate(pixelValues)
        return TIPSv2DPTOutput(
            depth: depthHead(feats, imageSize: (h, w)),
            normals: normalsHead(feats, imageSize: (h, w)),
            segmentation: segmentationHead(feats, imageSize: (h, w))
        )
    }
}
