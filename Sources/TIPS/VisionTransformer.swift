import MLX
import MLXNN
import MLXFast

// MARK: - PatchEmbed

public class PatchEmbed: Module, UnaryLayer {
    @ModuleInfo(key: "proj") var proj: Conv2d

    public init(patchSize: Int = 14, inChannels: Int = 3, embedDim: Int = 768) {
        self._proj.wrappedValue = Conv2d(
            inputChannels: inChannels,
            outputChannels: embedDim,
            kernelSize: .init(patchSize),
            stride: .init(patchSize)
        )
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: (B, H, W, C)
        let out = proj(x)  // (B, Hp, Wp, D)
        let b = out.dim(0)
        let hp = out.dim(1)
        let wp = out.dim(2)
        let d = out.dim(3)
        return out.reshaped(b, hp * wp, d)
    }
}

// MARK: - Attention

public class VisionAttention: Module {
    let numHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "qkv") var qkv: Linear
    @ModuleInfo(key: "proj") var proj: Linear

    public init(dim: Int, numHeads: Int) {
        self.numHeads = numHeads
        self.headDim = dim / numHeads
        self.scale = 1.0 / Float(dim / numHeads).squareRoot()
        self._qkv.wrappedValue = Linear(dim, dim * 3)
        self._proj.wrappedValue = Linear(dim, dim)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let b = x.dim(0)
        let n = x.dim(1)
        let c = x.dim(2)
        // (B, N, 3*C) -> (B, N, 3, H, Dh) -> (3, B, H, N, Dh)
        let qkvOut = qkv(x).reshaped(b, n, 3, numHeads, headDim).transposed(2, 0, 3, 1, 4)
        let q = qkvOut[0], k = qkvOut[1], v = qkvOut[2]
        let out = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: nil)
        // (B, H, N, Dh) -> (B, N, C)
        return proj(out.transposed(0, 2, 1, 3).reshaped(b, n, c))
    }
}

// MARK: - FFN

/// FFN kind used inside a `VisionBlock`.
public enum FFNLayer {
    case mlp
    case swiglu
}

/// Shared base class so `VisionBlock` can hold either FFN under a single
/// `@ModuleInfo(key: "mlp")` field without fighting Swift's type system.
open class VisionFFNBase: Module, UnaryLayer {
    /// Subclasses must override.
    open func callAsFunction(_ x: MLXArray) -> MLXArray {
        fatalError("VisionFFNBase.callAsFunction must be overridden")
    }
}

public final class VisionMlp: VisionFFNBase {
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear

    public init(dim: Int, hiddenDim: Int) {
        self._fc1.wrappedValue = Linear(dim, hiddenDim)
        self._fc2.wrappedValue = Linear(hiddenDim, dim)
        super.init()
    }

    public override func callAsFunction(_ x: MLXArray) -> MLXArray {
        fc2(gelu(fc1(x)))
    }
}

/// SwiGLU FFN matching DINOv2's `SwiGLUFFNFused`.
///
/// Hidden size is rounded via `(int(h * 2/3) + 7) / 8 * 8`, where `h` is the
/// nominal `mlpRatio * dim` hidden size. Weight layout matches torch:
/// `w12` (in → 2·hidden) and `w3` (hidden → in).
public final class VisionSwiGLU: VisionFFNBase {
    @ModuleInfo(key: "w12") var w12: Linear
    @ModuleInfo(key: "w3") var w3: Linear

    public init(dim: Int, hiddenDim: Int) {
        let rounded = ((hiddenDim * 2 / 3) + 7) / 8 * 8
        self._w12.wrappedValue = Linear(dim, 2 * rounded)
        self._w3.wrappedValue = Linear(rounded, dim)
        super.init()
    }

    public override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let x12 = w12(x)
        let parts = split(x12, parts: 2, axis: -1)
        return w3(silu(parts[0]) * parts[1])
    }
}

// MARK: - LayerScale

public class LayerScale: Module, UnaryLayer {
    @ParameterInfo(key: "gamma") var gamma: MLXArray

    public init(dim: Int, initValues: Float = 1.0) {
        self._gamma = ParameterInfo(wrappedValue: MLXArray.ones([dim]) * initValues, key: "gamma")
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        x * gamma
    }
}

// MARK: - Block

public class VisionBlock: Module {
    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "attn") var attn: VisionAttention
    @ModuleInfo(key: "ls1") var ls1: LayerScale
    @ModuleInfo(key: "norm2") var norm2: LayerNorm
    @ModuleInfo(key: "mlp") var mlp: VisionFFNBase
    @ModuleInfo(key: "ls2") var ls2: LayerScale

    public init(
        dim: Int,
        numHeads: Int,
        mlpRatio: Float = 4.0,
        initValues: Float = 1.0,
        ffnLayer: FFNLayer = .mlp
    ) {
        self._norm1.wrappedValue = LayerNorm(dimensions: dim, eps: 1e-6)
        self._attn.wrappedValue = VisionAttention(dim: dim, numHeads: numHeads)
        self._ls1.wrappedValue = LayerScale(dim: dim, initValues: initValues)
        self._norm2.wrappedValue = LayerNorm(dimensions: dim, eps: 1e-6)
        let hidden = Int(Float(dim) * mlpRatio)
        switch ffnLayer {
        case .mlp:
            self._mlp.wrappedValue = VisionMlp(dim: dim, hiddenDim: hidden)
        case .swiglu:
            self._mlp.wrappedValue = VisionSwiGLU(dim: dim, hiddenDim: hidden)
        }
        self._ls2.wrappedValue = LayerScale(dim: dim, initValues: initValues)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = x + ls1(attn(norm1(x)))
        out = out + ls2(mlp(norm2(out)))
        return out
    }
}

// MARK: - TIPSImageOutput

public struct TIPSImageOutput {
    /// (B, 1, D) — image-level feature
    public let clsToken: MLXArray
    /// (B, R, D) — register tokens
    public let registerTokens: MLXArray
    /// (B, N, D) — spatially-aware patch features
    public let patchTokens: MLXArray
}

// MARK: - Variant configs

/// Vision encoder size presets. Mirrors `VISION_VARIANTS` in
/// `python/mlx-tipsv2/mlx_tipsv2.py`, which in turn follows
/// `tips/pytorch/image_encoder.py:vit_{small,base,large,so400m,giant2}`.
public enum VisionVariant: String, CaseIterable {
    case S, B, L, So400m, g

    public var config: VisionVariantConfig {
        switch self {
        case .S:      return .init(embedDim: 384,  depth: 12, numHeads: 6,  mlpRatio: 4.0,            ffnLayer: .mlp)
        case .B:      return .init(embedDim: 768,  depth: 12, numHeads: 12, mlpRatio: 4.0,            ffnLayer: .mlp)
        case .L:      return .init(embedDim: 1024, depth: 24, numHeads: 16, mlpRatio: 4.0,            ffnLayer: .mlp)
        case .So400m: return .init(embedDim: 1152, depth: 27, numHeads: 16, mlpRatio: 4304.0 / 1152.0, ffnLayer: .mlp)
        case .g:      return .init(embedDim: 1536, depth: 40, numHeads: 24, mlpRatio: 4.0,            ffnLayer: .swiglu)
        }
    }
}

public struct VisionVariantConfig {
    public let embedDim: Int
    public let depth: Int
    public let numHeads: Int
    public let mlpRatio: Float
    public let ffnLayer: FFNLayer
    public let numRegisterTokens: Int
    public let initValues: Float
    public let patchSize: Int

    public init(
        embedDim: Int,
        depth: Int,
        numHeads: Int,
        mlpRatio: Float,
        ffnLayer: FFNLayer,
        numRegisterTokens: Int = 1,
        initValues: Float = 1.0,
        patchSize: Int = 14
    ) {
        self.embedDim = embedDim
        self.depth = depth
        self.numHeads = numHeads
        self.mlpRatio = mlpRatio
        self.ffnLayer = ffnLayer
        self.numRegisterTokens = numRegisterTokens
        self.initValues = initValues
        self.patchSize = patchSize
    }
}

// MARK: - VisionTransformer

public class VisionTransformer: Module {
    public let imgSize: Int
    public let patchSize: Int
    public let embedDim: Int
    public let numRegisterTokens: Int
    public let numPatches: Int

    @ModuleInfo(key: "patch_embed") var patchEmbed: PatchEmbed
    @ParameterInfo(key: "cls_token") var clsToken: MLXArray
    @ParameterInfo(key: "pos_embed") var posEmbed: MLXArray
    @ParameterInfo(key: "register_tokens") var registerTokens: MLXArray
    @ParameterInfo(key: "mask_token") var maskToken: MLXArray
    @ModuleInfo(key: "blocks") var blocks: [VisionBlock]
    @ModuleInfo(key: "norm") var norm: LayerNorm

    public init(
        imgSize: Int = 448,
        patchSize: Int = 14,
        embedDim: Int = 768,
        depth: Int = 12,
        numHeads: Int = 12,
        mlpRatio: Float = 4.0,
        numRegisterTokens: Int = 1,
        initValues: Float = 1.0,
        ffnLayer: FFNLayer = .mlp
    ) {
        self.imgSize = imgSize
        self.patchSize = patchSize
        self.embedDim = embedDim
        self.numRegisterTokens = numRegisterTokens
        let np = (imgSize / patchSize) * (imgSize / patchSize)
        self.numPatches = np

        self._patchEmbed.wrappedValue = PatchEmbed(patchSize: patchSize, inChannels: 3, embedDim: embedDim)
        self._clsToken = ParameterInfo(wrappedValue: MLXArray.zeros([1, 1, embedDim]), key: "cls_token")
        self._posEmbed = ParameterInfo(wrappedValue: MLXArray.zeros([1, np + 1, embedDim]), key: "pos_embed")
        self._registerTokens = ParameterInfo(wrappedValue: MLXArray.zeros([1, numRegisterTokens, embedDim]), key: "register_tokens")
        self._maskToken = ParameterInfo(wrappedValue: MLXArray.zeros([1, embedDim]), key: "mask_token")
        self._blocks.wrappedValue = (0..<depth).map { _ in
            VisionBlock(
                dim: embedDim,
                numHeads: numHeads,
                mlpRatio: mlpRatio,
                initValues: initValues,
                ffnLayer: ffnLayer
            )
        }
        self._norm.wrappedValue = LayerNorm(dimensions: embedDim, eps: 1e-6)
    }

    /// Build a variant-sized encoder (matches `build_vision_transformer` in Python).
    public convenience init(variant: VisionVariant, imgSize: Int = 448) {
        let c = variant.config
        self.init(
            imgSize: imgSize,
            patchSize: c.patchSize,
            embedDim: c.embedDim,
            depth: c.depth,
            numHeads: c.numHeads,
            mlpRatio: c.mlpRatio,
            numRegisterTokens: c.numRegisterTokens,
            initValues: c.initValues,
            ffnLayer: c.ffnLayer
        )
    }

    /// Bilinear-interpolate positional embeddings when input grid differs from training grid.
    private func interpolatePosEncoding(h: Int, w: Int) -> MLXArray {
        let h0 = h / patchSize
        let w0 = w / patchSize
        if h0 * w0 == numPatches {
            return posEmbed
        }
        let classPosEmb = posEmbed[0..., ..<1]
        let patchPosEmb = posEmbed[0..., 1...]
        let side = Int(Double(numPatches).squareRoot())
        let d = embedDim

        // Pull to CPU for bilinear resize
        MLX.eval(patchPosEmb)
        let flat = patchPosEmb.asArray(Float.self)  // (1, side*side, d)

        // Bilinear interpolation
        var out = [Float](repeating: 0, count: h0 * w0 * d)
        for row in 0..<h0 {
            for col in 0..<w0 {
                let fy = Float(row) * Float(side - 1) / Float(h0 - 1)
                let fx = Float(col) * Float(side - 1) / Float(w0 - 1)
                let y0 = min(Int(fy), side - 1)
                let y1 = min(y0 + 1, side - 1)
                let x0 = min(Int(fx), side - 1)
                let x1 = min(x0 + 1, side - 1)
                let wy = fy - Float(y0)
                let wx = fx - Float(x0)
                for di in 0..<d {
                    let g00 = flat[(y0 * side + x0) * d + di]
                    let g01 = flat[(y0 * side + x1) * d + di]
                    let g10 = flat[(y1 * side + x0) * d + di]
                    let g11 = flat[(y1 * side + x1) * d + di]
                    let v = (1 - wy) * ((1 - wx) * g00 + wx * g01) + wy * ((1 - wx) * g10 + wx * g11)
                    out[(row * w0 + col) * d + di] = v
                }
            }
        }
        let interp = MLXArray(out, [1, h0 * w0, d])
        return concatenated([classPosEmb, interp], axis: 1)
    }

    /// Embed patches, prepend CLS, add pos-encoding, insert register tokens.
    private func prepareTokens(_ pixelValues: MLXArray) -> MLXArray {
        let b = pixelValues.dim(0)
        let h = pixelValues.dim(1)
        let w = pixelValues.dim(2)

        var x = patchEmbed(pixelValues)
        let cls = broadcast(clsToken, to: [b, 1, embedDim])
        x = concatenated([cls, x], axis: 1)
        x = x + interpolatePosEncoding(h: h, w: w)

        let reg = broadcast(registerTokens, to: [b, numRegisterTokens, embedDim])
        x = concatenated([x[0..., ..<1], reg, x[0..., 1...]], axis: 1)
        return x
    }

    public func callAsFunction(_ pixelValues: MLXArray) -> TIPSImageOutput {
        var x = prepareTokens(pixelValues)
        for blk in blocks {
            x = blk(x)
        }
        x = norm(x)

        let r = numRegisterTokens
        return TIPSImageOutput(
            clsToken: x[0..., ..<1],
            registerTokens: x[0..., 1..<(1 + r)],
            patchTokens: x[0..., (1 + r)...]
        )
    }

    /// One intermediate feature map from `getIntermediateLayers`.
    public struct IntermediateLayer {
        /// `(B, N, D)` or `(B, h0, w0, D)` when `reshape == true`.
        public let patchTokens: MLXArray
        /// `(B, D)` — the CLS token at this block. `nil` when `returnClassToken == false`.
        public let classToken: MLXArray?
    }

    /// Return features from the last `n` blocks (or from a sequence of indices).
    ///
    /// Mirrors `VisionTransformer.get_intermediate_layers` in
    /// `python/mlx-tipsv2/mlx_tipsv2.py` / `tips/pytorch/image_encoder.py`.
    /// Encode image using value embeddings from the last attention block ("values trick" from MaskCLIP).
    ///
    /// In the last block the full Q·K softmax attention is replaced by the raw V projection,
    /// keeping the residual and FFN paths intact. This produces spatially richer features
    /// that improve zero-shot dense segmentation.
    ///
    /// Mirrors `encode_image_value_attention` in `TIPS_zeroshot_segmentation.ipynb`.
    ///
    /// - Returns: `(B, h0, w0, D)` NHWC patch features.
    public func encodeValueAttention(_ pixelValues: MLXArray) -> MLXArray {
        let b  = pixelValues.dim(0)
        let h  = pixelValues.dim(1)
        let w  = pixelValues.dim(2)
        var x  = prepareTokens(pixelValues)

        for (i, blk) in blocks.enumerated() {
            if i < blocks.count - 1 {
                x = blk(x)
            } else {
                // Values trick: skip QK softmax, use V projection only as the attention branch.
                let xN = blk.norm1(x)
                let bD = xN.dim(0), nD = xN.dim(1), cD = xN.dim(2)
                // (B, N, 3*C) → (3, B, H, N, Dh)
                let qkv = blk.attn.qkv(xN)
                    .reshaped(bD, nD, 3, blk.attn.numHeads, blk.attn.headDim)
                    .transposed(2, 0, 3, 1, 4)
                let v    = qkv[2]                                           // (B, H, N, Dh)
                let vOut = blk.attn.proj(v.transposed(0, 2, 1, 3).reshaped(bD, nD, cD))
                var xVal = x + blk.ls1(vOut)
                xVal     = xVal + blk.ls2(blk.mlp(blk.norm2(xVal)))
                x = xVal
            }
        }
        x = norm(x)

        let r  = numRegisterTokens
        let h0 = h / patchSize
        let w0 = w / patchSize
        return x[0..., (1 + r)...].reshaped(b, h0, w0, embedDim)  // (B, h0, w0, D)
    }

    public func getIntermediateLayers(
        _ pixelValues: MLXArray,
        n: Int = 1,
        reshape: Bool = false,
        returnClassToken: Bool = false,
        applyNorm: Bool = true
    ) -> [IntermediateLayer] {
        let total = blocks.count
        let start = max(0, total - n)
        return getIntermediateLayers(
            pixelValues,
            indices: Array(start..<total),
            reshape: reshape,
            returnClassToken: returnClassToken,
            applyNorm: applyNorm
        )
    }

    public func getIntermediateLayers(
        _ pixelValues: MLXArray,
        indices: [Int],
        reshape: Bool = false,
        returnClassToken: Bool = false,
        applyNorm: Bool = true
    ) -> [IntermediateLayer] {
        let b = pixelValues.dim(0)
        let h = pixelValues.dim(1)
        let w = pixelValues.dim(2)
        var x = prepareTokens(pixelValues)

        let take = Set(indices)
        var collected: [(Int, MLXArray)] = []
        for (i, blk) in blocks.enumerated() {
            x = blk(x)
            if take.contains(i) {
                collected.append((i, applyNorm ? norm(x) : x))
            }
        }
        precondition(collected.count == take.count,
                     "only \(collected.count) / \(take.count) blocks collected")

        // Preserve caller ordering.
        let lookup = Dictionary(uniqueKeysWithValues: collected)
        let ordered = indices.compactMap { lookup[$0] }

        let r = numRegisterTokens
        let h0 = h / patchSize
        let w0 = w / patchSize

        return ordered.map { f in
            var patches = f[0..., (1 + r)...]
            if reshape {
                patches = patches.reshaped(b, h0, w0, patches.dim(-1))
            }
            let cls: MLXArray? = returnClassToken ? f[0..., 0] : nil
            return IntermediateLayer(patchTokens: patches, classToken: cls)
        }
    }
}
