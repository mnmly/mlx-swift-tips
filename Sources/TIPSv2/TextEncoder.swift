import MLX
import MLXNN
import MLXFast
import Foundation

// MARK: - Sinusoidal positional embedding

/// Matches text_encoder.PositionalEmbedding (min=1, max=10_000).
func sinusoidalPosEmbed(seqLen: Int, dim: Int) -> MLXArray {
    let numTimescales = dim / 2
    let logInc = log(10_000.0) / Double(max(numTimescales - 1, 1))
    // inv_freq: (numTimescales,)
    let invFreqVals = (0..<numTimescales).map { Float(exp(-Double($0) * logInc)) }
    let invFreq = MLXArray(invFreqVals)
    // pos: (seqLen,)
    let pos = MLXArray(Array(0..<seqLen).map { Float($0) })
    // outer product: (seqLen, numTimescales)
    let scaled = outer(pos, invFreq)
    var signal = concatenated([sin(scaled), cos(scaled)], axis: 1)
    if dim % 2 == 1 {
        signal = concatenated([signal, MLXArray.zeros([seqLen, 1])], axis: 1)
    }
    return signal.expandedDimensions(axis: 0)  // (1, seqLen, dim)
}

// MARK: - TextAttention

public class TextAttention: Module {
    let numHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "in_proj") var inProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    public init(dim: Int, numHeads: Int) {
        self.numHeads = numHeads
        self.headDim = dim / numHeads
        self.scale = 1.0 / Float(dim / numHeads).squareRoot()
        self._inProj.wrappedValue = Linear(dim, 3 * dim)
        self._outProj.wrappedValue = Linear(dim, dim)
    }

    public func callAsFunction(_ x: MLXArray, keyPaddingMask: MLXArray) -> MLXArray {
        // x: (B, L, D); keyPaddingMask: (B, L), 1 = valid, 0 = pad
        let b = x.dim(0)
        let l = x.dim(1)
        let d = x.dim(2)
        let qkvOut = inProj(x).reshaped(b, l, 3, numHeads, headDim).transposed(2, 0, 3, 1, 4)
        let q = qkvOut[0], k = qkvOut[1], v = qkvOut[2]

        // Additive mask: 0 on valid keys, -1e9 on padded keys
        // keyPaddingMask shape (B, L) — 1 = valid; we need (B, 1, 1, L)
        let negInf = MLXArray(Float(-1e9)).asType(x.dtype)
        let zero = MLXArray(Float(0)).asType(x.dtype)
        let mask4d = which(keyPaddingMask[0..., .newAxis, .newAxis, 0...] .> 0, zero, negInf)

        let out = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: mask4d)
        return outProj(out.transposed(0, 2, 1, 3).reshaped(b, l, d))
    }
}

// MARK: - TextResBlock

public class TextResBlock: Module {
    @ModuleInfo(key: "attn") var attn: TextAttention
    @ModuleInfo(key: "ln_1") var ln1: LayerNorm
    @ModuleInfo(key: "ln_2") var ln2: LayerNorm
    @ModuleInfo(key: "mlp_c_fc") var mlpCFc: Linear
    @ModuleInfo(key: "mlp_c_proj") var mlpCProj: Linear

    public init(dim: Int, numHeads: Int, mlpDim: Int) {
        self._attn.wrappedValue = TextAttention(dim: dim, numHeads: numHeads)
        self._ln1.wrappedValue = LayerNorm(dimensions: dim)
        self._ln2.wrappedValue = LayerNorm(dimensions: dim)
        self._mlpCFc.wrappedValue = Linear(dim, mlpDim)
        self._mlpCProj.wrappedValue = Linear(mlpDim, dim)
    }

    public func callAsFunction(_ x: MLXArray, validMask: MLXArray) -> MLXArray {
        // validMask: (B, L) float — 1 for real tokens, 0 for padding
        var out = x + attn(ln1(x), keyPaddingMask: validMask)
        var h = ln2(out)
        h = relu(mlpCFc(h))
        h = h * validMask[0..., 0..., .newAxis]
        h = mlpCProj(h)
        h = h * validMask[0..., 0..., .newAxis]
        out = out + h
        return out
    }
}

// MARK: - Variant configs

/// Text encoder size presets. Mirrors `TEXT_VARIANTS` in
/// `python/mlx-tipsv2/mlx_tipsv2.py`, following
/// `tips/pytorch/run_text_encoder_inference.py:get_config`.
public enum TextVariant: String, CaseIterable {
    case S, B, L, So400m, g

    public var config: TextVariantConfig {
        switch self {
        case .S:      return .init(hiddenSize: 384,  mlpDim: 1536, numHeads: 6,  numLayers: 12)
        case .B:      return .init(hiddenSize: 768,  mlpDim: 3072, numHeads: 12, numLayers: 12)
        case .L:      return .init(hiddenSize: 1024, mlpDim: 4096, numHeads: 16, numLayers: 12)
        case .So400m: return .init(hiddenSize: 1152, mlpDim: 4304, numHeads: 16, numLayers: 27)
        case .g:      return .init(hiddenSize: 1536, mlpDim: 6144, numHeads: 24, numLayers: 12)
        }
    }
}

public struct TextVariantConfig {
    public let hiddenSize: Int
    public let mlpDim: Int
    public let numHeads: Int
    public let numLayers: Int

    public init(hiddenSize: Int, mlpDim: Int, numHeads: Int, numLayers: Int) {
        self.hiddenSize = hiddenSize
        self.mlpDim = mlpDim
        self.numHeads = numHeads
        self.numLayers = numLayers
    }
}

// MARK: - TextEncoder

public class TextEncoder: Module {
    public let dim: Int
    public let maxLen: Int

    @ModuleInfo(key: "token_embedding") var tokenEmbedding: Embedding
    @ModuleInfo(key: "blocks") var blocks: [TextResBlock]
    @ModuleInfo(key: "ln_final") var lnFinal: LayerNorm

    public init(
        vocabSize: Int = 32000,
        dim: Int = 768,
        numHeads: Int = 12,
        numLayers: Int = 12,
        mlpDim: Int = 3072,
        maxLen: Int = 64
    ) {
        self.dim = dim
        self.maxLen = maxLen
        self._tokenEmbedding.wrappedValue = Embedding(embeddingCount: vocabSize, dimensions: dim)
        self._blocks.wrappedValue = (0..<numLayers).map { _ in
            TextResBlock(dim: dim, numHeads: numHeads, mlpDim: mlpDim)
        }
        self._lnFinal.wrappedValue = LayerNorm(dimensions: dim)
    }

    /// Build a variant-sized text encoder (matches `build_text_encoder` in Python).
    public convenience init(variant: TextVariant, vocabSize: Int = 32000, maxLen: Int = 64) {
        let c = variant.config
        self.init(
            vocabSize: vocabSize,
            dim: c.hiddenSize,
            numHeads: c.numHeads,
            numLayers: c.numLayers,
            mlpDim: c.mlpDim,
            maxLen: maxLen
        )
    }

    public func callAsFunction(ids: MLXArray, paddings: MLXArray) -> MLXArray {
        // ids, paddings: (B, L); paddings == 1 on padding positions
        let l = ids.dim(1)
        let valid = (1 - paddings).asType(.float32)  // (B, L), 1 = real token

        var x = tokenEmbedding(ids) * Float(dim).squareRoot()
        x = x + sinusoidalPosEmbed(seqLen: l, dim: dim).asType(x.dtype)

        for blk in blocks {
            x = blk(x, validMask: valid)
        }
        x = lnFinal(x)

        // Average over valid tokens
        let denom = sum(valid, axes: [1], keepDims: true) + 1e-8
        let pooled = sum(x * valid[0..., 0..., .newAxis], axes: [1]) / denom
        return pooled
    }
}
