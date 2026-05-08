import Foundation
import CoreGraphics
import ImageIO
import MLX
import MLXNN

// MARK: - Errors

public enum TIPSPipelineError: Error, LocalizedError {
    case imageDecodeFailed
    case cgContextFailed
    case missingTokenizer
    case noLabels

    public var errorDescription: String? {
        switch self {
        case .imageDecodeFailed: return "Could not decode CGImage."
        case .cgContextFailed:   return "Could not allocate a CGContext for preprocessing."
        case .missingTokenizer:  return "Pipeline tokenizer is not set. Load via TIPSPipeline.fromPretrained(directory:)."
        case .noLabels:          return "Empty labels array."
        }
    }
}

// MARK: - TIPSPrediction

/// Output of `TIPSPipeline.callAsFunction(_:)`.
public struct TIPSPrediction {
    /// Backbone outputs keyed by name. Keys: `cls_token`, `register_tokens`, `patch_tokens`.
    public let outputs: [String: MLXArray]
    public let sourceWidth: Int
    public let sourceHeight: Int
    public let processedSize: Int

    public var clsToken: MLXArray?       { outputs["cls_token"] }
    public var registerTokens: MLXArray? { outputs["register_tokens"] }
    public var patchTokens: MLXArray?    { outputs["patch_tokens"] }

    /// Side length (in patches) of the square patch grid.
    public var patchGridSide: Int? {
        guard let pt = patchTokens else { return nil }
        return Int(Double(pt.dim(1)).squareRoot().rounded())
    }
}

// MARK: - TIPSPipeline

/// High-level pipeline for the contrastive TIPS backbone.
///
/// Wraps `TIPSModel` with `CGImage` preprocessing, an `[String: MLXArray]` predict
/// surface, and convenience operations (text encoding, zero-shot classification,
/// zero-shot segmentation).
///
/// The bare `TIPSModel` remains available as `pipeline.model` for callers that
/// need the lower-level forward pass.
public struct TIPSPipeline {
    public let model: TIPSModel
    public let dtype: DType
    public let imgSize: Int

    public init(model: TIPSModel, dtype: DType = .float32, imgSize: Int = 448) {
        self.model = model
        self.dtype = dtype
        self.imgSize = imgSize
    }

    // MARK: Preprocess

    /// Decode a `CGImage` to the NHWC float tensor the model expects.
    ///
    /// - Resizes to `size × size` (default: pipeline `imgSize`, typically 448).
    /// - Outputs shape `(1, size, size, 3)`, values in `[0, 1]`.
    /// - dtype matches the pipeline's `dtype`.
    ///
    /// Mirrors the byte path used across the demo notebooks: sRGB device RGB,
    /// `noneSkipLast` alpha, `pixel / 255` float conversion.
    public func preprocess(_ image: CGImage, size: Int? = nil) throws -> MLXArray {
        try TIPSPipeline.preprocess(image, size: size ?? imgSize, dtype: dtype)
    }

    /// Stateless byte path. Used by both pipelines.
    public static func preprocess(_ image: CGImage, size: Int, dtype: DType) throws -> MLXArray {
        let cs = CGColorSpaceCreateDeviceRGB()
        var buf = [UInt8](repeating: 0, count: size * size * 4)
        guard let ctx = CGContext(
            data: &buf, width: size, height: size,
            bitsPerComponent: 8, bytesPerRow: size * 4,
            space: cs, bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
        ) else { throw TIPSPipelineError.cgContextFailed }
        ctx.interpolationQuality = .high
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: size, height: size))

        var floats = [Float](repeating: 0, count: size * size * 3)
        var o = 0
        for i in stride(from: 0, to: buf.count, by: 4) {
            floats[o]     = Float(buf[i])     / 255
            floats[o + 1] = Float(buf[i + 1]) / 255
            floats[o + 2] = Float(buf[i + 2]) / 255
            o += 3
        }
        let arr = MLXArray(floats, [1, size, size, 3])
        return dtype == .float32 ? arr : arr.asType(dtype)
    }

    /// Convenience: load a `CGImage` from disk via ImageIO.
    public static func loadCGImage(at url: URL) throws -> CGImage {
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
              let cg  = CGImageSourceCreateImageAtIndex(src, 0, nil)
        else { throw TIPSPipelineError.imageDecodeFailed }
        return cg
    }

    // MARK: Predict

    /// Run the backbone on a preprocessed tensor and return all three feature
    /// maps. Includes `eval` so the returned arrays are materialised.
    public func predict(_ pixelValues: MLXArray) -> [String: MLXArray] {
        let out = model.encodeImage(pixelValues)
        MLX.eval(out.clsToken, out.registerTokens, out.patchTokens)
        return [
            "cls_token":       out.clsToken,
            "register_tokens": out.registerTokens,
            "patch_tokens":    out.patchTokens,
        ]
    }

    /// One-call convenience: `CGImage` → `TIPSPrediction`.
    public func callAsFunction(_ image: CGImage, size: Int? = nil) throws -> TIPSPrediction {
        let s = size ?? imgSize
        let pixels = try preprocess(image, size: s)
        let outputs = predict(pixels)
        return TIPSPrediction(
            outputs: outputs,
            sourceWidth: image.width,
            sourceHeight: image.height,
            processedSize: s
        )
    }

    // MARK: Text

    /// Tokenize and encode a batch of texts. Requires the pipeline's model to
    /// have a tokenizer (true when loaded via `fromPretrained(directory:)`).
    public func encodeText(_ texts: [String], maxLen: Int = 64) throws -> MLXArray {
        guard model.tokenizer != nil else { throw TIPSPipelineError.missingTokenizer }
        let out = try model.encodeText(texts, maxLen: maxLen)
        MLX.eval(out)
        return out
    }

    // MARK: Zero-shot classification

    /// Returns labels with their cosine similarity to the image CLS token,
    /// sorted descending.
    public func zeroShotClassify(_ image: CGImage, labels: [String], size: Int? = nil) throws
        -> [(label: String, score: Float)]
    {
        guard !labels.isEmpty else { throw TIPSPipelineError.noLabels }
        let prediction = try self(image, size: size)
        guard let cls = prediction.clsToken else { return [] }
        let imgFeat = l2Normalize(cls.squeezed(axis: 1))
        let txtFeat = l2Normalize(try encodeText(labels))
        let sim = matmul(imgFeat, txtFeat.transposed(1, 0))[0]
        MLX.eval(sim)
        let scores = sim.asArray(Float.self)
        return scores.indices
            .sorted { scores[$0] > scores[$1] }
            .map { (labels[$0], scores[$0]) }
    }

    // MARK: Zero-shot segmentation

    public struct ZeroShotSegmentation {
        /// Patch grid side (e.g., 32 for a 448-px image at patch 14).
        public let gridSide: Int
        /// Per-patch class indices, row-major, length `gridSide * gridSide`.
        public let labels: [Int32]
        /// `(N, C)` cosine similarity matrix.
        public let logits: MLXArray
    }

    /// Per-patch zero-shot segmentation against `labels`. Uses cosine
    /// similarity between l2-normalized patch tokens and label embeddings.
    public func zeroShotSegment(_ image: CGImage, labels: [String], size: Int? = nil) throws
        -> ZeroShotSegmentation
    {
        guard !labels.isEmpty else { throw TIPSPipelineError.noLabels }
        let prediction = try self(image, size: size)
        guard let patchTokens = prediction.patchTokens else {
            return .init(gridSide: 0, labels: [], logits: MLXArray.zeros([0, 0]))
        }
        let patches = l2Normalize(patchTokens[0])
        let txtFeat = l2Normalize(try encodeText(labels))
        let logits = matmul(patches, txtFeat.transposed(1, 0))
        let argmax = argMax(logits, axis: -1)
        MLX.eval(logits, argmax)

        let n = patches.dim(0)
        let side = Int(Double(n).squareRoot().rounded())
        return .init(gridSide: side, labels: argmax.asArray(Int32.self), logits: logits)
    }
}

// MARK: - fromPretrained

public extension TIPSPipeline {
    /// Load a pipeline from a directory containing `model.safetensors` and
    /// `tokenizer.model` (e.g., the HF snapshot for `google/tipsv2-b14`).
    static func fromPretrained(
        directory: URL,
        variant: TIPSWeightLoader.Variant = .B,
        imgSize: Int = 448,
        dtype: DType = .float32
    ) throws -> TIPSPipeline {
        let model = try TIPSWeightLoader.load(
            directory: directory, variant: variant, imgSize: imgSize, dtype: dtype
        )
        return TIPSPipeline(model: model, dtype: dtype, imgSize: imgSize)
    }

    /// Load a pipeline from a combined safetensors file.
    static func fromPretrained(
        safetensorsURL: URL,
        variant: TIPSWeightLoader.Variant = .B,
        imgSize: Int = 448,
        dtype: DType = .float32
    ) throws -> TIPSPipeline {
        let model = try TIPSWeightLoader.load(
            safetensorsURL: safetensorsURL, variant: variant, imgSize: imgSize, dtype: dtype
        )
        return TIPSPipeline(model: model, dtype: dtype, imgSize: imgSize)
    }

    /// Load a pipeline from split vision/text safetensors plus a
    /// SentencePiece tokenizer file.
    static func fromPretrained(
        visionSafetensorsURL: URL,
        textSafetensorsURL: URL,
        tokenizerURL: URL,
        variant: TIPSWeightLoader.Variant = .B,
        imgSize: Int = 448,
        dtype: DType = .float32
    ) throws -> TIPSPipeline {
        let model = try TIPSWeightLoader.load(
            visionSafetensorsURL: visionSafetensorsURL,
            textSafetensorsURL: textSafetensorsURL,
            tokenizerURL: tokenizerURL,
            variant: variant,
            imgSize: imgSize,
            dtype: dtype
        )
        return TIPSPipeline(model: model, dtype: dtype, imgSize: imgSize)
    }
}

// MARK: - TIPSDPTPrediction

public struct TIPSDPTPrediction {
    /// DPT head outputs keyed by name. Keys: `depth`, `normals`, `segmentation`.
    public let outputs: [String: MLXArray]
    public let sourceWidth: Int
    public let sourceHeight: Int
    public let processedSize: Int

    public var depth: MLXArray?        { outputs["depth"] }
    public var normals: MLXArray?      { outputs["normals"] }
    public var segmentation: MLXArray? { outputs["segmentation"] }
}

// MARK: - TIPSDPTPipeline

/// High-level pipeline for the DPT dense-prediction heads (depth / normals /
/// segmentation) on top of the TIPS backbone.
public struct TIPSDPTPipeline {
    public let model: TIPSDPTModel
    public let dtype: DType
    public let imgSize: Int

    public init(model: TIPSDPTModel, dtype: DType = .float32, imgSize: Int = 448) {
        self.model = model
        self.dtype = dtype
        self.imgSize = imgSize
    }

    public func preprocess(_ image: CGImage, size: Int? = nil) throws -> MLXArray {
        try TIPSPipeline.preprocess(image, size: size ?? imgSize, dtype: dtype)
    }

    /// Run all three heads and return `[String: MLXArray]`. Includes `eval`.
    public func predict(_ pixelValues: MLXArray) -> [String: MLXArray] {
        let out = model(pixelValues)
        MLX.eval(out.depth, out.normals, out.segmentation)
        return [
            "depth":        out.depth,
            "normals":      out.normals,
            "segmentation": out.segmentation,
        ]
    }

    public func callAsFunction(_ image: CGImage, size: Int? = nil) throws -> TIPSDPTPrediction {
        let s = size ?? imgSize
        let pixels = try preprocess(image, size: s)
        let outputs = predict(pixels)
        return TIPSDPTPrediction(
            outputs: outputs,
            sourceWidth: image.width,
            sourceHeight: image.height,
            processedSize: s
        )
    }
}

public extension TIPSDPTPipeline {
    /// Load from a DPT-head directory plus a backbone directory (the standard
    /// `google/tipsv2-*-dpt` + `google/tipsv2-*` HF snapshot pair).
    static func fromPretrained(
        dptDirectory: URL,
        backboneDirectory: URL,
        imgSize: Int = 448,
        dtype: DType = .float32
    ) throws -> TIPSDPTPipeline {
        let model = try TIPSWeightLoader.loadDPT(
            dptDirectory: dptDirectory,
            backboneDirectory: backboneDirectory,
            imgSize: imgSize,
            dtype: dtype
        )
        return TIPSDPTPipeline(model: model, dtype: dtype, imgSize: imgSize)
    }
}

// MARK: - MLXTIPS namespace

/// One-line convenience namespace for loading pipelines.
public enum MLXTIPS {
    /// Backbone pipeline from a HF snapshot directory.
    public static func fromPretrained(
        directory: URL,
        variant: TIPSWeightLoader.Variant = .B,
        imgSize: Int = 448,
        dtype: DType = .float32
    ) throws -> TIPSPipeline {
        try TIPSPipeline.fromPretrained(
            directory: directory, variant: variant, imgSize: imgSize, dtype: dtype
        )
    }

    /// DPT pipeline from a paired HF snapshot directory layout.
    public static func dptFromPretrained(
        dptDirectory: URL,
        backboneDirectory: URL,
        imgSize: Int = 448,
        dtype: DType = .float32
    ) throws -> TIPSDPTPipeline {
        try TIPSDPTPipeline.fromPretrained(
            dptDirectory: dptDirectory,
            backboneDirectory: backboneDirectory,
            imgSize: imgSize,
            dtype: dtype
        )
    }
}
