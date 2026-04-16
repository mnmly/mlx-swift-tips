import MLX
import MLXNN
import Foundation

// MARK: - TIPSModel

public class TIPSModel: Module {
    @ModuleInfo(key: "vision_encoder") public var visionEncoder: VisionTransformer
    @ModuleInfo(key: "text_encoder") public var textEncoder: TextEncoder

    /// Optional tokenizer — populated by `TIPSWeightLoader.load(directory:)`.
    public var tokenizer: TIPSTokenizer?

    public init(vision: VisionTransformer, text: TextEncoder, tokenizer: TIPSTokenizer? = nil) {
        self._visionEncoder.wrappedValue = vision
        self._textEncoder.wrappedValue = text
        self.tokenizer = tokenizer
    }

    public func encodeImage(_ pixelValues: MLXArray) -> TIPSImageOutput {
        visionEncoder(pixelValues)
    }

    /// Encode pre-tokenized text.
    /// - Parameters:
    ///   - ids: (B, L) int32 token IDs
    ///   - paddings: (B, L) int32, 1 = padding position
    public func encodeText(ids: MLXArray, paddings: MLXArray) -> MLXArray {
        textEncoder(ids: ids, paddings: paddings)
    }

    /// Tokenize and encode strings. Requires `tokenizer` to be set.
    public func encodeText(_ texts: [String], maxLen: Int = 64) throws -> MLXArray {
        guard let tok = tokenizer else {
            fatalError("TIPSModel: tokenizer not set. Load via TIPSWeightLoader.load(directory:).")
        }
        let (ids, paddings) = try tok.tokenize(texts, maxLen: maxLen)
        return encodeText(ids: ids, paddings: paddings)
    }

    public func encodeText(_ text: String, maxLen: Int = 64) throws -> MLXArray {
        try encodeText([text], maxLen: maxLen)
    }
}

// MARK: - Utilities

public func l2Normalize(_ x: MLXArray, axis: Int = -1) -> MLXArray {
    x / (MLX.norm(x, axis: [axis], keepDims: true) + 1e-8)
}
