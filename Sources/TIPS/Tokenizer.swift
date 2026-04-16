import MLX
import SentencepieceTokenizer
import Foundation

/// Wraps SentencepieceTokenizer to match the Python Tokenizer behaviour:
/// - lowercase input
/// - no BOS/EOS tokens
/// - pad with 0, mark padding positions as 1
public final class TIPSTokenizer {
    private let sp: SentencepieceTokenizer

    /// - Parameter modelPath: Path to `tokenizer.model` (SentencePiece model file)
    public init(modelPath: String) throws {
        // tokenOffset: 0 — TIPS uses raw SentencePiece IDs, not HuggingFace-shifted ones
        self.sp = try SentencepieceTokenizer(modelPath: modelPath, tokenOffset: 0)
    }

    /// Tokenize one or more strings.
    /// - Returns: `(ids, paddings)` both shape `(B, maxLen)` as `Int32` MLXArrays.
    ///   `paddings[b, i] == 1` means position `i` is padding for sequence `b`.
    public func tokenize(_ texts: [String], maxLen: Int = 64) throws -> (ids: MLXArray, paddings: MLXArray) {
        var tokensArr = [[Int32]](repeating: [Int32](repeating: 0, count: maxLen), count: texts.count)
        for (i, text) in texts.enumerated() {
            let ids = try sp.encode(text.lowercased())
            let n = min(ids.count, maxLen)
            for j in 0..<n {
                tokensArr[i][j] = Int32(ids[j])
            }
        }
        let flat = tokensArr.flatMap { $0 }
        let idsArray = MLXArray(flat, [texts.count, maxLen])
        let paddingsFlat = flat.map { Int32($0 == 0 ? 1 : 0) }
        let paddingsArray = MLXArray(paddingsFlat, [texts.count, maxLen])
        return (idsArray, paddingsArray)
    }

    public func tokenize(_ text: String, maxLen: Int = 64) throws -> (ids: MLXArray, paddings: MLXArray) {
        try tokenize([text], maxLen: maxLen)
    }
}
