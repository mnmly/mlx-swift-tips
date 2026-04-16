import MLX
import MLXNN
import SentencepieceTokenizer
import Foundation

public struct TIPSWeightLoader {

    // MARK: - Vision-only

    /// Load a vision encoder from a safetensors checkpoint.
    ///
    /// Accepts either combined HF-style checkpoints (keys prefixed with
    /// `vision_encoder.`) or split pre-converted vision safetensors (no prefix).
    public static func loadVisionEncoder(
        safetensorsURL: URL,
        variant: VisionVariant = .B,
        imgSize: Int = 448,
        dtype: DType = .float32
    ) throws -> VisionTransformer {
        let model = VisionTransformer(variant: variant, imgSize: imgSize)
        let raw = try MLX.loadArrays(url: safetensorsURL)
        let params = buildVisionWeights(raw, dtype: dtype)
        try model.update(
            parameters: ModuleParameters.unflattened(params),
            verify: [.noUnusedKeys]
        )
        MLX.eval(model.parameters())
        return model
    }

    // MARK: - Full model — combined safetensors

    /// Load the full `TIPSModel` from a combined safetensors checkpoint
    /// (keys prefixed with `vision_encoder.` / `text_encoder.`).
    public static func load(
        safetensorsURL: URL,
        variant: Variant = .B,
        imgSize: Int = 448,
        dtype: DType = .float32
    ) throws -> TIPSModel {
        let vision = VisionTransformer(variant: variant.vision, imgSize: imgSize)
        let text = TextEncoder(variant: variant.text)
        let model = TIPSModel(vision: vision, text: text)

        let raw = try MLX.loadArrays(url: safetensorsURL)

        var remapped = buildVisionWeights(raw, dtype: dtype).map { (k, v) in
            ("vision_encoder." + k, v)
        }
        remapped += buildTextWeights(raw, dtype: dtype).map { (k, v) in
            ("text_encoder." + k, v)
        }

        try model.update(
            parameters: ModuleParameters.unflattened(remapped),
            verify: [.noUnusedKeys]
        )
        MLX.eval(model.parameters())
        return model
    }

    /// Load a full model from a directory containing `model.safetensors` +
    /// `tokenizer.model`.
    public static func load(
        directory: URL,
        variant: Variant = .B,
        imgSize: Int = 448,
        dtype: DType = .float32
    ) throws -> TIPSModel {
        let safetensorsURL = directory.appendingPathComponent("model.safetensors")
        let tokenizerURL = directory.appendingPathComponent("tokenizer.model")
        let model = try load(
            safetensorsURL: safetensorsURL,
            variant: variant,
            imgSize: imgSize,
            dtype: dtype
        )
        model.tokenizer = try TIPSTokenizer(modelPath: tokenizerURL.path)
        return model
    }

    // MARK: - Full model — split safetensors pair

    /// Load a full model from a pair of pre-converted safetensors files
    /// (the output of `scripts/convert_npz_to_safetensors.py`).
    ///
    /// - Parameters:
    ///   - visionSafetensorsURL: vision weights, no prefix.
    ///   - textSafetensorsURL: text weights, no prefix.
    ///   - tokenizerURL: `tokenizer.model` (SentencePiece).
    public static func load(
        visionSafetensorsURL: URL,
        textSafetensorsURL: URL,
        tokenizerURL: URL? = nil,
        variant: Variant = .B,
        imgSize: Int = 448,
        dtype: DType = .float32
    ) throws -> TIPSModel {
        let vision = VisionTransformer(variant: variant.vision, imgSize: imgSize)
        let text = TextEncoder(variant: variant.text)
        let model = TIPSModel(vision: vision, text: text)

        let visionRaw = try MLX.loadArrays(url: visionSafetensorsURL)
        let textRaw = try MLX.loadArrays(url: textSafetensorsURL)

        var remapped = buildVisionWeights(visionRaw, dtype: dtype).map { (k, v) in
            ("vision_encoder." + k, v)
        }
        remapped += buildTextWeights(textRaw, dtype: dtype).map { (k, v) in
            ("text_encoder." + k, v)
        }

        try model.update(
            parameters: ModuleParameters.unflattened(remapped),
            verify: [.noUnusedKeys]
        )
        MLX.eval(model.parameters())

        if let tokenizerURL {
            model.tokenizer = try TIPSTokenizer(modelPath: tokenizerURL.path)
        }
        return model
    }

    // MARK: - Variant

    /// Paired vision + text variant for a given TIPS release.
    public struct Variant {
        public let vision: VisionVariant
        public let text: TextVariant

        public init(vision: VisionVariant, text: TextVariant) {
            self.vision = vision
            self.text = text
        }

        public static let S      = Variant(vision: .S,      text: .S)
        public static let B      = Variant(vision: .B,      text: .B)
        public static let L      = Variant(vision: .L,      text: .L)
        public static let So400m = Variant(vision: .So400m, text: .So400m)
        public static let g      = Variant(vision: .g,      text: .g)
    }

    // MARK: - Remap helpers

    /// Remap a raw state-dict to flat vision-encoder parameter names.
    ///
    /// Accepts both formats (mirrors `_build_vision_weights` in
    /// `python/mlx-tipsv2/mlx_tipsv2.py`):
    ///   - HF combined safetensors: keys start with `vision_encoder.`
    ///   - Pre-converted split safetensors: torch state_dict, no prefix
    static func buildVisionWeights(
        _ raw: [String: MLXArray],
        dtype: DType = .float32
    ) -> [(String, MLXArray)] {
        var out: [(String, MLXArray)] = []
        out.reserveCapacity(raw.count)
        for (key, value) in raw {
            var mk = key
            if mk.hasPrefix("vision_encoder.") {
                mk = String(mk.dropFirst("vision_encoder.".count))
            } else if mk.hasPrefix("text_encoder.") || mk == "temperature" {
                continue
            }
            var v = value
            if mk == "patch_embed.proj.weight" {
                // Conv2d weight: NCHW -> NHWC (MLX is channels-last).
                v = v.transposed(0, 2, 3, 1)
            }
            out.append((mk, v.asType(dtype)))
        }
        return out
    }

    /// Remap a raw state-dict to flat text-encoder parameter names.
    ///
    /// Accepts both combined HF safetensors (keys prefixed with `text_encoder.`)
    /// and split pre-converted files (no prefix, plus a possible scalar
    /// `temperature`). Mirrors `_build_text_weights` in Python.
    static func buildTextWeights(
        _ raw: [String: MLXArray],
        dtype: DType = .float32
    ) -> [(String, MLXArray)] {
        var out: [(String, MLXArray)] = []
        out.reserveCapacity(raw.count)
        for (key, value) in raw {
            var mk = key
            if mk.hasPrefix("text_encoder.") {
                mk = String(mk.dropFirst("text_encoder.".count))
            } else if mk.hasPrefix("vision_encoder.") {
                continue
            }
            if mk == "temperature" { continue }
            mk = remapTextKey(mk)
            out.append((mk, value.asType(dtype)))
        }
        return out
    }

    /// Remap a single torch text-encoder key to our flat layout.
    static func remapTextKey(_ key: String) -> String {
        var mk = key
        mk = mk.replacingOccurrences(of: "transformer.resblocks.", with: "blocks.")
        mk = mk.replacingOccurrences(of: "attn.in_proj_weight", with: "attn.in_proj.weight")
        mk = mk.replacingOccurrences(of: "attn.in_proj_bias", with: "attn.in_proj.bias")
        mk = mk.replacingOccurrences(of: "mlp.c_fc", with: "mlp_c_fc")
        mk = mk.replacingOccurrences(of: "mlp.c_proj", with: "mlp_c_proj")
        return mk
    }

    // MARK: - DPT weight loading

    /// Remap and transpose a DPT head state-dict from PyTorch layout to MLX layout.
    ///
    /// Mirrors `_remap_dpt_weights` in `mlx_tipsv2_dpt.py`:
    /// - `Conv2d`:          (out, in, kH, kW) → (out, kH, kW, in) via `.transposed(0, 2, 3, 1)`
    /// - `ConvTranspose2d`: (in, out, kH, kW) → (out, kH, kW, in) via `.transposed(1, 2, 3, 0)`
    ///
    /// The ConvTranspose layers are identified by key substrings
    /// `resize_layers.0.weight` and `resize_layers.1.weight`.
    public static func buildDPTWeights(
        _ raw: [String: MLXArray],
        dtype: DType = .float32
    ) -> [(String, MLXArray)] {
        var out: [(String, MLXArray)] = []
        out.reserveCapacity(raw.count)
        for (key, value) in raw {
            var v = value
            if key.hasSuffix(".weight") && v.ndim == 4 {
                if key.contains(".resize_layers.0.weight") || key.contains(".resize_layers.1.weight") {
                    // ConvTranspose2d: (in, out, kH, kW) -> (out, kH, kW, in)
                    v = v.transposed(1, 2, 3, 0)
                } else {
                    // Conv2d: (out, in, kH, kW) -> (out, kH, kW, in)
                    v = v.transposed(0, 2, 3, 1)
                }
            }
            out.append((key, v.asType(dtype)))
        }
        return out
    }

    /// Load a `TIPSDPTModel` from:
    ///   - `dptSafetensorsURL`: the DPT head weights (`model.safetensors` from
    ///     e.g. `google/tipsv2-b14-dpt`)
    ///   - `backboneSafetensorsURL`: the combined backbone safetensors
    ///     (`model.safetensors` from e.g. `google/tipsv2-b14`)
    ///   - `config`: parsed DPT variant configuration
    public static func loadDPT(
        dptSafetensorsURL: URL,
        backboneSafetensorsURL: URL,
        config: DPTVariantConfig,
        imgSize: Int = 448,
        dtype: DType = .float32
    ) throws -> TIPSDPTModel {
        // Build and load backbone
        let backbone = VisionTransformer(variant: config.backboneVariant, imgSize: imgSize)
        let backboneRaw = try MLX.loadArrays(url: backboneSafetensorsURL)
        let backboneParams = buildVisionWeights(backboneRaw, dtype: dtype)
        try backbone.update(
            parameters: ModuleParameters.unflattened(backboneParams),
            verify: [.noUnusedKeys]
        )
        MLX.eval(backbone.parameters())

        // Build DPT model
        let model = TIPSDPTModel(
            backbone: backbone,
            embedDim: config.embedDim,
            channels: config.channels,
            postProcessChannels: config.postProcessChannels,
            readoutType: config.readoutType,
            numDepthBins: config.numDepthBins,
            minDepth: config.minDepth,
            maxDepth: config.maxDepth,
            numSegClasses: config.numSegClasses,
            blockIndices: config.blockIndices
        )

        // Load DPT head weights
        let dptRaw = try MLX.loadArrays(url: dptSafetensorsURL)
        let dptParams = buildDPTWeights(dptRaw, dtype: dtype)
        try model.update(
            parameters: ModuleParameters.unflattened(dptParams),
            verify: [.noUnusedKeys]
        )
        MLX.eval(model.parameters())
        return model
    }

    /// Load a `TIPSDPTModel` from a directory pair:
    ///   - `dptDirectory`:      contains `model.safetensors` + `config.json`
    ///   - `backboneDirectory`: contains `model.safetensors` (backbone)
    public static func loadDPT(
        dptDirectory: URL,
        backboneDirectory: URL,
        imgSize: Int = 448,
        dtype: DType = .float32
    ) throws -> TIPSDPTModel {
        let config = try DPTVariantConfig(directory: dptDirectory)
        return try loadDPT(
            dptSafetensorsURL: dptDirectory.appendingPathComponent("model.safetensors"),
            backboneSafetensorsURL: backboneDirectory.appendingPathComponent("model.safetensors"),
            config: config,
            imgSize: imgSize,
            dtype: dtype
        )
    }
}

// MARK: - DPTVariantConfig

/// Parsed `config.json` from a `google/tipsv2-*-dpt` HuggingFace repo.
///
/// Mirrors `DPTVariantConfig` + `_load_dpt_config` in `mlx_tipsv2_dpt.py`.
public struct DPTVariantConfig: Codable {
    public let embedDim: Int
    public let channels: Int
    public let postProcessChannels: [Int]
    public let readoutType: String
    public let numDepthBins: Int
    public let minDepth: Float
    public let maxDepth: Float
    public let numSegClasses: Int
    public let blockIndices: [Int]
    public let backboneRepo: String

    /// The `VisionVariant` inferred from `backboneRepo`
    /// (e.g. `"google/tipsv2-b14"` → `.B`).
    public var backboneVariant: VisionVariant {
        let tail = backboneRepo.components(separatedBy: "-").last?.lowercased() ?? "b14"
        if tail.hasPrefix("so") { return .So400m }
        switch tail.first {
        case "l": return .L
        case "g": return .g
        default:  return .B
        }
    }

    enum CodingKeys: String, CodingKey {
        case embedDim          = "embed_dim"
        case channels
        case postProcessChannels = "post_process_channels"
        case readoutType       = "readout_type"
        case numDepthBins      = "num_depth_bins"
        case minDepth          = "min_depth"
        case maxDepth          = "max_depth"
        case numSegClasses     = "num_seg_classes"
        case blockIndices      = "block_indices"
        case backboneRepo      = "backbone_repo"
    }

    /// Decode from a `config.json` inside `directory`.
    public init(directory: URL) throws {
        let data = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        self = try JSONDecoder().decode(DPTVariantConfig.self, from: data)
    }
}
