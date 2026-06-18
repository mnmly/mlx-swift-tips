import Foundation
import CoreGraphics
import MLX

// MARK: - Config

/// Configuration for a ``TIPSSession``. Carries every knob both the CLI and the
/// SwiftUI app consume, with defaults that work for either.
public struct TIPSSessionConfig: @unchecked Sendable {
    /// Backbone snapshot directory (`model.safetensors` + `tokenizer.model`,
    /// e.g. the HF snapshot for `google/tipsv2-b14`).
    public var backboneDirectory: URL
    /// Optional DPT-head snapshot directory (`google/tipsv2-*-dpt`). When set,
    /// ``TIPSSession/load(_:)`` also loads the dense-prediction heads.
    public var dptDirectory: URL?
    /// Backbone variant (must match the checkpoint).
    public var variant: TIPSWeightLoader.Variant
    /// Square input resolution in pixels (default 448). Per-call `size:`
    /// parameters override this.
    public var resolution: Int
    /// Compute dtype.
    public var dtype: DType

    public init(
        backboneDirectory: URL,
        dptDirectory: URL? = nil,
        variant: TIPSWeightLoader.Variant = .B,
        resolution: Int = 448,
        dtype: DType = .float32
    ) {
        self.backboneDirectory = backboneDirectory
        self.dptDirectory = dptDirectory
        self.variant = variant
        self.resolution = resolution
        self.dtype = dtype
    }
}

// MARK: - Results

/// Zero-shot segmentation render: a blended overlay, the bare palette mask, and
/// a human-readable summary of the dominant detected classes.
public struct ZeroSegResult: @unchecked Sendable {
    public let overlay: CGImage
    public let mask: CGImage
    public let detected: String
}

// MARK: - Session

/// The shared compute driver for TIPSv2, consumed identically by the CLI
/// (`Examples/tips-cli`) and the SwiftUI app (`Examples/macOSExample`).
///
/// All non-presentation work lives here: loading the backbone (and optional DPT
/// heads), running each task (PCA features, zero-shot segmentation, depth /
/// normals / ADE20K segmentation), and rendering results to `CGImage` via
/// ``TIPSRender``. Frontends own only argument parsing / file pickers, cadence,
/// and how `CGImage`s are surfaced.
///
/// Threading contract: single-writer-at-a-time. The CLI is single-threaded; the
/// app drives the session from one detached `Task` at a time. Hence
/// `@unchecked Sendable` — do not drive one session from two concurrent callers.
public final class TIPSSession: @unchecked Sendable {

    public let config: TIPSSessionConfig
    /// The contrastive backbone pipeline (always present).
    public let tips: TIPSPipeline
    /// The DPT dense-prediction pipeline, if loaded.
    public private(set) var dpt: TIPSDPTPipeline?

    public var hasDPT: Bool { dpt != nil }

    public enum SessionError: Swift.Error, CustomStringConvertible {
        case dptNotLoaded
        public var description: String {
            switch self {
            case .dptNotLoaded:
                return "DPT heads not loaded. Set config.dptDirectory or call loadDPT(dptDirectory:)."
            }
        }
    }

    private init(config: TIPSSessionConfig, tips: TIPSPipeline, dpt: TIPSDPTPipeline?) {
        self.config = config
        self.tips = tips
        self.dpt = dpt
    }

    // MARK: Load

    /// Load the backbone (and, if `config.dptDirectory` is set, the DPT heads).
    /// This is the expensive step; call it once and reuse the session.
    public static func load(_ config: TIPSSessionConfig) throws -> TIPSSession {
        let tips = try TIPSPipeline.fromPretrained(
            directory: config.backboneDirectory,
            variant: config.variant,
            imgSize: config.resolution,
            dtype: config.dtype
        )
        MLX.eval(tips.model.parameters())

        var dpt: TIPSDPTPipeline?
        if let dptDir = config.dptDirectory {
            dpt = try TIPSDPTPipeline.fromPretrained(
                dptDirectory: dptDir,
                backboneDirectory: config.backboneDirectory,
                imgSize: config.resolution,
                dtype: config.dtype
            )
        }
        return TIPSSession(config: config, tips: tips, dpt: dpt)
    }

    /// Attach (or replace) the DPT heads on an already-loaded session — used by
    /// the GUI's incremental "Load DPT" flow. `backboneDirectory` defaults to
    /// the session's configured backbone.
    public func loadDPT(dptDirectory: URL, backboneDirectory: URL? = nil) throws {
        dpt = try TIPSDPTPipeline.fromPretrained(
            dptDirectory: dptDirectory,
            backboneDirectory: backboneDirectory ?? config.backboneDirectory,
            imgSize: config.resolution,
            dtype: config.dtype
        )
    }

    /// Load a `CGImage` from disk (ImageIO).
    public static func loadImage(at url: URL) throws -> CGImage {
        try TIPSPipeline.loadCGImage(at: url)
    }

    // MARK: PCA / feature visualisation

    /// Extract patch-level features for one image, for downstream PCA / K-means.
    public func spatialFeatures(_ image: CGImage, size: Int? = nil) throws -> SpatialFeatures {
        let s = size ?? config.resolution
        let pixels = try tips.preprocess(image, size: s)
        let out = tips.model.encodeImage(pixels)
        let feats = out.patchTokens[0].asType(.float32)  // (N, D)
        MLX.eval(feats)
        return SpatialFeatures(feats: feats, side: s / 14)
    }

    /// One-call PCA: returns the 3-component RGB image, the 1-component inferno
    /// "depth" image, and the cached features (reuse for ``kmeans(_:nClusters:)``).
    public func pca(_ image: CGImage, size: Int? = nil) throws
        -> (pca: CGImage, pcaDepth: CGImage, spatial: SpatialFeatures)
    {
        let sp = try spatialFeatures(image, size: size)
        return (TIPSRender.pca(sp), TIPSRender.pcaDepth(sp), sp)
    }

    /// K-means cluster map over cached spatial features.
    public func kmeans(_ spatial: SpatialFeatures, nClusters: Int) -> CGImage {
        TIPSRender.kmeans(spatial, nClusters: nClusters)
    }

    // MARK: Text

    /// Tokenize + encode texts (L2 normalisation left to the caller).
    public func encodeText(_ texts: [String], maxLen: Int = 64) throws -> MLXArray {
        try tips.encodeText(texts, maxLen: maxLen)
    }

    // MARK: Zero-shot segmentation (value-attention + TCL templates)

    /// Dense zero-shot segmentation: value-attention patch features matched
    /// against TCL-templated text embeddings for each label.
    public func zeroShotSegment(_ image: CGImage, labels: [String], size: Int? = nil) throws -> ZeroSegResult {
        let s = size ?? config.resolution
        let pixels = try tips.preprocess(image, size: s)

        // Value-attention features (1, h0, w0, D): bypass the standard forward.
        let featNHWC = tips.model.visionEncoder.encodeValueAttention(pixels)
        let h0 = featNHWC.dim(1), w0 = featNHWC.dim(2), D = featNHWC.dim(3)
        let patches = l2Normalize(featNHWC.reshaped(h0 * w0, D))  // (N, D)

        let textFeats = try encodeWithTemplates(labels: labels)  // (C, D)
        let C = labels.count

        // Per-patch similarity, then bilinearly upsample the score maps to (near)
        // the original image resolution BEFORE argmax — matching the reference
        // `vis_custom_semseg`. This yields smooth, full-resolution boundaries
        // instead of a blocky `h0×w0` patch grid. Output is capped on the long
        // side to bound memory/time.
        let sim = matmul(patches, textFeats.transposed(1, 0)).reshaped(1, h0, w0, C)  // (1,h0,w0,C)
        let maxSide = 1280
        let scale = min(1.0, Double(maxSide) / Double(max(image.width, image.height)))
        let outW = max(1, Int((Double(image.width) * scale).rounded()))
        let outH = max(1, Int((Double(image.height) * scale).rounded()))
        let simUp = bilinearResize(sim, targetH: outH, targetW: outW, alignCorners: false)  // (1,H,W,C)
        let labelMap = argMax(simUp, axis: -1)  // (1,H,W)
        MLX.eval(labelMap)
        let labels32 = labelMap.reshaped(outH * outW).asArray(Int32.self)

        let mask = TIPSRender.labelMask(labelMap: labels32, gridW: outW, gridH: outH, nLabels: C)
        let overlay = TIPSRender.labelOverlay(
            base: image, labelMap: labels32, gridW: outW, gridH: outH, nLabels: C)
        let detected = TIPSSession.detectedSummary(labelMap: labels32, labels: labels)
        return ZeroSegResult(overlay: overlay, mask: mask, detected: detected)
    }

    // MARK: DPT dense prediction

    /// Depth (turbo) + normals (RGB) renders. Requires DPT heads loaded.
    public func depthAndNormals(_ image: CGImage, size: Int? = nil) throws
        -> (depth: CGImage, normals: CGImage)
    {
        guard let dpt else { throw SessionError.dptNotLoaded }
        let pixels = try dpt.preprocess(image, size: size ?? config.resolution)
        let depth = dpt.model.predictDepth(pixels)      // (1, 1, H, W)
        let normals = dpt.model.predictNormals(pixels)  // (1, 3, H, W)
        MLX.eval(depth, normals)
        return (TIPSRender.depthTurbo(depth), TIPSRender.normals(normals))
    }

    /// ADE20K semantic segmentation render. Requires DPT heads loaded.
    public func segmentation(_ image: CGImage, size: Int? = nil) throws -> CGImage {
        guard let dpt else { throw SessionError.dptNotLoaded }
        let pixels = try dpt.preprocess(image, size: size ?? config.resolution)
        let seg = dpt.model.predictSegmentation(pixels)  // (1, C, H, W)
        MLX.eval(seg)
        return TIPSRender.ade20kSeg(seg)
    }

    // MARK: - Helpers

    /// TCL prompt-ensemble templates (mirrors the Python demo).
    private static let tclTemplates = [
        "itap of a {}.", "a bad photo of a {}.", "a origami {}.",
        "a photo of the large {}.", "a {} in a video game.", "art of the {}.",
        "a photo of the small {}.", "a photo of many {}.", "a photo of {}s.",
    ]

    /// Encode each label as the L2-normalized mean of its templated prompts.
    /// Returns `(C, D)`.
    private func encodeWithTemplates(labels: [String]) throws -> MLXArray {
        var classFeats: [MLXArray] = []
        for label in labels {
            let texts = TIPSSession.tclTemplates.map { $0.replacingOccurrences(of: "{}", with: label) }
            let normed = l2Normalize(try tips.encodeText(texts))
            classFeats.append(l2Normalize(normed.mean(axis: 0, keepDims: true)))
        }
        return concatenated(classFeats, axis: 0)
    }

    /// Top dominant labels (≥2% coverage), formatted "label (xx.x%)".
    private static func detectedSummary(labelMap: [Int32], labels: [String]) -> String {
        var counts = [Int: Int]()
        for v in labelMap { counts[Int(v), default: 0] += 1 }
        let total = max(labelMap.count, 1)
        return counts.sorted { $0.value > $1.value }
            .filter { Float($0.value) / Float(total) >= 0.02 && $0.key < labels.count }
            .prefix(8)
            .map { "\(labels[$0.key]) (\(String(format: "%.1f", Float($0.value) / Float(total) * 100))%)" }
            .joined(separator: ", ")
    }
}
