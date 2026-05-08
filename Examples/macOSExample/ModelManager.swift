import Foundation
import SwiftUI
import MLX
import MLXNN
import MLXLinalg
import MLXTIPS

// MARK: - Model Manager

@MainActor
final class ModelManager: ObservableObject {
    @Published var tipsPipeline: TIPSPipeline?
    @Published var dptPipeline: TIPSDPTPipeline?
    @Published var isLoadingTIPS = false
    @Published var isLoadingDPT = false
    @Published var statusMessage = "No model loaded"
    @Published var tipsVariantOption: TIPSVariantOption = .B
    @Published var resolution: Int = 448

    // MARK: - Load TIPS

    func loadTIPS(directory: URL) {
        isLoadingTIPS = true
        statusMessage = "Loading TIPS model…"
        let variant = tipsVariantOption.loaderVariant
        Task.detached(priority: .userInitiated) { [weak self] in
            do {
                let pipeline = try TIPSPipeline.fromPretrained(directory: directory, variant: variant)
                await MainActor.run {
                    self?.tipsPipeline = pipeline
                    self?.isLoadingTIPS = false
                    self?.statusMessage = "TIPS model loaded"
                }
            } catch {
                await MainActor.run {
                    self?.isLoadingTIPS = false
                    self?.statusMessage = "Failed: \(error.localizedDescription)"
                }
            }
        }
    }

    // MARK: - Load DPT

    func loadDPT(dptDirectory: URL, backboneDirectory: URL) {
        isLoadingDPT = true
        statusMessage = "Loading DPT model…"
        Task.detached(priority: .userInitiated) { [weak self] in
            do {
                let pipeline = try TIPSDPTPipeline.fromPretrained(
                    dptDirectory: dptDirectory,
                    backboneDirectory: backboneDirectory
                )
                await MainActor.run {
                    self?.dptPipeline = pipeline
                    self?.isLoadingDPT = false
                    self?.statusMessage = "DPT model loaded"
                }
            } catch {
                await MainActor.run {
                    self?.isLoadingDPT = false
                    self?.statusMessage = "DPT load failed: \(error.localizedDescription)"
                }
            }
        }
    }

    // MARK: - PCA

    func extractPCA(imageURL: URL) async throws -> (pca: NSImage, depthPCA: NSImage, spatial: SpatialFeatures) {
        guard let pipeline = tipsPipeline else { throw TIPSError.noModel }
        let res = resolution
        return try await Task.detached(priority: .userInitiated) {
            let cg = try TIPSPipeline.loadCGImage(at: imageURL)
            let pixels = try pipeline.preprocess(cg, size: res)
            let out = pipeline.model.encodeImage(pixels)
            let feats = out.patchTokens[0].asType(.float32)  // (N, D)
            let spatial = SpatialFeatures(feats: feats, side: res / 14)
            let pcaImg = try ImageUtils.renderPCA(spatial: spatial)
            let depthImg = try ImageUtils.renderPCADepth(spatial: spatial)
            return (pcaImg, depthImg, spatial)
        }.value
    }

    func computeKMeans(spatial: SpatialFeatures, nClusters: Int) async throws -> NSImage {
        return try await Task.detached(priority: .userInitiated) {
            try ImageUtils.renderKMeans(spatial: spatial, nClusters: nClusters)
        }.value
    }

    // MARK: - Zero-shot Segmentation

    func zeroShotSeg(imageURL: URL, labels: [String]) async throws -> (overlay: NSImage, mask: NSImage, detected: String) {
        guard let pipeline = tipsPipeline else { throw TIPSError.noModel }
        let res = resolution
        return try await Task.detached(priority: .userInitiated) {
            let cg = try TIPSPipeline.loadCGImage(at: imageURL)
            let pixels = try pipeline.preprocess(cg, size: res)
            let origImage = try ImageUtils.loadNSImage(url: imageURL)

            // Value-attention features: (1, h0, w0, D). This bypasses
            // pipeline.predict because we need the value-attention variant,
            // not the standard forward; pipeline.model exposes it.
            let featNHWC = pipeline.model.visionEncoder.encodeValueAttention(pixels)
            let h0 = featNHWC.dim(1), w0 = featNHWC.dim(2), D = featNHWC.dim(3)
            let patches = l2Normalize(featNHWC.reshaped(h0 * w0, D))  // (N, D)

            // Encode text with TCL templates
            let textFeats = try encodeWithTemplates(pipeline: pipeline, labels: labels)  // (C, D)
            eval(textFeats)

            let sim = matmul(patches, textFeats.transposed(1, 0))  // (N, C)
            let labelMap = argMax(sim, axis: -1)
            eval(labelMap)
            let labels32 = labelMap.asArray(Int32.self)

            return ImageUtils.renderSegmentation(
                labelMap: labels32,
                gridH: h0, gridW: w0,
                labels: labels,
                origImage: origImage
            )
        }.value
    }

    // MARK: - DPT Depth & Normals

    func predictDepthNormals(imageURL: URL) async throws -> (depth: NSImage, normals: NSImage) {
        guard let pipeline = dptPipeline else { throw TIPSError.noDPTModel }
        let res = resolution
        return try await Task.detached(priority: .userInitiated) {
            let cg = try TIPSPipeline.loadCGImage(at: imageURL)
            let pixels = try pipeline.preprocess(cg, size: res)
            let depth = pipeline.model.predictDepth(pixels)      // (1, 1, H, W)
            let normals = pipeline.model.predictNormals(pixels)  // (1, 3, H, W)
            eval(depth, normals)

            let depthImg = ImageUtils.renderDepthTurbo(depth: depth)
            let normalsImg = ImageUtils.renderNormals(normals: normals)
            return (depthImg, normalsImg)
        }.value
    }

    // MARK: - DPT Segmentation

    func predictSegmentation(imageURL: URL) async throws -> NSImage {
        guard let pipeline = dptPipeline else { throw TIPSError.noDPTModel }
        let res = resolution
        return try await Task.detached(priority: .userInitiated) {
            let cg = try TIPSPipeline.loadCGImage(at: imageURL)
            let pixels = try pipeline.preprocess(cg, size: res)
            let seg = pipeline.model.predictSegmentation(pixels)  // (1, C, H, W)
            eval(seg)
            return ImageUtils.renderADE20KSeg(seg: seg)
        }.value
    }
}

// MARK: - Supporting types

enum TIPSError: LocalizedError {
    case noModel, noDPTModel, imageLoadFailed

    var errorDescription: String? {
        switch self {
        case .noModel:     return "No TIPS model loaded. Please load a model first."
        case .noDPTModel:  return "No DPT model loaded. Please load a DPT model first."
        case .imageLoadFailed: return "Failed to load image."
        }
    }
}

/// Cached spatial features from a PCA extraction.
struct SpatialFeatures {
    let feats: MLXArray   // (N, D)  float32
    let side: Int         // sqrt(N)
}

// MARK: - TCL template encoding

private let tclTemplates = [
    "itap of a {}.",
    "a bad photo of a {}.",
    "a origami {}.",
    "a photo of the large {}.",
    "a {} in a video game.",
    "art of the {}.",
    "a photo of the small {}.",
    "a photo of many {}.",
    "a photo of {}s.",
]

func encodeWithTemplates(pipeline: TIPSPipeline, labels: [String]) throws -> MLXArray {
    var classFeats: [MLXArray] = []
    for label in labels {
        let texts = tclTemplates.map { $0.replacingOccurrences(of: "{}", with: label) }
        let feats = try pipeline.encodeText(texts)
        let normed = l2Normalize(feats)
        let avg = normed.mean(axis: 0, keepDims: true)
        classFeats.append(l2Normalize(avg))
    }
    return concatenated(classFeats, axis: 0)
}
