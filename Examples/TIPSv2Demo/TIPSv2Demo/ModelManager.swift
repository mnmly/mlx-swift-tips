import Foundation
import Combine
import SwiftUI
import MLXTIPS

// MARK: - Model Manager
//
// Thin presentation wrapper over the library-side `TIPSSession`. All compute +
// rendering lives in the library; this @MainActor type owns only @Published UI
// state. Heavy work is offloaded to `@concurrent nonisolated` static helpers
// (Sendable in, fresh/Sendable out) and driven from `Task {}` blocks that
// inherit MainActor isolation — so state mutations stay on the main actor
// without `[weak self]` / `MainActor.run` gymnastics.

@MainActor
final class ModelManager: ObservableObject {
    @Published var session: TIPSSession?
    @Published var isLoadingTIPS = false
    @Published var isLoadingDPT = false
    @Published var isDownloading = false
    @Published var downloadProgress: Double = 0
    @Published var statusMessage = "No model loaded"
    @Published var tipsVariantOption: TIPSVariantOption = .B
    @Published var resolution: Int = 448

    /// User-granted Hugging Face cache root (sandbox: an `NSOpenPanel`-selected
    /// folder Powerbox lets us write to). Reused across downloads + loads.
    @Published var cacheDir: URL?

    /// View-facing guards (recomputed each render; `session` is @Published).
    var tipsPipeline: TIPSPipeline? { session?.tips }
    var dptPipeline: TIPSDPTPipeline? { session?.dpt }

    private let cacheBookmarkKey = "hfCacheFolderBookmark"

    init() {
        restoreCacheBookmark()
    }

    // MARK: - Cache folder grant (security-scoped bookmark)

    /// Remember the user-granted cache folder as an app-scope security-scoped
    /// bookmark so it survives relaunch (needs the app-scope bookmarks
    /// entitlement; degrades to a per-launch grant if absent).
    func setCacheDir(_ url: URL) {
        cacheDir = url
        if let data = try? url.bookmarkData(
            options: [.withSecurityScope], includingResourceValuesForKeys: nil, relativeTo: nil) {
            UserDefaults.standard.set(data, forKey: cacheBookmarkKey)
        }
    }

    private func restoreCacheBookmark() {
        guard let data = UserDefaults.standard.data(forKey: cacheBookmarkKey) else { return }
        var stale = false
        guard let url = try? URL(
            resolvingBookmarkData: data, options: [.withSecurityScope],
            relativeTo: nil, bookmarkDataIsStale: &stale),
            url.startAccessingSecurityScopedResource()  // held for the app's lifetime
        else { return }
        cacheDir = url
        if stale { setCacheDir(url) }
    }

    // MARK: - Load (file pickers)

    func loadTIPS(directory: URL) {
        isLoadingTIPS = true
        statusMessage = "Loading TIPS model…"
        let variant = tipsVariantOption.loaderVariant
        let res = resolution
        Task {
            do {
                session = try await Self.makeSession(
                    backbone: directory, dpt: nil, variant: variant, resolution: res)
                statusMessage = "TIPS model loaded"
            } catch {
                statusMessage = "Failed: \(error.localizedDescription)"
            }
            isLoadingTIPS = false
        }
    }

    func loadDPT(dptDirectory: URL, backboneDirectory: URL) {
        isLoadingDPT = true
        statusMessage = "Loading DPT model…"
        let existing = session
        let variant = tipsVariantOption.loaderVariant
        let res = resolution
        Task {
            do {
                if let existing {
                    try await Self.attachDPT(existing, dpt: dptDirectory, backbone: backboneDirectory)
                    session = existing   // re-assign fires objectWillChange (dpt now set)
                } else {
                    session = try await Self.makeSession(
                        backbone: backboneDirectory, dpt: dptDirectory, variant: variant, resolution: res)
                }
                statusMessage = "DPT model loaded"
            } catch {
                statusMessage = "DPT load failed: \(error.localizedDescription)"
            }
            isLoadingDPT = false
        }
    }

    // MARK: - Download (HF Hub)

    /// Download a known repo into `cacheRoot` (a user-granted folder so the
    /// sandbox allows writing there) and auto-load it. For DPT repos, the
    /// matching backbone is fetched too.
    func download(repo: TIPSHub.Repo, into cacheRoot: URL) {
        isDownloading = true
        downloadProgress = 0
        cacheDir = cacheRoot
        statusMessage = "Downloading \(repo.displayName)…"
        Task {
            // Progress crosses the actor boundary via a Sendable continuation,
            // not by capturing `self` in the off-actor download closure.
            let (progress, continuation) = AsyncStream.makeStream(of: Double.self)
            let consumer = Task { for await p in progress { self.downloadProgress = p } }
            defer { consumer.cancel() }
            do {
                let urls = try await Self.resolveURLs(
                    repo: repo, cacheRoot: cacheRoot, progress: continuation)
                continuation.finish()
                downloadProgress = 1
                isDownloading = false
                tipsVariantOption = TIPSVariantOption(hubRepo: repo)
                if let dptURL = urls.dpt {
                    loadDPT(dptDirectory: dptURL, backboneDirectory: urls.backbone)
                } else {
                    loadTIPS(directory: urls.backbone)
                }
            } catch {
                continuation.finish()
                isDownloading = false
                statusMessage = "Download failed: \(error.localizedDescription)"
            }
        }
    }

    // MARK: - Tasks (offloaded compute → NSImage on the way out)

    func extractPCA(imageURL: URL) async throws -> (pca: NSImage, depthPCA: NSImage, spatial: SpatialFeatures) {
        guard let session else { throw TIPSError.noModel }
        return try await Self.computePCA(session: session, imageURL: imageURL, size: resolution)
    }

    func computeKMeans(spatial: SpatialFeatures, nClusters: Int) async throws -> NSImage {
        guard let session else { throw TIPSError.noModel }
        return try await Self.computeKMeans(session: session, spatial: spatial, nClusters: nClusters)
    }

    func zeroShotSeg(imageURL: URL, labels: [String]) async throws -> (overlay: NSImage, mask: NSImage, detected: String) {
        guard let session else { throw TIPSError.noModel }
        return try await Self.computeZeroSeg(session: session, imageURL: imageURL, labels: labels, size: resolution)
    }

    func predictDepthNormals(imageURL: URL) async throws -> (depth: NSImage, normals: NSImage) {
        guard let session, session.hasDPT else { throw TIPSError.noDPTModel }
        return try await Self.computeDepthNormals(session: session, imageURL: imageURL, size: resolution)
    }

    func predictSegmentation(imageURL: URL) async throws -> NSImage {
        guard let session, session.hasDPT else { throw TIPSError.noDPTModel }
        return try await Self.computeSegmentation(session: session, imageURL: imageURL, size: resolution)
    }

    // MARK: - Off-actor compute (concurrent executor; Sendable in, fresh out)

    @concurrent
    nonisolated static func makeSession(
        backbone: URL, dpt: URL?, variant: TIPSWeightLoader.Variant, resolution: Int
    ) async throws -> TIPSSession {
        try TIPSSession.load(.init(
            backboneDirectory: backbone, dptDirectory: dpt, variant: variant, resolution: resolution))
    }

    @concurrent
    nonisolated static func attachDPT(_ session: TIPSSession, dpt: URL, backbone: URL) async throws {
        try session.loadDPT(dptDirectory: dpt, backboneDirectory: backbone)
    }

    @concurrent
    nonisolated static func resolveURLs(
        repo: TIPSHub.Repo, cacheRoot: URL, progress: AsyncStream<Double>.Continuation
    ) async throws -> (backbone: URL, dpt: URL?) {
        let scoped = cacheRoot.startAccessingSecurityScopedResource()
        defer { if scoped { cacheRoot.stopAccessingSecurityScopedResource() } }
        let wantsDPT = repo.isDPT
        let backbone = try await TIPSHub.resolve(repo: repo.backboneRepo, cacheRoot: cacheRoot) {
            progress.yield($0 * (wantsDPT ? 0.5 : 1.0))
        }
        var dpt: URL?
        if wantsDPT {
            dpt = try await TIPSHub.resolve(repo: repo, cacheRoot: cacheRoot) {
                progress.yield(0.5 + $0 * 0.5)
            }
        }
        return (backbone, dpt)
    }

    @concurrent
    nonisolated static func computePCA(
        session: TIPSSession, imageURL: URL, size: Int
    ) async throws -> (pca: NSImage, depthPCA: NSImage, spatial: SpatialFeatures) {
        let cg = try TIPSSession.loadImage(at: imageURL)
        let (pca, pcaDepth, sp) = try session.pca(cg, size: size)
        return (ImageUtils.nsImage(pca), ImageUtils.nsImage(pcaDepth), sp)
    }

    @concurrent
    nonisolated static func computeKMeans(
        session: TIPSSession, spatial: SpatialFeatures, nClusters: Int
    ) async throws -> NSImage {
        ImageUtils.nsImage(session.kmeans(spatial, nClusters: nClusters))
    }

    @concurrent
    nonisolated static func computeZeroSeg(
        session: TIPSSession, imageURL: URL, labels: [String], size: Int
    ) async throws -> (overlay: NSImage, mask: NSImage, detected: String) {
        let cg = try TIPSSession.loadImage(at: imageURL)
        let r = try session.zeroShotSegment(cg, labels: labels, size: size)
        return (ImageUtils.nsImage(r.overlay), ImageUtils.nsImage(r.mask), r.detected)
    }

    @concurrent
    nonisolated static func computeDepthNormals(
        session: TIPSSession, imageURL: URL, size: Int
    ) async throws -> (depth: NSImage, normals: NSImage) {
        let cg = try TIPSSession.loadImage(at: imageURL)
        let (d, n) = try session.depthAndNormals(cg, size: size)
        return (ImageUtils.nsImage(d), ImageUtils.nsImage(n))
    }

    @concurrent
    nonisolated static func computeSegmentation(
        session: TIPSSession, imageURL: URL, size: Int
    ) async throws -> NSImage {
        let cg = try TIPSSession.loadImage(at: imageURL)
        return ImageUtils.nsImage(try session.segmentation(cg, size: size))
    }
}

// MARK: - Supporting types

enum TIPSError: LocalizedError {
    case noModel, noDPTModel, imageLoadFailed

    var errorDescription: String? {
        switch self {
        case .noModel:         return "No TIPS model loaded. Please load a model first."
        case .noDPTModel:      return "No DPT model loaded. Please load a DPT model first."
        case .imageLoadFailed: return "Failed to load image."
        }
    }
}
