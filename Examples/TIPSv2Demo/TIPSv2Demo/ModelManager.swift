import Foundation
import Combine
import SwiftUI
import MLXTIPS

// MARK: - Model Manager
//
// Thin presentation wrapper over the library-side `TIPSSession`. All compute +
// rendering lives in the library; this type owns only @Published UI state, the
// detached-Task hops, and CGImage→NSImage conversion. Keep it that way — if a
// new algorithm shows up here, it belongs in `TIPSSession`/`TIPSRender`.

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
        Task.detached(priority: .userInitiated) { [weak self] in
            do {
                let session = try TIPSSession.load(.init(
                    backboneDirectory: directory, variant: variant, resolution: res))
                await MainActor.run {
                    self?.session = session
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

    func loadDPT(dptDirectory: URL, backboneDirectory: URL) {
        isLoadingDPT = true
        statusMessage = "Loading DPT model…"
        let existing = session
        let variant = tipsVariantOption.loaderVariant
        let res = resolution
        Task.detached(priority: .userInitiated) { [weak self] in
            do {
                let session: TIPSSession
                if let existing {
                    try existing.loadDPT(dptDirectory: dptDirectory, backboneDirectory: backboneDirectory)
                    session = existing
                } else {
                    session = try TIPSSession.load(.init(
                        backboneDirectory: backboneDirectory, dptDirectory: dptDirectory,
                        variant: variant, resolution: res))
                }
                await MainActor.run {
                    self?.session = session  // reassign fires objectWillChange
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

    // MARK: - Download (HF Hub)

    /// Download a known repo into `cacheRoot` (a user-granted folder so the
    /// sandbox allows writing there) and auto-load it. For DPT repos, the
    /// matching backbone is fetched too.
    func download(repo: TIPSHub.Repo, into cacheRoot: URL) {
        isDownloading = true
        downloadProgress = 0
        cacheDir = cacheRoot
        statusMessage = "Downloading \(repo.displayName)…"
        let backboneRepo = repo.backboneRepo
        let wantsDPT = repo.isDPT
        Task.detached(priority: .userInitiated) { [weak self] in
            // Powerbox grants access to the panel-selected folder; keep the
            // security scope open for the whole download + load.
            let scoped = cacheRoot.startAccessingSecurityScopedResource()
            defer { if scoped { cacheRoot.stopAccessingSecurityScopedResource() } }
            do {
                let backboneURL = try await TIPSHub.resolve(repo: backboneRepo, cacheRoot: cacheRoot) { p in
                    Task { @MainActor in self?.downloadProgress = p * (wantsDPT ? 0.5 : 1.0) }
                }
                var dptURL: URL?
                if wantsDPT {
                    dptURL = try await TIPSHub.resolve(repo: repo, cacheRoot: cacheRoot) { p in
                        Task { @MainActor in self?.downloadProgress = 0.5 + p * 0.5 }
                    }
                }
                await MainActor.run {
                    guard let self else { return }
                    self.isDownloading = false
                    self.downloadProgress = 1
                    self.tipsVariantOption = TIPSVariantOption(hubRepo: repo)
                    if let dptURL {
                        self.loadDPT(dptDirectory: dptURL, backboneDirectory: backboneURL)
                    } else {
                        self.loadTIPS(directory: backboneURL)
                    }
                }
            } catch {
                await MainActor.run {
                    self?.isDownloading = false
                    self?.statusMessage = "Download failed: \(error.localizedDescription)"
                }
            }
        }
    }

    // MARK: - Tasks (delegate to TIPSSession, convert CGImage → NSImage)

    func extractPCA(imageURL: URL) async throws -> (pca: NSImage, depthPCA: NSImage, spatial: SpatialFeatures) {
        guard let session else { throw TIPSError.noModel }
        let res = resolution
        return try await Task.detached(priority: .userInitiated) {
            let cg = try TIPSSession.loadImage(at: imageURL)
            let (pca, pcaDepth, sp) = try session.pca(cg, size: res)
            return (ImageUtils.nsImage(pca), ImageUtils.nsImage(pcaDepth), sp)
        }.value
    }

    func computeKMeans(spatial: SpatialFeatures, nClusters: Int) async throws -> NSImage {
        guard let session else { throw TIPSError.noModel }
        return try await Task.detached(priority: .userInitiated) {
            ImageUtils.nsImage(session.kmeans(spatial, nClusters: nClusters))
        }.value
    }

    func zeroShotSeg(imageURL: URL, labels: [String]) async throws -> (overlay: NSImage, mask: NSImage, detected: String) {
        guard let session else { throw TIPSError.noModel }
        let res = resolution
        return try await Task.detached(priority: .userInitiated) {
            let cg = try TIPSSession.loadImage(at: imageURL)
            let r = try session.zeroShotSegment(cg, labels: labels, size: res)
            return (ImageUtils.nsImage(r.overlay), ImageUtils.nsImage(r.mask), r.detected)
        }.value
    }

    func predictDepthNormals(imageURL: URL) async throws -> (depth: NSImage, normals: NSImage) {
        guard let session, session.hasDPT else { throw TIPSError.noDPTModel }
        let res = resolution
        return try await Task.detached(priority: .userInitiated) {
            let cg = try TIPSSession.loadImage(at: imageURL)
            let (d, n) = try session.depthAndNormals(cg, size: res)
            return (ImageUtils.nsImage(d), ImageUtils.nsImage(n))
        }.value
    }

    func predictSegmentation(imageURL: URL) async throws -> NSImage {
        guard let session, session.hasDPT else { throw TIPSError.noDPTModel }
        let res = resolution
        return try await Task.detached(priority: .userInitiated) {
            let cg = try TIPSSession.loadImage(at: imageURL)
            return ImageUtils.nsImage(try session.segmentation(cg, size: res))
        }.value
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
