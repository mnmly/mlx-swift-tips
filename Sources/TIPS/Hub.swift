import Foundation
import Hub

/// Downloads TIPSv2 checkpoints from Hugging Face into the shared HF cache,
/// matching the layout `TIPSSession`/`TIPSPipeline` load from
/// (`~/.cache/huggingface/hub/models--google--…/snapshots/<rev>`).
///
/// Used by the GUI's "Download" button and the `tips-cli download` command, so
/// both frontends fetch identical files into the same cache.
public enum TIPSHub {

    /// Known `google/tipsv2-*` repositories.
    public enum Repo: String, CaseIterable, Sendable {
        case b14 = "google/tipsv2-b14"
        case l14 = "google/tipsv2-l14"
        case so400m14 = "google/tipsv2-so400m14"
        case g14 = "google/tipsv2-g14"
        case b14dpt = "google/tipsv2-b14-dpt"
        case l14dpt = "google/tipsv2-l14-dpt"
        case so400m14dpt = "google/tipsv2-so400m14-dpt"
        case g14dpt = "google/tipsv2-g14-dpt"

        public var repoId: String { rawValue }
        public var isDPT: Bool { rawValue.hasSuffix("-dpt") }

        public var displayName: String {
            switch self {
            case .b14: "TIPSv2 B/14"
            case .l14: "TIPSv2 L/14"
            case .so400m14: "TIPSv2 SO400M/14"
            case .g14: "TIPSv2 g/14"
            case .b14dpt: "TIPSv2 B/14 — DPT heads"
            case .l14dpt: "TIPSv2 L/14 — DPT heads"
            case .so400m14dpt: "TIPSv2 SO400M/14 — DPT heads"
            case .g14dpt: "TIPSv2 g/14 — DPT heads"
            }
        }

        /// Backbone variant this repo corresponds to.
        public var variant: TIPSWeightLoader.Variant {
            switch self {
            case .b14, .b14dpt: .B
            case .l14, .l14dpt: .L
            case .so400m14, .so400m14dpt: .So400m
            case .g14, .g14dpt: .g
            }
        }

        /// The backbone repo for this entry: itself for a backbone repo, or the
        /// matching `tipsv2-*` for a `-dpt` repo. DPT heads need their backbone.
        public var backboneRepo: Repo {
            switch self {
            case .b14, .b14dpt: .b14
            case .l14, .l14dpt: .l14
            case .so400m14, .so400m14dpt: .so400m14
            case .g14, .g14dpt: .g14
            }
        }
    }

    /// Default HF cache root (`~/.cache/huggingface/hub`) — the same location
    /// the model loaders search.
    public static var defaultCacheRoot: URL {
        FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".cache/huggingface/hub", isDirectory: true)
    }

    /// Locate an already-downloaded snapshot in the **standard `huggingface_hub`
    /// cache layout** (`<root>/models--<org>--<repo>/snapshots/<newest>`), if one
    /// with a `model.safetensors` exists. This is the layout the Python HF client
    /// writes, so this reuses checkpoints downloaded by `huggingface-cli` / the
    /// Python demos rather than re-fetching them.
    public static func cachedSnapshot(repo: Repo, cacheRoot: URL? = nil) -> URL? {
        let root = cacheRoot ?? defaultCacheRoot
        let dirName = "models--" + repo.repoId.replacingOccurrences(of: "/", with: "--")
        let snapshots = root.appendingPathComponent(dirName, isDirectory: true)
            .appendingPathComponent("snapshots", isDirectory: true)
        let fm = FileManager.default
        guard let entries = try? fm.contentsOfDirectory(
            at: snapshots, includingPropertiesForKeys: [.contentModificationDateKey]
        ) else { return nil }
        let newest = entries
            .filter { (try? $0.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true }
            .max { a, b in
                let da = (try? a.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
                let db = (try? b.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
                return da < db
            }
        guard let snap = newest,
              fm.fileExists(atPath: snap.appendingPathComponent("model.safetensors").path)
        else { return nil }
        return snap
    }

    /// Reuse an existing HF-cache snapshot (``cachedSnapshot(repo:cacheRoot:)``)
    /// if present, otherwise ``download(repo:cacheRoot:hubToken:progress:)``.
    /// This is what the CLI and the GUI call so already-downloaded models are
    /// never re-fetched.
    @discardableResult
    public static func resolve(
        repo: Repo,
        cacheRoot: URL? = nil,
        hubToken: String? = nil,
        progress: @escaping @Sendable (Double) -> Void = { _ in }
    ) async throws -> URL {
        if let local = cachedSnapshot(repo: repo, cacheRoot: cacheRoot) {
            progress(1)
            return local
        }
        return try await download(repo: repo, cacheRoot: cacheRoot, hubToken: hubToken, progress: progress)
    }

    /// Download `repo` into the HF cache and return the local snapshot directory
    /// (ready to pass to ``TIPSSessionConfig`` / `fromPretrained`).
    ///
    /// - Note: swift-transformers writes a flat `<root>/models/<org>/<repo>/`
    ///   layout, which differs from the Python client's
    ///   `models--<org>--<repo>/snapshots/<hash>/`. Prefer ``resolve(repo:cacheRoot:hubToken:progress:)``
    ///   so an existing Python-cache copy is reused instead of duplicated.
    ///
    /// - Parameters:
    ///   - repo: which checkpoint to fetch.
    ///   - cacheRoot: hub root (defaults to ``defaultCacheRoot``).
    ///   - hubToken: optional HF token for gated/private repos.
    ///   - progress: download fraction in `0...1` (called off the main actor).
    @discardableResult
    public static func download(
        repo: Repo,
        cacheRoot: URL? = nil,
        hubToken: String? = nil,
        progress: @escaping @Sendable (Double) -> Void = { _ in }
    ) async throws -> URL {
        let hub = HubApi(downloadBase: cacheRoot ?? defaultCacheRoot, hfToken: hubToken)
        // The Swift loaders only need the weights + config (+ tokenizer for the
        // backbone); skip the HF python modeling files.
        let matching = repo.isDPT
            ? ["config.json", "model.safetensors"]
            : ["config.json", "model.safetensors", "tokenizer.model"]
        return try await hub.snapshot(from: repo.repoId, matching: matching) { p in
            progress(p.fractionCompleted)
        }
    }
}
