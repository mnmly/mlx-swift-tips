/// `tips-cli` — run TIPSv2 tasks from the command line, driving the same
/// ``TIPSSession`` the SwiftUI app uses so the two stay byte-identical.
///
/// Examples:
///   tips-cli download google/tipsv2-b14
///   tips-cli pca      --repo google/tipsv2-b14 photo.jpg --out-dir out
///   tips-cli zeroseg  --repo google/tipsv2-b14 photo.jpg --labels "cat,dog,grass"
///   tips-cli depth    --repo google/tipsv2-b14 --dpt-repo google/tipsv2-b14-dpt photo.jpg
///
/// Build with `xcodebuild` (Metal toolchain) or `swift build` for a quick check.

import ArgumentParser
import CoreGraphics
import Foundation
import ImageIO
import MLXTIPS
import UniformTypeIdentifiers

// MARK: - Shared helpers

struct ModelOptions: ParsableArguments {
    @Option(name: .long, help: "Backbone snapshot directory (model.safetensors + tokenizer.model).")
    var backbone: String?

    @Option(name: .long, help: "HF backbone repo to download/reuse, e.g. google/tipsv2-b14.")
    var repo: String?

    @Option(name: .long, help: "Backbone variant: B | L | So400m | g.")
    var variant: String = "B"

    @Option(name: .long, help: "Square input resolution (px).")
    var resolution: Int = 448

    func resolveBackbone() async throws -> URL {
        try await resolveDir(path: backbone, repo: repo, what: "backbone")
    }

    var loaderVariant: TIPSWeightLoader.Variant {
        switch variant.lowercased() {
        case "l": return .L
        case "so400m", "so": return .So400m
        case "g": return .g
        default: return .B
        }
    }
}

/// Resolve a snapshot directory from an explicit path or by downloading a repo.
func resolveDir(path: String?, repo: String?, what: String) async throws -> URL {
    if let path { return URL(fileURLWithPath: (path as NSString).expandingTildeInPath) }
    if let repo {
        guard let known = TIPSHub.Repo(rawValue: repo) else {
            throw ValidationError("Unknown repo '\(repo)'. Known: \(TIPSHub.Repo.allCases.map(\.rawValue).joined(separator: ", "))")
        }
        // Reuse an existing HF-cache snapshot if present; download otherwise.
        let url = try await TIPSHub.resolve(repo: known) { p in
            FileHandle.standardError.write("\r  resolving \(known.displayName): \(Int(p * 100))%   ".data(using: .utf8)!)
        }
        FileHandle.standardError.write("\n".data(using: .utf8)!)
        return url
    }
    throw ValidationError("Provide --\(what) <dir> or --repo <id>.")
}

func writePNG(_ image: CGImage, to url: URL) throws {
    guard let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil)
    else { throw ValidationError("Could not create PNG destination at \(url.path)") }
    CGImageDestinationAddImage(dest, image, nil)
    guard CGImageDestinationFinalize(dest) else {
        throw ValidationError("Could not write PNG at \(url.path)")
    }
    FileHandle.standardError.write("wrote \(url.path)\n".data(using: .utf8)!)
}

func outDirURL(_ s: String) throws -> URL {
    let url = URL(fileURLWithPath: (s as NSString).expandingTildeInPath, isDirectory: true)
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    return url
}

// MARK: - Root

@main
struct TIPSCLI: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "tips-cli",
        abstract: "Run TIPSv2 image-encoder and DPT tasks via the shared TIPSSession.",
        subcommands: [Download.self, PCA.self, ZeroSeg.self, Depth.self, Segmentation.self]
    )
}

// MARK: download

extension TIPSCLI {
    struct Download: AsyncParsableCommand {
        static let configuration = CommandConfiguration(abstract: "Download a google/tipsv2-* checkpoint into the HF cache.")

        @Argument(help: "Repo id, e.g. google/tipsv2-b14 or google/tipsv2-b14-dpt.")
        var repo: String

        func run() async throws {
            guard let known = TIPSHub.Repo(rawValue: repo) else {
                throw ValidationError("Unknown repo. Known: \(TIPSHub.Repo.allCases.map(\.rawValue).joined(separator: ", "))")
            }
            let url = try await TIPSHub.download(repo: known) { p in
                FileHandle.standardError.write("\r  \(Int(p * 100))%   ".data(using: .utf8)!)
            }
            FileHandle.standardError.write("\n".data(using: .utf8)!)
            print(url.path)
        }
    }
}

// MARK: pca

extension TIPSCLI {
    struct PCA: AsyncParsableCommand {
        static let configuration = CommandConfiguration(abstract: "PCA / PCA-depth / K-means feature visualisations.")
        @OptionGroup var model: ModelOptions
        @Argument(help: "Input image.") var image: String
        @Option(name: .long, help: "Output directory.") var outDir: String = "."
        @Option(name: .long, help: "K-means clusters.") var clusters: Int = 8

        func run() async throws {
            let backbone = try await model.resolveBackbone()
            let session = try TIPSSession.load(.init(
                backboneDirectory: backbone, variant: model.loaderVariant, resolution: model.resolution))
            let cg = try TIPSSession.loadImage(at: URL(fileURLWithPath: (image as NSString).expandingTildeInPath))
            let out = try outDirURL(outDir)
            let (pca, pcaDepth, spatial) = try session.pca(cg)
            try writePNG(pca, to: out.appendingPathComponent("pca.png"))
            try writePNG(pcaDepth, to: out.appendingPathComponent("pca_depth.png"))
            try writePNG(session.kmeans(spatial, nClusters: clusters), to: out.appendingPathComponent("kmeans.png"))
        }
    }
}

// MARK: zeroseg

extension TIPSCLI {
    struct ZeroSeg: AsyncParsableCommand {
        static let configuration = CommandConfiguration(commandName: "zeroseg", abstract: "Zero-shot dense segmentation against text labels.")
        @OptionGroup var model: ModelOptions
        @Argument(help: "Input image.") var image: String
        @Option(name: .long, help: "Comma-separated class labels.") var labels: String
        @Option(name: .long, help: "Output directory.") var outDir: String = "."

        func run() async throws {
            let labelList = labels.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty }
            guard !labelList.isEmpty else { throw ValidationError("Provide --labels \"a,b,c\".") }
            let backbone = try await model.resolveBackbone()
            let session = try TIPSSession.load(.init(
                backboneDirectory: backbone, variant: model.loaderVariant, resolution: model.resolution))
            let cg = try TIPSSession.loadImage(at: URL(fileURLWithPath: (image as NSString).expandingTildeInPath))
            let out = try outDirURL(outDir)
            let result = try session.zeroShotSegment(cg, labels: labelList)
            try writePNG(result.overlay, to: out.appendingPathComponent("zeroseg_overlay.png"))
            try writePNG(result.mask, to: out.appendingPathComponent("zeroseg_mask.png"))
            print("detected: \(result.detected)")
        }
    }
}

// MARK: depth + normals

extension TIPSCLI {
    struct Depth: AsyncParsableCommand {
        static let configuration = CommandConfiguration(abstract: "DPT depth + normals.")
        @OptionGroup var model: ModelOptions
        @Option(name: .long, help: "DPT-head snapshot directory.") var dpt: String?
        @Option(name: .long, help: "HF DPT repo, e.g. google/tipsv2-b14-dpt.") var dptRepo: String?
        @Argument(help: "Input image.") var image: String
        @Option(name: .long, help: "Output directory.") var outDir: String = "."

        func run() async throws {
            let backbone = try await model.resolveBackbone()
            let dptDir = try await resolveDir(path: dpt, repo: dptRepo, what: "dpt")
            let session = try TIPSSession.load(.init(
                backboneDirectory: backbone, dptDirectory: dptDir,
                variant: model.loaderVariant, resolution: model.resolution))
            let cg = try TIPSSession.loadImage(at: URL(fileURLWithPath: (image as NSString).expandingTildeInPath))
            let out = try outDirURL(outDir)
            let (depth, normals) = try session.depthAndNormals(cg)
            try writePNG(depth, to: out.appendingPathComponent("depth.png"))
            try writePNG(normals, to: out.appendingPathComponent("normals.png"))
        }
    }
}

// MARK: segmentation (ADE20K)

extension TIPSCLI {
    struct Segmentation: AsyncParsableCommand {
        static let configuration = CommandConfiguration(commandName: "seg", abstract: "DPT ADE20K semantic segmentation.")
        @OptionGroup var model: ModelOptions
        @Option(name: .long, help: "DPT-head snapshot directory.") var dpt: String?
        @Option(name: .long, help: "HF DPT repo, e.g. google/tipsv2-b14-dpt.") var dptRepo: String?
        @Argument(help: "Input image.") var image: String
        @Option(name: .long, help: "Output directory.") var outDir: String = "."

        func run() async throws {
            let backbone = try await model.resolveBackbone()
            let dptDir = try await resolveDir(path: dpt, repo: dptRepo, what: "dpt")
            let session = try TIPSSession.load(.init(
                backboneDirectory: backbone, dptDirectory: dptDir,
                variant: model.loaderVariant, resolution: model.resolution))
            let cg = try TIPSSession.loadImage(at: URL(fileURLWithPath: (image as NSString).expandingTildeInPath))
            let out = try outDirURL(outDir)
            try writePNG(try session.segmentation(cg), to: out.appendingPathComponent("segmentation.png"))
        }
    }
}
