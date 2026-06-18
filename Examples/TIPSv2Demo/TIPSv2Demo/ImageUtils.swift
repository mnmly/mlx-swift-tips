import AppKit
import CoreGraphics
import Foundation
import MLXTIPS

// MARK: - AppKit glue
//
// All rendering (PCA, K-means, depth turbo, normals, segmentation palettes) now
// lives in the library as `CGImage`-returning `TIPSRender` functions, shared
// with the CLI. This frontend only bridges `CGImage` → `NSImage` and loads the
// base image used elsewhere in the UI.

// `nonisolated` — pure CoreGraphics/AppKit value conversions with no shared
// state; called from the off-actor (`@concurrent`) compute helpers.
enum ImageUtils {
    nonisolated static func loadNSImage(url: URL) throws -> NSImage {
        guard let img = NSImage(contentsOf: url) else { throw TIPSError.imageLoadFailed }
        return img
    }

    nonisolated static func nsImage(_ cg: CGImage) -> NSImage {
        NSImage(cgImage: cg, size: NSSize(width: cg.width, height: cg.height))
    }
}
