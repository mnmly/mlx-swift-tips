import XCTest
import MLX
@testable import MLXTIPS

/// Model-free coverage of the shared driver's rendering + Hub plumbing — the
/// parts the CLI and the SwiftUI app both depend on. `TIPSSession.load` itself
/// needs a checkpoint, so its contract is exercised by the `tips-cli` smoke run
/// rather than here.
final class SessionRenderTests: XCTestCase {

    func testMakeImageDimensions() {
        let w = 5, h = 3
        let img = TIPSRender.makeImage(rgbU8: [UInt8](repeating: 128, count: w * h * 3), width: w, height: h)
        XCTAssertEqual(img.width, w)
        XCTAssertEqual(img.height, h)
    }

    /// Deterministic synthetic features with per-row variance (so SVD is
    /// well-posed): row i, dim d = sin(i * 0.1 + d).
    private func syntheticSpatial(side: Int, dim: Int) -> SpatialFeatures {
        let n = side * side
        var vals = [Float](repeating: 0, count: n * dim)
        for i in 0..<n {
            for d in 0..<dim { vals[i * dim + d] = Foundation.sin(Float(i) * 0.1 + Float(d)) }
        }
        return SpatialFeatures(feats: MLXArray(vals, [n, dim]), side: side)
    }

    func testPCARendersAtGridSide() {
        let side = 8
        let spatial = syntheticSpatial(side: side, dim: 32)
        let pca = TIPSRender.pca(spatial)
        let pcaDepth = TIPSRender.pcaDepth(spatial)
        XCTAssertEqual(pca.width, side)
        XCTAssertEqual(pca.height, side)
        XCTAssertEqual(pcaDepth.width, side)
    }

    func testKMeansRendersAtGridSide() {
        let side = 8
        let img = TIPSRender.kmeans(syntheticSpatial(side: side, dim: 16), nClusters: 4)
        XCTAssertEqual(img.width, side)
        XCTAssertEqual(img.height, side)
    }

    func testHubBackboneRepoMapping() {
        XCTAssertEqual(TIPSHub.Repo.b14dpt.backboneRepo, .b14)
        XCTAssertEqual(TIPSHub.Repo.l14dpt.backboneRepo, .l14)
        XCTAssertEqual(TIPSHub.Repo.b14.backboneRepo, .b14)
        XCTAssertTrue(TIPSHub.Repo.g14dpt.isDPT)
        XCTAssertFalse(TIPSHub.Repo.g14.isDPT)
    }
}
