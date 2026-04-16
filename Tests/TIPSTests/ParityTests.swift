/// Numerical parity tests: compare Swift MLXTIPS outputs against Python MLX
/// reference values saved by `tests/generate_parity_fixtures.py`.
///
/// Prerequisites:
///   1. Run the fixture generator from the Python package root:
///        cd ../../python/mlx-tips
///        uv run python ../../swift/mlx-swift-tipsv2/tests/generate_parity_fixtures.py
///      This writes Tests/TIPSTests/Fixtures/parity_fixtures.safetensors.
///   2. Set TIPS_SNAPSHOT_DIR to the HF snapshot directory, or ensure the
///      default HF cache path is present.
///
/// Tests are skipped (not failed) when fixtures or snapshot are absent,
/// so CI without model weights continues to pass.

import XCTest
import MLX
@testable import MLXTIPS

final class ParityTests: XCTestCase {

    let atol: Float = 0.02

    private var fixtures: [String: MLXArray] = [:]
    private var snapshotURL: URL!
    private var _model: TIPSModel?

    // MARK: - Setup

    override func setUpWithError() throws {
        let fixturesURL = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .appendingPathComponent("Fixtures/parity_fixtures.safetensors")

        guard FileManager.default.fileExists(atPath: fixturesURL.path) else {
            throw XCTSkip(
                "Fixtures not found at \(fixturesURL.path). " +
                "Run `uv run python tests/generate_parity_fixtures.py` from python/mlx-tips first."
            )
        }
        fixtures = try MLX.loadArrays(url: fixturesURL)

        let defaultRoot = "~/.cache/huggingface/hub/models--google--tipsv2-b14/snapshots"
        let snapRoot = ProcessInfo.processInfo.environment["TIPS_SNAPSHOT_DIR"] ?? defaultRoot
        let rootURL = URL(fileURLWithPath: snapRoot)
        guard let snap = (try? FileManager.default.contentsOfDirectory(
            at: rootURL,
            includingPropertiesForKeys: [.contentModificationDateKey]
        ))?.max(by: { a, b in
            let da = (try? a.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
            let db = (try? b.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
            return da < db
        }),
        FileManager.default.fileExists(atPath: snap.appendingPathComponent("model.safetensors").path)
        else {
            throw XCTSkip("Snapshot not found. Set TIPS_SNAPSHOT_DIR or ensure HF cache is present.")
        }
        snapshotURL = snap
    }

    // MARK: - Vision

    func testVisionClsToken() throws {
        let out = try runVision()
        let ref = try require("cls_token")
        assertClose(out.clsToken, ref, label: "cls_token")
    }

    func testVisionRegisterTokens() throws {
        let out = try runVision()
        let ref = try require("register_tokens")
        assertClose(out.registerTokens, ref, label: "register_tokens")
    }

    func testVisionPatchTokens() throws {
        let out = try runVision()
        let ref = try require("patch_tokens")
        assertClose(out.patchTokens, ref, label: "patch_tokens")
    }

    // MARK: - Text

    func testTextEmbeddings() throws {
        let model = try loadModel()
        let ids  = try require("text_ids").asType(.int32)
        let pads = try require("text_paddings").asType(.int32)
        let ref  = try require("text_embeddings")

        let out = model.encodeText(ids: ids, paddings: pads)
        MLX.eval(out)
        assertClose(out, ref, label: "text_embeddings")
    }

    // MARK: - Private helpers

    private func loadModel() throws -> TIPSModel {
        if let m = _model { return m }
        let m = try TIPSWeightLoader.load(directory: snapshotURL, variant: .B)
        _model = m
        return m
    }

    private func runVision() throws -> TIPSImageOutput {
        let model = try loadModel()
        let image = try require("image_input")
        let out = model.encodeImage(image)
        MLX.eval(out.clsToken, out.registerTokens, out.patchTokens)
        return out
    }

    private func require(_ key: String) throws -> MLXArray {
        guard let arr = fixtures[key] else {
            throw XCTSkip("Fixture key '\(key)' missing — regenerate fixtures.")
        }
        return arr
    }

    private func assertClose(_ swift: MLXArray, _ ref: MLXArray, label: String) {
        XCTAssertEqual(swift.shape, ref.shape, "\(label): shape mismatch")
        let diff = abs(swift - ref)
        MLX.eval(diff)
        let maxErr = diff.max(keepDims: false).item(Float.self)
        let cos = cosineSim(swift, ref)
        print(String(format: "[%@]  max_err=%.2e  cosine=%.6f", label, maxErr, cos))
        XCTAssertLessThanOrEqual(maxErr, atol, "\(label): max abs error \(maxErr) exceeds atol \(atol)")
    }

    private func cosineSim(_ a: MLXArray, _ b: MLXArray) -> Float {
        let af = a.reshaped([-1]).asType(.float32)
        let bf = b.reshaped([-1]).asType(.float32)
        let dot = (af * bf).sum(keepDims: false)
        let na  = (af * af).sum(keepDims: false).sqrt()
        let nb  = (bf * bf).sum(keepDims: false).sqrt()
        MLX.eval(dot, na, nb)
        return dot.item(Float.self) / (na.item(Float.self) * nb.item(Float.self) + 1e-8)
    }
}
