import XCTest
import MLX
import MLXNN
@testable import MLXTIPS

final class ShapeTests: XCTestCase {

    // MARK: - Vision encoder components

    func testPatchEmbed() {
        let m = PatchEmbed(patchSize: 14, inChannels: 3, embedDim: 768)
        let x = MLXArray.zeros([1, 448, 448, 3])
        let out = m(x)
        XCTAssertEqual(out.shape, [1, 1024, 768])
    }

    func testVisionAttention() {
        let m = VisionAttention(dim: 64, numHeads: 4)
        let x = MLXArray.zeros([2, 10, 64])
        let out = m(x)
        XCTAssertEqual(out.shape, [2, 10, 64])
    }

    func testLayerScale() {
        let m = LayerScale(dim: 64, initValues: 1.0)
        let x = MLXArray.zeros([2, 10, 64])
        let out = m(x)
        XCTAssertEqual(out.shape, [2, 10, 64])
    }

    func testVisionBlock() {
        let m = VisionBlock(dim: 64, numHeads: 4)
        let x = MLXArray.zeros([2, 10, 64])
        let out = m(x)
        XCTAssertEqual(out.shape, [2, 10, 64])
    }

    func testVisionBlockSwiGLU() {
        let m = VisionBlock(dim: 64, numHeads: 4, ffnLayer: .swiglu)
        let x = MLXArray.zeros([2, 10, 64])
        let out = m(x)
        XCTAssertEqual(out.shape, [2, 10, 64])
        XCTAssertTrue(m.mlp is VisionSwiGLU)
    }

    func testVisionSwiGLUHiddenRounding() {
        // hiddenDim=256 -> (256*2/3 + 7)/8*8 = (170 + 7)/8*8 = 176
        let m = VisionSwiGLU(dim: 64, hiddenDim: 256)
        XCTAssertEqual(m.w12.weight.shape, [2 * 176, 64])
        XCTAssertEqual(m.w3.weight.shape, [64, 176])
    }

    func testVisionVariantConfigs() {
        XCTAssertEqual(VisionVariant.B.config.embedDim, 768)
        XCTAssertEqual(VisionVariant.L.config.depth, 24)
        XCTAssertEqual(VisionVariant.g.config.ffnLayer, .swiglu)
        XCTAssertEqual(VisionVariant.So400m.config.numHeads, 16)
    }

    func testTextVariantConfigs() {
        XCTAssertEqual(TextVariant.B.config.hiddenSize, 768)
        XCTAssertEqual(TextVariant.So400m.config.numLayers, 27)
        XCTAssertEqual(TextVariant.g.config.mlpDim, 6144)
    }

    func testGetIntermediateLayers() {
        let m = VisionTransformer(
            imgSize: 56, patchSize: 14, embedDim: 64, depth: 4,
            numHeads: 4, mlpRatio: 2.0, numRegisterTokens: 1, initValues: 1.0
        )
        let x = MLXArray.zeros([1, 56, 56, 3])

        // Last 2 blocks; patches only.
        let feats = m.getIntermediateLayers(x, n: 2)
        XCTAssertEqual(feats.count, 2)
        XCTAssertEqual(feats[0].patchTokens.shape, [1, 16, 64])
        XCTAssertNil(feats[0].classToken)

        // With class token + reshape.
        let feats2 = m.getIntermediateLayers(x, n: 1, reshape: true, returnClassToken: true)
        XCTAssertEqual(feats2.count, 1)
        XCTAssertEqual(feats2[0].patchTokens.shape, [1, 4, 4, 64])
        XCTAssertEqual(feats2[0].classToken?.shape, [1, 64])

        // Explicit indices.
        let feats3 = m.getIntermediateLayers(x, indices: [0, 3])
        XCTAssertEqual(feats3.count, 2)
    }

    func testVisionTransformerSmall() {
        // Use a smaller config to keep test fast
        let m = VisionTransformer(
            imgSize: 56, patchSize: 14, embedDim: 64, depth: 2,
            numHeads: 4, mlpRatio: 2.0, numRegisterTokens: 1, initValues: 1.0
        )
        let x = MLXArray.zeros([1, 56, 56, 3])
        let out = m(x)
        // patches: (56/14)^2 = 16
        XCTAssertEqual(out.clsToken.shape, [1, 1, 64])
        XCTAssertEqual(out.registerTokens.shape, [1, 1, 64])
        XCTAssertEqual(out.patchTokens.shape, [1, 16, 64])
    }

    // MARK: - Text encoder components

    func testSinusoidalPosEmbed() {
        let emb = sinusoidalPosEmbed(seqLen: 64, dim: 768)
        XCTAssertEqual(emb.shape, [1, 64, 768])
    }

    func testTextAttention() {
        let m = TextAttention(dim: 64, numHeads: 4)
        let x = MLXArray.zeros([2, 10, 64])
        let mask = MLXArray.ones([2, 10], dtype: .float32)
        let out = m(x, keyPaddingMask: mask)
        XCTAssertEqual(out.shape, [2, 10, 64])
    }

    func testTextResBlock() {
        let m = TextResBlock(dim: 64, numHeads: 4, mlpDim: 256)
        let x = MLXArray.zeros([2, 10, 64])
        let mask = MLXArray.ones([2, 10], dtype: .float32)
        let out = m(x, validMask: mask)
        XCTAssertEqual(out.shape, [2, 10, 64])
    }

    func testTextEncoderSmall() {
        let m = TextEncoder(
            vocabSize: 100, dim: 64, numHeads: 4, numLayers: 2,
            mlpDim: 256, maxLen: 16
        )
        let ids = MLXArray.zeros([2, 16], dtype: .int32)
        let paddings = MLXArray.zeros([2, 16], dtype: .int32)
        let out = m(ids: ids, paddings: paddings)
        XCTAssertEqual(out.shape, [2, 64])
    }

    // MARK: - Full model

    func testFullModelSmall() {
        let vision = VisionTransformer(
            imgSize: 56, patchSize: 14, embedDim: 64, depth: 2,
            numHeads: 4, mlpRatio: 2.0, numRegisterTokens: 1, initValues: 1.0
        )
        let text = TextEncoder(
            vocabSize: 100, dim: 64, numHeads: 4, numLayers: 2,
            mlpDim: 256, maxLen: 16
        )
        let model = TIPSModel(vision: vision, text: text)

        let img = MLXArray.zeros([1, 56, 56, 3])
        let imgOut = model.encodeImage(img)
        XCTAssertEqual(imgOut.clsToken.shape, [1, 1, 64])
        XCTAssertEqual(imgOut.patchTokens.shape, [1, 16, 64])

        let ids = MLXArray.zeros([2, 16], dtype: .int32)
        let paddings = MLXArray.zeros([2, 16], dtype: .int32)
        let txtOut = model.encodeText(ids: ids, paddings: paddings)
        XCTAssertEqual(txtOut.shape, [2, 64])
    }

    // MARK: - Weight key remapping

    func testBuildVisionWeightsPrefixAgnostic() {
        // Combined HF-style: prefixed keys.
        let combined: [String: MLXArray] = [
            "vision_encoder.cls_token": MLXArray.zeros([1, 1, 4]),
            "vision_encoder.patch_embed.proj.weight": MLXArray.zeros([4, 3, 14, 14]),
            "text_encoder.token_embedding.weight": MLXArray.zeros([10, 4]),
            "temperature": MLXArray(Float(1.0)),
        ]
        let combinedOut = Dictionary(uniqueKeysWithValues:
            TIPSWeightLoader.buildVisionWeights(combined))
        XCTAssertNotNil(combinedOut["cls_token"])
        XCTAssertEqual(combinedOut["patch_embed.proj.weight"]?.shape, [4, 14, 14, 3])
        XCTAssertNil(combinedOut["token_embedding.weight"])

        // Split pre-converted: no prefix.
        let split: [String: MLXArray] = [
            "cls_token": MLXArray.zeros([1, 1, 4]),
            "patch_embed.proj.weight": MLXArray.zeros([4, 3, 14, 14]),
        ]
        let splitOut = Dictionary(uniqueKeysWithValues:
            TIPSWeightLoader.buildVisionWeights(split))
        XCTAssertNotNil(splitOut["cls_token"])
        XCTAssertEqual(splitOut["patch_embed.proj.weight"]?.shape, [4, 14, 14, 3])
    }

    func testBuildTextWeightsPrefixAgnostic() {
        let combined: [String: MLXArray] = [
            "text_encoder.transformer.resblocks.0.attn.in_proj_weight": MLXArray.zeros([12, 4]),
            "text_encoder.transformer.resblocks.0.mlp.c_fc.weight": MLXArray.zeros([16, 4]),
            "vision_encoder.cls_token": MLXArray.zeros([1, 1, 4]),
            "temperature": MLXArray(Float(1.0)),
        ]
        let combinedOut = Dictionary(uniqueKeysWithValues:
            TIPSWeightLoader.buildTextWeights(combined))
        XCTAssertNotNil(combinedOut["blocks.0.attn.in_proj.weight"])
        XCTAssertNotNil(combinedOut["blocks.0.mlp_c_fc.weight"])
        XCTAssertNil(combinedOut["cls_token"])
        XCTAssertNil(combinedOut["temperature"])

        let split: [String: MLXArray] = [
            "transformer.resblocks.0.attn.in_proj_weight": MLXArray.zeros([12, 4]),
            "ln_final.weight": MLXArray.zeros([4]),
            "temperature": MLXArray(Float(1.0)),
        ]
        let splitOut = Dictionary(uniqueKeysWithValues:
            TIPSWeightLoader.buildTextWeights(split))
        XCTAssertNotNil(splitOut["blocks.0.attn.in_proj.weight"])
        XCTAssertNotNil(splitOut["ln_final.weight"])
        XCTAssertNil(splitOut["temperature"])
    }

    func testTextKeyRemap() {
        let cases: [(String, String)] = [
            ("transformer.resblocks.0.attn.in_proj_weight", "blocks.0.attn.in_proj.weight"),
            ("transformer.resblocks.0.attn.in_proj_bias", "blocks.0.attn.in_proj.bias"),
            ("transformer.resblocks.0.attn.out_proj.weight", "blocks.0.attn.out_proj.weight"),
            ("transformer.resblocks.0.mlp.c_fc.weight", "blocks.0.mlp_c_fc.weight"),
            ("transformer.resblocks.0.mlp.c_proj.bias", "blocks.0.mlp_c_proj.bias"),
            ("ln_final.weight", "ln_final.weight"),
            ("token_embedding.weight", "token_embedding.weight"),
        ]
        for (input, expected) in cases {
            XCTAssertEqual(TIPSWeightLoader.remapTextKey(input), expected, "Failed for key: \(input)")
        }
    }
}
