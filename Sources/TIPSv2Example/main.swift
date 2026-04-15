/// TIPSv2 example — mirrors the three PyTorch demo notebooks:
///
///   TIPS_Demo.ipynb                       → --task zeroshot  (zero-shot classification)
///   TIPS_Demo.ipynb                       → --task pca       (PCA patch-feature visualisation)
///   TIPS_Demo.ipynb / zeroshot_seg…ipynb  → --task seg       (zero-shot dense segmentation)
///   mlx_tipsv2_dpt.py                     → --task dpt       (DPT depth / normals / seg heads)
///
/// Usage:
///   tipsv2-example [options] <image> <label> [<label> ...]
///
/// Options:
///   --model-dir  <dir>          Directory with model.safetensors + tokenizer.model
///   --split      <v.st> <t.st> <tok>   Split safetensors files
///   --variant    B|L|So400m|g   Model variant (default: B)
///   --dpt-dir    <dir>          DPT head model directory (model.safetensors + config.json)
///   --backbone-dir <dir>        Backbone directory for DPT loading
///   --task       zeroshot|pca|seg|dpt|all  (default: zeroshot)
///   --out-dir    <dir>          Where to write output images (default: .)
///   --img-size   <n>            Resize input to n×n (default: 448)

import Foundation
import CoreGraphics
import ImageIO
import MLX
import MLXNN
import MLXLinalg
import MLXTIPSv2

// MARK: - Argument parsing

struct Args {
    enum LoadMode {
        case directory(URL)
        case split(vision: URL, text: URL, tokenizer: URL)
        case dpt(dptDir: URL, backboneDir: URL)
    }

    var loadMode: LoadMode = .directory(URL(fileURLWithPath: "."))
    var variant: TIPSv2WeightLoader.Variant = .B
    var task: String = "zeroshot"
    var outDir: URL = URL(fileURLWithPath: ".")
    var imgSize: Int = 448
    var imagePath: String = ""
    var labels: [String] = []
    // vseg options
    var useSlide: Bool = false
    var stride: Int = 336
    var useTemplates: Bool = false
    // fg options
    var trainDir: URL? = nil
    var maskDir: URL? = nil
    var fgC: Float = 0.1
}

func parseArgs() -> Args {
    var a = Args()
    var argv = CommandLine.arguments.dropFirst()

    func next(_ flag: String) -> String {
        guard let v = argv.popFirst() else {
            fputs("Missing value after \(flag)\n", stderr); exit(2)
        }
        return v
    }

    while let arg = argv.first {
        switch arg {
        case "--model-dir":
            argv.removeFirst()
            a.loadMode = .directory(URL(fileURLWithPath: next("--model-dir")))
        case "--split":
            argv.removeFirst()
            let v = URL(fileURLWithPath: next("--split"))
            let t = URL(fileURLWithPath: next("--split(text)"))
            let tok = URL(fileURLWithPath: next("--split(tok)"))
            a.loadMode = .split(vision: v, text: t, tokenizer: tok)
        case "--dpt-dir":
            argv.removeFirst()
            let dpt = URL(fileURLWithPath: next("--dpt-dir"))
            // backbone-dir must follow immediately
            guard argv.first == "--backbone-dir" else {
                fputs("--dpt-dir requires --backbone-dir <path> immediately after\n", stderr); exit(2)
            }
            argv.removeFirst()
            let bb = URL(fileURLWithPath: next("--backbone-dir"))
            a.loadMode = .dpt(dptDir: dpt, backboneDir: bb)
        case "--variant":
            argv.removeFirst()
            a.variant = variantFromString(next("--variant"))
        case "--task":
            argv.removeFirst()
            a.task = next("--task")
        case "--out-dir":
            argv.removeFirst()
            a.outDir = URL(fileURLWithPath: next("--out-dir"))
        case "--img-size":
            argv.removeFirst()
            a.imgSize = Int(next("--img-size")) ?? 448
        case "--stride":
            argv.removeFirst()
            a.stride = Int(next("--stride")) ?? 336
        case "--slide":
            argv.removeFirst()
            a.useSlide = true
        case "--templates":
            argv.removeFirst()
            a.useTemplates = true
        case "--train-dir":
            argv.removeFirst()
            a.trainDir = URL(fileURLWithPath: next("--train-dir"))
        case "--mask-dir":
            argv.removeFirst()
            a.maskDir = URL(fileURLWithPath: next("--mask-dir"))
        case "--fg-c":
            argv.removeFirst()
            a.fgC = Float(next("--fg-c")) ?? 0.1
        default:
            argv.removeFirst()
            if a.imagePath.isEmpty {
                a.imagePath = arg
            } else {
                a.labels.append(arg)
            }
        }
    }

    if a.imagePath.isEmpty {
        fputs("Usage: tipsv2-example [options] <image> [<label> ...]\n", stderr)
        fputs("  --model-dir  <dir>    combined safetensors directory\n", stderr)
        fputs("  --split <v> <t> <tok> split safetensors files\n", stderr)
        fputs("  --dpt-dir <dir> --backbone-dir <dir>  DPT mode\n", stderr)
        fputs("  --variant B|L|So400m|g\n", stderr)
        fputs("  --task zeroshot|pca|seg|dpt|all\n", stderr)
        fputs("  --out-dir <dir>       output directory for saved images\n", stderr)
        fputs("  --img-size <n>        resize input (default: 448)\n", stderr)
        exit(2)
    }
    return a
}

func variantFromString(_ s: String) -> TIPSv2WeightLoader.Variant {
    switch s.lowercased() {
    case "l":      return .L
    case "so400m", "so": return .So400m
    case "g":      return .g
    default:       return .B
    }
}

// MARK: - Image I/O (CGImage ↔ MLXArray)

/// Load an image from disk, resize to `size × size`, return NHWC float32 [0,1].
/// Mirrors `preprocess_image` / `load_image_bytes` from the notebooks.
func loadImage(at path: String, size: Int) throws -> MLXArray {
    let url = URL(fileURLWithPath: path)
    guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
          let cg  = CGImageSourceCreateImageAtIndex(src, 0, nil)
    else { throw NSError(domain: "TIPS", code: 1,
                         userInfo: [NSLocalizedDescriptionKey: "Cannot load \(path)"]) }

    let cs = CGColorSpaceCreateDeviceRGB()
    var buf = [UInt8](repeating: 0, count: size * size * 4)
    guard let ctx = CGContext(data: &buf, width: size, height: size,
                              bitsPerComponent: 8, bytesPerRow: size * 4,
                              space: cs,
                              bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
    else { throw NSError(domain: "TIPS", code: 2,
                         userInfo: [NSLocalizedDescriptionKey: "CGContext failed"]) }
    ctx.interpolationQuality = .high
    ctx.draw(cg, in: CGRect(x: 0, y: 0, width: size, height: size))

    var floats = [Float](repeating: 0, count: size * size * 3)
    var o = 0
    for i in stride(from: 0, to: buf.count, by: 4) {
        floats[o]     = Float(buf[i])     / 255.0
        floats[o + 1] = Float(buf[i + 1]) / 255.0
        floats[o + 2] = Float(buf[i + 2]) / 255.0
        o += 3
    }
    return MLXArray(floats, [1, size, size, 3])
}

/// Save an NHWC uint8 array `[1, H, W, C]` (C = 3) as PNG.
func saveRGB(_ arr: MLXArray, to url: URL) throws {
    MLX.eval(arr)
    let h = arr.dim(1), w = arr.dim(2)
    let flat = arr.reshaped(h * w * 3).asArray(UInt8.self)

    var rgba = [UInt8](repeating: 255, count: h * w * 4)
    for i in 0..<(h * w) {
        rgba[i * 4]     = flat[i * 3]
        rgba[i * 4 + 1] = flat[i * 3 + 1]
        rgba[i * 4 + 2] = flat[i * 3 + 2]
    }

    let cs = CGColorSpaceCreateDeviceRGB()
    guard let ctx = CGContext(data: &rgba, width: w, height: h,
                              bitsPerComponent: 8, bytesPerRow: w * 4,
                              space: cs,
                              bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue),
          let cgImg = ctx.makeImage()
    else { throw NSError(domain: "TIPS", code: 3,
                         userInfo: [NSLocalizedDescriptionKey: "Could not create image"]) }

    guard let dest = CGImageDestinationCreateWithURL(url as CFURL, "public.png" as CFString, 1, nil)
    else { throw NSError(domain: "TIPS", code: 4,
                         userInfo: [NSLocalizedDescriptionKey: "Cannot write \(url.path)"]) }
    CGImageDestinationAddImage(dest, cgImg, nil)
    CGImageDestinationFinalize(dest)
}

/// Save a float32 NCHW single-channel image `[1, 1, H, W]` as a greyscale PNG.
/// Values are normalised to [0, 255] using min/max across the image.
func saveGreyscale(_ arr: MLXArray, to url: URL) throws {
    MLX.eval(arr)
    // arr: (1, 1, H, W) — squeeze to (H, W)
    let squeezed = arr.squeezed()
    let h = squeezed.dim(0), w = squeezed.dim(1)
    let flat = squeezed.asArray(Float.self)
    let mn = flat.min() ?? 0
    let mx = flat.max() ?? 1
    let range = max(mx - mn, 1e-8)
    let bytes = flat.map { UInt8(((($0 - mn) / range) * 255).clamped(to: 0...255)) }

    var rgba = [UInt8](repeating: 255, count: h * w * 4)
    for i in 0..<(h * w) {
        rgba[i * 4] = bytes[i]; rgba[i * 4 + 1] = bytes[i]; rgba[i * 4 + 2] = bytes[i]
    }

    let cs = CGColorSpaceCreateDeviceRGB()
    guard let ctx = CGContext(data: &rgba, width: w, height: h,
                              bitsPerComponent: 8, bytesPerRow: w * 4,
                              space: cs,
                              bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue),
          let cgImg = ctx.makeImage()
    else { throw NSError(domain: "TIPS", code: 3,
                         userInfo: [NSLocalizedDescriptionKey: "Could not create image"]) }

    guard let dest = CGImageDestinationCreateWithURL(url as CFURL, "public.png" as CFString, 1, nil)
    else { throw NSError(domain: "TIPS", code: 4,
                         userInfo: [NSLocalizedDescriptionKey: "Cannot write \(url.path)"]) }
    CGImageDestinationAddImage(dest, cgImg, nil)
    CGImageDestinationFinalize(dest)
}

extension Float {
    func clamped(to range: ClosedRange<Float>) -> Float { min(max(self, range.lowerBound), range.upperBound) }
}

// MARK: - Stable per-class colour palette (deterministic)

/// Deterministic per-class RGB palette via a simple LCG hash.
/// Mirrors the `np.random.default_rng(0)` palette used in the Python demos.
func makePalette(count: Int) -> [[UInt8]] {
    var seed: UInt64 = 0
    func nextUInt8() -> UInt8 {
        seed = seed &* 6364136223846793005 &+ 1442695040888963407
        return UInt8((seed >> 33) & 0xFF)
    }
    return (0..<count).map { _ in [nextUInt8(), nextUInt8(), nextUInt8()] }
}

// MARK: - Tasks

/// Task 1 — Zero-shot classification (mirrors TIPS_Demo.ipynb cell 9).
///
/// Encodes the image CLS token and a list of text labels, prints ranked cosine
/// similarities — identical to `zero_shot_classification` in `example.py`.
func runZeroshot(model: TIPSv2Model, pixels: MLXArray, labels: [String]) throws {
    guard !labels.isEmpty else { print("  [zeroshot] no labels — skipping"); return }
    let imgOut = model.encodeImage(pixels)
    // CLS token: (1, 1, D) → (1, D)
    let imgFeat = l2Normalize(imgOut.clsToken.squeezed(axis: 1))
    let txtFeat = l2Normalize(try model.encodeText(labels))
    let sim = matmul(imgFeat, txtFeat.transposed(1, 0))[0]  // (N,)
    eval(sim)

    let scores = sim.asArray(Float.self)
    let ranked = scores.indices.sorted { scores[$0] > scores[$1] }
    print("  zero-shot ranking:")
    for i in ranked {
        print(String(format: "    %+.3f  %@", scores[i], labels[i]))
    }
}

/// Task 2 — PCA patch-feature visualisation (mirrors TIPS_Demo.ipynb `dense_feature_pca`).
///
/// Reduces D-dimensional patch features to 3 RGB channels via SVD, normalises
/// to [0, 255] and saves as `pca.png`.
func runPCA(model: TIPSv2Model, pixels: MLXArray, outDir: URL) throws {
    let imgOut = model.encodeImage(pixels)
    // patch_tokens: (1, N, D) → (N, D)
    let feats = imgOut.patchTokens[0].asType(.float32)  // (N, D)
    let n = feats.dim(0)

    // Centre
    let mean = feats.mean(axis: 0, keepDims: true)             // (1, D)
    let centred = feats - mean                                  // (N, D)

    // Thin SVD — mlx-swift returns (U, S, Vt); SVD is CPU-only in MLX
    MLX.eval(centred)
    let cpuStream = MLX.StreamOrDevice.cpu
    let (_, _, vt) = MLXLinalg.svd(centred, stream: cpuStream)  // Vt: (D, D) or (min(N,D), D)

    // Project onto top-3 components
    let top3 = vt[..<3]                                        // (3, D)
    let proj = matmul(centred, top3.transposed(1, 0))          // (N, 3)

    // Normalise each channel to [0, 255]
    MLX.eval(proj)
    let projF = proj.asArray(Float.self)                        // flat [N*3], row-major
    var rgb = [Float](repeating: 0, count: n * 3)
    for c in 0..<3 {
        var mn = Float.greatestFiniteMagnitude
        var mx = -Float.greatestFiniteMagnitude
        for i in 0..<n { let v = projF[i * 3 + c]; mn = min(mn, v); mx = max(mx, v) }
        let range = max(mx - mn, 1e-8)
        for i in 0..<n { rgb[i * 3 + c] = ((projF[i * 3 + c] - mn) / range * 255) }
    }

    let side = Int(Double(n).squareRoot().rounded())
    let rgbU8 = rgb.map { UInt8($0.clamped(to: 0...255)) }
    let imgArray = MLXArray(rgbU8, [1, side, side, 3])

    let outURL = outDir.appendingPathComponent("pca.png")
    try saveRGB(imgArray, to: outURL)
    print("  PCA heatmap → \(outURL.path)")
}

/// Task 3 — Zero-shot dense segmentation (mirrors TIPS_Demo.ipynb `zero_shot_segmentation`
/// and TIPS_zeroshot_segmentation.ipynb).
///
/// Computes cosine similarity between per-patch features and text label embeddings,
/// takes argmax, colours each patch with a deterministic palette, and saves
/// `seg.png` (patch-resolution) and `seg_labels.txt`.
func runSegmentation(model: TIPSv2Model, pixels: MLXArray, labels: [String], outDir: URL) throws {
    guard !labels.isEmpty else { print("  [seg] no labels — skipping"); return }

    let imgOut = model.encodeImage(pixels)
    let patches = l2Normalize(imgOut.patchTokens[0])            // (N, D)
    let txtFeat = l2Normalize(try model.encodeText(labels))     // (C, D)

    let logits = matmul(patches, txtFeat.transposed(1, 0))      // (N, C)
    let labelMap = argMax(logits, axis: -1)                     // (N,)
    eval(labelMap)

    let n = patches.dim(0)
    let side = Int(Double(n).squareRoot().rounded())
    let labelInts = labelMap.asArray(Int32.self)
    let palette = makePalette(count: labels.count)

    var pixelData = [UInt8](repeating: 0, count: side * side * 3)
    for i in 0..<(side * side) {
        let cls = Int(labelInts[i])
        let color = palette[min(cls, palette.count - 1)]
        pixelData[i * 3] = color[0]; pixelData[i * 3 + 1] = color[1]; pixelData[i * 3 + 2] = color[2]
    }
    let segArray = MLXArray(pixelData, [1, side, side, 3])

    let segURL = outDir.appendingPathComponent("seg.png")
    try saveRGB(segArray, to: segURL)
    print("  segmentation map → \(segURL.path)  (\(labels.count) classes, \(side)×\(side) patches)")

    // Print label index counts
    var counts = [Int: Int]()
    for v in labelInts { counts[Int(v), default: 0] += 1 }
    let topLabels = counts.sorted { $0.value > $1.value }.prefix(5)
    print("  top-5 predicted classes:")
    for (idx, count) in topLabels {
        print(String(format: "    %5.1f%%  %@", Float(count) / Float(n) * 100, labels[idx]))
    }
}

/// Task 4 — DPT dense-prediction (mirrors `mlx_tipsv2_dpt.py` __main__ and `load_tipsv2_dpt`).
///
/// Produces depth, normals, and segmentation maps using the DPT heads.
/// Saves `dpt_depth.png`, `dpt_normals.png`, `dpt_seg.png`.
func runDPT(dptModel: TIPSv2DPTModel, pixels: MLXArray, outDir: URL) throws {
    let h = pixels.dim(1), w = pixels.dim(2)
    print("  running DPT heads on \(h)×\(w) image …")

    // ---- Depth ----
    let depth = dptModel.predictDepth(pixels)   // (1, 1, H, W)
    eval(depth)
    let depthURL = outDir.appendingPathComponent("dpt_depth.png")
    try saveGreyscale(depth, to: depthURL)
    let depthVals = depth.squeezed().asArray(Float.self)
    let depthMin = depthVals.min() ?? 0, depthMax = depthVals.max() ?? 0
    print(String(format: "  depth → %@  (min %.3f m, max %.3f m)", depthURL.path, depthMin, depthMax))

    // ---- Normals ----
    let normals = dptModel.predictNormals(pixels)   // (1, 3, H, W)
    eval(normals)
    // Map [-1,1] normals to [0,255] RGB: n * 0.5 + 0.5
    let normalsNHWC = normals.transposed(0, 2, 3, 1)    // (1, H, W, 3)
    let normalsVis = ((normalsNHWC * 0.5 + 0.5) * 255).asType(.uint8)
    let normalsURL = outDir.appendingPathComponent("dpt_normals.png")
    try saveRGB(normalsVis, to: normalsURL)
    print("  normals → \(normalsURL.path)")

    // ---- Segmentation ----
    let seg = dptModel.predictSegmentation(pixels)      // (1, numClasses, H, W)
    eval(seg)
    // Argmax over channel dim → (1, H, W)
    let segLabels = argMax(seg[0], axis: 0)             // (H, W)
    let numClasses = seg.dim(1)
    let palette = makePalette(count: numClasses)
    let labelArr = segLabels.asArray(Int32.self)
    let segH = seg.dim(2), segW = seg.dim(3)
    var segRGB = [UInt8](repeating: 0, count: segH * segW * 3)
    for i in 0..<(segH * segW) {
        let c = palette[min(Int(labelArr[i]), palette.count - 1)]
        segRGB[i * 3] = c[0]; segRGB[i * 3 + 1] = c[1]; segRGB[i * 3 + 2] = c[2]
    }
    let segArray = MLXArray(segRGB, [1, segH, segW, 3])
    let segURL = outDir.appendingPathComponent("dpt_seg.png")
    try saveRGB(segArray, to: segURL)
    print("  DPT segmentation → \(segURL.path)  (\(numClasses) classes)")
}

// MARK: - Short-side image loading

/// Load an image resizing so its short side = `shortSide`, preserving aspect ratio. Returns NHWC float32 [0,1].
func loadImageShortSide(at path: String, shortSide: Int) throws -> MLXArray {
    let url = URL(fileURLWithPath: path)
    guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
          let cg  = CGImageSourceCreateImageAtIndex(src, 0, nil)
    else { throw NSError(domain: "TIPS", code: 1,
                         userInfo: [NSLocalizedDescriptionKey: "Cannot load \(path)"]) }

    let origH = cg.height, origW = cg.width
    let newH: Int, newW: Int
    if origW <= origH {
        newW = shortSide
        newH = max(shortSide, Int(Float(shortSide) * Float(origH) / Float(origW)))
    } else {
        newH = shortSide
        newW = max(shortSide, Int(Float(shortSide) * Float(origW) / Float(origH)))
    }

    let cs = CGColorSpaceCreateDeviceRGB()
    var buf = [UInt8](repeating: 0, count: newH * newW * 4)
    guard let ctx = CGContext(data: &buf, width: newW, height: newH,
                              bitsPerComponent: 8, bytesPerRow: newW * 4,
                              space: cs,
                              bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
    else { throw NSError(domain: "TIPS", code: 2,
                         userInfo: [NSLocalizedDescriptionKey: "CGContext failed"]) }
    ctx.interpolationQuality = .high
    ctx.draw(cg, in: CGRect(x: 0, y: 0, width: newW, height: newH))

    var floats = [Float](repeating: 0, count: newH * newW * 3)
    var o = 0
    for i in stride(from: 0, to: buf.count, by: 4) {
        floats[o]     = Float(buf[i])     / 255.0
        floats[o + 1] = Float(buf[i + 1]) / 255.0
        floats[o + 2] = Float(buf[i + 2]) / 255.0
        o += 3
    }
    return MLXArray(floats, [1, newH, newW, 3])
}

// MARK: - Value-attention segmentation (vseg)

/// TCL prompt templates (9) from https://github.com/khanrc/tcl/blob/main/datasets/templates.py
private let tclTemplates: [String] = [
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

/// Encode each label through all TCL templates, normalize each, mean-pool, re-normalize.
/// Returns (C, D) feature matrix, l2-normalized per row.
func encodeTextWithTemplates(model: TIPSv2Model, labels: [String]) throws -> MLXArray {
    var classFeats: [MLXArray] = []
    for label in labels {
        let texts = tclTemplates.map { $0.replacingOccurrences(of: "{}", with: label) }
        let feats = try model.encodeText(texts)                  // (T, D)
        let normed = l2Normalize(feats)                          // (T, D)
        let avg = normed.mean(axis: 0, keepDims: true)           // (1, D)
        classFeats.append(l2Normalize(avg))                      // (1, D)
    }
    return concatenated(classFeats, axis: 0)                     // (C, D)
}

/// Compute value-attention cosine map for a whole image.
/// - Parameters:
///   - visionEncoder: the vision transformer
///   - pixels: NHWC (1, H, W, 3)
///   - textFeats: (C, D) l2-normalized text features
/// - Returns: (C, h, w) cosine similarity map at patch resolution
func predictWhole(visionEncoder: VisionTransformer, pixels: MLXArray, textFeats: MLXArray) -> MLXArray {
    let feats = visionEncoder.encodeValueAttention(pixels)       // (1, h, w, D)
    let h = feats.dim(1), w = feats.dim(2), D = feats.dim(3)
    let C = textFeats.dim(0)
    let patches = l2Normalize(feats.reshaped(h * w, D))          // (N, D)
    let cos = matmul(textFeats, patches.transposed(1, 0))        // (C, N)
    return cos.reshaped(C, h, w)                                 // (C, h, w)
}

/// Sliding-window inference (mirrors TCL/predict_slide).
/// Crops (side × side) windows with `stride`, upsamples each to pixel space,
/// accumulates softmax probabilities, returns pixel-resolution argmax label map (H × W) flat Int32.
func predictSlide(
    visionEncoder: VisionTransformer,
    pixels1: MLXArray,          // (H, W, 3) — no batch dim
    textFeats: MLXArray,        // (C, D)
    side: Int, stride: Int,
    H: Int, W: Int, C: Int
) -> [Int32] {
    var probs  = [Float](repeating: 0, count: C * H * W)
    var counts = [Float](repeating: 0, count: H * W)

    let hGrids = max(H - side + stride - 1, 0) / stride + 1
    let wGrids = max(W - side + stride - 1, 0) / stride + 1

    for i in 0..<hGrids {
        for j in 0..<wGrids {
            var y1 = i * stride, x1 = j * stride
            let y2 = min(y1 + side, H), x2 = min(x1 + side, W)
            y1 = max(y2 - side, 0); x1 = max(x2 - side, 0)
            let winH = y2 - y1, winW = x2 - x1

            // Crop window and add batch dim
            let window = pixels1[y1..<y2, x1..<x2, 0...].reshaped(1, winH, winW, 3)

            // Compute cosine map at patch resolution (C, h_p, w_p)
            let cosMap = predictWhole(visionEncoder: visionEncoder, pixels: window, textFeats: textFeats)
            let hp = cosMap.dim(1), wp = cosMap.dim(2)

            // Upsample (C, hp, wp) → (C, winH, winW) via bilinear (alignCorners=false)
            // MLXNN.Upsample expects NHWC (1, hp, wp, C)
            let cosNHWC = cosMap.transposed(1, 2, 0).reshaped(1, hp, wp, C)
            let scaleH = Float(winH) / Float(hp)
            let scaleW = Float(winW) / Float(wp)
            let up = MLXNN.Upsample(scaleFactor: [scaleH, scaleW], mode: .linear(alignCorners: false))
            let upResult = up(cosNHWC)   // (1, winH, winW, C)
            let upCHW = upResult[0].transposed(2, 0, 1)  // (C, winH, winW)
            eval(upCHW)
            let upArr = upCHW.asArray(Float.self)

            // Softmax along C and accumulate
            for py in 0..<winH {
                for px in 0..<winW {
                    var maxVal = -Float.greatestFiniteMagnitude
                    for c in 0..<C {
                        let v = upArr[c * winH * winW + py * winW + px]
                        if v > maxVal { maxVal = v }
                    }
                    var sum: Float = 0
                    var expVals = [Float](repeating: 0, count: C)
                    for c in 0..<C {
                        expVals[c] = exp(upArr[c * winH * winW + py * winW + px] - maxVal)
                        sum += expVals[c]
                    }
                    let gy = y1 + py, gx = x1 + px
                    for c in 0..<C {
                        probs[c * H * W + gy * W + gx] += expVals[c] / sum
                    }
                    counts[gy * W + gx] += 1
                }
            }
        }
    }

    // Argmax per pixel
    var labelMap = [Int32](repeating: 0, count: H * W)
    for gy in 0..<H {
        for gx in 0..<W {
            let cnt = counts[gy * W + gx]
            guard cnt > 0 else { continue }
            var bestC = 0, bestIdx = 0
            var bestP: Float = -1
            for c in 0..<C {
                let p = probs[c * H * W + gy * W + gx] / cnt
                if p > bestP { bestP = p; bestC = c }
            }
            labelMap[gy * W + gx] = Int32(bestC)
            _ = bestIdx  // suppress warning
        }
    }
    return labelMap
}

/// Task 5 — Value-attention zero-shot segmentation (mirrors TIPS_zeroshot_segmentation.ipynb).
func runValueSeg(
    model: TIPSv2Model,
    pixels: MLXArray,
    labels: [String],
    outDir: URL,
    useSlide: Bool,
    stride: Int,
    useTemplates: Bool,
    imgSize: Int
) throws {
    guard !labels.isEmpty else { print("  [vseg] no labels — skipping"); return }

    print("  encoding text\(useTemplates ? " (TCL templates)" : "") …")
    let textFeats: MLXArray
    if useTemplates {
        textFeats = try encodeTextWithTemplates(model: model, labels: labels)
    } else {
        textFeats = l2Normalize(try model.encodeText(labels))
    }
    eval(textFeats)

    let H = pixels.dim(1), W = pixels.dim(2)
    let C = textFeats.dim(0)
    let palette = makePalette(count: labels.count)

    let labelMap: [Int32]
    let outH: Int, outW: Int
    if useSlide {
        print("  sliding-window inference (side=\(imgSize), stride=\(stride)) …")
        labelMap = predictSlide(
            visionEncoder: model.visionEncoder,
            pixels1: pixels[0],
            textFeats: textFeats,
            side: imgSize, stride: stride,
            H: H, W: W, C: C
        )
        outH = H; outW = W
    } else {
        print("  whole-image inference …")
        let cosMap = predictWhole(visionEncoder: model.visionEncoder, pixels: pixels, textFeats: textFeats)
        eval(cosMap)
        outH = cosMap.dim(1); outW = cosMap.dim(2)
        let argmaxMap = argMax(cosMap, axis: 0)
        eval(argmaxMap)
        labelMap = argmaxMap.asArray(Int32.self)
    }

    var pixelData = [UInt8](repeating: 0, count: outH * outW * 3)
    for i in 0..<(outH * outW) {
        let cls = Int(labelMap[i])
        let color = palette[min(cls, palette.count - 1)]
        pixelData[i * 3] = color[0]; pixelData[i * 3 + 1] = color[1]; pixelData[i * 3 + 2] = color[2]
    }
    let segArray = MLXArray(pixelData, [1, outH, outW, 3])
    let outURL = outDir.appendingPathComponent("vseg.png")
    try saveRGB(segArray, to: outURL)
    let mode = useSlide ? "slide" : "whole"
    print("  vseg (\(mode)) → \(outURL.path)  (\(outH)×\(outW), \(labels.count) classes)")
}

// MARK: - Foreground segmentation (fg)

/// 3×3 median filter on a 2D Float grid (hPatches × wPatches).
func medianFilter3x3(_ grid: [Float], rows: Int, cols: Int) -> [Float] {
    var out = grid
    for r in 0..<rows {
        for c in 0..<cols {
            var window: [Float] = []
            for dr in -1...1 {
                for dc in -1...1 {
                    let nr = max(0, min(rows - 1, r + dr))
                    let nc = max(0, min(cols - 1, c + dc))
                    window.append(grid[nr * cols + nc])
                }
            }
            window.sort()
            out[r * cols + c] = window[4]  // median of 9
        }
    }
    return out
}

/// Load the alpha channel of an RGBA PNG, box-filter to patch grid.
/// Returns flat [Float] values in [0,1] of size (hPatches × wPatches).
func loadMaskPatches(at path: String, imgSize: Int = 448, patchSize: Int = 14) throws -> [Float] {
    let url = URL(fileURLWithPath: path)
    guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
          let cg  = CGImageSourceCreateImageAtIndex(src, 0, nil)
    else { throw NSError(domain: "TIPS", code: 1,
                         userInfo: [NSLocalizedDescriptionKey: "Cannot load mask \(path)"]) }

    let origH = cg.height, origW = cg.width
    let hPatches = imgSize / patchSize
    let wPatches = max(1, (origW * imgSize) / (origH * patchSize))
    let resH = hPatches * patchSize
    let resW = wPatches * patchSize

    // Render with alpha
    var buf = [UInt8](repeating: 0, count: resH * resW * 4)
    let cs = CGColorSpaceCreateDeviceRGB()
    guard let ctx = CGContext(data: &buf, width: resW, height: resH,
                              bitsPerComponent: 8, bytesPerRow: resW * 4,
                              space: cs,
                              bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
    else { throw NSError(domain: "TIPS", code: 2,
                         userInfo: [NSLocalizedDescriptionKey: "CGContext failed"]) }
    ctx.draw(cg, in: CGRect(x: 0, y: 0, width: resW, height: resH))

    // Box filter: average alpha in each patchSize×patchSize block
    var patches = [Float](repeating: 0, count: hPatches * wPatches)
    for ph in 0..<hPatches {
        for pw in 0..<wPatches {
            var sum: Float = 0
            for dy in 0..<patchSize {
                for dx in 0..<patchSize {
                    let y = ph * patchSize + dy
                    let x = pw * patchSize + dx
                    sum += Float(buf[(y * resW + x) * 4 + 3]) / 255.0
                }
            }
            patches[ph * wPatches + pw] = sum / Float(patchSize * patchSize)
        }
    }
    return patches
}

/// L2-regularised logistic regression trained with gradient descent using MLX.
/// - Parameters:
///   - xs: (N, D) feature matrix
///   - ys: (N,) binary labels in {0, 1}
///   - C: inverse regularization strength (higher = less regularization)
///   - maxIter: number of gradient steps
///   - lr: learning rate
/// - Returns: (weights (D,), bias (1,))
func trainBinaryLR(xs: MLXArray, ys: MLXArray, C: Float = 0.1, maxIter: Int = 1000, lr: Float = 0.1) -> (MLXArray, MLXArray) {
    let D = xs.dim(1)
    var w = MLXArray.zeros([D])
    var b = MLXArray.zeros([1])

    for _ in 0..<maxIter {
        let logits = (matmul(xs, w.reshaped(D, 1)) + b).reshaped(xs.dim(0))  // (N,)
        // sigmoid
        let probs = 1 / (1 + exp(0 - logits))                               // (N,)
        let err = probs - ys                                                  // (N,)
        let gradW = matmul(xs.transposed(1, 0), err.reshaped(xs.dim(0), 1))
                        .reshaped(D) / Float(xs.dim(0)) + w / C
        let gradB = err.mean(keepDims: true)
        w = w - lr * gradW
        b = b - lr * gradB
    }
    eval(w, b)
    return (w, b)
}

/// Task 6 — Binary foreground segmentation via logistic regression on TIPS patch features.
/// Mirrors TIPS_foreground_segmentation_Demo.ipynb.
func runForegroundSeg(
    model: TIPSv2Model,
    pixels: MLXArray,
    trainDir: URL,
    maskDir: URL,
    outDir: URL,
    imgSize: Int = 448,
    patchSize: Int = 14,
    fgC: Float = 0.1
) throws {
    // Enumerate training images and masks (sorted by name for pairing)
    let imgExts = Set(["jpg", "jpeg", "png"])
    let trainImages = try FileManager.default.contentsOfDirectory(at: trainDir, includingPropertiesForKeys: nil)
        .filter { imgExts.contains($0.pathExtension.lowercased()) }
        .sorted { $0.lastPathComponent < $1.lastPathComponent }
    let maskFiles = try FileManager.default.contentsOfDirectory(at: maskDir, includingPropertiesForKeys: nil)
        .filter { $0.pathExtension.lowercased() == "png" }
        .sorted { $0.lastPathComponent < $1.lastPathComponent }

    guard !trainImages.isEmpty else {
        fputs("  [fg] no training images found in \(trainDir.path)\n", stderr); return
    }
    guard trainImages.count == maskFiles.count else {
        fputs("  [fg] mismatch: \(trainImages.count) images vs \(maskFiles.count) masks\n", stderr); return
    }

    print("  Extracting patch features from \(trainImages.count) training images …")
    var xsFlat = [Float]()
    var ysFlat = [Float]()
    var D = 0

    for (imgFile, maskFile) in zip(trainImages, maskFiles) {
        let trainPixels = try loadImageShortSide(at: imgFile.path, shortSide: imgSize)
        let patchLabels = try loadMaskPatches(at: maskFile.path, imgSize: imgSize, patchSize: patchSize)

        let layers = model.visionEncoder.getIntermediateLayers(trainPixels, n: 1, reshape: false, applyNorm: true)
        let feats = layers[0].patchTokens[0]   // (N, D)
        eval(feats)
        let N = feats.dim(0)
        D = feats.dim(1)
        let featsArr = feats.asArray(Float.self)

        for i in 0..<min(N, patchLabels.count) {
            let label = patchLabels[i]
            if label < 0.01 || label > 0.99 {
                xsFlat.append(contentsOf: featsArr[(i * D)..<((i + 1) * D)])
                ysFlat.append(label > 0.5 ? 1.0 : 0.0)
            }
        }
    }

    guard !xsFlat.isEmpty else {
        fputs("  [fg] no usable patches (all labels ambiguous)\n", stderr); return
    }
    print("  Training logistic regression on \(ysFlat.count) patches (D=\(D), C=\(fgC)) …")

    let xs = MLXArray(xsFlat, [ysFlat.count, D])
    let ys = MLXArray(ysFlat, [ysFlat.count])
    let (w, b) = trainBinaryLR(xs: xs, ys: ys, C: fgC, maxIter: 1000, lr: 0.1)

    // Inference on test image (short-side resize for non-square input)
    let testH = pixels.dim(1), testW = pixels.dim(2)
    let layers = model.visionEncoder.getIntermediateLayers(pixels, n: 1, reshape: false, applyNorm: true)
    let testFeats = layers[0].patchTokens[0]   // (N, D)
    eval(testFeats)

    let hPatches = testH / patchSize
    let wPatches = testW / patchSize

    // Logistic prediction: sigmoid(x @ w + b)
    let logits = (matmul(testFeats, w.reshaped(D, 1)) + b).reshaped(testFeats.dim(0))
    let fgProbs = 1 / (1 + exp(0 - logits))
    eval(fgProbs)
    var fgArr = fgProbs.asArray(Float.self)  // (N,)
    if fgArr.count > hPatches * wPatches { fgArr = Array(fgArr.prefix(hPatches * wPatches)) }

    // Median filter
    let fgFiltered = medianFilter3x3(fgArr, rows: hPatches, cols: wPatches)

    // Save as greyscale PNG
    let fgU8 = fgFiltered.map { UInt8(($0 * 255).clamped(to: 0...255)) }
    var rgba = [UInt8](repeating: 255, count: hPatches * wPatches * 4)
    for i in 0..<(hPatches * wPatches) {
        rgba[i*4] = fgU8[i]; rgba[i*4+1] = fgU8[i]; rgba[i*4+2] = fgU8[i]
    }
    let cs = CGColorSpaceCreateDeviceRGB()
    guard let ctx = CGContext(data: &rgba, width: wPatches, height: hPatches,
                              bitsPerComponent: 8, bytesPerRow: wPatches * 4,
                              space: cs,
                              bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue),
          let cgImg = ctx.makeImage()
    else { throw NSError(domain: "TIPS", code: 3, userInfo: [NSLocalizedDescriptionKey: "CGContext failed"]) }
    let outURL = outDir.appendingPathComponent("fg.png")
    guard let dest = CGImageDestinationCreateWithURL(outURL as CFURL, "public.png" as CFString, 1, nil)
    else { throw NSError(domain: "TIPS", code: 4, userInfo: [NSLocalizedDescriptionKey: "Cannot write \(outURL.path)"]) }
    CGImageDestinationAddImage(dest, cgImg, nil)
    CGImageDestinationFinalize(dest)
    print("  foreground map → \(outURL.path)  (\(hPatches)×\(wPatches) patches)")
}

// MARK: - Entry point

let args = parseArgs()

// Create output directory if needed
try FileManager.default.createDirectory(at: args.outDir, withIntermediateDirectories: true)

// Load image
print("Loading image: \(args.imagePath) (resizing to \(args.imgSize)×\(args.imgSize))")
let pixels = try loadImage(at: args.imagePath, size: args.imgSize)

let tasks: [String] = args.task == "all"
    ? ["zeroshot", "pca", "seg"]
    : [args.task]

// vseg and fg need separate handling — they share the backbone but have different load paths
let needsBackbone = tasks.contains(where: { ["zeroshot","pca","seg","vseg","fg"].contains($0) })

// DPT is a separate model load path
if tasks.contains("dpt") || args.task == "all" {
    guard case .dpt(let dptDir, let bbDir) = args.loadMode else {
        fputs("--task dpt requires --dpt-dir <dir> --backbone-dir <dir>\n", stderr)
        exit(2)
    }
    print("Loading DPT model from \(dptDir.path) (backbone: \(bbDir.path)) …")
    let dptModel = try TIPSv2WeightLoader.loadDPT(
        dptDirectory: dptDir,
        backboneDirectory: bbDir
    )
    print("ready.\n")
    print("--- dpt ---")
    try runDPT(dptModel: dptModel, pixels: pixels, outDir: args.outDir)
    print()
}

// All non-DPT tasks share the same TIPSv2Model
let nonDPTTasks = tasks.filter { $0 != "dpt" }
if !nonDPTTasks.isEmpty && needsBackbone {
    print("Loading TIPSv2 model …")
    let model: TIPSv2Model
    switch args.loadMode {
    case .directory(let dir):
        model = try TIPSv2WeightLoader.load(directory: dir, variant: args.variant)
    case .split(let v, let t, let tok):
        model = try TIPSv2WeightLoader.load(
            visionSafetensorsURL: v,
            textSafetensorsURL: t,
            tokenizerURL: tok,
            variant: args.variant
        )
    case .dpt:
        // Should not reach here — handled above.
        fatalError("unreachable")
    }
    print("ready.\n")

    for task in nonDPTTasks {
        print("--- \(task) ---")
        switch task {
        case "zeroshot":
            try runZeroshot(model: model, pixels: pixels, labels: args.labels)
        case "pca":
            try runPCA(model: model, pixels: pixels, outDir: args.outDir)
        case "seg":
            try runSegmentation(model: model, pixels: pixels, labels: args.labels, outDir: args.outDir)
        case "vseg":
            try runValueSeg(
                model: model,
                pixels: pixels,
                labels: args.labels,
                outDir: args.outDir,
                useSlide: args.useSlide,
                stride: args.stride,
                useTemplates: args.useTemplates,
                imgSize: args.imgSize
            )
        case "fg":
            guard let trainDir = args.trainDir, let maskDir = args.maskDir else {
                fputs("--task fg requires --train-dir <dir> and --mask-dir <dir>\n", stderr)
                exit(2)
            }
            try runForegroundSeg(
                model: model,
                pixels: pixels,
                trainDir: trainDir,
                maskDir: maskDir,
                outDir: args.outDir,
                imgSize: args.imgSize,
                fgC: args.fgC
            )
        default:
            fputs("Unknown task: \(task)\n", stderr)
        }
        print()
    }
}
