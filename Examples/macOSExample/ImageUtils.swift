import Foundation
import AppKit
import CoreGraphics
import ImageIO
import MLX
import MLXLinalg
import MLXTIPS

// MARK: - Image I/O

enum ImageUtils {

    // MARK: Load
    //
    // CGImage→MLXArray decode lives in `TIPSPipeline.preprocess(_:)` (in
    // the library). `loadNSImage` here only loads the AppKit image used as
    // the base for blended segmentation overlays.

    static func loadNSImage(url: URL) throws -> NSImage {
        guard let img = NSImage(contentsOf: url) else { throw TIPSError.imageLoadFailed }
        return img
    }

    // MARK: PCA

    /// PCA 3-component RGB visualisation (mirrors `vis_pca` in app.py).
    /// Uses SVD, projects onto top-3 principal components, then applies sigmoid.
    static func renderPCA(spatial: SpatialFeatures) throws -> NSImage {
        let feats = spatial.feats        // (N, D)
        let N = feats.dim(0)
        let side = spatial.side

        let mean = feats.mean(axis: 0, keepDims: true)
        let centred = feats - mean
        MLX.eval(centred)

        let (_, _, vt) = MLXLinalg.svd(centred, stream: .cpu)
        let top3 = vt[..<3]
        let proj = matmul(centred, top3.transposed(1, 0))  // (N, 3)
        MLX.eval(proj)
        let projF = proj.asArray(Float.self)

        // Per-channel: centre by mean, scale by std, then sigmoid(2*x)
        var rgb = [Float](repeating: 0, count: N * 3)
        for c in 0..<3 {
            var sum: Float = 0
            for i in 0..<N { sum += projF[i * 3 + c] }
            let mu = sum / Float(N)
            var var2: Float = 0
            for i in 0..<N { let d = projF[i * 3 + c] - mu; var2 += d * d }
            let std = max(Float(var2 / Float(N)).squareRoot(), 1e-8)
            for i in 0..<N {
                let z = (projF[i * 3 + c] - mu) / std  // whitened
                rgb[i * 3 + c] = 1 / (1 + exp(-2 * z)) // sigmoid
            }
        }

        let u8 = rgb.map { UInt8(($0 * 255).clamped(to: 0...255)) }
        return makeNSImage(rgbU8: u8, width: side, height: side)
    }

    /// 1st PCA component → inferno colormap (mirrors `vis_depth` in app.py).
    static func renderPCADepth(spatial: SpatialFeatures) throws -> NSImage {
        let feats = spatial.feats
        let N = feats.dim(0)
        let side = spatial.side

        let mean = feats.mean(axis: 0, keepDims: true)
        let centred = feats - mean
        MLX.eval(centred)

        let (_, _, vt) = MLXLinalg.svd(centred, stream: .cpu)
        let top1 = vt[..<1]
        let proj = matmul(centred, top1.transposed(1, 0))  // (N, 1)
        MLX.eval(proj)
        let projF = proj.asArray(Float.self)

        let mn = projF.min() ?? 0, mx = projF.max() ?? 1
        let range = max(mx - mn, 1e-8)

        var u8 = [UInt8](repeating: 0, count: N * 3)
        for i in 0..<N {
            let t = (projF[i] - mn) / range
            let (r, g, b) = infernoColor(t)
            u8[i * 3]     = UInt8((r * 255).clamped(to: 0...255))
            u8[i * 3 + 1] = UInt8((g * 255).clamped(to: 0...255))
            u8[i * 3 + 2] = UInt8((b * 255).clamped(to: 0...255))
        }
        return makeNSImage(rgbU8: u8, width: side, height: side)
    }

    // MARK: K-means

    /// Simple Lloyd's K-means on spatial features, coloured via tab20 palette.
    static func renderKMeans(spatial: SpatialFeatures, nClusters: Int) throws -> NSImage {
        let feats = spatial.feats   // (N, D)
        MLX.eval(feats)
        let N = feats.dim(0)
        let side = spatial.side
        let featsF = feats.asArray(Float.self)
        let D = feats.dim(1)

        // K-means: 20 iterations, random init from data
        var centroids = initCentroids(featsF: featsF, N: N, D: D, k: nClusters)
        var assignments = [Int](repeating: 0, count: N)

        for _ in 0..<20 {
            // Assign
            for i in 0..<N {
                var bestDist = Float.greatestFiniteMagnitude
                var bestC = 0
                for c in 0..<nClusters {
                    var dist: Float = 0
                    for d in 0..<D {
                        let diff = featsF[i * D + d] - centroids[c * D + d]
                        dist += diff * diff
                    }
                    if dist < bestDist { bestDist = dist; bestC = c }
                }
                assignments[i] = bestC
            }
            // Update
            var newCentroids = [Float](repeating: 0, count: nClusters * D)
            var counts = [Int](repeating: 0, count: nClusters)
            for i in 0..<N {
                let c = assignments[i]
                counts[c] += 1
                for d in 0..<D { newCentroids[c * D + d] += featsF[i * D + d] }
            }
            for c in 0..<nClusters where counts[c] > 0 {
                for d in 0..<D { newCentroids[c * D + d] /= Float(counts[c]) }
            }
            centroids = newCentroids
        }

        let palette = tab20Palette(n: nClusters)
        var u8 = [UInt8](repeating: 0, count: N * 3)
        for i in 0..<N {
            let c = assignments[i]
            u8[i * 3]     = palette[c * 3]
            u8[i * 3 + 1] = palette[c * 3 + 1]
            u8[i * 3 + 2] = palette[c * 3 + 2]
        }
        return makeNSImage(rgbU8: u8, width: side, height: side)
    }

    // MARK: Segmentation overlay

    static func renderSegmentation(
        labelMap: [Int32],
        gridH: Int, gridW: Int,
        labels: [String],
        origImage: NSImage
    ) -> (overlay: NSImage, mask: NSImage, detected: String) {
        let n = labels.count
        let palette = tab20Palette(n: max(n, 1))

        var u8 = [UInt8](repeating: 0, count: gridH * gridW * 3)
        for i in 0..<(gridH * gridW) {
            let c = min(Int(labelMap[i]), n - 1)
            u8[i * 3]     = palette[c * 3]
            u8[i * 3 + 1] = palette[c * 3 + 1]
            u8[i * 3 + 2] = palette[c * 3 + 2]
        }
        let mask = makeNSImage(rgbU8: u8, width: gridW, height: gridH)

        // Build overlay (blend)
        let overlay = blendImages(base: origImage, segRGB: u8, gridW: gridW, gridH: gridH, alpha: 0.6)

        // Count dominant labels
        var counts = [Int: Int]()
        for v in labelMap { counts[Int(v), default: 0] += 1 }
        let total = labelMap.count
        let sorted = counts.sorted { $0.value > $1.value }
        let detected = sorted
            .filter { Float($0.value) / Float(total) >= 0.02 && $0.key < labels.count }
            .prefix(8)
            .map { "\(labels[$0.key]) (\(String(format: "%.1f", Float($0.value) / Float(total) * 100))%)" }
            .joined(separator: ", ")

        return (overlay, mask, detected)
    }

    // MARK: DPT Depth (turbo colormap)

    static func renderDepthTurbo(depth: MLXArray) -> NSImage {
        // depth: (1, 1, H, W)
        let h = depth.dim(2), w = depth.dim(3)
        let flat = depth.squeezed().asArray(Float.self)
        let mn = flat.min() ?? 0, mx = flat.max() ?? 1
        let range = max(mx - mn, 1e-8)

        var u8 = [UInt8](repeating: 0, count: h * w * 3)
        for i in 0..<(h * w) {
            let t = (flat[i] - mn) / range
            let (r, g, b) = turboColor(t)
            u8[i * 3]     = UInt8((r * 255).clamped(to: 0...255))
            u8[i * 3 + 1] = UInt8((g * 255).clamped(to: 0...255))
            u8[i * 3 + 2] = UInt8((b * 255).clamped(to: 0...255))
        }
        return makeNSImage(rgbU8: u8, width: w, height: h)
    }

    // MARK: DPT Normals

    static func renderNormals(normals: MLXArray) -> NSImage {
        // normals: (1, 3, H, W), values in [-1, 1]
        let h = normals.dim(2), w = normals.dim(3)
        // NCHW → NHWC
        let nhwc = normals.transposed(0, 2, 3, 1)  // (1, H, W, 3)
        MLX.eval(nhwc)
        let flat = nhwc.asArray(Float.self)
        var u8 = [UInt8](repeating: 0, count: h * w * 3)
        for i in 0..<(h * w * 3) {
            u8[i] = UInt8(((flat[i] * 0.5 + 0.5) * 255).clamped(to: 0...255))
        }
        return makeNSImage(rgbU8: u8, width: w, height: h)
    }

    // MARK: DPT Segmentation (ADE20K)

    static func renderADE20KSeg(seg: MLXArray) -> NSImage {
        // seg: (1, C, H, W) — argmax over C
        let h = seg.dim(2), w = seg.dim(3)
        let labels = argMax(seg[0], axis: 0)  // (H, W)
        MLX.eval(labels)
        let labelArr = labels.asArray(Int32.self)

        var u8 = [UInt8](repeating: 0, count: h * w * 3)
        for i in 0..<(h * w) {
            let cls = min(Int(labelArr[i]), ade20kPalette.count / 3 - 1)
            u8[i * 3]     = ade20kPalette[cls * 3]
            u8[i * 3 + 1] = ade20kPalette[cls * 3 + 1]
            u8[i * 3 + 2] = ade20kPalette[cls * 3 + 2]
        }
        return makeNSImage(rgbU8: u8, width: w, height: h)
    }

    // MARK: - Helpers

    static func makeNSImage(rgbU8: [UInt8], width: Int, height: Int) -> NSImage {
        var rgba = [UInt8](repeating: 255, count: width * height * 4)
        for i in 0..<(width * height) {
            rgba[i * 4]     = rgbU8[i * 3]
            rgba[i * 4 + 1] = rgbU8[i * 3 + 1]
            rgba[i * 4 + 2] = rgbU8[i * 3 + 2]
        }
        let cs = CGColorSpaceCreateDeviceRGB()
        guard var mutableRGBA = Optional(rgba),
              let ctx = CGContext(
                data: &mutableRGBA,
                width: width, height: height,
                bitsPerComponent: 8, bytesPerRow: width * 4,
                space: cs,
                bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
              ),
              let cgImg = ctx.makeImage()
        else { return NSImage() }
        return NSImage(cgImage: cgImg, size: NSSize(width: width, height: height))
    }

    private static func blendImages(base: NSImage, segRGB: [UInt8], gridW: Int, gridH: Int, alpha: Float) -> NSImage {
        // Resize orig image to grid size and blend
        let cs = CGColorSpaceCreateDeviceRGB()
        var baseBuf = [UInt8](repeating: 0, count: gridW * gridH * 4)
        guard let ctx = CGContext(
            data: &baseBuf, width: gridW, height: gridH,
            bitsPerComponent: 8, bytesPerRow: gridW * 4,
            space: cs, bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
        ) else { return base }
        if let cgImg = base.cgImage(forProposedRect: nil, context: nil, hints: nil) {
            ctx.draw(cgImg, in: CGRect(x: 0, y: 0, width: gridW, height: gridH))
        }

        var blended = [UInt8](repeating: 255, count: gridW * gridH * 4)
        for i in 0..<(gridW * gridH) {
            let br = Float(baseBuf[i * 4])     / 255
            let bg = Float(baseBuf[i * 4 + 1]) / 255
            let bb = Float(baseBuf[i * 4 + 2]) / 255
            let sr = Float(segRGB[i * 3])      / 255
            let sg = Float(segRGB[i * 3 + 1])  / 255
            let sb = Float(segRGB[i * 3 + 2])  / 255
            blended[i * 4]     = UInt8(((1 - alpha) * br + alpha * sr) * 255)
            blended[i * 4 + 1] = UInt8(((1 - alpha) * bg + alpha * sg) * 255)
            blended[i * 4 + 2] = UInt8(((1 - alpha) * bb + alpha * sb) * 255)
        }
        guard let ctx2 = CGContext(
            data: &blended, width: gridW, height: gridH,
            bitsPerComponent: 8, bytesPerRow: gridW * 4,
            space: cs, bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
        ), let cgImg = ctx2.makeImage()
        else { return base }
        return NSImage(cgImage: cgImg, size: NSSize(width: gridW, height: gridH))
    }
}

// MARK: - Float clamping

extension Float {
    func clamped(to r: ClosedRange<Float>) -> Float { Swift.min(Swift.max(self, r.lowerBound), r.upperBound) }
}

// MARK: - Colormap helpers

/// Turbo colormap interpolated from control points.
func turboColor(_ t: Float) -> (Float, Float, Float) {
    let pts: [(Float, Float, Float, Float)] = [
        (0.0,  0.189, 0.072, 0.230),
        (0.1,  0.243, 0.293, 0.806),
        (0.2,  0.182, 0.484, 0.968),
        (0.3,  0.091, 0.655, 0.843),
        (0.4,  0.003, 0.793, 0.597),
        (0.5,  0.247, 0.898, 0.350),
        (0.6,  0.659, 0.966, 0.131),
        (0.7,  0.949, 0.890, 0.101),
        (0.8,  0.988, 0.649, 0.084),
        (0.9,  0.895, 0.330, 0.064),
        (1.0,  0.478, 0.054, 0.031),
    ]
    return lerpColormap(pts, t: t)
}

/// Inferno colormap interpolated from control points.
func infernoColor(_ t: Float) -> (Float, Float, Float) {
    let pts: [(Float, Float, Float, Float)] = [
        (0.0, 0.001, 0.000, 0.014),
        (0.1, 0.083, 0.016, 0.133),
        (0.2, 0.221, 0.026, 0.259),
        (0.3, 0.350, 0.041, 0.302),
        (0.4, 0.490, 0.068, 0.273),
        (0.5, 0.641, 0.119, 0.181),
        (0.6, 0.779, 0.237, 0.095),
        (0.7, 0.894, 0.428, 0.032),
        (0.8, 0.970, 0.648, 0.068),
        (0.9, 0.998, 0.863, 0.318),
        (1.0, 0.988, 0.998, 0.645),
    ]
    return lerpColormap(pts, t: t)
}

private func lerpColormap(_ pts: [(Float, Float, Float, Float)], t: Float) -> (Float, Float, Float) {
    let tc = Swift.max(0, Swift.min(1, t))
    for i in 0..<(pts.count - 1) {
        let (t0, r0, g0, b0) = pts[i]
        let (t1, r1, g1, b1) = pts[i + 1]
        if tc <= t1 {
            let f = (tc - t0) / (t1 - t0)
            return (r0 + f * (r1 - r0), g0 + f * (g1 - g0), b0 + f * (b1 - b0))
        }
    }
    let last = pts.last!
    return (last.1, last.2, last.3)
}

/// Tab20 palette (20 distinct colours), repeated if n > 20.
func tab20Palette(n: Int) -> [UInt8] {
    let base: [(UInt8, UInt8, UInt8)] = [
        (31,119,180),(174,199,232),(255,127,14),(255,187,120),
        (44,160,44),(152,223,138),(214,39,40),(255,152,150),
        (148,103,189),(197,176,213),(140,86,75),(196,156,148),
        (227,119,194),(247,182,210),(127,127,127),(199,199,199),
        (188,189,34),(219,219,141),(23,190,207),(158,218,229),
    ]
    var out = [UInt8]()
    out.reserveCapacity(n * 3)
    for i in 0..<n {
        let (r, g, b) = base[i % base.count]
        out.append(r); out.append(g); out.append(b)
    }
    return out
}

// MARK: - K-means init (K-means++ style: pick first randomly then furthest)

private func initCentroids(featsF: [Float], N: Int, D: Int, k: Int) -> [Float] {
    var centroids = [Float](repeating: 0, count: k * D)
    // Pick first centroid: index 0
    var chosen = [Int]()
    chosen.append(0)
    for di in 0..<D { centroids[di] = featsF[di] }

    for c in 1..<k {
        // Find point furthest from nearest centroid
        var bestDist: Float = -1
        var bestIdx = 0
        for i in 0..<N {
            var minD = Float.greatestFiniteMagnitude
            for ci in 0..<c {
                var dist: Float = 0
                for d in 0..<D {
                    let diff = featsF[i * D + d] - centroids[ci * D + d]
                    dist += diff * diff
                }
                if dist < minD { minD = dist }
            }
            if minD > bestDist { bestDist = minD; bestIdx = i }
        }
        chosen.append(bestIdx)
        for d in 0..<D { centroids[c * D + d] = featsF[bestIdx * D + d] }
    }
    return centroids
}

// MARK: - ADE20K Palette (150 classes, HSV golden-ratio spacing)

let ade20kPalette: [UInt8] = {
    var out = [UInt8](repeating: 0, count: 151 * 3)
    for i in 1...150 {
        let hue = Float(i) * 0.618033988749895
        let hue1 = hue - floor(hue)
        let sat = Float(0.65 + 0.35 * Float((i * 7) % 5) / 4.0)
        let val = Float(0.70 + 0.30 * Float((i * 11) % 3) / 2.0)
        let (r, g, b) = hsvToRgb(h: hue1, s: sat, v: val)
        out[i * 3]     = UInt8((r * 255).clamped(to: 0...255))
        out[i * 3 + 1] = UInt8((g * 255).clamped(to: 0...255))
        out[i * 3 + 2] = UInt8((b * 255).clamped(to: 0...255))
    }
    return out
}()

private func hsvToRgb(h: Float, s: Float, v: Float) -> (Float, Float, Float) {
    let i = Int(h * 6)
    let f = h * 6 - Float(i)
    let p = v * (1 - s)
    let q = v * (1 - f * s)
    let t = v * (1 - (1 - f) * s)
    switch i % 6 {
    case 0: return (v, t, p)
    case 1: return (q, v, p)
    case 2: return (p, v, t)
    case 3: return (p, q, v)
    case 4: return (t, p, v)
    default: return (v, p, q)
    }
}
