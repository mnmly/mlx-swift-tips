import Foundation
import CoreGraphics
import MLX
import MLXLinalg

// MARK: - Spatial features (preview cache)

/// Patch-level features extracted from one image, cached so the GUI can render
/// several visualisations (PCA, PCA-depth, K-means) without recomputing the
/// backbone forward pass.
///
/// `@unchecked Sendable`: the only non-`Sendable` member is the immutable
/// `MLXArray`, which is safe to read across the single detached worker the
/// session contract already assumes (see ``TIPSSession``).
public struct SpatialFeatures: @unchecked Sendable {
    /// `(N, D)` float32 patch features.
    public let feats: MLXArray
    /// Patch grid side length (`sqrt(N)`).
    public let side: Int

    public init(feats: MLXArray, side: Int) {
        self.feats = feats
        self.side = side
    }
}

// MARK: - TIPSRender

/// Pure CoreGraphics rendering of model outputs to `CGImage`. Shared by every
/// frontend so the CLI and the SwiftUI app produce byte-identical images.
///
/// No AppKit / SwiftUI here — `CGImage` is the library's presentation-neutral
/// currency; frontends wrap it (`NSImage(cgImage:)`, `Image(decorative:)`, PNG
/// encode) however they like.
public enum TIPSRender {

    // MARK: PCA

    /// PCA 3-component RGB visualisation (mirrors `vis_pca`): SVD → top-3
    /// principal components → per-channel whiten → `sigmoid(2·z)`.
    public static func pca(_ spatial: SpatialFeatures) -> CGImage {
        let feats = spatial.feats
        let N = feats.dim(0)
        let side = spatial.side

        let centred = feats - feats.mean(axis: 0, keepDims: true)
        MLX.eval(centred)
        let (_, _, vt) = MLXLinalg.svd(centred, stream: .cpu)
        let proj = matmul(centred, vt[..<3].transposed(1, 0))  // (N, 3)
        MLX.eval(proj)
        let projF = proj.asArray(Float.self)

        var rgb = [Float](repeating: 0, count: N * 3)
        for c in 0..<3 {
            var sum: Float = 0
            for i in 0..<N { sum += projF[i * 3 + c] }
            let mu = sum / Float(N)
            var var2: Float = 0
            for i in 0..<N { let d = projF[i * 3 + c] - mu; var2 += d * d }
            let std = max((var2 / Float(N)).squareRoot(), 1e-8)
            for i in 0..<N {
                let z = (projF[i * 3 + c] - mu) / std
                rgb[i * 3 + c] = 1 / (1 + exp(-2 * z))
            }
        }
        let u8 = rgb.map { UInt8((($0 * 255)).clamped(to: 0...255)) }
        return makeImage(rgbU8: u8, width: side, height: side)
    }

    /// 1st PCA component → inferno colormap (mirrors `vis_depth`).
    public static func pcaDepth(_ spatial: SpatialFeatures) -> CGImage {
        let feats = spatial.feats
        let N = feats.dim(0)
        let side = spatial.side

        let centred = feats - feats.mean(axis: 0, keepDims: true)
        MLX.eval(centred)
        let (_, _, vt) = MLXLinalg.svd(centred, stream: .cpu)
        let proj = matmul(centred, vt[..<1].transposed(1, 0))  // (N, 1)
        MLX.eval(proj)
        let projF = proj.asArray(Float.self)

        let mn = projF.min() ?? 0, mx = projF.max() ?? 1
        let range = max(mx - mn, 1e-8)
        var u8 = [UInt8](repeating: 0, count: N * 3)
        for i in 0..<N {
            let (r, g, b) = infernoColor((projF[i] - mn) / range)
            u8[i * 3]     = UInt8((r * 255).clamped(to: 0...255))
            u8[i * 3 + 1] = UInt8((g * 255).clamped(to: 0...255))
            u8[i * 3 + 2] = UInt8((b * 255).clamped(to: 0...255))
        }
        return makeImage(rgbU8: u8, width: side, height: side)
    }

    // MARK: K-means

    /// Lloyd's K-means (20 iters, K-means++ init) over spatial features,
    /// coloured via the tab20 palette.
    public static func kmeans(_ spatial: SpatialFeatures, nClusters: Int) -> CGImage {
        let feats = spatial.feats
        MLX.eval(feats)
        let N = feats.dim(0)
        let D = feats.dim(1)
        let side = spatial.side
        let featsF = feats.asArray(Float.self)

        var centroids = initCentroids(featsF: featsF, N: N, D: D, k: nClusters)
        var assignments = [Int](repeating: 0, count: N)
        for _ in 0..<20 {
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
        return makeImage(rgbU8: u8, width: side, height: side)
    }

    // MARK: Zero-shot segmentation (palette mask + overlay)

    /// Colour a per-patch label grid with the tab20 palette.
    public static func labelMask(
        labelMap: [Int32], gridW: Int, gridH: Int, nLabels: Int
    ) -> CGImage {
        let palette = tab20Palette(n: max(nLabels, 1))
        var u8 = [UInt8](repeating: 0, count: gridH * gridW * 3)
        for i in 0..<(gridH * gridW) {
            let c = min(Int(labelMap[i]), nLabels - 1)
            u8[i * 3]     = palette[c * 3]
            u8[i * 3 + 1] = palette[c * 3 + 1]
            u8[i * 3 + 2] = palette[c * 3 + 2]
        }
        return makeImage(rgbU8: u8, width: gridW, height: gridH)
    }

    /// Blend a per-patch label grid over a base image at `alpha`.
    public static func labelOverlay(
        base: CGImage, labelMap: [Int32], gridW: Int, gridH: Int, nLabels: Int, alpha: Float = 0.6
    ) -> CGImage {
        let palette = tab20Palette(n: max(nLabels, 1))
        var segRGB = [UInt8](repeating: 0, count: gridH * gridW * 3)
        for i in 0..<(gridH * gridW) {
            let c = min(Int(labelMap[i]), nLabels - 1)
            segRGB[i * 3]     = palette[c * 3]
            segRGB[i * 3 + 1] = palette[c * 3 + 1]
            segRGB[i * 3 + 2] = palette[c * 3 + 2]
        }
        return blend(base: base, segRGB: segRGB, gridW: gridW, gridH: gridH, alpha: alpha)
    }

    // MARK: DPT depth / normals / segmentation

    /// `(1, 1, H, W)` depth → min-max normalised turbo colormap.
    public static func depthTurbo(_ depth: MLXArray) -> CGImage {
        let h = depth.dim(2), w = depth.dim(3)
        let flat = depth.squeezed().asArray(Float.self)
        let mn = flat.min() ?? 0, mx = flat.max() ?? 1
        let range = max(mx - mn, 1e-8)
        var u8 = [UInt8](repeating: 0, count: h * w * 3)
        for i in 0..<(h * w) {
            let (r, g, b) = turboColor((flat[i] - mn) / range)
            u8[i * 3]     = UInt8((r * 255).clamped(to: 0...255))
            u8[i * 3 + 1] = UInt8((g * 255).clamped(to: 0...255))
            u8[i * 3 + 2] = UInt8((b * 255).clamped(to: 0...255))
        }
        return makeImage(rgbU8: u8, width: w, height: h)
    }

    /// `(1, 3, H, W)` normals in `[-1, 1]` → RGB (`n·0.5 + 0.5`).
    public static func normals(_ normals: MLXArray) -> CGImage {
        let h = normals.dim(2), w = normals.dim(3)
        let nhwc = normals.transposed(0, 2, 3, 1)  // (1, H, W, 3)
        MLX.eval(nhwc)
        let flat = nhwc.asArray(Float.self)
        var u8 = [UInt8](repeating: 0, count: h * w * 3)
        for i in 0..<(h * w * 3) {
            u8[i] = UInt8(((flat[i] * 0.5 + 0.5) * 255).clamped(to: 0...255))
        }
        return makeImage(rgbU8: u8, width: w, height: h)
    }

    /// `(1, C, H, W)` class logits → argmax → ADE20K palette.
    public static func ade20kSeg(_ seg: MLXArray) -> CGImage {
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
        return makeImage(rgbU8: u8, width: w, height: h)
    }

    // MARK: - Pixel-buffer → CGImage

    /// Build an RGB `CGImage` from a tightly-packed RGB byte buffer.
    public static func makeImage(rgbU8: [UInt8], width: Int, height: Int) -> CGImage {
        var rgba = [UInt8](repeating: 255, count: width * height * 4)
        for i in 0..<(width * height) {
            rgba[i * 4]     = rgbU8[i * 3]
            rgba[i * 4 + 1] = rgbU8[i * 3 + 1]
            rgba[i * 4 + 2] = rgbU8[i * 3 + 2]
        }
        let cs = CGColorSpaceCreateDeviceRGB()
        let ctx = CGContext(
            data: &rgba, width: width, height: height,
            bitsPerComponent: 8, bytesPerRow: width * 4,
            space: cs, bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
        )!
        return ctx.makeImage()!
    }

    private static func blend(
        base: CGImage, segRGB: [UInt8], gridW: Int, gridH: Int, alpha: Float
    ) -> CGImage {
        let cs = CGColorSpaceCreateDeviceRGB()
        var baseBuf = [UInt8](repeating: 0, count: gridW * gridH * 4)
        let ctx = CGContext(
            data: &baseBuf, width: gridW, height: gridH,
            bitsPerComponent: 8, bytesPerRow: gridW * 4,
            space: cs, bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
        )!
        ctx.draw(base, in: CGRect(x: 0, y: 0, width: gridW, height: gridH))

        var blended = [UInt8](repeating: 255, count: gridW * gridH * 4)
        for i in 0..<(gridW * gridH) {
            for ch in 0..<3 {
                let b = Float(baseBuf[i * 4 + ch]) / 255
                let s = Float(segRGB[i * 3 + ch]) / 255
                blended[i * 4 + ch] = UInt8((((1 - alpha) * b + alpha * s) * 255).clamped(to: 0...255))
            }
        }
        let ctx2 = CGContext(
            data: &blended, width: gridW, height: gridH,
            bitsPerComponent: 8, bytesPerRow: gridW * 4,
            space: cs, bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
        )!
        return ctx2.makeImage()!
    }
}

// MARK: - Float clamp

extension Float {
    func clamped(to r: ClosedRange<Float>) -> Float {
        Swift.min(Swift.max(self, r.lowerBound), r.upperBound)
    }
}

// MARK: - Colormaps

func turboColor(_ t: Float) -> (Float, Float, Float) {
    lerpColormap([
        (0.0, 0.189, 0.072, 0.230), (0.1, 0.243, 0.293, 0.806),
        (0.2, 0.182, 0.484, 0.968), (0.3, 0.091, 0.655, 0.843),
        (0.4, 0.003, 0.793, 0.597), (0.5, 0.247, 0.898, 0.350),
        (0.6, 0.659, 0.966, 0.131), (0.7, 0.949, 0.890, 0.101),
        (0.8, 0.988, 0.649, 0.084), (0.9, 0.895, 0.330, 0.064),
        (1.0, 0.478, 0.054, 0.031),
    ], t: t)
}

func infernoColor(_ t: Float) -> (Float, Float, Float) {
    lerpColormap([
        (0.0, 0.001, 0.000, 0.014), (0.1, 0.083, 0.016, 0.133),
        (0.2, 0.221, 0.026, 0.259), (0.3, 0.350, 0.041, 0.302),
        (0.4, 0.490, 0.068, 0.273), (0.5, 0.641, 0.119, 0.181),
        (0.6, 0.779, 0.237, 0.095), (0.7, 0.894, 0.428, 0.032),
        (0.8, 0.970, 0.648, 0.068), (0.9, 0.998, 0.863, 0.318),
        (1.0, 0.988, 0.998, 0.645),
    ], t: t)
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

/// Tab20 palette (20 distinct colours), repeated if `n > 20`.
func tab20Palette(n: Int) -> [UInt8] {
    let base: [(UInt8, UInt8, UInt8)] = [
        (31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
        (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
        (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
        (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
        (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229),
    ]
    var out = [UInt8]()
    out.reserveCapacity(n * 3)
    for i in 0..<n {
        let (r, g, b) = base[i % base.count]
        out.append(r); out.append(g); out.append(b)
    }
    return out
}

private func initCentroids(featsF: [Float], N: Int, D: Int, k: Int) -> [Float] {
    var centroids = [Float](repeating: 0, count: k * D)
    for di in 0..<D { centroids[di] = featsF[di] }
    for c in 1..<k {
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
        for d in 0..<D { centroids[c * D + d] = featsF[bestIdx * D + d] }
    }
    return centroids
}

/// ADE20K 150-class palette (HSV golden-ratio spacing).
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
