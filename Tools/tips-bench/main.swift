/// `tips-bench` — benchmark the Swift MLX TIPS image encoder.
///
/// Mirrors the protocol of `Benchmarks/torch_tips_bench.py` so the two are
/// directly comparable: same iteration count, same image, same model variant,
/// same dtype.
///
/// Build with `xcodebuild` (not `swift run`) so MLX gets a Metal-capable
/// toolchain:
///
///     xcodebuild -scheme tips-bench -destination 'platform=macOS' \
///         -configuration release -derivedDataPath .xcdd build
///     .xcdd/Build/Products/Release/tips-bench --hf-repo \
///         --variant B --input <image> --iters 20 --warmup 3
///
/// Output is a single-line JSON dict on stdout (machine-aggregable) plus a
/// human-readable summary on stderr.

import Foundation
import CoreGraphics
import ImageIO
import MLX
import MLXTIPS

// MARK: - CLI

struct BenchArgs {
    var variant: TIPSWeightLoader.Variant = .B
    var snapshot: URL?
    var hfRepo: String?    // repo id (e.g., "google/tipsv2-b14") or "auto"
    var inputPath: String = ""
    var iters: Int = 20
    var warmup: Int = 3
    var includeLoad: Bool = false
    var dtype: DType = .float32
}

let variantHFSlug: [String: String] = [
    "B": "b14", "L": "l14", "So400m": "so400m14", "g": "g14",
]

func parseArgs() -> BenchArgs {
    var a = BenchArgs()
    var argv = CommandLine.arguments.dropFirst()

    func next(_ flag: String) -> String {
        guard let v = argv.popFirst() else {
            FileHandle.standardError.write("Missing value after \(flag)\n".data(using: .utf8)!)
            exit(2)
        }
        return v
    }

    while let arg = argv.first {
        switch arg {
        case "--variant":
            argv.removeFirst()
            a.variant = variantFromString(next("--variant"))
        case "--snapshot":
            argv.removeFirst()
            a.snapshot = URL(fileURLWithPath: (next("--snapshot") as NSString).expandingTildeInPath)
        case "--hf-repo":
            argv.removeFirst()
            // Optional inline value; accept "auto" or "<org>/<repo>"
            if let next = argv.first, !next.hasPrefix("--") {
                argv.removeFirst()
                a.hfRepo = next
            } else {
                a.hfRepo = "auto"
            }
        case "--input":
            argv.removeFirst()
            a.inputPath = next("--input")
        case "--iters":
            argv.removeFirst()
            a.iters = Int(next("--iters")) ?? 20
        case "--warmup":
            argv.removeFirst()
            a.warmup = Int(next("--warmup")) ?? 3
        case "--include-load":
            argv.removeFirst()
            a.includeLoad = true
        case "--dtype":
            argv.removeFirst()
            a.dtype = next("--dtype").lowercased() == "fp16" ? .float16 : .float32
        default:
            FileHandle.standardError.write("Unknown flag: \(arg)\n".data(using: .utf8)!)
            exit(2)
        }
    }

    if a.inputPath.isEmpty {
        FileHandle.standardError.write("""
        Usage: tips-bench --input <image> [options]
          --variant B|L|So400m|g     (default: B)
          --hf-repo [<org>/<repo>]   auto-derive when no value
          --snapshot <dir>           HF snapshot dir (alternative to --hf-repo)
          --iters <n>                (default: 20)
          --warmup <n>               (default: 3)
          --include-load             end-to-end mode (re-decode image each iter)
          --dtype fp32|fp16          (default: fp32)
        \n
        """.data(using: .utf8)!)
        exit(2)
    }
    return a
}

func variantFromString(_ s: String) -> TIPSWeightLoader.Variant {
    switch s.lowercased() {
    case "l":           return .L
    case "so400m", "so": return .So400m
    case "g":           return .g
    default:            return .B
    }
}

func variantString(_ v: TIPSWeightLoader.Variant) -> String {
    switch (v.vision, v.text) {
    case (.L, .L):           return "L"
    case (.So400m, .So400m): return "So400m"
    case (.g, .g):           return "g"
    default:                 return "B"
    }
}

// MARK: - Snapshot resolution

func resolveSnapshot(args: BenchArgs) -> URL {
    if let snap = args.snapshot { return snap }
    if let repo = args.hfRepo {
        let repoId = repo.contains("/") ? repo : "google/tipsv2-\(variantHFSlug[variantString(args.variant)] ?? "b14")"
        // Locate the latest snapshot in the HF cache.
        let cache = FileManager.default
            .homeDirectoryForCurrentUser
            .appendingPathComponent(".cache/huggingface/hub")
            .appendingPathComponent("models--\(repoId.replacingOccurrences(of: "/", with: "--"))")
            .appendingPathComponent("snapshots")
        if let snaps = try? FileManager.default.contentsOfDirectory(
            at: cache, includingPropertiesForKeys: [.contentModificationDateKey]
        ),
           let latest = snaps.max(by: { a, b in
               let da = (try? a.resourceValues(forKeys: [.contentModificationDateKey])
                   .contentModificationDate) ?? .distantPast
               let db = (try? b.resourceValues(forKeys: [.contentModificationDateKey])
                   .contentModificationDate) ?? .distantPast
               return da < db
           }) {
            return latest
        }
        FileHandle.standardError.write("""
        --hf-repo set but no snapshot found at \(cache.path).
        Run the Python fixture generator first or pass --snapshot <dir>.
        """.data(using: .utf8)!)
        exit(2)
    }
    FileHandle.standardError.write("Pass --hf-repo or --snapshot.\n".data(using: .utf8)!)
    exit(2)
}

// MARK: - Image loading (matches TIPSPipeline.preprocess byte path)

func loadCGImage(at path: String) -> CGImage {
    let url = URL(fileURLWithPath: path)
    guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
          let cg  = CGImageSourceCreateImageAtIndex(src, 0, nil)
    else {
        FileHandle.standardError.write("Cannot decode \(path)\n".data(using: .utf8)!)
        exit(1)
    }
    return cg
}

// MARK: - Bench

func now() -> Double { CFAbsoluteTimeGetCurrent() }

func percentile(_ xs: [Double], _ p: Double) -> Double {
    let sorted = xs.sorted()
    let idx = Int(Double(sorted.count - 1) * p)
    return sorted[idx]
}

let args = parseArgs()
let snap = resolveSnapshot(args: args)

FileHandle.standardError.write("snapshot: \(snap.path)\n".data(using: .utf8)!)
FileHandle.standardError.write("device:   mlx-metal\n".data(using: .utf8)!)

// Cold model load (timed).
let loadStart = now()
let pipeline = try TIPSPipeline.fromPretrained(directory: snap, variant: args.variant, dtype: args.dtype)
MLX.eval(pipeline.model.parameters())
let loadS = now() - loadStart

let cgImage = loadCGImage(at: args.inputPath)

// Preload tensor when not in --include-load mode.
let preloaded: MLXArray? = args.includeLoad ? nil : try {
    let t = try pipeline.preprocess(cgImage)
    MLX.eval(t)
    return t
}()

func runOnce() throws -> Double {
    let t0 = now()
    let pixels: MLXArray
    if args.includeLoad {
        // Decode + preprocess each iter.
        pixels = try pipeline.preprocess(cgImage)
    } else {
        pixels = preloaded!
    }
    let outputs = pipeline.predict(pixels)
    // `predict` already calls eval, which is the sync point. The CFAbsoluteTime
    // call after this point measures wall-clock for the full forward+sync.
    _ = outputs  // suppress unused warning
    return now() - t0
}

// Warmup
for _ in 0..<args.warmup {
    _ = try runOnce()
}

// Timed loop
var times: [Double] = []
times.reserveCapacity(args.iters)
for _ in 0..<args.iters {
    times.append(try runOnce())
}

let mean   = times.reduce(0, +) / Double(times.count)
let median = percentile(times, 0.5)
let mn     = times.min() ?? 0
let mx     = times.max() ?? 0

let mode = args.includeLoad ? "end-to-end" : "inference-only"
let dtypeStr = args.dtype == .float16 ? "fp16" : "fp32"

// JSON for machine aggregation.
struct Result: Encodable {
    let backend: String
    let variant: String
    let mode: String
    let dtype: String
    let iters: Int
    let warmup: Int
    let load_s: Double
    let times: [Double]
    let mean_s: Double
    let median_s: Double
    let min_s: Double
    let max_s: Double
}

let result = Result(
    backend: "mlx-swift",
    variant: variantString(args.variant),
    mode: mode,
    dtype: dtypeStr,
    iters: args.iters,
    warmup: args.warmup,
    load_s: loadS,
    times: times,
    mean_s: mean,
    median_s: median,
    min_s: mn,
    max_s: mx
)

let json = try JSONEncoder().encode(result)
FileHandle.standardOutput.write(json)
FileHandle.standardOutput.write("\n".data(using: .utf8)!)

let summary = String(
    format: "\n[%@/%@/%@]  median=%.1f ms  mean=%.1f ms  min=%.1f ms  max=%.1f ms  load=%.2f s\n",
    "mlx-swift", mode, dtypeStr,
    median * 1000, mean * 1000, mn * 1000, mx * 1000, loadS
)
FileHandle.standardError.write(summary.data(using: .utf8)!)
