// swift-tools-version: 5.9
// Swift port of TIPSv2. Upstream provenance (see README → "Porting provenance"):
//   google-deepmind/tips @ 4db271d  — PyTorch reference (DPT "Scenic parity")
import PackageDescription

let package = Package(
    name: "mlx-swift-tips",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .library(
            name: "MLXTIPS",
            targets: ["MLXTIPS"]
        ),
        .executable(
            name: "tips-bench",
            targets: ["tips-bench"]
        ),
        .executable(
            name: "tips-cli",
            targets: ["tips-cli"]
        ),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.17.0"),
        .package(url: "https://github.com/jkrukowski/swift-sentencepiece", from: "0.0.6"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.0"),
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.2.0"),
        .package(url: "https://github.com/swiftlang/swift-docc-plugin", from: "1.4.3"),
    ],
    targets: [
        .target(
            name: "MLXTIPS",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXLinalg", package: "mlx-swift"),
                .product(name: "SentencepieceTokenizer", package: "swift-sentencepiece"),
                .product(name: "Hub", package: "swift-transformers"),
            ],
            path: "Sources/TIPS"
        ),
        .executableTarget(
            name: "tips-bench",
            dependencies: [
                "MLXTIPS",
                .product(name: "MLX", package: "mlx-swift"),
            ],
            path: "Tools/tips-bench"
        ),
        .executableTarget(
            name: "tips-cli",
            dependencies: [
                "MLXTIPS",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
            path: "Examples/tips-cli"
        ),
        .testTarget(
            name: "MLXTIPSTests",
            dependencies: ["MLXTIPS"],
            path: "Tests/TIPSTests"
        ),
    ]
)
