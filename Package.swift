// swift-tools-version: 5.9
// Swift port of the Python MLX TIPSv2 implementation:
// https://github.com/mnmly/mlx-tips
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
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.17.0"),
        .package(url: "https://github.com/jkrukowski/swift-sentencepiece", from: "0.0.6")
    ],
    targets: [
        .target(
            name: "MLXTIPS",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "SentencepieceTokenizer", package: "swift-sentencepiece"),
            ],
            path: "Sources/TIPS"
        ),
        .testTarget(
            name: "MLXTIPSTests",
            dependencies: ["MLXTIPS"],
            path: "Tests/TIPSTests"
        ),
    ]
)
