// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "mlx-swift-tipsv2",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .library(
            name: "MLXTIPSv2",
            targets: ["MLXTIPSv2"]
        ),
        .executable(
            name: "tipsv2-example",
            targets: ["TIPSv2Example"]
        ),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.17.0"),
        .package(path: "../swift-sentencepiece"),
    ],
    targets: [
        .target(
            name: "MLXTIPSv2",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "SentencepieceTokenizer", package: "swift-sentencepiece"),
            ],
            path: "Sources/TIPSv2"
        ),
        .executableTarget(
            name: "TIPSv2Example",
            dependencies: [
                "MLXTIPSv2",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXLinalg", package: "mlx-swift"),
            ],
            path: "Sources/TIPSv2Example"
        ),
        .testTarget(
            name: "MLXTIPSv2Tests",
            dependencies: ["MLXTIPSv2"],
            path: "Tests/TIPSv2Tests"
        ),
    ]
)
