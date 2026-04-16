// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "NotebookMirror",
    platforms: [
        .macOS(.v14)
    ],
    dependencies: [
        .package(path: "../.."),
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.17.0"),
    ],
    targets: [
        .executableTarget(
            name: "NotebookMirror",
            dependencies: [
                .product(name: "MLXTIPS", package: "mlx-swift-tipsv2"),
                .product(name: "MLXLinalg", package: "mlx-swift"),
            ],
            path: "."
        ),
    ]
)
