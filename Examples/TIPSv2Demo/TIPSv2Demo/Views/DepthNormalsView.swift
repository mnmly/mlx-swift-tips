import SwiftUI

struct DepthNormalsView: View {
    @EnvironmentObject var mm: ModelManager

    @State private var imageURL: URL?
    @State private var depthImage: NSImage?
    @State private var normalsImage: NSImage?
    @State private var isBusy = false
    @State private var errorMessage: String?

    var body: some View {
        ZStack {
            HSplitView {
                // Left: input
                VStack(spacing: 12) {
                    ImageDropZone(imageURL: $imageURL, label: "Drop image or click to open")
                        .frame(minHeight: 200)

                    if mm.dptPipeline == nil {
                        Text("Load a DPT model to use this tab.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .multilineTextAlignment(.center)
                    }

                    Button("Predict Depth & Normals") { predict() }
                        .disabled(imageURL == nil || mm.dptPipeline == nil || isBusy)
                        .buttonStyle(.borderedProminent)

                    if let err = errorMessage {
                        Text(err).foregroundStyle(.red).font(.caption)
                    }

                    Spacer()
                }
                .padding()
                .frame(minWidth: 240, maxWidth: 340)

                // Right: outputs side by side
                HStack(spacing: 12) {
                    ResultImageView(image: depthImage, label: "Depth (turbo)")
                    ResultImageView(image: normalsImage, label: "Surface Normals")
                }
                .padding()
                .frame(minWidth: 400)
            }

            if isBusy {
                BusyOverlay(message: "Running DPT depth & normals…")
            }
        }
    }

    private func predict() {
        guard let url = imageURL else { return }
        isBusy = true
        errorMessage = nil
        Task {
            do {
                let (depth, normals) = try await mm.predictDepthNormals(imageURL: url)
                depthImage = depth
                normalsImage = normals
            } catch {
                errorMessage = error.localizedDescription
            }
            isBusy = false
        }
    }
}
