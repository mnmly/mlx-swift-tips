import SwiftUI

struct SegmentationView: View {
    @EnvironmentObject var mm: ModelManager

    @State private var imageURL: URL?
    @State private var segImage: NSImage?
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

                    Text("ADE20K supervised segmentation (150 classes) using a DPT head on a frozen TIPS v2 encoder.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)

                    Button("Segment") { segment() }
                        .disabled(imageURL == nil || mm.dptPipeline == nil || isBusy)
                        .buttonStyle(.borderedProminent)

                    if let err = errorMessage {
                        Text(err).foregroundStyle(.red).font(.caption)
                    }

                    Spacer()
                }
                .padding()
                .frame(minWidth: 240, maxWidth: 340)

                // Right: output
                ResultImageView(image: segImage, label: "DPT Segmentation (ADE20K)")
                    .padding()
                    .frame(minWidth: 400)
            }

            if isBusy {
                BusyOverlay(message: "Running DPT segmentation…")
            }
        }
    }

    private func segment() {
        guard let url = imageURL else { return }
        isBusy = true
        errorMessage = nil
        Task {
            do {
                segImage = try await mm.predictSegmentation(imageURL: url)
            } catch {
                errorMessage = error.localizedDescription
            }
            isBusy = false
        }
    }
}
