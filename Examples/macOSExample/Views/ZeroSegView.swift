import SwiftUI

struct ZeroSegView: View {
    @EnvironmentObject var mm: ModelManager

    @State private var imageURL: URL?
    @State private var classText = "sky, tree, road, building, person"
    @State private var overlayImage: NSImage?
    @State private var maskImage: NSImage?
    @State private var detectedText = ""
    @State private var isBusy = false
    @State private var errorMessage: String?

    var body: some View {
        ZStack {
            HSplitView {
                // Left: input
                VStack(alignment: .leading, spacing: 12) {
                    ImageDropZone(imageURL: $imageURL, label: "Drop image or click to open")
                        .frame(minHeight: 200)

                    VStack(alignment: .leading, spacing: 4) {
                        Text("Class names (comma-separated)")
                            .font(.caption).foregroundStyle(.secondary)
                        TextEditor(text: $classText)
                            .font(.system(.body, design: .monospaced))
                            .frame(height: 80)
                            .border(Color.secondary.opacity(0.3))
                    }

                    Button("Segment") { runSegmentation() }
                        .disabled(imageURL == nil || mm.tipsPipeline == nil || isBusy || classText.trimmingCharacters(in: .whitespaces).isEmpty)
                        .buttonStyle(.borderedProminent)

                    if !detectedText.isEmpty {
                        VStack(alignment: .leading, spacing: 4) {
                            Text("Detected").font(.caption.bold())
                            Text(detectedText)
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                        .padding(8)
                        .background(Color(nsColor: .controlBackgroundColor), in: RoundedRectangle(cornerRadius: 6))
                    }

                    if let err = errorMessage {
                        Text(err).foregroundStyle(.red).font(.caption)
                    }

                    Spacer()
                }
                .padding()
                .frame(minWidth: 240, maxWidth: 340)

                // Right: outputs
                TabView {
                    ResultImageView(image: overlayImage, label: "Overlay")
                        .padding()
                        .tabItem { Text("Overlay") }

                    ResultImageView(image: maskImage, label: "Mask")
                        .padding()
                        .tabItem { Text("Mask") }
                }
                .frame(minWidth: 400)
            }

            if isBusy {
                BusyOverlay(message: "Segmenting…")
            }
        }
    }

    private func runSegmentation() {
        guard let url = imageURL else { return }
        let labels = classText
            .split(separator: ",")
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
        guard !labels.isEmpty else { return }

        isBusy = true
        errorMessage = nil
        Task {
            do {
                let (overlay, mask, detected) = try await mm.zeroShotSeg(imageURL: url, labels: labels)
                overlayImage = overlay
                maskImage = mask
                detectedText = detected
            } catch {
                errorMessage = error.localizedDescription
            }
            isBusy = false
        }
    }
}
