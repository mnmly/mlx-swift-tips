import SwiftUI

struct PCAView: View {
    @EnvironmentObject var mm: ModelManager

    @State private var imageURL: URL?
    @State private var pcaImage: NSImage?
    @State private var depthImage: NSImage?
    @State private var kmeansImage: NSImage?
    @State private var spatial: SpatialFeatures?
    @State private var nClusters = 6
    @State private var isBusy = false
    @State private var busyMessage = ""
    @State private var errorMessage: String?

    var body: some View {
        ZStack {
            HSplitView {
                // Left: input
                VStack(spacing: 12) {
                    ImageDropZone(imageURL: $imageURL, label: "Drop image or click to open")
                        .frame(minHeight: 200)

                    Button("Extract Features") { extractFeatures() }
                        .disabled(imageURL == nil || mm.tipsPipeline == nil || isBusy)
                        .buttonStyle(.borderedProminent)

                    if let err = errorMessage {
                        Text(err).foregroundStyle(.red).font(.caption)
                    }
                }
                .padding()
                .frame(minWidth: 240, maxWidth: 340)

                // Right: outputs
                VStack(spacing: 0) {
                    TabView {
                        ResultImageView(image: pcaImage, label: "PCA → RGB")
                            .padding()
                            .tabItem { Text("PCA") }

                        ResultImageView(image: depthImage, label: "1st PCA component (inferno)")
                            .padding()
                            .tabItem { Text("Depth") }

                        VStack(spacing: 12) {
                            HStack {
                                Text("Clusters: \(nClusters)")
                                Slider(value: Binding(
                                    get: { Double(nClusters) },
                                    set: { nClusters = Int($0) }
                                ), in: 2...20, step: 1)
                                Button("Re-cluster") { reCluster() }
                                    .disabled(spatial == nil || isBusy)
                            }
                            .padding(.horizontal)
                            ResultImageView(image: kmeansImage, label: "K-means clusters")
                                .padding(.horizontal)
                        }
                        .padding(.vertical)
                        .tabItem { Text("K-means") }
                    }
                }
                .frame(minWidth: 400)
            }

            if isBusy {
                BusyOverlay(message: busyMessage)
            }
        }
    }

    private func extractFeatures() {
        guard let url = imageURL else { return }
        isBusy = true
        busyMessage = "Extracting features…"
        errorMessage = nil
        Task {
            do {
                let (pca, depth, sp) = try await mm.extractPCA(imageURL: url)
                pcaImage = pca
                depthImage = depth
                spatial = sp
                // Also compute initial k-means
                busyMessage = "Clustering…"
                kmeansImage = try await mm.computeKMeans(spatial: sp, nClusters: nClusters)
            } catch {
                errorMessage = error.localizedDescription
            }
            isBusy = false
        }
    }

    private func reCluster() {
        guard let sp = spatial else { return }
        isBusy = true
        busyMessage = "Clustering…"
        Task {
            do {
                kmeansImage = try await mm.computeKMeans(spatial: sp, nClusters: nClusters)
            } catch {
                errorMessage = error.localizedDescription
            }
            isBusy = false
        }
    }
}
