import SwiftUI
import UniformTypeIdentifiers
import MLXTIPS

// MARK: - Variant enum for the picker (wraps TIPSWeightLoader.Variant)

enum TIPSVariantOption: String, CaseIterable, Identifiable, Hashable {
    case B, L, So400m, g
    var id: String { rawValue }
    var label: String {
        switch self {
        case .B:      return "B/14"
        case .L:      return "L/14"
        case .So400m: return "SO400m"
        case .g:      return "g/14"
        }
    }
    var loaderVariant: TIPSWeightLoader.Variant {
        switch self {
        case .B:      return .B
        case .L:      return .L
        case .So400m: return .So400m
        case .g:      return .g
        }
    }

    /// Map a Hub repo (or its DPT counterpart) to the matching picker option.
    init(hubRepo: TIPSHub.Repo) {
        switch hubRepo.backboneRepo {
        case .l14:      self = .L
        case .so400m14: self = .So400m
        case .g14:      self = .g
        default:        self = .B
        }
    }
}

struct ContentView: View {
    @EnvironmentObject var modelManager: ModelManager

    var body: some View {
        TabView {
            PCAView()
                .tabItem { Label("PCA & Features", systemImage: "chart.bar.xaxis") }

            ZeroSegView()
                .tabItem { Label("Zero-shot Seg", systemImage: "paintbrush.pointed") }

            DepthNormalsView()
                .tabItem { Label("Depth & Normals", systemImage: "mountain.2") }

            SegmentationView()
                .tabItem { Label("Segmentation", systemImage: "theatermasks") }
        }
        .toolbar {
            ToolbarItemGroup(placement: .automatic) {
                ModelControlsView()
            }
        }
        .padding(8)
    }
}

// MARK: - Shared model controls toolbar

struct ModelControlsView: View {
    @EnvironmentObject var mm: ModelManager

    var body: some View {
        HStack(spacing: 12) {
            // Variant picker
            Picker("Variant", selection: $mm.tipsVariantOption) {
                ForEach(TIPSVariantOption.allCases) { opt in
                    Text(opt.label).tag(opt)
                }
            }
            .pickerStyle(.menu)
            .frame(width: 110)

            // Resolution picker
            Picker("Resolution", selection: $mm.resolution) {
                ForEach([224, 336, 448, 672, 896, 1120, 1344], id: \.self) { r in
                    Text("\(r)").tag(r)
                }
            }
            .pickerStyle(.menu)
            .frame(width: 80)

            // Download from Hugging Face
            Menu {
                Section("Backbone") {
                    ForEach([TIPSHub.Repo.b14, .l14, .so400m14, .g14], id: \.self) { repo in
                        Button(repo.displayName) { chooseCacheAndDownload(repo) }
                    }
                }
                Section("Backbone + DPT heads") {
                    ForEach([TIPSHub.Repo.b14dpt, .l14dpt, .so400m14dpt, .g14dpt], id: \.self) { repo in
                        Button(repo.displayName) { chooseCacheAndDownload(repo) }
                    }
                }
            } label: {
                Label(
                    mm.isDownloading ? "Downloading \(Int(mm.downloadProgress * 100))%" : "Download",
                    systemImage: "icloud.and.arrow.down"
                )
            }
            .disabled(mm.isDownloading)

            // Load TIPS button
            Button {
                let panel = NSOpenPanel()
                panel.canChooseFiles = false
                panel.canChooseDirectories = true
                panel.prompt = "Select TIPS model directory"
                panel.directoryURL = mm.cacheDir ?? TIPSHub.defaultCacheRoot
                if panel.runModal() == .OK, let url = panel.url {
                    mm.loadTIPS(directory: url)
                }
            } label: {
                Label(
                    mm.isLoadingTIPS ? "Loading…" : (mm.tipsPipeline != nil ? "TIPS: Loaded" : "Load TIPS"),
                    systemImage: mm.tipsPipeline != nil ? "checkmark.circle.fill" : "folder"
                )
            }
            .disabled(mm.isLoadingTIPS)

            // Load DPT button
            Button {
                let panel = NSOpenPanel()
                panel.canChooseFiles = false
                panel.canChooseDirectories = true
                panel.prompt = "Select DPT model directory"
                panel.directoryURL = mm.cacheDir ?? TIPSHub.defaultCacheRoot
                if panel.runModal() == .OK, let dptURL = panel.url {
                    let panel2 = NSOpenPanel()
                    panel2.canChooseFiles = false
                    panel2.canChooseDirectories = true
                    panel2.prompt = "Select backbone directory"
                    panel2.directoryURL = mm.cacheDir ?? TIPSHub.defaultCacheRoot
                    if panel2.runModal() == .OK, let bbURL = panel2.url {
                        mm.loadDPT(dptDirectory: dptURL, backboneDirectory: bbURL)
                    }
                }
            } label: {
                Label(
                    mm.isLoadingDPT ? "Loading…" : (mm.dptPipeline != nil ? "DPT: Loaded" : "Load DPT"),
                    systemImage: mm.dptPipeline != nil ? "checkmark.circle.fill" : "folder"
                )
            }
            .disabled(mm.isLoadingDPT)

            Text(mm.statusMessage)
                .foregroundStyle(.secondary)
                .lineLimit(1)
                .frame(maxWidth: 200)
        }
    }

    /// Resolve a writable cache folder (reusing a prior grant, else prompting
    /// with the HF cache as the default), then download into it. The panel
    /// grant is what lets the sandboxed app write to `~/.cache/huggingface`.
    private func chooseCacheAndDownload(_ repo: TIPSHub.Repo) {
        if let dir = mm.cacheDir {
            mm.download(repo: repo, into: dir)
            return
        }
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.canCreateDirectories = true
        panel.prompt = "Use as Cache"
        panel.message = "Choose the Hugging Face cache folder to download models into."
        panel.directoryURL = TIPSHub.defaultCacheRoot
        if panel.runModal() == .OK, let url = panel.url {
            mm.setCacheDir(url)   // persist the grant for next launch
            mm.download(repo: repo, into: url)
        }
    }
}

// MARK: - Reusable drop zone / image picker

struct ImageDropZone: View {
    @Binding var imageURL: URL?
    let label: String

    var body: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 10)
                .fill(Color(nsColor: .controlBackgroundColor))
                .overlay(
                    RoundedRectangle(cornerRadius: 10)
                        .stroke(style: StrokeStyle(lineWidth: 1.5, dash: [6]))
                        .foregroundStyle(.secondary)
                )

            if let url = imageURL, let nsImg = NSImage(contentsOf: url) {
                Image(nsImage: nsImg)
                    .resizable()
                    .scaledToFit()
                    .padding(4)
            } else {
                VStack(spacing: 8) {
                    Image(systemName: "photo")
                        .font(.largeTitle)
                        .foregroundStyle(.secondary)
                    Text(label)
                        .foregroundStyle(.secondary)
                }
            }
        }
        .onTapGesture { pickImage() }
        .dropDestination(for: URL.self) { urls, _ in
            guard let url = urls.first else { return false }
            imageURL = url
            return true
        }
        .contextMenu {
            Button("Choose Image…") { pickImage() }
            if imageURL != nil {
                Button("Clear") { imageURL = nil }
            }
        }
    }

    private func pickImage() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.image]
        panel.canChooseDirectories = false
        if panel.runModal() == .OK { imageURL = panel.url }
    }
}

// MARK: - Output image view

struct ResultImageView: View {
    let image: NSImage?
    let label: String

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(label)
                .font(.caption)
                .foregroundStyle(.secondary)
            ZStack {
                RoundedRectangle(cornerRadius: 8)
                    .fill(Color(nsColor: .controlBackgroundColor))
                if let img = image {
                    Image(nsImage: img)
                        .resizable()
                        .scaledToFit()
                        .padding(2)
                } else {
                    Text("—")
                        .foregroundStyle(.tertiary)
                }
            }
        }
    }
}

// MARK: - Loading overlay

struct BusyOverlay: View {
    let message: String

    var body: some View {
        ZStack {
            Color.black.opacity(0.25).ignoresSafeArea()
            VStack(spacing: 12) {
                ProgressView()
                Text(message)
                    .font(.headline)
            }
            .padding(24)
            .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 12))
        }
    }
}
