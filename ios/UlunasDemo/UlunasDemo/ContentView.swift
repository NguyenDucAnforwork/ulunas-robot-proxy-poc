import SwiftUI

struct ContentView: View {
    @StateObject private var audio = AudioEngineManager()
    @State private var selectedBackend: UlunasBackend = .runtimeFreeC
    @State private var errorMessage: String?
    @State private var savedFiles: (noisy: URL, enhanced: URL)?

    var body: some View {
        NavigationView {
            Form {
                Section("Feedback risk") {
                    if audio.feedbackRiskWarning {
                        Label("No wired headphones detected. This project has NO echo cancellation -- playing enhanced audio through the built-in speaker while the mic is open WILL feed back into the mic. Connect wired (Lightning) headphones before starting.", systemImage: "exclamationmark.triangle.fill")
                            .foregroundColor(.red)
                    } else {
                        Label("Wired/Bluetooth output detected: \(audio.currentRoute)", systemImage: "checkmark.circle.fill")
                            .foregroundColor(.green)
                    }
                }

                Section("Backend") {
                    Picker("Backend", selection: $selectedBackend) {
                        Text("Runtime-free C (primary)").tag(UlunasBackend.runtimeFreeC)
                        Text("ONNX Runtime (reference)").tag(UlunasBackend.onnxRuntime)
                    }
                    .disabled(audio.isRunning)
                }

                Section("Control") {
                    Toggle("Bypass (passthrough, no enhancement)", isOn: $audio.isBypassed)
                    Button(audio.isRunning ? "Stop" : "Start") {
                        toggleRunning()
                    }
                    if let errorMessage {
                        Text(errorMessage).foregroundColor(.red).font(.caption)
                    }
                }

                Section("Audio route (actual, not assumed)") {
                    LabeledContent("Input sample rate", value: "\(Int(audio.inputSampleRate)) Hz")
                    LabeledContent("Input channels", value: "\(audio.inputChannelCount)")
                    LabeledContent("Current route", value: audio.currentRoute)
                }

                Section("Live stats") {
                    LabeledContent("Mean inference / hop", value: String(format: "%.2f ms", audio.meanInferenceMs))
                    LabeledContent("P95 inference / hop", value: String(format: "%.2f ms", audio.p95InferenceMs))
                    LabeledContent("RTF", value: String(format: "%.3f", audio.rtf))
                    LabeledContent("Underruns", value: "\(audio.underrunCount)")
                    LabeledContent("Overruns", value: "\(audio.overrunCount)")
                }

                Section("Recording") {
                    Button("Save noisy + enhanced WAV") {
                        saveRecording()
                    }
                    .disabled(!audio.isRunning)
                    if let savedFiles {
                        Text("Saved:\n\(savedFiles.noisy.lastPathComponent)\n\(savedFiles.enhanced.lastPathComponent)")
                            .font(.caption)
                    }
                }
            }
            .navigationTitle("UL-UNAS Demo")
            .onAppear {
                // Recording starts immediately so a "save" click captures the buffer accumulated
                // since Start -- see AudioEngineManager.startRecording, called from toggleRunning.
            }
        }
    }

    private func toggleRunning() {
        errorMessage = nil
        if audio.isRunning {
            audio.stop()
        } else {
            do {
                try audio.start(backend: selectedBackend)
                let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
                audio.startRecording(baseURL: docs)
            } catch {
                errorMessage = error.localizedDescription
            }
        }
    }

    private func saveRecording() {
        savedFiles = audio.stopRecording()
        // Restart recording immediately so streaming continues uninterrupted for the user.
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        audio.startRecording(baseURL: docs)
    }
}

#Preview {
    ContentView()
}
