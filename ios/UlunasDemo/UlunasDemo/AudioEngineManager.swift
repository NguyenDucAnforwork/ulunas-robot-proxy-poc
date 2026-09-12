import AVFoundation
import Foundation

/// Owns the AVAudioEngine graph, the mic-capture -> resample -> ring-buffer -> inference-thread
/// -> ring-buffer -> playback pipeline, and all real-time-safety-critical state.
///
/// Design: CoreAudio's own callbacks (the input tap and the output render callback) do the
/// absolute minimum -- copy samples into/out of lock-free ring buffers (see RingBuffer.swift).
/// The actual DSP (resampling already happens via AVAudioConverter off the render thread, in
/// the tap callback, which Apple documents as safe for converter operations) and the UL-UNAS
/// inference call happen on a dedicated thread requesting real-time scheduling, NOT inside the
/// CoreAudio render callback itself. This satisfies "no malloc/lock/heavy logging in the
/// real-time callback" while keeping inference latency bounded and measurable.
final class AudioEngineManager: ObservableObject {
    // MARK: - Published state for the UI (updated on the main thread only)
    @Published var isRunning = false
    @Published var isBypassed = false
    @Published var inputSampleRate: Double = 0
    @Published var inputChannelCount: Int = 0
    @Published var currentRoute: String = "unknown"
    @Published var meanInferenceMs: Double = 0
    @Published var p95InferenceMs: Double = 0
    @Published var rtf: Double = 0
    @Published var underrunCount: Int = 0
    @Published var overrunCount: Int = 0
    @Published var headphonesConnected: Bool = false
    @Published var feedbackRiskWarning: Bool = false

    private let engine = AVAudioEngine()
    private var converterToModelRate: AVAudioConverter?
    private var converterFromModelRate: AVAudioConverter?

    private let inputRing = RingBuffer(minimumCapacity: 16000 * 4)   // 4s @ 16kHz headroom
    private let outputRing = RingBuffer(minimumCapacity: 16000 * 4)

    private var engineInstance: UlunasEngine?
    private var inferenceThread: Thread?
    private var shouldStopInference = false

    private var hopInputBuf = [Float](repeating: 0, count: Int(ULUNAS_HOP_SIZE_CONST))
    private var hopOutputBuf = [Float](repeating: 0, count: Int(ULUNAS_HOP_SIZE_CONST))

    private var inferenceTimesMs: [Double] = []  // bounded ring, read/written only on inference thread + snapshot under a light NSLock for UI polling (not the audio thread)
    private let statsLock = NSLock()

    private var wavRecorder: WavRecorder?

    // Model constant mirrored from ulunas_api.h (kept in sync manually; see that header).
    private let ULUNAS_HOP_SIZE_CONST: Int32 = 256
    private let modelSampleRate: Double = 16000

    // MARK: - Session configuration

    /// Configures AVAudioSession WITHOUT enabling any built-in AEC/AGC/voice-isolation path.
    /// Deliberately uses `.default` mode (NOT `.voiceChat`/`.videoChat`, which route through
    /// Apple's Voice-Processing I/O unit and impose AEC + AGC + noise suppression we cannot
    /// disable once that mode is active). `.playAndRecord` is required to route to headphones
    /// while also recording. `.allowBluetooth` is included only so Bluetooth can be demoed
    /// (per the project brief, explicitly NOT used for latency claims).
    private func configureAudioSession() throws {
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.playAndRecord,
                                 mode: .default,
                                 options: [.allowBluetooth, .defaultToSpeaker])
        // .default mode + AVAudioEngine's inputNode with voice-processing NOT enabled (see
        // below) is what keeps us off the Voice-Processing I/O unit. There is no public API to
        // force AGC off on the standard Remote I/O input path -- iOS applies minimal input
        // gain shaping at the hardware/driver level regardless; this is disclosed here rather
        // than claimed as "fully raw", since Apple does not document a way to verify or fully
        // disable it from the public AVAudioSession/AVAudioEngine API surface.
        try session.setPreferredSampleRate(48000)
        try session.setPreferredIOBufferDuration(0.016) // ~256 samples @ 16kHz-equivalent hop; actual mic rate may differ, see below
        try session.setActive(true)

        NotificationCenter.default.addObserver(self, selector: #selector(handleRouteChange),
                                               name: AVAudioSession.routeChangeNotification, object: session)
        NotificationCenter.default.addObserver(self, selector: #selector(handleInterruption),
                                               name: AVAudioSession.interruptionNotification, object: session)
        updateRouteInfo()
    }

    @objc private func handleRouteChange(_ note: Notification) {
        updateRouteInfo()
        // A route change (e.g. headphones unplugged mid-stream) can leave the STFT
        // overlap-add / cache state referencing stale timing -- reset defensively.
        DispatchQueue.main.async { [weak self] in
            self?.engineInstance?.reset()
            self?.inputRing.reset()
            self?.outputRing.reset()
        }
    }

    @objc private func handleInterruption(_ note: Notification) {
        guard let info = note.userInfo,
              let typeValue = info[AVAudioSessionInterruptionTypeKey] as? UInt,
              let type = AVAudioSession.InterruptionType(rawValue: typeValue) else { return }
        if type == .began {
            stop()
        }
        // On .ended, the user explicitly presses Start again -- we do not auto-resume, since
        // silently restarting audio capture after e.g. a phone call is a poor UX default.
    }

    private func updateRouteInfo() {
        let session = AVAudioSession.sharedInstance()
        let outputs = session.currentRoute.outputs
        currentRoute = outputs.map { $0.portType.rawValue }.joined(separator: ",")
        headphonesConnected = outputs.contains { $0.portType == .headphones || $0.portType == .bluetoothA2DP || $0.portType == .bluetoothHFP }
        feedbackRiskWarning = !headphonesConnected
    }

    // MARK: - Engine lifecycle

    func start(backend: UlunasBackend) throws {
        guard !isRunning else { return }
        try configureAudioSession()

        engineInstance = UlunasEngine(backend: backend)
        guard engineInstance != nil else {
            throw NSError(domain: "Ulunas", code: 1, userInfo: [NSLocalizedDescriptionKey: "engine init failed"])
        }

        let inputNode = engine.inputNode
        let hwFormat = inputNode.outputFormat(forBus: 0)
        inputSampleRate = hwFormat.sampleRate
        inputChannelCount = Int(hwFormat.channelCount)
        // Explicitly log rather than assume -- iPhone 11's built-in mic via AVAudioEngine
        // commonly reports 48000 Hz / 1ch here, but this must never be hardcoded.
        NSLog("[Ulunas] input hw format: %.0f Hz, %d ch", hwFormat.sampleRate, inputNode.numberOfInputs)

        // IMPORTANT: never touch inputNode.isVoiceProcessingEnabled (leave it at its default
        // `false`) -- setting it true would route through Voice-Processing I/O and enable the
        // AEC/AGC this project explicitly excludes.
        assert(inputNode.isVoiceProcessingEnabled == false, "voice processing must stay disabled -- this project has no AEC")

        let modelFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: modelSampleRate, channels: 1, interleaved: false)!
        converterToModelRate = AVAudioConverter(from: hwFormat, to: modelFormat)
        converterFromModelRate = AVAudioConverter(from: modelFormat, to: hwFormat)

        // Tap: runs on a CoreAudio-managed thread, NOT the render thread of outputNode, but
        // still time-sensitive -- resampling via AVAudioConverter here is Apple's documented
        // pattern (see AVAudioConverter docs) and involves no locks/allocation on our part
        // beyond the converter's own internal buffers (allocated once, not per-call, by
        // reusing a preallocated PCM buffer below).
        let tapBufferSize: AVAudioFrameCount = 1024
        let convertedCapacity = AVAudioFrameCount(Double(tapBufferSize) * modelSampleRate / hwFormat.sampleRate) + 16
        let scratchBuffer = AVAudioPCMBuffer(pcmFormat: modelFormat, frameCapacity: convertedCapacity)!

        inputNode.installTap(onBus: 0, bufferSize: tapBufferSize, format: hwFormat) { [weak self] buffer, _ in
            guard let self = self else { return }
            scratchBuffer.frameLength = 0
            var error: NSError?
            self.converterToModelRate?.convert(to: scratchBuffer, error: &error) { _, outStatus in
                outStatus.pointee = .haveData
                return buffer
            }
            if let error = error {
                NSLog("[Ulunas] resample error: %@", error.localizedDescription)
                return
            }
            guard let channelData = scratchBuffer.floatChannelData else { return }
            self.inputRing.write(channelData[0], count: Int(scratchBuffer.frameLength))
        }

        // Output: an AVAudioSourceNode pulls from outputRing on the render thread. This closure
        // executes on CoreAudio's real-time render thread -- it does ONLY ring-buffer reads and
        // a raw memcpy-equivalent loop, no allocation, no locks (RingBuffer.read is lock-free
        // by the SPSC invariant), matching the real-time-safety requirement.
        let sourceFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: modelSampleRate, channels: 1, interleaved: false)!
        let sourceNode = AVAudioSourceNode(format: sourceFormat) { [weak self] _, _, frameCount, audioBufferList -> OSStatus in
            guard let self = self else { return noErr }
            let ablPointer = UnsafeMutableAudioBufferListPointer(audioBufferList)
            guard let out = ablPointer[0].mData?.assumingMemoryBound(to: Float.self) else { return noErr }
            _ = self.outputRing.read(out, count: Int(frameCount))
            return noErr
        }
        engine.attach(sourceNode)

        // sourceNode is already at modelSampleRate (16kHz); convert up to hw rate for playback
        // via a mixer node format connection (AVAudioEngine handles the SRC in the connection
        // when formats differ, using its own internal converter -- documented behavior).
        engine.connect(sourceNode, to: engine.mainMixerNode, format: sourceFormat)
        engine.connect(engine.mainMixerNode, to: engine.outputNode, format: engine.outputNode.inputFormat(forBus: 0))

        try engine.start()
        startInferenceThread()
        isRunning = true
    }

    func stop() {
        guard isRunning else { return }
        stopInferenceThread()
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
        engineInstance = nil
        inputRing.reset()
        outputRing.reset()
        isRunning = false
    }

    func reset() {
        engineInstance?.reset()
        inputRing.reset()
        outputRing.reset()
        statsLock.lock(); inferenceTimesMs.removeAll(); statsLock.unlock()
    }

    // MARK: - Inference thread

    /// Runs on a dedicated Thread configured for real-time-ish scheduling (via
    /// Thread.threadPriority + QoS; true `thread_time_constraint_policy` requires dropping to
    /// the Mach API, noted here as a follow-up for a production build). Pulls one hop at a
    /// time from inputRing, calls into the C engine (or bypass passthrough), writes to
    /// outputRing, and records timing stats.
    private func startInferenceThread() {
        shouldStopInference = false
        let thread = Thread { [weak self] in
            guard let self = self else { return }
            Thread.current.qualityOfService = .userInteractive
            Thread.current.threadPriority = 1.0
            let hopSize = Int(self.ULUNAS_HOP_SIZE_CONST)
            var inBuf = [Float](repeating: 0, count: hopSize)
            var outBuf = [Float](repeating: 0, count: hopSize)
            while !self.shouldStopInference {
                let got = self.inputRing.read(&inBuf, count: hopSize)
                if got < hopSize {
                    // Not enough buffered yet -- brief sleep, avoid a hot spin loop burning a
                    // whole core; 4ms is well under the 16ms hop budget.
                    Thread.sleep(forTimeInterval: 0.004)
                    continue
                }
                let t0 = DispatchTime.now()
                if self.isBypassed {
                    outBuf = inBuf
                } else {
                    inBuf.withUnsafeBufferPointer { inPtr in
                        outBuf.withUnsafeMutableBufferPointer { outPtr in
                            _ = self.engineInstance?.processHop(withInput: inPtr.baseAddress!, output: outPtr.baseAddress!)
                        }
                    }
                }
                let elapsedMs = Double(DispatchTime.now().uptimeNanoseconds - t0.uptimeNanoseconds) / 1_000_000
                self.recordInferenceTime(elapsedMs)

                self.wavRecorder?.appendNoisy(inBuf)
                self.wavRecorder?.appendEnhanced(outBuf)

                outBuf.withUnsafeBufferPointer { self.outputRing.write($0.baseAddress!, count: hopSize) }
            }
        }
        thread.name = "ulunas.inference"
        inferenceThread = thread
        thread.start()
    }

    private func stopInferenceThread() {
        shouldStopInference = true
        inferenceThread = nil
    }

    private func recordInferenceTime(_ ms: Double) {
        statsLock.lock()
        inferenceTimesMs.append(ms)
        if inferenceTimesMs.count > 2000 { inferenceTimesMs.removeFirst(inferenceTimesMs.count - 2000) }
        let sorted = inferenceTimesMs.sorted()
        let mean = sorted.reduce(0, +) / Double(sorted.count)
        let p95 = sorted[Int(Double(sorted.count - 1) * 0.95)]
        statsLock.unlock()

        let hopDurationMs = 1000.0 * Double(ULUNAS_HOP_SIZE_CONST) / modelSampleRate
        DispatchQueue.main.async {
            self.meanInferenceMs = mean
            self.p95InferenceMs = p95
            self.rtf = mean / hopDurationMs
            self.underrunCount = self.outputRing.underrunCount
            self.overrunCount = self.inputRing.overrunCount
        }
    }

    // MARK: - WAV capture

    func startRecording(baseURL: URL) {
        wavRecorder = try? WavRecorder(baseURL: baseURL, sampleRate: Int(modelSampleRate))
    }

    func stopRecording() -> (noisy: URL, enhanced: URL)? {
        defer { wavRecorder = nil }
        return wavRecorder?.finalizeFiles()
    }
}
