import Foundation

/// Writes noisy and enhanced streams to two PCM16 mono WAV files, appended hop-by-hop.
/// File I/O happens on the inference thread's own timeline but is NOT in the CoreAudio
/// real-time callback -- `FileHandle.write` here is acceptable since this runs on the
/// dedicated inference thread, not the audio render thread (see AudioEngineManager).
/// For a stricter real-time guarantee, buffer to memory and flush on a separate background
/// queue instead; left as-is here since the inference thread already tolerates a few ms of
/// jitter (it is not the hard real-time boundary, the CoreAudio callbacks are).
final class WavRecorder {
    private let noisyURL: URL
    private let enhancedURL: URL
    private var noisyHandle: FileHandle
    private var enhancedHandle: FileHandle
    private let sampleRate: Int
    private var noisyFrameCount: Int = 0
    private var enhancedFrameCount: Int = 0

    init(baseURL: URL, sampleRate: Int) throws {
        self.sampleRate = sampleRate
        noisyURL = baseURL.appendingPathComponent("noisy_\(Int(Date().timeIntervalSince1970)).wav")
        enhancedURL = baseURL.appendingPathComponent("enhanced_\(Int(Date().timeIntervalSince1970)).wav")
        FileManager.default.createFile(atPath: noisyURL.path, contents: Self.emptyWavHeader(sampleRate: sampleRate))
        FileManager.default.createFile(atPath: enhancedURL.path, contents: Self.emptyWavHeader(sampleRate: sampleRate))
        noisyHandle = try FileHandle(forWritingTo: noisyURL)
        enhancedHandle = try FileHandle(forWritingTo: enhancedURL)
        noisyHandle.seekToEndOfFile()
        enhancedHandle.seekToEndOfFile()
    }

    func appendNoisy(_ samples: [Float]) {
        noisyHandle.write(Self.floatToPCM16(samples))
        noisyFrameCount += samples.count
    }

    func appendEnhanced(_ samples: [Float]) {
        enhancedHandle.write(Self.floatToPCM16(samples))
        enhancedFrameCount += samples.count
    }

    func finalizeFiles() -> (noisy: URL, enhanced: URL) {
        noisyHandle.closeFile()
        enhancedHandle.closeFile()
        Self.patchHeader(url: noisyURL, sampleRate: sampleRate, frameCount: noisyFrameCount)
        Self.patchHeader(url: enhancedURL, sampleRate: sampleRate, frameCount: enhancedFrameCount)
        return (noisyURL, enhancedURL)
    }

    private static func floatToPCM16(_ samples: [Float]) -> Data {
        var data = Data(capacity: samples.count * 2)
        for s in samples {
            let clamped = max(-1.0, min(1.0, s))
            let v = Int16(clamped * 32767.0)
            withUnsafeBytes(of: v.littleEndian) { data.append(contentsOf: $0) }
        }
        return data
    }

    private static func emptyWavHeader(sampleRate: Int) -> Data {
        header(sampleRate: sampleRate, dataSize: 0)
    }

    private static func patchHeader(url: URL, sampleRate: Int, frameCount: Int) {
        guard let handle = try? FileHandle(forWritingTo: url) else { return }
        defer { handle.closeFile() }
        let dataSize = frameCount * 2
        handle.seek(toFileOffset: 0)
        handle.write(header(sampleRate: sampleRate, dataSize: dataSize))
    }

    private static func header(sampleRate: Int, dataSize: Int) -> Data {
        var data = Data()
        func append(_ s: String) { data.append(s.data(using: .ascii)!) }
        func appendU32(_ v: UInt32) { withUnsafeBytes(of: v.littleEndian) { data.append(contentsOf: $0) } }
        func appendU16(_ v: UInt16) { withUnsafeBytes(of: v.littleEndian) { data.append(contentsOf: $0) } }

        append("RIFF"); appendU32(UInt32(36 + dataSize)); append("WAVE")
        append("fmt "); appendU32(16)
        appendU16(1)                      // PCM
        appendU16(1)                      // mono
        appendU32(UInt32(sampleRate))
        appendU32(UInt32(sampleRate * 2)) // byte rate
        appendU16(2)                      // block align
        appendU16(16)                     // bits per sample
        append("data"); appendU32(UInt32(dataSize))
        return data
    }
}
