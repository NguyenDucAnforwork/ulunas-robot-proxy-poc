import Foundation

/// Single-producer/single-consumer float ring buffer used to move audio between CoreAudio's
/// real-time threads (mic tap, output render callback) and the dedicated inference thread,
/// without locks or allocation on the real-time path.
///
/// Correctness note: `writeIndex` is mutated ONLY by the producer thread, `readIndex` ONLY by
/// the consumer thread -- this is the SPSC invariant and must never be violated (an earlier
/// draft of this file had the producer rewrite `readIndex` on overrun, which is a data race;
/// fixed below by having the producer simply refuse to overwrite unread data instead). Indices
/// are plain `Int`, relying on aligned-word loads/stores being atomic on arm64 in practice; for
/// a production-hardened version, back these with `Atomics.ManagedAtomic<Int>` (swift-atomics)
/// and appropriate `.acquire`/`.release` orderings instead of relying on the informal
/// happens-before relationship CoreAudio's callback scheduling provides.
final class RingBuffer {
    private var storage: [Float]
    private let capacity: Int
    private let mask: Int
    private var writeIndex: Int = 0 // producer-owned
    private var readIndex: Int = 0  // consumer-owned

    private(set) var underrunCount: Int = 0
    private(set) var overrunCount: Int = 0

    init(minimumCapacity: Int) {
        var cap = 1
        while cap < minimumCapacity { cap <<= 1 }
        capacity = cap
        mask = cap - 1
        storage = [Float](repeating: 0, count: cap)
    }

    /// Producer side: write `count` samples. Never allocates. If the buffer is full (consumer
    /// isn't keeping up), drops the *new* incoming samples and counts an overrun -- it does
    /// NOT touch `readIndex`, which belongs exclusively to the consumer.
    func write(_ samples: UnsafePointer<Float>, count: Int) {
        let free = capacity - (writeIndex - readIndex)
        let n = min(count, max(0, free))
        if n < count { overrunCount += 1 }
        storage.withUnsafeMutableBufferPointer { buf in
            for i in 0..<n {
                buf[writeIndex & mask] = samples[i]
                writeIndex += 1
            }
        }
    }

    /// Consumer side: read up to `count` samples into `dest`. Returns the number actually read;
    /// zero-fills the remainder and counts an underrun if not enough data was available.
    func read(_ dest: UnsafeMutablePointer<Float>, count: Int) -> Int {
        let available = writeIndex - readIndex
        let toRead = min(available, count)
        storage.withUnsafeBufferPointer { buf in
            for i in 0..<toRead {
                dest[i] = buf[readIndex & mask]
                readIndex += 1
            }
        }
        if toRead < count {
            for i in toRead..<count { dest[i] = 0 }
            underrunCount += 1
        }
        return toRead
    }

    /// Must only be called while both producer and consumer are stopped (e.g. engine paused).
    func reset() {
        writeIndex = 0
        readIndex = 0
        underrunCount = 0
        overrunCount = 0
    }
}
