/* TEMPORARY benchmark helper (not committed, not production code). Generates the exact
 * deterministic input both the C and Python sides will consume, so cross-language output
 * parity is a byte-identical-input comparison rather than relying on replicating C++'s
 * std::mt19937/uniform_real_distribution algorithm in Python (implementation-defined,
 * error-prone to match exactly). Same seed/distribution/hop-count convention as the
 * existing (unmodified) mobile/c_neon/src/benchmark_latency.cpp.
 */
#include <cstdio>
#include <random>
#include <vector>

int main() {
    const int HOP = 256;
    const int WARMUP = 100;
    const int MEASURED = 2000;
    const int TOTAL_HOPS = WARMUP + MEASURED;

    // 1) end-to-end PCM input: same RNG/distribution/draw-order as benchmark_latency.cpp
    {
        std::mt19937 rng(1234);
        std::uniform_real_distribution<float> dist(-0.2f, 0.2f);
        std::vector<float> buf(TOTAL_HOPS * HOP);
        for (auto& v : buf) v = dist(rng);
        FILE* f = fopen("input_pcm.bin", "wb");
        fwrite(buf.data(), sizeof(float), buf.size(), f);
        fclose(f);
        printf("wrote input_pcm.bin: %zu floats (%d hops x %d)\n", buf.size(), TOTAL_HOPS, HOP);
    }

    // 2) model-only spectral input: 257 bins x 2 (re,im) per frame, same seed/distribution
    {
        const int FREQ = 257;
        std::mt19937 rng(1234);
        std::uniform_real_distribution<float> dist(-0.2f, 0.2f);
        std::vector<float> buf(TOTAL_HOPS * FREQ * 2);
        for (auto& v : buf) v = dist(rng);
        FILE* f = fopen("input_spec.bin", "wb");
        fwrite(buf.data(), sizeof(float), buf.size(), f);
        fclose(f);
        printf("wrote input_spec.bin: %zu floats (%d frames x %d x 2)\n", buf.size(), TOTAL_HOPS, FREQ);
    }
    return 0;
}
