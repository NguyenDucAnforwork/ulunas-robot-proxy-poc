/* TEMPORARY benchmark harness (not committed, not production code). Model-only level:
 * calls ulunas_process_frame_spec directly (bypasses STFT/OLA/analysis-buffer framing),
 * the same public C API function documented in mobile/c_neon/src/ulunas_full.h as having
 * "the exact same I/O contract as the streaming ONNX graph" -- i.e. the same contract as
 * Python's StreamULUNAS.forward(). Same timing mechanism (std::chrono::high_resolution_clock)
 * as benchmark_latency.cpp. No production file under mobile/c_neon/ is modified. */
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <vector>
#include "ulunas_full.h"

int main() {
    const int FREQ = 257;
    const int WARMUP = 100;
    const int MEASURED = 2000;

    FILE* fin = fopen("input_spec.bin", "rb");
    std::vector<float> all_in((WARMUP + MEASURED) * FREQ * 2);
    size_t n = fread(all_in.data(), sizeof(float), all_in.size(), fin);
    fclose(fin);
    if (n != all_in.size()) { fprintf(stderr, "short read\n"); return 1; }

    UlunasState* st;
    ulunas_init(&st);
    std::vector<float> out(FREQ * 2);

    for (int i = 0; i < WARMUP; ++i) {
        ulunas_process_frame_spec(st, &all_in[i * FREQ * 2], out.data());
    }

    std::vector<double> frame_times_ms(MEASURED);
    std::vector<float> all_out(MEASURED * FREQ * 2);
    for (int i = 0; i < MEASURED; ++i) {
        const float* in_ptr = &all_in[(WARMUP + i) * FREQ * 2];
        auto t0 = std::chrono::high_resolution_clock::now();
        ulunas_process_frame_spec(st, in_ptr, out.data());
        auto t1 = std::chrono::high_resolution_clock::now();
        frame_times_ms[i] = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::copy(out.begin(), out.end(), all_out.begin() + i * FREQ * 2);
    }

    FILE* fout = fopen("output_spec_c.bin", "wb");
    fwrite(all_out.data(), sizeof(float), all_out.size(), fout);
    fclose(fout);

    std::vector<double> sorted_times = frame_times_ms;
    std::sort(sorted_times.begin(), sorted_times.end());
    double sum = 0;
    for (double t : sorted_times) sum += t;
    double mean_ms = sum / MEASURED;
    double p50 = sorted_times[(size_t)(0.50 * MEASURED)];
    double p95 = sorted_times[(size_t)(0.95 * MEASURED)];
    double p99 = sorted_times[(size_t)(0.99 * MEASURED)];
    double max_ms = sorted_times.back();
    double hop_duration_ms = 1000.0 * 256 / 16000; // same 16ms convention as end-to-end, for RTF comparability

    printf("n_frames=%d hop_duration_ms=%.2f\n", MEASURED, hop_duration_ms);
    printf("mean_ms=%.5f p50_ms=%.5f p95_ms=%.5f p99_ms=%.5f max_ms=%.5f\n",
           mean_ms, p50, p95, p99, max_ms);
    printf("RTF_mean=%.5f RTF_p95=%.5f\n", mean_ms / hop_duration_ms, p95 / hop_duration_ms);

    ulunas_destroy(st);
    return 0;
}
