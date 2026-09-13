/* Per-hop inference latency benchmark for the runtime-free C engine.
 *
 * IMPORTANT, per project rules: any number produced by this program when run under an
 * emulator (qemu-aarch64-static) or on a non-target host (x86-64) is explicitly a
 * SIMULATED/EMULATED number, NOT a measurement of real Cortex-A53 or Apple A13 silicon.
 * QEMU user-mode translates ARM64 instructions to run on the host CPU; its timing reflects
 * (host CPU speed x QEMU translation overhead), which has no fixed, predictable relationship
 * to real ARM silicon performance -- it can be faster OR slower than real hardware depending
 * on the specific instructions and host. This program prints that caveat on every run so the
 * number is never mistaken for a real-device benchmark downstream. Reported at the user's
 * explicit request for *a* number now, accepting this caveat, while real hardware remains
 * unavailable (BLOCKED_EXTERNAL for genuine Cortex-A53/A13 measurement). */
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include "ulunas_full.h"

int main(int argc, char** argv) {
    int n_hops = argc > 1 ? atoi(argv[1]) : 2000; // default ~32s of audio at 16kHz/hop=256
    int warmup_hops = argc > 2 ? atoi(argv[2]) : 100;

    UlunasState* st;
    ulunas_init(&st);

    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-0.2f, 0.2f);
    std::vector<float> in(ULUNAS_HOP), out(ULUNAS_HOP);

    for (int i = 0; i < warmup_hops; ++i) {
        for (auto& v : in) v = dist(rng);
        ulunas_process_hop(st, in.data(), out.data());
    }

    std::vector<double> hop_times_ms(n_hops);
    for (int i = 0; i < n_hops; ++i) {
        for (auto& v : in) v = dist(rng);
        auto t0 = std::chrono::high_resolution_clock::now();
        ulunas_process_hop(st, in.data(), out.data());
        auto t1 = std::chrono::high_resolution_clock::now();
        hop_times_ms[i] = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    std::vector<double> sorted_times = hop_times_ms;
    std::sort(sorted_times.begin(), sorted_times.end());
    double sum = 0;
    for (double t : sorted_times) sum += t;
    double mean_ms = sum / n_hops;
    double p50 = sorted_times[(size_t)(0.50 * n_hops)];
    double p95 = sorted_times[(size_t)(0.95 * n_hops)];
    double p99 = sorted_times[(size_t)(0.99 * n_hops)];
    double max_ms = sorted_times.back();

    double hop_duration_ms = 1000.0 * ULUNAS_HOP / ULUNAS_FS; // 16ms
    double rtf_mean = mean_ms / hop_duration_ms;
    double rtf_p95 = p95 / hop_duration_ms;

    printf("n_hops=%d hop_duration_ms=%.2f\n", n_hops, hop_duration_ms);
    printf("mean_ms=%.5f p50_ms=%.5f p95_ms=%.5f p99_ms=%.5f max_ms=%.5f\n",
           mean_ms, p50, p95, p99, max_ms);
    printf("RTF_mean=%.5f RTF_p95=%.5f\n", rtf_mean, rtf_p95);
    printf("real_time_capable(RTF_mean<1)=%s\n", rtf_mean < 1.0 ? "yes" : "no");
    printf("\n*** CAVEAT: this number reflects whatever machine ran this binary (see argv[0]\n");
    printf("*** and how it was invoked -- x86 host directly, or qemu-aarch64-static emulating\n");
    printf("*** ARM64). It is NOT a measurement of real Cortex-A53/Apple-A13 silicon. See\n");
    printf("*** MOBILE_BENCHMARK.md for what is and isn't a real-hardware number. ***\n");

    ulunas_destroy(st);
    return 0;
}
