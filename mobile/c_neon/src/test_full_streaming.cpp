/* Full-utterance streaming parity: feeds a real utterance through ulunas_process_hop,
 * hop by hop (as a real audio callback would), and compares the reconstructed waveform
 * against the PyTorch OFFLINE model's output on the same utterance.
 *
 * Expected: NOT bit-exact from sample 0, because ulunas_process_hop uses a causal
 * zero-initialized analysis window (see the design-choice comment in ulunas_full.cpp),
 * while the offline PyTorch reference uses torch.stft's default center=True (whole-signal
 * reflect-padding). This only affects the FIRST ~1 window (32ms / 512 samples) of the
 * output; every frame after that should match to float precision since the analysis
 * window content becomes identical regardless of the padding convention. Reports BOTH
 * numbers, does not hide the transient. */
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include "ulunas_full.h"

int main() {
    FILE* fin = fopen("/tmp/streaming_test_input.f32", "rb");
    fseek(fin, 0, SEEK_END);
    long n_bytes = ftell(fin);
    fseek(fin, 0, SEEK_SET);
    int n_samples = (int)(n_bytes / sizeof(float));
    std::vector<float> input(n_samples);
    fread(input.data(), sizeof(float), n_samples, fin);
    fclose(fin);

    FILE* fref = fopen("/tmp/streaming_test_offline_ref.f32", "rb");
    std::vector<float> ref(n_samples);
    fread(ref.data(), sizeof(float), n_samples, fref);
    fclose(fref);

    UlunasState* st;
    if (ulunas_init(&st) != 0) { fprintf(stderr, "init failed\n"); return 1; }

    std::vector<float> output(n_samples, 0.0f);
    int n_hops = n_samples / ULUNAS_HOP;
    for (int h = 0; h < n_hops; ++h) {
        ulunas_process_hop(st, &input[h * ULUNAS_HOP], &output[h * ULUNAS_HOP]);
    }
    /* leftover samples (< 1 hop) are ignored for this test, matches n_hops*HOP truncation */

    ulunas_destroy(st);

    FILE* fout = fopen("/tmp/streaming_test_c_output.f32", "wb");
    fwrite(output.data(), sizeof(float), output.size(), fout);
    fclose(fout);

    int n_valid = n_hops * ULUNAS_HOP;
    /* Empirically measured (via cross-correlation, see MOBILE_BENCHMARK.md) total output
     * latency of this streaming implementation: ULUNAS_HOP (256) samples MORE than the
     * model's own inherent 2-hop (512-sample) algorithmic latency established by ulunas.py's
     * own causality test -- i.e. output_c[i + DELAY] corresponds to offline_ref[i]. This is a
     * real, understood property of this straightforward 50%-overlap OLA buffering scheme
     * (not a bug): the accumulator position emitted each call is only fully formed once the
     * NEXT frame's contribution lands, and this implementation emits on a 1-call-delayed
     * schedule relative to the theoretical minimum. Total API latency: 3 hops = 48ms. */
    const int DELAY = ULUNAS_HOP;
    double max_err_all = 0.0, max_err_steady = 0.0;
    int startup_samples = ULUNAS_WIN + DELAY;
    int any_nan = 0;
    for (int i = 0; i + DELAY < n_valid; ++i) {
        float out_v = output[i + DELAY];
        if (!std::isfinite(out_v)) any_nan = 1;
        double e = fabs((double)out_v - (double)ref[i]);
        if (e > max_err_all) max_err_all = e;
        if (i >= startup_samples && e > max_err_steady) max_err_steady = e;
    }

    printf("n_samples=%d n_hops=%d output_delay_samples=%d (%.1fms)\n", n_valid, n_hops, DELAY, 1000.0*DELAY/ULUNAS_FS);
    printf("max_abs_err (ALL samples, incl. startup transient): %.8f\n", max_err_all);
    printf("max_abs_err (steady-state, samples >= %d after delay compensation): %.8f\n", startup_samples, max_err_steady);
    printf("any_nan_or_inf: %s\n", any_nan ? "YES (FAIL)" : "no");
    printf("RESULT (steady-state <= 1e-4 required): %s\n", (max_err_steady <= 1e-4 && !any_nan) ? "PASS" : "FAIL");
    return (max_err_steady <= 1e-4 && !any_nan) ? 0 : 1;
}
