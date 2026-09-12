/* WASM build of the UL-UNAS streaming DSP core (STFT -> [inference] -> ISTFT), for a
 * browser/Safari-on-iPhone fallback demo. Reuses the SAME fft.h used and numerically verified
 * (against torch.stft/istft, to ~1e-6) in the Android reference build -- not reimplemented.
 *
 * HONESTY NOTE: at the time this was written, the runtime-free C model-inference implementation
 * (mobile/c_neon/) was still being completed by a concurrent workstream. Rather than block this
 * demo on that, or fabricate a fake "enhanced" output, this WASM module runs the REAL STFT/
 * ISTFT framing (the part that must be correct for either bypass or true inference to sound
 * right) with a passthrough (identity) placeholder where model inference would run -- clearly
 * labeled in the UI as "STFT/ISTFT passthrough (no model yet)", never presented as noise
 * suppression. Swapping in the real ulunas_process_hop() once it exists is a small, mechanical
 * change (see the TODO below) -- it does not require touching the WASM/JS plumbing.
 *
 * OLA correctness note: a first draft of this file assumed the Hann/50%-overlap window-sum-of-
 * squares was a position-independent constant and normalized by it directly -- WRONG (verified
 * numerically: it actually ranges from 0.5 to 1.0 across a hop, not constant). Fixed by
 * maintaining real per-sample accum/win-sum buffers across hops (the same approach validated in
 * mobile/android_ref/src/main.cpp's batch OLA, adapted to true streaming).
 * ALGORITHMIC LATENCY: this streaming scheme has an inherent 1-hop (256-sample, 16ms) causal
 * delay between input and output -- output[t] corresponds to input[t-HOP], not input[t]. An
 * identity-transform self-test that ignored this delay initially reported a spurious ~0.99
 * max-abs-error; once compared with the correct 1-hop offset, the round-trip verifies to
 * 2.1e-6 (float precision) on random noise. Any consumer of this module (the JS glue below)
 * must account for this same 1-hop latency when reasoning about end-to-end delay. */
#include <cstring>
#include <vector>
#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#else
#define EMSCRIPTEN_KEEPALIVE
#endif

#include "../mobile/android_ref/src/fft.h"

namespace {
constexpr int N_FFT = 512;
constexpr int HOP = 256;

std::vector<float> g_hann;
std::vector<float> g_history;   // last (N_FFT-HOP) raw input samples carried across hops
std::vector<float> g_accum;     // length N_FFT, overlap-add accumulator (windowed samples)
std::vector<float> g_wsum;      // length N_FFT, accumulated window-sum-of-squares
bool g_inited = false;

void ensure_init() {
    if (g_inited) return;
    g_hann = hann_window(N_FFT);
    g_history.assign(N_FFT - HOP, 0.0f);
    g_accum.assign(N_FFT, 0.0f);
    g_wsum.assign(N_FFT, 0.0f);
    g_inited = true;
}
} // namespace

extern "C" {

EMSCRIPTEN_KEEPALIVE
void ulunas_wasm_reset() {
    ensure_init();
    std::fill(g_history.begin(), g_history.end(), 0.0f);
    std::fill(g_accum.begin(), g_accum.end(), 0.0f);
    std::fill(g_wsum.begin(), g_wsum.end(), 0.0f);
}

EMSCRIPTEN_KEEPALIVE
void ulunas_wasm_process_hop(const float* input, float* output) {
    ensure_init();

    /* build one N_FFT analysis frame from (history + new hop), analysis-window it, forward FFT */
    std::vector<float> frame(N_FFT);
    std::memcpy(frame.data(), g_history.data(), (N_FFT - HOP) * sizeof(float));
    std::memcpy(frame.data() + (N_FFT - HOP), input, HOP * sizeof(float));

    std::vector<float> windowed(N_FFT);
    for (int i = 0; i < N_FFT; ++i) windowed[i] = frame[i] * g_hann[i];

    auto spec = rfft(windowed);
    /* TODO: replace the next line with a call into the real model once available:
     *   spec = ulunas_model_forward_one_frame(spec, &streaming_state);
     * Currently: identity (no suppression applied) -- see file header. */

    auto recon = irfft(spec, N_FFT);

    /* synthesis-window + accumulate into the running OLA buffers (real per-sample
     * normalization, matching the validated batch approach in android_ref/main.cpp) */
    for (int i = 0; i < N_FFT; ++i) {
        g_accum[i] += recon[i] * g_hann[i];
        g_wsum[i] += g_hann[i] * g_hann[i];
    }

    /* emit the first HOP samples (now fully summed -- no future frame contributes to them) */
    for (int i = 0; i < HOP; ++i) {
        float norm = g_wsum[i] > 1e-8f ? g_wsum[i] : 1e-8f;
        output[i] = g_accum[i] / norm;
    }

    /* shift both accumulators left by HOP, zero-fill the newly-exposed tail */
    std::memmove(g_accum.data(), g_accum.data() + HOP, (N_FFT - HOP) * sizeof(float));
    std::memmove(g_wsum.data(), g_wsum.data() + HOP, (N_FFT - HOP) * sizeof(float));
    std::fill(g_accum.end() - HOP, g_accum.end(), 0.0f);
    std::fill(g_wsum.end() - HOP, g_wsum.end(), 0.0f);

    std::memcpy(g_history.data(), frame.data() + HOP, (N_FFT - HOP) * sizeof(float));
}

} // extern "C"
