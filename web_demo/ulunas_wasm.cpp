/* WASM build of the REAL UL-UNAS streaming inference engine, for a browser/Safari-on-iPhone
 * fallback demo. This wraps the exact runtime-free C engine (mobile/c_neon/src/ulunas_full.cpp,
 * the same code validated in mobile/c_neon/ against ONNX/PyTorch parity to 1e-6-1e-7 and
 * benchmarked on real ARM in ARM_FULL_GRAPH_RESULTS.md) with real F_PROXY_ROBOT trained
 * weights -- not a reimplementation, not a placeholder.
 *
 * HISTORY: earlier versions of this file (see git history) shipped a hand-rolled STFT/
 * identity/ISTFT pipeline as a placeholder, because the runtime-free C model implementation
 * was still being completed by a concurrent workstream at the time and this demo was written
 * not to block on it. That placeholder has been replaced now that the real engine exists --
 * this file is now a thin wrapper: `ulunas_wasm_process_hop` is a direct pass-through to the
 * real `ulunas_process_hop`, doing zero DSP of its own (the real engine's own STFT/mask/OLA,
 * from mobile/c_neon/src/ulunas_full.cpp, already handles framing end-to-end).
 *
 * ALGORITHMIC LATENCY: ulunas_process_hop has the same causal algorithmic latency documented
 * in ulunas_full.h (2 hops = 32ms) -- output at call N corresponds to input delayed by that
 * amount; the first 2 calls' output is a startup transient (causal zero-history warm-up), not
 * a bug. Any consumer of this module (the JS glue in index.html) must account for this same
 * latency when reasoning about end-to-end delay. */
#include "../mobile/c_neon/src/ulunas_full.h"
#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#else
#define EMSCRIPTEN_KEEPALIVE
#endif

namespace {
UlunasState* g_state = nullptr;
} // namespace

extern "C" {

EMSCRIPTEN_KEEPALIVE
void ulunas_wasm_reset() {
    if (g_state) {
        ulunas_reset(g_state);
    } else {
        ulunas_init(&g_state);
    }
}

EMSCRIPTEN_KEEPALIVE
void ulunas_wasm_process_hop(const float* input, float* output) {
    if (!g_state) ulunas_init(&g_state);
    ulunas_process_hop(g_state, input, output);
}

} // extern "C"
