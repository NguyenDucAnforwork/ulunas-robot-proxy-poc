/* Generates the native-x86 reference output for the WASM parity test (test_wasm_node.js).
 * Same deterministic pseudo-random input generator as that file (kept in sync manually --
 * both are tiny and this avoids a cross-language shared-RNG dependency). Runs the REAL
 * ulunas_process_hop (same source file compiled to WASM) natively and dumps raw float32 LE
 * PCM so the Node test can diff native vs WASM output on identical input. */
#include <cstdio>
#include <cstdint>
#include <vector>
#include "ulunas_full.h"

int main() {
    const int HOP = 256;
    const int N_HOPS = 40;

    UlunasState* state = nullptr;
    ulunas_init(&state);

    int32_t seed = 12345;
    auto rnd = [&]() {
        seed = (int32_t)((seed * 1103515245 + 12345) & 0x7fffffff);
        return (seed / (float)0x7fffffff) * 2.0f - 1.0f;
    };

    std::vector<float> full_in(N_HOPS * HOP), full_out(N_HOPS * HOP);
    for (int i = 0; i < N_HOPS * HOP; ++i) full_in[i] = rnd() * 0.5f;

    for (int h = 0; h < N_HOPS; ++h) {
        ulunas_process_hop(state, &full_in[h * HOP], &full_out[h * HOP]);
    }

    FILE* f = fopen("reference_output.bin", "wb");
    fwrite(full_out.data(), sizeof(float), full_out.size(), f);
    fclose(f);
    ulunas_destroy(state);
    printf("wrote reference_output.bin: %zu floats\n", full_out.size());
    return 0;
}
