/* Both independently compiled graphs run on the SAME in-process input. No fixtures. */
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include "ulunas_full.h"

struct FoldedUlunasState;
extern "C" {
int folded_ulunas_init(FoldedUlunasState **);
void folded_ulunas_destroy(FoldedUlunasState *);
void folded_ulunas_reset(FoldedUlunasState *);
void folded_ulunas_process_frame_spec(FoldedUlunasState *, const float *, float *);
int folded_ulunas_process_hop(FoldedUlunasState *, const float *, float *);
}

int main() {
    UlunasState *ref = nullptr;
    FoldedUlunasState *folded = nullptr;
    if (ulunas_init(&ref) || folded_ulunas_init(&folded)) return 1;
    std::mt19937 rng(20260913);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    bool pass = true;
    for (int pcm = 0; pcm < 2; ++pcm) {
        ulunas_reset(ref);
        folded_ulunas_reset(folded);
        double max_abs = 0;
        const int n = pcm ? ULUNAS_HOP : 514;
        for (int frame = 0; frame < 20; ++frame) {
            float in[514], a[514], b[514];
            for (int i = 0; i < n; ++i) in[i] = dist(rng);
            if (pcm) {
                ulunas_process_hop(ref, in, a);
                folded_ulunas_process_hop(folded, in, b);
            } else {
                ulunas_process_frame_spec(ref, in, a);
                folded_ulunas_process_frame_spec(folded, in, b);
            }
            for (int i = 0; i < n; ++i) {
                if (!std::isfinite(a[i]) || !std::isfinite(b[i])) pass = false;
                max_abs = std::max(max_abs, std::fabs(double(a[i]) - double(b[i])));
            }
        }
        printf("%s: 20 frames, max_abs_error=%.10g (limit=1e-4)\n",
               pcm ? "PCM full pipeline" : "STFT full graph", max_abs);
        pass = pass && max_abs <= 1e-4;
    }
    ulunas_destroy(ref);
    folded_ulunas_destroy(folded);
    printf("BN fold parity: %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
