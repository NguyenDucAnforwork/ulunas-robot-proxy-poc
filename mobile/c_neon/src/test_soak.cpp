/* Long soak test: several minutes of continuous streaming (concatenated + looped real
 * utterances and noise), checking for NaN/Inf, and (under ASan) memory issues. */
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <random>
#include <vector>
#include "ulunas_full.h"

int main(int argc, char** argv) {
    int target_seconds = argc > 1 ? atoi(argv[1]) : 60;
    UlunasState* st;
    ulunas_init(&st);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-0.3f, 0.3f);
    std::vector<float> in(ULUNAS_HOP), out(ULUNAS_HOP);

    int n_hops = (target_seconds * ULUNAS_FS) / ULUNAS_HOP;
    int any_nan = 0;
    long total_samples = 0;
    for (int h = 0; h < n_hops; ++h) {
        for (auto& v : in) v = dist(rng) + 0.1f * sinf((float)total_samples * 0.01f); /* tone+noise mix */
        ulunas_process_hop(st, in.data(), out.data());
        for (float v : out) if (!std::isfinite(v)) any_nan = 1;
        total_samples += ULUNAS_HOP;
        if (h % (ULUNAS_FS / ULUNAS_HOP * 10) == 0) /* every ~10s */
            printf("  %.0fs processed, any_nan_so_far=%d\n", (double)total_samples / ULUNAS_FS, any_nan);
    }

    ulunas_destroy(st);
    printf("Total: %.1fs (%d hops) processed. any_nan_or_inf=%s\n",
           (double)total_samples / ULUNAS_FS, n_hops, any_nan ? "YES (FAIL)" : "no");
    printf("RESULT: %s\n", any_nan ? "FAIL" : "PASS");
    return any_nan ? 1 : 0;
}
