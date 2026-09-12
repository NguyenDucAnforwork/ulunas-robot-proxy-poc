/* Edge cases: silence, impulse, random noise, reset-then-reprocess determinism, NaN/Inf. */
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <random>
#include <vector>
#include "ulunas_full.h"

static int check_finite(const std::vector<float>& v, const char* name) {
    for (float x : v) if (!std::isfinite(x)) { printf("%s: FAIL (non-finite value found)\n", name); return 0; }
    printf("%s: PASS (all finite)\n", name);
    return 1;
}

int main() {
    int all_pass = 1;
    UlunasState* st;
    ulunas_init(&st);

    /* silence */
    {
        std::vector<float> in(ULUNAS_HOP, 0.0f), out(ULUNAS_HOP);
        for (int i = 0; i < 200; ++i) ulunas_process_hop(st, in.data(), out.data());
        std::vector<float> outv(out.begin(), out.end());
        all_pass &= check_finite(outv, "silence (200 hops)");
    }
    ulunas_reset(st);

    /* impulse */
    {
        std::vector<float> in(ULUNAS_HOP, 0.0f), out(ULUNAS_HOP);
        in[0] = 1.0f;
        std::vector<float> collected;
        for (int i = 0; i < 50; ++i) {
            ulunas_process_hop(st, in.data(), out.data());
            in[0] = 0.0f; /* only first hop has the impulse */
            collected.insert(collected.end(), out.begin(), out.end());
        }
        all_pass &= check_finite(collected, "impulse (50 hops)");
    }
    ulunas_reset(st);

    /* random noise */
    {
        std::mt19937 rng(123);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        std::vector<float> in(ULUNAS_HOP), out(ULUNAS_HOP);
        std::vector<float> collected;
        for (int i = 0; i < 500; ++i) {
            for (auto& v : in) v = dist(rng);
            ulunas_process_hop(st, in.data(), out.data());
            collected.insert(collected.end(), out.begin(), out.end());
        }
        all_pass &= check_finite(collected, "random noise (500 hops, ~8s)");
    }

    /* reset determinism: reset then reprocess the SAME random sequence must give the SAME output */
    {
        ulunas_reset(st);
        std::mt19937 rng(999);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        std::vector<std::vector<float>> hops(30, std::vector<float>(ULUNAS_HOP));
        for (auto& h : hops) for (auto& v : h) v = dist(rng);

        std::vector<float> out(ULUNAS_HOP);
        std::vector<float> run1, run2;
        for (auto& h : hops) { ulunas_process_hop(st, h.data(), out.data()); run1.insert(run1.end(), out.begin(), out.end()); }

        ulunas_reset(st);
        for (auto& h : hops) { ulunas_process_hop(st, h.data(), out.data()); run2.insert(run2.end(), out.begin(), out.end()); }

        double max_diff = 0;
        for (size_t i = 0; i < run1.size(); ++i) max_diff = std::max(max_diff, (double)fabs(run1[i] - run2[i]));
        printf("reset-then-reprocess determinism: max_diff=%.10f %s\n", max_diff, max_diff == 0.0 ? "PASS (bit-exact)" : "FAIL");
        all_pass &= (max_diff == 0.0);
    }

    ulunas_destroy(st);
    printf("\n=== EDGE CASES: %s ===\n", all_pass ? "ALL PASS" : "SOME FAILED");
    return all_pass ? 0 : 1;
}
