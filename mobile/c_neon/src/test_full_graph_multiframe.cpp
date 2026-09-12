/* Full-graph multi-frame parity: feeds 20 consecutive real spectral frames (from a real
 * utterance) through ulunas_process_frame_spec, starting from zero caches (matching the
 * ONNX session's own zero-initialized caches), and compares every frame's output AND the
 * evolving cache state is implicitly exercised (caches thread frame-to-frame inside
 * UlunasState) against the ONNX reference computed the identical way. This is the first
 * test that catches cache-threading bugs across frames (single-frame tests only exercise
 * cache-in/cache-out once from a zero start). */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include "ulunas_full.h"

static float* load_npy_f32(const char* path, long* out_n) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    char magic[6]; fread(magic, 1, 6, f);
    unsigned char ver[2]; fread(ver, 1, 2, f);
    unsigned short header_len; fread(&header_len, 2, 1, f);
    char* header = (char*)malloc(header_len + 1);
    fread(header, 1, header_len, f);
    long data_start = ftell(f);
    fseek(f, 0, SEEK_END);
    long data_end = ftell(f);
    long n = (data_end - data_start) / sizeof(float);
    fseek(f, data_start, SEEK_SET);
    float* buf = (float*)malloc(n * sizeof(float));
    fread(buf, sizeof(float), n, f);
    fclose(f); free(header);
    *out_n = n;
    return buf;
}

int main() {
    long n_in, n_out;
    float* inputs = load_npy_f32("/tmp/multi_frame_inputs.npy", &n_in);   /* (1,257,T,2) */
    float* outputs_ref = load_npy_f32("/tmp/multi_frame_outputs.npy", &n_out);
    int n_frames = (int)(n_in / (257 * 2));
    printf("n_frames=%d\n", n_frames);

    UlunasState* st;
    if (ulunas_init(&st) != 0) { fprintf(stderr, "init failed\n"); return 1; }

    double max_err_overall = 0.0;
    int any_nan = 0;
    for (int t = 0; t < n_frames; ++t) {
        float mix_spec[257 * 2];
        /* numpy array is (1,257,T,2) C-order: index = ((f*T)+t)*2 + ri */
        for (int f = 0; f < 257; ++f) {
            mix_spec[f * 2 + 0] = inputs[(f * n_frames + t) * 2 + 0];
            mix_spec[f * 2 + 1] = inputs[(f * n_frames + t) * 2 + 1];
        }
        float enh_spec[257 * 2];
        ulunas_process_frame_spec(st, mix_spec, enh_spec);

        double frame_max_err = 0.0;
        for (int f = 0; f < 257; ++f) {
            for (int ri = 0; ri < 2; ++ri) {
                float v = enh_spec[f * 2 + ri];
                if (!std::isfinite(v)) any_nan = 1;
                float ref = outputs_ref[(f * n_frames + t) * 2 + ri];
                double e = fabs(v - ref);
                if (e > frame_max_err) frame_max_err = e;
            }
        }
        printf("frame %2d: max_abs_err=%.8f\n", t, frame_max_err);
        if (frame_max_err > max_err_overall) max_err_overall = frame_max_err;
    }

    ulunas_destroy(st);

    printf("\noverall max_abs_err over %d frames: %.8f\n", n_frames, max_err_overall);
    printf("any_nan_or_inf: %s\n", any_nan ? "YES (FAIL)" : "no");
    printf("RESULT: %s\n", (max_err_overall < 1e-3 && !any_nan) ? "PASS" : "FAIL");
    return (max_err_overall < 1e-3 && !any_nan) ? 0 : 1;
}
