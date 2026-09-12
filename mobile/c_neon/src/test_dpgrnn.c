/* Validates DPGRNN layer 0 (dual-path grouped RNN: bidirectional intra-RNN over frequency,
 * recomputed fresh each frame; cached unidirectional inter-RNN over time, one hidden state
 * per frequency bin) against StreamULUNAS._stream_dpgrnn. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "kernels.h"
#include "../generated/ulunas_weights.h"

static float* load_npy_f32(const char* path, int* out_n) {
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
    int n = (int)((data_end - data_start) / sizeof(float));
    fseek(f, data_start, SEEK_SET);
    float* buf = (float*)malloc(n * sizeof(float));
    fread(buf, sizeof(float), n, f);
    fclose(f); free(header);
    *out_n = n;
    return buf;
}

#define C 16
#define F 33

int main() {
    int n;
    float* x_cf = load_npy_f32("/tmp/ref_dpgrnn_input.npy", &n); /* (1,16,1,33) -> [C][F] */

    /* permute to [F][C] for the frequency-sequence view */
    float x_fc[F * C];
    for (int f = 0; f < F; ++f)
        for (int c = 0; c < C; ++c)
            x_fc[f * C + c] = x_cf[c * F + f];

    /* ---- intra_rnn: bidirectional GRNN over F=33, split C=16 into 2x8, hidden=4 each ---- */
    float g1_in[F * 8], g2_in[F * 8];
    for (int f = 0; f < F; ++f) {
        memcpy(g1_in + f * 8, x_fc + f * C, 8 * sizeof(float));
        memcpy(g2_in + f * 8, x_fc + f * C + 8, 8 * sizeof(float));
    }
    float g1_out[F * 8], g2_out[F * 8]; /* each: [F][2*4=8] */
    gru_bidirectional_seq(g1_in, F, 8, 4,
        w_dpgrnn_0_intra_rnn_rnn1_weight_ih_l0, w_dpgrnn_0_intra_rnn_rnn1_weight_hh_l0,
        w_dpgrnn_0_intra_rnn_rnn1_bias_ih_l0, w_dpgrnn_0_intra_rnn_rnn1_bias_hh_l0,
        w_dpgrnn_0_intra_rnn_rnn1_weight_ih_l0_reverse, w_dpgrnn_0_intra_rnn_rnn1_weight_hh_l0_reverse,
        w_dpgrnn_0_intra_rnn_rnn1_bias_ih_l0_reverse, w_dpgrnn_0_intra_rnn_rnn1_bias_hh_l0_reverse, g1_out);
    gru_bidirectional_seq(g2_in, F, 8, 4,
        w_dpgrnn_0_intra_rnn_rnn2_weight_ih_l0, w_dpgrnn_0_intra_rnn_rnn2_weight_hh_l0,
        w_dpgrnn_0_intra_rnn_rnn2_bias_ih_l0, w_dpgrnn_0_intra_rnn_rnn2_bias_hh_l0,
        w_dpgrnn_0_intra_rnn_rnn2_weight_ih_l0_reverse, w_dpgrnn_0_intra_rnn_rnn2_weight_hh_l0_reverse,
        w_dpgrnn_0_intra_rnn_rnn2_bias_ih_l0_reverse, w_dpgrnn_0_intra_rnn_rnn2_bias_hh_l0_reverse, g2_out);

    float intra_gru_out[F * 16]; /* concat g1_out, g2_out per position -> 16 */
    for (int f = 0; f < F; ++f) {
        memcpy(intra_gru_out + f * 16, g1_out + f * 8, 8 * sizeof(float));
        memcpy(intra_gru_out + f * 16 + 8, g2_out + f * 8, 8 * sizeof(float));
    }

    /* intra_fc: Linear(16,16) per frequency position */
    float intra_x[F * 16];
    for (int f = 0; f < F; ++f)
        for (int o = 0; o < 16; ++o) {
            float acc = w_dpgrnn_0_intra_fc_bias[o];
            for (int j = 0; j < 16; ++j) acc += w_dpgrnn_0_intra_fc_weight[o * 16 + j] * intra_gru_out[f * 16 + j];
            intra_x[f * 16 + o] = acc;
        }

    /* intra_ln: LayerNorm over the FULL (F=33,C=16) block jointly */
    layernorm_apply(intra_x, F, 16, w_dpgrnn_0_intra_ln_weight, w_dpgrnn_0_intra_ln_bias, 1e-8f);

    float intra_out[F * 16];
    for (int i = 0; i < F * 16; ++i) intra_out[i] = x_fc[i] + intra_x[i];

    /* ---- inter_rnn: cached unidirectional GRNN, ONE hidden state pair per frequency bin ---- */
    float inter_cache[F * 16]; /* [F][16], 16 = 8+8 halves */
    memset(inter_cache, 0, sizeof(inter_cache));

    float inter_gru_out[F * 16];
    for (int f = 0; f < F; ++f) {
        float in1[8], in2[8];
        memcpy(in1, intra_out + f * 16, 8 * sizeof(float));
        memcpy(in2, intra_out + f * 16 + 8, 8 * sizeof(float));
        float* h1 = inter_cache + f * 16;
        float* h2 = inter_cache + f * 16 + 8;
        gru_step(in1, 8, h1, 8, w_dpgrnn_0_inter_rnn_rnn1_weight_ih_l0, w_dpgrnn_0_inter_rnn_rnn1_weight_hh_l0,
                w_dpgrnn_0_inter_rnn_rnn1_bias_ih_l0, w_dpgrnn_0_inter_rnn_rnn1_bias_hh_l0);
        gru_step(in2, 8, h2, 8, w_dpgrnn_0_inter_rnn_rnn2_weight_ih_l0, w_dpgrnn_0_inter_rnn_rnn2_weight_hh_l0,
                w_dpgrnn_0_inter_rnn_rnn2_bias_ih_l0, w_dpgrnn_0_inter_rnn_rnn2_bias_hh_l0);
        memcpy(inter_gru_out + f * 16, h1, 8 * sizeof(float));
        memcpy(inter_gru_out + f * 16 + 8, h2, 8 * sizeof(float));
    }

    float inter_x[F * 16];
    for (int f = 0; f < F; ++f)
        for (int o = 0; o < 16; ++o) {
            float acc = w_dpgrnn_0_inter_fc_bias[o];
            for (int j = 0; j < 16; ++j) acc += w_dpgrnn_0_inter_fc_weight[o * 16 + j] * inter_gru_out[f * 16 + j];
            inter_x[f * 16 + o] = acc;
        }
    layernorm_apply(inter_x, F, 16, w_dpgrnn_0_inter_ln_weight, w_dpgrnn_0_inter_ln_bias, 1e-8f);

    float inter_out[F * 16];
    for (int i = 0; i < F * 16; ++i) inter_out[i] = intra_out[i] + inter_x[i];

    /* permute back [F][C] -> [C][F] */
    float dual_out[C * F];
    for (int f = 0; f < F; ++f)
        for (int c = 0; c < C; ++c)
            dual_out[c * F + f] = inter_out[f * 16 + c];

    int n_out, n_cache;
    float* ref_out = load_npy_f32("/tmp/ref_dpgrnn_out.npy", &n_out);
    float* ref_cache = load_npy_f32("/tmp/ref_dpgrnn_inter_cache_out.npy", &n_cache);

    double e_out = 0, e_cache = 0;
    for (int i = 0; i < C * F; ++i) { double e = fabs(dual_out[i] - ref_out[i]); if (e > e_out) e_out = e; }
    for (int i = 0; i < F * 16; ++i) { double e = fabs(inter_cache[i] - ref_cache[i]); if (e > e_cache) e_cache = e; }

    printf("dpgrnn output max abs err: %.8f\n", e_out);
    printf("dpgrnn inter_cache_out max abs err: %.8f\n", e_cache);
    printf("PASS: %s\n", (e_out < 1e-3 && e_cache < 1e-3) ? "YES" : "NO");
    return 0;
}
