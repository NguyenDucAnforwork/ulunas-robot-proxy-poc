/* Validates encoder block 2 (XDWSBlock: pconv -> depthwise dconv -> cTFA, no residual/final
 * shuffle) against StreamULUNAS._stream_xdws, using block1's real output as input. */
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

int main() {
    int n;
    float* x_in = load_npy_f32("/tmp/ref_block2_input.npy", &n); /* (1,24,1,33) */
    int C = 24, F = 33, groups = 2;

    /* ---- pconv: pointwise grouped conv (24->24), BN, AffinePReLU(width=33), Shuffle ---- */
    float pconv_out[24 * 33];
    conv2d_causal_1frame(x_in, C, 1, F, w_encoder_en_convs_2_pconv_0_weight,
                         w_encoder_en_convs_2_pconv_0_bias, C, 1, 1, 0, groups, pconv_out, F);
    batchnorm_apply(pconv_out, C, F, w_encoder_en_convs_2_pconv_1_weight,
                    w_encoder_en_convs_2_pconv_1_bias, w_encoder_en_convs_2_pconv_1_running_mean,
                    w_encoder_en_convs_2_pconv_1_running_var, 1e-5f);
    affine_prelu_apply(pconv_out, C, F, w_encoder_en_convs_2_pconv_2_affine_weight,
                       w_encoder_en_convs_2_pconv_2_affine_bias, w_encoder_en_convs_2_pconv_2_slope_weight);
    float pconv_shuffled[24 * 33];
    shuffle_apply(pconv_out, C / 2, F, pconv_shuffled);

    /* ---- dconv: depthwise causal conv (groups=24), kt=2,kf=3,stride_f=1,pf=1, BN, AffinePReLU ---- */
    int KT = 2, KF = 3, stride_f = 1, pf = 1;
    float conv_cache[24 * 1 * 33]; /* CONV_CACHE_SHAPES[2] = (24,1,33) */
    memset(conv_cache, 0, sizeof(conv_cache));
    float dconv_in[24 * 2 * 33];
    for (int c = 0; c < C; ++c) {
        memcpy(dconv_in + c * 2 * 33, conv_cache + c * 33, 33 * sizeof(float));
        memcpy(dconv_in + c * 2 * 33 + 33, pconv_shuffled + c * 33, 33 * sizeof(float));
    }
    float dconv_out[24 * 33];
    conv2d_causal_1frame(dconv_in, C, KT, F, w_encoder_en_convs_2_dconv_1_weight,
                         w_encoder_en_convs_2_dconv_1_bias, C, KF, stride_f, pf, C, dconv_out, F);
    float new_conv_cache[24 * 33];
    for (int c = 0; c < C; ++c) memcpy(new_conv_cache + c * 33, dconv_in + c * 2 * 33 + 33, 33 * sizeof(float));

    batchnorm_apply(dconv_out, C, F, w_encoder_en_convs_2_dconv_2_weight,
                    w_encoder_en_convs_2_dconv_2_bias, w_encoder_en_convs_2_dconv_2_running_mean,
                    w_encoder_en_convs_2_dconv_2_running_var, 1e-5f);
    affine_prelu_apply(dconv_out, C, F, w_encoder_en_convs_2_dconv_3_affine_weight,
                       w_encoder_en_convs_2_dconv_3_affine_bias, w_encoder_en_convs_2_dconv_3_slope_weight);

    /* ---- cTFA(channels=24, width=33): same shapes as block1's cTFA ---- */
    float tfa_cache[48];
    memset(tfa_cache, 0, sizeof(tfa_cache));
    float zt[24];
    for (int c = 0; c < C; ++c) {
        float acc = 0;
        for (int f = 0; f < F; ++f) acc += dconv_out[c * F + f] * dconv_out[c * F + f];
        zt[c] = acc / F;
    }
    gru_step(zt, C, tfa_cache, 48, w_encoder_en_convs_2_dconv_4_ta_gru_weight_ih_l0,
            w_encoder_en_convs_2_dconv_4_ta_gru_weight_hh_l0, w_encoder_en_convs_2_dconv_4_ta_gru_bias_ih_l0,
            w_encoder_en_convs_2_dconv_4_ta_gru_bias_hh_l0);
    float at[24];
    for (int c = 0; c < C; ++c) {
        float acc = w_encoder_en_convs_2_dconv_4_ta_fc_bias[c];
        for (int j = 0; j < 48; ++j) acc += w_encoder_en_convs_2_dconv_4_ta_fc_weight[c * 48 + j] * tfa_cache[j];
        at[c] = sigmoidf_(acc);
    }
    int r = 4, F_pad = 36, H = 9;
    float fa_in[36]; memset(fa_in, 0, sizeof(fa_in));
    for (int f = 0; f < F; ++f) {
        float acc = 0;
        for (int c = 0; c < C; ++c) acc += dconv_out[c * F + f] * dconv_out[c * F + f];
        fa_in[f] = acc / C;
    }
    float fa_gru_out[9 * 8];
    gru_bidirectional_seq(fa_in, H, r, r,
        w_encoder_en_convs_2_dconv_4_fa_gru_weight_ih_l0, w_encoder_en_convs_2_dconv_4_fa_gru_weight_hh_l0,
        w_encoder_en_convs_2_dconv_4_fa_gru_bias_ih_l0, w_encoder_en_convs_2_dconv_4_fa_gru_bias_hh_l0,
        w_encoder_en_convs_2_dconv_4_fa_gru_weight_ih_l0_reverse, w_encoder_en_convs_2_dconv_4_fa_gru_weight_hh_l0_reverse,
        w_encoder_en_convs_2_dconv_4_fa_gru_bias_ih_l0_reverse, w_encoder_en_convs_2_dconv_4_fa_gru_bias_hh_l0_reverse,
        fa_gru_out);
    float fa_fc_out[9 * 4];
    for (int h = 0; h < H; ++h)
        for (int o = 0; o < r; ++o) {
            float acc = w_encoder_en_convs_2_dconv_4_fa_fc_bias[o];
            for (int j = 0; j < 8; ++j) acc += w_encoder_en_convs_2_dconv_4_fa_fc_weight[o * 8 + j] * fa_gru_out[h * 8 + j];
            fa_fc_out[h * r + o] = acc;
        }
    float af[33];
    for (int f = 0; f < F; ++f) af[f] = sigmoidf_(fa_fc_out[f]);

    for (int c = 0; c < C; ++c)
        for (int f = 0; f < F; ++f)
            dconv_out[c * F + f] = at[c] * dconv_out[c * F + f] * af[f];

    int n_out, n_cache, n_tfa;
    float* ref_out = load_npy_f32("/tmp/ref_block2_out.npy", &n_out);
    float* ref_cache = load_npy_f32("/tmp/ref_block2_conv_cache_out.npy", &n_cache);
    float* ref_tfa = load_npy_f32("/tmp/ref_block2_tfa_cache_out.npy", &n_tfa);

    double e_out = 0, e_cache = 0, e_tfa = 0;
    for (int i = 0; i < 24 * 33; ++i) { double e = fabs(dconv_out[i] - ref_out[i]); if (e > e_out) e_out = e; }
    for (int i = 0; i < 24 * 33; ++i) { double e = fabs(new_conv_cache[i] - ref_cache[i]); if (e > e_cache) e_cache = e; }
    for (int i = 0; i < 48; ++i) { double e = fabs(tfa_cache[i] - ref_tfa[i]); if (e > e_tfa) e_tfa = e; }

    printf("block2 output max abs err: %.8f\n", e_out);
    printf("block2 conv_cache_out max abs err: %.8f\n", e_cache);
    printf("block2 tfa_cache_out max abs err: %.8f\n", e_tfa);
    printf("PASS: %s\n", (e_out < 1e-3 && e_cache < 1e-4 && e_tfa < 1e-3) ? "YES" : "NO");
    return 0;
}
