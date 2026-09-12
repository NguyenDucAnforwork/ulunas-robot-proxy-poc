/* Validates encoder block 1 (XMBBlocks, the first downsampling + grouped-pointwise-conv +
 * shuffle stage) against StreamULUNAS._stream_xmb, using block0's real output as input. */
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
    float* x_in = load_npy_f32("/tmp/ref_block1_input.npy", &n); /* (1,12,1,65) */
    int C_in = 12, F_in = 65;

    /* ---- pconv1: pointwise grouped conv (12->24, groups=2), BN, AffinePReLU(width=65), Shuffle ---- */
    int C_mid = 24, groups = 2;
    float pconv1_out[24 * 65];
    conv2d_causal_1frame(x_in, C_in, /*KT=*/1, F_in, w_encoder_en_convs_1_pconv1_0_weight,
                         w_encoder_en_convs_1_pconv1_0_bias, C_mid, /*KF=*/1, /*stride_f=*/1,
                         /*pf=*/0, groups, pconv1_out, F_in);
    batchnorm_apply(pconv1_out, C_mid, F_in, w_encoder_en_convs_1_pconv1_1_weight,
                    w_encoder_en_convs_1_pconv1_1_bias, w_encoder_en_convs_1_pconv1_1_running_mean,
                    w_encoder_en_convs_1_pconv1_1_running_var, 1e-5f);
    affine_prelu_apply(pconv1_out, C_mid, F_in, w_encoder_en_convs_1_pconv1_2_affine_weight,
                       w_encoder_en_convs_1_pconv1_2_affine_bias, w_encoder_en_convs_1_pconv1_2_slope_weight);
    float pconv1_shuffled[24 * 65];
    shuffle_apply(pconv1_out, C_mid / 2, F_in, pconv1_shuffled);

    /* ---- dconv: depthwise causal conv (groups=24), kt=2,kf=3,stride_f=2,pf=1, BN, AffinePReLU(width=33) ---- */
    int KT = 2, KF = 3, stride_f = 2, pf = 1, F_out = 33;
    float conv_cache[24 * 1 * 65]; /* CONV_CACHE_SHAPES[1] = (24,1,65) */
    memset(conv_cache, 0, sizeof(conv_cache));
    /* my conv kernel expects [C][KT][F] contiguous per channel -- interleave cache/current per channel */
    float dconv_in_fixed[24 * 2 * 65];
    for (int c = 0; c < 24; ++c) {
        memcpy(dconv_in_fixed + c * 2 * 65, conv_cache + c * 65, 65 * sizeof(float));
        memcpy(dconv_in_fixed + c * 2 * 65 + 65, pconv1_shuffled + c * 65, 65 * sizeof(float));
    }
    float dconv_out[24 * 33];
    conv2d_causal_1frame(dconv_in_fixed, C_mid, KT, F_in, w_encoder_en_convs_1_dconv_1_weight,
                         w_encoder_en_convs_1_dconv_1_bias, C_mid, KF, stride_f, pf, /*groups=*/C_mid,
                         dconv_out, F_out);
    float new_conv_cache[24 * 65];
    for (int c = 0; c < 24; ++c) memcpy(new_conv_cache + c * 65, dconv_in_fixed + c * 2 * 65 + 65, 65 * sizeof(float));

    batchnorm_apply(dconv_out, C_mid, F_out, w_encoder_en_convs_1_dconv_2_weight,
                    w_encoder_en_convs_1_dconv_2_bias, w_encoder_en_convs_1_dconv_2_running_mean,
                    w_encoder_en_convs_1_dconv_2_running_var, 1e-5f);
    affine_prelu_apply(dconv_out, C_mid, F_out, w_encoder_en_convs_1_dconv_3_affine_weight,
                       w_encoder_en_convs_1_dconv_3_affine_bias, w_encoder_en_convs_1_dconv_3_slope_weight);

    /* ---- pconv2: pointwise grouped conv (24->24, groups=2), BN (no activation) ---- */
    float pconv2_out[24 * 33];
    conv2d_causal_1frame(dconv_out, C_mid, 1, F_out, w_encoder_en_convs_1_pconv2_0_weight,
                         w_encoder_en_convs_1_pconv2_0_bias, C_mid, 1, 1, 0, groups, pconv2_out, F_out);
    batchnorm_apply(pconv2_out, C_mid, F_out, w_encoder_en_convs_1_pconv2_1_weight,
                    w_encoder_en_convs_1_pconv2_1_bias, w_encoder_en_convs_1_pconv2_1_running_mean,
                    w_encoder_en_convs_1_pconv2_1_running_var, 1e-5f);

    /* ---- cTFA(channels=24, width=33): ta_gru hidden=48 (cached), fa r=4 H=9 (fresh) ---- */
    float tfa_cache[48];
    memset(tfa_cache, 0, sizeof(tfa_cache));
    float zt[24];
    for (int c = 0; c < C_mid; ++c) {
        float acc = 0;
        for (int f = 0; f < F_out; ++f) acc += pconv2_out[c * F_out + f] * pconv2_out[c * F_out + f];
        zt[c] = acc / F_out;
    }
    gru_step(zt, C_mid, tfa_cache, 48, w_encoder_en_convs_1_pconv2_2_ta_gru_weight_ih_l0,
            w_encoder_en_convs_1_pconv2_2_ta_gru_weight_hh_l0, w_encoder_en_convs_1_pconv2_2_ta_gru_bias_ih_l0,
            w_encoder_en_convs_1_pconv2_2_ta_gru_bias_hh_l0);
    float at[24];
    for (int c = 0; c < C_mid; ++c) {
        float acc = w_encoder_en_convs_1_pconv2_2_ta_fc_bias[c];
        for (int j = 0; j < 48; ++j) acc += w_encoder_en_convs_1_pconv2_2_ta_fc_weight[c * 48 + j] * tfa_cache[j];
        at[c] = sigmoidf_(acc);
    }
    int r = 4, pad_len = 3, F_pad = 36, H = 9;
    float fa_in[36]; memset(fa_in, 0, sizeof(fa_in));
    for (int f = 0; f < F_out; ++f) {
        float acc = 0;
        for (int c = 0; c < C_mid; ++c) acc += pconv2_out[c * F_out + f] * pconv2_out[c * F_out + f];
        fa_in[f] = acc / C_mid;
    }
    float fa_gru_out[9 * 8];
    gru_bidirectional_seq(fa_in, H, r, r,
        w_encoder_en_convs_1_pconv2_2_fa_gru_weight_ih_l0, w_encoder_en_convs_1_pconv2_2_fa_gru_weight_hh_l0,
        w_encoder_en_convs_1_pconv2_2_fa_gru_bias_ih_l0, w_encoder_en_convs_1_pconv2_2_fa_gru_bias_hh_l0,
        w_encoder_en_convs_1_pconv2_2_fa_gru_weight_ih_l0_reverse, w_encoder_en_convs_1_pconv2_2_fa_gru_weight_hh_l0_reverse,
        w_encoder_en_convs_1_pconv2_2_fa_gru_bias_ih_l0_reverse, w_encoder_en_convs_1_pconv2_2_fa_gru_bias_hh_l0_reverse,
        fa_gru_out);
    float fa_fc_out[9 * 4];
    for (int h = 0; h < H; ++h)
        for (int o = 0; o < r; ++o) {
            float acc = w_encoder_en_convs_1_pconv2_2_fa_fc_bias[o];
            for (int j = 0; j < 8; ++j) acc += w_encoder_en_convs_1_pconv2_2_fa_fc_weight[o * 8 + j] * fa_gru_out[h * 8 + j];
            fa_fc_out[h * r + o] = acc;
        }
    float af[33];
    for (int f = 0; f < F_out; ++f) af[f] = sigmoidf_(fa_fc_out[f]);

    for (int c = 0; c < C_mid; ++c)
        for (int f = 0; f < F_out; ++f)
            pconv2_out[c * F_out + f] = at[c] * pconv2_out[c * F_out + f] * af[f];

    /* residual: shapes differ (12,65) vs (24,33) -> skip add */

    /* final shuffle (groups==2, not is_last) */
    float final_out[24 * 33];
    shuffle_apply(pconv2_out, C_mid / 2, F_out, final_out);

    int n_out, n_cache, n_tfa;
    float* ref_out = load_npy_f32("/tmp/ref_block1_out.npy", &n_out);
    float* ref_cache = load_npy_f32("/tmp/ref_block1_conv_cache_out.npy", &n_cache);
    float* ref_tfa = load_npy_f32("/tmp/ref_block1_tfa_cache_out.npy", &n_tfa);

    double e_out = 0, e_cache = 0, e_tfa = 0;
    for (int i = 0; i < 24 * 33; ++i) { double e = fabs(final_out[i] - ref_out[i]); if (e > e_out) e_out = e; }
    for (int i = 0; i < 24 * 65; ++i) { double e = fabs(new_conv_cache[i] - ref_cache[i]); if (e > e_cache) e_cache = e; }
    for (int i = 0; i < 48; ++i) { double e = fabs(tfa_cache[i] - ref_tfa[i]); if (e > e_tfa) e_tfa = e; }

    printf("block1 output max abs err: %.8f\n", e_out);
    printf("block1 conv_cache_out max abs err: %.8f\n", e_cache);
    printf("block1 tfa_cache_out max abs err: %.8f\n", e_tfa);
    printf("PASS: %s\n", (e_out < 1e-3 && e_cache < 1e-4 && e_tfa < 1e-3) ? "YES" : "NO");
    return 0;
}
