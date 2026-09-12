/* Validates ERB.bm + encoder block 0 (XConvBlock) against a Python-extracted reference
 * (StreamULUNAS._stream_xconv with zero initial cache, frame index 5 of a real test wav). */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "kernels.h"
#include "../generated/ulunas_weights.h"

#define NFFT 512
#define FREQ 257
#define ERB_LOW 65
#define ERB_HIGH 64
#define ERB_TOTAL 129 /* 65 + 64 */

static float* load_npy_f32(const char* path, int* out_n) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    char magic[6]; fread(magic, 1, 6, f);
    unsigned char ver[2]; fread(ver, 1, 2, f);
    unsigned short header_len; fread(&header_len, 2, 1, f);
    char* header = (char*)malloc(header_len + 1);
    fread(header, 1, header_len, f);
    header[header_len] = 0;
    /* crude shape parse: find total element count from 'shape': (...) */
    long data_start = ftell(f);
    fseek(f, 0, SEEK_END);
    long data_end = ftell(f);
    int n = (int)((data_end - data_start) / sizeof(float));
    fseek(f, data_start, SEEK_SET);
    float* buf = (float*)malloc(n * sizeof(float));
    fread(buf, sizeof(float), n, f);
    fclose(f);
    free(header);
    *out_n = n;
    return buf;
}

int main() {
    int n;
    float* mix = load_npy_f32("/tmp/test_frame_input.npy", &n); /* (1,257,1,2) = 514 floats */

    /* --- magnitude (log10 of norm over the re/im pair) --- */
    float feat[FREQ];
    for (int f = 0; f < FREQ; ++f) {
        float re = mix[f * 2 + 0], im = mix[f * 2 + 1];
        float mag = sqrtf(re * re + im * im);
        if (mag < 1e-12f) mag = 1e-12f;
        feat[f] = log10f(mag);
    }

    /* --- ERB.bm: low 65 bins raw, high 192 bins -> 64 via erb_fc (no bias) --- */
    float feat_erb[ERB_TOTAL];
    for (int i = 0; i < ERB_LOW; ++i) feat_erb[i] = feat[i];
    for (int o = 0; o < ERB_HIGH; ++o) {
        float acc = 0.0f;
        for (int i = 0; i < (FREQ - ERB_LOW); ++i)
            acc += feat[ERB_LOW + i] * w_erb_erb_fc_weight[o * (FREQ - ERB_LOW) + i];
        feat_erb[ERB_LOW + o] = acc;
    }

    int n_ref;
    float* ref_feat_erb = load_npy_f32("/tmp/ref_feat_erb.npy", &n_ref);
    double erb_err = 0;
    for (int i = 0; i < ERB_TOTAL; ++i) { double e = fabs(feat_erb[i] - ref_feat_erb[i]); if (e > erb_err) erb_err = e; }
    printf("erb.bm max abs err: %.8f\n", erb_err);

    /* --- block 0: XConvBlock, in=1ch, out=12ch, kt=3,kf=3, stride_f=2, groups=1, width=65 --- */
    int C_in = 1, C_out = 12, KT = 3, KF = 3, stride_f = 2, pf = 1;
    int F_in = ERB_TOTAL; /* 129 */
    int F_out = 65;

    /* conv_cache: shape (1,2,129) i.e. (C_in=1, KT-1=2, F=129), zero-initialized */
    float conv_cache[1 * 2 * 129];
    memset(conv_cache, 0, sizeof(conv_cache));

    /* build the (C_in, KT, F) input = [cache(2 frames), current(1 frame)] -- RAW, unpadded */
    float conv_in[1 * 3 * 129];
    memcpy(conv_in, conv_cache, sizeof(conv_cache));
    memcpy(conv_in + 2 * 129, feat_erb, sizeof(feat_erb));

    float conv_out[12 * 65];
    conv2d_causal_1frame(conv_in, C_in, KT, F_in, w_encoder_en_convs_0_ops_1_weight,
                         w_encoder_en_convs_0_ops_1_bias, C_out, KF, stride_f, pf, /*groups=*/1,
                         conv_out, F_out);

    /* new conv_cache = conv_in[:, 1:, :] (drop oldest frame, shift) */
    float new_conv_cache[1 * 2 * 129];
    memcpy(new_conv_cache, conv_in + 129, 2 * 129 * sizeof(float));

    printf("DEBUG after conv: %.6f %.6f %.6f %.6f %.6f\n",
           conv_out[0], conv_out[1], conv_out[2], conv_out[3], conv_out[4]);

    /* BatchNorm2d(12) */
    batchnorm_apply(conv_out, C_out, F_out, w_encoder_en_convs_0_ops_2_weight,
                    w_encoder_en_convs_0_ops_2_bias, w_encoder_en_convs_0_ops_2_running_mean,
                    w_encoder_en_convs_0_ops_2_running_var, 1e-5f);

    /* AffinePReLU(12, width=65) */
    printf("DEBUG after bn: %.6f %.6f %.6f %.6f %.6f\n",
           conv_out[0], conv_out[1], conv_out[2], conv_out[3], conv_out[4]);

    affine_prelu_apply(conv_out, C_out, F_out, w_encoder_en_convs_0_ops_3_affine_weight,
                       w_encoder_en_convs_0_ops_3_affine_bias, w_encoder_en_convs_0_ops_3_slope_weight);

    printf("DEBUG after affineprelu: %.6f %.6f %.6f %.6f %.6f\n",
           conv_out[0], conv_out[1], conv_out[2], conv_out[3], conv_out[4]);

    /* cTFA(channels=12, width=65): ta_gru hidden=24 (cached), fa r=4 H=17 (fresh each frame) */
    float tfa_cache[24];
    memset(tfa_cache, 0, sizeof(tfa_cache));

    /* zt = mean(x^2, over F) per channel -> (C,) */
    float zt[12];
    for (int c = 0; c < C_out; ++c) {
        float acc = 0;
        for (int f = 0; f < F_out; ++f) acc += conv_out[c * F_out + f] * conv_out[c * F_out + f];
        zt[c] = acc / F_out;
    }
    printf("DEBUG zt: %.6f %.6f %.6f %.6f %.6f\n", zt[0], zt[1], zt[2], zt[3], zt[4]);
    gru_step(zt, C_out, tfa_cache, 24, w_encoder_en_convs_0_ops_4_ta_gru_weight_ih_l0,
            w_encoder_en_convs_0_ops_4_ta_gru_weight_hh_l0, w_encoder_en_convs_0_ops_4_ta_gru_bias_ih_l0,
            w_encoder_en_convs_0_ops_4_ta_gru_bias_hh_l0);
    /* ta_fc: Linear(24,12) then sigmoid -> at[c] */
    float at[12];
    for (int c = 0; c < C_out; ++c) {
        float acc = w_encoder_en_convs_0_ops_4_ta_fc_bias[c];
        for (int j = 0; j < 24; ++j) acc += w_encoder_en_convs_0_ops_4_ta_fc_weight[c * 24 + j] * tfa_cache[j];
        at[c] = sigmoidf_(acc);
    }

    /* FA: mean(x^2, over C) -> (F,) = (65,), pad to 68, reshape (17,4), bidirectional GRU, fc(8->4), unpad */
    int r = 4, pad_len = 3, F_pad = 68, H = 17;
    float fa_in[68];
    memset(fa_in, 0, sizeof(fa_in));
    for (int f = 0; f < F_out; ++f) {
        float acc = 0;
        for (int c = 0; c < C_out; ++c) acc += conv_out[c * F_out + f] * conv_out[c * F_out + f];
        fa_in[f] = acc / C_out;
    }
    /* reshape (17,4) is already the natural flat layout of fa_in viewed as [H][r] */
    float fa_gru_out[17 * 8]; /* [H, 2*r] */
    gru_bidirectional_seq(fa_in, H, r, r,
                          w_encoder_en_convs_0_ops_4_fa_gru_weight_ih_l0, w_encoder_en_convs_0_ops_4_fa_gru_weight_hh_l0,
                          w_encoder_en_convs_0_ops_4_fa_gru_bias_ih_l0, w_encoder_en_convs_0_ops_4_fa_gru_bias_hh_l0,
                          w_encoder_en_convs_0_ops_4_fa_gru_weight_ih_l0_reverse, w_encoder_en_convs_0_ops_4_fa_gru_weight_hh_l0_reverse,
                          w_encoder_en_convs_0_ops_4_fa_gru_bias_ih_l0_reverse, w_encoder_en_convs_0_ops_4_fa_gru_bias_hh_l0_reverse,
                          fa_gru_out);
    float fa_fc_out[17 * 4];
    for (int h = 0; h < H; ++h) {
        for (int o = 0; o < r; ++o) {
            float acc = w_encoder_en_convs_0_ops_4_fa_fc_bias[o];
            for (int j = 0; j < 8; ++j) acc += w_encoder_en_convs_0_ops_4_fa_fc_weight[o * 8 + j] * fa_gru_out[h * 8 + j];
            fa_fc_out[h * r + o] = acc;
        }
    }
    float af[65]; /* unpad to F_out=65, sigmoid */
    for (int f = 0; f < F_out; ++f) af[f] = sigmoidf_(fa_fc_out[f]);

    /* y = at[c] * x[c,f] * af[f] */
    for (int c = 0; c < C_out; ++c)
        for (int f = 0; f < F_out; ++f)
            conv_out[c * F_out + f] = at[c] * conv_out[c * F_out + f] * af[f];

    /* --- compare to reference --- */
    int n_out;
    float* ref_out = load_npy_f32("/tmp/ref_block0_out.npy", &n_out);
    int n_cache;
    float* ref_cache = load_npy_f32("/tmp/ref_block0_conv_cache_out.npy", &n_cache);
    int n_tfa;
    float* ref_tfa = load_npy_f32("/tmp/ref_block0_tfa_cache_out.npy", &n_tfa);

    double max_err_out = 0, max_err_cache = 0, max_err_tfa = 0;
    for (int i = 0; i < 12 * 65; ++i) { double e = fabs(conv_out[i] - ref_out[i]); if (e > max_err_out) max_err_out = e; }
    for (int i = 0; i < 2 * 129; ++i) { double e = fabs(new_conv_cache[i] - ref_cache[i]); if (e > max_err_cache) max_err_cache = e; }
    for (int i = 0; i < 24; ++i) { double e = fabs(tfa_cache[i] - ref_tfa[i]); if (e > max_err_tfa) max_err_tfa = e; }

    printf("block0 output max abs err: %.8f\n", max_err_out);
    printf("block0 conv_cache_out max abs err: %.8f\n", max_err_cache);
    printf("block0 tfa_cache_out max abs err: %.8f\n", max_err_tfa);
    printf("PASS: %s\n", (erb_err < 1e-4 && max_err_out < 1e-3 && max_err_cache < 1e-4 && max_err_tfa < 1e-3) ? "YES" : "NO");
    return 0;
}
