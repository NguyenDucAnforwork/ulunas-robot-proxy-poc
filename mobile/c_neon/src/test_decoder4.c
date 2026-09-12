/* Validates decoder block 4 (final XConvBlock, use_deconv=True, is_last=True: ConvTranspose2d
 * -> BatchNorm -> (no AffinePReLU, is_last) -> cTFA -> (no final Shuffle, is_last)) against
 * StreamULUNAS._stream_xconv on the decoder side. This is the piece that specifically
 * exercises convtranspose2d_causal_1frame wired into the same block pattern already
 * validated with regular Conv2d on the encoder side. */
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
    float* x_in = load_npy_f32("/tmp/ref_dec4_input.npy", &n); /* (1,12,1,65) */
    int C_in = 12, C_out = 1, KT = 3, KF = 3, stride_f = 2, pf = 1, F_in = 65, F_out = 129;

    float conv_cache[12 * 2 * 65]; /* CONV_CACHE_SHAPES[5] = (12,2,65) */
    memset(conv_cache, 0, sizeof(conv_cache));
    float conv_in[12 * 3 * 65];
    for (int c = 0; c < C_in; ++c) {
        memcpy(conv_in + c * 3 * 65, conv_cache + c * 2 * 65, 2 * 65 * sizeof(float));
        memcpy(conv_in + c * 3 * 65 + 2 * 65, x_in + c * 65, 65 * sizeof(float));
    }

    float conv_out[1 * 129];
    convtranspose2d_causal_1frame(conv_in, C_in, /*T_in=*/3, F_in, w_decoder_de_convs_4_ops_1_weight,
                                  w_decoder_de_convs_4_ops_1_bias, C_out, KT, KF, stride_f, pf,
                                  /*groups=*/1, conv_out, F_out);

    float new_conv_cache[12 * 2 * 65];
    for (int c = 0; c < C_in; ++c)
        memcpy(new_conv_cache + c * 2 * 65, conv_in + c * 3 * 65 + 65, 2 * 65 * sizeof(float));

    batchnorm_apply(conv_out, C_out, F_out, w_decoder_de_convs_4_ops_2_weight,
                    w_decoder_de_convs_4_ops_2_bias, w_decoder_de_convs_4_ops_2_running_mean,
                    w_decoder_de_convs_4_ops_2_running_var, 1e-5f);
    /* is_last=True: NO AffinePReLU */

    /* cTFA(channels=1, width=129) */
    float tfa_cache[2];
    memset(tfa_cache, 0, sizeof(tfa_cache));
    float zt[1];
    { float acc = 0; for (int f = 0; f < F_out; ++f) acc += conv_out[f] * conv_out[f]; zt[0] = acc / F_out; }
    gru_step(zt, 1, tfa_cache, 2, w_decoder_de_convs_4_ops_4_ta_gru_weight_ih_l0,
            w_decoder_de_convs_4_ops_4_ta_gru_weight_hh_l0, w_decoder_de_convs_4_ops_4_ta_gru_bias_ih_l0,
            w_decoder_de_convs_4_ops_4_ta_gru_bias_hh_l0);
    float at;
    { float acc = w_decoder_de_convs_4_ops_4_ta_fc_bias[0];
      for (int j = 0; j < 2; ++j) acc += w_decoder_de_convs_4_ops_4_ta_fc_weight[j] * tfa_cache[j];
      at = sigmoidf_(acc); }

    int r = 4, F_pad = 132, H = 33;
    float fa_in[132]; memset(fa_in, 0, sizeof(fa_in));
    for (int f = 0; f < F_out; ++f) fa_in[f] = conv_out[f] * conv_out[f]; /* mean over C=1 is identity */
    float fa_gru_out[33 * 8];
    gru_bidirectional_seq(fa_in, H, r, r,
        w_decoder_de_convs_4_ops_4_fa_gru_weight_ih_l0, w_decoder_de_convs_4_ops_4_fa_gru_weight_hh_l0,
        w_decoder_de_convs_4_ops_4_fa_gru_bias_ih_l0, w_decoder_de_convs_4_ops_4_fa_gru_bias_hh_l0,
        w_decoder_de_convs_4_ops_4_fa_gru_weight_ih_l0_reverse, w_decoder_de_convs_4_ops_4_fa_gru_weight_hh_l0_reverse,
        w_decoder_de_convs_4_ops_4_fa_gru_bias_ih_l0_reverse, w_decoder_de_convs_4_ops_4_fa_gru_bias_hh_l0_reverse,
        fa_gru_out);
    float fa_fc_out[33 * 4];
    for (int h = 0; h < H; ++h)
        for (int o = 0; o < r; ++o) {
            float acc = w_decoder_de_convs_4_ops_4_fa_fc_bias[o];
            for (int j = 0; j < 8; ++j) acc += w_decoder_de_convs_4_ops_4_fa_fc_weight[o * 8 + j] * fa_gru_out[h * 8 + j];
            fa_fc_out[h * r + o] = acc;
        }
    float af[129];
    for (int f = 0; f < F_out; ++f) af[f] = sigmoidf_(fa_fc_out[f]);

    for (int f = 0; f < F_out; ++f) conv_out[f] = at * conv_out[f] * af[f];
    /* is_last=True: NO final Shuffle */

    int n_out, n_cache, n_tfa;
    float* ref_out = load_npy_f32("/tmp/ref_dec4_out.npy", &n_out);
    float* ref_cache = load_npy_f32("/tmp/ref_dec4_conv_cache_out.npy", &n_cache);
    float* ref_tfa = load_npy_f32("/tmp/ref_dec4_tfa_cache_out.npy", &n_tfa);

    double e_out = 0, e_cache = 0, e_tfa = 0;
    for (int i = 0; i < 129; ++i) { double e = fabs(conv_out[i] - ref_out[i]); if (e > e_out) e_out = e; }
    for (int i = 0; i < 12 * 2 * 65; ++i) { double e = fabs(new_conv_cache[i] - ref_cache[i]); if (e > e_cache) e_cache = e; }
    for (int i = 0; i < 2; ++i) { double e = fabs(tfa_cache[i] - ref_tfa[i]); if (e > e_tfa) e_tfa = e; }

    printf("decoder4 output max abs err: %.8f\n", e_out);
    printf("decoder4 conv_cache_out max abs err: %.8f\n", e_cache);
    printf("decoder4 tfa_cache_out max abs err: %.8f\n", e_tfa);
    printf("PASS: %s\n", (e_out < 1e-3 && e_cache < 1e-4 && e_tfa < 1e-3) ? "YES" : "NO");
    return 0;
}
