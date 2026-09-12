/* Regression test: the NEW generic conv_bn_act_ctfa / pointwise_bn_act / ctfa_apply /
 * dpgrnn helpers (kernels.h) must reproduce the SAME outputs as the original hand-written
 * block0/block1/block2/decoder4/dpgrnn tests, which were themselves validated against the
 * real PyTorch StreamULUNAS reference. This isolates "did the refactor break anything" from
 * "is the math right" (already answered by the original tests). */
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

static double max_abs_err(const float* a, const float* b, int n) {
    double m = 0;
    for (int i = 0; i < n; ++i) { double e = fabs(a[i] - b[i]); if (e > m) m = e; }
    return m;
}

static int all_pass = 1;
static void check(const char* name, double err, double thresh) {
    printf("%-40s max_err=%.8f thresh=%.1e %s\n", name, err, thresh, err < thresh ? "PASS" : "FAIL");
    if (!(err < thresh)) all_pass = 0;
}


static void test_dpgrnn_generic() {
    int n;
    float* x_in = load_npy_f32("/tmp/ref_dpgrnn_input.npy", &n);
    float inter_cache[33*16]; memset(inter_cache, 0, sizeof(inter_cache));
    float out[16*33];
    DpgrnnSpec s = {0};
    s.intra_wih1=w_dpgrnn_0_intra_rnn_rnn1_weight_ih_l0; s.intra_whh1=w_dpgrnn_0_intra_rnn_rnn1_weight_hh_l0;
    s.intra_bih1=w_dpgrnn_0_intra_rnn_rnn1_bias_ih_l0; s.intra_bhh1=w_dpgrnn_0_intra_rnn_rnn1_bias_hh_l0;
    s.intra_wih1_rev=w_dpgrnn_0_intra_rnn_rnn1_weight_ih_l0_reverse; s.intra_whh1_rev=w_dpgrnn_0_intra_rnn_rnn1_weight_hh_l0_reverse;
    s.intra_bih1_rev=w_dpgrnn_0_intra_rnn_rnn1_bias_ih_l0_reverse; s.intra_bhh1_rev=w_dpgrnn_0_intra_rnn_rnn1_bias_hh_l0_reverse;
    s.intra_wih2=w_dpgrnn_0_intra_rnn_rnn2_weight_ih_l0; s.intra_whh2=w_dpgrnn_0_intra_rnn_rnn2_weight_hh_l0;
    s.intra_bih2=w_dpgrnn_0_intra_rnn_rnn2_bias_ih_l0; s.intra_bhh2=w_dpgrnn_0_intra_rnn_rnn2_bias_hh_l0;
    s.intra_wih2_rev=w_dpgrnn_0_intra_rnn_rnn2_weight_ih_l0_reverse; s.intra_whh2_rev=w_dpgrnn_0_intra_rnn_rnn2_weight_hh_l0_reverse;
    s.intra_bih2_rev=w_dpgrnn_0_intra_rnn_rnn2_bias_ih_l0_reverse; s.intra_bhh2_rev=w_dpgrnn_0_intra_rnn_rnn2_bias_hh_l0_reverse;
    s.intra_fc_w=w_dpgrnn_0_intra_fc_weight; s.intra_fc_b=w_dpgrnn_0_intra_fc_bias;
    s.intra_ln_w=w_dpgrnn_0_intra_ln_weight; s.intra_ln_b=w_dpgrnn_0_intra_ln_bias;
    s.inter_wih1=w_dpgrnn_0_inter_rnn_rnn1_weight_ih_l0; s.inter_whh1=w_dpgrnn_0_inter_rnn_rnn1_weight_hh_l0;
    s.inter_bih1=w_dpgrnn_0_inter_rnn_rnn1_bias_ih_l0; s.inter_bhh1=w_dpgrnn_0_inter_rnn_rnn1_bias_hh_l0;
    s.inter_wih2=w_dpgrnn_0_inter_rnn_rnn2_weight_ih_l0; s.inter_whh2=w_dpgrnn_0_inter_rnn_rnn2_weight_hh_l0;
    s.inter_bih2=w_dpgrnn_0_inter_rnn_rnn2_bias_ih_l0; s.inter_bhh2=w_dpgrnn_0_inter_rnn_rnn2_bias_hh_l0;
    s.inter_fc_w=w_dpgrnn_0_inter_fc_weight; s.inter_fc_b=w_dpgrnn_0_inter_fc_bias;
    s.inter_ln_w=w_dpgrnn_0_inter_ln_weight; s.inter_ln_b=w_dpgrnn_0_inter_ln_bias;

    dpgrnn_forward(&s, x_in, inter_cache, out);

    float* ref_out = load_npy_f32("/tmp/ref_dpgrnn_out.npy", &n);
    float* ref_cache = load_npy_f32("/tmp/ref_dpgrnn_inter_cache_out.npy", &n);
    check("dpgrnn(generic) output", max_abs_err(out, ref_out, 16*33), 1e-3);
    check("dpgrnn(generic) inter_cache", max_abs_err(inter_cache, ref_cache, 33*16), 1e-3);
}

int main() {
    int n;

    /* ---- block0 (XConvBlock, encoder) via generic conv_bn_act_ctfa ---- */
    {
        float* feat_erb = load_npy_f32("/tmp/ref_feat_erb.npy", &n);
        float conv_cache[1 * 2 * 129]; memset(conv_cache, 0, sizeof(conv_cache));
        float tfa_cache[24]; memset(tfa_cache, 0, sizeof(tfa_cache));
        float out[12 * 65];

        ConvBnActCtfaSpec spec = {0};
        spec.conv_w = w_encoder_en_convs_0_ops_1_weight; spec.conv_b = w_encoder_en_convs_0_ops_1_bias;
        spec.bn_w = w_encoder_en_convs_0_ops_2_weight; spec.bn_b = w_encoder_en_convs_0_ops_2_bias;
        spec.bn_rm = w_encoder_en_convs_0_ops_2_running_mean; spec.bn_rv = w_encoder_en_convs_0_ops_2_running_var;
        spec.affine_w = w_encoder_en_convs_0_ops_3_affine_weight; spec.affine_b = w_encoder_en_convs_0_ops_3_affine_bias;
        spec.slope = w_encoder_en_convs_0_ops_3_slope_weight;
        spec.has_ctfa = 1;
        spec.ctfa.C = 12; spec.ctfa.F = 65; spec.ctfa.ta_hidden = 24;
        spec.ctfa.ta_wih = w_encoder_en_convs_0_ops_4_ta_gru_weight_ih_l0;
        spec.ctfa.ta_whh = w_encoder_en_convs_0_ops_4_ta_gru_weight_hh_l0;
        spec.ctfa.ta_bih = w_encoder_en_convs_0_ops_4_ta_gru_bias_ih_l0;
        spec.ctfa.ta_bhh = w_encoder_en_convs_0_ops_4_ta_gru_bias_hh_l0;
        spec.ctfa.ta_fc_w = w_encoder_en_convs_0_ops_4_ta_fc_weight; spec.ctfa.ta_fc_b = w_encoder_en_convs_0_ops_4_ta_fc_bias;
        spec.ctfa.fa_wih_f = w_encoder_en_convs_0_ops_4_fa_gru_weight_ih_l0;
        spec.ctfa.fa_whh_f = w_encoder_en_convs_0_ops_4_fa_gru_weight_hh_l0;
        spec.ctfa.fa_bih_f = w_encoder_en_convs_0_ops_4_fa_gru_bias_ih_l0;
        spec.ctfa.fa_bhh_f = w_encoder_en_convs_0_ops_4_fa_gru_bias_hh_l0;
        spec.ctfa.fa_wih_b = w_encoder_en_convs_0_ops_4_fa_gru_weight_ih_l0_reverse;
        spec.ctfa.fa_whh_b = w_encoder_en_convs_0_ops_4_fa_gru_weight_hh_l0_reverse;
        spec.ctfa.fa_bih_b = w_encoder_en_convs_0_ops_4_fa_gru_bias_ih_l0_reverse;
        spec.ctfa.fa_bhh_b = w_encoder_en_convs_0_ops_4_fa_gru_bias_hh_l0_reverse;
        spec.ctfa.fa_fc_w = w_encoder_en_convs_0_ops_4_fa_fc_weight; spec.ctfa.fa_fc_b = w_encoder_en_convs_0_ops_4_fa_fc_bias;
        spec.C_in = 1; spec.C_out = 12; spec.KT = 3; spec.KF = 3; spec.stride_f = 2; spec.pf = 1;
        spec.groups = 1; spec.F_in = 129; spec.F_out = 65; spec.use_deconv = 0;

        conv_bn_act_ctfa(&spec, feat_erb, conv_cache, tfa_cache, out);

        float* ref_out = load_npy_f32("/tmp/ref_block0_out.npy", &n);
        float* ref_cache = load_npy_f32("/tmp/ref_block0_conv_cache_out.npy", &n);
        float* ref_tfa = load_npy_f32("/tmp/ref_block0_tfa_cache_out.npy", &n);
        check("block0(generic) output", max_abs_err(out, ref_out, 12*65), 1e-3);
        check("block0(generic) conv_cache", max_abs_err(conv_cache, ref_cache, 2*129), 1e-4);
        check("block0(generic) tfa_cache", max_abs_err(tfa_cache, ref_tfa, 24), 1e-3);
    }

    /* ---- block2 (XDWSBlock, encoder) via generic pointwise_bn_act + conv_bn_act_ctfa ---- */
    {
        float* x_in = load_npy_f32("/tmp/ref_block2_input.npy", &n); /* (1,24,1,33) */
        float pconv_out[24 * 33];
        PointwiseSpec pw = {0};
        pw.conv_w = w_encoder_en_convs_2_pconv_0_weight; pw.conv_b = w_encoder_en_convs_2_pconv_0_bias;
        pw.bn_w = w_encoder_en_convs_2_pconv_1_weight; pw.bn_b = w_encoder_en_convs_2_pconv_1_bias;
        pw.bn_rm = w_encoder_en_convs_2_pconv_1_running_mean; pw.bn_rv = w_encoder_en_convs_2_pconv_1_running_var;
        pw.affine_w = w_encoder_en_convs_2_pconv_2_affine_weight; pw.affine_b = w_encoder_en_convs_2_pconv_2_affine_bias;
        pw.slope = w_encoder_en_convs_2_pconv_2_slope_weight;
        pw.C_in = 24; pw.C_out = 24; pw.groups = 2; pw.F = 33; pw.do_shuffle = 1;
        pointwise_bn_act(&pw, x_in, pconv_out);

        float conv_cache[24 * 1 * 33]; memset(conv_cache, 0, sizeof(conv_cache));
        float tfa_cache[48]; memset(tfa_cache, 0, sizeof(tfa_cache));
        float out[24 * 33];
        ConvBnActCtfaSpec spec = {0};
        spec.conv_w = w_encoder_en_convs_2_dconv_1_weight; spec.conv_b = w_encoder_en_convs_2_dconv_1_bias;
        spec.bn_w = w_encoder_en_convs_2_dconv_2_weight; spec.bn_b = w_encoder_en_convs_2_dconv_2_bias;
        spec.bn_rm = w_encoder_en_convs_2_dconv_2_running_mean; spec.bn_rv = w_encoder_en_convs_2_dconv_2_running_var;
        spec.affine_w = w_encoder_en_convs_2_dconv_3_affine_weight; spec.affine_b = w_encoder_en_convs_2_dconv_3_affine_bias;
        spec.slope = w_encoder_en_convs_2_dconv_3_slope_weight;
        spec.has_ctfa = 1;
        spec.ctfa.C = 24; spec.ctfa.F = 33; spec.ctfa.ta_hidden = 48;
        spec.ctfa.ta_wih = w_encoder_en_convs_2_dconv_4_ta_gru_weight_ih_l0;
        spec.ctfa.ta_whh = w_encoder_en_convs_2_dconv_4_ta_gru_weight_hh_l0;
        spec.ctfa.ta_bih = w_encoder_en_convs_2_dconv_4_ta_gru_bias_ih_l0;
        spec.ctfa.ta_bhh = w_encoder_en_convs_2_dconv_4_ta_gru_bias_hh_l0;
        spec.ctfa.ta_fc_w = w_encoder_en_convs_2_dconv_4_ta_fc_weight; spec.ctfa.ta_fc_b = w_encoder_en_convs_2_dconv_4_ta_fc_bias;
        spec.ctfa.fa_wih_f = w_encoder_en_convs_2_dconv_4_fa_gru_weight_ih_l0;
        spec.ctfa.fa_whh_f = w_encoder_en_convs_2_dconv_4_fa_gru_weight_hh_l0;
        spec.ctfa.fa_bih_f = w_encoder_en_convs_2_dconv_4_fa_gru_bias_ih_l0;
        spec.ctfa.fa_bhh_f = w_encoder_en_convs_2_dconv_4_fa_gru_bias_hh_l0;
        spec.ctfa.fa_wih_b = w_encoder_en_convs_2_dconv_4_fa_gru_weight_ih_l0_reverse;
        spec.ctfa.fa_whh_b = w_encoder_en_convs_2_dconv_4_fa_gru_weight_hh_l0_reverse;
        spec.ctfa.fa_bih_b = w_encoder_en_convs_2_dconv_4_fa_gru_bias_ih_l0_reverse;
        spec.ctfa.fa_bhh_b = w_encoder_en_convs_2_dconv_4_fa_gru_bias_hh_l0_reverse;
        spec.ctfa.fa_fc_w = w_encoder_en_convs_2_dconv_4_fa_fc_weight; spec.ctfa.fa_fc_b = w_encoder_en_convs_2_dconv_4_fa_fc_bias;
        spec.C_in = 24; spec.C_out = 24; spec.KT = 2; spec.KF = 3; spec.stride_f = 1; spec.pf = 1;
        spec.groups = 24; spec.F_in = 33; spec.F_out = 33; spec.use_deconv = 0;

        conv_bn_act_ctfa(&spec, pconv_out, conv_cache, tfa_cache, out);

        float* ref_out = load_npy_f32("/tmp/ref_block2_out.npy", &n);
        float* ref_cache = load_npy_f32("/tmp/ref_block2_conv_cache_out.npy", &n);
        float* ref_tfa = load_npy_f32("/tmp/ref_block2_tfa_cache_out.npy", &n);
        check("block2(generic) output", max_abs_err(out, ref_out, 24*33), 1e-3);
        check("block2(generic) conv_cache", max_abs_err(conv_cache, ref_cache, 33), 1e-4);
        check("block2(generic) tfa_cache", max_abs_err(tfa_cache, ref_tfa, 48), 1e-3);
    }

    /* ---- decoder4 (XConvBlock, is_last, ConvTranspose2d) via generic conv_bn_act_ctfa ---- */
    {
        float* x_in = load_npy_f32("/tmp/ref_dec4_input.npy", &n); /* (1,12,1,65) */
        float conv_cache[12 * 2 * 65]; memset(conv_cache, 0, sizeof(conv_cache));
        float tfa_cache[2]; memset(tfa_cache, 0, sizeof(tfa_cache));
        float out[1 * 129];

        ConvBnActCtfaSpec spec = {0};
        spec.conv_w = w_decoder_de_convs_4_ops_1_weight; spec.conv_b = w_decoder_de_convs_4_ops_1_bias;
        spec.bn_w = w_decoder_de_convs_4_ops_2_weight; spec.bn_b = w_decoder_de_convs_4_ops_2_bias;
        spec.bn_rm = w_decoder_de_convs_4_ops_2_running_mean; spec.bn_rv = w_decoder_de_convs_4_ops_2_running_var;
        spec.affine_w = NULL; /* is_last: no AffinePReLU */
        spec.has_ctfa = 1;
        spec.ctfa.C = 1; spec.ctfa.F = 129; spec.ctfa.ta_hidden = 2;
        spec.ctfa.ta_wih = w_decoder_de_convs_4_ops_4_ta_gru_weight_ih_l0;
        spec.ctfa.ta_whh = w_decoder_de_convs_4_ops_4_ta_gru_weight_hh_l0;
        spec.ctfa.ta_bih = w_decoder_de_convs_4_ops_4_ta_gru_bias_ih_l0;
        spec.ctfa.ta_bhh = w_decoder_de_convs_4_ops_4_ta_gru_bias_hh_l0;
        spec.ctfa.ta_fc_w = w_decoder_de_convs_4_ops_4_ta_fc_weight; spec.ctfa.ta_fc_b = w_decoder_de_convs_4_ops_4_ta_fc_bias;
        spec.ctfa.fa_wih_f = w_decoder_de_convs_4_ops_4_fa_gru_weight_ih_l0;
        spec.ctfa.fa_whh_f = w_decoder_de_convs_4_ops_4_fa_gru_weight_hh_l0;
        spec.ctfa.fa_bih_f = w_decoder_de_convs_4_ops_4_fa_gru_bias_ih_l0;
        spec.ctfa.fa_bhh_f = w_decoder_de_convs_4_ops_4_fa_gru_bias_hh_l0;
        spec.ctfa.fa_wih_b = w_decoder_de_convs_4_ops_4_fa_gru_weight_ih_l0_reverse;
        spec.ctfa.fa_whh_b = w_decoder_de_convs_4_ops_4_fa_gru_weight_hh_l0_reverse;
        spec.ctfa.fa_bih_b = w_decoder_de_convs_4_ops_4_fa_gru_bias_ih_l0_reverse;
        spec.ctfa.fa_bhh_b = w_decoder_de_convs_4_ops_4_fa_gru_bias_hh_l0_reverse;
        spec.ctfa.fa_fc_w = w_decoder_de_convs_4_ops_4_fa_fc_weight; spec.ctfa.fa_fc_b = w_decoder_de_convs_4_ops_4_fa_fc_bias;
        spec.C_in = 12; spec.C_out = 1; spec.KT = 3; spec.KF = 3; spec.stride_f = 2; spec.pf = 1;
        spec.groups = 1; spec.F_in = 65; spec.F_out = 129; spec.use_deconv = 1;

        conv_bn_act_ctfa(&spec, x_in, conv_cache, tfa_cache, out);

        float* ref_out = load_npy_f32("/tmp/ref_dec4_out.npy", &n);
        float* ref_cache = load_npy_f32("/tmp/ref_dec4_conv_cache_out.npy", &n);
        float* ref_tfa = load_npy_f32("/tmp/ref_dec4_tfa_cache_out.npy", &n);
        check("decoder4(generic) output", max_abs_err(out, ref_out, 129), 1e-3);
        check("decoder4(generic) conv_cache", max_abs_err(conv_cache, ref_cache, 2*65), 1e-4);
        check("decoder4(generic) tfa_cache", max_abs_err(tfa_cache, ref_tfa, 2), 1e-3);
    }

    test_dpgrnn_generic();

    printf("\n=== OVERALL: %s ===\n", all_pass ? "ALL PASS" : "SOME FAILED");
    return all_pass ? 0 : 1;
}
