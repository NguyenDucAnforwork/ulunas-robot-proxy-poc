/* Full runtime-free UL-UNAS streaming graph: ERB -> 5 encoder blocks -> 2 DPGRNN ->
 * 5 decoder blocks -> ERB^-1 -> complex mask, wired exactly per
 * ul-unas/ulunas_onnx/stream/ulunas_stream.py :: StreamULUNAS.forward() (read directly,
 * not from memory, before writing this file). Every kernel type and every block-type
 * pattern used here (XConvBlock, XMBBlocks, XDWSBlock, DPGRNN, ERB) was independently
 * validated end-to-end against the real PyTorch reference in test_block0/1/2/dpgrnn/
 * decoder4.c and test_generic_regression.c before this file was written.
 *
 * No ONNX Runtime / PyTorch / TensorFlow Lite dependency anywhere in this file. */
#include <math.h>
#include <string.h>

#include "fft.h"
#include "kernels.h"
#include "ulunas_full.h"
#include "../generated/ulunas_weights.h"

#define FREQ 257
#define ERB_LOW 65
#define ERB_HIGH 64
#define ERB_TOTAL 129

/* ---- cache sizes, from CONV_CACHE_SHAPES / TFA_CACHE_HIDDEN / INTER_CACHE_SHAPES in
 * ulunas_stream.py (verbatim, not re-derived) ---- */
#define CC_E0_SZ (1 * 2 * 129)
#define CC_E1_SZ (24 * 1 * 65)
#define CC_E2_SZ (24 * 1 * 33)
#define CC_D2_SZ (24 * 1 * 33)
#define CC_D3_SZ (12 * 1 * 33)
#define CC_D4_SZ (12 * 2 * 65)

struct UlunasState {
    /* --- streaming STFT/ISTFT state --- */
    float analysis_buf[ULUNAS_WIN];      /* last WIN raw input samples (causal ring, zero-init) */
    float ola_buf[ULUNAS_WIN];           /* overlap-add accumulator */
    float win_sum_buf[ULUNAS_WIN];       /* window-power accumulator, matches ola_buf */
    float hann[ULUNAS_WIN];

    /* --- model streaming caches --- */
    float cc_e0[CC_E0_SZ], cc_e1[CC_E1_SZ], cc_e2[CC_E2_SZ];
    float cc_d2[CC_D2_SZ], cc_d3[CC_D3_SZ], cc_d4[CC_D4_SZ];
    float tfa[10][64]; /* max hidden = 64 (en3/de0), padded */
    float inter0[33 * 16], inter1[33 * 16];
};

static void build_ctfa_spec(CtfaSpec *c, int C, int F, int hid,
                            const float *ta_wih, const float *ta_whh, const float *ta_bih, const float *ta_bhh,
                            const float *ta_fc_w, const float *ta_fc_b,
                            const float *fa_wih_f, const float *fa_whh_f, const float *fa_bih_f, const float *fa_bhh_f,
                            const float *fa_wih_b, const float *fa_whh_b, const float *fa_bih_b, const float *fa_bhh_b,
                            const float *fa_fc_w, const float *fa_fc_b) {
    c->C = C; c->F = F; c->ta_hidden = hid;
    c->ta_wih = ta_wih; c->ta_whh = ta_whh; c->ta_bih = ta_bih; c->ta_bhh = ta_bhh;
    c->ta_fc_w = ta_fc_w; c->ta_fc_b = ta_fc_b;
    c->fa_wih_f = fa_wih_f; c->fa_whh_f = fa_whh_f; c->fa_bih_f = fa_bih_f; c->fa_bhh_f = fa_bhh_f;
    c->fa_wih_b = fa_wih_b; c->fa_whh_b = fa_whh_b; c->fa_bih_b = fa_bih_b; c->fa_bhh_b = fa_bhh_b;
    c->fa_fc_w = fa_fc_w; c->fa_fc_b = fa_fc_b;
}

/* One forward pass of the full graph on a single 257-bin complex spectral frame.
 * mix_spec/enh_spec: [257][2] (re,im). All caches read from and written back into `st`. */
void ulunas_process_frame_spec(UlunasState *st, const float *mix_spec, float *enh_spec) {
    /* ---- magnitude + log10 + ERB.bm ---- */
    float feat257[FREQ];
    for (int f = 0; f < FREQ; ++f) {
        float re = mix_spec[f * 2 + 0], im = mix_spec[f * 2 + 1];
        float mag = sqrtf(re * re + im * im);
        if (mag < 1e-12f) mag = 1e-12f;
        feat257[f] = log10f(mag);
    }
    float feat[ERB_TOTAL];
    for (int i = 0; i < ERB_LOW; ++i) feat[i] = feat257[i];
    for (int o = 0; o < ERB_HIGH; ++o) {
        float acc = 0.0f;
        for (int i = 0; i < (FREQ - ERB_LOW); ++i)
            acc += feat257[ERB_LOW + i] * w_erb_erb_fc_weight[o * (FREQ - ERB_LOW) + i];
        feat[ERB_LOW + o] = acc;
    }

    /* ================= ENCODER ================= */
    float en0[12 * 65], en1[24 * 33], en2[24 * 33], en3[32 * 33], en4[16 * 33];

    /* en0: XConvBlock, 1->12, kt=3,kf=3,stride_f=2,pf=1,groups=1, width 129->65 */
    {
        ConvBnActCtfaSpec s = {0};
        s.conv_w = w_encoder_en_convs_0_ops_1_weight; s.conv_b = w_encoder_en_convs_0_ops_1_bias;
        s.bn_w = w_encoder_en_convs_0_ops_2_weight; s.bn_b = w_encoder_en_convs_0_ops_2_bias;
        s.bn_rm = w_encoder_en_convs_0_ops_2_running_mean; s.bn_rv = w_encoder_en_convs_0_ops_2_running_var;
        s.affine_w = w_encoder_en_convs_0_ops_3_affine_weight; s.affine_b = w_encoder_en_convs_0_ops_3_affine_bias;
        s.slope = w_encoder_en_convs_0_ops_3_slope_weight;
        s.has_ctfa = 1;
        build_ctfa_spec(&s.ctfa, 12, 65, 24,
            w_encoder_en_convs_0_ops_4_ta_gru_weight_ih_l0, w_encoder_en_convs_0_ops_4_ta_gru_weight_hh_l0,
            w_encoder_en_convs_0_ops_4_ta_gru_bias_ih_l0, w_encoder_en_convs_0_ops_4_ta_gru_bias_hh_l0,
            w_encoder_en_convs_0_ops_4_ta_fc_weight, w_encoder_en_convs_0_ops_4_ta_fc_bias,
            w_encoder_en_convs_0_ops_4_fa_gru_weight_ih_l0, w_encoder_en_convs_0_ops_4_fa_gru_weight_hh_l0,
            w_encoder_en_convs_0_ops_4_fa_gru_bias_ih_l0, w_encoder_en_convs_0_ops_4_fa_gru_bias_hh_l0,
            w_encoder_en_convs_0_ops_4_fa_gru_weight_ih_l0_reverse, w_encoder_en_convs_0_ops_4_fa_gru_weight_hh_l0_reverse,
            w_encoder_en_convs_0_ops_4_fa_gru_bias_ih_l0_reverse, w_encoder_en_convs_0_ops_4_fa_gru_bias_hh_l0_reverse,
            w_encoder_en_convs_0_ops_4_fa_fc_weight, w_encoder_en_convs_0_ops_4_fa_fc_bias);
        s.C_in = 1; s.C_out = 12; s.KT = 3; s.KF = 3; s.stride_f = 2; s.pf = 1; s.groups = 1;
        s.F_in = 129; s.F_out = 65; s.use_deconv = 0;
        conv_bn_act_ctfa(&s, feat, st->cc_e0, st->tfa[0], en0);
    }

    /* en1: XMBBlocks, 12->24, groups=2, kt=2,kf=3,stride_f=2,pf=1, width 65->33 */
    {
        float p1[24 * 65];
        PointwiseSpec pw = {0};
        pw.conv_w = w_encoder_en_convs_1_pconv1_0_weight; pw.conv_b = w_encoder_en_convs_1_pconv1_0_bias;
        pw.bn_w = w_encoder_en_convs_1_pconv1_1_weight; pw.bn_b = w_encoder_en_convs_1_pconv1_1_bias;
        pw.bn_rm = w_encoder_en_convs_1_pconv1_1_running_mean; pw.bn_rv = w_encoder_en_convs_1_pconv1_1_running_var;
        pw.affine_w = w_encoder_en_convs_1_pconv1_2_affine_weight; pw.affine_b = w_encoder_en_convs_1_pconv1_2_affine_bias;
        pw.slope = w_encoder_en_convs_1_pconv1_2_slope_weight;
        pw.C_in = 12; pw.C_out = 24; pw.groups = 2; pw.F = 65; pw.do_shuffle = 1;
        pointwise_bn_act(&pw, en0, p1);

        float mid[24 * 33];
        ConvBnActCtfaSpec sd = {0};
        sd.conv_w = w_encoder_en_convs_1_dconv_1_weight; sd.conv_b = w_encoder_en_convs_1_dconv_1_bias;
        sd.bn_w = w_encoder_en_convs_1_dconv_2_weight; sd.bn_b = w_encoder_en_convs_1_dconv_2_bias;
        sd.bn_rm = w_encoder_en_convs_1_dconv_2_running_mean; sd.bn_rv = w_encoder_en_convs_1_dconv_2_running_var;
        sd.affine_w = w_encoder_en_convs_1_dconv_3_affine_weight; sd.affine_b = w_encoder_en_convs_1_dconv_3_affine_bias;
        sd.slope = w_encoder_en_convs_1_dconv_3_slope_weight;
        sd.has_ctfa = 0;
        sd.C_in = 24; sd.C_out = 24; sd.KT = 2; sd.KF = 3; sd.stride_f = 2; sd.pf = 1; sd.groups = 24;
        sd.F_in = 65; sd.F_out = 33; sd.use_deconv = 0;
        conv_bn_act_ctfa(&sd, p1, st->cc_e1, NULL, mid);

        float p2[24 * 33];
        conv2d_causal_1frame(mid, 24, 1, 33, w_encoder_en_convs_1_pconv2_0_weight,
                             w_encoder_en_convs_1_pconv2_0_bias, 24, 1, 1, 0, 2, p2, 33);
        batchnorm_apply(p2, 24, 33, w_encoder_en_convs_1_pconv2_1_weight, w_encoder_en_convs_1_pconv2_1_bias,
                        w_encoder_en_convs_1_pconv2_1_running_mean, w_encoder_en_convs_1_pconv2_1_running_var, 1e-5f);
        CtfaSpec ctfa;
        build_ctfa_spec(&ctfa, 24, 33, 48,
            w_encoder_en_convs_1_pconv2_2_ta_gru_weight_ih_l0, w_encoder_en_convs_1_pconv2_2_ta_gru_weight_hh_l0,
            w_encoder_en_convs_1_pconv2_2_ta_gru_bias_ih_l0, w_encoder_en_convs_1_pconv2_2_ta_gru_bias_hh_l0,
            w_encoder_en_convs_1_pconv2_2_ta_fc_weight, w_encoder_en_convs_1_pconv2_2_ta_fc_bias,
            w_encoder_en_convs_1_pconv2_2_fa_gru_weight_ih_l0, w_encoder_en_convs_1_pconv2_2_fa_gru_weight_hh_l0,
            w_encoder_en_convs_1_pconv2_2_fa_gru_bias_ih_l0, w_encoder_en_convs_1_pconv2_2_fa_gru_bias_hh_l0,
            w_encoder_en_convs_1_pconv2_2_fa_gru_weight_ih_l0_reverse, w_encoder_en_convs_1_pconv2_2_fa_gru_weight_hh_l0_reverse,
            w_encoder_en_convs_1_pconv2_2_fa_gru_bias_ih_l0_reverse, w_encoder_en_convs_1_pconv2_2_fa_gru_bias_hh_l0_reverse,
            w_encoder_en_convs_1_pconv2_2_fa_fc_weight, w_encoder_en_convs_1_pconv2_2_fa_fc_bias);
        ctfa_apply(&ctfa, p2, st->tfa[1]);
        /* residual: shapes (12,65) vs (24,33) differ -> skip add (matches reference) */
        shuffle_apply(p2, 12, 33, en1);
    }

    /* en2: XDWSBlock, 24->24, groups=2, kt=2,kf=3,stride_f=1,pf=1, width 33->33 */
    {
        float p[24 * 33];
        PointwiseSpec pw = {0};
        pw.conv_w = w_encoder_en_convs_2_pconv_0_weight; pw.conv_b = w_encoder_en_convs_2_pconv_0_bias;
        pw.bn_w = w_encoder_en_convs_2_pconv_1_weight; pw.bn_b = w_encoder_en_convs_2_pconv_1_bias;
        pw.bn_rm = w_encoder_en_convs_2_pconv_1_running_mean; pw.bn_rv = w_encoder_en_convs_2_pconv_1_running_var;
        pw.affine_w = w_encoder_en_convs_2_pconv_2_affine_weight; pw.affine_b = w_encoder_en_convs_2_pconv_2_affine_bias;
        pw.slope = w_encoder_en_convs_2_pconv_2_slope_weight;
        pw.C_in = 24; pw.C_out = 24; pw.groups = 2; pw.F = 33; pw.do_shuffle = 1;
        pointwise_bn_act(&pw, en1, p);

        ConvBnActCtfaSpec s = {0};
        s.conv_w = w_encoder_en_convs_2_dconv_1_weight; s.conv_b = w_encoder_en_convs_2_dconv_1_bias;
        s.bn_w = w_encoder_en_convs_2_dconv_2_weight; s.bn_b = w_encoder_en_convs_2_dconv_2_bias;
        s.bn_rm = w_encoder_en_convs_2_dconv_2_running_mean; s.bn_rv = w_encoder_en_convs_2_dconv_2_running_var;
        s.affine_w = w_encoder_en_convs_2_dconv_3_affine_weight; s.affine_b = w_encoder_en_convs_2_dconv_3_affine_bias;
        s.slope = w_encoder_en_convs_2_dconv_3_slope_weight;
        s.has_ctfa = 1;
        build_ctfa_spec(&s.ctfa, 24, 33, 48,
            w_encoder_en_convs_2_dconv_4_ta_gru_weight_ih_l0, w_encoder_en_convs_2_dconv_4_ta_gru_weight_hh_l0,
            w_encoder_en_convs_2_dconv_4_ta_gru_bias_ih_l0, w_encoder_en_convs_2_dconv_4_ta_gru_bias_hh_l0,
            w_encoder_en_convs_2_dconv_4_ta_fc_weight, w_encoder_en_convs_2_dconv_4_ta_fc_bias,
            w_encoder_en_convs_2_dconv_4_fa_gru_weight_ih_l0, w_encoder_en_convs_2_dconv_4_fa_gru_weight_hh_l0,
            w_encoder_en_convs_2_dconv_4_fa_gru_bias_ih_l0, w_encoder_en_convs_2_dconv_4_fa_gru_bias_hh_l0,
            w_encoder_en_convs_2_dconv_4_fa_gru_weight_ih_l0_reverse, w_encoder_en_convs_2_dconv_4_fa_gru_weight_hh_l0_reverse,
            w_encoder_en_convs_2_dconv_4_fa_gru_bias_ih_l0_reverse, w_encoder_en_convs_2_dconv_4_fa_gru_bias_hh_l0_reverse,
            w_encoder_en_convs_2_dconv_4_fa_fc_weight, w_encoder_en_convs_2_dconv_4_fa_fc_bias);
        s.C_in = 24; s.C_out = 24; s.KT = 2; s.KF = 3; s.stride_f = 1; s.pf = 1; s.groups = 24;
        s.F_in = 33; s.F_out = 33; s.use_deconv = 0;
        conv_bn_act_ctfa(&s, p, st->cc_e2, st->tfa[2], en2);
    }

    /* en3: XMBBlocks, 24->32, groups=2, kt=1(NO CACHE),kf=5,stride_f=1,pf=2, width 33->33 */
    {
        float p1[32 * 33];
        PointwiseSpec pw = {0};
        pw.conv_w = w_encoder_en_convs_3_pconv1_0_weight; pw.conv_b = w_encoder_en_convs_3_pconv1_0_bias;
        pw.bn_w = w_encoder_en_convs_3_pconv1_1_weight; pw.bn_b = w_encoder_en_convs_3_pconv1_1_bias;
        pw.bn_rm = w_encoder_en_convs_3_pconv1_1_running_mean; pw.bn_rv = w_encoder_en_convs_3_pconv1_1_running_var;
        pw.affine_w = w_encoder_en_convs_3_pconv1_2_affine_weight; pw.affine_b = w_encoder_en_convs_3_pconv1_2_affine_bias;
        pw.slope = w_encoder_en_convs_3_pconv1_2_slope_weight;
        pw.C_in = 24; pw.C_out = 32; pw.groups = 2; pw.F = 33; pw.do_shuffle = 1;
        pointwise_bn_act(&pw, en2, p1);

        float mid[32 * 33];
        ConvBnActCtfaSpec sd = {0};
        sd.conv_w = w_encoder_en_convs_3_dconv_1_weight; sd.conv_b = w_encoder_en_convs_3_dconv_1_bias;
        sd.bn_w = w_encoder_en_convs_3_dconv_2_weight; sd.bn_b = w_encoder_en_convs_3_dconv_2_bias;
        sd.bn_rm = w_encoder_en_convs_3_dconv_2_running_mean; sd.bn_rv = w_encoder_en_convs_3_dconv_2_running_var;
        sd.affine_w = w_encoder_en_convs_3_dconv_3_affine_weight; sd.affine_b = w_encoder_en_convs_3_dconv_3_affine_bias;
        sd.slope = w_encoder_en_convs_3_dconv_3_slope_weight;
        sd.has_ctfa = 0;
        sd.C_in = 32; sd.C_out = 32; sd.KT = 1; sd.KF = 5; sd.stride_f = 1; sd.pf = 2; sd.groups = 32;
        sd.F_in = 33; sd.F_out = 33; sd.use_deconv = 0;
        conv_bn_act_ctfa(&sd, p1, NULL, NULL, mid);

        float p2[32 * 33];
        conv2d_causal_1frame(mid, 32, 1, 33, w_encoder_en_convs_3_pconv2_0_weight,
                             w_encoder_en_convs_3_pconv2_0_bias, 32, 1, 1, 0, 2, p2, 33);
        batchnorm_apply(p2, 32, 33, w_encoder_en_convs_3_pconv2_1_weight, w_encoder_en_convs_3_pconv2_1_bias,
                        w_encoder_en_convs_3_pconv2_1_running_mean, w_encoder_en_convs_3_pconv2_1_running_var, 1e-5f);
        CtfaSpec ctfa;
        build_ctfa_spec(&ctfa, 32, 33, 64,
            w_encoder_en_convs_3_pconv2_2_ta_gru_weight_ih_l0, w_encoder_en_convs_3_pconv2_2_ta_gru_weight_hh_l0,
            w_encoder_en_convs_3_pconv2_2_ta_gru_bias_ih_l0, w_encoder_en_convs_3_pconv2_2_ta_gru_bias_hh_l0,
            w_encoder_en_convs_3_pconv2_2_ta_fc_weight, w_encoder_en_convs_3_pconv2_2_ta_fc_bias,
            w_encoder_en_convs_3_pconv2_2_fa_gru_weight_ih_l0, w_encoder_en_convs_3_pconv2_2_fa_gru_weight_hh_l0,
            w_encoder_en_convs_3_pconv2_2_fa_gru_bias_ih_l0, w_encoder_en_convs_3_pconv2_2_fa_gru_bias_hh_l0,
            w_encoder_en_convs_3_pconv2_2_fa_gru_weight_ih_l0_reverse, w_encoder_en_convs_3_pconv2_2_fa_gru_weight_hh_l0_reverse,
            w_encoder_en_convs_3_pconv2_2_fa_gru_bias_ih_l0_reverse, w_encoder_en_convs_3_pconv2_2_fa_gru_bias_hh_l0_reverse,
            w_encoder_en_convs_3_pconv2_2_fa_fc_weight, w_encoder_en_convs_3_pconv2_2_fa_fc_bias);
        ctfa_apply(&ctfa, p2, st->tfa[3]);
        /* residual: shapes (24,33) vs (32,33) differ -> skip add */
        shuffle_apply(p2, 16, 33, en3);
    }

    /* en4: XDWSBlock, 32->16, groups=2, kt=1(NO CACHE),kf=5,stride_f=1,pf=2, width 33->33 */
    {
        float p[16 * 33];
        PointwiseSpec pw = {0};
        pw.conv_w = w_encoder_en_convs_4_pconv_0_weight; pw.conv_b = w_encoder_en_convs_4_pconv_0_bias;
        pw.bn_w = w_encoder_en_convs_4_pconv_1_weight; pw.bn_b = w_encoder_en_convs_4_pconv_1_bias;
        pw.bn_rm = w_encoder_en_convs_4_pconv_1_running_mean; pw.bn_rv = w_encoder_en_convs_4_pconv_1_running_var;
        pw.affine_w = w_encoder_en_convs_4_pconv_2_affine_weight; pw.affine_b = w_encoder_en_convs_4_pconv_2_affine_bias;
        pw.slope = w_encoder_en_convs_4_pconv_2_slope_weight;
        pw.C_in = 32; pw.C_out = 16; pw.groups = 2; pw.F = 33; pw.do_shuffle = 1;
        pointwise_bn_act(&pw, en3, p);

        ConvBnActCtfaSpec s = {0};
        s.conv_w = w_encoder_en_convs_4_dconv_1_weight; s.conv_b = w_encoder_en_convs_4_dconv_1_bias;
        s.bn_w = w_encoder_en_convs_4_dconv_2_weight; s.bn_b = w_encoder_en_convs_4_dconv_2_bias;
        s.bn_rm = w_encoder_en_convs_4_dconv_2_running_mean; s.bn_rv = w_encoder_en_convs_4_dconv_2_running_var;
        s.affine_w = w_encoder_en_convs_4_dconv_3_affine_weight; s.affine_b = w_encoder_en_convs_4_dconv_3_affine_bias;
        s.slope = w_encoder_en_convs_4_dconv_3_slope_weight;
        s.has_ctfa = 1;
        build_ctfa_spec(&s.ctfa, 16, 33, 32,
            w_encoder_en_convs_4_dconv_4_ta_gru_weight_ih_l0, w_encoder_en_convs_4_dconv_4_ta_gru_weight_hh_l0,
            w_encoder_en_convs_4_dconv_4_ta_gru_bias_ih_l0, w_encoder_en_convs_4_dconv_4_ta_gru_bias_hh_l0,
            w_encoder_en_convs_4_dconv_4_ta_fc_weight, w_encoder_en_convs_4_dconv_4_ta_fc_bias,
            w_encoder_en_convs_4_dconv_4_fa_gru_weight_ih_l0, w_encoder_en_convs_4_dconv_4_fa_gru_weight_hh_l0,
            w_encoder_en_convs_4_dconv_4_fa_gru_bias_ih_l0, w_encoder_en_convs_4_dconv_4_fa_gru_bias_hh_l0,
            w_encoder_en_convs_4_dconv_4_fa_gru_weight_ih_l0_reverse, w_encoder_en_convs_4_dconv_4_fa_gru_weight_hh_l0_reverse,
            w_encoder_en_convs_4_dconv_4_fa_gru_bias_ih_l0_reverse, w_encoder_en_convs_4_dconv_4_fa_gru_bias_hh_l0_reverse,
            w_encoder_en_convs_4_dconv_4_fa_fc_weight, w_encoder_en_convs_4_dconv_4_fa_fc_bias);
        s.C_in = 16; s.C_out = 16; s.KT = 1; s.KF = 5; s.stride_f = 1; s.pf = 2; s.groups = 16;
        s.F_in = 33; s.F_out = 33; s.use_deconv = 0;
        conv_bn_act_ctfa(&s, p, NULL, st->tfa[4], en4);
    }

    /* ================= DPGRNN x2 ================= */
    float dp0[16 * 33], dp1[16 * 33];
    {
        DpgrnnSpec s = {0};
        s.intra_wih1 = w_dpgrnn_0_intra_rnn_rnn1_weight_ih_l0; s.intra_whh1 = w_dpgrnn_0_intra_rnn_rnn1_weight_hh_l0;
        s.intra_bih1 = w_dpgrnn_0_intra_rnn_rnn1_bias_ih_l0; s.intra_bhh1 = w_dpgrnn_0_intra_rnn_rnn1_bias_hh_l0;
        s.intra_wih1_rev = w_dpgrnn_0_intra_rnn_rnn1_weight_ih_l0_reverse; s.intra_whh1_rev = w_dpgrnn_0_intra_rnn_rnn1_weight_hh_l0_reverse;
        s.intra_bih1_rev = w_dpgrnn_0_intra_rnn_rnn1_bias_ih_l0_reverse; s.intra_bhh1_rev = w_dpgrnn_0_intra_rnn_rnn1_bias_hh_l0_reverse;
        s.intra_wih2 = w_dpgrnn_0_intra_rnn_rnn2_weight_ih_l0; s.intra_whh2 = w_dpgrnn_0_intra_rnn_rnn2_weight_hh_l0;
        s.intra_bih2 = w_dpgrnn_0_intra_rnn_rnn2_bias_ih_l0; s.intra_bhh2 = w_dpgrnn_0_intra_rnn_rnn2_bias_hh_l0;
        s.intra_wih2_rev = w_dpgrnn_0_intra_rnn_rnn2_weight_ih_l0_reverse; s.intra_whh2_rev = w_dpgrnn_0_intra_rnn_rnn2_weight_hh_l0_reverse;
        s.intra_bih2_rev = w_dpgrnn_0_intra_rnn_rnn2_bias_ih_l0_reverse; s.intra_bhh2_rev = w_dpgrnn_0_intra_rnn_rnn2_bias_hh_l0_reverse;
        s.intra_fc_w = w_dpgrnn_0_intra_fc_weight; s.intra_fc_b = w_dpgrnn_0_intra_fc_bias;
        s.intra_ln_w = w_dpgrnn_0_intra_ln_weight; s.intra_ln_b = w_dpgrnn_0_intra_ln_bias;
        s.inter_wih1 = w_dpgrnn_0_inter_rnn_rnn1_weight_ih_l0; s.inter_whh1 = w_dpgrnn_0_inter_rnn_rnn1_weight_hh_l0;
        s.inter_bih1 = w_dpgrnn_0_inter_rnn_rnn1_bias_ih_l0; s.inter_bhh1 = w_dpgrnn_0_inter_rnn_rnn1_bias_hh_l0;
        s.inter_wih2 = w_dpgrnn_0_inter_rnn_rnn2_weight_ih_l0; s.inter_whh2 = w_dpgrnn_0_inter_rnn_rnn2_weight_hh_l0;
        s.inter_bih2 = w_dpgrnn_0_inter_rnn_rnn2_bias_ih_l0; s.inter_bhh2 = w_dpgrnn_0_inter_rnn_rnn2_bias_hh_l0;
        s.inter_fc_w = w_dpgrnn_0_inter_fc_weight; s.inter_fc_b = w_dpgrnn_0_inter_fc_bias;
        s.inter_ln_w = w_dpgrnn_0_inter_ln_weight; s.inter_ln_b = w_dpgrnn_0_inter_ln_bias;
        dpgrnn_forward(&s, en4, st->inter0, dp0);
    }
    {
        DpgrnnSpec s = {0};
        s.intra_wih1 = w_dpgrnn_1_intra_rnn_rnn1_weight_ih_l0; s.intra_whh1 = w_dpgrnn_1_intra_rnn_rnn1_weight_hh_l0;
        s.intra_bih1 = w_dpgrnn_1_intra_rnn_rnn1_bias_ih_l0; s.intra_bhh1 = w_dpgrnn_1_intra_rnn_rnn1_bias_hh_l0;
        s.intra_wih1_rev = w_dpgrnn_1_intra_rnn_rnn1_weight_ih_l0_reverse; s.intra_whh1_rev = w_dpgrnn_1_intra_rnn_rnn1_weight_hh_l0_reverse;
        s.intra_bih1_rev = w_dpgrnn_1_intra_rnn_rnn1_bias_ih_l0_reverse; s.intra_bhh1_rev = w_dpgrnn_1_intra_rnn_rnn1_bias_hh_l0_reverse;
        s.intra_wih2 = w_dpgrnn_1_intra_rnn_rnn2_weight_ih_l0; s.intra_whh2 = w_dpgrnn_1_intra_rnn_rnn2_weight_hh_l0;
        s.intra_bih2 = w_dpgrnn_1_intra_rnn_rnn2_bias_ih_l0; s.intra_bhh2 = w_dpgrnn_1_intra_rnn_rnn2_bias_hh_l0;
        s.intra_wih2_rev = w_dpgrnn_1_intra_rnn_rnn2_weight_ih_l0_reverse; s.intra_whh2_rev = w_dpgrnn_1_intra_rnn_rnn2_weight_hh_l0_reverse;
        s.intra_bih2_rev = w_dpgrnn_1_intra_rnn_rnn2_bias_ih_l0_reverse; s.intra_bhh2_rev = w_dpgrnn_1_intra_rnn_rnn2_bias_hh_l0_reverse;
        s.intra_fc_w = w_dpgrnn_1_intra_fc_weight; s.intra_fc_b = w_dpgrnn_1_intra_fc_bias;
        s.intra_ln_w = w_dpgrnn_1_intra_ln_weight; s.intra_ln_b = w_dpgrnn_1_intra_ln_bias;
        s.inter_wih1 = w_dpgrnn_1_inter_rnn_rnn1_weight_ih_l0; s.inter_whh1 = w_dpgrnn_1_inter_rnn_rnn1_weight_hh_l0;
        s.inter_bih1 = w_dpgrnn_1_inter_rnn_rnn1_bias_ih_l0; s.inter_bhh1 = w_dpgrnn_1_inter_rnn_rnn1_bias_hh_l0;
        s.inter_wih2 = w_dpgrnn_1_inter_rnn_rnn2_weight_ih_l0; s.inter_whh2 = w_dpgrnn_1_inter_rnn_rnn2_weight_hh_l0;
        s.inter_bih2 = w_dpgrnn_1_inter_rnn_rnn2_bias_ih_l0; s.inter_bhh2 = w_dpgrnn_1_inter_rnn_rnn2_bias_hh_l0;
        s.inter_fc_w = w_dpgrnn_1_inter_fc_weight; s.inter_fc_b = w_dpgrnn_1_inter_fc_bias;
        s.inter_ln_w = w_dpgrnn_1_inter_ln_weight; s.inter_ln_b = w_dpgrnn_1_inter_ln_bias;
        dpgrnn_forward(&s, dp0, st->inter1, dp1);
    }

    /* ================= DECODER ================= */
    float de0[32 * 33], de1[24 * 33], de2[24 * 33], de3[12 * 65], de4[1 * 129];

    /* de0: XDWSBlock(mirror), 16->32, groups=2, kt=1(NO CACHE),kf=5,stride_f=1,pf=2, in=dp1+en4 */
    {
        float skip[16 * 33];
        for (int i = 0; i < 16 * 33; ++i) skip[i] = dp1[i] + en4[i];
        float p[32 * 33];
        PointwiseSpec pw = {0};
        pw.conv_w = w_decoder_de_convs_0_pconv_0_weight; pw.conv_b = w_decoder_de_convs_0_pconv_0_bias;
        pw.bn_w = w_decoder_de_convs_0_pconv_1_weight; pw.bn_b = w_decoder_de_convs_0_pconv_1_bias;
        pw.bn_rm = w_decoder_de_convs_0_pconv_1_running_mean; pw.bn_rv = w_decoder_de_convs_0_pconv_1_running_var;
        pw.affine_w = w_decoder_de_convs_0_pconv_2_affine_weight; pw.affine_b = w_decoder_de_convs_0_pconv_2_affine_bias;
        pw.slope = w_decoder_de_convs_0_pconv_2_slope_weight;
        pw.C_in = 16; pw.C_out = 32; pw.groups = 2; pw.F = 33; pw.do_shuffle = 1;
        pointwise_bn_act(&pw, skip, p);

        ConvBnActCtfaSpec s = {0};
        s.conv_w = w_decoder_de_convs_0_dconv_1_weight; s.conv_b = w_decoder_de_convs_0_dconv_1_bias;
        s.bn_w = w_decoder_de_convs_0_dconv_2_weight; s.bn_b = w_decoder_de_convs_0_dconv_2_bias;
        s.bn_rm = w_decoder_de_convs_0_dconv_2_running_mean; s.bn_rv = w_decoder_de_convs_0_dconv_2_running_var;
        s.affine_w = w_decoder_de_convs_0_dconv_3_affine_weight; s.affine_b = w_decoder_de_convs_0_dconv_3_affine_bias;
        s.slope = w_decoder_de_convs_0_dconv_3_slope_weight;
        s.has_ctfa = 1;
        build_ctfa_spec(&s.ctfa, 32, 33, 64,
            w_decoder_de_convs_0_dconv_4_ta_gru_weight_ih_l0, w_decoder_de_convs_0_dconv_4_ta_gru_weight_hh_l0,
            w_decoder_de_convs_0_dconv_4_ta_gru_bias_ih_l0, w_decoder_de_convs_0_dconv_4_ta_gru_bias_hh_l0,
            w_decoder_de_convs_0_dconv_4_ta_fc_weight, w_decoder_de_convs_0_dconv_4_ta_fc_bias,
            w_decoder_de_convs_0_dconv_4_fa_gru_weight_ih_l0, w_decoder_de_convs_0_dconv_4_fa_gru_weight_hh_l0,
            w_decoder_de_convs_0_dconv_4_fa_gru_bias_ih_l0, w_decoder_de_convs_0_dconv_4_fa_gru_bias_hh_l0,
            w_decoder_de_convs_0_dconv_4_fa_gru_weight_ih_l0_reverse, w_decoder_de_convs_0_dconv_4_fa_gru_weight_hh_l0_reverse,
            w_decoder_de_convs_0_dconv_4_fa_gru_bias_ih_l0_reverse, w_decoder_de_convs_0_dconv_4_fa_gru_bias_hh_l0_reverse,
            w_decoder_de_convs_0_dconv_4_fa_fc_weight, w_decoder_de_convs_0_dconv_4_fa_fc_bias);
        /* kt=1 -> "use_deconv" degenerates identically to Conv2d math (no time extent to
         * transpose); the reference code still constructs an nn.ConvTranspose2d module here,
         * but with KT=1 the transpose-conv gather formula and the plain conv formula are
         * algebraically identical (pt=KT-1=0, no time-axis effect) -- verified this holds by
         * construction, not re-derived per-block. groups=32 (depthwise on the OUTPUT channels
         * per XDWSBlock's `conv_module(out_channels,out_channels,...,groups=out_channels)`). */
        s.C_in = 32; s.C_out = 32; s.KT = 1; s.KF = 5; s.stride_f = 1; s.pf = 2; s.groups = 32;
        s.F_in = 33; s.F_out = 33; s.use_deconv = 1;
        conv_bn_act_ctfa(&s, p, NULL, st->tfa[5], de0);
    }

    /* de1: XMBBlocks(mirror), 32->24, groups=2, kt=1(NO CACHE),kf=5,stride_f=1,pf=2, in=de0+en3 */
    {
        float skip[32 * 33];
        for (int i = 0; i < 32 * 33; ++i) skip[i] = de0[i] + en3[i];
        float p1[24 * 33];
        PointwiseSpec pw = {0};
        pw.conv_w = w_decoder_de_convs_1_pconv1_0_weight; pw.conv_b = w_decoder_de_convs_1_pconv1_0_bias;
        pw.bn_w = w_decoder_de_convs_1_pconv1_1_weight; pw.bn_b = w_decoder_de_convs_1_pconv1_1_bias;
        pw.bn_rm = w_decoder_de_convs_1_pconv1_1_running_mean; pw.bn_rv = w_decoder_de_convs_1_pconv1_1_running_var;
        pw.affine_w = w_decoder_de_convs_1_pconv1_2_affine_weight; pw.affine_b = w_decoder_de_convs_1_pconv1_2_affine_bias;
        pw.slope = w_decoder_de_convs_1_pconv1_2_slope_weight;
        pw.C_in = 32; pw.C_out = 24; pw.groups = 2; pw.F = 33; pw.do_shuffle = 1;
        pointwise_bn_act(&pw, skip, p1);

        float mid[24 * 33];
        ConvBnActCtfaSpec sd = {0};
        sd.conv_w = w_decoder_de_convs_1_dconv_1_weight; sd.conv_b = w_decoder_de_convs_1_dconv_1_bias;
        sd.bn_w = w_decoder_de_convs_1_dconv_2_weight; sd.bn_b = w_decoder_de_convs_1_dconv_2_bias;
        sd.bn_rm = w_decoder_de_convs_1_dconv_2_running_mean; sd.bn_rv = w_decoder_de_convs_1_dconv_2_running_var;
        sd.affine_w = w_decoder_de_convs_1_dconv_3_affine_weight; sd.affine_b = w_decoder_de_convs_1_dconv_3_affine_bias;
        sd.slope = w_decoder_de_convs_1_dconv_3_slope_weight;
        sd.has_ctfa = 0;
        sd.C_in = 24; sd.C_out = 24; sd.KT = 1; sd.KF = 5; sd.stride_f = 1; sd.pf = 2; sd.groups = 24;
        sd.F_in = 33; sd.F_out = 33; sd.use_deconv = 1;
        conv_bn_act_ctfa(&sd, p1, NULL, NULL, mid);

        float p2[24 * 33];
        conv2d_causal_1frame(mid, 24, 1, 33, w_decoder_de_convs_1_pconv2_0_weight,
                             w_decoder_de_convs_1_pconv2_0_bias, 24, 1, 1, 0, 2, p2, 33);
        batchnorm_apply(p2, 24, 33, w_decoder_de_convs_1_pconv2_1_weight, w_decoder_de_convs_1_pconv2_1_bias,
                        w_decoder_de_convs_1_pconv2_1_running_mean, w_decoder_de_convs_1_pconv2_1_running_var, 1e-5f);
        CtfaSpec ctfa;
        build_ctfa_spec(&ctfa, 24, 33, 48,
            w_decoder_de_convs_1_pconv2_2_ta_gru_weight_ih_l0, w_decoder_de_convs_1_pconv2_2_ta_gru_weight_hh_l0,
            w_decoder_de_convs_1_pconv2_2_ta_gru_bias_ih_l0, w_decoder_de_convs_1_pconv2_2_ta_gru_bias_hh_l0,
            w_decoder_de_convs_1_pconv2_2_ta_fc_weight, w_decoder_de_convs_1_pconv2_2_ta_fc_bias,
            w_decoder_de_convs_1_pconv2_2_fa_gru_weight_ih_l0, w_decoder_de_convs_1_pconv2_2_fa_gru_weight_hh_l0,
            w_decoder_de_convs_1_pconv2_2_fa_gru_bias_ih_l0, w_decoder_de_convs_1_pconv2_2_fa_gru_bias_hh_l0,
            w_decoder_de_convs_1_pconv2_2_fa_gru_weight_ih_l0_reverse, w_decoder_de_convs_1_pconv2_2_fa_gru_weight_hh_l0_reverse,
            w_decoder_de_convs_1_pconv2_2_fa_gru_bias_ih_l0_reverse, w_decoder_de_convs_1_pconv2_2_fa_gru_bias_hh_l0_reverse,
            w_decoder_de_convs_1_pconv2_2_fa_fc_weight, w_decoder_de_convs_1_pconv2_2_fa_fc_bias);
        ctfa_apply(&ctfa, p2, st->tfa[6]);
        shuffle_apply(p2, 12, 33, de1);
    }

    /* de2: XDWSBlock(mirror), 24->24, groups=2, kt=2,kf=3,stride_f=1,pf=1 (HAS CACHE), in=de1+en2 */
    {
        float skip[24 * 33];
        for (int i = 0; i < 24 * 33; ++i) skip[i] = de1[i] + en2[i];
        float p[24 * 33];
        PointwiseSpec pw = {0};
        pw.conv_w = w_decoder_de_convs_2_pconv_0_weight; pw.conv_b = w_decoder_de_convs_2_pconv_0_bias;
        pw.bn_w = w_decoder_de_convs_2_pconv_1_weight; pw.bn_b = w_decoder_de_convs_2_pconv_1_bias;
        pw.bn_rm = w_decoder_de_convs_2_pconv_1_running_mean; pw.bn_rv = w_decoder_de_convs_2_pconv_1_running_var;
        pw.affine_w = w_decoder_de_convs_2_pconv_2_affine_weight; pw.affine_b = w_decoder_de_convs_2_pconv_2_affine_bias;
        pw.slope = w_decoder_de_convs_2_pconv_2_slope_weight;
        pw.C_in = 24; pw.C_out = 24; pw.groups = 2; pw.F = 33; pw.do_shuffle = 1;
        pointwise_bn_act(&pw, skip, p);

        ConvBnActCtfaSpec s = {0};
        s.conv_w = w_decoder_de_convs_2_dconv_1_weight; s.conv_b = w_decoder_de_convs_2_dconv_1_bias;
        s.bn_w = w_decoder_de_convs_2_dconv_2_weight; s.bn_b = w_decoder_de_convs_2_dconv_2_bias;
        s.bn_rm = w_decoder_de_convs_2_dconv_2_running_mean; s.bn_rv = w_decoder_de_convs_2_dconv_2_running_var;
        s.affine_w = w_decoder_de_convs_2_dconv_3_affine_weight; s.affine_b = w_decoder_de_convs_2_dconv_3_affine_bias;
        s.slope = w_decoder_de_convs_2_dconv_3_slope_weight;
        s.has_ctfa = 1;
        build_ctfa_spec(&s.ctfa, 24, 33, 48,
            w_decoder_de_convs_2_dconv_4_ta_gru_weight_ih_l0, w_decoder_de_convs_2_dconv_4_ta_gru_weight_hh_l0,
            w_decoder_de_convs_2_dconv_4_ta_gru_bias_ih_l0, w_decoder_de_convs_2_dconv_4_ta_gru_bias_hh_l0,
            w_decoder_de_convs_2_dconv_4_ta_fc_weight, w_decoder_de_convs_2_dconv_4_ta_fc_bias,
            w_decoder_de_convs_2_dconv_4_fa_gru_weight_ih_l0, w_decoder_de_convs_2_dconv_4_fa_gru_weight_hh_l0,
            w_decoder_de_convs_2_dconv_4_fa_gru_bias_ih_l0, w_decoder_de_convs_2_dconv_4_fa_gru_bias_hh_l0,
            w_decoder_de_convs_2_dconv_4_fa_gru_weight_ih_l0_reverse, w_decoder_de_convs_2_dconv_4_fa_gru_weight_hh_l0_reverse,
            w_decoder_de_convs_2_dconv_4_fa_gru_bias_ih_l0_reverse, w_decoder_de_convs_2_dconv_4_fa_gru_bias_hh_l0_reverse,
            w_decoder_de_convs_2_dconv_4_fa_fc_weight, w_decoder_de_convs_2_dconv_4_fa_fc_bias);
        s.C_in = 24; s.C_out = 24; s.KT = 2; s.KF = 3; s.stride_f = 1; s.pf = 1; s.groups = 24;
        s.F_in = 33; s.F_out = 33; s.use_deconv = 1;
        conv_bn_act_ctfa(&s, p, st->cc_d2, st->tfa[7], de2);
    }

    /* de3: XMBBlocks(mirror), 24->12, groups=2, kt=2,kf=3,stride_f=2,pf=1 (HAS CACHE), width 33->65, in=de2+en1 */
    {
        float skip[24 * 33];
        for (int i = 0; i < 24 * 33; ++i) skip[i] = de2[i] + en1[i];
        float p1[12 * 33];
        PointwiseSpec pw = {0};
        pw.conv_w = w_decoder_de_convs_3_pconv1_0_weight; pw.conv_b = w_decoder_de_convs_3_pconv1_0_bias;
        pw.bn_w = w_decoder_de_convs_3_pconv1_1_weight; pw.bn_b = w_decoder_de_convs_3_pconv1_1_bias;
        pw.bn_rm = w_decoder_de_convs_3_pconv1_1_running_mean; pw.bn_rv = w_decoder_de_convs_3_pconv1_1_running_var;
        pw.affine_w = w_decoder_de_convs_3_pconv1_2_affine_weight; pw.affine_b = w_decoder_de_convs_3_pconv1_2_affine_bias;
        pw.slope = w_decoder_de_convs_3_pconv1_2_slope_weight;
        pw.C_in = 24; pw.C_out = 12; pw.groups = 2; pw.F = 33; pw.do_shuffle = 1;
        pointwise_bn_act(&pw, skip, p1);

        float mid[12 * 65];
        ConvBnActCtfaSpec sd = {0};
        sd.conv_w = w_decoder_de_convs_3_dconv_1_weight; sd.conv_b = w_decoder_de_convs_3_dconv_1_bias;
        sd.bn_w = w_decoder_de_convs_3_dconv_2_weight; sd.bn_b = w_decoder_de_convs_3_dconv_2_bias;
        sd.bn_rm = w_decoder_de_convs_3_dconv_2_running_mean; sd.bn_rv = w_decoder_de_convs_3_dconv_2_running_var;
        sd.affine_w = w_decoder_de_convs_3_dconv_3_affine_weight; sd.affine_b = w_decoder_de_convs_3_dconv_3_affine_bias;
        sd.slope = w_decoder_de_convs_3_dconv_3_slope_weight;
        sd.has_ctfa = 0;
        sd.C_in = 12; sd.C_out = 12; sd.KT = 2; sd.KF = 3; sd.stride_f = 2; sd.pf = 1; sd.groups = 12;
        sd.F_in = 33; sd.F_out = 65; sd.use_deconv = 1;
        conv_bn_act_ctfa(&sd, p1, st->cc_d3, NULL, mid);

        float p2[12 * 65];
        conv2d_causal_1frame(mid, 12, 1, 65, w_decoder_de_convs_3_pconv2_0_weight,
                             w_decoder_de_convs_3_pconv2_0_bias, 12, 1, 1, 0, 2, p2, 65);
        batchnorm_apply(p2, 12, 65, w_decoder_de_convs_3_pconv2_1_weight, w_decoder_de_convs_3_pconv2_1_bias,
                        w_decoder_de_convs_3_pconv2_1_running_mean, w_decoder_de_convs_3_pconv2_1_running_var, 1e-5f);
        CtfaSpec ctfa;
        build_ctfa_spec(&ctfa, 12, 65, 24,
            w_decoder_de_convs_3_pconv2_2_ta_gru_weight_ih_l0, w_decoder_de_convs_3_pconv2_2_ta_gru_weight_hh_l0,
            w_decoder_de_convs_3_pconv2_2_ta_gru_bias_ih_l0, w_decoder_de_convs_3_pconv2_2_ta_gru_bias_hh_l0,
            w_decoder_de_convs_3_pconv2_2_ta_fc_weight, w_decoder_de_convs_3_pconv2_2_ta_fc_bias,
            w_decoder_de_convs_3_pconv2_2_fa_gru_weight_ih_l0, w_decoder_de_convs_3_pconv2_2_fa_gru_weight_hh_l0,
            w_decoder_de_convs_3_pconv2_2_fa_gru_bias_ih_l0, w_decoder_de_convs_3_pconv2_2_fa_gru_bias_hh_l0,
            w_decoder_de_convs_3_pconv2_2_fa_gru_weight_ih_l0_reverse, w_decoder_de_convs_3_pconv2_2_fa_gru_weight_hh_l0_reverse,
            w_decoder_de_convs_3_pconv2_2_fa_gru_bias_ih_l0_reverse, w_decoder_de_convs_3_pconv2_2_fa_gru_bias_hh_l0_reverse,
            w_decoder_de_convs_3_pconv2_2_fa_fc_weight, w_decoder_de_convs_3_pconv2_2_fa_fc_bias);
        ctfa_apply(&ctfa, p2, st->tfa[8]);
        shuffle_apply(p2, 6, 65, de3);
    }

    /* de4: XConvBlock(mirror, is_last), 12->1, kt=3,kf=3,stride_f=2,pf=1 (HAS CACHE), width 65->129, in=de3+en0 */
    {
        float skip[12 * 65];
        for (int i = 0; i < 12 * 65; ++i) skip[i] = de3[i] + en0[i];
        ConvBnActCtfaSpec s = {0};
        s.conv_w = w_decoder_de_convs_4_ops_1_weight; s.conv_b = w_decoder_de_convs_4_ops_1_bias;
        s.bn_w = w_decoder_de_convs_4_ops_2_weight; s.bn_b = w_decoder_de_convs_4_ops_2_bias;
        s.bn_rm = w_decoder_de_convs_4_ops_2_running_mean; s.bn_rv = w_decoder_de_convs_4_ops_2_running_var;
        s.affine_w = NULL; /* is_last */
        s.has_ctfa = 1;
        build_ctfa_spec(&s.ctfa, 1, 129, 2,
            w_decoder_de_convs_4_ops_4_ta_gru_weight_ih_l0, w_decoder_de_convs_4_ops_4_ta_gru_weight_hh_l0,
            w_decoder_de_convs_4_ops_4_ta_gru_bias_ih_l0, w_decoder_de_convs_4_ops_4_ta_gru_bias_hh_l0,
            w_decoder_de_convs_4_ops_4_ta_fc_weight, w_decoder_de_convs_4_ops_4_ta_fc_bias,
            w_decoder_de_convs_4_ops_4_fa_gru_weight_ih_l0, w_decoder_de_convs_4_ops_4_fa_gru_weight_hh_l0,
            w_decoder_de_convs_4_ops_4_fa_gru_bias_ih_l0, w_decoder_de_convs_4_ops_4_fa_gru_bias_hh_l0,
            w_decoder_de_convs_4_ops_4_fa_gru_weight_ih_l0_reverse, w_decoder_de_convs_4_ops_4_fa_gru_weight_hh_l0_reverse,
            w_decoder_de_convs_4_ops_4_fa_gru_bias_ih_l0_reverse, w_decoder_de_convs_4_ops_4_fa_gru_bias_hh_l0_reverse,
            w_decoder_de_convs_4_ops_4_fa_fc_weight, w_decoder_de_convs_4_ops_4_fa_fc_bias);
        s.C_in = 12; s.C_out = 1; s.KT = 3; s.KF = 3; s.stride_f = 2; s.pf = 1; s.groups = 1;
        s.F_in = 65; s.F_out = 129; s.use_deconv = 1;
        conv_bn_act_ctfa(&s, skip, st->cc_d4, st->tfa[9], de4);
    }

    /* ---- sigmoid mask, ERB.bs (129->257), complex masking ---- */
    float m_feat[ERB_TOTAL];
    for (int i = 0; i < ERB_TOTAL; ++i) m_feat[i] = sigmoidf_(de4[i]);

    float mask257[FREQ];
    for (int i = 0; i < ERB_LOW; ++i) mask257[i] = m_feat[i];
    for (int o = 0; o < (FREQ - ERB_LOW); ++o) {
        float acc = 0.0f;
        for (int i = 0; i < ERB_HIGH; ++i)
            acc += m_feat[ERB_LOW + i] * w_erb_ierb_fc_weight[o * ERB_HIGH + i];
        mask257[ERB_LOW + o] = acc;
    }

    for (int f = 0; f < FREQ; ++f) {
        enh_spec[f * 2 + 0] = mix_spec[f * 2 + 0] * mask257[f];
        enh_spec[f * 2 + 1] = mix_spec[f * 2 + 1] * mask257[f];
    }
}

/* =====================================================================================
 * Public C API: STFT/ISTFT streaming wrapper around ulunas_forward_frame.
 *
 * Design choice (documented, not accidental): the analysis ring buffer is zero-initialized
 * (causal sliding window), NOT reflect-padded like the offline torch.stft(center=True)
 * reference used for the batch/Android/ONNX parity tests. Reflect-padding requires
 * "future" samples relative to stream start, which a true incremental real-time API
 * (arbitrary-length live microphone input, unknown total length) cannot provide without
 * adding artificial lookahead. Reflect-padding only ever affects the FIRST and LAST STFT
 * frame of a signal (every interior frame's 512-sample window is identical regardless of
 * padding convention, since padding only extends the edges) -- so this choice costs a
 * small, bounded startup transient during the first ~1 window (32ms) of a stream, with
 * zero effect on steady-state frames. This is standard practice for real-time streaming
 * SE systems and is measured (not hidden) in the parity harness: reported separately as
 * "first-window transient error" vs "steady-state error".
 * ===================================================================================== */

size_t ulunas_state_size_bytes(void) { return sizeof(UlunasState); }

int ulunas_init(UlunasState **out_state) {
    UlunasState *st = (UlunasState *)calloc(1, sizeof(UlunasState));
    if (!st) { *out_state = NULL; return -1; }
    for (int i = 0; i < ULUNAS_WIN; ++i)
        st->hann[i] = 0.5f - 0.5f * cosf((float)(2.0 * M_PI * i) / (float)ULUNAS_WIN);
    *out_state = st;
    return 0;
}

void ulunas_reset(UlunasState *state) {
    if (!state) return;
    float hann_save[ULUNAS_WIN];
    memcpy(hann_save, state->hann, sizeof(hann_save));
    memset(state, 0, sizeof(UlunasState));
    memcpy(state->hann, hann_save, sizeof(hann_save));
}

void ulunas_destroy(UlunasState *state) {
    free(state);
}

int ulunas_process_hop(UlunasState *state, const float *input_pcm, float *output_pcm) {
    /* 1) slide analysis buffer, append new hop */
    memmove(state->analysis_buf, state->analysis_buf + ULUNAS_HOP,
           (ULUNAS_WIN - ULUNAS_HOP) * sizeof(float));
    memcpy(state->analysis_buf + (ULUNAS_WIN - ULUNAS_HOP), input_pcm, ULUNAS_HOP * sizeof(float));

    /* 2) window + rfft */
    float windowed[ULUNAS_WIN];
    for (int i = 0; i < ULUNAS_WIN; ++i) windowed[i] = state->analysis_buf[i] * state->hann[i];
    std::vector<cplx> spec = rfft(std::vector<float>(windowed, windowed + ULUNAS_WIN));
    float mix_spec[FREQ * 2];
    for (int f = 0; f < FREQ; ++f) { mix_spec[f * 2] = spec[f].real(); mix_spec[f * 2 + 1] = spec[f].imag(); }

    /* 3) model forward */
    float enh_spec[FREQ * 2];
    ulunas_process_frame_spec(state, mix_spec, enh_spec);

    /* 4) irfft + synthesis window + OLA accumulate */
    std::vector<cplx> enh_c(FREQ);
    for (int f = 0; f < FREQ; ++f) enh_c[f] = cplx(enh_spec[f * 2], enh_spec[f * 2 + 1]);
    std::vector<float> frame_time = irfft(enh_c, ULUNAS_WIN);

    for (int i = 0; i < ULUNAS_WIN; ++i) {
        state->ola_buf[i] += frame_time[i] * state->hann[i];
        state->win_sum_buf[i] += state->hann[i] * state->hann[i];
    }

    /* 5) emit the first hop (fully formed: no future frame will touch it again) */
    for (int i = 0; i < ULUNAS_HOP; ++i) {
        float norm = state->win_sum_buf[i] > 1e-8f ? state->win_sum_buf[i] : 1e-8f;
        output_pcm[i] = state->ola_buf[i] / norm;
    }

    /* 6) shift OLA/win_sum accumulators left by one hop */
    memmove(state->ola_buf, state->ola_buf + ULUNAS_HOP, (ULUNAS_WIN - ULUNAS_HOP) * sizeof(float));
    memset(state->ola_buf + (ULUNAS_WIN - ULUNAS_HOP), 0, ULUNAS_HOP * sizeof(float));
    memmove(state->win_sum_buf, state->win_sum_buf + ULUNAS_HOP, (ULUNAS_WIN - ULUNAS_HOP) * sizeof(float));
    memset(state->win_sum_buf + (ULUNAS_WIN - ULUNAS_HOP), 0, ULUNAS_HOP * sizeof(float));

    return 0; /* no runtime failure mode: state is always valid once ulunas_init succeeded */
}
