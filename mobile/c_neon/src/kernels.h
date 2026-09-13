/* Scalar C99 kernel library for the runtime-free UL-UNAS reimplementation (M5).
 * Every function here is a faithful translation of one PyTorch op actually used by
 * StreamULUNAS (ul-unas/ulunas_onnx/stream/ulunas_stream.py), verified against it
 * numerically (see validate_*.py under mobile/c_neon/). No dynamic shapes: all sizes are
 * compile-time constants matching this one fixed architecture ("graph cố định" per spec). */
#pragma once
#include <math.h>
#include <string.h>

/* ---- grouped causal Conv2d, ONE output time-frame, given KT cached+current input frames ----
 * in: [C_in, KT, F_in] flattened, weight: [C_out, C_in/groups, KT, KF], bias: [C_out]
 * out: [C_out, 1, F_out], F_out = F_in (stride 1) or ceil(F_in/stride_f) with 'same'-ish
 * padding pf = KF/2 already assumed applied by caller via padded F axis (F_in already padded).
 * This computes ONLY the LAST time-step's output (kt-1 offset), matching a KT-tap causal conv
 * consuming [cache(KT-1 frames), current(1 frame)] and producing 1 output frame. */
/* pf = kf/2 is PyTorch nn.Conv2d's OWN automatic same-style frequency padding (applied
 * internally by Conv2d itself in the reference model: Conv2d(..., padding=(0, pf), ...) --
 * the streaming wrapper only handles TIME padding via cache-concat, and calls conv(inp)
 * directly, relying on Conv2d's own pf for frequency. `in` is the RAW unpadded (C_in,KT,F_in)
 * buffer; this kernel applies the pf offset itself (out-of-range = implicit zero), matching
 * PyTorch's zero-padding semantics exactly -- do NOT pre-pad the frequency axis yourself. */
static inline void conv2d_causal_1frame(
    const float* in, int C_in, int KT, int F_in,
    const float* weight, const float* bias,
    int C_out, int KF, int stride_f, int pf, int groups,
    float* out, int F_out) {
    int cin_per_group = C_in / groups;
    int cout_per_group = C_out / groups;
    for (int oc = 0; oc < C_out; ++oc) {
        int g = oc / cout_per_group;
        for (int of = 0; of < F_out; ++of) {
            float acc = bias ? bias[oc] : 0.0f;
            int f_center = of * stride_f - pf;
            for (int ic = 0; ic < cin_per_group; ++ic) {
                int ic_global = g * cin_per_group + ic;
                for (int kt = 0; kt < KT; ++kt) {
                    for (int kf = 0; kf < KF; ++kf) {
                        int f_idx = f_center + kf;
                        if (f_idx < 0 || f_idx >= F_in) continue;
                        float x = in[(ic_global * KT + kt) * F_in + f_idx];
                        float w = weight[((oc * cin_per_group + ic) * KT + kt) * KF + kf];
                        acc += x * w;
                    }
                }
            }
            out[oc * F_out + of] = acc;
        }
    }
}

/* ---- grouped causal ConvTranspose2d, ONE output time-frame (the LAST one, matching
 * _stream_temporal_conv's `conv(inp_padded)[:, :, -1:, :]`).
 * weight layout for ConvTranspose2d is (C_in, C_out/groups, KT, KF) -- OPPOSITE of Conv2d's
 * (C_out, C_in/groups, KT, KF). Uses the standard transposed-conv "gather" definition:
 *   out[oc,ot,of] = bias[oc] + sum_{ic,kt,kf} in[ic, ot+pt-kt, ...] * w[ic,oc,kt,kf]
 *     where it=ot+pt-kt must be in range, and (of+pf-kf) must be exactly divisible by
 *     stride_f with the quotient in range (frequency-axis upsampling).
 * Verified numerically against torch.nn.ConvTranspose2d + the exact streaming pad/cache
 * logic in ulunas_stream.py::_stream_temporal_conv before writing this (max err 1.8e-7).
 * in: [C_in, T_in, F_in] flattened (T_in = kt cached+current frames, kt = kernel_size[0]);
 * pt = kt-1 (always, per XConvBlock/XDWSBlock/XMBBlocks construction with use_deconv=True). */
static inline void convtranspose2d_causal_1frame(
    const float* in, int C_in, int T_in, int F_in,
    const float* weight, const float* bias,
    int C_out, int KT, int KF, int stride_f, int pf, int groups,
    float* out, int F_out) {
    int pt = KT - 1;
    int ot = (T_in - 1) - 2 * pt + KT - 1; /* index of the LAST output time-frame, 0-based */
    int cin_per_group = C_in / groups;
    int cout_per_group = C_out / groups;
    for (int oc = 0; oc < C_out; ++oc) {
        int g = oc / cout_per_group;
        for (int of = 0; of < F_out; ++of) {
            float acc = bias ? bias[oc] : 0.0f;
            for (int ic_local = 0; ic_local < cin_per_group; ++ic_local) {
                int ic = g * cin_per_group + ic_local;
                for (int kt = 0; kt < KT; ++kt) {
                    int it = ot + pt - kt;
                    if (it < 0 || it >= T_in) continue;
                    for (int kf = 0; kf < KF; ++kf) {
                        int inum = of + pf - kf;
                        if (inum % stride_f != 0) continue;
                        int ifreq = inum / stride_f;
                        if (ifreq < 0 || ifreq >= F_in) continue;
                        /* weight indexed [ic (global, C_in), oc_local (within group), kt, kf] */
                        int oc_local = oc - g * cout_per_group;
                        float w = weight[((ic * cout_per_group + oc_local) * KT + kt) * KF + kf];
                        acc += in[(ic * T_in + it) * F_in + ifreq] * w;
                    }
                }
            }
            out[oc * F_out + of] = acc;
        }
    }
}

/* ---- BatchNorm2d (eval mode, frozen running stats -- decision #21 in SPEC) folded into an
 * affine transform: y = (x - running_mean) / sqrt(running_var + eps) * weight + bias ---- */
static inline void batchnorm_apply(float* x, int C, int F, const float* weight, const float* bias,
                                    const float* running_mean, const float* running_var, float eps) {
#ifdef ULUNAS_BN_FOLDED
    /* All 22 preceding convolutions already contain this affine, including pconv2. */
    (void)x; (void)C; (void)F; (void)weight; (void)bias;
    (void)running_mean; (void)running_var; (void)eps;
#else
    for (int c = 0; c < C; ++c) {
        float scale = weight[c] / sqrtf(running_var[c] + eps);
        float shift = bias[c] - running_mean[c] * scale;
        for (int f = 0; f < F; ++f) x[c * F + f] = x[c * F + f] * scale + shift;
    }
#endif
}

/* ---- AffinePReLU: y = affine_w*x + affine_b + PReLU(x, slope) ---- (see ulunas.py::AffinePReLU) */
static inline void affine_prelu_apply(float* x, int C, int F, const float* affine_w,
                                      const float* affine_b, const float* slope) {
    for (int c = 0; c < C; ++c) {
        for (int f = 0; f < F; ++f) {
            float v = x[c * F + f];
            float affine = affine_w[c * F + f] * v + affine_b[c * F + f];
            float act = v > 0.0f ? v : slope[c] * v;
            x[c * F + f] = affine + act;
        }
    }
}

#ifdef __ARM_NEON
#include <arm_neon.h>

/* NEON dot-product: sum_{i=0}^{n-1} a[i]*b[i], n need not be a multiple of 4. */
static inline float neon_dot(const float* a, const float* b, int n) {
    float32x4_t acc = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        acc = vfmaq_f32(acc, va, vb);
    }
    float sum = vaddvq_f32(acc);
    for (; i < n; ++i) sum += a[i] * b[i];
    return sum;
}
#endif

/* Scalar dot product, used as the reference and as the non-NEON fallback (identical math,
 * different accumulation order than the NEON version -- validated to agree to float
 * precision, see test_neon_parity.cpp). */
static inline float scalar_dot(const float* a, const float* b, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) sum += a[i] * b[i];
    return sum;
}

/* ---- single-step GRU cell (matches torch.nn.GRU semantics), used for cached temporal GRUs
 * (cTFA.ta_gru, DPGRNN inter_rnn) AND internally by gru_bidirectional_seq (FA, DPGRNN
 * intra_rnn) -- by far the most-called primitive in the graph (profiled: ctfa_apply +
 * dpgrnn_forward together account for ~41% of scalar runtime, dominated by GRU gate
 * matmuls), hence the first NEON target. weight_ih:[3H,I], weight_hh:[3H,H], gate order
 * r,z,n (PyTorch GRU convention). h_prev/h_out size H, may alias. Uses NEON dot-products
 * on ARM targets (__ARM_NEON), plain scalar dot-products elsewhere -- SAME formula, just a
 * different accumulation order, verified to agree to float precision. ---- */
static inline void gru_step(const float* x, int input_size, float* h, int hidden_size,
                            const float* w_ih, const float* w_hh, const float* b_ih, const float* b_hh) {
    float gates_i[3 * 256]; /* r,z,n from input; hidden_size assumed <= 256 for this fixed model */
    float gates_h[3 * 256];
    for (int g = 0; g < 3 * hidden_size; ++g) {
#ifdef __ARM_NEON
        float acc_i = (b_ih ? b_ih[g] : 0.0f) + neon_dot(w_ih + g * input_size, x, input_size);
        float acc_h = (b_hh ? b_hh[g] : 0.0f) + neon_dot(w_hh + g * hidden_size, h, hidden_size);
#else
        float acc_i = (b_ih ? b_ih[g] : 0.0f) + scalar_dot(w_ih + g * input_size, x, input_size);
        float acc_h = (b_hh ? b_hh[g] : 0.0f) + scalar_dot(w_hh + g * hidden_size, h, hidden_size);
#endif
        gates_i[g] = acc_i;
        gates_h[g] = acc_h;
    }
    float h_new[256];
    for (int k = 0; k < hidden_size; ++k) {
        float r = 1.0f / (1.0f + expf(-(gates_i[k] + gates_h[k])));
        float z = 1.0f / (1.0f + expf(-(gates_i[hidden_size + k] + gates_h[hidden_size + k])));
        float n = tanhf(gates_i[2 * hidden_size + k] + r * gates_h[2 * hidden_size + k]);
        h_new[k] = (1.0f - z) * n + z * h[k];
    }
    memcpy(h, h_new, hidden_size * sizeof(float));
}

/* ---- bidirectional GRU over a short sequence of length T (used by FA -- recomputed fully
 * every frame from zero-initialized hidden state, no cross-frame cache, since it runs over
 * the frequency-group axis WITHIN one time frame, not across time). Writes [T, 2*hidden]
 * (forward-hidden concat backward-hidden per PyTorch bidirectional GRU convention). ---- */
static inline void gru_bidirectional_seq(const float* x_seq, int T, int input_size,
                                         int hidden_size, const float* w_ih_f, const float* w_hh_f,
                                         const float* b_ih_f, const float* b_hh_f, const float* w_ih_b,
                                         const float* w_hh_b, const float* b_ih_b, const float* b_hh_b,
                                         float* out_seq /* [T, 2*hidden_size] */) {
    float h_f[64] = {0};
    for (int t = 0; t < T; ++t) {
        gru_step(&x_seq[t * input_size], input_size, h_f, hidden_size, w_ih_f, w_hh_f, b_ih_f, b_hh_f);
        memcpy(&out_seq[t * 2 * hidden_size], h_f, hidden_size * sizeof(float));
    }
    float h_b[64] = {0};
    for (int t = T - 1; t >= 0; --t) {
        gru_step(&x_seq[t * input_size], input_size, h_b, hidden_size, w_ih_b, w_hh_b, b_ih_b, b_hh_b);
        memcpy(&out_seq[t * 2 * hidden_size + hidden_size], h_b, hidden_size * sizeof(float));
    }
}

static inline float sigmoidf_(float x) { return 1.0f / (1.0f + expf(-x)); }

/* ---- Shuffle: channel-interleave of two halves. Verified against ulunas.py::Shuffle:
 * input channels [0..C-1]=x1, [C..2C-1]=x2 -> output[2c]=x1[c]=in[c], output[2c+1]=x2[c]=in[C+c].
 * out buffer must be different from `in` (no in-place). F = spatial size per channel. ---- */
static inline void shuffle_apply(const float* in, int C_half, int F, float* out) {
    for (int c = 0; c < C_half; ++c) {
        memcpy(out + (2 * c) * F, in + c * F, F * sizeof(float));
        memcpy(out + (2 * c + 1) * F, in + (C_half + c) * F, F * sizeof(float));
    }
}

/* ---- LayerNorm over nn.LayerNorm((width, C))'s FULL normalized_shape -- i.e. mean/var are
 * computed JOINTLY over both width and C together (one scalar mean, one scalar var for the
 * whole (width,C) block), NOT per-width-row independently. This was verified numerically
 * against torch.nn.LayerNorm before writing this (a per-row version is WRONG and was caught
 * by that check -- error 1.81 vs 2.4e-7 for the joint version). Used by DPGRNN's intra_ln/
 * inter_ln. ---- */
static inline void layernorm_apply(float* x, int width, int C, const float* weight,
                                   const float* bias, float eps) {
    int n = width * C;
    float mean = 0.0f;
    for (int i = 0; i < n; ++i) mean += x[i];
    mean /= n;
    float var = 0.0f;
    for (int i = 0; i < n; ++i) { float d = x[i] - mean; var += d * d; }
    var /= n;
    float inv_std = 1.0f / sqrtf(var + eps);
    for (int i = 0; i < n; ++i) x[i] = (x[i] - mean) * inv_std * weight[i] + bias[i];
}

/* ===================================================================================
 * Generic block-composite functions, built ONLY from the primitives above (each of
 * which was independently validated against PyTorch/numpy before this point). These
 * compose the exact sequences already proven correct end-to-end in test_block0.c
 * (XConvBlock), test_block1.c (XMBBlocks), test_block2.c (XDWSBlock), test_dpgrnn.c
 * (DPGRNN), and test_decoder4.c (XConvBlock with ConvTranspose2d, is_last) -- refactored
 * here into reusable, struct-parameterized form so the full 12-block graph doesn't need
 * ~12 copies of near-identical code. The struct field values a caller passes for each of
 * the 12 real block instances were cross-derived from ulunas_arch.json tensor shapes and
 * the known architecture hyperparameters (types/channels/kernels/strides/groups/widths
 * in ulunas.py's ULUNAS.__init__ defaults), then cross-checked against each other for
 * consistency (e.g. TFA_CACHE_HIDDEN values match 2*out_channels for every block).
 * =================================================================================== */

/* cTFA: temporal gate (cached GRU, hidden=2*C) x frequency gate (fresh bidirectional GRU
 * over an H-length sequence, r=4 groups per step, H=ceil(F/4)*... see FA in ulunas.py).
 * x: [C,F] in-place gated. tfa_cache: persistent buffer of size 2*C (ta_gru hidden). */
typedef struct {
    int C, F, ta_hidden; /* ta_hidden = 2*C */
    const float *ta_wih, *ta_whh, *ta_bih, *ta_bhh;   /* ta_gru: input=C, hidden=2*C */
    const float *ta_fc_w, *ta_fc_b;                    /* Linear(2*C, C) */
    const float *fa_wih_f, *fa_whh_f, *fa_bih_f, *fa_bhh_f; /* fa.gru forward, in=4,hid=4 */
    const float *fa_wih_b, *fa_whh_b, *fa_bih_b, *fa_bhh_b; /* fa.gru reverse */
    const float *fa_fc_w, *fa_fc_b;                    /* Linear(8,4) */
} CtfaSpec;

static inline void ctfa_apply(const CtfaSpec *s, float *x, float *tfa_cache) {
    int C = s->C, F = s->F;
    float zt[256];
    for (int c = 0; c < C; ++c) {
        float acc = 0;
        for (int f = 0; f < F; ++f) acc += x[c * F + f] * x[c * F + f];
        zt[c] = acc / F;
    }
    gru_step(zt, C, tfa_cache, s->ta_hidden, s->ta_wih, s->ta_whh, s->ta_bih, s->ta_bhh);
    float at[256];
    for (int c = 0; c < C; ++c) {
        float acc = s->ta_fc_b[c];
        for (int j = 0; j < s->ta_hidden; ++j) acc += s->ta_fc_w[c * s->ta_hidden + j] * tfa_cache[j];
        at[c] = sigmoidf_(acc);
    }

    const int r = 4;
    int remainder = F % r;
    int pad_len = remainder ? (r - remainder) : 0;
    int F_pad = F + pad_len;
    int H = F_pad / r;
    float fa_in[264]; /* max F_pad encountered: 132 (width=129); generous headroom */
    memset(fa_in, 0, sizeof(float) * F_pad);
    for (int f = 0; f < F; ++f) {
        float acc = 0;
        for (int c = 0; c < C; ++c) acc += x[c * F + f] * x[c * F + f];
        fa_in[f] = acc / C;
    }
    float fa_gru_out[33 * 8]; /* max H=33 (width=129) * 2*r=8 */
    gru_bidirectional_seq(fa_in, H, r, r, s->fa_wih_f, s->fa_whh_f, s->fa_bih_f, s->fa_bhh_f,
                          s->fa_wih_b, s->fa_whh_b, s->fa_bih_b, s->fa_bhh_b, fa_gru_out);
    float fa_fc_out[33 * 4];
    for (int h = 0; h < H; ++h)
        for (int o = 0; o < r; ++o) {
            float acc = s->fa_fc_b[o];
            for (int j = 0; j < 8; ++j) acc += s->fa_fc_w[o * 8 + j] * fa_gru_out[h * 8 + j];
            fa_fc_out[h * r + o] = acc;
        }
    float af[264];
    for (int f = 0; f < F; ++f) af[f] = sigmoidf_(fa_fc_out[f]);

    for (int c = 0; c < C; ++c)
        for (int f = 0; f < F; ++f)
            x[c * F + f] = at[c] * x[c * F + f] * af[f];
}

/* Pointwise (1x1) grouped conv + BN + AffinePReLU + optional Shuffle. Used for XMBBlocks'
 * pconv1 and XDWSBlock's pconv (both always KT=1,KF=1,stride=1,pf=0, no cache). */
typedef struct {
    const float *conv_w, *conv_b, *bn_w, *bn_b, *bn_rm, *bn_rv, *affine_w, *affine_b, *slope;
    int C_in, C_out, groups, F;
    int do_shuffle;
} PointwiseSpec;

static inline void pointwise_bn_act(const PointwiseSpec *s, const float *x_in, float *x_out) {
    conv2d_causal_1frame(x_in, s->C_in, 1, s->F, s->conv_w, s->conv_b, s->C_out, 1, 1, 0,
                         s->groups, x_out, s->F);
    batchnorm_apply(x_out, s->C_out, s->F, s->bn_w, s->bn_b, s->bn_rm, s->bn_rv, 1e-5f);
    affine_prelu_apply(x_out, s->C_out, s->F, s->affine_w, s->affine_b, s->slope);
    if (s->do_shuffle) {
        float tmp[32 * 65]; /* max C_out*F encountered among pointwise stages */
        memcpy(tmp, x_out, sizeof(float) * s->C_out * s->F);
        shuffle_apply(tmp, s->C_out / 2, s->F, x_out);
    }
}

/* Temporal (or 1x1) conv/deconv + BN + optional AffinePReLU + optional cTFA, with an
 * optional persistent temporal cache (NULL => kt=1, no cache, e.g. encoder blocks 3/4 and
 * decoder blocks 0/1). Covers: XConvBlock's ops[1..5] (has_affine=1 unless is_last,
 * has_ctfa=1 always), XDWSBlock's dconv (has_affine=1, has_ctfa=1), XMBBlocks' middle
 * dconv (has_affine=1, has_ctfa=0) and pconv2 (has_affine=0, has_ctfa=1, KT=KF=1,
 * conv_cache=NULL always since pointwise). use_deconv selects Conv2d vs ConvTranspose2d
 * (weight layout differs accordingly -- caller must pass the correctly-shaped weight
 * tensor either way; the two kernel functions already handle their respective layouts). */
typedef struct {
    const float *conv_w, *conv_b, *bn_w, *bn_b, *bn_rm, *bn_rv;
    const float *affine_w, *affine_b, *slope; /* NULL => skip (is_last case) */
    int has_ctfa;
    CtfaSpec ctfa; /* only used if has_ctfa */
    int C_in, C_out, KT, KF, stride_f, pf, groups, F_in, F_out;
    int use_deconv;
} ConvBnActCtfaSpec;

static inline void conv_bn_act_ctfa(const ConvBnActCtfaSpec *s, const float *x_in,
                                    float *conv_cache /* NULL if KT==1 */, float *tfa_cache,
                                    float *x_out) {
    if (s->KT == 1) {
        if (s->use_deconv) {
            convtranspose2d_causal_1frame(x_in, s->C_in, 1, s->F_in, s->conv_w, s->conv_b,
                                          s->C_out, s->KT, s->KF, s->stride_f, s->pf, s->groups,
                                          x_out, s->F_out);
        } else {
            conv2d_causal_1frame(x_in, s->C_in, 1, s->F_in, s->conv_w, s->conv_b, s->C_out,
                                 s->KF, s->stride_f, s->pf, s->groups, x_out, s->F_out);
        }
    } else {
        int cache_frame_count = s->KT - 1;
        float conv_in[32 * 3 * 65]; /* generous: max C_in*KT*F_in among all instances */
        for (int c = 0; c < s->C_in; ++c) {
            memcpy(conv_in + c * s->KT * s->F_in, conv_cache + c * cache_frame_count * s->F_in,
                  (size_t)cache_frame_count * s->F_in * sizeof(float));
            memcpy(conv_in + c * s->KT * s->F_in + cache_frame_count * s->F_in,
                  x_in + c * s->F_in, (size_t)s->F_in * sizeof(float));
        }
        if (s->use_deconv) {
            convtranspose2d_causal_1frame(conv_in, s->C_in, s->KT, s->F_in, s->conv_w, s->conv_b,
                                          s->C_out, s->KT, s->KF, s->stride_f, s->pf, s->groups,
                                          x_out, s->F_out);
        } else {
            conv2d_causal_1frame(conv_in, s->C_in, s->KT, s->F_in, s->conv_w, s->conv_b, s->C_out,
                                 s->KF, s->stride_f, s->pf, s->groups, x_out, s->F_out);
        }
        for (int c = 0; c < s->C_in; ++c)
            memcpy(conv_cache + c * cache_frame_count * s->F_in,
                  conv_in + c * s->KT * s->F_in + s->F_in, (size_t)cache_frame_count * s->F_in * sizeof(float));
    }

    batchnorm_apply(x_out, s->C_out, s->F_out, s->bn_w, s->bn_b, s->bn_rm, s->bn_rv, 1e-5f);
    if (s->affine_w) affine_prelu_apply(x_out, s->C_out, s->F_out, s->affine_w, s->affine_b, s->slope);
    if (s->has_ctfa) ctfa_apply(&s->ctfa, x_out, tfa_cache);
}

/* DPGRNN: dual-path grouped RNN, fixed at C=16,width(F)=33,hidden=16 for this architecture
 * (both dpgrnn.0 and dpgrnn.1 share these dims). intra_rnn is a fresh bidirectional GRNN
 * over the F=33 frequency axis (no cross-frame cache, like cTFA's fa); inter_rnn is a
 * cached unidirectional GRNN with one independent hidden-state pair per frequency bin
 * (inter_cache: [33][16], 8+8 per GRNN half). x: [C=16,F=33] in, out same shape (may alias
 * a separate buffer); inter_cache updated in place. */
typedef struct {
    const float *intra_wih1, *intra_whh1, *intra_bih1, *intra_bhh1;
    const float *intra_wih1_rev, *intra_whh1_rev, *intra_bih1_rev, *intra_bhh1_rev;
    const float *intra_wih2, *intra_whh2, *intra_bih2, *intra_bhh2;
    const float *intra_wih2_rev, *intra_whh2_rev, *intra_bih2_rev, *intra_bhh2_rev;
    const float *intra_fc_w, *intra_fc_b, *intra_ln_w, *intra_ln_b;
    const float *inter_wih1, *inter_whh1, *inter_bih1, *inter_bhh1;
    const float *inter_wih2, *inter_whh2, *inter_bih2, *inter_bhh2;
    const float *inter_fc_w, *inter_fc_b, *inter_ln_w, *inter_ln_b;
} DpgrnnSpec;

#define DPGRNN_C 16
#define DPGRNN_F 33

static inline void dpgrnn_forward(const DpgrnnSpec *s, const float *x_cf /* [16][33] */,
                                  float *inter_cache /* [33][16] */, float *out_cf /* [16][33] */) {
    float x_fc[DPGRNN_F * DPGRNN_C];
    for (int f = 0; f < DPGRNN_F; ++f)
        for (int c = 0; c < DPGRNN_C; ++c) x_fc[f * DPGRNN_C + c] = x_cf[c * DPGRNN_F + f];

    float g1_in[DPGRNN_F * 8], g2_in[DPGRNN_F * 8];
    for (int f = 0; f < DPGRNN_F; ++f) {
        memcpy(g1_in + f * 8, x_fc + f * DPGRNN_C, 8 * sizeof(float));
        memcpy(g2_in + f * 8, x_fc + f * DPGRNN_C + 8, 8 * sizeof(float));
    }
    float g1_out[DPGRNN_F * 8], g2_out[DPGRNN_F * 8];
    gru_bidirectional_seq(g1_in, DPGRNN_F, 8, 4, s->intra_wih1, s->intra_whh1, s->intra_bih1, s->intra_bhh1,
                          s->intra_wih1_rev, s->intra_whh1_rev, s->intra_bih1_rev, s->intra_bhh1_rev, g1_out);
    gru_bidirectional_seq(g2_in, DPGRNN_F, 8, 4, s->intra_wih2, s->intra_whh2, s->intra_bih2, s->intra_bhh2,
                          s->intra_wih2_rev, s->intra_whh2_rev, s->intra_bih2_rev, s->intra_bhh2_rev, g2_out);

    float intra_gru_out[DPGRNN_F * 16];
    for (int f = 0; f < DPGRNN_F; ++f) {
        memcpy(intra_gru_out + f * 16, g1_out + f * 8, 8 * sizeof(float));
        memcpy(intra_gru_out + f * 16 + 8, g2_out + f * 8, 8 * sizeof(float));
    }
    float intra_x[DPGRNN_F * 16];
    for (int f = 0; f < DPGRNN_F; ++f)
        for (int o = 0; o < 16; ++o) {
            float acc = s->intra_fc_b[o];
            for (int j = 0; j < 16; ++j) acc += s->intra_fc_w[o * 16 + j] * intra_gru_out[f * 16 + j];
            intra_x[f * 16 + o] = acc;
        }
    layernorm_apply(intra_x, DPGRNN_F, 16, s->intra_ln_w, s->intra_ln_b, 1e-8f);

    float intra_out[DPGRNN_F * 16];
    for (int i = 0; i < DPGRNN_F * 16; ++i) intra_out[i] = x_fc[i] + intra_x[i];

    float inter_gru_out[DPGRNN_F * 16];
    for (int f = 0; f < DPGRNN_F; ++f) {
        float in1[8], in2[8];
        memcpy(in1, intra_out + f * 16, 8 * sizeof(float));
        memcpy(in2, intra_out + f * 16 + 8, 8 * sizeof(float));
        float* h1 = inter_cache + f * 16;
        float* h2 = inter_cache + f * 16 + 8;
        gru_step(in1, 8, h1, 8, s->inter_wih1, s->inter_whh1, s->inter_bih1, s->inter_bhh1);
        gru_step(in2, 8, h2, 8, s->inter_wih2, s->inter_whh2, s->inter_bih2, s->inter_bhh2);
        memcpy(inter_gru_out + f * 16, h1, 8 * sizeof(float));
        memcpy(inter_gru_out + f * 16 + 8, h2, 8 * sizeof(float));
    }
    float inter_x[DPGRNN_F * 16];
    for (int f = 0; f < DPGRNN_F; ++f)
        for (int o = 0; o < 16; ++o) {
            float acc = s->inter_fc_b[o];
            for (int j = 0; j < 16; ++j) acc += s->inter_fc_w[o * 16 + j] * inter_gru_out[f * 16 + j];
            inter_x[f * 16 + o] = acc;
        }
    layernorm_apply(inter_x, DPGRNN_F, 16, s->inter_ln_w, s->inter_ln_b, 1e-8f);

    float inter_out[DPGRNN_F * 16];
    for (int i = 0; i < DPGRNN_F * 16; ++i) inter_out[i] = intra_out[i] + inter_x[i];

    for (int f = 0; f < DPGRNN_F; ++f)
        for (int c = 0; c < DPGRNN_C; ++c) out_cf[c * DPGRNN_F + f] = inter_out[f * 16 + c];
}
