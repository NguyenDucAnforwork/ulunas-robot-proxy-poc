// Real-ARM microbenchmark for UL-UNAS's two claimed inference optimizations, run on an
// actual AWS Graviton (Neoverse, aarch64) instance -- NOT x86, NOT QEMU. Copies the exact
// kernel formulas from mobile/c_neon/src/kernels.h verbatim (not reimplemented from memory)
// with representative-but-synthetic sizes (max GRU input=32/hidden=48, max BN channel*freq
// =32*65, per generated/ulunas_arch.json), since the full 697-node graph + real trained
// weights (4.7MB header) could not be transferred through this EC2 policy's only channel
// (user-data, no SSH/S3/key-pair permission granted). Answers two items the x86-only
// BN_FOLD_RESULTS.md and PROJECT_STATUS.md explicitly flagged as blocked: (1) a real ARM
// number for BN folding (previously x86-only), (2) a real ARM number for NEON vs scalar
// dot-product (previously correctness-only on qemu, never timed).
#include <arm_neon.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>

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
static inline float scalar_dot(const float* a, const float* b, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) sum += a[i] * b[i];
    return sum;
}
static inline void gru_step_neon(const float* x, int input_size, float* h, int hidden_size,
                                  const float* w_ih, const float* w_hh, const float* b_ih, const float* b_hh) {
    float gates_i[3 * 256], gates_h[3 * 256];
    for (int g = 0; g < 3 * hidden_size; ++g) {
        gates_i[g] = b_ih[g] + neon_dot(w_ih + g * input_size, x, input_size);
        gates_h[g] = b_hh[g] + neon_dot(w_hh + g * hidden_size, h, hidden_size);
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
static inline void gru_step_scalar(const float* x, int input_size, float* h, int hidden_size,
                                    const float* w_ih, const float* w_hh, const float* b_ih, const float* b_hh) {
    float gates_i[3 * 256], gates_h[3 * 256];
    for (int g = 0; g < 3 * hidden_size; ++g) {
        gates_i[g] = b_ih[g] + scalar_dot(w_ih + g * input_size, x, input_size);
        gates_h[g] = b_hh[g] + scalar_dot(w_hh + g * hidden_size, h, hidden_size);
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
static inline void bn_apply_unfolded(float* x, int C, int F, const float* weight, const float* bias,
                                      const float* rm, const float* rv, float eps) {
    for (int c = 0; c < C; ++c) {
        float scale = weight[c] / sqrtf(rv[c] + eps);
        float shift = bias[c] - rm[c] * scale;
        for (int f = 0; f < F; ++f) x[c * F + f] = x[c * F + f] * scale + shift;
    }
}

static float randf() { return (float)rand() / RAND_MAX * 2.0f - 1.0f; }

template <typename F>
double time_ns_per_call(F&& fn, int warmup, int iters) {
    for (int i = 0; i < warmup; ++i) fn();
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) fn();
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::nano>(t1 - t0).count() / iters;
}

int main() {
    srand(20260913);
    const int WARMUP = 200, ITERS = 20000;

    // ---- 1) NEON vs scalar dot-product, n=48 (max GRU hidden dim in the real model) ----
    const int N = 48;
    float a[N], b[N];
    for (int i = 0; i < N; ++i) { a[i] = randf(); b[i] = randf(); }
    volatile float sink = 0;
    double t_scalar_dot = time_ns_per_call([&]{ sink = scalar_dot(a, b, N); }, WARMUP, ITERS);
    double t_neon_dot   = time_ns_per_call([&]{ sink = neon_dot(a, b, N); }, WARMUP, ITERS);

    // ---- 2) GRU cell step, input=32 hidden=48 (largest encoder ta_gru in the real model) ----
    const int IN = 32, HID = 48;
    float x[IN], h_s[HID], h_n[HID], w_ih[3*HID*IN], w_hh[3*HID*HID], bi[3*HID], bh[3*HID];
    for (int i = 0; i < IN; ++i) x[i] = randf();
    for (int i = 0; i < HID; ++i) h_s[i] = h_n[i] = randf();
    for (int i = 0; i < 3*HID*IN; ++i) w_ih[i] = randf() * 0.1f;
    for (int i = 0; i < 3*HID*HID; ++i) w_hh[i] = randf() * 0.1f;
    for (int i = 0; i < 3*HID; ++i) { bi[i] = randf() * 0.1f; bh[i] = randf() * 0.1f; }
    double t_gru_scalar = time_ns_per_call([&]{ gru_step_scalar(x, IN, h_s, HID, w_ih, w_hh, bi, bh); }, WARMUP, ITERS);
    double t_gru_neon   = time_ns_per_call([&]{ gru_step_neon(x, IN, h_n, HID, w_ih, w_hh, bi, bh); }, WARMUP, ITERS);

    // ---- 3) BatchNorm apply (unfolded) vs folded (no-op), C=32 F=65 (max pointwise stage size) ----
    const int C = 32, FSZ = 65;
    float xb[C*FSZ], bn_w[C], bn_b[C], bn_rm[C], bn_rv[C];
    for (int i = 0; i < C*FSZ; ++i) xb[i] = randf();
    for (int i = 0; i < C; ++i) { bn_w[i] = 1.0f + randf()*0.1f; bn_b[i] = randf()*0.1f; bn_rm[i] = randf()*0.1f; bn_rv[i] = 1.0f + fabsf(randf()); }
    double t_bn_unfolded = time_ns_per_call([&]{ bn_apply_unfolded(xb, C, FSZ, bn_w, bn_b, bn_rm, bn_rv, 1e-5f); }, WARMUP, ITERS);
    double t_bn_folded = time_ns_per_call([&]{ /* folded: no-op, exactly matching kernels.h ULUNAS_BN_FOLDED path */ (void)xb; }, WARMUP, ITERS);

    printf("=== REAL ARM (aarch64) MICROBENCHMARK RESULTS ===\n");
    printf("dot_product,n=%d: scalar=%.2f ns/call, neon=%.2f ns/call, speedup=%.3fx\n",
           N, t_scalar_dot, t_neon_dot, t_scalar_dot / t_neon_dot);
    printf("gru_cell,input=%d,hidden=%d: scalar=%.2f ns/call, neon=%.2f ns/call, speedup=%.3fx\n",
           IN, HID, t_gru_scalar, t_gru_neon, t_gru_scalar / t_gru_neon);
    printf("batchnorm_apply,C=%d,F=%d: unfolded=%.2f ns/call, folded(no-op)=%.2f ns/call, delta=%.2f ns/call\n",
           C, FSZ, t_bn_unfolded, t_bn_folded, t_bn_unfolded - t_bn_folded);
    printf("sink=%f (prevents dead-code elimination)\n", sink);
    printf("=== END RESULTS ===\n");
    return 0;
}
