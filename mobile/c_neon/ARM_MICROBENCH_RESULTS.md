# Real-ARM (AWS Graviton) microbenchmark: NEON and BatchNorm-folding

Measured 2026-09-13. First **real ARM silicon** numbers for this project — every prior
number for these two optimizations was either x86-host (`BN_FOLD_RESULTS.md`) or
QEMU aarch64 correctness-only (never timed, per `PROJECT_STATUS.md`). This closes that
specific gap; it does **not** replace the still-open Cortex-A53/iPhone hardware benchmark.

## What this is and isn't

- **Real hardware**: AWS `t4g.small` (Graviton2, Neoverse-N1, aarch64), Amazon Linux 2023,
  GCC 11.5.0, `-O2 -std=c++17`, no cross-compiler/emulator involved.
- **Not the full model**: the full 697-node graph + real trained weights (a 4.7MB generated
  header) could not be transferred to the instance — the IAM policy available for this
  benchmark allows only `RunInstances`/`TerminateInstances`/`GetConsoleOutput` (no SSH key
  pair, no security-group ingress, no S3), so the only data channel in is EC2 user-data
  (16KB limit) and the only channel out is console output. Instead, the exact kernel
  formulas (`neon_dot`, `gru_step`, `batchnorm_apply`) were copied verbatim from
  `src/kernels.h` into a small self-contained microbenchmark (`src/arm_microbench.cpp`,
  ~5KB) and run with synthetic-but-representative sizes read from
  `generated/ulunas_arch.json` (max GRU input=32/hidden=48, max BN pointwise-stage
  size C=32×F=65). This measures the real per-primitive cost on real ARM NEON hardware,
  **not** end-to-end hop latency/RTF for the real model — that still requires shipping
  real weights to a reachable ARM host (see Next steps).
- **Neoverse-N1, not Cortex-A53 or Apple Silicon**: Graviton2 is a server-class ARMv8.2-A
  core, not the mobile/embedded Cortex-A53 the iPhone-11-class target uses, and not Apple's
  A13. Both implement the same NEON ISA, so the *qualitative* finding (NEON meaningfully
  beats scalar on real ARM, unlike anything an x86 host could show) transfers; the
  *absolute* ns numbers do not, and are not claimed to.

## Results

| Primitive | Scalar | NEON / folded | Speedup / delta |
|---|---|---|---|
| dot-product, n=48 | 62.70 ns/call | 8.01 ns/call | **7.83x** |
| GRU cell step, input=32, hidden=48 | 10714.56 ns/call | 4202.85 ns/call | **2.55x** |
| BatchNorm apply, C=32, F=65 (2080 elem) | 1443.78 ns/call (unfolded) | 0.00 ns/call (folded, compiled out) | **1443.78 ns/call removed** |

20000 measured iterations after 200 warmup per primitive, `std::chrono::steady_clock`,
seed 20260913, `volatile` sink to block dead-code elimination on the dot-product result.
Full console log: `arm_graviton_console_output.log`. Source: `src/arm_microbench.cpp`.

## Reading these honestly

- **NEON dot-product**: 7.83x on real ARM — this is the number x86 structurally cannot
  produce (x86 has no NEON at all; the previous QEMU runs only checked correctness, never
  timing, since QEMU's instruction emulation is not representative of real-silicon
  throughput). This confirms NEON is a real, large win for the GRU gate matmuls specifically
  on ARM hardware, as the `kernels.h` comment predicted ("by far the most-called primitive
  ... hence the first NEON target").
- **Full GRU cell speedup is much smaller than the raw dot-product speedup** (2.55x vs
  7.83x) because a GRU step is dominated by `3*hidden_size` separate short dot-products
  (n=32 and n=48, small enough that NEON's fixed per-call overhead matters) plus
  non-vectorized `expf`/`tanhf` calls per gate — the amortized win is real but far from
  the ideal 4-wide NEON speedup.
- **BatchNorm removal cost, real ARM**: 1443.78 ns for one C=32×F=65 pointwise-stage BN
  application — this is a real ARM number for exactly the operation `BN_FOLD_RESULTS.md`
  could only measure on x86. It is **not** directly comparable to the x86 hop-level
  −6.65% figure, and it is **not** extrapolated to a full-graph ARM percentage here: the
  22 real BN layers have a range of C×F sizes (this is the largest pointwise stage, an
  upper-bound single-layer estimate, not an average), and without the full graph's
  per-hop total on ARM there is no honest denominator to divide by. Reporting the
  raw per-call ns instead of guessing a percentage.

## Next steps (unblocks the remaining PROJECT_STATUS.md BLOCKED_EXTERNAL items)

1. **Full-graph real-ARM hop latency**: needs either (a) broader IAM permissions on this
   AWS key (S3 or a key-pair/SSH path, to transfer the real 4.7MB weight header + full
   `ulunas_full.cpp` graph and run the actual `measure_bn_fold.sh`/`benchmark_latency.cpp`
   unmodified), or (b) a Graviton instance type this policy already allows plus a
   permitted way to get >16KB of data in (e.g., add `s3:GetObject`/`s3:PutObject` scoped to
   one bucket, or `ec2-instance-connect:SendSSHPublicKey` scoped to this instance).
2. **Real mobile silicon (Cortex-A53 / iPhone 11)**: still requires physical hardware or
   AWS EC2 Mac + Xcode + Device Farm, per `EXPERIMENT_SWEEP.md` §5 and
   `IOS_BUILD_AND_DEMO.md` — Graviton is a real, useful ARM data point but does not
   substitute for the actual target device.
