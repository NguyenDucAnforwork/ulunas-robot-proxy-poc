# BatchNorm folding: implementation and host measurements

Measured 2026-09-13. Opt-in export-time folding of all 22 frozen BatchNorm layers
passes the self-contained parity checks and reduces measured x86 mean hop latency
by **6.65%** (0.06266 ms/hop). Default inference and `build_and_test.sh` remain unfused.
No commits were made; all task changes are under `mobile/c_neon/`.

> **Correction (follow-up session, 2026-09-13):** the checkpoint used below,
> `F_GENERIC.tar`, is M1's *generic*-domain fine-tune, not `F_PROXY_ROBOT.tar` — the
> actual robot-proxy-adapted checkpoint this whole project targets. `export_weights.py`
> now takes an explicit `--checkpoint` argument (previously hardcoded) specifically
> because this mislabel was caught and needed fixing. Re-running the identical protocol
> against the correct `F_PROXY_ROBOT.tar` (SHA-256 `70c2a7d2...4a163`, verified against
> `REPRODUCTION.md`) gives: **parity 2.47e-6 spectral / 9.30e-6 PCM** (both still ≪1e-4),
> **x86 latency −6.70%** (0.94203ms → 0.87892ms) — the same conclusion, confirming the
> latency effect is architecture-driven (same graph shape) rather than an artifact of
> which checkpoint's weights were loaded. The measurements below are left unmodified as
> the historical record of what was actually measured that session; treat the numbers in
> this correction note, not the ones below, as authoritative for `F_PROXY_ROBOT`. See
> `ARM_FULL_GRAPH_RESULTS.md` for the equivalent real-ARM re-run.

## Implementation

- `export_weights.py --fold-bn` writes `generated/ulunas_weights_folded.h` and
  `generated/ulunas_arch_folded.json`, leaving the reference export separate.
  Float32 arithmetic uses `scale = gamma / sqrt(var + 1e-5)`,
  `shift = beta - mean * scale`, `W' = W * scale`, `b' = b * scale + shift`.
- The fixed architecture mapping checks coverage of exactly 22 BNs. Conv2d scales
  output-axis weight slices. Decoder temporal ConvTranspose2d weights are viewed
  as `[groups, Cin/groups, Cout/groups, KT, KF]` and scaled by the corresponding
  output channel. Decoder pointwise layers remain Conv2d. Shape and finite-value
  checks reject incompatible exports. All current convolutions have biases.
- `src/ulunas_full.cpp` selects the folded header only with `-DULUNAS_BN_FOLDED`.
  `src/kernels.h` makes `batchnorm_apply` a no-op in that mode, covering composite
  helpers and the four direct pconv2 BN calls. Generated header guards reject
  mismatched modes. BN arrays are retained for source compatibility but are unused
  in the folded graph; they are not rewritten to approximate identity parameters.
- Reference tensor values are unchanged; the reference header diff only adds a
  build-mode guard. No runtime dependency or initialization-time folding is added.
- `src/test_bn_fold_parity.cpp` links independently compiled reference/folded
  graphs (folded API symbols renamed), with independent streaming states and
  identical in-process inputs. `measure_bn_fold.sh` builds and runs this test,
  both edge tests, and sequential alternating latency benchmarks.

## Numerical parity

Checkpoint: `/content/project/checkpoints/F_GENERIC.tar` (successfully loaded).
SHA-256: `11a20e25b06c446b2e205b10c3778a658480c0622b229a8c3b82c8949e320fb5`.
Both exports contain 409 float tensors / 172,290 float32 values.

| Check | Result |
| --- | --- |
| Full graph, 20 consecutive 257-bin complex STFT frames | Max absolute error **1.043081284e-7** |
| Full PCM pipeline, 20 consecutive 256-sample hops | Max absolute error **6.675720215e-6** |
| Reference `test_edge_cases` | All pass |
| Folded `test_edge_cases` | All pass |
| Reset/reprocess determinism, each build | Bit-exact, max difference **0** |

Parity inputs use `std::mt19937(20260913)`, uniform floats in [-1, 1], with RNG
continuing from the spectral case into the PCM case. States reset between cases;
all outputs are checked for finiteness. The fixed acceptance limit is `1e-4`,
including startup frames/hops, with no discarded transient and no tolerance increase.
These are finite-sequence folded-vs-reference checks, not renewed PyTorch parity
or an exhaustive bound on all possible inputs.

Existing edge tests cover silence (200 hops), impulse (50 hops), random noise
(500 hops), and repeated identical input after reset (30 hops per run).
No fixture-dependent block/full tests were used. A GCC `-std=c99 -O2` smoke compile
of the folded generated header plus `kernels.h` also passed; the existing full
pipeline/benchmark uses its C++ wrapper.

## Native x86 latency (NOT ARM/iPhone)

Host: x86_64 Intel Xeon CPU @ 2.20 GHz, KVM guest, 12 logical CPUs.
Compiler: g++ 13.3.0, `-O2 -std=c++17`, no fast-math or architecture tuning flags.
Native host execution, no emulator or ARM cross-compiler. CPU affinity was not
pinned. A concurrent training job may affect scheduling; these are host measurements,
not isolated target-device performance guarantees.

Unmodified `src/benchmark_latency.cpp`: 2,000 measured hops after 100 warmup hops
per run; random PCM seed 1234, amplitude [-0.2, 0.2]. One step is one 256-sample hop
at 16 kHz (16 ms of audio). Timed region is `ulunas_process_hop`, including the
existing FFT/OLA wrapper and graph; input generation is outside the timed region.
Run order: reference/folded, folded/reference, reference/folded; no simultaneous
benchmark processes.

| Pair | Build | Mean ms/step | p95 ms/step | RTF_mean | RTF_p95 |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | Reference | 0.94729 | 0.97347 | 0.05921 | 0.06084 |
| 1 | Folded | 0.88165 | 0.90602 | 0.05510 | 0.05663 |
| 2 | Reference | 0.93321 | 0.96609 | 0.05833 | 0.06038 |
| 2 | Folded | 0.87761 | 0.90369 | 0.05485 | 0.05648 |
| 3 | Reference | 0.94551 | 0.97830 | 0.05909 | 0.06114 |
| 3 | Folded | 0.87877 | 0.91298 | 0.05492 | 0.05706 |

Averaging the three per-run statistics:

| Build | Mean ms/step | Average per-run p95 ms | RTF_mean* | Average per-run RTF_p95* |
| --- | ---: | ---: | ---: | ---: |
| Reference | 0.94200 | 0.97262 | 0.05888 | 0.06079 |
| Folded | 0.87934 | 0.90756 | 0.05496 | 0.05672 |
| Change | **-6.65%** | **-6.69%** | **-6.65%** | **-6.69%** |

*Aggregate RTFs and percentage changes computed from ms / 16 before rounding.
The average of run p95s is not a pooled percentile. Percentage change is
`100 * (folded / reference - 1)`.

The payoff is modest: about 63 microseconds per hop on this host, consistent with
removing a small part of the total work. All three pairs favor folding, but timing
also includes compiler code-generation effects and host noise; it does not directly
measure BN's isolated compute fraction. No claim is made about NEON, Cortex-A53,
Apple A13, or any real ARM/iPhone latency. Target-device validation remains necessary.

## Reproduce

From the repository root, with the checkpoint available and Python torch/numpy installed:

```bash
./mobile/c_neon/measure_bn_fold.sh
```

This exports both variants, builds with `-O2`, enforces parity/edge-test failures,
and runs the six 2000/100 benchmarks. Logs and binaries are under
`mobile/c_neon/build/bn_fold/` (build artifacts are ignored by git).
Optional `CPU=N ./mobile/c_neon/measure_bn_fold.sh` pins benchmark execution to an
allowed logical CPU; that option was not used for the measurements above.

For just an opt-in folded benchmark:

```bash
cd mobile/c_neon
python3 export_weights.py --fold-bn
g++ -O2 -std=c++17 -Isrc -DULUNAS_BN_FOLDED \
  src/benchmark_latency.cpp src/ulunas_full.cpp -o build/benchmark_bn_folded
./build/benchmark_bn_folded 2000 100
```

For the reference, export without `--fold-bn` and compile without
`-DULUNAS_BN_FOLDED`. The original fixture-dependent `build_and_test.sh` is unchanged.
