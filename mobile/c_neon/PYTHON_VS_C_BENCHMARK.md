# Python/PyTorch vs C reference: fair latency + parity benchmark

Measured 2026-09-13. Answers: how much faster is the runtime-free C engine than the
streaming PyTorch reference it was ported from, on identical input, at two levels
(model-only and end-to-end), with no optimization (no BN-folding, no source changes)
on either side. Scripts: `pybench/` (ad hoc, not part of the production build).

## 1. What was used

**Python** — `mobile/onnx_export/ulunas_onnx/stream/ulunas_stream.py::StreamULUNAS`,
loaded exactly as `mobile/onnx_export/export_finetuned_stream.py` does (`ULUNAS()` loads
the checkpoint, `state_dict()` copied into `StreamULUNAS(strict=True)`). This is the
same streaming reference class already used to validate this C code's parity — not
written from scratch for this benchmark. `SEtrain_adapted/infer.py` was inspected and
excluded: it imports `models.gtcrn_end2end.GTCRN` — legacy GTCRN from the original
template repo, not UL-UNAS.

**C** — `mobile/c_neon/src/ulunas_full.cpp`, unmodified production code, called via
`ulunas_process_hop` (end-to-end) and `ulunas_process_frame_spec` (model-only, "the
exact same I/O contract as the streaming ONNX graph" per its own header comment — i.e.
the same contract as `StreamULUNAS.forward()`). Built **without** `-DULUNAS_BN_FOLDED`
(reference/unoptimized build).

**Checkpoint: `F_PROXY_ROBOT.tar`**, not `F_GENERIC.tar` — `generated/ulunas_weights.h`
(the real weights compiled into the C binary) was corrected to this checkpoint earlier
in this session (see `ARM_FULL_GRAPH_RESULTS.md`), so it is the checkpoint that actually
"corresponds to the C graph" right now. The Python side loads the same checkpoint so the
parity numbers below are a meaningful same-weights comparison.

## 2. Hardware / toolchain

| | |
|---|---|
| Compiler (C) | g++ 13.3.0, `-O2 -std=c++17 -I<mobile/c_neon/src>` (identical flags to `benchmark_latency.cpp`'s own reference build) |
| PyTorch | 2.11.0+cu128, CPU, `torch.set_num_threads(1)`, `torch.set_num_interop_threads(1)` |
| CPU | Intel Xeon @ 2.20GHz (x86_64, KVM guest, 12 vCPU) — same machine for both sides |
| OS | Linux 6.6.122 |
| Timing | C: `std::chrono::high_resolution_clock` (same mechanism as `benchmark_latency.cpp`); Python: `time.perf_counter_ns()` |
| Input | seed=1234, `std::mt19937` + `uniform_real_distribution(-0.2,0.2)` (same as `benchmark_latency.cpp`), dumped to a file so C and Python consume byte-identical input rather than relying on replicating C++'s RNG algorithm in Python |
| Protocol | 100 warmup hops (discarded) + 2000 measured hops/frames, 3 runs per side |

## 3. Latency results

### End-to-end (PCM hop → STFT → model → complex mask → iSTFT/OLA → PCM hop)

| Run | C mean (ms) | Python mean (ms) |
|---|---:|---:|
| 1 | 0.93278 | 30.35537 |
| 2 | 0.95630 | 31.03910 |
| 3 | 0.94830 | 31.27574 |
| **avg** | **0.94579** | **30.89007** |

| Metric | C (avg of 3) | Python (avg of 3) |
|---|---:|---:|
| p50 (ms) | 0.94487 | 30.76063 |
| p95 (ms) | 0.97323 | 31.62217 |
| p99 (ms) | 1.00782 | 32.55234 |
| max (ms) | 1.35008 | 37.08241 |
| RTF_mean (16ms/hop) | **0.0591** | **1.9306** (not real-time on 1 thread) |

**C speedup = 32.66x. Latency reduction = 96.94% (−29.94 ms/hop).**

### Model-only (spectral frame → model → enhanced spectral frame)

| Run | C mean (ms) | Python mean (ms) |
|---|---:|---:|
| 1 | 0.90222 | 30.27103 |
| 2 | 0.91549 | 30.91884 |
| 3 | 0.91831 | 30.98939 |
| **avg** | **0.91201** | **30.72642** |

| Metric | C (avg of 3) | Python (avg of 3) |
|---|---:|---:|
| p50 (ms) | 0.90617 | 30.58584 |
| p95 (ms) | 0.96213 | 31.58921 |
| p99 (ms) | 1.06587 | 32.74470 |
| max (ms) | 1.19068 | 39.65277 |
| RTF_mean (16ms/hop) | **0.0570** | **1.9204** |

**C speedup = 33.69x. Latency reduction = 97.03% (−29.81 ms/hop).**

Model-only and end-to-end are close on both sides (C: 0.912 vs 0.946ms; Python: 30.73 vs
30.89ms) — STFT/OLA is a small fraction of total cost; almost all latency on both sides
is the model forward pass itself.

## 4. Output parity (same weights, same input)

| | End-to-end (PCM) | Model-only (spectral) |
|---|---:|---:|
| max abs error | **1.49e-7** | **8.38e-8** |
| RMSE | 9.49e-9 | 1.69e-9 |
| NaN/Inf (C) | none | none |
| NaN/Inf (Python) | none | none |

Float32-machine-epsilon-level agreement, unchanged when excluding the first 1/2/4/8
hops/frames — **no startup-transient blowup**, unlike the earlier native-vs-WASM
comparison in `../web_demo/README.md`. The difference: that comparison was two
*compilers* of the *identical* C++ source (where an ill-conditioned early-frame OLA
division amplified ordinary cross-compiler rounding differences); this comparison is
two fully independent implementations in different languages/libraries (PyTorch/MKL vs
a hand-written radix-2 FFT + scalar kernels) agreeing to near machine precision — a
much stronger, more reassuring result about the C port's fidelity to the trained model.

## 5. What keeps this from being perfectly apples-to-apples

- `-O2` is an optimization flag that was **not** disabled — comparing against `-O0`
  would be an unfair strawman; `-O2` is this project's own established baseline
  (`benchmark_latency.cpp`, `measure_bn_fold.sh` both use it).
- Python's cost is dominated by PyTorch eager-mode dispatch overhead (many small ops,
  each crossing the Python/C++ boundary) — this is *why* Python is slow here, not
  because the model does more arithmetic; a TorchScript/ONNX Runtime-compiled path
  would likely close much of this gap (out of scope for this benchmark).
- STFT/iFFT use different libraries: C's own radix-2 Cooley-Tukey (`fft.h`) vs
  `torch.fft.rfft/irfft` (MKL). Both follow the same convention (periodic Hann,
  unnormalized forward / 1/n-normalized inverse), which is exactly why parity stays
  this tight despite being independent implementations.
- Both ran on a cloud x86 VM, not a mobile ARM target — this answers "how much faster
  is C than Python," not "what will this be on a phone."

## 6. Reproduce

See `pybench/README.md`.
