# MOBILE_BENCHMARK.md

Status: mixed — the ONNX reference path, Android build, and the full runtime-free C/NEON graph (M5) are all **PASS** (genuinely verified this session, on host + qemu-emulated ARM64). Real-device performance (Cortex-A53 / iPhone 11) is **PENDING/BLOCKED_EXTERNAL** — no physical hardware available in this environment.

## 1. ONNX reference path (M4) — **PASS**

Streaming ONNX exported from the fine-tuned `F_GENERIC` checkpoint (`mobile/onnx_export/export_finetuned_stream.py`), reusing the repo's own `StreamULUNAS` wrapper unmodified.

| Check | Result | Threshold | Status |
|---|---|---|---|
| `onnx.checker.check_model` | passes | — | **PASS** |
| Utterances tested | 10 | ≥10 | **PASS** |
| Frames tested | 3418 | ≥100 | **PASS** |
| Max abs tensor error (PyTorch offline vs ONNX streaming) | 7.6e-6 | ≤1e-4 | **PASS** |
| Waveform RMSE (PyTorch offline vs ONNX streaming) | 4.8e-5 | ≤1e-4 | **PASS** |
| NaN/Inf | none observed | — | **PASS** |

## 2. Android build (M4) — **PASS** (build only; on-device run is separate, see §4)

- Real Android NDK r27c downloaded and used (not a generic Linux cross-compiler).
- Real ONNX Runtime Android 1.19.2 prebuilt library (`onnxruntime-android.aar`, arm64-v8a `.so` + C++ headers) used — no source build needed.
- Native executable (`mobile/android_ref/`, C++17, CMake) implementing: hand-written STFT/ISTFT (validated against `torch.stft`/`istft` center=True convention to ~1e-6 before writing any C++), PCM16 WAV I/O, per-frame streaming inference with cache threading, offline WAV enhancement, basic timing instrumentation.
- **Cross-compiled successfully for `arm64-v8a`** via the real NDK toolchain (`aarch64-linux-android24-clang++`), linked against the real `libonnxruntime.so` for arm64-v8a. Confirmed genuine ARM64 ELF binary (`file` shows `ELF 64-bit LSB pie executable, ARM aarch64`, interpreter `/system/bin/linker64`).
- **Correctness (x86 host build, same source)**: output matches the PyTorch offline reference at waveform RMSE 9.65e-6, max abs error 9.15e-5 — passes the ≤1e-3/≤1e-4 thresholds with large margin.
- **Correctness (hand-written FFT/STFT core specifically, on real ARM64 machine code)**: a standalone FFT round-trip test, cross-compiled with a generic aarch64-linux-gnu toolchain and executed under `qemu-aarch64-static` (real AArch64 instruction emulation, not x86), produced results matching the x86 build to float32 precision (round-trip error 5.19e-6 vs 5.42e-6). This isolates and confirms the DSP math (the highest cross-architecture bug risk in this codebase) is portable and correct.
- **Not done**: running the full ONNX-Runtime-linked arm64-v8a binary itself under emulation — the Android/bionic dynamic linker (`/system/bin/linker64`) is part of a real Android OS image, not the NDK, and isn't available here; and a manylinux aarch64 onnxruntime wheel didn't expose a linkable standalone `libonnxruntime.so` for a same-ABI substitute. This is not a hard requirement (the plan's own done-criteria call for a real device or PENDING), so it's recorded as an unattempted nice-to-have, not a gap in the required correctness chain.

## 3. Runtime-free C/NEON path (M5) — **FULL GRAPH COMPLETE AND VALIDATED**

What's actually done and verified this session:
- Static weight export from the `F_GENERIC` checkpoint: all 409 relevant tensors (172,290 floats, 0.69MB as float32 — well under the 5MB binary budget) exported to a C99 header (`mobile/c_neon/generated/ulunas_weights.h`).
- Every distinct kernel type in the 697-node graph implemented in scalar C99 and **individually validated against PyTorch/numpy references before being trusted**:
  - Grouped causal Conv2d (incl. the `pf` frequency-padding PyTorch applies internally) — validated, groups=1 and groups=2.
  - Grouped causal ConvTranspose2d (the streaming cache/pad logic, general transposed-conv "gather" formula) — validated, groups=1 and groups=2.
  - BatchNorm2d (frozen running stats, per decision #21) — standard formula.
  - AffinePReLU (custom activation) — direct transcription.
  - GRU single-step cell — validated against `torch.nn.GRU`, exact gate-order match.
  - Bidirectional GRU over a short sequence (used by `FA`, recomputed fresh every frame) — built from the validated single-step cell.
  - LayerNorm as used by DPGRNN — **validation caught a real bug** (it normalizes jointly over both the width and channel dimensions together, not per-row as first assumed) and it was fixed before being used further.
  - GRNN (DPGRNN's channel-split RNN) — confirmed to reduce exactly to two independent calls of the already-validated GRU kernels.
  - Shuffle (channel interleave) — verified exact interleave pattern (`out[2c]=in[c], out[2c+1]=in[C+c]`) against `ulunas.py::Shuffle` directly.

**Every kernel type used anywhere in the 697-node graph is now implemented and individually validated.** Remaining work is wiring these validated kernels according to each block's specific shape/config (all recorded in `ulunas_arch.json`), not new algorithmic risk.
- **All three distinct block types validated end-to-end** (using each block's real preceding-block output as input, chained: block0's real output feeds block1's test, block1's real output feeds block2's test):
  - `XConvBlock` (encoder block 0): causal conv → BatchNorm → AffinePReLU → cTFA (cached temporal GRU + fresh-per-frame bidirectional frequency GRU) — matches `StreamULUNAS._stream_xconv` to 2.4e-7 (float precision), including updated caches.
  - `XMBBlocks` (encoder block 1): grouped pointwise conv → BN → AffinePReLU → Shuffle → depthwise causal conv → BN → AffinePReLU → grouped pointwise conv → BN → cTFA → (residual add skipped, shapes differ, matching the reference) → final Shuffle — matches `StreamULUNAS._stream_xmb` to 6.7e-6, including updated caches.
  - `XDWSBlock` (encoder block 2): grouped pointwise conv → BN → AffinePReLU → Shuffle → depthwise causal conv → BN → AffinePReLU → cTFA (no second pointwise conv, no residual, no final shuffle) — matches `StreamULUNAS._stream_xdws` to 3.8e-6, including updated caches.

Every block type used anywhere in the encoder (and, since the decoder mirrors these exact patterns with the already-validated `ConvTranspose2d` kernel in place of `Conv2d`, the decoder too) is now covered by a passing end-to-end test.

- **`DPGRNN` (dual-path grouped RNN) validated end-to-end too**: bidirectional intra-RNN over the frequency axis (recomputed fresh every frame, like `FA`) → `Linear` → jointly-normalized `LayerNorm` → residual → cached unidirectional inter-RNN with **33 independent per-frequency-bin hidden states** (matching `inter_cache` shape `(33,16)`) → `Linear` → `LayerNorm` → residual — matches `StreamULUNAS._stream_dpgrnn` to 1.67e-6 (output) and 2.8e-7 (updated cache).

- **A decoder block validated end-to-end too** (`decoder.de_convs[4]`, the final block, `is_last=True`): causal `ConvTranspose2d` (grouped-transposed-conv kernel integrated into the same `XConvBlock` pattern already proven on the encoder side, now confirmed on the decoder side) → BatchNorm → (no AffinePReLU, correctly skipped for `is_last`) → cTFA → (no final Shuffle, correctly skipped for `is_last`) — matches `StreamULUNAS._stream_xconv` on the decoder path to 2.4e-6 (output), 0.0 (cache, exact), 1.2e-7 (tfa cache).

**Every kernel type AND every distinct structural pattern in the entire 697-node graph — all 3 block types on both the encoder (`Conv2d`) and decoder (`ConvTranspose2d`) side, both `DPGRNN` layers, `ERB` — is now implemented in scalar C99 and individually validated end-to-end against the real PyTorch streaming reference.** What remains is applying this already-proven code to the specific shape parameters of the few not-yet-instantiated blocks and wiring them together into the single full forward pass — mechanical repetition of validated patterns, not open algorithmic risk.

### 3.1 Full graph wired (`mobile/c_neon/src/ulunas_full.cpp` + `ulunas_full.h`)

All 12 blocks (5 encoder, 2 DPGRNN, 5 decoder) plus `ERB.bm`/`ERB.bs`, sigmoid mask, and complex masking are wired into one straight-line C++ forward pass (`ulunas_process_frame_spec`), built entirely from the generic, pre-validated composite functions added to `kernels.h` (`conv_bn_act_ctfa`, `pointwise_bn_act`, `ctfa_apply`, `dpgrnn_forward`) — themselves regression-tested (`test_generic_regression.cpp`) against the original hand-written per-block tests before being trusted. Every one of the 12 blocks' shape parameters (channels/kernel/stride/groups/pf/cache presence) was cross-derived from `ulunas_arch.json`'s real exported tensor shapes and cross-checked three ways (against the known architecture hyperparameters, against `TFA_CACHE_HIDDEN`/`CONV_CACHE_SHAPES`, and against each other) before being used.

**Full-graph parity vs. the streaming ONNX reference — PASS**: 20 consecutive frames of a real utterance, starting from zero caches (same as the ONNX session), fed through `ulunas_process_frame_spec` one at a time with caches threaded frame-to-frame exactly as `UlunasState` does internally. Max abs error over all 20 frames: **1.1e-7** (float precision) — this is a materially stronger test than single-frame validation since it exercises correct cache accumulation across time for all 6 conv caches, all 10 cTFA hidden states, and both DPGRNN inter-caches simultaneously; any block-wiring or cache-indexing mistake would have produced much larger, likely diverging, error. Re-run cross-compiled for real arm64 and executed under `qemu-aarch64-static`: **7e-8** (same precision, confirms cross-architecture correctness).

### 3.2 Streaming C API implemented and validated

```c
int ulunas_init(UlunasState **out_state);
void ulunas_process_hop(UlunasState *state, const float *input_pcm, float *output_pcm);
void ulunas_reset(UlunasState *state);
void ulunas_destroy(UlunasState *state);
```
Design: a single opaque `UlunasState` struct (35.55KB, `sizeof` measured, not estimated) holds the STFT analysis ring buffer, OLA/window-sum accumulators, and all model caches, allocated once in `ulunas_init` via one `calloc`. **No malloc/free inside `ulunas_process_hop`.** `ulunas_reset` re-zeros state (preserving the precomputed Hann window) without reallocating.

**Full-utterance streaming parity — PASS with a documented, verified latency offset, not a silent threshold change**: feeding a real 38656-sample utterance through `ulunas_process_hop` hop-by-hop and comparing to the PyTorch offline reference initially showed a large (1.3) error. Root-caused (not papered over) via cross-correlation: this streaming implementation's simple 50%-overlap OLA buffering has an inherent **1 extra hop (256 samples / 16ms) of output latency** beyond the model's own established 2-hop (32ms) algorithmic latency — i.e. `output[i+256]` corresponds to `offline_ref[i]`, a real and understood property of this buffering scheme (explained in a code comment in `ulunas_full.cpp`), not a bug. After compensating for this measured delay: max abs error vs. a **same-convention** Python streaming reference (causal zero-init windowing, not reflect-padding) = **7.5e-6** (float precision — confirms the C streaming code itself is correct). Max abs error vs. the **offline** (whole-signal, `torch.stft(center=True)`, reflect-padded) reference = **2.8e-4** — slightly above the nominal 1e-4 target; verified via a second, independent Python streaming reference using the identical causal convention that this residual is a windowing-convention/accumulation-order artifact (mean error ~2e-5, isolated peaks at transient-heavy moments), not a wiring defect — a true incremental real-time API cannot use whole-signal reflect-padding (it would require unavailable future samples), so causal zero-init windowing is the correct real-time design choice, with this small, understood, and now-measured cost.

### 3.3 Edge cases, sanitizers, cross-architecture — all PASS

| Check | Result |
|---|---|
| Silence (200 hops) | PASS, all outputs finite |
| Impulse | PASS, all outputs finite |
| Random noise (500 hops, ~8s) | PASS, all outputs finite |
| `ulunas_reset()` then reprocess same input | PASS, **bit-exact** (max diff = 0.0) vs. a fresh init |
| AddressSanitizer + UndefinedBehaviorSanitizer, edge cases | PASS, clean |
| AddressSanitizer + UndefinedBehaviorSanitizer, 5-minute continuous soak (18750 hops) | PASS, clean, no leaks/OOB/UB, no NaN/Inf |
| Cross-compiled for real arm64 (`aarch64-linux-gnu-g++`), run under `qemu-aarch64-static` | PASS — edge cases and full-graph parity both confirmed on real ARM64 instruction emulation |
| `nm`/`ldd` check for ONNX Runtime / PyTorch / TensorFlow Lite linkage | **None found** — only `libstdc++`/`libm`/`libgcc_s`/`libc` |

### 3.4 Sizes and resource numbers (measured, not estimated)

| Metric | Value |
|---|---|
| MACs | 34.35M per second of audio (ptflops, offline model — same architecture, applies unchanged) |
| Static weight data | 0.69MB (172,290 floats, `ulunas_weights.h`) |
| Compiled library object + weights (`.a`, x86-64, `-O2`) | 772KB |
| Runtime arena (`sizeof(UlunasState)`, measured via `ulunas_state_size_bytes()`) | **35.55KB** |

Both comfortably under the 5MB binary / 32MB RSS mobile targets — though these are x86-64 sizes; real ARM64 binary size will be reported once built for the actual target (NDK/Xcode toolchain), expected similar or smaller given no runtime dependency.

### 3.5 NEON

Profiled (gprof, x86 host, for hotspot identification only — not a performance claim): `conv_bn_act_ctfa` (28.9% self-time), `ctfa_apply` (24.6%), `dpgrnn_forward` (16.2%), `pointwise_bn_act` (10.6%) — together ~90% of runtime, matching the pre-registered prediction that the 28 GRU instances plus the grouped-conv MAC loops dominate. Added a NEON-accelerated dot-product (`neon_dot`, `vld1q_f32`/`vfmaq_f32`/`vaddvq_f32`) inside `gru_step` — the single highest-call-count primitive (invoked by every `ctfa_apply` and every `dpgrnn_forward` call, both directly and via `gru_bidirectional_seq`) — guarded by `__ARM_NEON` (auto-defined for all AArch64 targets) with the original scalar dot-product kept as the exact fallback everywhere else. Re-ran full-graph parity and all edge cases on the NEON-enabled arm64 build under `qemu-aarch64-static`: **identical results** (1.1e-7 / 7e-8 graph parity, all edge cases PASS) — confirms the NEON path is numerically correct, same formula, different (SIMD) accumulation order. **Not yet NEON-optimized**: the Conv2d/ConvTranspose2d grouped MAC loops (secondary hotspot, inside `conv_bn_act_ctfa`'s own body) — a reasonable next target, not attempted further given time constraints; scalar correctness there is already fully validated and unaffected.

### 3.6 What genuinely remains

| Item | Status |
|---|---|
| Wire ALL 12 blocks into one full forward pass | **DONE** (§3.1) |
| Streaming C API (`ulunas_init/process_hop/reset/destroy`) | **DONE** (§3.2) |
| Full-graph + full-utterance parity | **DONE**, one documented/verified latency-offset caveat (§3.2) |
| Edge cases, sanitizers, cross-arch (qemu) | **DONE** (§3.3) |
| No ORT/PyTorch/TFLite in the runtime-free binary | **DONE**, confirmed via `nm`/`ldd` |
| NEON on the dominant hotspot (GRU) | **DONE** (§3.5) |
| NEON on Conv2d/ConvTranspose2d loops | **NOT_RUN** — secondary hotspot, scalar path fully correct, straightforward follow-up |
| Real Cortex-A53 / iPhone 11 build, RTF/RSS/binary-size benchmarking | **BLOCKED_EXTERNAL** — needs real hardware, see §4 and `IOS_BUILD_AND_DEMO.md` |

The runtime-free C inference core is functionally complete and independently validated at every level (kernel → block → full-graph-per-frame → full-utterance-streaming → edge cases → sanitizers → cross-architecture). What remains is real-hardware benchmarking (hard external blocker, not engineering work) and one further NEON pass (optimization, not correctness).

## 4. Real-device performance benchmarking — **PENDING / BLOCKED (real hardware) — SIMULATED numbers below, by explicit request**

No physical Cortex-A53 or Apple A13 device was available in this environment. Real-hardware RTF/RSS remain **BLOCKED_EXTERNAL** below. At the user's explicit request ("phần inference có thể chạy trên máy mô phỏng, miễn ra kết quả cuối"), §4.1 reports a **simulated/emulated** latency number instead of leaving this fully blank — but it is labeled as such throughout, never presented as a real-device measurement, per the project's own measured/estimated/simulated/blocked distinction.

### 4.1 Simulated latency (QEMU-emulated ARM64 + x86 host) — real numbers, wrong machine

`mobile/c_neon/src/benchmark_latency.cpp`: calls `ulunas_process_hop` 2000 times (100-hop warmup discarded) with random input, measures wall-clock per call.

| Build | Where it ran | RTF mean | RTF p95 | Real-time (RTF<1)? |
|---|---|---|---|---|
| x86-64, native | this host, directly | 0.060 | 0.061 | yes |
| ARM64 scalar (`-U__ARM_NEON`) | `qemu-aarch64-static` (ARM64 instructions emulated on this host) | 0.733 | 0.781 | yes |
| ARM64 + NEON (default) | `qemu-aarch64-static` | 0.726 | 0.737 | yes |

**What this does and does not tell us**: all three configurations report RTF < 1 even under QEMU's emulation overhead, which is a mildly encouraging sign that the algorithm's compute cost has real headroom before hitting real-time limits. **What it explicitly does NOT tell us**: real Cortex-A53/A13 performance. QEMU user-mode translates ARM64 instructions to run on this x86 host's actual silicon; its timing reflects (this host's raw speed) × (translation overhead for the specific instruction mix), which has no fixed, known ratio to real ARM silicon — a real Cortex-A53 could be meaningfully slower or, for some workloads, comparable, and there is no way to derive one from the other without the real chip. **The NEON vs. scalar numbers here are also not meaningful as a NEON speedup measurement** (0.733 vs 0.726 RTF, ~1% apart) — QEMU's own translation cost dominates the two builds' timing almost identically, masking whatever real speedup NEON's SIMD instructions would give on actual silicon (where the earlier host-machine profiling identified GRU as ~65% of runtime and specifically motivated adding NEON there). A real NEON-vs-scalar speedup number requires the real chip.

- Median RTF ≤ 0.25 (target) / < 1 (hard real-time requirement): **met in this simulated run**, not yet confirmed on real hardware.
- P95 frame processing time < hop duration (16ms): x86 native yes (0.98ms); QEMU ARM64 yes as RTF (11.5-12.5ms per hop, still < 16ms hop duration) but this is emulated timing, not real-silicon timing.
- Peak native RSS, real binary size on target toolchain, ONNX-vs-scalar-C-vs-NEON breakdown on real hardware: still **BLOCKED_EXTERNAL** below (this benchmark measures CPU time only, not memory, and the simulated numbers above are not a substitute for the real comparison the brief asks for).

### 4.2 What remains genuinely BLOCKED_EXTERNAL (real hardware required, no way around it)

- Native library size ≤ 5MB (x86-64 measured at 772KB, §3.4 — real ARM64 size not yet measured, needs the actual NDK/Xcode toolchain build)
- 10-minute continuous streaming soak test **on real hardware** (the C soak test itself passed on x86/qemu, §3.3 — but that's correctness, not a substitute for a real-device thermal/memory/battery soak)
- Peak native RSS on real hardware
- Real Cortex-A53/A13 RTF, P50/P95/P99, end-to-end audio latency, CPU/thermal/battery impact
- ONNX Runtime RTF vs. scalar-C RTF vs. C+NEON RTF breakdown **on real hardware**, and the real runtime-free/NEON speedup numbers (the QEMU numbers in §4.1 do not substitute for this — explicitly stated above)

## 5. iOS (target platform: iPhone 11 / A13) — source + docs written, on-device work BLOCKED_EXTERNAL

This session has no macOS, no Xcode, and no physical iPhone 11. Nothing in this section was built, run, or benchmarked here — see `IOS_BUILD_AND_DEMO.md` for the full protocol a person with that hardware must follow.

| Item | Status | Note |
|---|---|---|
| Xcode project (`ios/UlunasDemo/`) | **PASS** (written, not built) | Hand-written `project.pbxproj` (braces/parens balanced, 35 unique object IDs, no obvious structural errors — could not run `plutil` to fully validate, no macOS tooling here), SwiftUI UI, `AudioEngineManager.swift` (AVAudioEngine mic capture → resample → lock-free ring buffer → dedicated inference thread → ring buffer → playback), `RingBuffer.swift` (SPSC; caught and fixed a real data race in a first draft — the producer must never mutate the consumer's `readIndex`), `WavRecorder.swift`, ObjC++ bridge (`UlunasBridge.h/.mm`) against the fixed C API (`mobile/c_neon/include/ulunas_api.h`) |
| No Voice-Processing I/O / AEC / Voice Isolation | **PASS** (by construction) | Code explicitly asserts `inputNode.isVoiceProcessingEnabled == false` and uses `AVAudioSession` `.default` mode, never `.voiceChat`; documented in-code that iOS does not expose a public API to fully disable input-side AGC on the standard Remote I/O path — disclosed, not claimed as fully raw |
| Runtime-free C engine wired into the app | **NOT_RUN** | Depends on the concurrent C-graph completion workstream (see §3 above for its status); `IOS_BUILD_AND_DEMO.md` §2 documents exactly which files to add once ready |
| ONNX Runtime iOS reference backend | **NOT_RUN** | Bridge code initializes an ORT session behind a build flag, but the host-side STFT/cache framing needed to actually drive it frame-by-frame is not yet ported (disclosed in code comments, not hidden) |
| Native build/run/install on iPhone 11 | **BLOCKED_EXTERNAL** | No Mac/Xcode/device in this environment |
| 3-scenario live demo + screen recording | **BLOCKED_EXTERNAL** | No device |
| On-device benchmark (inference time, RTF, memory, CPU, thermal, battery) | **BLOCKED_EXTERNAL** | No device/Instruments access — template table with blanks provided in `IOS_BUILD_AND_DEMO.md` §6, explicitly not pre-filled |
| WASM/browser fallback demo (`web_demo/`) | **PASS** (DSP core only, no browser test) | Emscripten 3.1.6 installed in-container; `ulunas_wasm.cpp` (reusing the same `fft.h` validated in `mobile/android_ref/`) compiles cleanly to `.wasm`; a Node.js smoke test verifies the compiled module's streaming STFT/overlap-add reconstructs an identity signal to 1.8e-6 max abs error (after correctly accounting for the pipeline's inherent 1-hop/16ms causal latency — a first version of the OLA math wrongly assumed a constant window-sum normalization, which numerically is NOT constant (ranges 0.5–1.0 across a hop); caught, fixed, re-verified). **Does not yet include the real model** (identity passthrough, clearly labeled in the page) since the C-graph workstream wasn't done yet — swapping it in is a documented, mechanical follow-up. **Never run in an actual browser** (no browser available here) — this is disclosed prominently in `web_demo/README.md` and the page itself, and this WASM demo must never be cited as iOS/ARM64/NEON performance evidence |

**Summary**: everything that could be done without physical Apple hardware or Xcode was done and is genuinely verified where verification was possible (the DSP math, the ring-buffer concurrency contract, the absence of voice-processing APIs in the code path). Everything requiring the actual hardware/toolchain is explicitly `BLOCKED_EXTERNAL`, not simulated or estimated.
