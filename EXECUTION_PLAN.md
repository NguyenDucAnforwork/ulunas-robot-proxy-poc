# EXECUTION_PLAN.md — Takeover plan for robot-proxy + full runtime-free C + iOS

Verified against real artifacts on disk before writing this (not just prior reports):
`checkpoints/{F_GENERIC,F_GENERIC_ASR}.tar` exist (809KB/810KB, timestamps match training_logs), `evaluation/evaluation_results.{json,csv}` exist with real numbers matching `EXPERIMENT_REPORT.md`, `training_logs/{F_GENERIC,F_GENERIC_ASR}/*.json` exist with the resource/lambda/grad-check numbers cited in the report. **No discrepancy found between reports and artifacts.** GPU confirmed idle (0% util, 80GB free) at takeover time 2026-09-12 14:26 UTC.

## Status of every requirement in the new brief

| # | Item | Status | Note |
|---|---|---|---|
| 1 | M1 results (F_GENERIC/F_GENERIC_ASR train+eval) | **PASS** | Verified real, preserved untouched |
| 2 | ONNX parity, Android NDK build | **PASS** | Verified real (`mobile/onnx_export/out/parity_results.json`, `mobile/android_ref/build_arm64/ulunas_ref` ELF) |
| 3 | Individual C kernels + block patterns validated | **PASS** | `mobile/c_neon/build/test_{block0,block1,block2,dpgrnn,decoder4}` all pass |
| 4 | Full 697-node runtime-free C graph | **PENDING → executing now** | Only isolated blocks done; full wiring not started |
| 5 | Robot-proxy dataset (public ego-noise + procedural) | **PENDING → executing now** | Renamed F_ROBOT→F_PROXY_ROBOT per instruction #3, nothing trained yet under either name |
| 6 | F_PROXY_ROBOT / F_PROXY_ROBOT_ASR training | **PENDING → executing now** | Budget-constrained, pilot first |
| 7 | RQ1/RQ2/RQ3 full 6-condition eval | **PENDING → executing now** | RQ1/RQ2 partially answered in M1 (generic only); RQ3 needs proxy models |
| 8 | QC failure analysis with real cases | **PENDING → executing now** | Was schema-only/BLOCKED in M1 since no eval existed yet for the relevant conditions; now has real data to analyze |
| 9 | iOS native app (Swift/ObjC++/C) | **PENDING → executing now** | Source + build docs will be written; **actual build/run/benchmark is BLOCKED_EXTERNAL — no macOS/Xcode/physical iPhone 11 in this environment** |
| 10 | Web/WASM fallback demo | **PENDING → attempting** | Buildable without Xcode; testing on a real iPhone Safari is still BLOCKED_EXTERNAL |
| 11 | Real iPhone 11 benchmark, demo video | **BLOCKED_EXTERNAL** | No physical device, no macOS — stated upfront, not discovered later |
| 12 | PROJECT_STATUS.md | **PENDING → written last** | After all above settle |

## Environment constraints discovered (stated now, not hidden)

- This session runs in a Linux Colab-style container (confirmed via `uname`/no Xcode/no macOS tooling). **There is no path to actually compiling, signing, or running an iOS app, nor to physically benchmarking an iPhone 11, from here.** Per the brief's own instructions (§9, §11), this is handled by: writing complete iOS source + build instructions, and marking on-device build/run/benchmark/demo-video as `BLOCKED_EXTERNAL` — not fabricating results. This is flagged now, upfront, per "chỉ dừng để hỏi... khi gặp hard external blocker" — this is exactly that blocker, already resolved by the brief's own fallback instructions, so I'm proceeding without stopping to ask.
- GPU budget: treating "3-4h" as the budget for *this* takeover's new GPU work (robot-proxy training), separate from what M1 already used. Will time-box accordingly and report actual usage, not assume.

## Execution order (this session)

1. Dispatch two parallel forks (no GPU needed, independent of the training below):
   - Fork A: complete the full 697-node runtime-free C graph (extends the already-validated kernel library).
   - Fork B: write the iOS Xcode project source (Swift + ObjC++/C bridge), `IOS_BUILD_AND_DEMO.md`, and attempt a WASM fallback build.
2. Main thread (GPU-critical path): source/verify license for public ego-noise data (KU Leuven UAV, Robovox), build procedural motor/fan/servo noise generator, build robot-proxy mixture manifests (no leakage), run a pilot to size the step budget, train `F_PROXY_ROBOT` and `F_PROXY_ROBOT_ASR` (same step count, enforced).
3. Run the full 6-condition evaluation (N0, P0, F_GENERIC, F_GENERIC_ASR, F_PROXY_ROBOT, F_PROXY_ROBOT_ASR) with paired bootstrap CIs, answer RQ1/RQ2/RQ3 with real numbers.
4. QC failure analysis on the real per-utterance results (best/worst 10, SIG-BAK tradeoff investigation).
5. Merge fork results, finalize `MOBILE_BENCHMARK.md`, `IOS_BUILD_AND_DEMO.md`, `ROBOT_PROXY_DATA.md`, `DATA_LICENSES.md`.
6. Write final `PROJECT_STATUS.md` with the required Requirement/Evidence/Status/Limitation table, `PASS`/`FAIL`/`BLOCKED_EXTERNAL`/`NOT_RUN` only.

No stopping for routine technical choices; decisions made along the way are logged in the relevant doc with a one-line rationale, per instruction.
