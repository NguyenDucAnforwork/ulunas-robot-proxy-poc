# EXPERIMENT_SWEEP.md — 90-minute train + inference optimization run (2026-09-13)

Autonomous run to (a) improve the robot-proxy UL-UNAS model with a small parallel
training sweep on one idle A100-80GB and (b) land a real, measured inference
optimization. Data-download time excluded from the 90-min compute budget by request.
Everything below is grounded in measurements taken **on this server this session** —
where a number is an estimate or from a different machine it says so.

> Read alongside: `EXPERIMENT_REPORT.md` (the RQ1/RQ2/RQ3 baseline), `MOBILE_BENCHMARK.md`
> (runtime-free C core), `mobile/c_neon/BN_FOLD_RESULTS.md` (inference-opt detail).

---

## 0. TL;DR

- **GPU reality (measured):** the model is *tiny* (171,328 params) and **launch/latency-bound**,
  not FLOP-bound. A single batch-8 training step = **79 ms = 12.6 steps/s**, using only
  **~930 MB / 45 % util** of an A100-80GB. Naive parallel processes just time-slice
  (4 jobs → 5.4 steps/s each, agg capped ~22 steps/s). **CUDA MPS is the unlock:**
  4 concurrent jobs → **11.8 steps/s each** (≈full single-job speed), 8 jobs → 8.7 each.
  So the "maximize GPU" answer here is *many concurrent small jobs under MPS*, not big batches.
- **Training sweep:** 4 arms, batch 8, seed 43, DNS3 init, 18k steps each, run **concurrently
  under MPS** in ~30 min total: `control`, `lr_sched` (warmup+cosine), `preserve`
  (speech-preservation loss targeting the SIG over-suppression regression), `snr_curriculum`.
  **Fast SI-SDR proxy (§4.1):** no arm beats control. **Full 150-mixture DNSMOS verdict
  (§4.2, added in the follow-up session):** `lr_sched`'s *best* checkpoint robustly beats
  control on every metric (SIG/BAK/OVRL/SI-SDR/STOI, all sub-material) — the opposite of
  what the final-step proxy suggested, because checkpoint selection (best-val vs final-step)
  changes the ranking. `preserve` makes exactly the SIG-for-BAK trade it was designed to
  make (+0.0075 SIG, −0.0113 BAK, both robust) but an order of magnitude too small to close
  the −0.02 to −0.03 SIG regression RQ3 found. `snr_curriculum` is neutral. No material win
  for any arm in 18k steps on this data build.
- **Inference opt (done, measured):** **BatchNorm folding** into preceding convs — opt-in
  `-DULUNAS_BN_FOLDED`, numerically parity-safe (1.04e-7 spectral / 6.68e-6 PCM vs reference),
  **−6.65 % x86 hop latency**. x86-host only, *not* an ARM/iPhone number. See
  `mobile/c_neon/BN_FOLD_RESULTS.md`.

---

## 1. Hardware & measured throughput (this server)

`NVIDIA A100-SXM4-80GB`, torch 2.11.0+cu128, 12 vCPU.

| Config | steps/s per job | aggregate steps/s | samples/s | VRAM/job |
|---|---|---|---|---|
| 1× batch 8 | 12.6 | 12.6 | 103 | 930 MB |
| 1× batch 16 | 12.1 | 12.1 | 193 | 1808 MB |
| 1× batch 32 | 7.8 | 7.8 | 248 | 3586 MB |
| 1× batch 128 | 2.3 | 2.3 | 290 (ceiling) | 14236 MB |
| 4× batch 8, **no MPS** | 5.4 | 21.8 | 176 | 930 MB |
| 4× batch 8, **MPS** | **11.8** | **47** | **377** | 930 MB |
| 8× batch 8, **MPS** | 8.7 | 70 | 560 | 930 MB |

**Why:** 171 k params + many *small, sequential* GRU/conv ops ⇒ per-kernel launch latency
dominates; the A100's SMs are mostly idle. Big batch raises samples/s but saturates ~290
(single-stream ceiling). Separate CUDA contexts time-slice without MPS (no true concurrency).
**MPS** lets kernels from the 4 arms interleave on idle SMs, so each arm runs at ~full speed —
the genuine way to "use the whole GPU" for this workload. This is itself an interview-worthy
finding: for ultra-light streaming models, *concurrency*, not batch size, unlocks the GPU.

---

## 2. Data pipeline (rebuilt this session; deviations documented)

`data/` is gitignored, so the domain data was rebuilt from scratch:

- **Clean speech:** LibriSpeech dev-clean (openslr) — reliable, full split
  (train 1983 / val 157 / test 563 utterances), same speaker split as the original.
- **UAV ego-noise:** KU Leuven (DOI `10.48804/PZAVUC`), datafiles 706/702/699 →
  resampled to 16 kHz mono → `train_motor_low` / `dev_motor_mid` / `test_motor_high`. OK.
- **Procedural motor/fan/servo noise:** `procedural_noise.py`, seeded — 77 clips. OK.
- **Generic (DEMAND) noise — DEVIATION:** Zenodo (record 1227121) **504-Gateway-Timed-out
  on 10 of 12 `*_16k.zip`** this session; only DKITCHEN + OOFFICE downloaded. Rather than
  stall, each was **time-sliced 70/15/15 into train/val/test** (`make_manifests.py`) —
  disjoint time regions ⇒ **no train/val/test leakage** (leakage check passes). Consequence:
  lower generic-noise diversity than the original 12-env split. **All 4 arms share this
  identical data**, so the arm-vs-arm comparison is unaffected; only comparison to the
  *published* F_PROXY_ROBOT absolute numbers is (do not compare across the two data builds).

Repro-fixes made to the committed scripts: single-level DEMAND extraction path + tolerance
for missing DEMAND/UAV envs (`split_manifest.py`, `build_robot_proxy_manifest.py`); a
`/content/project` → repo symlink resolves the code's hardcoded paths.

---

## 3. Training sweep design (Workstream B)

Baseline held fixed (= F_PROXY_ROBOT): DNS3 init, `HybridLoss(lamda_ri=30, lamda_mag=70,
lamda_sisnr=1.0, compress=0.3)`, Adam lr 1e-5, batch 8, 4 s segments, seed 43, grad-clip 3,
22 BN frozen, noise mix 50/35/15, SNR U[-5,15]. One script `SEtrain_adapted/train_arm.py`,
one deliberate delta per arm:

| Arm | Delta vs baseline | Hypothesis / target |
|---|---|---|
| `control` | none (reference at this data/depth) | anchor |
| `lr_sched` | warmup 500 steps 1e-6→1e-5, then cosine → 1e-6 | does constant-LR leave convergence on the table at equal steps? |
| `preserve` | `lamda_sisnr` 1.0→0.3 **+** speech-preservation penalty (see below) | fix the **SIG over-suppression regression** (RQ3's core quality problem) |
| `snr_curriculum` | first 6k steps SNR U[5,15] (easy) → then U[-5,15] | improve SIG / deletions at low SNR |

**Speech-preservation loss** (`preserve` arm): `mean(relu(|clean|^0.3 − |enh|^0.3)^2)` over the
STFT — penalizes enhanced compressed-magnitude falling *below* clean (over-suppression) but not
exceeding it. Weight 20 (calibratable). Directly discourages deleting speech energy.

**Dropped** (justified in the plan): longer-than-baseline training (infeasible <90 min),
ASR/CTC loss (proven dead-end, RQ2), GAN/DNSMOS-direct (too risky for the window).

All four run **concurrently under MPS**, batch 8, 18k steps, ~30 min wall total.
Selection metric = **best validation SE-loss** (+ SI-SDR); full DNSMOS/WER is a slower,
separate confirmation (`eval_robot_proxy.py`) noted as follow-up.

Reproduce: `python3 train_arm.py --arm <arm> --steps 18000 --out_tag arm_<arm>` (see §6).

---

## 4. Sweep results

All 4 arms ran **concurrently under MPS** to 18k steps in **~32 min total** (9.2–9.4
steps/s each; 941 MB VRAM each; budget used 07:37→08:10). No NaN/Inf.

### 4.1 Proxy metric — validation SI-SDR (comparable across arms)

| Arm | val SI-SDR | Δ vs control |
|---|---|---|
| **arm_control** | **18.512 dB** | — |
| arm_snr_curriculum | 18.511 dB | −0.001 (tie) |
| arm_lr_sched | 18.450 dB | −0.062 |
| arm_preserve | 18.415 dB | −0.097 |

> **Do not compare arms by validation SE-loss:** `arm_preserve` uses a different loss
> (`lamda_sisnr` 0.3 + preservation term), so its SE-loss (1.53) is on a different scale
> than the others' (~0.23). SI-SDR is the common, comparable proxy.

**Reading:** on the fast SI-SDR proxy at 18k steps, **no intervention beats control.**
`snr_curriculum` ties it; `lr_sched` and `preserve` are marginally *lower* on SI-SDR
(−0.06 / −0.10 dB). Crucially, **SI-SDR is exactly the waveform-fidelity quantity the
`preserve` arm deliberately trades away** to keep speech magnitude — so a small SI-SDR
dip is *consistent with* its hypothesis, not a refutation. The `preserve` question can
only be answered by **DNSMOS SIG** (§4.2). `lr_sched`'s cosine decay to 1e-6 starves the
last steps of learning rate, so at equal 18k steps it slightly trails constant-LR — i.e.
the schedule did not help convergence within this budget.

### 4.2 Perceptual verdict — reduced DNSMOS eval (150 paired test mixtures)

DNSMOS P.835 SIG/BAK/OVRL + SI-SDR + STOI, same 150 deterministic robot-proxy test
mixtures for every arm (paired), each arm's **`_best.tar`** checkpoint (selected by
validation SE-loss during training, per §3). WER (wav2vec2) skipped for the time
window — noted as follow-up. Script: `SEtrain_adapted/eval_arms_dnsmos.py`, full
per-utterance output in `evaluation/eval_arms_dnsmos_results.json`.

| Arm | SIG | BAK | OVRL | SI-SDR (dB) | STOI |
|---|---|---|---|---|---|
| arm_control | 3.145 | 3.913 | 2.867 | 17.589 | 0.9314 |
| arm_lr_sched | 3.155 | 3.929 | 2.881 | 17.803 | 0.9326 |
| arm_preserve | 3.153 | 3.902 | 2.869 | 17.593 | 0.9325 |
| arm_snr_curriculum | 3.146 | 3.915 | 2.869 | 17.589 | 0.9316 |

**Paired bootstrap CI95% (Δ = arm − control, n=150, 2000 resamples), materiality
threshold |ΔSIG|/|ΔOVRL| > 0.03 per the project's own established convention
(`EXPERIMENT_REPORT.md`):**

| Arm | ΔSIG | ΔBAK | ΔOVRL | ΔSI-SDR | ΔSTOI |
|---|---|---|---|---|---|
| lr_sched | +0.0098 [+0.0011,+0.0188] ✅robust | +0.0152 [+0.0075,+0.0238] ✅robust | +0.0141 [+0.0064,+0.0224] ✅robust | +0.214dB [+0.148,+0.291] ✅robust | +0.0012 [+0.0007,+0.0018] ✅robust |
| preserve | +0.0075 [+0.0024,+0.0132] ✅robust | **−0.0113 [−0.0159,−0.0068]** ✅robust(worse) | +0.0016 [−0.0027,+0.0062] not robust | +0.0042 [−0.0282,+0.0388] not robust | +0.0011 [+0.0008,+0.0014] ✅robust |
| snr_curriculum | +0.0011 [−0.0020,+0.0043] not robust | +0.0015 [−0.0017,+0.0045] not robust | +0.0018 [−0.0010,+0.0046] not robust | −0.0005 [−0.0289,+0.0298] not robust | +0.0002 [+0.0001,+0.0004] ✅robust(trivial) |

All deltas above are **statistically robust but sub-material** (< 0.03 DNSMOS) — none
would clear this project's own pre-registered materiality bar. Reading each arm honestly:

- **`preserve` hypothesis is directionally confirmed, but too small to matter.** SIG
  improves (+0.0075, robust) with a BAK cost (−0.0113, robust) — exactly the SIG-for-BAK
  trade the speech-preservation loss was designed to make (§3). But the magnitude is an
  order smaller than what needs fixing: RQ3's own QC found `F_PROXY_ROBOT` vs `F_GENERIC`
  SIG regression of −0.021 to −0.034 (worst cases down to −0.34 to −0.97, `QC_FAILURE_ANALYSIS.md`
  §4). A weight-20 preservation penalty nudges SIG back by +0.0075 — real, but nowhere near
  enough to close that gap. OVRL is a wash (CI crosses 0): the SIG gain and BAK loss roughly
  cancel in the composite score.
- **`lr_sched` actually wins on every metric here — the opposite of §4.1's read.** The
  fast SI-SDR proxy in §4.1 compared *final-step* (18k) validation numbers on a 9-item
  val set and found `lr_sched` slightly behind control. This full 150-item test-set eval
  uses each arm's **best-validation checkpoint**, not the final step — and `lr_sched`'s
  best checkpoint (likely earlier than step 18000, since its LR has already decayed by
  then) generalizes better: robust gains on SIG/BAK/OVRL/SI-SDR/STOI, all still sub-material
  but consistently positive. **Lesson for next time:** a fast same-step proxy and a
  best-checkpoint full eval are not measuring the same thing — checkpoint selection matters
  more than the 18k-step endpoint comparison suggested.
- **`snr_curriculum` is confirmed neutral.** Every CI crosses (or nearly crosses) zero —
  no detectable effect from the easy→hard SNR schedule at this step budget, consistent
  with the §4.1 SI-SDR tie.

**Overall verdict for the sweep:** no arm produces a material win over `control` in 18k
steps on this (reduced, single-session) data build. `lr_sched` is the best all-around
candidate for a longer run (robust, uniformly positive, cheap to keep). `preserve` is the
right *mechanism* for the SIG regression but needs a substantially higher penalty weight
(or more steps) to reach material size — a natural next ablation, not a dead end.

---

## 5. Inference optimization (Workstream A)

**BatchNorm folding — done and measured** (`mobile/c_neon/BN_FOLD_RESULTS.md`):
opt-in `-DULUNAS_BN_FOLDED`; 22 frozen BNs folded into the preceding conv at export
(`export_weights.py --fold-bn`). Parity vs reference **1.04e-7** (spectral) / **6.68e-6**
(PCM), edge-cases pass both builds, reset-determinism bit-exact. **x86 hop latency
0.942 → 0.879 ms (−6.65 %)**, RTF 0.0589 → 0.0550. **x86-host only — NOT an ARM/iPhone
number.** Payoff is modest by design (BN is a small compute fraction), but it is a real,
parity-safe, classic deployment win with the default build left untouched.

Higher-payoff inference levers (NEON on grouped Conv/ConvTranspose, fp16-SIMD GRU) require
**real ARM silicon** to measure meaningfully (x86 has no NEON; qemu timing is not predictive).
Recommended next step: AWS Graviton for a real-ARM profile, then EC2 Mac + Device Farm to
unblock the iPhone build/benchmark — see the Codex plan in the session history.

---

## 6. Reproduce

```bash
# 0. paths + data (see §2). /content/project is a symlink to the repo.
python3 SEtrain_adapted/make_manifests.py            # speech + time-sliced generic noise
python3 SEtrain_adapted/build_robot_proxy_manifest.py # + UAV + procedural

# 1. MPS (the key to using the GPU) + 4 concurrent arms
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log
nvidia-cuda-mps-control -d
for arm in control lr_sched preserve snr_curriculum; do
  python3 SEtrain_adapted/train_arm.py --arm $arm --steps 18000 --out_tag arm_$arm &
done; wait
python3 SEtrain_adapted/compare_arms.py            # ranked proxy-metric table

# 2. inference opt
./mobile/c_neon/measure_bn_fold.sh                 # parity + x86 latency before/after
```

---

## 7. Lessons learned (for reproduction & interview)

1. **Measure the bottleneck before "optimizing the GPU."** The instinct is bigger batches;
   the measurement said *launch-bound*, and **MPS + concurrency** (not batch size) was the
   real 2–4× lever. A 171 k-param streaming model cannot saturate an A100 by batch alone.
2. **MPS turns an idle A100 into a 4-way experiment farm** for tiny models: 4 arms at
   ~full single-job speed simultaneously, ~free. This is what made a 4-arm sweep fit in ~30 min.
3. **Data-download is the real risk, not compute.** Zenodo 504'd on 10/12 DEMAND files;
   a time-sliced 2-env fallback (leakage-free) kept the run alive. Always have a degrade path,
   and keep the comparison *internally* fair even when absolute realism drops.
4. **Keep the comparison honest under substitution:** all arms share the exact same
   (reduced) data, so arm-vs-arm conclusions hold; only cross-build (vs published) comparison
   is invalidated — state that explicitly rather than quietly comparing.
5. **Fold what's frozen.** 22 frozen BNs → fold into conv for a free, parity-safe inference
   win; the honest payoff is small (−6.65 % x86) because BN is a small compute slice — report
   the real number, don't inflate it, and never call an x86 number an ARM number.
6. **Parallelize the human, too:** the CPU-only inference track (BN-fold) ran on a Codex
   subagent while the GPU training track ran here — independent subtrees, no coordination cost.
7. **Proxy metrics for fast screens, full metrics for verdicts:** DNSMOS/WER eval is CPU-bound
   and slow; validation SE-loss + SI-SDR gives a same-window directional signal, with the
   perceptual verdict deferred — but note SE-loss won't fully capture the SIG effect the
   `preserve` arm targets, so treat its screen as necessary-not-sufficient. **Confirmed by
   §4.2:** the full DNSMOS eval flipped `lr_sched`'s ranking relative to the fast proxy,
   because the proxy compared final-step (18k) validation numbers while the full eval used
   each arm's best-validation checkpoint — a different point in training. When a proxy and
   a full eval disagree, check whether they're even scoring the same checkpoint before
   trusting either.
8. **A mechanism can be real and still not material.** `preserve`'s SIG-for-BAK trade
   showed up exactly as designed (both deltas robust/CI-excludes-0) but at 1/3 the size of
   the regression it was meant to fix. Confirming a mechanism direction is not the same as
   confirming it's big enough to ship — report the ratio, not just the sign.
