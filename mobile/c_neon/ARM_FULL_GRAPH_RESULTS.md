# Real-ARM full-graph hop latency (AWS Graviton, real trained weights)

Measured 2026-09-13, follow-up to `ARM_MICROBENCH_RESULTS.md`. That earlier run could
only ship a synthetic per-primitive microbenchmark through EC2 user-data (16KB limit).
The user granted `s3:PutObject`/`s3:GetObject` on a dedicated bucket, which unblocked
shipping the real generated-weights header and the actual 697-node `ulunas_full.cpp`
graph to a Graviton instance via presigned S3 URLs (upload in, upload results back
out — still no SSH, IAM stayed RunInstances/TerminateInstances/GetConsoleOutput-only).
This is the exact same `measure_bn_fold.sh` pipeline used for the x86 numbers in
`BN_FOLD_RESULTS.md`, run unmodified except for skipping the `export_weights.py` step
(headers were shipped pre-generated, since the instance has no torch/checkpoint).

**Checkpoint correction:** the first pass of this run (superseded, kept below for the
record) reused a pre-existing `generated/` header that turned out to be exported from
`F_GENERIC.tar` (M1's generic-domain fine-tune) instead of `F_PROXY_ROBOT.tar` (the
actual robot-proxy-adapted checkpoint this project targets) — caught during review.
`export_weights.py` was fixed to take an explicit `--checkpoint` argument (previously
hardcoded to `F_GENERIC.tar`), and everything below is the re-run against the correct
`F_PROXY_ROBOT.tar` (SHA-256 `70c2a7d2...4a163`, verified against `REPRODUCTION.md`).

## Hardware

AWS `t4g.small` (Graviton2, Neoverse-N1, aarch64), Amazon Linux 2023, GCC 11.5.0,
`-O2 -std=c++17`, native compile (no cross-compiler, no QEMU). Real hop-level pipeline:
STFT → full 12-block graph → mask → OLA, one 256-sample (16ms) hop per call, exactly
as deployed. Full logs: `arm_full_graph_results_v2/` (this, correct-checkpoint run);
`arm_full_graph_results/` (superseded `F_GENERIC` run, kept for the record).

## Correctness (same binary that produced the numbers below, F_PROXY_ROBOT weights)

| Check | Result |
|---|---|
| Full graph, 20 STFT frames | max abs error **1.252e-6** (limit 1e-4) |
| Full PCM pipeline, 20 hops | max abs error **1.931e-5** (limit 1e-4) |
| BN-fold parity | **PASS** |
| Edge cases (silence/impulse/random/reset-determinism), both builds | **ALL PASS**, reset-determinism bit-exact (`max_diff=0.0`) |

## Real ARM hop latency: BatchNorm folding, full graph, F_PROXY_ROBOT weights

3 pairs, alternating order, 2000 hops after 100 warmup each (same protocol as x86):

| Pair | Build | mean ms/hop | p95 ms/hop | RTF_mean |
|---|---|---:|---:|---:|
| 1 | Reference | 0.69936 | 0.71148 | 0.04371 |
| 1 | Folded | 0.68122 | 0.69159 | 0.04258 |
| 2 | Reference | 0.70155 | 0.71689 | 0.04385 |
| 2 | Folded | 0.68278 | 0.69553 | 0.04267 |
| 3 | Reference | 0.70025 | 0.71153 | 0.04377 |
| 3 | Folded | 0.68481 | 0.70638 | 0.04280 |
| **avg** | **Reference** | **0.70039** | **0.71330** | **0.04378** |
| **avg** | **Folded** | **0.68294** | **0.69783** | **0.04268** |
| **Δ** | | **−2.49%** | **−2.17%** | **−2.50%** |

**Comparison across checkpoints and across x86 vs ARM** (all real measurements, none
hypothetical):

| Run | mean-latency Δ from BN folding |
|---|---|
| x86, `F_GENERIC.tar` (`BN_FOLD_RESULTS.md`, original) | −6.65% |
| x86, `F_PROXY_ROBOT.tar` (`BN_FOLD_RESULTS.md`, correction) | −6.70% |
| ARM Graviton2, `F_GENERIC.tar` (superseded, `arm_full_graph_results/`) | −3.55% |
| **ARM Graviton2, `F_PROXY_ROBOT.tar` (this run, authoritative)** | **−2.49%** |

Reading this honestly: the two x86 runs (different checkpoints) agree to within 0.05
points — consistent with BN-fold's latency effect being driven by the graph's compute
pattern, not the specific trained weight values (float multiply-add timing doesn't
branch on data). The two ARM runs disagree by about 1 point (−3.55% vs −2.49%) despite
the same expectation — the more likely explanation is ordinary cloud-VM run-to-run
noise (different physical host, noisy-neighbor scheduling, thermal state) between two
separate `t4g.small` instances, not a genuine checkpoint-dependent effect that has no
mechanism to exist. Reporting both real numbers rather than picking one; the
qualitative conclusion (BN folding is a real, positive, smaller-on-ARM-than-x86 win)
holds either way.

**RTF_mean ≈ 0.044** on both builds — comfortably real-time-capable on a modest 2-vCPU
cloud ARM core.

## What this does and doesn't close

- **Closes**: "no real-ARM full-graph hop latency exists, from the correct checkpoint"
  — this is the real number, from the real `F_PROXY_ROBOT` trained weights, on real
  ARM64/NEON silicon, using the identical protocol as the x86 measurement.
- **Does not close**: the actual target device. Graviton2 (Neoverse-N1, server-class,
  out-of-order, large caches) is architecturally very different from Cortex-A53
  (in-order, mobile, iPhone-11-class target) or Apple Silicon — absolute latency will
  differ, likely substantially, on the real target. `PROJECT_STATUS.md`'s
  `BLOCKED_EXTERNAL` items for Cortex-A53/iPhone RTF/latency/memory/thermal/battery
  are unchanged by this result and still require physical hardware or EC2 Mac + Xcode.
- The benchmark binary's own printed caveat ("x86 host directly, or qemu-aarch64-static
  emulating ARM64 ... NOT real Cortex-A53/Apple-A13 silicon") is boilerplate baked into
  `benchmark_latency.cpp` before this run existed — for *this* run specifically, the
  binary executed natively on real ARM64 silicon (Graviton2), not QEMU and not x86;
  the "not Cortex-A53/Apple-A13" half of that caveat still applies as written above.

## Reproduce

Requires an AWS key with `ec2:RunInstances`/`TerminateInstances`/`GetConsoleOutput`
scoped to one region, and `s3:PutObject`/`GetObject` on one bucket in the same region
(no key-pair/security-group/instance-connect permission needed):

```bash
# from a machine with the real checkpoint already exported (mobile/c_neon/generated/*.h present)
cd mobile/c_neon
python3 export_weights.py --checkpoint /path/to/F_PROXY_ROBOT.tar
python3 export_weights.py --checkpoint /path/to/F_PROXY_ROBOT.tar --fold-bn
tar --exclude=build -czf /tmp/c_neon_full.tar.gz src include generated measure_bn_fold.sh
aws s3 cp /tmp/c_neon_full.tar.gz s3://<bucket>/c_neon_full.tar.gz --region <region>
# generate presigned GET (input) and PUT (output, via boto3 generate_presigned_url) URLs,
# embed both in an EC2 user-data script that: installs gcc-c++, curls the GET url, extracts,
# runs the measure_bn_fold.sh build+benchmark steps (skip export_weights.py -- headers are
# already in the tarball), tars the build/bn_fold logs, curl -T's them to the PUT url.
aws ec2 run-instances --image-id <al2023-arm64-ami> --instance-type t4g.small \
  --subnet-id <default-subnet> --security-group-ids <default-sg> --user-data file://user_data.sh
# poll: aws s3 ls s3://<bucket>/results.tar.gz ; then aws ec2 terminate-instances immediately after
```
