# Real-ARM full-graph hop latency (AWS Graviton, real trained weights)

Measured 2026-09-13, follow-up to `ARM_MICROBENCH_RESULTS.md`. That earlier run could
only ship a synthetic per-primitive microbenchmark through EC2 user-data (16KB limit).
The user granted `s3:PutObject`/`s3:GetObject` on a dedicated bucket, which unblocked
shipping the **real** 4.7MB generated-weights header (from the real `F_GENERIC.tar`
checkpoint) and the actual 697-node `ulunas_full.cpp` graph to a second Graviton
instance via presigned S3 URLs (upload in, upload results back out — still no SSH,
IAM stayed RunInstances/TerminateInstances/GetConsoleOutput-only). This is the exact
same `measure_bn_fold.sh` pipeline used for the x86 numbers in `BN_FOLD_RESULTS.md`,
run unmodified except for skipping the `export_weights.py` step (headers were shipped
pre-generated, since the instance has no torch/checkpoint).

## Hardware

AWS `t4g.small` (Graviton2, Neoverse-N1, aarch64), Amazon Linux 2023, GCC 11.5.0,
`-O2 -std=c++17`, native compile (no cross-compiler, no QEMU). Real hop-level pipeline:
STFT → full 12-block graph → mask → OLA, one 256-sample (16ms) hop per call, exactly
as deployed. Full logs: `arm_full_graph_results/`.

## Correctness (same binary that produced the numbers below)

| Check | Result |
|---|---|
| Full graph, 20 STFT frames | max abs error **1.006e-7** (limit 1e-4) |
| Full PCM pipeline, 20 hops | max abs error **2.384e-5** (limit 1e-4) |
| BN-fold parity | **PASS** |
| Edge cases (silence/impulse/random/reset-determinism), both builds | **ALL PASS**, reset-determinism bit-exact (`max_diff=0.0`) |

## Real ARM hop latency: BatchNorm folding, full graph

3 pairs, alternating order, 2000 hops after 100 warmup each (same protocol as the x86 run):

| Pair | Build | mean ms/hop | p95 ms/hop | RTF_mean |
|---|---|---:|---:|---:|
| 1 | Reference | 0.69784 | 0.70801 | 0.04362 |
| 1 | Folded | 0.68036 | 0.69306 | 0.04252 |
| 2 | Reference | 0.73449 | 0.78461 | 0.04591 |
| 2 | Folded | 0.68147 | 0.69245 | 0.04259 |
| 3 | Reference | 0.69894 | 0.70971 | 0.04368 |
| 3 | Folded | 0.69381 | 0.69258 | 0.04336 |
| **avg** | **Reference** | **0.71042** | **0.73411** | **0.04440** |
| **avg** | **Folded** | **0.68521** | **0.69270** | **0.04282** |
| **Δ** | | **−3.55%** | **−5.64%** | **−3.56%** |

**This is a real ARM number, and it is smaller than the x86 figure.** x86 host showed
−6.65% mean-latency; real Graviton2/Neoverse-N1 shows **−3.55%**. Both are real,
neither is wrong — they're different CPUs with different relative costs for BN's
divide+multiply-add vs. the surrounding conv/GRU work, and this is exactly why the
project's own docs insisted the x86 number could not be called an ARM number. The
honest reading: BN folding is a real, small, parity-safe win on ARM too, just smaller
than on this particular x86 host — report the number measured, not the one hoped for.

**RTF_mean ≈ 0.043–0.046** on both builds means the model uses well under 5% of one
hop's real-time budget on a modest 2-vCPU cloud ARM core — comfortably real-time
capable (`RTF < 1`) with large headroom, though this is Graviton2 server silicon, not
the Cortex-A53/iPhone-class mobile target (see caveats).

## What this does and doesn't close

- **Closes**: "no real-ARM full-graph hop latency exists" — this is the real number,
  from the real trained checkpoint's weights, on real ARM64/NEON silicon, using the
  identical protocol as the x86 measurement.
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
cd mobile/c_neon && tar --exclude=build -czf /tmp/c_neon_full.tar.gz src include generated measure_bn_fold.sh
aws s3 cp /tmp/c_neon_full.tar.gz s3://<bucket>/c_neon_full.tar.gz --region <region>
# generate presigned GET (input) and PUT (output, via boto3 generate_presigned_url) URLs,
# embed both in an EC2 user-data script that: installs gcc-c++, curls the GET url, extracts,
# runs the measure_bn_fold.sh build+benchmark steps (skip export_weights.py -- headers are
# already in the tarball), tars the build/bn_fold logs, curl -T's them to the PUT url.
aws ec2 run-instances --image-id <al2023-arm64-ami> --instance-type t4g.small \
  --subnet-id <default-subnet> --security-group-ids <default-sg> --user-data file://user_data.sh
# poll: aws s3 ls s3://<bucket>/results.tar.gz ; then aws ec2 terminate-instances immediately after
```
