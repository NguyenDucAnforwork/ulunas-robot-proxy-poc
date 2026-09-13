# REPRODUCTION.md — M1 (generic) + robot-proxy training/evaluation

Environment used: Linux, 1× NVIDIA A100-SXM4-80GB, CUDA 13.0, Python 3.13. Dependency versions pinned in `requirements_pinned.txt` (captured via `pip freeze` after installation — exact versions actually used this run, not a hand-written wishlist).

## 1. Get the reference repos and checkpoint

```bash
git clone --depth 1 https://github.com/Xiaobin-Rong/ul-unas.git
git clone --depth 1 https://github.com/Xiaobin-Rong/SEtrain.git
```

## 2. Adapt SEtrain (already done under `SEtrain_adapted/`)

```bash
cp -r SEtrain/* SEtrain_adapted/
cp ul-unas/ulunas.py SEtrain_adapted/models/ulunas.py
```
Then the new files in `SEtrain_adapted/` (`generic_dataset.py`, `asr_loss.py`, `dnsmos_onnxruntime.py`, `train_generic.py`, `eval_generic.py`, `split_manifest.py`) replace/extend the template's `train.py`/`dataloader.py`/`evaluate.py`.

## 3. Get the data (see `data/manifests/data_manifest.csv` for exact URLs/hashes/license verification dates)

```bash
wget https://www.openslr.org/resources/12/dev-clean.tar.gz            # LibriSpeech, CC BY 4.0
wget https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz  # CC BY 4.0
# DEMAND: 12 environments at 16kHz, e.g.
wget "https://zenodo.org/records/1227121/files/DKITCHEN_16k.zip?download=1"
# ... (11 more environments, see data_manifest.csv for the full list)
```
Extract, then generate the split manifests (no leakage — verified by an assertion in the script itself):
```bash
cd SEtrain_adapted && python3 split_manifest.py
```

## 4. Fine-tune

```bash
# benchmark first (300-500 steps) to size the run to your time budget
python3 train_generic.py --variant se_only --steps 400 --benchmark --out_tag bench_se_only
python3 train_generic.py --variant se_asr  --steps 400 --benchmark --out_tag bench_se_asr

# full runs (same step count for both, per the plan's controlled-comparison requirement)
python3 train_generic.py --variant se_only --steps 18000 --out_tag F_GENERIC
python3 train_generic.py --variant se_asr  --steps 18000 --out_tag F_GENERIC_ASR
```
`lambda_asr` for the ASR-loss run is computed automatically via a gradient-scale probe on the first batch (see `training_logs/F_GENERIC_ASR/lambda_probe.json`) unless passed explicitly with `--lambda_asr`.

## 5. Evaluate

```bash
python3 eval_generic.py
```
Produces `evaluation/evaluation_results.{json,csv}`, `evaluation/evaluation_raw_per_utterance.json` (per-utterance values, for any further statistical analysis), and 10 audio comparison sets under `audio_samples/`. This step is CPU-bound (DNSMOS) and took roughly 15-20 minutes on this machine for 563+300 test utterances × 4 conditions when nothing else was competing for CPU cores.

## 6. Mobile / ONNX export (M4, done for the fine-tuned checkpoint)

```bash
cd mobile/onnx_export && python3 export_finetuned_stream.py   # exports ONNX + runs parity check
cd mobile/android_ref/build_arm64
cmake -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-24 ..
make
```

## 7. Runtime-free C kernels (M5, partial — see `MOBILE_BENCHMARK.md` for exact status)

```bash
cd mobile/c_neon && python3 export_weights.py   # dumps checkpoint to a C header
cd build && gcc -O2 -std=c99 -I../src ../src/test_block0.c -lm -o test_block0 && ./test_block0
# (test_block1.c, test_block2.c, test_dpgrnn.c, test_decoder4.c similarly)
```

## 8. Robot-proxy data + training + evaluation (see `ROBOT_PROXY_DATA.md`, `DATA_LICENSES.md` for full detail)

```bash
# public UAV ego-noise (KU Leuven, CC-BY-NC-SA-4.0 -- NonCommercial, see DATA_LICENSES.md)
# NOTE: the DOI in some briefs (10.48804/TLUJBE) 404s -- use 10.48804/PZAVUC, verified working
curl -sL "https://rdr.kuleuven.be/api/access/datafile/706" -o RPM4000_Channel_01.wav  # train
curl -sL "https://rdr.kuleuven.be/api/access/datafile/702" -o RPM5000_Channel_01.wav  # val
curl -sL "https://rdr.kuleuven.be/api/access/datafile/699" -o RPM6000_Channel_01.wav  # test
# (resample to 16kHz; see build_robot_proxy_manifest.py)

# procedural mechanical noise (fully synthetic, seeded, reproducible)
python3 procedural_noise.py

# combined manifest (50% generic / 35% UAV / 15% procedural BY SAMPLING WEIGHT, not duration --
# see ROBOT_PROXY_DATA.md §2 for why)
python3 build_robot_proxy_manifest.py

# train (same checkpoint/seed/optimizer/steps as F_GENERIC, only noise pool + loss differ)
python3 train_robot_proxy.py --variant se_only --steps 18000 --out_tag F_PROXY_ROBOT
python3 train_robot_proxy.py --variant se_asr  --steps 18000 --out_tag F_PROXY_ROBOT_ASR

# full 6-condition evaluation (N0/P0/F_GENERIC/F_GENERIC_ASR/F_PROXY_ROBOT/F_PROXY_ROBOT_ASR)
# on a NEW robot-proxy-domain test set -- this re-evaluates F_GENERIC/F_GENERIC_ASR too, since
# they were never tested on this domain in M1. Adds whisper-tiny.en as a second ASR (WER
# generalization check) alongside the primary wav2vec2-base-960h.
python3 eval_robot_proxy.py
```

This eval run is significantly slower than M1's (roughly 6/4 more conditions × 2 ASR decodes instead of 1) — budget well over an hour of mostly-CPU-bound (DNSMOS) + some GPU (model + 2×ASR decode) time.

## 9. Checkpoint SHA256 hashes (verify against `hf download banhchungtuongot/ulunas-robot-proxy-poc`)

```
11a20e25b06c446b2e205b10c3778a658480c0622b229a8c3b82c8949e320fb5  F_GENERIC.tar
b9c0621e91e1094a2e366f53228fb2b24340f00dba19a32bdfe34b0b22d494e8  F_GENERIC_ASR.tar
70c2a7d26af6d11d701cd683fd67b07d1979fc0dbe2f77434851a5d9b4aa8c29  F_PROXY_ROBOT.tar
5394d0c012e500b37d4c2076cf5b847d9db07348c135bd15deb3f6904fa0167e  F_PROXY_ROBOT_best.tar
36d523c1647dda9ca8f81b3cfc4819bc1694b9776a0688fa1dd356be5c3e5b3c  F_PROXY_ROBOT_ASR.tar
0f9f23f7c752c5139cddfc38bc3dae57de56d33f49c10dc0e4a403801ec6ded4  F_PROXY_ROBOT_ASR_best.tar
5c23c0ba63fe01ba9180674431e60a8d62398de5cf1e3607ac7329edf5bfe10c  F_PROXY_ROBOT_LOWSUP.tar
894b29ca0277e5d80ed853d91c6e518fd7e556028b683f7d1a9057fbac8e019b  F_PROXY_ROBOT_LOWSUP_best.tar
2e575697f5292575da89767d70daf31c899490580dda26b4f679ac74351e9863  F_PROXY_ROBOT_LOWSISNR.tar
76573cc974a37556eebbeb7d208ac35b6d1fb1e241060f8b955477408bca4863  F_PROXY_ROBOT_LOWSISNR_best.tar
812d67d4b8a4fd90b1f12b78e649f27d1a30da18f235e4007b8712a27d4e1819  F_PROXY_ROBOT_LOWSISNR2.tar
9af06884c19615bdfdad647c3991ccc894bfd4f85b5f69bc12677fa4d76dc745  F_PROXY_ROBOT_LOWSISNR2_best.tar
```

`F_PROXY_ROBOT_LOWSUP`: reduced-suppression ablation (`lamda_mag` 70→35), see `QC_FAILURE_ANALYSIS.md` §4.3c — real negative result, not an improvement.

`F_PROXY_ROBOT_LOWSISNR`: reduced SI-SNR-weight ablation (`lamda_sisnr` 1.0→0.3, `lamda_ri`/`lamda_mag` back to 30/70), see `QC_FAILURE_ANALYSIS.md` §4.3d — small but statistically robust joint SIG/BAK/OVRL improvement, more pronounced in the UAV-motor_high+fan tradeoff subset; still sub-material (<0.03 DNSMOS), WER not robustly changed.

`F_PROXY_ROBOT_LOWSISNR2`: pushes further (`lamda_sisnr` 0.3→0.1), see `QC_FAILURE_ANALYSIS.md` §4.3e — SIG/BAK/OVRL improvement continues monotonically (SIG in the tradeoff subset now approaches, but stays under, the 0.03 materiality line), but overall SI-SDR is now robustly *worse* (-0.08dB vs original) — a real perceptual-quality/waveform-fidelity tradeoff, not a free improvement.

These are the exact checkpoints produced and uploaded in this session (computed via `sha256sum`, not re-derived) — training is not bit-exact-reproducible run-to-run (GPU float non-determinism, unseeded `DataLoader` workers), so a fresh training run will produce *equivalent* but not *hash-identical* checkpoints. Use these hashes only to verify you have the exact same uploaded files, not to validate a fresh retrain.

## Known gaps / things a fresh run should watch for

- QUT-NOISE download will fail with HTTP 403 (server-side block, not a code bug) — DEMAND alone is used.
- CUDA-based DNSMOS silently falls back to CPU on this environment (missing `libcublasLt.so.13` for the installed onnxruntime-gpu/CUDA version pairing) — this is why evaluation is CPU-bound; not a correctness issue, just slower than it could be.
- `eval_generic.py`'s JSON writes use `default=lambda o: float(o)` as a safety net for stray numpy scalar types — if you see a `TypeError: Object of type ... is not JSON serializable` again, that's the place to look first (it happened once during this project, after ~90 minutes of otherwise-successful computation, from `si_sdr()`/`stoi()` returning numpy floats).
