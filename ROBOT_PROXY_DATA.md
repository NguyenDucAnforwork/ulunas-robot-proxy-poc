# ROBOT_PROXY_DATA.md

No real robot exists for this project. Everything below is a **proxy**: real public UAV ego-noise + procedural mechanical-noise simulation, mixed with the same clean speech used for `F_GENERIC`. Nothing here is called "real robot performance" anywhere in this project — see `EXPERIMENT_REPORT.md` RQ3 for the explicit scope limitation.

## 1. Noise sources

### 1.1 Public UAV ego-noise (real recordings)

Source: KU Leuven RDR, "Replication Data for: Speech enhancement using ego-noise references with a microphone array embedded in an unmanned aerial vehicle" (Tengan, Dietzen, et al., ICA 2022), `doi:10.48804/PZAVUC`.

**Discrepancy note**: the persistentId given in the task brief (`doi:10.48804/TLUJBE`) returns HTTP 404 — it does not exist. The correct, working DOI for this exact dataset (UAV ego-noise, microphone array, Leuven) was `doi:10.48804/PZAVUC`, found via the same search that surfaced this dataset during earlier planning in this project. Used the working one; flagging the discrepancy rather than silently substituting.

- **Content**: 16-channel microphone array embedded in a real UAV, recorded in a semi-anechoic chamber at Siemens Digital Industries Software, Leuven, at 3 fixed rotor speeds: 4000, 5000, 6000 RPM. Channels 1-12 are the main array; channels 13-16 are under the propeller pairs. A separate `Speech_Ref` file per RPM level is an external reference mic near a loudspeaker, used by the original authors for speech-presence-probability — **not used here** (it's a reference mic, not ego-noise, and the brief explicitly forbids using a reference microphone at inference).
- **Channel used**: **Channel_01 only**, for every RPM level — one single on-device array element, satisfying "chỉ dùng một channel microphone gắn trên thiết bị" (single-channel constraint).
- **License**: **CC-BY-NC-SA-4.0** — confirmed directly via the Dataverse API (`/api/datasets/:persistentId/`), not just the web page. **Non-commercial only.** This restricts any downstream commercial use of models trained partly on this data; flagged prominently here and in `THIRD_PARTY_NOTICES.md`.
- **Access**: direct, no login/registration required (confirmed by successful anonymous download via the Dataverse REST API).
- **Format as downloaded**: 44.1kHz, 16-bit PCM, mono per channel, ~32s per RPM session. Resampled to 16kHz for this project.
- **Split assignment (no leakage)**: each RPM level is one distinct, independent recording session — assigned whole to one split, never chunked across splits:
  - RPM4000 → **train** (`motor_low`)
  - RPM5000 → **val** (`motor_mid`)
  - RPM6000 → **test** (`motor_high`)
- **Real limitation**: only ~32 seconds of raw audio per split, 3 sessions total. Reused via random 4-second crops at different SNRs across many training mixtures (same reuse strategy already used for DEMAND in M1) — genuinely low source diversity, documented, not hidden.

### 1.2 Robovox — attempted, BLOCKED

`robovox.univ-avignon.fr` explicitly includes real robot ego-noise ("internal robot noises (robot activators)"), which would have been a better match than UAV noise. **Access requires registering for a CodaBench challenge** to receive a dataset link; no license was stated on the public page. Per the brief's explicit rule ("nếu yêu cầu tài khoản... không bypass, ghi BLOCKED"), this is marked **BLOCKED** — not signed up for, not downloaded.

### 1.3 Generic environmental noise

Reused unmodified: the exact DEMAND train/val/test split already built and leak-checked for M1 (`data/manifests/noise_split.json`) — 8/2/2 environments, 40/10/10 minutes.

### 1.4 Procedural mechanical noise (simulation, not real data)

`SEtrain_adapted/procedural_noise.py` generates 7 documented, seeded noise families:

| State | Method |
|---|---|
| `idle` | low-level band-limited (100-2000Hz) noise, deliberately quiet |
| `fan` | broadband (200-6000Hz) filtered white noise |
| `motor_low` | harmonic synthesis, fundamental ~60Hz + 8 harmonics, slow random-walk drift |
| `motor_high` | same, fundamental ~180Hz |
| `rpm_transition` | harmonic synthesis with a linear frequency ramp between two random endpoints (up or down) — **the one state the real UAV data cannot provide**, since each UAV recording is a single fixed RPM |
| `servo_impulse` | Poisson-ish click train (exponential-decay tone bursts) over a low chassis hum |
| `moving_surface` | band-limited (20-300Hz) rumble + random low-frequency "bump" impulses |

All clips also get: a mild synthetic microphone frequency-response tilt (±3dB/octave), random gain (±4dB), and a 15% chance of mild clipping — matching the brief's "clipping nhẹ, gain variation và device frequency response" requirement.

Every clip's exact seed and the full `PROCEDURAL_NOISE_PARAMS` dict are saved to `data/procedural_noise/manifest.json` and `params.json` for exact reproducibility. **Split leakage prevention**: train/val/test each use a disjoint base seed (1000/2000/3000) folded into every clip's seed — no two splits can ever produce the same waveform, by construction (not just "checked", literally impossible by the seeding scheme).

Generated: 77 clips, 10.3 minutes total (train 5.6min/42 clips, val 1.9min/14 clips, test 2.8min/21 clips).

## 2. Mixture composition — target vs. actual

Target from the brief: 50% generic / 35% UAV / 15% procedural.

**By raw file duration**, DEMAND (40+10+10 min) vastly outweighs UAV (32s×3) and procedural (10.3min total) — a duration-based split would put UAV at <5% regardless of intent. **Resolution: the ratio is implemented as a per-mixture WEIGHTED SAMPLING PROBABILITY** (`robot_proxy_dataset.py::SOURCE_WEIGHTS = {"generic":0.50,"uav":0.35,"procedural":0.15}`), not a duration allocation — every training/eval mixture independently draws its noise source with exactly those probabilities, then crops a random 4-second segment (with a random SNR) from whichever file that source's pool contains. This hits the target ratio exactly in terms of mixture composition, at the cost of heavy reuse of the small UAV/procedural pools (documented above, not hidden) — this is standard practice in SE data augmentation and is the correct fix for "public data insufficient → adjust ratio, record the real one," except here no ratio adjustment was actually needed once framed as sampling probability rather than duration.

## 3. Distance/angle/RIR proxy — explicit simplification

No real RIR measurement or physical acoustic simulation (e.g. pyroomacoustics image-source method) is used. Per the brief's own priority rule ("RIR ưu tiên phụ, không được làm trễ thí nghiệm chính"), distance (0.5/1/2m) and angle (0°/45°/90°/180°) are represented by a lightweight, physically-motivated **proxy**: distance applies 1/d amplitude attenuation plus a mild low-pass filter (air/diffraction loss increases with distance); angle beyond 0° further lowers the cutoff frequency and applies a small extra attenuation (approximating off-axis microphone directivity loss). See `robot_proxy_dataset.py::distance_angle_proxy()`. **This is not a physical RIR/HRTF model** — it is called a "proxy" throughout this project precisely to avoid overstating its realism.

## 4. Robot state coverage

All 7 requested states are represented: `idle, fan, motor_low, motor_high, rpm_transition, servo_impulse, moving_surface` come from procedural synthesis (plus `motor_low`/`motor_high` also draw from the real UAV recordings at RPM4000/6000); `motor_mid` (RPM5000) was added as a documented extension since 5000 RPM doesn't cleanly fit "low" or "high"; `generic_background` (DEMAND) is the 50%-weight non-robot-specific noise floor.

## 5. Manifests and reproduction

- `data/manifests/robot_proxy_noise_split.json` — combined manifest (all 3 sources, all splits, with `robot_state`/`source`/`license` fields per entry).
- `data/manifests/robot_proxy_noise_ratio_report.json` — raw-duration ratio by split (informational; see §2 for why the ACTUAL mixture ratio is sampling-based, not this raw number).
- `data/procedural_noise/{manifest.json,params.json}` — exact procedural noise reproduction data.
- `SEtrain_adapted/build_robot_proxy_manifest.py`, `SEtrain_adapted/procedural_noise.py`, `SEtrain_adapted/robot_proxy_dataset.py` — the scripts that produce all of the above; re-running them reproduces byte-identical procedural noise (fixed seeds) and the same manifest structure.
- **No corrupted-file check needed for procedural data** (generated fresh, always valid); UAV/DEMAND files were verified readable via `soundfile.info()` during manifest building (any unreadable file would have raised an exception there, and none did).
