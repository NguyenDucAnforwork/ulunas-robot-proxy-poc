# INPUT_SPEC.md — UL-UNAS Robot/Generic Deployment Input Specification (DRAFT)

Status: **DRAFT — test-grid and framework defined; final thresholds marked PENDING real-device validation.** Per the plan's own rule, final thresholds must come from actual tests, not be copied blindly from the test grid below. No physical robot device or Cortex-A53 hardware was available during M1, so the "Recommended / Degraded / Unsupported" boundaries below are **starting points derived from the model's training conditions and general speech-enhancement/ASR literature**, not yet validated against real hardware measurements. Each PENDING row names exactly what test would confirm or revise it.

## 1. Fixed format (not negotiable — matches the model exactly)

| Property | Value | Source |
|---|---|---|
| Sample rate | 16000 Hz | `ulunas.py`, verified §1.1 of SPEC_PLAN |
| Channel count | 1 (mono) | Model input is `(B, n_samples)`, single channel only |
| PCM format | 16-bit signed PCM (PCM16) | Matches `audio/*.wav` samples in `ul-unas` repo, verified via `soundfile.info()` |
| Frame / hop | win=512 samples (32ms), hop=256 samples (16ms) | STFT params, §1.1 |
| Algorithmic latency | 32ms (2×hop) | Verified via repo's own causality test |

## 2. Robot simplification (per your decision #4 — must be stated explicitly, not assumed)

This model, in this PoC, is scoped for:
- **One microphone** in the robot (no array, no beamforming).
- **Single-channel noise suppression only.**
- **No echo cancellation (AEC)** — the operating assumption is the robot does **not** play audio through its own speaker while listening for a command. If a future deployment plays TTS/audio while listening, this model provides **no AEC protection** and will treat robot-emitted audio as just another interfering signal (likely poorly suppressed, since it wasn't trained on the robot's own played-back audio as a distinct noise class).
- No beamforming (would require a microphone array, which this PoC does not have).

Any input that violates these assumptions (multi-mic array feed, simultaneous robot playback) is **out of the supported envelope by construction**, not a threshold to be tuned.

## 3. dBFS vs dB SPL — kept strictly separate

- **dBFS** (dB Full Scale): a measure of digital signal level *inside the PCM file*, relative to the maximum representable amplitude (0 dBFS = full scale). Everything measurable from a WAV file (active speech level, peak level, clipping) in this spec is dBFS.
- **dB SPL** (Sound Pressure Level): a physical acoustic measure at the microphone diaphragm, in Pascals referenced to a physical threshold. This depends on microphone sensitivity/gain calibration, which we do not have for the robot's actual microphone.
- **We do not infer dB SPL from PCM dBFS.** Any statement about "how loud" the physical environment was would require a calibrated SPL meter measurement alongside the recording — not done in M1, and not assumed.

## 4. Test grid (planned, per your instruction #15 and the original brief)

| Axis | Values |
|---|---|
| SNR | -5, 0, 5, 10, 20 dB |
| Distance | 20, 50, 100, 200 cm (generic); 50, 100, 200 cm (robot-specific, per decision #15) |
| Angle | 0°, 45°, 90° |
| Active speech level | -30, -24, -18, -12 dBFS |
| Clipping | with / without |
| Field | near-field / far-field |
| Device processing | default (AGC/NS on) / unprocessed, if the device supports switching |
| Robot state (robot-specific) | idle / moving / turning / servo-active |

**Status: PENDING.** None of these cells have been executed against real hardware yet — M1 only covers synthetic on-the-fly mixtures (LibriSpeech + DEMAND) and the Speech-Commands-based command-eval set, not this physical grid. Running this grid requires the real-robot recording setup described in SPEC_PLAN §3.6 (M3), not yet scheduled.

## 5. Recommended / Degraded / Unsupported zones (DRAFT thresholds — see status note above)

| Zone | SNR | Active speech level | Clipping | Distance | Notes |
|---|---|---|---|---|---|
| **Recommended** | ≥ 5 dB | -24 to -18 dBFS | none | ≤ 100cm (robot), ≤ 100cm (generic) | Matches the SNR range (-5 to 15 dB) used during M1 fine-tuning; extrapolation beyond this range is unvalidated |
| **Degraded but supported** | -5 to 5 dB | -30 to -24 dBFS, or -18 to -12 dBFS | isolated clipped samples (<0.1% of frame) | 100-200cm | Expect measurably lower SIG/WER per RQ1 findings (see EXPERIMENT_REPORT.md); still usable, not silently equivalent to "recommended" |
| **Unsupported / reject** | < -5 dB | < -30 dBFS (near-silence) or > -12 dBFS (near full-scale, high clip risk) | sustained clipping (>1% of frame) | > 200cm | Below the training SNR floor, or physically unrecoverable signal (clipped/near-silent) |

**These specific numeric boundaries are DRAFT** — set from the -5..15dB SNR range actually used in M1 training/eval (a real, verified fact) extrapolated with a margin (a judgment call, not a measurement) for the two ends. They must be revised once the real-device test grid (§4) is actually run, per the plan's own rule against blind copying.

## 6. Warn / reject conditions

- **Reject**: sample rate ≠ 16000 Hz without resampling; channel count > 1 without downmixing; input shorter than 1 STFT frame (512 samples).
- **Warn**: active speech level outside the "Recommended" band (§5); DC offset detected (mean |sample| > a small threshold, e.g. 0.01 of full scale, over a 1s window); silence ratio > 90% of the buffer (likely no speech present); clipping detected (any sample at ±32767 for PCM16).

## 7. Open items before this spec can be finalized

1. Real-device test grid (§4) execution — blocked on robot hardware access (same blocker as SPEC_PLAN §7).
2. Microphone calibration for any dB SPL statement — not attempted, not planned for M1/M2.
3. AGC/AEC/on-device NS interaction — cannot be characterized without the actual target robot's audio pipeline.
