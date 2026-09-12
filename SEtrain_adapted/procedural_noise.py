"""
Procedural robot-like mechanical noise generator (motor harmonics+drift, fan broadband, RPM
transitions, servo click/impulse, chassis vibration, wheel/surface rumble, clipping/gain/
device-response coloring). All parameters and the RNG seed are fixed and logged so every
clip is exactly reproducible -- see PROCEDURAL_NOISE_PARAMS below and the seed passed in.

This is explicitly NOT real robot data -- it is a documented simulation, used only because no
real robot/ego-noise recording device is available (per project decision). Never call its
output "real robot noise".
"""
import json
import os

import numpy as np
import soundfile as sf
from scipy import signal

FS = 16000

PROCEDURAL_NOISE_PARAMS = {
    "motor_fundamental_hz_range": [40, 220],
    "motor_n_harmonics": 8,
    "motor_harmonic_rolloff": 0.6,   # amplitude of harmonic k = rolloff^k
    "motor_drift_depth_hz": 3.0,     # slow random-walk drift on the fundamental
    "motor_drift_rate_hz": 0.3,      # how fast the drift random-walk moves
    "fan_broadband_lowcut_hz": 200,
    "fan_broadband_highcut_hz": 6000,
    "rpm_transition_start_hz_range": [40, 90],
    "rpm_transition_end_hz_range": [150, 220],
    "servo_click_rate_hz_range": [0.5, 4.0],
    "servo_click_duration_ms": 8,
    "chassis_vibration_hz_range": [15, 60],
    "wheel_rumble_lowcut_hz": 20,
    "wheel_rumble_highcut_hz": 300,
    "wheel_bump_rate_hz_range": [0.2, 1.5],
    "device_response_tilt_db_per_octave_range": [-3, 3],
    "clip_probability": 0.15,
    "clip_threshold_range": [0.85, 0.98],
    "gain_variation_db_range": [-4, 4],
}


def _rng(seed):
    return np.random.default_rng(seed)


def _apply_device_response(x, rng, fs=FS):
    """Mild spectral tilt to emulate a generic MEMS mic frequency response, +- a few dB/octave."""
    tilt = rng.uniform(*PROCEDURAL_NOISE_PARAMS["device_response_tilt_db_per_octave_range"])
    n = len(x)
    freqs = np.fft.rfftfreq(n, 1 / fs)
    freqs_safe = np.maximum(freqs, 1.0)
    octaves_from_1k = np.log2(freqs_safe / 1000.0)
    gain_db = tilt * octaves_from_1k
    gain = 10 ** (gain_db / 20)
    X = np.fft.rfft(x)
    X = X * gain
    return np.fft.irfft(X, n=n).astype(np.float32)


def _apply_gain_and_clip(x, rng):
    gain_db = rng.uniform(*PROCEDURAL_NOISE_PARAMS["gain_variation_db_range"])
    x = x * (10 ** (gain_db / 20))
    if rng.random() < PROCEDURAL_NOISE_PARAMS["clip_probability"]:
        thresh = rng.uniform(*PROCEDURAL_NOISE_PARAMS["clip_threshold_range"])
        peak = np.max(np.abs(x)) + 1e-8
        x = np.clip(x / peak, -thresh, thresh) * peak
    peak = np.max(np.abs(x)) + 1e-8
    if peak > 0.98:
        x = x / peak * 0.98
    return x.astype(np.float32)


def gen_motor(duration_s, seed, fundamental_hz=None):
    rng = _rng(seed)
    n = int(duration_s * FS)
    t = np.arange(n) / FS
    if fundamental_hz is None:
        fundamental_hz = rng.uniform(*PROCEDURAL_NOISE_PARAMS["motor_fundamental_hz_range"])
    drift_rate = PROCEDURAL_NOISE_PARAMS["motor_drift_rate_hz"]
    drift = PROCEDURAL_NOISE_PARAMS["motor_drift_depth_hz"] * np.cumsum(rng.normal(0, 1, n)) / np.sqrt(n)
    drift = signal.lfilter([drift_rate], [1, -(1 - drift_rate)], drift)
    inst_freq = fundamental_hz + drift
    phase = 2 * np.pi * np.cumsum(inst_freq) / FS
    x = np.zeros(n, dtype=np.float64)
    rolloff = PROCEDURAL_NOISE_PARAMS["motor_harmonic_rolloff"]
    for k in range(1, PROCEDURAL_NOISE_PARAMS["motor_n_harmonics"] + 1):
        x += (rolloff ** (k - 1)) * np.sin(k * phase)
    x = x / np.max(np.abs(x))
    x = _apply_device_response(x.astype(np.float32), rng)
    return _apply_gain_and_clip(x, rng)


def gen_fan(duration_s, seed):
    rng = _rng(seed)
    n = int(duration_s * FS)
    white = rng.normal(0, 1, n)
    sos = signal.butter(4, [PROCEDURAL_NOISE_PARAMS["fan_broadband_lowcut_hz"],
                            PROCEDURAL_NOISE_PARAMS["fan_broadband_highcut_hz"]],
                        btype="band", fs=FS, output="sos")
    x = signal.sosfilt(sos, white)
    x = x / (np.max(np.abs(x)) + 1e-8)
    x = _apply_device_response(x.astype(np.float32), rng)
    return _apply_gain_and_clip(x, rng)


def gen_rpm_transition(duration_s, seed):
    rng = _rng(seed)
    n = int(duration_s * FS)
    t = np.arange(n) / FS
    f0 = rng.uniform(*PROCEDURAL_NOISE_PARAMS["rpm_transition_start_hz_range"])
    f1 = rng.uniform(*PROCEDURAL_NOISE_PARAMS["rpm_transition_end_hz_range"])
    if rng.random() < 0.5:
        f0, f1 = f1, f0  # ramp down instead of up, randomly
    inst_freq = f0 + (f1 - f0) * (t / duration_s)
    phase = 2 * np.pi * np.cumsum(inst_freq) / FS
    x = np.zeros(n, dtype=np.float64)
    rolloff = PROCEDURAL_NOISE_PARAMS["motor_harmonic_rolloff"]
    for k in range(1, PROCEDURAL_NOISE_PARAMS["motor_n_harmonics"] + 1):
        x += (rolloff ** (k - 1)) * np.sin(k * phase)
    x = x / np.max(np.abs(x))
    x = _apply_device_response(x.astype(np.float32), rng)
    return _apply_gain_and_clip(x, rng)


def gen_servo_impulse(duration_s, seed):
    rng = _rng(seed)
    n = int(duration_s * FS)
    x = np.zeros(n, dtype=np.float64)
    click_rate = rng.uniform(*PROCEDURAL_NOISE_PARAMS["servo_click_rate_hz_range"])
    click_len = int(PROCEDURAL_NOISE_PARAMS["servo_click_duration_ms"] / 1000 * FS)
    t_click = np.arange(click_len) / FS
    n_clicks = int(duration_s * click_rate)
    for _ in range(n_clicks):
        pos = rng.integers(0, max(1, n - click_len))
        env = np.exp(-t_click / (PROCEDURAL_NOISE_PARAMS["servo_click_duration_ms"] / 1000 / 3))
        click_freq = rng.uniform(800, 3000)
        click = env * np.sin(2 * np.pi * click_freq * t_click) * rng.uniform(0.5, 1.0)
        x[pos:pos + click_len] += click
    # low-level chassis hum under the clicks
    hum_freq = rng.uniform(*PROCEDURAL_NOISE_PARAMS["chassis_vibration_hz_range"])
    x += 0.05 * np.sin(2 * np.pi * hum_freq * np.arange(n) / FS)
    x = x / (np.max(np.abs(x)) + 1e-8)
    x = _apply_device_response(x.astype(np.float32), rng)
    return _apply_gain_and_clip(x, rng)


def gen_moving_surface(duration_s, seed):
    rng = _rng(seed)
    n = int(duration_s * FS)
    white = rng.normal(0, 1, n)
    sos = signal.butter(4, [PROCEDURAL_NOISE_PARAMS["wheel_rumble_lowcut_hz"],
                            PROCEDURAL_NOISE_PARAMS["wheel_rumble_highcut_hz"]],
                        btype="band", fs=FS, output="sos")
    rumble = signal.sosfilt(sos, white)
    bump_rate = rng.uniform(*PROCEDURAL_NOISE_PARAMS["wheel_bump_rate_hz_range"])
    n_bumps = int(duration_s * bump_rate)
    bump_len = int(0.05 * FS)
    t_bump = np.arange(bump_len) / FS
    for _ in range(n_bumps):
        pos = rng.integers(0, max(1, n - bump_len))
        env = np.exp(-t_bump / 0.02)
        rumble[pos:pos + bump_len] += env * rng.uniform(1.5, 3.0) * rng.standard_normal(bump_len)
    x = rumble / (np.max(np.abs(rumble)) + 1e-8)
    x = _apply_device_response(x.astype(np.float32), rng)
    return _apply_gain_and_clip(x, rng)


def gen_idle(duration_s, seed):
    rng = _rng(seed)
    n = int(duration_s * FS)
    white = rng.normal(0, 1, n) * 0.3
    sos = signal.butter(2, [100, 2000], btype="band", fs=FS, output="sos")
    x = signal.sosfilt(sos, white)
    x = x / (np.max(np.abs(x)) + 1e-8) * 0.4  # deliberately low-level, "idle" = quiet ambient hum
    x = _apply_device_response(x.astype(np.float32), rng)
    return _apply_gain_and_clip(x, rng)


GENERATORS = {
    "idle": gen_idle,
    "fan": gen_fan,
    "motor_low": lambda d, s: gen_motor(d, s, fundamental_hz=60),
    "motor_high": lambda d, s: gen_motor(d, s, fundamental_hz=180),
    "rpm_transition": gen_rpm_transition,
    "servo_impulse": gen_servo_impulse,
    "moving_surface": gen_moving_surface,
}


def generate_split(out_dir, split_name, states, n_clips_per_state, duration_s, base_seed):
    """Different base_seed per split guarantees non-overlapping procedural noise instances
    across train/dev/test (no two splits ever draw the same RNG stream)."""
    os.makedirs(out_dir, exist_ok=True)
    manifest = []
    for state in states:
        gen_fn = GENERATORS[state]
        for i in range(n_clips_per_state):
            seed = hash((split_name, state, i, base_seed)) % (2 ** 31)
            x = gen_fn(duration_s, seed)
            fname = f"{state}_{split_name}_{i:03d}.wav"
            path = os.path.join(out_dir, fname)
            sf.write(path, x, FS)
            manifest.append({"path": path, "state": state, "split": split_name,
                             "seed": seed, "duration": duration_s, "source": "procedural"})
    return manifest


if __name__ == "__main__":
    OUT = "/content/project/data/procedural_noise"
    states = list(GENERATORS.keys())
    all_manifest = []
    all_manifest += generate_split(f"{OUT}/train", "train", states, n_clips_per_state=6, duration_s=8.0, base_seed=1000)
    all_manifest += generate_split(f"{OUT}/dev", "dev", states, n_clips_per_state=2, duration_s=8.0, base_seed=2000)
    all_manifest += generate_split(f"{OUT}/test", "test", states, n_clips_per_state=3, duration_s=8.0, base_seed=3000)

    with open(f"{OUT}/manifest.json", "w") as f:
        json.dump({"params": PROCEDURAL_NOISE_PARAMS, "clips": all_manifest}, f, indent=2)
    with open(f"{OUT}/params.json", "w") as f:
        json.dump(PROCEDURAL_NOISE_PARAMS, f, indent=2)

    total_dur = sum(c["duration"] for c in all_manifest)
    print(f"Generated {len(all_manifest)} clips, {total_dur/60:.1f} min total, states={states}")
    for split in ["train", "dev", "test"]:
        n = sum(1 for c in all_manifest if c["split"] == split)
        d = sum(c["duration"] for c in all_manifest if c["split"] == split)
        print(f"  {split}: {n} clips, {d/60:.1f} min")
