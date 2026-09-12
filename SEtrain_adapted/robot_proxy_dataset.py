"""
Robot-proxy on-the-fly mixture dataset. Same clean-speech pool as GENERIC (LibriSpeech
speech_split.json, same train/val/test speaker sets as F_GENERIC used -- decision: fair
comparison requires identical clean speech). Noise is drawn via WEIGHTED SOURCE SAMPLING
(50% generic / 35% UAV / 15% procedural per mixture) rather than duration-based pooling,
since the public UAV pool (~32s per split) and procedural pool (~2-6min per split) are far
smaller than DEMAND (~10-40min per split) -- this is the "adjust ratio, record actual" case
from the brief, resolved by sampling weight rather than shrinking the generic pool, which
would waste already-available, license-clear data for no reason. The 32s UAV recording is
therefore reused across many mixtures via random 4s crops at different SNRs -- a real
diversity limitation (only 3 distinct UAV recording sessions total), documented in
ROBOT_PROXY_DATA.md, not hidden.

Distance/angle are represented as attenuation + first-order lowpass "proxies" (see
distance_angle_proxy()) rather than physically simulated RIRs, per the brief's own priority
rule: RIR/physical simulation is a secondary nice-to-have that must never delay the main
experiment. This is a documented simplification, not a physical acoustic model.
"""
import json
import random

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from generic_dataset import mix_snr, _load_segment

FS = 16000
SOURCE_WEIGHTS = {"generic": 0.50, "uav": 0.35, "procedural": 0.15}
DISTANCE_PROXY_M = [0.5, 1.0, 2.0]
ANGLE_PROXY_DEG = [0, 45, 90, 180]


def distance_angle_proxy(x, distance_m, angle_deg, rng):
    """Simple physically-motivated (not physically simulated) proxy: distance attenuates
    level ~1/d and adds mild high-frequency rolloff (air/diffraction loss); angle beyond 0
    adds extra high-frequency loss (off-axis mic directivity) and a small level drop.
    Documented simplification -- not a real RIR/HRTF."""
    atten = 1.0 / max(distance_m, 0.1)
    from scipy import signal
    cutoff = 8000 / (1 + 0.5 * distance_m)
    angle_factor = 1.0 - 0.35 * (angle_deg / 180.0)
    cutoff *= (0.6 + 0.4 * angle_factor)
    cutoff = min(cutoff, 7900)
    sos = signal.butter(2, cutoff, btype="low", fs=FS, output="sos")
    y = signal.sosfilt(sos, x)
    y = y * atten * angle_factor
    return y.astype(np.float32)


class RobotProxyDataset(Dataset):
    def __init__(self, speech_manifest_path, noise_manifest_path, split, segment_seconds=4.0,
                 snr_range=(-5, 15), virtual_length=2000, seed=43, with_transcript=False):
        with open(speech_manifest_path) as f:
            self.speech_items = json.load(f)[split]
        with open(noise_manifest_path) as f:
            noise_items = json.load(f)[split]
        self.noise_by_source = {"generic": [], "uav": [], "procedural": []}
        for n in noise_items:
            self.noise_by_source[n["source"]].append(n)
        for src, items in self.noise_by_source.items():
            assert len(items) > 0, f"no noise items for source={src} in split={split}"

        self.split = split
        self.segment_len = int(segment_seconds * FS)
        self.snr_range = snr_range
        self.virtual_length = virtual_length if split == "train" else len(self.speech_items)
        self.base_seed = seed
        self.with_transcript = with_transcript

    def __len__(self):
        return self.virtual_length

    def _pick_noise(self, rng):
        src = rng.choices(list(SOURCE_WEIGHTS.keys()), weights=list(SOURCE_WEIGHTS.values()), k=1)[0]
        item = rng.choice(self.noise_by_source[src])
        return item, src

    def __getitem__(self, idx):
        # Deterministic per-index seed (not fresh entropy) for BOTH train and val/test, so
        # F_PROXY_ROBOT and F_PROXY_ROBOT_ASR draw the exact same mixture sequence at every
        # step -- required for the fair-comparison rule ("cùng mixture manifest và thứ tự
        # sample"). DataLoader uses no shuffle, so idx order is 0..virtual_length-1 every run.
        rng = random.Random(self.base_seed * 100003 + idx)
        if self.split == "train":
            speech_item = rng.choice(self.speech_items)
        else:
            speech_item = self.speech_items[idx % len(self.speech_items)]

        noise_item, source = self._pick_noise(rng)
        snr_db = rng.uniform(*self.snr_range)
        distance_m = rng.choice(DISTANCE_PROXY_M)
        angle_deg = rng.choice(ANGLE_PROXY_DEG)

        clean = _load_segment(speech_item["path"], self.segment_len, rng)
        noise = _load_segment(noise_item["path"], self.segment_len, rng)
        noise = distance_angle_proxy(noise, distance_m, angle_deg, rng)

        mix, clean_ref = mix_snr(clean, noise, snr_db, rng)

        meta = {"robot_state": noise_item["robot_state"], "noise_source": source,
                "distance_m": distance_m, "angle_deg": angle_deg, "snr_db": snr_db}

        if self.with_transcript:
            return (torch.from_numpy(mix), torch.from_numpy(clean_ref),
                    speech_item["transcript"] or "", meta)
        return torch.from_numpy(mix), torch.from_numpy(clean_ref)
