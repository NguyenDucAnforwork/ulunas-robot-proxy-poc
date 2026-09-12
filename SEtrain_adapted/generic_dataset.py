"""
GENERIC-domain on-the-fly noisy/clean mixture dataset for UL-UNAS fine-tuning.
Clean speech: LibriSpeech dev-clean subset (English, read speech).
Noise: DEMAND (16kHz, single channel).
No pre-generated mixtures are stored on disk; every __getitem__ call synthesizes a fresh mixture.
"""
import json
import random

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

FS = 16000


def _load_segment(path, n_samples, rng):
    info = sf.info(path)
    total = info.frames
    if total <= n_samples:
        audio, _ = sf.read(path, dtype="float32")
        if len(audio) < n_samples:
            audio = np.pad(audio, (0, n_samples - len(audio)))
        return audio
    start = rng.randint(0, total - n_samples)
    audio, _ = sf.read(path, dtype="float32", start=start, stop=start + n_samples)
    return audio


def mix_snr(clean, noise, snr_db, rng, eps=1e-8):
    """Scale noise to hit the target SNR (dB) against clean, then random-gain and peak-normalize.
    Mirrors the spirit of SEtrain/prepare_datasets/gen_DNS3_datasets.py::mk_mixture (adapted to
    single-source, no reverberant/low-reverb pair since we have no RIR in M1)."""
    clean_power = np.mean(clean ** 2) + eps
    noise_power = np.mean(noise ** 2) + eps
    target_noise_power = clean_power / (10 ** (snr_db / 10))
    noise_scaled = noise * np.sqrt(target_noise_power / noise_power)

    amp = 0.3 + 0.5 * rng.random()  # random overall gain, matches mk_mixture's amp draw
    clean_g = amp * clean
    noise_g = amp * noise_scaled
    mix = clean_g + noise_g

    peak = max(np.max(np.abs(mix)), np.max(np.abs(clean_g))) + eps
    if peak > 1.0:
        mix = mix / peak
        clean_g = clean_g / peak
    return mix.astype(np.float32), clean_g.astype(np.float32)


class GenericSEDataset(Dataset):
    def __init__(self, speech_manifest_path, noise_manifest_path, split, segment_seconds=4.0,
                 snr_range=(-5, 15), virtual_length=2000, seed=43, with_transcript=False):
        with open(speech_manifest_path) as f:
            self.speech_items = json.load(f)[split]
        with open(noise_manifest_path) as f:
            self.noise_items = json.load(f)[split]
        self.split = split
        self.segment_len = int(segment_seconds * FS)
        self.snr_range = snr_range
        self.virtual_length = virtual_length if split == "train" else len(self.speech_items)
        self.base_seed = seed
        self.with_transcript = with_transcript

    def __len__(self):
        return self.virtual_length

    def __getitem__(self, idx):
        rng = random.Random(self.base_seed * 100003 + idx if self.split != "train" else None)
        if self.split == "train":
            rng = random.Random()  # true randomness each epoch for train (on-the-fly augmentation)
            speech_item = rng.choice(self.speech_items)
        else:
            speech_item = self.speech_items[idx % len(self.speech_items)]
        noise_item = rng.choice(self.noise_items)
        snr_db = rng.uniform(*self.snr_range)

        clean = _load_segment(speech_item["path"], self.segment_len, rng)
        noise = _load_segment(noise_item["path"], self.segment_len, rng)
        mix, clean_ref = mix_snr(clean, noise, snr_db, rng)

        if self.with_transcript:
            return (torch.from_numpy(mix), torch.from_numpy(clean_ref),
                    speech_item["transcript"] or "")
        return torch.from_numpy(mix), torch.from_numpy(clean_ref)
