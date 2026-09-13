"""Build manifests for the 90-min sweep. Speech = LibriSpeech dev-clean (via
split_manifest.build_speech_manifest). Generic noise: Zenodo DEMAND 16k was
504-timing-out for all but DKITCHEN/OOFFICE on 2026-09-13, so we time-slice
those two recordings 70/15/15 into train/val/test (leakage-free: disjoint time
regions of each recording). Documented deviation from the original 12-env DEMAND
split -- all 4 sweep arms share this identical noise, so the arm-vs-arm
comparison is unaffected; only comparison to the *published* baseline is.
"""
import json, os
import numpy as np
import soundfile as sf
import split_manifest as sm

MANIFEST_DIR = "/content/project/data/manifests"
DEMAND = "/content/project/data/demand/extracted"
GEN_OUT = "/content/project/data/demand/sliced"
FS = 16000
ENVS = ["DKITCHEN", "OOFFICE"]        # the two that downloaded successfully
SPLITS = {"train": (0.0, 0.70), "val": (0.70, 0.85), "test": (0.85, 1.0)}


def build_generic_noise():
    os.makedirs(GEN_OUT, exist_ok=True)
    noise = {"train": [], "val": [], "test": []}
    for env in ENVS:
        p = f"{DEMAND}/{env}/ch01.wav"
        x, sr = sf.read(p)
        assert sr == FS
        n = len(x)
        for split, (a, b) in SPLITS.items():
            seg = x[int(a * n):int(b * n)]
            outp = f"{GEN_OUT}/{env}_{split}.wav"
            sf.write(outp, seg, FS)
            noise[split].append({"path": outp, "env": f"{env}_{split}",
                                 "duration": len(seg) / FS})
    return noise


def main():
    os.makedirs(MANIFEST_DIR, exist_ok=True)
    speech, dur, train_spk = sm.build_speech_manifest()
    with open(f"{MANIFEST_DIR}/speech_split.json", "w") as f:
        json.dump(speech, f, indent=2)
    for s in ["train", "val", "test"]:
        print(f"speech[{s}]: {len(speech[s])} utt, {dur[s]/3600:.2f}h")
    noise = build_generic_noise()
    with open(f"{MANIFEST_DIR}/noise_split.json", "w") as f:
        json.dump(noise, f, indent=2)
    for s in ["train", "val", "test"]:
        tot = sum(n["duration"] for n in noise[s])
        print(f"generic-noise[{s}]: {len(noise[s])} files, {tot/60:.1f} min")


if __name__ == "__main__":
    main()
