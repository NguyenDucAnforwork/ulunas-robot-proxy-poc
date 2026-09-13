"""
Build train/val/test split manifests for the GENERIC domain (LibriSpeech dev-clean clean speech +
DEMAND noise), split by SOURCE (speaker / noise-environment) to prevent leakage across splits.
Run once; writes JSON manifests under data/manifests/.
"""
import json
import os
import glob
import soundfile as sf

LIBRISPEECH_ROOT = "/content/project/data/librispeech/LibriSpeech/dev-clean"
DEMAND_ROOT = "/content/project/data/demand/extracted"
OUT_DIR = "/content/project/data/manifests"

# Chosen by utterance-count so that: test >= 200 utterances, val ~20-30 min, train ~4-6h.
# (see SPEC_PLAN.md section 2.5 / this run's console log for the by-speaker utterance counts)
TEST_SPEAKERS = ["3752", "6313", "2277", "1462", "3081", "2428"]
VAL_SPEAKERS = ["422", "2902", "3576", "1673"]

NOISE_TRAIN = ["DKITCHEN", "OOFFICE", "PCAFETER", "TCAR", "STRAFFIC", "SPSQUARE", "NPARK", "OMEETING"]
NOISE_VAL = ["DWASHING", "PSTATION"]
NOISE_TEST = ["TBUS", "TMETRO"]


def build_speech_manifest():
    all_speakers = sorted(os.listdir(LIBRISPEECH_ROOT))
    all_speakers = [s for s in all_speakers if os.path.isdir(os.path.join(LIBRISPEECH_ROOT, s))]

    assert set(TEST_SPEAKERS).isdisjoint(VAL_SPEAKERS), "leakage: test/val speaker overlap"
    train_speakers = [s for s in all_speakers if s not in TEST_SPEAKERS and s not in VAL_SPEAKERS]
    assert set(train_speakers).isdisjoint(TEST_SPEAKERS)
    assert set(train_speakers).isdisjoint(VAL_SPEAKERS)

    manifest = {"train": [], "val": [], "test": []}
    duration = {"train": 0.0, "val": 0.0, "test": 0.0}

    for split_name, speakers in [("train", train_speakers), ("val", VAL_SPEAKERS), ("test", TEST_SPEAKERS)]:
        for spk in speakers:
            flacs = glob.glob(os.path.join(LIBRISPEECH_ROOT, spk, "**", "*.flac"), recursive=True)
            for f in flacs:
                # transcript lookup
                chapter_dir = os.path.dirname(f)
                trans_files = glob.glob(os.path.join(chapter_dir, "*.trans.txt"))
                utt_id = os.path.splitext(os.path.basename(f))[0]
                transcript = None
                for tf in trans_files:
                    with open(tf) as fh:
                        for line in fh:
                            uid, text = line.strip().split(" ", 1)
                            if uid == utt_id:
                                transcript = text
                                break
                    if transcript is not None:
                        break
                info = sf.info(f)
                manifest[split_name].append({
                    "path": f, "speaker": spk, "utt_id": utt_id,
                    "duration": info.duration, "transcript": transcript,
                })
                duration[split_name] += info.duration

    return manifest, duration, train_speakers


def build_noise_manifest():
    manifest = {"train": [], "val": [], "test": []}
    for split_name, envs in [("train", NOISE_TRAIN), ("val", NOISE_VAL), ("test", NOISE_TEST)]:
        for env in envs:
            # DEMAND *_16k.zip extracts to <env>/ch01.wav (single level); tolerate the
            # legacy double-<env> layout too. Skip envs that failed to download (flaky
            # Zenodo mirror) -- assert below that each split still has >=1 env.
            cands = [os.path.join(DEMAND_ROOT, env, "ch01.wav"),
                     os.path.join(DEMAND_ROOT, env, env, "ch01.wav")]
            f = next((c for c in cands if os.path.exists(c)), None)
            if f is None:
                print(f"[split_manifest] WARN missing DEMAND env {env}, skipping")
                continue
            info = sf.info(f)
            manifest[split_name].append({"path": f, "env": env, "duration": info.duration})
        assert len(manifest[split_name]) > 0, f"no DEMAND envs available for {split_name}"
    assert set(NOISE_TRAIN).isdisjoint(NOISE_VAL)
    assert set(NOISE_TRAIN).isdisjoint(NOISE_TEST)
    assert set(NOISE_VAL).isdisjoint(NOISE_TEST)
    return manifest


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    speech_manifest, duration, train_speakers = build_speech_manifest()
    noise_manifest = build_noise_manifest()

    for split in ["train", "val", "test"]:
        print(f"speech[{split}]: {len(speech_manifest[split])} utterances, "
              f"{duration[split]/3600:.2f}h")
    for split in ["train", "val", "test"]:
        total = sum(n["duration"] for n in noise_manifest[split])
        print(f"noise[{split}]: {len(noise_manifest[split])} files, {total/60:.1f} min")

    print("train speakers:", train_speakers)

    with open(os.path.join(OUT_DIR, "speech_split.json"), "w") as f:
        json.dump(speech_manifest, f, indent=2)
    with open(os.path.join(OUT_DIR, "noise_split.json"), "w") as f:
        json.dump(noise_manifest, f, indent=2)

    # leakage assertions across the two dimensions we control
    train_spk_set = {u["speaker"] for u in speech_manifest["train"]}
    val_spk_set = {u["speaker"] for u in speech_manifest["val"]}
    test_spk_set = {u["speaker"] for u in speech_manifest["test"]}
    assert train_spk_set.isdisjoint(val_spk_set)
    assert train_spk_set.isdisjoint(test_spk_set)
    assert val_spk_set.isdisjoint(test_spk_set)
    print("LEAKAGE CHECK PASSED: speaker sets disjoint across train/val/test, noise-env sets disjoint across train/val/test.")
