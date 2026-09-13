"""
Build the combined robot-proxy noise manifest: 50% generic (reuse M1's DEMAND split) +
35% public UAV ego-noise (KU Leuven, CC-BY-NC-SA-4.0) + 15% procedural motor/fan/servo noise.

Split assignment (documented, no leakage):
- generic (DEMAND): reuses M1's exact train/dev/test env split (already leak-checked).
- UAV: each RPM session is an entire, distinct recording -> RPM4000->train, RPM5000->dev,
  RPM6000->test. No chunk from one RPM session appears in another split.
- procedural: different RNG seed per split (train=1000,dev=2000,test=3000 base seeds) ->
  no literal sample overlap possible across splits, by construction.

Target ratio is by DURATION, not file count. Actual achievable ratio is reported and used
(instruction: if public data is insufficient, adjust the ratio but record the real one).
"""
import json
import os

import soundfile as sf

MANIFEST_DIR = "/content/project/data/manifests"
UAV_DIR = "/content/project/data/uav_kuleuven"
PROC_DIR = "/content/project/data/procedural_noise"
OUT_PATH = f"{MANIFEST_DIR}/robot_proxy_noise_split.json"

TARGET_RATIO = {"generic": 0.50, "uav": 0.35, "procedural": 0.15}


def main():
    with open(f"{MANIFEST_DIR}/noise_split.json") as f:
        generic = json.load(f)  # {"train":[...],"val":[...],"test":[...]}

    uav_files = {
        "train": [("train_motor_low_16k.wav", "motor_low")],
        "val": [("dev_motor_mid_16k.wav", "motor_mid")],
        "test": [("test_motor_high_16k.wav", "motor_high")],
    }

    with open(f"{PROC_DIR}/manifest.json") as f:
        proc_manifest = json.load(f)["clips"]
    proc_by_split = {"train": [], "val": [], "test": []}
    split_name_map = {"train": "train", "dev": "val", "test": "test"}
    for c in proc_manifest:
        proc_by_split[split_name_map[c["split"]]].append(c)

    combined = {"train": [], "val": [], "test": []}
    for split in ["train", "val", "test"]:
        for item in generic[split]:
            info = sf.info(item["path"])
            combined[split].append({"path": item["path"], "duration": info.duration,
                                    "source": "generic", "robot_state": "generic_background",
                                    "env": item.get("env")})
        for fname, state in uav_files[split]:
            path = f"{UAV_DIR}/{fname}"
            if not os.path.exists(path):
                print(f"[build_manifest] WARN missing UAV {path}, skipping (KU Leuven dl may have failed)")
                continue
            info = sf.info(path)
            combined[split].append({"path": path, "duration": info.duration,
                                    "source": "uav", "robot_state": state,
                                    "license": "CC-BY-NC-SA-4.0", "rpm_session": fname})
        for c in proc_by_split[split]:
            info = sf.info(c["path"])
            combined[split].append({"path": c["path"], "duration": info.duration,
                                    "source": "procedural", "robot_state": c["state"],
                                    "seed": c["seed"]})

    report = {"target_ratio": TARGET_RATIO, "actual_by_split": {}}
    for split in ["train", "val", "test"]:
        total = sum(c["duration"] for c in combined[split])
        by_source = {}
        for src in ["generic", "uav", "procedural"]:
            d = sum(c["duration"] for c in combined[split] if c["source"] == src)
            by_source[src] = {"duration_s": d, "ratio": d / total if total > 0 else 0}
        report["actual_by_split"][split] = {"total_duration_s": total, "by_source": by_source}

    with open(OUT_PATH, "w") as f:
        json.dump(combined, f, indent=2)
    with open(f"{MANIFEST_DIR}/robot_proxy_noise_ratio_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))

    # leakage check: no file path appears in more than one split
    all_paths = {}
    for split in ["train", "val", "test"]:
        for c in combined[split]:
            p = c["path"]
            assert p not in all_paths, f"LEAKAGE: {p} in both {all_paths[p]} and {split}"
            all_paths[p] = split
    print("LEAKAGE CHECK PASSED: no noise file appears in more than one split.")


if __name__ == "__main__":
    main()
