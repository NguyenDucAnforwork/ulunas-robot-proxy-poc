"""
Perceptual verdict for the 4-arm sweep (EXPERIMENT_SWEEP.md §4.2): DNSMOS P.835
SIG/BAK/OVRL + SI-SDR + STOI on 150 deterministic, paired robot-proxy test mixtures,
same mixtures for every arm.

Tests the `preserve` arm's actual hypothesis (does the speech-preservation loss reduce
SIG over-suppression vs `arm_control`?) which the fast SI-SDR proxy in §4.1 cannot answer
-- SI-SDR is exactly the waveform-fidelity quantity `preserve` deliberately trades away.

Test-set construction reuses eval_robot_proxy.py's build_robot_proxy_synth_test_set()
logic (same SEED, same speech_split.json/robot_proxy_noise_split.json test splits) but
capped to the first N_TEST items for a fast, deterministic, paired 150-mixture subset --
this is a reduced screen, not the full 563-utterance evaluation.
"""
import json
import os
import random
import sys

import numpy as np
import soundfile as sf
import torch
from pystoi import stoi as stoi_fn

sys.path.insert(0, os.path.dirname(__file__))
from models.ulunas import ULUNAS
from generic_dataset import mix_snr
from robot_proxy_dataset import distance_angle_proxy, DISTANCE_PROXY_M, ANGLE_PROXY_DEG, SOURCE_WEIGHTS
from eval_generic import si_sdr, paired_bootstrap_delta_ci
from dnsmos_onnxruntime import DNSMOSOnnxRuntime

FS = 16000
MANIFEST_DIR = "/content/project/data/manifests"
CKPT_DIR = "/content/project/checkpoints/experiments_20260913"
OUT_DIR = "/content/project/evaluation"
SEED = 54321
N_TEST = 150

ARMS = ["arm_control", "arm_lr_sched", "arm_preserve", "arm_snr_curriculum"]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_model(tag):
    model = ULUNAS().to(device).eval()
    ckpt = torch.load(f"{CKPT_DIR}/{tag}_best.tar", map_location=device)
    model.load_state_dict(ckpt["model"])
    return model


@torch.inference_mode()
def enhance(model, noisy_np):
    x = torch.from_numpy(noisy_np).unsqueeze(0).to(device)
    return model(x).squeeze(0).cpu().numpy()


def build_test_set(n=N_TEST):
    with open(f"{MANIFEST_DIR}/speech_split.json") as f:
        speech_items = json.load(f)["test"][:n]
    with open(f"{MANIFEST_DIR}/robot_proxy_noise_split.json") as f:
        noise_items = json.load(f)["test"]
    noise_by_source = {"generic": [], "uav": [], "procedural": []}
    for it in noise_items:
        noise_by_source[it["source"]].append(it)

    rng = random.Random(SEED)
    items = []
    for i, sp in enumerate(speech_items):
        src = rng.choices(list(SOURCE_WEIGHTS.keys()), weights=list(SOURCE_WEIGHTS.values()), k=1)[0]
        noise_item = rng.choice(noise_by_source[src])
        snr_db = rng.uniform(-5, 15)
        distance_m = rng.choice(DISTANCE_PROXY_M)
        angle_deg = rng.choice(ANGLE_PROXY_DEG)

        clean, _ = sf.read(sp["path"], dtype="float32")
        n_info = sf.info(noise_item["path"])
        n_start = rng.randint(0, max(0, n_info.frames - len(clean)))
        noise_seg, _ = sf.read(noise_item["path"], dtype="float32", start=n_start,
                                stop=min(n_info.frames, n_start + len(clean)))
        if len(noise_seg) < len(clean):
            noise_seg = np.pad(noise_seg, (0, len(clean) - len(noise_seg)))
        noise_seg = distance_angle_proxy(noise_seg, distance_m, angle_deg, rng)
        mix, clean_ref = mix_snr(clean, noise_seg, snr_db, rng)

        items.append({"idx": i, "utt_id": sp["utt_id"], "noise_source": src,
                      "robot_state": noise_item["robot_state"], "snr_db": snr_db,
                      "mix": mix, "clean": clean_ref})
    return items


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Building {N_TEST}-mixture deterministic paired test set...", flush=True)
    items = build_test_set()
    print(f"  {len(items)} items", flush=True)

    dnsmos = DNSMOSOnnxRuntime()
    models = {tag: load_model(tag) for tag in ARMS}

    per_arm = {tag: {"sig": [], "bak": [], "ovrl": [], "sisdr": [], "stoi": []} for tag in ARMS}

    for i_item, item in enumerate(items):
        if i_item % 25 == 0:
            print(f"  progress: {i_item}/{len(items)}", flush=True)
        for tag in ARMS:
            enhanced = enhance(models[tag], item["mix"])
            enhanced = enhanced[:len(item["clean"])]
            clean = item["clean"][:len(enhanced)]
            d = dnsmos(enhanced, FS)
            pa = per_arm[tag]
            pa["sig"].append(d["SIG"])
            pa["bak"].append(d["BAK"])
            pa["ovrl"].append(d["OVRL"])
            pa["sisdr"].append(float(si_sdr(clean, enhanced)))
            try:
                pa["stoi"].append(float(stoi_fn(clean, enhanced, FS, extended=False)))
            except Exception:
                pa["stoi"].append(float("nan"))

    results = {"n_test": len(items), "seed": SEED, "conditions": {}}
    for tag in ARMS:
        pa = per_arm[tag]
        results["conditions"][tag] = {
            "dnsmos_SIG_mean": float(np.mean(pa["sig"])),
            "dnsmos_BAK_mean": float(np.mean(pa["bak"])),
            "dnsmos_OVRL_mean": float(np.mean(pa["ovrl"])),
            "sisdr_mean": float(np.mean(pa["sisdr"])),
            "stoi_mean": float(np.nanmean(pa["stoi"])),
            "_raw": pa,
        }

    ci = {}
    base = per_arm["arm_control"]
    for tag in ARMS:
        if tag == "arm_control":
            continue
        out = {}
        for metric in ["sig", "bak", "ovrl", "sisdr", "stoi"]:
            delta, lo, hi = paired_bootstrap_delta_ci(per_arm[tag][metric], base[metric])
            out[f"delta_{metric}"] = {"delta": delta, "ci95_lo": lo, "ci95_hi": hi}
        ci[f"{tag}_vs_control"] = out
    results["paired_bootstrap_ci_vs_control"] = ci

    with open(f"{OUT_DIR}/eval_arms_dnsmos_results.json", "w") as f:
        json.dump(results, f, indent=2, default=lambda o: float(o))

    print(f"\n{'arm':22s} {'SIG':>7s} {'BAK':>7s} {'OVRL':>7s} {'SI-SDR':>8s} {'STOI':>7s}")
    for tag in ARMS:
        c = results["conditions"][tag]
        print(f"{tag:22s} {c['dnsmos_SIG_mean']:>7.3f} {c['dnsmos_BAK_mean']:>7.3f} "
              f"{c['dnsmos_OVRL_mean']:>7.3f} {c['sisdr_mean']:>8.3f} {c['stoi_mean']:>7.4f}")

    print("\nPaired bootstrap CI95% vs arm_control:")
    for tag, out in ci.items():
        print(f"  {tag}: " + ", ".join(
            f"d{m.split('_')[1]}={v['delta']:+.4f}[{v['ci95_lo']:+.4f},{v['ci95_hi']:+.4f}]"
            for m, v in out.items()))

    print(f"\nWrote {OUT_DIR}/eval_arms_dnsmos_results.json")


if __name__ == "__main__":
    main()
