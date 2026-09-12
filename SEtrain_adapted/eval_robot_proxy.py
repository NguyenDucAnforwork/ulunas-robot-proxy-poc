"""
Full 6-condition evaluation (N0, P0, F_GENERIC, F_GENERIC_ASR, F_PROXY_ROBOT,
F_PROXY_ROBOT_ASR) on a NEW robot-proxy-domain synthetic test set (same LibriSpeech test
speech as M1, mixed with the robot-proxy noise pool -- 50/35/15 generic/UAV/procedural,
distance/angle proxy). This is genuinely new evaluation work: F_GENERIC/F_GENERIC_ASR were
only ever evaluated on the GENERIC test set in M1; RQ3 requires evaluating them on the
ROBOT-PROXY test set too, for a same-test-set comparison against F_PROXY_ROBOT(_ASR).

Adds a second frozen ASR backend (whisper-tiny.en) as a generalization check on WER, per
the brief -- primary WER is still wav2vec2-base-960h (same backend used in the ASR loss).
"""
import csv
import json
import os
import random
import sys

import jiwer
import numpy as np
import soundfile as sf
import torch
from pystoi import stoi as stoi_fn
from transformers import WhisperForConditionalGeneration, WhisperProcessor

sys.path.insert(0, os.path.dirname(__file__))
from models.ulunas import ULUNAS
from generic_dataset import mix_snr
from robot_proxy_dataset import distance_angle_proxy, DISTANCE_PROXY_M, ANGLE_PROXY_DEG, SOURCE_WEIGHTS
from dnsmos_onnxruntime import DNSMOSOnnxRuntime
from eval_generic import si_sdr, ASRDecoder, bootstrap_ci, paired_bootstrap_delta_ci

FS = 16000
MANIFEST_DIR = "/content/project/data/manifests"
SC_ROOT = "/content/project/data/speech_commands/extracted"
PRETRAINED_CKPT = "/content/repos/ul-unas/checkpoints/model_trained_on_dns3.tar"
FINETUNED_DIR = "/content/project/checkpoints"
OUT_DIR = "/content/project/evaluation"
AUDIO_SAMPLE_DIR = "/content/project/audio_samples_robot_proxy"
SEED = 54321
N_BOOTSTRAP = 2000

device = torch.device("cuda:0")

CONDITIONS = ["N0", "P0", "F_GENERIC", "F_GENERIC_ASR", "F_PROXY_ROBOT", "F_PROXY_ROBOT_ASR"]


def load_model(tag):
    model = ULUNAS().to(device).eval()
    if tag == "P0":
        ckpt = torch.load(PRETRAINED_CKPT, map_location=device)
    else:
        ckpt = torch.load(f"{FINETUNED_DIR}/{tag}.tar", map_location=device)
    model.load_state_dict(ckpt["model"])
    return model


@torch.inference_mode()
def enhance(model, noisy_np):
    x = torch.from_numpy(noisy_np).unsqueeze(0).to(device)
    return model(x).squeeze(0).cpu().numpy()


class WhisperDecoder:
    def __init__(self):
        self.processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
        self.model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en").to(device).eval()
        self.forced_decoder_ids = self.processor.get_decoder_prompt_ids(language="en", task="transcribe")

    @torch.inference_mode()
    def transcribe(self, wav_np):
        inputs = self.processor(wav_np, sampling_rate=FS, return_tensors="pt").input_features.to(device)
        ids = self.model.generate(inputs, forced_decoder_ids=self.forced_decoder_ids, max_new_tokens=128)
        text = self.processor.batch_decode(ids, skip_special_tokens=True)[0]
        return text.strip().upper()


def build_robot_proxy_synth_test_set():
    with open(f"{MANIFEST_DIR}/speech_split.json") as f:
        speech_items = json.load(f)["test"]
    with open(f"{MANIFEST_DIR}/robot_proxy_noise_split.json") as f:
        noise_items = json.load(f)["test"]
    noise_by_source = {"generic": [], "uav": [], "procedural": []}
    for n in noise_items:
        noise_by_source[n["source"]].append(n)

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

        items.append({"idx": i, "utt_id": sp["utt_id"], "transcript": sp["transcript"],
                     "noise_source": src, "robot_state": noise_item["robot_state"],
                     "snr_db": snr_db, "distance_m": distance_m, "angle_deg": angle_deg,
                     "mix": mix, "clean": clean_ref})
    return items


def build_robot_proxy_command_test_set(n_words=30, per_word=10):
    with open(f"{SC_ROOT}/testing_list.txt") as f:
        lines = [l.strip() for l in f if l.strip()]
    by_word = {}
    for l in lines:
        w = l.split("/")[0]
        by_word.setdefault(w, []).append(l)
    words = sorted(w for w in by_word if w != "_background_noise_")[:n_words]

    with open(f"{MANIFEST_DIR}/robot_proxy_noise_split.json") as f:
        noise_items = json.load(f)["test"]
    noise_by_source = {"generic": [], "uav": [], "procedural": []}
    for n in noise_items:
        noise_by_source[n["source"]].append(n)

    rng = random.Random(SEED + 1)
    items = []
    for w in words:
        picks = rng.sample(by_word[w], min(per_word, len(by_word[w])))
        for p in picks:
            path = f"{SC_ROOT}/{p}"
            clean, sr = sf.read(path, dtype="float32")
            src = rng.choices(list(SOURCE_WEIGHTS.keys()), weights=list(SOURCE_WEIGHTS.values()), k=1)[0]
            noise_item = rng.choice(noise_by_source[src])
            snr_db = rng.uniform(-5, 15)
            distance_m = rng.choice(DISTANCE_PROXY_M)
            angle_deg = rng.choice(ANGLE_PROXY_DEG)
            n_info = sf.info(noise_item["path"])
            n_start = rng.randint(0, max(0, n_info.frames - len(clean)))
            noise_seg, _ = sf.read(noise_item["path"], dtype="float32", start=n_start,
                                    stop=min(n_info.frames, n_start + len(clean)))
            if len(noise_seg) < len(clean):
                noise_seg = np.pad(noise_seg, (0, len(clean) - len(noise_seg)))
            noise_seg = distance_angle_proxy(noise_seg, distance_m, angle_deg, rng)
            mix, clean_ref = mix_snr(clean, noise_seg, snr_db, rng)
            items.append({"word": w, "mix": mix, "clean": clean_ref, "snr_db": snr_db,
                         "noise_source": src, "robot_state": noise_item["robot_state"]})
    return items


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(AUDIO_SAMPLE_DIR, exist_ok=True)

    print("Building robot-proxy synthetic test set...")
    synth_items = build_robot_proxy_synth_test_set()
    print(f"  {len(synth_items)} items")
    print("Building robot-proxy command test set...")
    cmd_items = build_robot_proxy_command_test_set()
    print(f"  {len(cmd_items)} items")

    dnsmos = DNSMOSOnnxRuntime()
    asr = ASRDecoder()
    whisper = WhisperDecoder()

    models = {}
    for tag in CONDITIONS:
        if tag == "N0":
            models[tag] = None
        else:
            models[tag] = load_model(tag)

    per_condition = {tag: {"dnsmos": [], "wer_ref": [], "wer_hyp": [], "wer_hyp_whisper": [],
                           "sisdr": [], "stoi": [], "snr_db": [], "noise_source": [], "robot_state": []}
                     for tag in CONDITIONS}
    per_condition_cmd = {tag: {"word": [], "hyp": []} for tag in CONDITIONS}

    print("Evaluating synthetic test set (6 conditions)...", flush=True)
    for i_item, item in enumerate(synth_items):
        if i_item % 25 == 0:
            print(f"  progress: {i_item}/{len(synth_items)}", flush=True)
        for tag in CONDITIONS:
            enhanced = item["mix"] if tag == "N0" else enhance(models[tag], item["mix"])
            enhanced = enhanced[:len(item["clean"])]
            clean = item["clean"][:len(enhanced)]

            d = dnsmos(enhanced, FS)
            pc = per_condition[tag]
            pc["dnsmos"].append(d)
            pc["sisdr"].append(float(si_sdr(clean, enhanced)))
            try:
                s = float(stoi_fn(clean, enhanced, FS, extended=False))
            except Exception:
                s = float("nan")
            pc["stoi"].append(s)
            hyp = asr.transcribe(enhanced)
            hyp_w = whisper.transcribe(enhanced)
            pc["wer_ref"].append(item["transcript"] or "")
            pc["wer_hyp"].append(hyp)
            pc["wer_hyp_whisper"].append(hyp_w)
            pc["snr_db"].append(item["snr_db"])
            pc["noise_source"].append(item["noise_source"])
            pc["robot_state"].append(item["robot_state"])

    with open(f"{OUT_DIR}/checkpoint_robot_proxy_synthetic_raw.json", "w") as f:
        json.dump(per_condition, f, indent=2, default=lambda o: float(o))
    print(f"Checkpoint saved: {OUT_DIR}/checkpoint_robot_proxy_synthetic_raw.json")

    print("Evaluating command test set (6 conditions)...", flush=True)
    for item in cmd_items:
        for tag in CONDITIONS:
            enhanced = item["mix"] if tag == "N0" else enhance(models[tag], item["mix"])
            enhanced = enhanced[:len(item["clean"])]
            hyp = asr.transcribe(enhanced)
            per_condition_cmd[tag]["word"].append(item["word"])
            per_condition_cmd[tag]["hyp"].append(hyp)

    # ---- aggregate ----
    results = {"synthetic_test_n": len(synth_items), "command_test_n": len(cmd_items),
               "conditions": {}}

    for tag in CONDITIONS:
        pc = per_condition[tag]
        sig = [d["SIG"] for d in pc["dnsmos"]]
        bak = [d["BAK"] for d in pc["dnsmos"]]
        ovrl = [d["OVRL"] for d in pc["dnsmos"]]
        wer_overall = jiwer.wer(pc["wer_ref"], pc["wer_hyp"])
        wer_overall_whisper = jiwer.wer(pc["wer_ref"], pc["wer_hyp_whisper"])
        per_utt_wer = [jiwer.wer([r], [h]) if r else float("nan") for r, h in zip(pc["wer_ref"], pc["wer_hyp"])]
        per_utt_wer_whisper = [jiwer.wer([r], [h]) if r else float("nan") for r, h in zip(pc["wer_ref"], pc["wer_hyp_whisper"])]

        cmd = per_condition_cmd[tag]
        exact = [1.0 if h.strip().lower() == w.lower() else 0.0 for w, h in zip(cmd["word"], cmd["hyp"])]
        deletions = [1.0 if h.strip() == "" else 0.0 for h in cmd["hyp"]]
        substitutions = [1.0 if (h.strip() != "" and h.strip().lower() != w.lower()) else 0.0
                          for w, h in zip(cmd["word"], cmd["hyp"])]

        results["conditions"][tag] = {
            "dnsmos_SIG_mean": float(np.mean(sig)), "dnsmos_BAK_mean": float(np.mean(bak)),
            "dnsmos_OVRL_mean": float(np.mean(ovrl)),
            "sisdr_mean": float(np.mean(pc["sisdr"])), "stoi_mean": float(np.nanmean(pc["stoi"])),
            "wer_corpus_primary_wav2vec2": float(wer_overall),
            "wer_corpus_secondary_whisper_tiny_en": float(wer_overall_whisper),
            "wer_mean_per_utt_primary": float(np.nanmean(per_utt_wer)),
            "wer_mean_per_utt_secondary": float(np.nanmean(per_utt_wer_whisper)),
            "command_exact_match_acc": float(np.mean(exact)),
            "command_deletion_rate": float(np.mean(deletions)),
            "command_substitution_rate": float(np.mean(substitutions)),
        }
        results["conditions"][tag]["_raw"] = {
            "sig": sig, "bak": bak, "ovrl": ovrl, "sisdr": pc["sisdr"], "stoi": pc["stoi"],
            "per_utt_wer": per_utt_wer, "per_utt_wer_whisper": per_utt_wer_whisper,
            "cmd_exact": exact, "snr_db": pc["snr_db"], "noise_source": pc["noise_source"],
            "robot_state": pc["robot_state"],
        }

    # ---- breakdown by SNR bucket and noise source (RQ1 requirement) ----
    snr_bins = [(-5, 0), (0, 5), (5, 10), (10, 15)]
    breakdown = {}
    for tag in CONDITIONS:
        raw = results["conditions"][tag]["_raw"]
        breakdown[tag] = {"by_snr_bin": {}, "by_noise_source": {}, "by_robot_state": {}}
        snr_arr = np.array(raw["snr_db"])
        wer_arr = np.array(raw["per_utt_wer"])
        ovrl_arr = np.array(raw["ovrl"])
        for lo, hi in snr_bins:
            mask = (snr_arr >= lo) & (snr_arr < hi)
            if mask.sum() > 0:
                breakdown[tag]["by_snr_bin"][f"{lo}_{hi}dB"] = {
                    "n": int(mask.sum()), "wer_mean": float(np.nanmean(wer_arr[mask])),
                    "ovrl_mean": float(np.mean(ovrl_arr[mask]))}
        for src in ["generic", "uav", "procedural"]:
            mask = np.array([s == src for s in raw["noise_source"]])
            if mask.sum() > 0:
                breakdown[tag]["by_noise_source"][src] = {
                    "n": int(mask.sum()), "wer_mean": float(np.nanmean(wer_arr[mask])),
                    "ovrl_mean": float(np.mean(ovrl_arr[mask]))}
        for state in set(raw["robot_state"]):
            mask = np.array([s == state for s in raw["robot_state"]])
            if mask.sum() > 0:
                breakdown[tag]["by_robot_state"][state] = {
                    "n": int(mask.sum()), "wer_mean": float(np.nanmean(wer_arr[mask])),
                    "ovrl_mean": float(np.mean(ovrl_arr[mask]))}
    results["breakdown"] = breakdown

    # ---- paired bootstrap CI: key comparisons for RQ1/RQ2/RQ3 ----
    def do_ci(tag_a, tag_b, metric_keys):
        out = {}
        ra = results["conditions"][tag_a]["_raw"]
        rb = results["conditions"][tag_b]["_raw"]
        for metric in metric_keys:
            delta, lo, hi = paired_bootstrap_delta_ci(ra[metric], rb[metric])
            out[f"delta_{metric}"] = {"delta": delta, "ci95_lo": lo, "ci95_hi": hi}
        return out

    metric_keys = ["sig", "bak", "ovrl", "sisdr", "stoi", "per_utt_wer"]
    ci = {
        "RQ1_P0_vs_N0": do_ci("P0", "N0", metric_keys),
        "RQ1_F_GENERIC_vs_N0": do_ci("F_GENERIC", "N0", metric_keys),
        "RQ1_F_PROXY_ROBOT_vs_N0": do_ci("F_PROXY_ROBOT", "N0", metric_keys),
        "RQ2_F_GENERIC_ASR_vs_F_GENERIC": do_ci("F_GENERIC_ASR", "F_GENERIC", metric_keys),
        "RQ2_F_PROXY_ROBOT_ASR_vs_F_PROXY_ROBOT": do_ci("F_PROXY_ROBOT_ASR", "F_PROXY_ROBOT", metric_keys),
        "RQ3_F_PROXY_ROBOT_vs_F_GENERIC": do_ci("F_PROXY_ROBOT", "F_GENERIC", metric_keys),
        "RQ3_F_PROXY_ROBOT_ASR_vs_F_GENERIC_ASR": do_ci("F_PROXY_ROBOT_ASR", "F_GENERIC_ASR", metric_keys),
    }
    results["paired_bootstrap_ci"] = ci

    raw_dump = {tag: results["conditions"][tag].pop("_raw") for tag in CONDITIONS}
    with open(f"{OUT_DIR}/evaluation_robot_proxy_raw_per_utterance.json", "w") as f:
        json.dump(raw_dump, f, indent=2, default=lambda o: float(o))
    with open(f"{OUT_DIR}/evaluation_robot_proxy_results.json", "w") as f:
        json.dump(results, f, indent=2, default=lambda o: float(o))

    with open(f"{OUT_DIR}/evaluation_robot_proxy_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "metric", "value"])
        for tag in CONDITIONS:
            for k, v in results["conditions"][tag].items():
                w.writerow([tag, k, v])

    # ---- audio samples ----
    print("Saving audio comparison samples...")
    n_samples = min(10, len(synth_items))
    picked = random.Random(SEED).sample(synth_items, n_samples)
    for i, item in enumerate(picked):
        sf.write(f"{AUDIO_SAMPLE_DIR}/sample{i:02d}_clean.wav", item["clean"], FS)
        sf.write(f"{AUDIO_SAMPLE_DIR}/sample{i:02d}_noisy.wav", item["mix"], FS)
        for tag in ["P0", "F_GENERIC", "F_GENERIC_ASR", "F_PROXY_ROBOT", "F_PROXY_ROBOT_ASR"]:
            enh = enhance(models[tag], item["mix"])[:len(item["clean"])]
            sf.write(f"{AUDIO_SAMPLE_DIR}/sample{i:02d}_{tag}.wav", enh, FS)
        with open(f"{AUDIO_SAMPLE_DIR}/sample{i:02d}_meta.json", "w") as f:
            json.dump({"utt_id": item["utt_id"], "transcript": item["transcript"],
                      "noise_source": item["noise_source"], "robot_state": item["robot_state"],
                      "snr_db": item["snr_db"], "distance_m": item["distance_m"],
                      "angle_deg": item["angle_deg"]}, f, indent=2)

    print("=== ROBOT-PROXY EVAL DONE ===")
    print(json.dumps({k: v for k, v in results.items() if k not in ("conditions",)}, indent=2)[:2000])


if __name__ == "__main__":
    main()
