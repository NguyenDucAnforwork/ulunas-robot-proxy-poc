"""
M1 evaluation: N0 (noisy), P0 (pretrained UL-UNAS), F_GENERIC, F_GENERIC_ASR.

- Synthetic paired test set (LibriSpeech test speakers x DEMAND test noise, FIXED seed so all
  4 conditions see the identical degraded input): DNSMOS (onnxruntime-direct), WER
  (facebook/wav2vec2-base-960h greedy CTC decode), SI-SDR, STOI (decision #23: intrusive
  metrics mandatory here since we have a time-aligned clean reference).
- Command test set (Google Speech Commands official test split x same noise sources): command
  exact-match accuracy, deletion rate, substitution rate (decision #24: replaces FRR/FAR).
- Paired bootstrap CI of every metric vs N0 and vs P0.
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
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

sys.path.insert(0, os.path.dirname(__file__))
from models.ulunas import ULUNAS
from generic_dataset import mix_snr
from dnsmos_onnxruntime import DNSMOSOnnxRuntime

FS = 16000
MANIFEST_DIR = "/content/project/data/manifests"
SC_ROOT = "/content/project/data/speech_commands/extracted"
PRETRAINED_CKPT = "/content/repos/ul-unas/checkpoints/model_trained_on_dns3.tar"
FINETUNED_DIR = "/content/project/checkpoints"
OUT_DIR = "/content/project/evaluation"
AUDIO_SAMPLE_DIR = "/content/project/audio_samples"
SEED = 12345
N_BOOTSTRAP = 2000

device = torch.device("cuda:0")


def si_sdr(reference, estimate, eps=1e-8):
    reference = reference - np.mean(reference)
    estimate = estimate - np.mean(estimate)
    proj = (np.sum(estimate * reference) / (np.sum(reference ** 2) + eps)) * reference
    noise = estimate - proj
    return 10 * np.log10((np.sum(proj ** 2) + eps) / (np.sum(noise ** 2) + eps))


def load_model(tag):
    model = ULUNAS().to(device).eval()
    if tag == "P0":
        ckpt = torch.load(PRETRAINED_CKPT, map_location=device)
        model.load_state_dict(ckpt["model"])
    else:
        ckpt = torch.load(f"{FINETUNED_DIR}/{tag}.tar", map_location=device)
        model.load_state_dict(ckpt["model"])
    return model


@torch.inference_mode()
def enhance(model, noisy_np):
    x = torch.from_numpy(noisy_np).unsqueeze(0).to(device)
    y = model(x)
    return y.squeeze(0).cpu().numpy()


def build_synthetic_test_set():
    with open(f"{MANIFEST_DIR}/speech_split.json") as f:
        speech_items = json.load(f)["test"]
    with open(f"{MANIFEST_DIR}/noise_split.json") as f:
        noise_items = json.load(f)["test"]

    rng = random.Random(SEED)
    items = []
    for i, sp in enumerate(speech_items):
        noise = rng.choice(noise_items)
        snr_db = rng.uniform(-5, 15)
        clean, _ = sf.read(sp["path"], dtype="float32")
        n_info = sf.info(noise["path"])
        n_start = rng.randint(0, max(0, n_info.frames - len(clean)))
        noise_seg, _ = sf.read(noise["path"], dtype="float32", start=n_start,
                                stop=min(n_info.frames, n_start + len(clean)))
        if len(noise_seg) < len(clean):
            noise_seg = np.pad(noise_seg, (0, len(clean) - len(noise_seg)))
        mix, clean_ref = mix_snr(clean, noise_seg, snr_db, rng)
        items.append({
            "idx": i, "utt_id": sp["utt_id"], "transcript": sp["transcript"],
            "noise_env": noise["env"], "snr_db": snr_db,
            "mix": mix, "clean": clean_ref,
        })
    return items


def build_command_test_set(n_words=30, per_word=10):
    with open(f"{SC_ROOT}/testing_list.txt") as f:
        lines = [l.strip() for l in f if l.strip()]
    by_word = {}
    for l in lines:
        w = l.split("/")[0]
        by_word.setdefault(w, []).append(l)
    words = sorted(w for w in by_word if w != "_background_noise_")[:n_words]

    with open(f"{MANIFEST_DIR}/noise_split.json") as f:
        noise_items = json.load(f)["test"]

    rng = random.Random(SEED + 1)
    items = []
    for w in words:
        picks = rng.sample(by_word[w], min(per_word, len(by_word[w])))
        for p in picks:
            path = f"{SC_ROOT}/{p}"
            clean, sr = sf.read(path, dtype="float32")
            assert sr == FS
            noise = rng.choice(noise_items)
            snr_db = rng.uniform(-5, 15)
            n_info = sf.info(noise["path"])
            n_start = rng.randint(0, max(0, n_info.frames - len(clean)))
            noise_seg, _ = sf.read(noise["path"], dtype="float32", start=n_start,
                                    stop=min(n_info.frames, n_start + len(clean)))
            if len(noise_seg) < len(clean):
                noise_seg = np.pad(noise_seg, (0, len(clean) - len(noise_seg)))
            mix, clean_ref = mix_snr(clean, noise_seg, snr_db, rng)
            items.append({"word": w, "mix": mix, "clean": clean_ref, "snr_db": snr_db,
                          "noise_env": noise["env"]})
    return items


class ASRDecoder:
    def __init__(self):
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").to(device).eval()

    @torch.inference_mode()
    def transcribe(self, wav_np):
        x = torch.from_numpy(wav_np).unsqueeze(0).to(device)
        x = (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-7)
        logits = self.model(input_values=x).logits
        pred_ids = torch.argmax(logits, dim=-1)
        text = self.processor.batch_decode(pred_ids)[0]
        return text.strip()


def bootstrap_ci(values, n=N_BOOTSTRAP, seed=0):
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return None, None, None
    means = [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n)]
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(np.mean(arr)), float(lo), float(hi)


def paired_bootstrap_delta_ci(values_a, values_b, n=N_BOOTSTRAP, seed=0):
    """CI on (mean(a) - mean(b)) using paired resampling (same utterance indices for both)."""
    rng = np.random.default_rng(seed)
    a = np.asarray(values_a, dtype=np.float64)
    b = np.asarray(values_b, dtype=np.float64)
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    if len(a) == 0:
        return None, None, None
    idx_pool = np.arange(len(a))
    deltas = []
    for _ in range(n):
        idx = rng.choice(idx_pool, size=len(idx_pool), replace=True)
        deltas.append((a[idx] - b[idx]).mean())
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return float(np.mean(a) - np.mean(b)), float(lo), float(hi)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(AUDIO_SAMPLE_DIR, exist_ok=True)

    print("Building fixed synthetic test set...")
    synth_items = build_synthetic_test_set()
    print(f"  {len(synth_items)} synthetic paired utterances")

    print("Building command test set...")
    cmd_items = build_command_test_set()
    print(f"  {len(cmd_items)} command utterances")

    dnsmos = DNSMOSOnnxRuntime(device="cuda")  # GPU was idle during the CPU-bound DNSMOS phase in the first run
    asr = ASRDecoder()

    conditions = ["N0", "P0", "F_GENERIC", "F_GENERIC_ASR"]
    models = {}
    for tag in conditions:
        if tag == "N0":
            models[tag] = None  # identity: enhanced = noisy
        else:
            models[tag] = load_model(tag)

    per_condition = {tag: {"dnsmos": [], "wer_ref": [], "wer_hyp": [], "sisdr": [], "stoi": []}
                     for tag in conditions}
    per_condition_cmd = {tag: {"word": [], "hyp": []} for tag in conditions}

    print("Evaluating synthetic test set...")
    for i_item, item in enumerate(synth_items):
        if i_item % 25 == 0:
            print(f"  synthetic progress: {i_item}/{len(synth_items)}", flush=True)
        for tag in conditions:
            enhanced = item["mix"] if tag == "N0" else enhance(models[tag], item["mix"])
            enhanced = enhanced[:len(item["clean"])]
            clean = item["clean"][:len(enhanced)]

            d = dnsmos(enhanced, FS)
            per_condition[tag]["dnsmos"].append(d)
            per_condition[tag]["sisdr"].append(float(si_sdr(clean, enhanced)))
            try:
                s = float(stoi_fn(clean, enhanced, FS, extended=False))
            except Exception:
                s = float("nan")
            per_condition[tag]["stoi"].append(s)
            hyp = asr.transcribe(enhanced)
            per_condition[tag]["wer_ref"].append(item["transcript"] or "")
            per_condition[tag]["wer_hyp"].append(hyp)

    # crash-safety checkpoint: save raw synthetic-set results before starting the command
    # set, so a bug in the (much cheaper) command-set/aggregation code can never lose the
    # expensive DNSMOS/ASR computation above again.
    with open(f"{OUT_DIR}/checkpoint_synthetic_raw.json", "w") as f:
        json.dump(per_condition, f, indent=2, default=lambda o: float(o))
    print(f"Checkpoint saved: {OUT_DIR}/checkpoint_synthetic_raw.json")

    print("Evaluating command test set...")
    for item in cmd_items:
        for tag in conditions:
            enhanced = item["mix"] if tag == "N0" else enhance(models[tag], item["mix"])
            enhanced = enhanced[:len(item["clean"])]
            hyp = asr.transcribe(enhanced)
            per_condition_cmd[tag]["word"].append(item["word"])
            per_condition_cmd[tag]["hyp"].append(hyp)

    # ---- aggregate ----
    results = {"synthetic_test_n": len(synth_items), "command_test_n": len(cmd_items),
               "conditions": {}}

    for tag in conditions:
        pc = per_condition[tag]
        sig = [d["SIG"] for d in pc["dnsmos"]]
        bak = [d["BAK"] for d in pc["dnsmos"]]
        ovrl = [d["OVRL"] for d in pc["dnsmos"]]
        wer_overall = jiwer.wer(pc["wer_ref"], pc["wer_hyp"])
        per_utt_wer = [jiwer.wer([r], [h]) if r else float("nan") for r, h in zip(pc["wer_ref"], pc["wer_hyp"])]

        cmd = per_condition_cmd[tag]
        exact = [1.0 if h.strip().lower() == w.lower() else 0.0 for w, h in zip(cmd["word"], cmd["hyp"])]
        deletions = [1.0 if h.strip() == "" else 0.0 for h in cmd["hyp"]]
        substitutions = [1.0 if (h.strip() != "" and h.strip().lower() != w.lower()) else 0.0
                          for w, h in zip(cmd["word"], cmd["hyp"])]

        results["conditions"][tag] = {
            "dnsmos_SIG_mean": float(np.mean(sig)), "dnsmos_SIG_median": float(np.median(sig)),
            "dnsmos_BAK_mean": float(np.mean(bak)), "dnsmos_BAK_median": float(np.median(bak)),
            "dnsmos_OVRL_mean": float(np.mean(ovrl)), "dnsmos_OVRL_median": float(np.median(ovrl)),
            "sisdr_mean": float(np.mean(pc["sisdr"])), "sisdr_median": float(np.median(pc["sisdr"])),
            "stoi_mean": float(np.nanmean(pc["stoi"])), "stoi_median": float(np.nanmedian(pc["stoi"])),
            "wer_corpus": float(wer_overall),
            "wer_mean_per_utt": float(np.nanmean(per_utt_wer)),
            "command_exact_match_acc": float(np.mean(exact)),
            "command_deletion_rate": float(np.mean(deletions)),
            "command_substitution_rate": float(np.mean(substitutions)),
        }
        # keep raw per-utterance arrays for bootstrap CI section below
        results["conditions"][tag]["_raw"] = {
            "sig": sig, "bak": bak, "ovrl": ovrl, "sisdr": pc["sisdr"], "stoi": pc["stoi"],
            "per_utt_wer": per_utt_wer, "cmd_exact": exact,
        }

    # paired bootstrap CI: each fine-tuned/pretrained condition vs N0, and vs P0
    ci_results = {}
    for tag in ["P0", "F_GENERIC", "F_GENERIC_ASR"]:
        ci_results[tag] = {}
        for baseline in ["N0", "P0"] if tag != "P0" else ["N0"]:
            for metric in ["sig", "bak", "ovrl", "sisdr", "stoi"]:
                a = results["conditions"][tag]["_raw"][metric]
                b = results["conditions"][baseline]["_raw"][metric]
                delta, lo, hi = paired_bootstrap_delta_ci(a, b)
                ci_results[tag][f"delta_{metric}_vs_{baseline}"] = {"delta": delta, "ci95_lo": lo, "ci95_hi": hi}
            a_wer = results["conditions"][tag]["_raw"]["per_utt_wer"]
            b_wer = results["conditions"][baseline]["_raw"]["per_utt_wer"]
            delta, lo, hi = paired_bootstrap_delta_ci(a_wer, b_wer)
            ci_results[tag][f"delta_wer_vs_{baseline}"] = {"delta": delta, "ci95_lo": lo, "ci95_hi": hi}

    results["paired_bootstrap_ci"] = ci_results

    # strip raw arrays before writing the compact JSON/CSV (keep them in a separate file for audit)
    raw_dump = {tag: results["conditions"][tag].pop("_raw") for tag in conditions}
    with open(f"{OUT_DIR}/evaluation_raw_per_utterance.json", "w") as f:
        json.dump(raw_dump, f, indent=2, default=lambda o: float(o))

    with open(f"{OUT_DIR}/evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2, default=lambda o: float(o))

    with open(f"{OUT_DIR}/evaluation_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "metric", "value"])
        for tag in conditions:
            for k, v in results["conditions"][tag].items():
                w.writerow([tag, k, v])

    # ---- audio comparison samples (>=10) ----
    print("Saving audio comparison samples...")
    n_samples = min(10, len(synth_items))
    picked = random.Random(SEED).sample(synth_items, n_samples)
    for i, item in enumerate(picked):
        sf.write(f"{AUDIO_SAMPLE_DIR}/sample{i:02d}_clean.wav", item["clean"], FS)
        sf.write(f"{AUDIO_SAMPLE_DIR}/sample{i:02d}_noisy.wav", item["mix"], FS)
        for tag in ["P0", "F_GENERIC", "F_GENERIC_ASR"]:
            enh = enhance(models[tag], item["mix"])[:len(item["clean"])]
            sf.write(f"{AUDIO_SAMPLE_DIR}/sample{i:02d}_{tag}.wav", enh, FS)
        with open(f"{AUDIO_SAMPLE_DIR}/sample{i:02d}_meta.json", "w") as f:
            json.dump({"utt_id": item["utt_id"], "transcript": item["transcript"],
                       "noise_env": item["noise_env"], "snr_db": item["snr_db"]}, f, indent=2)

    print("=== EVAL DONE ===")
    print(json.dumps(results, indent=2)[:3000])


if __name__ == "__main__":
    main()
