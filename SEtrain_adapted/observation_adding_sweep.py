"""
Observation-adding sweep: output = (1-alpha)*enhanced + alpha*noisy, for
alpha in {0, 0.05, 0.10, 0.20, 0.30}, on F_PROXY_ROBOT (the recommended SE-only model --
ASR loss showed no material benefit, see EXPERIMENT_REPORT.md RQ2).

Goal: test whether blending back a small amount of the noisy signal mitigates the SIG-BAK
over-suppression tradeoff found in QC_FAILURE_ANALYSIS.md §4.2 (concentrated in UAV
motor_high and procedural fan noise), without destroying the WER/SI-SDR/STOI gains.

Per the brief: never pick alpha from DNSMOS OVRL alone -- report SIG/BAK/OVRL/WER/SI-SDR/STOI
separately for every alpha, and break out the UAV/fan subset specifically since that's where
the tradeoff was found to concentrate.
"""
import json
import os
import sys

import jiwer
import numpy as np
import torch
from pystoi import stoi as stoi_fn

sys.path.insert(0, os.path.dirname(__file__))
from models.ulunas import ULUNAS
from eval_robot_proxy import build_robot_proxy_synth_test_set, si_sdr, ASRDecoder
from dnsmos_onnxruntime import DNSMOSOnnxRuntime

FS = 16000
CKPT = "/content/project/checkpoints/F_PROXY_ROBOT.tar"
OUT_DIR = "/content/project/evaluation"
ALPHAS = [0.0, 0.05, 0.10, 0.20, 0.30]

device = torch.device("cuda:0")


@torch.inference_mode()
def enhance(model, noisy_np):
    x = torch.from_numpy(noisy_np).unsqueeze(0).to(device)
    return model(x).squeeze(0).cpu().numpy()


def main():
    print("Loading F_PROXY_ROBOT and rebuilding the same deterministic test set...")
    model = ULUNAS().to(device).eval()
    model.load_state_dict(torch.load(CKPT, map_location=device)["model"])

    items = build_robot_proxy_synth_test_set()
    dnsmos = DNSMOSOnnxRuntime()
    asr = ASRDecoder()

    per_alpha = {a: {"sig": [], "bak": [], "ovrl": [], "sisdr": [], "stoi": [],
                     "wer_ref": [], "wer_hyp": [], "noise_source": [], "robot_state": []}
                for a in ALPHAS}

    for i, item in enumerate(items):
        if i % 50 == 0:
            print(f"  progress: {i}/{len(items)}", flush=True)
        enh = enhance(model, item["mix"])[:len(item["clean"])]
        noisy = item["mix"][:len(enh)]
        clean = item["clean"][:len(enh)]

        for a in ALPHAS:
            blended = (1 - a) * enh + a * noisy
            d = dnsmos(blended, FS)
            pc = per_alpha[a]
            pc["sig"].append(d["SIG"]); pc["bak"].append(d["BAK"]); pc["ovrl"].append(d["OVRL"])
            pc["sisdr"].append(float(si_sdr(clean, blended)))
            try:
                pc["stoi"].append(float(stoi_fn(clean, blended, FS, extended=False)))
            except Exception:
                pc["stoi"].append(float("nan"))
            hyp = asr.transcribe(blended)
            pc["wer_ref"].append(item["transcript"] or "")
            pc["wer_hyp"].append(hyp)
            pc["noise_source"].append(item["noise_source"])
            pc["robot_state"].append(item["robot_state"])

    results = {}
    for a in ALPHAS:
        pc = per_alpha[a]
        per_utt_wer = [jiwer.wer([r], [h]) if r else float("nan") for r, h in zip(pc["wer_ref"], pc["wer_hyp"])]
        wer_corpus = jiwer.wer(pc["wer_ref"], pc["wer_hyp"])

        # UAV motor_high + procedural fan subset (where the SIG-BAK tradeoff concentrates)
        tradeoff_mask = [(s == "uav" and rs == "motor_high") or (s == "procedural" and rs == "fan")
                         for s, rs in zip(pc["noise_source"], pc["robot_state"])]
        idx_tradeoff = [i for i, m in enumerate(tradeoff_mask) if m]

        results[f"alpha_{a}"] = {
            "n": len(items),
            "sig_mean": float(np.mean(pc["sig"])), "bak_mean": float(np.mean(pc["bak"])),
            "ovrl_mean": float(np.mean(pc["ovrl"])), "sisdr_mean": float(np.mean(pc["sisdr"])),
            "stoi_mean": float(np.nanmean(pc["stoi"])), "wer_corpus": float(wer_corpus),
            "wer_mean_per_utt": float(np.nanmean(per_utt_wer)),
            "tradeoff_subset_n": len(idx_tradeoff),
            "tradeoff_subset_sig_mean": float(np.mean([pc["sig"][i] for i in idx_tradeoff])) if idx_tradeoff else None,
            "tradeoff_subset_bak_mean": float(np.mean([pc["bak"][i] for i in idx_tradeoff])) if idx_tradeoff else None,
            "tradeoff_subset_ovrl_mean": float(np.mean([pc["ovrl"][i] for i in idx_tradeoff])) if idx_tradeoff else None,
            "tradeoff_subset_wer_mean": float(np.nanmean([per_utt_wer[i] for i in idx_tradeoff])) if idx_tradeoff else None,
        }
        print(f"alpha={a}: SIG={results[f'alpha_{a}']['sig_mean']:.3f} "
              f"BAK={results[f'alpha_{a}']['bak_mean']:.3f} "
              f"OVRL={results[f'alpha_{a}']['ovrl_mean']:.3f} "
              f"WER={results[f'alpha_{a}']['wer_corpus']:.4f} "
              f"| tradeoff-subset(n={len(idx_tradeoff)}) SIG={results[f'alpha_{a}']['tradeoff_subset_sig_mean']:.3f} "
              f"BAK={results[f'alpha_{a}']['tradeoff_subset_bak_mean']:.3f} "
              f"WER={results[f'alpha_{a}']['tradeoff_subset_wer_mean']:.4f}", flush=True)

    with open(f"{OUT_DIR}/observation_adding_sweep_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("=== SWEEP DONE ===")


if __name__ == "__main__":
    main()
