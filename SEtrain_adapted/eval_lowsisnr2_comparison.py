"""
Second step of the lamda_sisnr ablation: pushes lamda_sisnr further down (0.3->0.1,
F_PROXY_ROBOT_LOWSISNR2) to test whether the §4.3d improvement (lamda_sisnr 1.0->0.3) is
monotonic and grows, or plateaus/reverses. Compares all four conditions -- N0, F_PROXY_ROBOT
(lamda_sisnr=1.0, original), F_PROXY_ROBOT_LOWSISNR (lamda_sisnr=0.3),
F_PROXY_ROBOT_LOWSISNR2 (lamda_sisnr=0.1) -- on the same 563-utterance robot-proxy test set,
with paired bootstrap CI for LOWSISNR2 vs original AND LOWSISNR2 vs LOWSISNR (0.3), overall
and on the UAV-motor_high + procedural-fan "tradeoff subset" from QC_FAILURE_ANALYSIS.md §4.2.
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
from eval_robot_proxy import build_robot_proxy_synth_test_set, si_sdr, ASRDecoder, paired_bootstrap_delta_ci
from dnsmos_onnxruntime import DNSMOSOnnxRuntime

FS = 16000
OUT_DIR = "/content/project/evaluation"
CONDITIONS = ["N0", "F_PROXY_ROBOT", "F_PROXY_ROBOT_LOWSISNR", "F_PROXY_ROBOT_LOWSISNR2"]
device = torch.device("cuda:0")


def load_model(tag):
    model = ULUNAS().to(device).eval()
    ckpt = torch.load(f"/content/project/checkpoints/{tag}.tar", map_location=device)
    model.load_state_dict(ckpt["model"])
    return model


@torch.inference_mode()
def enhance(model, noisy_np):
    x = torch.from_numpy(noisy_np).unsqueeze(0).to(device)
    return model(x).squeeze(0).cpu().numpy()


def main():
    items = build_robot_proxy_synth_test_set()
    dnsmos = DNSMOSOnnxRuntime()
    asr = ASRDecoder()
    models = {tag: (None if tag == "N0" else load_model(tag)) for tag in CONDITIONS}

    per_cond = {tag: {"sig": [], "bak": [], "ovrl": [], "sisdr": [], "stoi": [],
                     "per_utt_wer": [], "noise_source": [], "robot_state": []}
               for tag in CONDITIONS}

    for i, item in enumerate(items):
        if i % 50 == 0:
            print(f"  progress: {i}/{len(items)}", flush=True)
        for tag in CONDITIONS:
            enh = item["mix"] if tag == "N0" else enhance(models[tag], item["mix"])
            enh = enh[:len(item["clean"])]
            clean = item["clean"][:len(enh)]
            d = dnsmos(enh, FS)
            pc = per_cond[tag]
            pc["sig"].append(d["SIG"]); pc["bak"].append(d["BAK"]); pc["ovrl"].append(d["OVRL"])
            pc["sisdr"].append(float(si_sdr(clean, enh)))
            try:
                pc["stoi"].append(float(stoi_fn(clean, enh, FS, extended=False)))
            except Exception:
                pc["stoi"].append(float("nan"))
            hyp = asr.transcribe(enh)
            ref = item["transcript"] or ""
            pc["per_utt_wer"].append(jiwer.wer([ref], [hyp]) if ref else float("nan"))
            pc["noise_source"].append(item["noise_source"])
            pc["robot_state"].append(item["robot_state"])

    tradeoff_mask = [(s == "uav" and rs == "motor_high") or (s == "procedural" and rs == "fan")
                     for s, rs in zip(per_cond["N0"]["noise_source"], per_cond["N0"]["robot_state"])]
    idx_tradeoff = [i for i, m in enumerate(tradeoff_mask) if m]

    results = {"n": len(items), "tradeoff_subset_n": len(idx_tradeoff), "conditions": {}}
    for tag in CONDITIONS:
        pc = per_cond[tag]
        results["conditions"][tag] = {
            "sig_mean": float(np.mean(pc["sig"])), "bak_mean": float(np.mean(pc["bak"])),
            "ovrl_mean": float(np.mean(pc["ovrl"])), "sisdr_mean": float(np.mean(pc["sisdr"])),
            "stoi_mean": float(np.nanmean(pc["stoi"])),
            "wer_mean": float(np.nanmean(pc["per_utt_wer"])),
            "tradeoff_sig_mean": float(np.mean([pc["sig"][i] for i in idx_tradeoff])),
            "tradeoff_bak_mean": float(np.mean([pc["bak"][i] for i in idx_tradeoff])),
            "tradeoff_ovrl_mean": float(np.mean([pc["ovrl"][i] for i in idx_tradeoff])),
            "tradeoff_wer_mean": float(np.nanmean([pc["per_utt_wer"][i] for i in idx_tradeoff])),
        }

    def ci_for(a_tag, b_tag, metric, idx=None):
        a = per_cond[a_tag][metric]
        b = per_cond[b_tag][metric]
        if idx is not None:
            a = [a[i] for i in idx]; b = [b[i] for i in idx]
        delta, lo, hi = paired_bootstrap_delta_ci(a, b)
        return {"delta": delta, "ci95_lo": lo, "ci95_hi": hi}

    metrics = ["sig", "bak", "ovrl", "sisdr", "stoi", "per_utt_wer"]
    results["ci_lowsisnr2_vs_original_overall"] = {
        m: ci_for("F_PROXY_ROBOT_LOWSISNR2", "F_PROXY_ROBOT", m) for m in metrics
    }
    results["ci_lowsisnr2_vs_original_tradeoff_subset"] = {
        m: ci_for("F_PROXY_ROBOT_LOWSISNR2", "F_PROXY_ROBOT", m, idx_tradeoff) for m in metrics
    }
    results["ci_lowsisnr2_vs_lowsisnr03_overall"] = {
        m: ci_for("F_PROXY_ROBOT_LOWSISNR2", "F_PROXY_ROBOT_LOWSISNR", m) for m in metrics
    }
    results["ci_lowsisnr2_vs_lowsisnr03_tradeoff_subset"] = {
        m: ci_for("F_PROXY_ROBOT_LOWSISNR2", "F_PROXY_ROBOT_LOWSISNR", m, idx_tradeoff) for m in metrics
    }

    with open(f"{OUT_DIR}/lowsisnr2_comparison_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps({k: v for k, v in results.items() if k != "conditions"}, indent=2))
    for tag in CONDITIONS:
        print(tag, results["conditions"][tag])
    print("=== DONE ===")


if __name__ == "__main__":
    main()
