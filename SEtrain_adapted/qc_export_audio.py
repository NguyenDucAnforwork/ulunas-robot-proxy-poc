"""
Export actual audio (noisy/clean/enhanced) + spectrograms for the worst-10, best-10, and
SIG-BAK-tradeoff-top-10 cases identified by qc_analysis.py, for model_tag F_PROXY_ROBOT_ASR
(the primary robot-proxy model). Re-derives the same deterministic synthetic test set (same
seed as eval_robot_proxy.py) to recover the exact audio for the flagged indices.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
from scipy import signal

sys.path.insert(0, os.path.dirname(__file__))
from eval_robot_proxy import build_robot_proxy_synth_test_set, load_model, enhance, FS

QC_DIR = "/content/project/qc_analysis"
AUDIO_OUT = f"{QC_DIR}/case_audio"


def spectrogram_png(wav, path, title):
    f, t, Sxx = signal.spectrogram(wav, fs=FS, nperseg=512, noverlap=256)
    plt.figure(figsize=(6, 3))
    plt.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-12), shading="gouraud", cmap="magma")
    plt.ylabel("Hz"); plt.xlabel("s"); plt.title(title)
    plt.colorbar(label="dB")
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close()


def main(model_tag="F_PROXY_ROBOT_ASR"):
    os.makedirs(AUDIO_OUT, exist_ok=True)
    print("Rebuilding deterministic synthetic test set (same seed)...")
    items = build_robot_proxy_synth_test_set()
    model = load_model(model_tag)

    for case_file in ["worst_10_cases", "best_10_cases", "sig_bak_tradeoff_top10"]:
        with open(f"{QC_DIR}/{case_file}.json") as f:
            cases = json.load(f)
        for rank, case in enumerate(cases):
            idx = case["idx"]
            item = items[idx]
            enh = enhance(model, item["mix"])[:len(item["clean"])]
            clean = item["clean"][:len(enh)]
            noisy = item["mix"][:len(enh)]

            prefix = f"{AUDIO_OUT}/{case_file}_{rank:02d}_idx{idx}"
            sf.write(f"{prefix}_clean.wav", clean, FS)
            sf.write(f"{prefix}_noisy.wav", noisy, FS)
            sf.write(f"{prefix}_enhanced.wav", enh, FS)
            spectrogram_png(clean, f"{prefix}_spec_clean.png", f"clean (idx={idx})")
            spectrogram_png(noisy, f"{prefix}_spec_noisy.png",
                            f"noisy SNR={item['snr_db']:.1f}dB state={item['robot_state']}")
            spectrogram_png(enh, f"{prefix}_spec_enhanced.png",
                            f"enhanced WER={case['wer']:.2f}")

            with open(f"{prefix}_meta.json", "w") as f:
                json.dump({**case, "utt_id": item["utt_id"], "transcript": item["transcript"],
                          "distance_m": item.get("distance_m"), "angle_deg": item.get("angle_deg")},
                         f, indent=2)
        print(f"{case_file}: exported {len(cases)} cases")


if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else "F_PROXY_ROBOT_ASR"
    main(tag)
