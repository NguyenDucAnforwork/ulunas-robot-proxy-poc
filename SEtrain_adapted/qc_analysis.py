"""
QC failure analysis on the real robot-proxy evaluation results (evaluation_robot_proxy_raw_
per_utterance.json + the synthetic test set metadata). Produces best/worst-10 case studies
with spectrograms, investigates the SIG-BAK tradeoff hypothesis, and slices failures by
robot_state/SNR/noise_source. Run AFTER eval_robot_proxy.py has produced its outputs.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

OUT_DIR = "/content/project/evaluation"
QC_DIR = "/content/project/qc_analysis"
FS = 16000


def load_data():
    with open(f"{OUT_DIR}/evaluation_robot_proxy_raw_per_utterance.json") as f:
        raw = json.load(f)
    with open(f"{OUT_DIR}/evaluation_robot_proxy_results.json") as f:
        results = json.load(f)
    return raw, results


def spectrogram_png(wav, path, title):
    f, t, Sxx = None, None, None
    from scipy import signal
    f, t, Sxx = signal.spectrogram(wav, fs=FS, nperseg=512, noverlap=256)
    plt.figure(figsize=(6, 3))
    plt.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-12), shading="gouraud", cmap="magma")
    plt.ylabel("Hz"); plt.xlabel("s"); plt.title(title)
    plt.colorbar(label="dB")
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close()


def main(model_tag="F_PROXY_ROBOT_ASR"):
    raw, results = load_data()
    os.makedirs(QC_DIR, exist_ok=True)

    model_raw = raw[model_tag]
    n0_raw = raw["N0"]

    sig = np.array(model_raw["sig"])
    bak = np.array(model_raw["bak"])
    ovrl = np.array(model_raw["ovrl"])
    wer = np.array(model_raw["per_utt_wer"])
    sig_n0 = np.array(n0_raw["sig"])

    delta_sig = sig - sig_n0
    delta_bak = bak - np.array(n0_raw["bak"])

    # SIG-BAK tradeoff hypothesis: cases where BAK improved a lot but SIG got WORSE
    tradeoff_score = delta_bak - delta_sig  # high = "BAK up but SIG down/less up" pattern
    worst_idx = np.argsort(wer)[-10:][::-1]  # highest WER = worst
    best_idx = np.argsort(wer)[:10]  # lowest WER = best
    tradeoff_idx = np.argsort(tradeoff_score)[-10:][::-1]

    def dump_case(idx_list, name):
        cases = []
        for i in idx_list:
            cases.append({
                "idx": int(i), "wer": float(wer[i]), "sig": float(sig[i]), "bak": float(bak[i]),
                "ovrl": float(ovrl[i]), "delta_sig_vs_N0": float(delta_sig[i]),
                "delta_bak_vs_N0": float(delta_bak[i]), "snr_db": model_raw["snr_db"][i],
                "noise_source": model_raw["noise_source"][i], "robot_state": model_raw["robot_state"][i],
            })
        with open(f"{QC_DIR}/{name}.json", "w") as f:
            json.dump(cases, f, indent=2)
        return cases

    worst_cases = dump_case(worst_idx, "worst_10_cases")
    best_cases = dump_case(best_idx, "best_10_cases")
    tradeoff_cases = dump_case(tradeoff_idx, "sig_bak_tradeoff_top10")

    # breakdown summaries already in results["breakdown"] -- re-surface the key ones
    breakdown_summary = {tag: results["breakdown"].get(tag, {}) for tag in
                         ["N0", "P0", "F_GENERIC", "F_GENERIC_ASR", "F_PROXY_ROBOT", "F_PROXY_ROBOT_ASR"]}

    with open(f"{QC_DIR}/breakdown_summary.json", "w") as f:
        json.dump(breakdown_summary, f, indent=2)

    # SIG-BAK scatter plot across all 6 conditions
    plt.figure(figsize=(6, 6))
    for tag in ["N0", "P0", "F_GENERIC", "F_GENERIC_ASR", "F_PROXY_ROBOT", "F_PROXY_ROBOT_ASR"]:
        r = raw[tag]
        plt.scatter(r["bak"], r["sig"], s=8, alpha=0.4, label=tag)
    plt.xlabel("BAK"); plt.ylabel("SIG"); plt.legend(fontsize=7)
    plt.title("SIG vs BAK across conditions (robot-proxy test set)")
    plt.tight_layout()
    plt.savefig(f"{QC_DIR}/sig_bak_scatter.png", dpi=120)
    plt.close()

    print(f"QC analysis for {model_tag} written to {QC_DIR}/")
    print(f"Worst-case mean WER: {np.mean([c['wer'] for c in worst_cases]):.3f}")
    print(f"Best-case mean WER: {np.mean([c['wer'] for c in best_cases]):.3f}")
    print(f"Top SIG-BAK-tradeoff case delta_bak={tradeoff_cases[0]['delta_bak_vs_N0']:.3f} "
          f"delta_sig={tradeoff_cases[0]['delta_sig_vs_N0']:.3f}")


if __name__ == "__main__":
    import sys
    tag = sys.argv[1] if len(sys.argv) > 1 else "F_PROXY_ROBOT_ASR"
    main(tag)
