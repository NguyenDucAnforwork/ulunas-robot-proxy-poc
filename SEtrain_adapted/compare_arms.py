"""Aggregate the 4-arm sweep: read each arm's loss_log.json + resource_usage.json
and print a ranked table by proxy metric (final validation SE-loss, then SI-SDR).
Usage: python3 compare_arms.py TAG1 TAG2 ..."""
import json, os, sys

LOG_DIR = "/content/project/training_logs"


def last_val(tag):
    p = f"{LOG_DIR}/{tag}/loss_log.json"
    if not os.path.exists(p):
        return None
    log = json.load(open(p))
    vals = [e for e in log if "val_se_loss" in e]
    ru = {}
    rp = f"{LOG_DIR}/{tag}/resource_usage.json"
    if os.path.exists(rp):
        ru = json.load(open(rp))
    best = min((e["val_se_loss"] for e in vals), default=None)
    last = vals[-1] if vals else {}
    return {"tag": tag, "n_val": len(vals),
            "final_val_se": last.get("val_se_loss"),
            "final_val_sisdr": last.get("val_si_sdr"),
            "best_val_se": best,
            "steps_per_sec": ru.get("steps_per_sec"),
            "steps": ru.get("steps")}


if __name__ == "__main__":
    tags = sys.argv[1:] or ["arm_control", "arm_lr_sched", "arm_preserve", "arm_snr_curriculum"]
    rows = [r for r in (last_val(t) for t in tags) if r]
    rows.sort(key=lambda r: (r["best_val_se"] if r["best_val_se"] is not None else 1e9))
    print(f"{'arm':22s} {'best_val_SE':>12s} {'final_val_SE':>12s} {'final_SI-SDR':>12s} {'steps/s':>8s} {'steps':>7s}")
    base = next((r for r in rows if r["tag"].endswith("control")), None)
    for r in rows:
        d = ""
        if base and base["best_val_se"] and r["best_val_se"] is not None and r is not base:
            d = f"  (Δ vs control {r['best_val_se']-base['best_val_se']:+.4f})"
        print(f"{r['tag']:22s} {r['best_val_se']:>12.4f} {r['final_val_se']:>12.4f} "
              f"{r['final_val_sisdr']:>12.2f} {str(r['steps_per_sec'])[:6]:>8s} {str(r['steps']):>7s}{d}")
    json.dump(rows, open(f"{LOG_DIR}/sweep_comparison.json", "w"), indent=2)
    print(f"\nwrote {LOG_DIR}/sweep_comparison.json")
