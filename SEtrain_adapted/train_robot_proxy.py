"""
Fine-tune UL-UNAS on the ROBOT-PROXY domain (LibriSpeech clean speech -- SAME pool as
F_GENERIC -- mixed with 50% generic/35% public-UAV/15% procedural noise per
robot_proxy_dataset.py), producing F_PROXY_ROBOT (SE-loss only) or F_PROXY_ROBOT_ASR
(SE-loss + frozen-ASR loss). Mirrors train_generic.py's fine-tune procedure exactly (same
checkpoint-loading, BN-freeze, optimizer/LR) -- only the dataset differs. Deterministic
per-index mixture sampling (see robot_proxy_dataset.py) guarantees F_PROXY_ROBOT and
F_PROXY_ROBOT_ASR see the identical mixture sequence, satisfying the fair-comparison
requirement (same manifest, same sample order, same seed, same optimizer/steps -- only
the loss differs).
"""
import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "models"))

from models.ulunas import ULUNAS
from loss_factory import HybridLoss
from robot_proxy_dataset import RobotProxyDataset
from asr_loss import FrozenCTCLoss, gradient_scale_probe
from train_generic import set_seed, freeze_bn_running_stats, build_model, SEED, PRETRAINED_CKPT

MANIFEST_DIR = "/content/project/data/manifests"
LOG_DIR = "/content/project/training_logs"
CKPT_DIR = "/content/project/checkpoints"


def config_hash(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]


def manifest_hash(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def make_loaders(batch_size, segment_seconds, with_transcript):
    speech_path = f"{MANIFEST_DIR}/speech_split.json"
    noise_path = f"{MANIFEST_DIR}/robot_proxy_noise_split.json"
    train_ds = RobotProxyDataset(speech_path, noise_path, split="train",
                                 segment_seconds=segment_seconds, virtual_length=100000,
                                 with_transcript=with_transcript)
    val_ds = RobotProxyDataset(speech_path, noise_path, split="val",
                               segment_seconds=segment_seconds, with_transcript=with_transcript)
    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=2, drop_last=False)
    return train_loader, val_loader, manifest_hash(speech_path), manifest_hash(noise_path)


def run(variant, total_steps, batch_size, segment_seconds, benchmark_only, lambda_asr_arg, out_tag,
        lamda_ri=30, lamda_mag=70, lamda_sisnr=1.0):
    set_seed(SEED)
    device = torch.device("cuda:0")
    with_asr = variant == "se_asr"

    model = build_model(device)
    model.train()
    n_bn = freeze_bn_running_stats(model)

    # lamda_ri/lamda_mag default to HybridLoss's own original values (30/70) -- only differ
    # when explicitly overridden (e.g. --lamda_mag 35 for a reduced-suppression ablation).
    se_loss_fn = HybridLoss(lamda_ri=lamda_ri, lamda_mag=lamda_mag, lamda_sisnr=lamda_sisnr).to(device)
    asr_loss_fn = FrozenCTCLoss(device) if with_asr else None

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    train_loader, val_loader, speech_hash, noise_hash = make_loaders(
        batch_size, segment_seconds, with_transcript=with_asr)
    train_iter = iter(train_loader)

    os.makedirs(f"{LOG_DIR}/{out_tag}", exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)

    cfg = {"variant": variant, "total_steps": total_steps, "batch_size": batch_size,
           "segment_seconds": segment_seconds, "seed": SEED, "lr": 1e-5,
           "pretrained_ckpt": PRETRAINED_CKPT, "speech_manifest_hash": speech_hash,
           "noise_manifest_hash": noise_hash, "lamda_ri": lamda_ri, "lamda_mag": lamda_mag,
           "lamda_sisnr": lamda_sisnr}
    cfg_hash = config_hash(cfg)
    cfg["config_hash"] = cfg_hash
    with open(f"{LOG_DIR}/{out_tag}/config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"[{out_tag}] config_hash={cfg_hash} speech_manifest_hash={speech_hash} noise_manifest_hash={noise_hash}")

    torch.cuda.reset_peak_memory_stats(device)
    lambda_asr = lambda_asr_arg
    grad_probe_info = None

    if with_asr and lambda_asr is None:
        noisy0, clean0, trans0, meta0 = next(train_iter)
        noisy0, clean0 = noisy0.to(device), clean0.to(device)
        lambda_asr, gnorm_se, gnorm_asr = gradient_scale_probe(
            model, se_loss_fn, asr_loss_fn, noisy0, clean0, list(trans0), target_ratio=0.2)
        grad_probe_info = {"lambda_asr": lambda_asr, "grad_norm_se": gnorm_se,
                            "grad_norm_asr_raw": gnorm_asr, "target_ratio": 0.2}
        print(f"[gradient-scale probe] grad_norm_se={gnorm_se:.6f} grad_norm_asr_raw={gnorm_asr:.6f} "
              f"-> lambda_asr={lambda_asr:.6f}")
        with open(f"{LOG_DIR}/{out_tag}/lambda_probe.json", "w") as f:
            json.dump(grad_probe_info, f, indent=2)
        model.train()
        freeze_bn_running_stats(model)
        train_iter = iter(train_loader)

    loss_log = []
    grad_check_logged = False
    nan_inf_detected = False
    best_val_loss = float("inf")
    t0 = time.time()

    for step in range(1, total_steps + 1):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        if with_asr:
            noisy, clean, transcripts, meta = batch
            transcripts = list(transcripts)
        else:
            noisy, clean = batch
        noisy, clean = noisy.to(device), clean.to(device)

        optimizer.zero_grad()
        enhanced = model(noisy)
        l_se = se_loss_fn(enhanced, clean)

        if with_asr:
            l_asr = asr_loss_fn(enhanced, transcripts)
            loss = l_se + lambda_asr * l_asr
        else:
            l_asr = torch.tensor(0.0)
            loss = l_se

        if not torch.isfinite(loss):
            nan_inf_detected = True
            print(f"[{out_tag}] WARNING: non-finite loss at step {step}, skipping update")
            optimizer.zero_grad()
            continue

        loss.backward()

        if with_asr and not grad_check_logged:
            first_layer = model.encoder.en_convs[0].ops[1].weight
            grad_norm_first_layer = first_layer.grad.detach().norm().item() if first_layer.grad is not None else 0.0
            with open(f"{LOG_DIR}/{out_tag}/grad_check.json", "w") as f:
                json.dump({"first_layer_grad_norm_with_asr_loss": grad_norm_first_layer,
                          "nonzero": grad_norm_first_layer > 0.0}, f, indent=2)
            print(f"[grad check] first-layer grad norm with L_ASR included = {grad_norm_first_layer:.8f} "
                  f"({'NONZERO -- OK' if grad_norm_first_layer > 0 else 'ZERO -- BUG'})")
            grad_check_logged = True

        torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
        optimizer.step()

        loss_log.append({"step": step, "loss_total": loss.item(), "loss_se": l_se.item(),
                         "loss_asr": (l_asr.item() if with_asr else None),
                         "elapsed_s": time.time() - t0})

        if step % 50 == 0 or step == 1:
            print(f"[{out_tag}] step {step}/{total_steps} loss_total={loss.item():.4f} "
                  f"loss_se={l_se.item():.4f} "
                  f"{'loss_asr=%.4f' % l_asr.item() if with_asr else ''} "
                  f"elapsed={time.time()-t0:.1f}s", flush=True)

        if step % 1000 == 0 and not benchmark_only:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for vb in val_loader:
                    vnoisy, vclean = (vb[0], vb[1]) if not with_asr else (vb[0], vb[1])
                    vnoisy, vclean = vnoisy.to(device), vclean.to(device)
                    venh = model(vnoisy)
                    val_losses.append(se_loss_fn(venh, vclean).item())
            val_loss = float(np.mean(val_losses))
            print(f"[{out_tag}] step {step} VAL SE-loss={val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({"model": model.state_dict(), "variant": variant, "step": step},
                          f"{CKPT_DIR}/{out_tag}_best.tar")
            model.train()
            freeze_bn_running_stats(model)

        if benchmark_only and step >= total_steps:
            break

    total_wall_s = time.time() - t0
    peak_vram_mb = torch.cuda.max_memory_allocated(device) / 1e6
    steps_per_sec = total_steps / total_wall_s

    resource_usage = {"variant": variant, "total_steps": total_steps, "batch_size": batch_size,
                      "segment_seconds": segment_seconds, "total_wall_clock_s": total_wall_s,
                      "steps_per_sec": steps_per_sec, "peak_vram_mb": peak_vram_mb,
                      "lambda_asr": lambda_asr, "n_batchnorm_frozen": n_bn,
                      "nan_inf_detected": nan_inf_detected, "best_val_se_loss": best_val_loss,
                      "config_hash": cfg_hash}
    with open(f"{LOG_DIR}/{out_tag}/resource_usage.json", "w") as f:
        json.dump(resource_usage, f, indent=2)

    with open(f"{LOG_DIR}/{out_tag}/loss_curve.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step", "loss_total", "loss_se", "loss_asr", "elapsed_s"])
        w.writeheader()
        w.writerows(loss_log)

    if not benchmark_only:
        torch.save({"model": model.state_dict(), "variant": variant, "total_steps": total_steps},
                   f"{CKPT_DIR}/{out_tag}.tar")

    print(f"=== DONE [{out_tag}] steps={total_steps} wall_clock={total_wall_s:.1f}s "
          f"({steps_per_sec:.3f} steps/s) peak_vram={peak_vram_mb:.1f}MB nan_inf={nan_inf_detected} ===")
    return resource_usage


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=["se_only", "se_asr"], required=True)
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--segment_seconds", type=float, default=4.0)
    p.add_argument("--benchmark", action="store_true")
    p.add_argument("--lambda_asr", type=float, default=None)
    p.add_argument("--out_tag", type=str, required=True)
    p.add_argument("--lamda_ri", type=float, default=30, help="HybridLoss RI-loss weight (default 30, same as M1/F_PROXY_ROBOT)")
    p.add_argument("--lamda_mag", type=float, default=70, help="HybridLoss magnitude-loss weight (default 70, same as M1/F_PROXY_ROBOT)")
    p.add_argument("--lamda_sisnr", type=float, default=1.0, help="HybridLoss SI-SNR term weight (default 1.0, same as M1/F_PROXY_ROBOT -- the SISNR term previously had this fixed, unweighted)")
    args = p.parse_args()

    run(args.variant, args.steps, args.batch_size, args.segment_seconds,
        args.benchmark, args.lambda_asr, args.out_tag, args.lamda_ri, args.lamda_mag, args.lamda_sisnr)
