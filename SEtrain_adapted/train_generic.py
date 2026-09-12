"""
Fine-tune UL-UNAS on the GENERIC domain (LibriSpeech + DEMAND), producing F_GENERIC (SE-loss only)
or F_GENERIC_ASR (SE-loss + frozen-CTC ASR loss), implementing decisions #8/#11/#12/#19/#21/#22:

- Load ONLY model weights from the pretrained DNS3 checkpoint (optimizer/scheduler NOT restored).
- Fresh optimizer, LR = 1e-5 constant (no warmup/decay -- short fine-tune, see SPEC_PLAN §3.2).
- BatchNorm running statistics frozen (kept at pretrained values) throughout fine-tuning.
- Frozen ASR (`facebook/wav2vec2-base-960h`) in `.eval()` mode but NOT `torch.no_grad()`.
- F_GENERIC and F_GENERIC_ASR share: init checkpoint, clean-speech pool, step count, batch,
  seed, optimizer, LR schedule. GPU-time is NOT required to match -- measured and reported.
"""
import argparse
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
from generic_dataset import GenericSEDataset
from asr_loss import FrozenCTCLoss, gradient_scale_probe

SEED = 43
PRETRAINED_CKPT = "/content/repos/ul-unas/checkpoints/model_trained_on_dns3.tar"
MANIFEST_DIR = "/content/project/data/manifests"
LOG_DIR = "/content/project/training_logs"
CKPT_DIR = "/content/project/checkpoints"


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def freeze_bn_running_stats(model):
    """Put every BatchNorm2d submodule in eval() mode so running_mean/running_var are frozen
    at their pretrained values, while the rest of the model (and BN's own affine weight/bias,
    if trainable) stays in train() mode. Must be re-applied after every model.train() call,
    since nn.Module.train() recursively flips all children back to train mode."""
    n = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
            n += 1
    return n


def build_model(device):
    model = ULUNAS().to(device)
    ckpt = torch.load(PRETRAINED_CKPT, map_location=device)
    model.load_state_dict(ckpt["model"])  # weights ONLY -- optimizer/scheduler in ckpt ignored
    return model


def make_loaders(batch_size, segment_seconds, with_transcript):
    train_ds = GenericSEDataset(
        f"{MANIFEST_DIR}/speech_split.json", f"{MANIFEST_DIR}/noise_split.json",
        split="train", segment_seconds=segment_seconds, virtual_length=100000,
        with_transcript=with_transcript,
    )
    val_ds = GenericSEDataset(
        f"{MANIFEST_DIR}/speech_split.json", f"{MANIFEST_DIR}/noise_split.json",
        split="val", segment_seconds=segment_seconds, with_transcript=with_transcript,
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=2, drop_last=False)
    return train_loader, val_loader


def run(variant, total_steps, batch_size, segment_seconds, benchmark_only, lambda_asr_arg, out_tag):
    set_seed(SEED)
    device = torch.device("cuda:0")
    with_asr = variant == "se_asr"

    model = build_model(device)
    model.train()
    n_bn = freeze_bn_running_stats(model)

    se_loss_fn = HybridLoss().to(device)
    asr_loss_fn = FrozenCTCLoss(device) if with_asr else None

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    train_loader, val_loader = make_loaders(batch_size, segment_seconds, with_transcript=with_asr)
    train_iter = iter(train_loader)

    os.makedirs(f"{LOG_DIR}/{out_tag}", exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)

    torch.cuda.reset_peak_memory_stats(device)
    lambda_asr = lambda_asr_arg
    grad_probe_info = None

    if with_asr and lambda_asr is None:
        noisy0, clean0, trans0 = next(train_iter)
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
        train_iter = iter(train_loader)  # restart so the probe batch isn't "spent" oddly

    loss_log = []
    grad_check_logged = False
    t0 = time.time()

    for step in range(1, total_steps + 1):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        if with_asr:
            noisy, clean, transcripts = batch
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

        loss.backward()

        if with_asr and not grad_check_logged:
            first_layer = model.encoder.en_convs[0].ops[1].weight
            grad_norm_first_layer = first_layer.grad.detach().norm().item() if first_layer.grad is not None else 0.0
            with open(f"{LOG_DIR}/{out_tag}/grad_check.json", "w") as f:
                json.dump({
                    "first_layer_grad_norm_with_asr_loss": grad_norm_first_layer,
                    "nonzero": grad_norm_first_layer > 0.0,
                }, f, indent=2)
            print(f"[grad check] first-layer grad norm with L_ASR included = {grad_norm_first_layer:.8f} "
                  f"({'NONZERO -- OK' if grad_norm_first_layer > 0 else 'ZERO -- BUG'})")
            grad_check_logged = True

        torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
        optimizer.step()

        loss_log.append({
            "step": step, "loss_total": loss.item(), "loss_se": l_se.item(),
            "loss_asr": (l_asr.item() if with_asr else None),
            "elapsed_s": time.time() - t0,
        })

        if step % 50 == 0 or step == 1:
            print(f"[{out_tag}] step {step}/{total_steps} loss_total={loss.item():.4f} "
                  f"loss_se={l_se.item():.4f} "
                  f"{'loss_asr=%.4f' % l_asr.item() if with_asr else ''} "
                  f"elapsed={time.time()-t0:.1f}s")

        if benchmark_only and step >= total_steps:
            break

    total_wall_s = time.time() - t0
    peak_vram_mb = torch.cuda.max_memory_allocated(device) / 1e6
    steps_per_sec = total_steps / total_wall_s

    resource_usage = {
        "variant": variant, "total_steps": total_steps, "batch_size": batch_size,
        "segment_seconds": segment_seconds, "total_wall_clock_s": total_wall_s,
        "steps_per_sec": steps_per_sec, "peak_vram_mb": peak_vram_mb,
        "lambda_asr": lambda_asr, "n_batchnorm_frozen": n_bn,
    }
    with open(f"{LOG_DIR}/{out_tag}/resource_usage.json", "w") as f:
        json.dump(resource_usage, f, indent=2)

    import csv
    with open(f"{LOG_DIR}/{out_tag}/loss_curve.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step", "loss_total", "loss_se", "loss_asr", "elapsed_s"])
        w.writeheader()
        w.writerows(loss_log)

    if not benchmark_only:
        torch.save({"model": model.state_dict(), "variant": variant, "total_steps": total_steps},
                   f"{CKPT_DIR}/{out_tag}.tar")

    print(f"=== DONE [{out_tag}] steps={total_steps} wall_clock={total_wall_s:.1f}s "
          f"({steps_per_sec:.3f} steps/s) peak_vram={peak_vram_mb:.1f}MB ===")
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
    args = p.parse_args()

    run(args.variant, args.steps, args.batch_size, args.segment_seconds,
        args.benchmark, args.lambda_asr, args.out_tag)
