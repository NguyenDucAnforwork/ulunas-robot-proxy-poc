"""
train_arm.py -- parameterized robot-proxy fine-tuning for the 90-min autonomous
optimization sweep (2026-09-13). One script, four arms, all sharing the exact
F_PROXY_ROBOT baseline (DNS3 init, batch 8, seed 43, Adam, HybridLoss lamda
30/70/1.0, 50/35/15 noise mix, SNR[-5,15]) EXCEPT for one deliberate delta each:

  arm=control        : baseline, no change (reference at this depth/seed)
  arm=lr_sched       : LR warmup(500 steps 1e-6->base) + cosine decay ->1e-6
  arm=preserve       : lamda_sisnr 1.0->0.3 + speech-preservation penalty on
                       over-attenuated (compressed) magnitude bins (Codex B2)
  arm=snr_curriculum : first --curr_switch steps SNR[5,15] (easy) then SNR[-5,15]

Runs under CUDA MPS so several arms share one A100 at near-full per-arm speed.
Proxy selection metric = validation SE-loss (+ SI-SDR on val). Full DNSMOS/WER
is a separate, slower confirmation step (eval_robot_proxy.py).

Reproduce: see EXPERIMENT_SWEEP.md.
"""
import argparse, json, math, os, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.ulunas import ULUNAS
from loss_factory import HybridLoss
from robot_proxy_dataset import RobotProxyDataset
from train_generic import set_seed, freeze_bn_running_stats, SEED, PRETRAINED_CKPT

MANIFEST_DIR = "/content/project/data/manifests"
LOG_DIR = "/content/project/training_logs"
CKPT_DIR = "/content/project/checkpoints"
FS = 16000


def build_model(device):
    model = ULUNAS().to(device)
    ckpt = torch.load(PRETRAINED_CKPT, map_location=device)
    model.load_state_dict(ckpt["model"])
    return model


def make_loader(split, batch_size, segment_seconds, snr_range, virtual_length):
    ds = RobotProxyDataset(
        f"{MANIFEST_DIR}/speech_split.json",
        f"{MANIFEST_DIR}/robot_proxy_noise_split.json",
        split=split, segment_seconds=segment_seconds,
        snr_range=snr_range, virtual_length=virtual_length,
    )
    nw = 4 if split == "train" else 2
    return DataLoader(ds, batch_size=batch_size, num_workers=nw,
                      drop_last=(split == "train"))


# ---- speech-preservation loss (arm=preserve) --------------------------------
_pres_win = None
def preservation_loss(pred, clean, n_fft=512, hop=256, win=512, c=0.3, eps=1e-8):
    """Penalize enhanced compressed-magnitude falling BELOW clean (over-suppression),
    not exceeding it. Directly targets the SIG regression from over-suppression."""
    global _pres_win
    if _pres_win is None or _pres_win.device != pred.device:
        _pres_win = torch.hann_window(win).to(pred.device)
    P = torch.stft(pred, n_fft, hop, win, _pres_win, return_complex=True)
    C = torch.stft(clean, n_fft, hop, win, _pres_win, return_complex=True)
    pm = torch.abs(P).clamp(eps) ** c
    cm = torch.abs(C).clamp(eps) ** c
    shortfall = torch.clamp(cm - pm, min=0.0)
    return torch.mean(shortfall ** 2)


def si_sdr(est, ref, eps=1e-8):
    ref = ref - ref.mean(dim=-1, keepdim=True)
    est = est - est.mean(dim=-1, keepdim=True)
    alpha = (torch.sum(est * ref, dim=-1, keepdim=True)) / (torch.sum(ref ** 2, dim=-1, keepdim=True) + eps)
    proj = alpha * ref
    noise = est - proj
    return torch.mean(10 * torch.log10((torch.sum(proj ** 2, dim=-1) + eps) / (torch.sum(noise ** 2, dim=-1) + eps)))


def lr_at(step, base_lr, total_steps, warmup=500, floor=1e-6):
    if step <= warmup:
        return floor + (base_lr - floor) * step / warmup
    prog = (step - warmup) / max(1, total_steps - warmup)
    return floor + 0.5 * (base_lr - floor) * (1 + math.cos(math.pi * prog))


def run(args):
    set_seed(SEED)
    device = torch.device("cuda:0")
    model = build_model(device); model.train()
    n_bn = freeze_bn_running_stats(model)

    lamda_sisnr = 0.3 if args.arm == "preserve" else 1.0
    se_loss_fn = HybridLoss(lamda_ri=30, lamda_mag=70, lamda_sisnr=lamda_sisnr).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.base_lr)

    # SNR curriculum: start easy
    init_snr = (5, 15) if args.arm == "snr_curriculum" else (-5, 15)
    train_loader = make_loader("train", args.batch_size, 4.0, init_snr, 100000)
    val_loader = make_loader("val", args.batch_size, 4.0, (-5, 15), 2000)
    train_iter = iter(train_loader)

    os.makedirs(f"{LOG_DIR}/{args.out_tag}", exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)
    cfg = vars(args).copy(); cfg.update({"seed": SEED, "n_bn_frozen": n_bn,
        "lamda_sisnr": lamda_sisnr, "init_snr": init_snr, "pretrained": PRETRAINED_CKPT})
    with open(f"{LOG_DIR}/{args.out_tag}/config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"[{args.out_tag}] arm={args.arm} steps={args.steps} batch={args.batch_size} "
          f"base_lr={args.base_lr} lamda_sisnr={lamda_sisnr} bn_frozen={n_bn}", flush=True)

    loss_log, best_val = [], float("inf")
    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time()
    switched = False
    for step in range(1, args.steps + 1):
        if args.arm == "lr_sched":
            for g in opt.param_groups:
                g["lr"] = lr_at(step, args.base_lr, args.steps)
        if args.arm == "snr_curriculum" and not switched and step > args.curr_switch:
            train_loader = make_loader("train", args.batch_size, 4.0, (-5, 15), 100000)
            train_iter = iter(train_loader); switched = True
            print(f"[{args.out_tag}] SNR curriculum switch -> [-5,15] at step {step}", flush=True)

        try:
            noisy, clean = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader); noisy, clean = next(train_iter)
        noisy, clean = noisy.to(device), clean.to(device)

        opt.zero_grad()
        enh = model(noisy)
        l_se = se_loss_fn(enh, clean)
        if args.arm == "preserve":
            l_pres = preservation_loss(enh, clean)
            loss = l_se + args.preserve_weight * l_pres
        else:
            l_pres = torch.tensor(0.0); loss = l_se
        if not torch.isfinite(loss):
            print(f"[{args.out_tag}] non-finite loss @ {step}, skip", flush=True)
            opt.zero_grad(); continue
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 3.0)
        opt.step()

        if step % 50 == 0 or step == 1:
            loss_log.append({"step": step, "loss_total": loss.item(),
                             "loss_se": l_se.item(), "loss_pres": float(l_pres),
                             "lr": opt.param_groups[0]["lr"], "elapsed_s": time.time() - t0})
        if step % 200 == 0 or step == 1:
            print(f"[{args.out_tag}] step {step}/{args.steps} loss_se={l_se.item():.4f} "
                  f"pres={float(l_pres):.4f} lr={opt.param_groups[0]['lr']:.2e} "
                  f"t={time.time()-t0:.0f}s", flush=True)

        if step % args.val_every == 0 or step == args.steps:
            model.eval()
            vses, vsdr = [], []
            with torch.no_grad():
                for vb in val_loader:
                    vn, vc = vb[0].to(device), vb[1].to(device)
                    ve = model(vn)
                    vses.append(se_loss_fn(ve, vc).item())
                    vsdr.append(si_sdr(ve, vc).item())
            vloss = float(np.mean(vses)); vsi = float(np.mean(vsdr))
            print(f"[{args.out_tag}] *** VAL step {step}: SE-loss={vloss:.4f} SI-SDR={vsi:.2f}dB", flush=True)
            loss_log.append({"step": step, "val_se_loss": vloss, "val_si_sdr": vsi})
            if vloss < best_val:
                best_val = vloss
                torch.save({"model": model.state_dict(), "arm": args.arm, "step": step,
                            "val_se_loss": vloss, "val_si_sdr": vsi},
                           f"{CKPT_DIR}/{args.out_tag}_best.tar")
            model.train(); freeze_bn_running_stats(model)
            with open(f"{LOG_DIR}/{args.out_tag}/loss_log.json", "w") as f:
                json.dump(loss_log, f)

    total = time.time() - t0
    peak = torch.cuda.max_memory_allocated(device) / 1e6
    torch.save({"model": model.state_dict(), "arm": args.arm, "step": args.steps},
               f"{CKPT_DIR}/{args.out_tag}.tar")
    res = {"arm": args.arm, "out_tag": args.out_tag, "steps": args.steps,
           "batch_size": args.batch_size, "base_lr": args.base_lr,
           "total_wall_s": total, "steps_per_sec": args.steps / total,
           "peak_vram_mb": peak, "best_val_se_loss": best_val}
    with open(f"{LOG_DIR}/{args.out_tag}/resource_usage.json", "w") as f:
        json.dump(res, f, indent=2)
    print(f"=== [{args.out_tag}] DONE {total:.1f}s ({args.steps/total:.2f} steps/s) "
          f"best_val={best_val:.4f} peak_vram={peak:.0f}MB ===", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--arm", required=True,
                   choices=["control", "lr_sched", "preserve", "snr_curriculum"])
    p.add_argument("--steps", type=int, default=18000)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--base_lr", type=float, default=1e-5)
    p.add_argument("--out_tag", required=True)
    p.add_argument("--val_every", type=int, default=2000)
    p.add_argument("--preserve_weight", type=float, default=20.0)
    p.add_argument("--curr_switch", type=int, default=6000)
    run(p.parse_args())
