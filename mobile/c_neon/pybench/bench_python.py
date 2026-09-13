"""
TEMPORARY benchmark script (not committed, not production code). Streaming PyTorch
reference for UL-UNAS, using the SAME class already used to validate the C engine's
parity (mobile/onnx_export/ulunas_onnx/stream/ulunas_stream.py::StreamULUNAS, driven by
mobile/onnx_export/export_finetuned_stream.py's exact load pattern) -- not the legacy
SEtrain_adapted/infer.py (that is a different, GTCRN-era script, not the UL-UNAS
streaming reference; confirmed by inspection before writing this).

Two levels, matching mobile/c_neon's harness_model_only.cpp / harness_e2e.cpp:
  - model-only:  spectral frame -> StreamULUNAS.forward() -> enhanced spectral frame
  - end-to-end:  PCM hop -> STFT (torch.fft.rfft, same Hann/512/256 convention as
                 mobile/c_neon/src/fft.h) -> model -> iSTFT/OLA (torch.fft.irfft +
                 window-sum-normalized overlap-add, same convention as
                 mobile/c_neon/src/ulunas_full.cpp::ulunas_process_hop) -> PCM hop.

Checkpoint: F_PROXY_ROBOT.tar -- matches what mobile/c_neon/generated/ulunas_weights.h
was exported from in this session (corrected from an earlier F_GENERIC.tar mismatch),
so cross-language output parity is a meaningful same-weights comparison.

Deterministic input is read from input_pcm.bin / input_spec.bin (produced by
dump_input.cpp, C++ std::mt19937(1234) uniform(-0.2,0.2) -- same distribution the task
specifies, generated once so both languages consume byte-identical input rather than
relying on replicating C++'s RNG algorithm in Python).
"""
import sys
import time

import numpy as np
import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

REPO = "/content/ulunas-robot-proxy-poc"
sys.path.insert(0, f"{REPO}/SEtrain_adapted")
sys.path.insert(0, f"{REPO}/mobile/onnx_export/ulunas_onnx/stream")

from models.ulunas import ULUNAS  # noqa: E402
from ulunas_stream import StreamULUNAS  # noqa: E402

CKPT = f"{REPO}/checkpoints/F_PROXY_ROBOT.tar"
WARMUP = 100
MEASURED = 2000
HOP = 256
FREQ = 257
WIN = 512
FS = 16000

device = torch.device("cpu")


def load_model():
    model = ULUNAS().to(device).eval()
    model.load_state_dict(torch.load(CKPT, map_location=device)["model"])
    stream_model = StreamULUNAS().to(device).eval()
    stream_model.load_state_dict(model.state_dict(), strict=True)
    return stream_model


def summarize(times_ms, label):
    t = np.sort(np.asarray(times_ms))
    n = len(t)
    mean_ms = float(t.mean())
    p50 = float(t[int(0.50 * n)])
    p95 = float(t[int(0.95 * n)])
    p99 = float(t[int(0.99 * n)])
    max_ms = float(t[-1])
    hop_ms = 1000.0 * HOP / FS
    print(f"[{label}] n={n} hop_duration_ms={hop_ms:.2f}")
    print(f"[{label}] mean_ms={mean_ms:.5f} p50_ms={p50:.5f} p95_ms={p95:.5f} "
          f"p99_ms={p99:.5f} max_ms={max_ms:.5f}")
    print(f"[{label}] RTF_mean={mean_ms/hop_ms:.5f} RTF_p95={p95/hop_ms:.5f}")
    return {"mean_ms": mean_ms, "p50_ms": p50, "p95_ms": p95, "p99_ms": p99, "max_ms": max_ms}


def bench_model_only(stream_model):
    data = np.fromfile("input_spec.bin", dtype=np.float32).reshape(-1, FREQ, 2)
    assert data.shape[0] == WARMUP + MEASURED, data.shape

    conv_cache, tfa_cache, inter_cache = StreamULUNAS.init_caches(batch_size=1, device=device)
    with torch.no_grad():
        for i in range(WARMUP):
            xi = torch.from_numpy(data[i])[None, :, None, :]
            _, conv_cache, tfa_cache, inter_cache = stream_model(xi, conv_cache, tfa_cache, inter_cache)

        times_ns = np.zeros(MEASURED, dtype=np.int64)
        outs = np.zeros((MEASURED, FREQ, 2), dtype=np.float32)
        for i in range(MEASURED):
            xi = torch.from_numpy(data[WARMUP + i])[None, :, None, :]
            t0 = time.perf_counter_ns()
            yi, conv_cache, tfa_cache, inter_cache = stream_model(xi, conv_cache, tfa_cache, inter_cache)
            t1 = time.perf_counter_ns()
            times_ns[i] = t1 - t0
            outs[i] = yi[0, :, 0, :].numpy()

    outs.tofile("output_spec_py.bin")
    return summarize(times_ns / 1e6, "model-only")


def bench_e2e(stream_model):
    pcm = np.fromfile("input_pcm.bin", dtype=np.float32).reshape(-1, HOP)
    assert pcm.shape[0] == WARMUP + MEASURED, pcm.shape

    hann = torch.hann_window(WIN, periodic=True)  # matches fft.h::hann_window (0.5-0.5cos(2*pi*i/n))
    analysis_buf = torch.zeros(WIN)
    ola_buf = torch.zeros(WIN)
    win_sum_buf = torch.zeros(WIN)
    conv_cache, tfa_cache, inter_cache = StreamULUNAS.init_caches(batch_size=1, device=device)

    def process_hop(hop_in):
        nonlocal analysis_buf, ola_buf, win_sum_buf, conv_cache, tfa_cache, inter_cache
        analysis_buf = torch.cat([analysis_buf[HOP:], hop_in])
        windowed = analysis_buf * hann
        spec = torch.fft.rfft(windowed, n=WIN)  # unnormalized forward, matches fft.h::rfft
        mix_spec = torch.view_as_real(spec)[None, :, None, :]  # (1,257,1,2)
        with torch.no_grad():
            enh_spec, conv_cache, tfa_cache, inter_cache = stream_model(
                mix_spec, conv_cache, tfa_cache, inter_cache)
        enh_c = torch.complex(enh_spec[0, :, 0, 0], enh_spec[0, :, 0, 1])
        frame_time = torch.fft.irfft(enh_c, n=WIN)  # 1/n-normalized inverse, matches fft.h::irfft
        ola_buf = ola_buf + frame_time * hann
        win_sum_buf = win_sum_buf + hann * hann
        norm = torch.clamp(win_sum_buf[:HOP], min=1e-8)
        out_hop = ola_buf[:HOP] / norm
        ola_buf = torch.cat([ola_buf[HOP:], torch.zeros(HOP)])
        win_sum_buf = torch.cat([win_sum_buf[HOP:], torch.zeros(HOP)])
        return out_hop

    for i in range(WARMUP):
        process_hop(torch.from_numpy(pcm[i]))

    times_ns = np.zeros(MEASURED, dtype=np.int64)
    outs = np.zeros((MEASURED, HOP), dtype=np.float32)
    for i in range(MEASURED):
        t0 = time.perf_counter_ns()
        out_hop = process_hop(torch.from_numpy(pcm[WARMUP + i]))
        t1 = time.perf_counter_ns()
        times_ns[i] = t1 - t0
        outs[i] = out_hop.numpy()

    outs.tofile("output_pcm_py.bin")
    return summarize(times_ns / 1e6, "end-to-end")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "both"
    stream_model = load_model()
    if mode in ("model-only", "both"):
        bench_model_only(stream_model)
    if mode in ("e2e", "both"):
        bench_e2e(stream_model)
