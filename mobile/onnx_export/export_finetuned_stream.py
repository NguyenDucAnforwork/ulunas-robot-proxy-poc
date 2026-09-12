"""
Export the fine-tuned F_GENERIC checkpoint (produced in M1) to streaming ONNX, reusing the
exact StreamULUNAS wrapper already shipped in ul-unas/ulunas_onnx/stream/ulunas_stream.py
(architecture/cache logic verified during the planning gate) -- only the checkpoint path and
input audio differ from the repo's own __main__ block.
"""
import os
import sys
import time

import numpy as np
import onnx
import onnxruntime
import soundfile as sf
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ulunas_onnx", "stream"))
sys.path.insert(0, "/content/project/SEtrain_adapted")
sys.path.insert(0, "/content/project/SEtrain_adapted/models")

from ulunas_stream import StreamULUNAS
from models.ulunas import ULUNAS

FINETUNED_CKPT = "/content/project/checkpoints/F_GENERIC.tar"
OUT_DIR = "/content/project/mobile/onnx_export/out"
TEST_WAV_DIR = "/content/project/mobile/onnx_export/parity_wavs"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cpu")

    model = ULUNAS().to(device).eval()
    model.load_state_dict(torch.load(FINETUNED_CKPT, map_location=device)["model"])

    stream_model = StreamULUNAS().to(device).eval()
    stream_model.load_state_dict(model.state_dict(), strict=True)

    # use one of the audio_samples noisy files produced during M1 eval as the export/parity input
    test_wavs = sorted([f for f in os.listdir(TEST_WAV_DIR) if f.endswith("_noisy.wav")])
    assert len(test_wavs) >= 10, f"expected >=10 audio samples, found {len(test_wavs)}"

    stft_window = torch.hann_window(512)

    def run_offline(x):
        with torch.no_grad():
            return model(x)

    def run_streaming(x_spec, conv_cache, tfa_cache, inter_cache):
        ys = []
        with torch.no_grad():
            for i in range(x_spec.shape[2]):
                xi = x_spec[:, :, i:i + 1, :]
                yi, conv_cache, tfa_cache, inter_cache = stream_model(xi, conv_cache, tfa_cache, inter_cache)
                ys.append(yi)
        return torch.cat(ys, dim=2), conv_cache, tfa_cache, inter_cache

    # --- export ONNX using the first test utterance as dummy/trace input ---
    x0, fs0 = sf.read(os.path.join(TEST_WAV_DIR, test_wavs[0]), dtype="float32")
    assert fs0 == 16000
    x0_t = torch.from_numpy(x0)[None]
    x0_spec_c = torch.stft(x0_t, n_fft=512, hop_length=256, win_length=512, window=stft_window, return_complex=True)
    x0_spec = torch.view_as_real(x0_spec_c)

    conv_cache, tfa_cache, inter_cache = StreamULUNAS.init_caches(batch_size=1, device=device)

    onnx_path = f"{OUT_DIR}/ulunas_finetuned_stream.onnx"
    simple_path = onnx_path.replace(".onnx", "_simple.onnx")
    for p in [onnx_path, simple_path]:
        if os.path.exists(p):
            os.remove(p)

    dummy_mix = x0_spec[:, :, 0:1, :]
    torch.onnx.export(
        stream_model, (dummy_mix, conv_cache, tfa_cache, inter_cache), onnx_path,
        input_names=["mix", "conv_cache", "tfa_cache", "inter_cache"],
        output_names=["enh", "conv_cache_out", "tfa_cache_out", "inter_cache_out"],
        opset_version=11, do_constant_folding=False, verbose=False,
    )
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX checker: PASS")

    from onnxsim import simplify
    model_simp, check = simplify(onnx_model)
    assert check
    onnx.save(model_simp, simple_path)
    print(f"Saved {onnx_path} and {simple_path}")

    # --- parity: >=10 utterances, >=100 frames total ---
    session = onnxruntime.InferenceSession(simple_path, None, providers=["CPUExecutionProvider"])

    max_abs_err_frame = 0.0
    max_abs_err_wave = 0.0
    total_frames = 0
    per_utt_results = []

    for wav_name in test_wavs[:10]:
        x, fs = sf.read(os.path.join(TEST_WAV_DIR, wav_name), dtype="float32")
        assert fs == 16000
        x_t = torch.from_numpy(x)[None]

        y_offline = run_offline(x_t).numpy().squeeze()

        x_spec_c = torch.stft(x_t, n_fft=512, hop_length=256, win_length=512, window=stft_window, return_complex=True)
        x_spec = torch.view_as_real(x_spec_c)

        conv_cache, tfa_cache, inter_cache = StreamULUNAS.init_caches(batch_size=1, device=device)
        ys_stream, *_ = run_streaming(x_spec, conv_cache, tfa_cache, inter_cache)
        ys_stream_c = torch.complex(ys_stream[..., 0], ys_stream[..., 1])
        y_stream = torch.istft(ys_stream_c[0], n_fft=512, hop_length=256, win_length=512,
                               window=stft_window, onesided=True, length=x_t.shape[1]).numpy()

        conv_cache_np = np.zeros([1, conv_cache.shape[1]], dtype="float32")
        tfa_cache_np = np.zeros([1, tfa_cache.shape[1]], dtype="float32")
        inter_cache_np = np.zeros([1, inter_cache.shape[1]], dtype="float32")
        inputs = x_spec.numpy()
        onnx_frames = []
        for i in range(inputs.shape[-2]):
            out_i, conv_cache_np, tfa_cache_np, inter_cache_np = session.run(
                [], {"mix": inputs[..., i:i + 1, :], "conv_cache": conv_cache_np,
                     "tfa_cache": tfa_cache_np, "inter_cache": inter_cache_np})
            onnx_frames.append(out_i)
        onnx_spec = np.concatenate(onnx_frames, axis=2)

        frame_err = np.abs(onnx_spec - ys_stream.numpy()).max()
        max_abs_err_frame = max(max_abs_err_frame, frame_err)
        total_frames += inputs.shape[-2]

        onnx_spec_t = torch.from_numpy(onnx_spec)
        onnx_spec_c = torch.complex(onnx_spec_t[..., 0], onnx_spec_t[..., 1])
        y_onnx = torch.istft(onnx_spec_c[0], n_fft=512, hop_length=256, win_length=512,
                             window=stft_window, onesided=True, length=x_t.shape[1]).numpy()

        wave_rmse_onnx_vs_offline = np.sqrt(np.mean((y_onnx - y_offline) ** 2))
        wave_rmse_stream_vs_offline = np.sqrt(np.mean((y_stream - y_offline) ** 2))
        max_abs_err_wave = max(max_abs_err_wave, wave_rmse_onnx_vs_offline)

        per_utt_results.append({
            "wav": wav_name, "n_frames": int(inputs.shape[-2]),
            "max_abs_frame_err_onnx_vs_streaming_torch": float(frame_err),
            "wave_rmse_onnx_vs_offline": float(wave_rmse_onnx_vs_offline),
            "wave_rmse_streaming_torch_vs_offline": float(wave_rmse_stream_vs_offline),
            "has_nan_or_inf": bool(not np.all(np.isfinite(y_onnx))),
        })

    result = {
        "n_utterances_tested": len(per_utt_results),
        "total_frames_tested": int(total_frames),
        "max_abs_frame_err_over_all_utt": float(max_abs_err_frame),
        "max_wave_rmse_over_all_utt": float(max_abs_err_wave),
        "pass_max_abs_error_1e4": bool(max_abs_err_frame <= 1e-4),
        "pass_waveform_rmse_1e4": bool(max_abs_err_wave <= 1e-4),
        "per_utterance": per_utt_results,
    }

    import json
    with open(f"{OUT_DIR}/parity_results.json", "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps({k: v for k, v in result.items() if k != "per_utterance"}, indent=2))


if __name__ == "__main__":
    main()
