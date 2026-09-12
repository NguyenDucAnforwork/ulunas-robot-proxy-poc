"""
DNSMOS P.835 (SIG/BAK/OVRL) + P808_MOS, computed by calling `onnxruntime.InferenceSession`
directly on the two ONNX files shipped in SEtrain/DNSMOS/DNSMOS/ -- NO espnet2 dependency
(decision #13).

Preprocessing and output mapping are a line-by-line port of the actual reference
implementation (espnet2/enh/layers/dnsmos.py :: DNSMOS_local, which is itself a port of
microsoft/DNS-Challenge/DNSMOS/dnsmos_local.py), fetched and read directly from
https://raw.githubusercontent.com/espnet/espnet/master/espnet2/enh/layers/dnsmos.py on
2026-09-12 -- not reimplemented from memory. Verified to match:
  - INPUT_LENGTH = 9.01s segments, 1s hop, pad-by-doubling audio shorter than one segment.
  - primary model ("sig_bak_ovr.onnx"): input_1 = raw waveform (1, 144160) float32,
    output = (1, 3) -> [sig_raw, bak_raw, ovr_raw], in that order.
  - p808 model ("model_v8.onnx"): input_1 = log-mel spectrogram (1, frames, 120), computed on
    audio_seg[:-160] (drops the last hop_length samples), via
    librosa.feature.melspectrogram(y=seg, sr=16000, n_fft=321, hop_length=160, n_mels=120),
    then (power_to_db(mel, ref=np.max) + 40) / 40, transposed to (frames, 120).
  - non-personalized polynomial correction (is_personalized_MOS=False, matching SEtrain's
    evaluate.py / calculate_nonintrusive_dnsmos.py, which never passes that flag):
      p_sig = polyval([-0.08397278, 1.22083953, 0.0052439], sig_raw)
      p_bak = polyval([-0.13166888, 1.60915514, -0.39604546], bak_raw)
      p_ovr = polyval([-0.06766283, 1.11546468, 0.04602535], ovr_raw)
  - final score = mean across all 9.01s/1s-hop segments.
"""
import librosa
import numpy as np
import onnxruntime as ort

SAMPLING_RATE = 16000
INPUT_LENGTH = 9.01

SIG_COEF = [-0.08397278, 1.22083953, 0.0052439]
BAK_COEF = [-0.13166888, 1.60915514, -0.39604546]
OVR_COEF = [-0.06766283, 1.11546468, 0.04602535]

DNSMOS_DIR = "/content/repos/SEtrain/DNSMOS/DNSMOS"


class DNSMOSOnnxRuntime:
    def __init__(self, device="cpu"):
        provider = "CUDAExecutionProvider" if device == "cuda" else "CPUExecutionProvider"
        self.primary_sess = ort.InferenceSession(f"{DNSMOS_DIR}/sig_bak_ovr.onnx", providers=[provider])
        self.p808_sess = ort.InferenceSession(f"{DNSMOS_DIR}/model_v8.onnx", providers=[provider])

    @staticmethod
    def _melspec(audio):
        mel = librosa.feature.melspectrogram(y=audio, sr=SAMPLING_RATE, n_fft=321, hop_length=160, n_mels=120)
        mel_db = (librosa.power_to_db(mel, ref=np.max) + 40) / 40
        return mel_db.T  # (frames, 120)

    def __call__(self, audio, fs=16000):
        assert fs == SAMPLING_RATE, "resample to 16kHz before calling (this PoC never needs to resample)"
        audio = np.asarray(audio, dtype=np.float32)

        len_samples = int(INPUT_LENGTH * SAMPLING_RATE)
        while len(audio) < len_samples:
            audio = np.append(audio, audio)

        num_hops = int(np.floor(len(audio) / SAMPLING_RATE) - INPUT_LENGTH) + 1
        hop_len_samples = SAMPLING_RATE

        sig_raw_l, bak_raw_l, ovr_raw_l = [], [], []
        sig_l, bak_l, ovr_l = [], [], []
        p808_l = []

        for idx in range(num_hops):
            seg = audio[int(idx * hop_len_samples): int((idx + INPUT_LENGTH) * hop_len_samples)]
            if len(seg) < len_samples:
                continue

            input_features = seg.astype("float32")[np.newaxis, :]
            p808_features = self._melspec(seg[:-160]).astype("float32")[np.newaxis, :, :]

            p808_mos = self.p808_sess.run(None, {"input_1": p808_features})[0][0][0]
            sig_raw, bak_raw, ovr_raw = self.primary_sess.run(None, {"input_1": input_features})[0][0]

            sig = np.polyval(SIG_COEF, sig_raw)
            bak = np.polyval(BAK_COEF, bak_raw)
            ovr = np.polyval(OVR_COEF, ovr_raw)

            sig_raw_l.append(sig_raw); bak_raw_l.append(bak_raw); ovr_raw_l.append(ovr_raw)
            sig_l.append(sig); bak_l.append(bak); ovr_l.append(ovr)
            p808_l.append(p808_mos)

        return {
            "OVRL_raw": float(np.mean(ovr_raw_l)), "SIG_raw": float(np.mean(sig_raw_l)),
            "BAK_raw": float(np.mean(bak_raw_l)), "OVRL": float(np.mean(ovr_l)),
            "SIG": float(np.mean(sig_l)), "BAK": float(np.mean(bak_l)),
            "P808_MOS": float(np.mean(p808_l)),
        }


if __name__ == "__main__":
    import soundfile as sf
    dnsmos = DNSMOSOnnxRuntime()
    wav, fs = sf.read("/content/repos/ul-unas/audio/clean/0119.wav", dtype="float32")
    print(dnsmos(wav, fs))
