/* UL-UNAS Android/ARM64 reference: offline WAV enhancement via streaming ONNX inference.
 *
 * Pipeline: read PCM16 mono WAV -> reflect-pad + frame + Hann window + real FFT (matches
 * PyTorch's torch.stft(center=True) to ~1e-6, verified numerically against a Python
 * reference before writing this file) -> feed each 257-bin complex frame + 3 cache tensors
 * to the streaming ONNX graph (ulunas_finetuned_stream_simple.onnx) one frame at a time,
 * threading the caches forward -> inverse real FFT + Hann synthesis window + overlap-add
 * with per-sample window-sum normalization (matches torch.istft to ~1e-6) -> crop the
 * reflect-pad -> write PCM16 mono WAV.
 *
 * This is the "offline WAV enhancement" + "streaming state" + "PCM16 I/O" + "STFT->model->
 * iSTFT" reference required for M4. It is ONNX-Runtime-based (reference/baseline path only,
 * per SPEC_PLAN -- the runtime-free C/NEON path is separate, M5).
 */
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "fft.h"
#include "wav_io.h"
#include "onnxruntime_cxx_api.h"

static const int N_FFT = 512;
static const int HOP = 256;
static const int PAD = N_FFT / 2; // 256, reflect-pad each side to match torch.stft(center=True)
static const int FREQ_BINS = N_FFT / 2 + 1; // 257

static const int CONV_CACHE_SIZE = 5358;
static const int TFA_CACHE_SIZE = 402;
static const int INTER_CACHE_SIZE = 1056;

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <model.onnx> <input.wav> <output.wav>\n", argv[0]);
        return 1;
    }
    const char* model_path = argv[1];
    const char* input_path = argv[2];
    const char* output_path = argv[3];

    WavData input = read_wav_pcm16_mono(input_path);
    if (input.sample_rate != 16000) {
        fprintf(stderr, "WARNING: input sample rate is %u, model expects 16000\n", input.sample_rate);
    }
    int n_samples = (int)input.samples.size();

    // ---- reflect-pad + frame + window + rfft (STFT analysis, matches torch.stft center=True) ----
    std::vector<float> padded(n_samples + 2 * PAD);
    for (int i = 0; i < PAD; ++i) padded[PAD - 1 - i] = input.samples[i + 1 <= n_samples - 1 ? i + 1 : n_samples - 1];
    for (int i = 0; i < n_samples; ++i) padded[PAD + i] = input.samples[i];
    for (int i = 0; i < PAD; ++i) {
        int src = n_samples - 2 - i;
        padded[PAD + n_samples + i] = input.samples[src >= 0 ? src : 0];
    }

    std::vector<float> win = hann_window(N_FFT);
    int num_frames = 1 + ((int)padded.size() - N_FFT) / HOP;

    std::vector<std::vector<cplx>> spec_frames(num_frames);
    for (int t = 0; t < num_frames; ++t) {
        std::vector<float> seg(N_FFT);
        for (int i = 0; i < N_FFT; ++i) seg[i] = padded[t * HOP + i] * win[i];
        spec_frames[t] = rfft(seg);
    }

    // ---- ONNX Runtime session ----
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ulunas_android_ref");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1); // single-thread streaming per SPEC (RTF measured single-thread)
    Ort::Session session(env, model_path, session_options);

    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::vector<float> conv_cache(CONV_CACHE_SIZE, 0.0f);
    std::vector<float> tfa_cache(TFA_CACHE_SIZE, 0.0f);
    std::vector<float> inter_cache(INTER_CACHE_SIZE, 0.0f);

    std::vector<std::vector<cplx>> enh_frames(num_frames);

    const char* input_names[] = {"mix", "conv_cache", "tfa_cache", "inter_cache"};
    const char* output_names[] = {"enh", "conv_cache_out", "tfa_cache_out", "inter_cache_out"};

    std::vector<double> frame_times_ms;
    frame_times_ms.reserve(num_frames);

    for (int t = 0; t < num_frames; ++t) {
        std::vector<float> mix_in(FREQ_BINS * 2);
        for (int f = 0; f < FREQ_BINS; ++f) {
            mix_in[f * 2 + 0] = spec_frames[t][f].real();
            mix_in[f * 2 + 1] = spec_frames[t][f].imag();
        }

        int64_t mix_shape[4] = {1, FREQ_BINS, 1, 2};
        int64_t conv_shape[2] = {1, CONV_CACHE_SIZE};
        int64_t tfa_shape[2] = {1, TFA_CACHE_SIZE};
        int64_t inter_shape[2] = {1, INTER_CACHE_SIZE};

        std::vector<Ort::Value> inputs;
        inputs.push_back(Ort::Value::CreateTensor<float>(mem_info, mix_in.data(), mix_in.size(), mix_shape, 4));
        inputs.push_back(Ort::Value::CreateTensor<float>(mem_info, conv_cache.data(), conv_cache.size(), conv_shape, 2));
        inputs.push_back(Ort::Value::CreateTensor<float>(mem_info, tfa_cache.data(), tfa_cache.size(), tfa_shape, 2));
        inputs.push_back(Ort::Value::CreateTensor<float>(mem_info, inter_cache.data(), inter_cache.size(), inter_shape, 2));

        auto t0 = std::chrono::high_resolution_clock::now();
        auto outputs = session.Run(Ort::RunOptions{nullptr}, input_names, inputs.data(), inputs.size(),
                                    output_names, 4);
        auto t1 = std::chrono::high_resolution_clock::now();
        frame_times_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());

        float* enh_out = outputs[0].GetTensorMutableData<float>();
        enh_frames[t].resize(FREQ_BINS);
        for (int f = 0; f < FREQ_BINS; ++f) enh_frames[t][f] = cplx(enh_out[f * 2], enh_out[f * 2 + 1]);

        memcpy(conv_cache.data(), outputs[1].GetTensorMutableData<float>(), CONV_CACHE_SIZE * sizeof(float));
        memcpy(tfa_cache.data(), outputs[2].GetTensorMutableData<float>(), TFA_CACHE_SIZE * sizeof(float));
        memcpy(inter_cache.data(), outputs[3].GetTensorMutableData<float>(), INTER_CACHE_SIZE * sizeof(float));
    }

    // ---- ISTFT: irfft + synthesis window + overlap-add + win-sum normalization + crop pad ----
    int padded_out_len = (num_frames - 1) * HOP + N_FFT;
    std::vector<double> ola(padded_out_len, 0.0);
    std::vector<double> win_sum(padded_out_len, 0.0);

    for (int t = 0; t < num_frames; ++t) {
        std::vector<float> frame_time = irfft(enh_frames[t], N_FFT);
        for (int i = 0; i < N_FFT; ++i) {
            ola[t * HOP + i] += (double)(frame_time[i] * win[i]);
            win_sum[t * HOP + i] += (double)(win[i] * win[i]);
        }
    }

    std::vector<float> output(n_samples, 0.0f);
    for (int i = 0; i < n_samples; ++i) {
        int src = PAD + i;
        double norm = win_sum[src] > 1e-11 ? win_sum[src] : 1e-11;
        output[i] = (float)(ola[src] / norm);
    }

    write_wav_pcm16_mono(output_path, output, input.sample_rate);

    double total_ms = 0;
    for (double v : frame_times_ms) total_ms += v;
    double frame_duration_ms = 1000.0 * HOP / 16000.0;
    double rtf = (total_ms / num_frames) / frame_duration_ms;
    double p95_ms = 0;
    {
        std::vector<double> sorted_t = frame_times_ms;
        std::sort(sorted_t.begin(), sorted_t.end());
        p95_ms = sorted_t[(size_t)(0.95 * sorted_t.size())];
    }

    printf("frames=%d mean_frame_ms=%.4f p95_frame_ms=%.4f frame_duration_ms=%.4f RTF=%.4f\n",
           num_frames, total_ms / num_frames, p95_ms, frame_duration_ms, rtf);
    printf("NOTE: this timing is from whatever host this binary is run on. It is NOT a valid\n"
           "Cortex-A53 performance number unless run on real Cortex-A53 hardware -- see\n"
           "MOBILE_BENCHMARK.md for the PENDING/BLOCKED status until real hardware is available.\n");

    return 0;
}
