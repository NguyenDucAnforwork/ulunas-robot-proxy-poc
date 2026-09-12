/* Standalone, ONNX-Runtime-free correctness check of the hand-written FFT/STFT/ISTFT core
 * (fft.h) -- built for both x86_64 (host) and aarch64 (via a generic glibc cross-compiler,
 * run under qemu-aarch64-static) to confirm the DSP math is bit-identical across
 * architectures, isolating this from any ONNX Runtime linking concerns. */
#include <cstdio>
#include <random>
#include "fft.h"

int main() {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> x(512);
    for (auto& v : x) v = dist(rng);

    auto spec = rfft(x);
    auto recon = irfft(spec, 512);

    double max_err = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        double e = std::abs((double)x[i] - (double)recon[i]);
        if (e > max_err) max_err = e;
    }

    auto win = hann_window(512);
    double win_sum = 0;
    for (float w : win) win_sum += w;

    printf("rfft/irfft round-trip max_err=%.10f\n", max_err);
    printf("hann_window sum=%.10f (expect ~256.0)\n", win_sum);
    printf("spec[0]=(%.6f,%.6f) spec[1]=(%.6f,%.6f) spec[256]=(%.6f,%.6f)\n",
           spec[0].real(), spec[0].imag(), spec[1].real(), spec[1].imag(),
           spec[256].real(), spec[256].imag());
    return 0;
}
