/* Minimal radix-2 Cooley-Tukey FFT/IFFT for power-of-two sizes (used with N=512, matching
 * UL-UNAS's n_fft=512). Self-contained -- no external FFT library, so the Android reference
 * build has no extra native dependency beyond ONNX Runtime. */
#pragma once
#include <cmath>
#include <complex>
#include <vector>

using cplx = std::complex<float>;

inline void fft_inplace(std::vector<cplx>& a, bool invert) {
    const size_t n = a.size();
    for (size_t i = 1, j = 0; i < n; ++i) {
        size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }
    for (size_t len = 2; len <= n; len <<= 1) {
        float ang = (float)(2 * M_PI / len) * (invert ? 1.0f : -1.0f);
        cplx wlen(std::cos(ang), std::sin(ang));
        for (size_t i = 0; i < n; i += len) {
            cplx w(1.0f, 0.0f);
            for (size_t k = 0; k < len / 2; ++k) {
                cplx u = a[i + k];
                cplx v = a[i + k + len / 2] * w;
                a[i + k] = u + v;
                a[i + k + len / 2] = u - v;
                w *= wlen;
            }
        }
    }
    if (invert) {
        for (auto& x : a) x /= (float)n;
    }
}

/* Real-input onesided FFT: returns n/2+1 complex bins, matching torch.stft(onesided=True). */
inline std::vector<cplx> rfft(const std::vector<float>& real_in) {
    size_t n = real_in.size();
    std::vector<cplx> a(n);
    for (size_t i = 0; i < n; ++i) a[i] = cplx(real_in[i], 0.0f);
    fft_inplace(a, false);
    a.resize(n / 2 + 1);
    return a;
}

/* Inverse of rfft: reconstruct full spectrum via conjugate symmetry, IFFT, return real part. */
inline std::vector<float> irfft(const std::vector<cplx>& onesided, size_t n) {
    std::vector<cplx> full(n);
    size_t half = n / 2 + 1;
    for (size_t i = 0; i < half; ++i) full[i] = onesided[i];
    for (size_t i = half; i < n; ++i) full[i] = std::conj(onesided[n - i]);
    fft_inplace(full, true);
    std::vector<float> out(n);
    for (size_t i = 0; i < n; ++i) out[i] = full[i].real();
    return out;
}

/* Matches torch.hann_window(n) with default periodic=True: 0.5 - 0.5*cos(2*pi*i/n). */
inline std::vector<float> hann_window(size_t n) {
    std::vector<float> w(n);
    for (size_t i = 0; i < n; ++i)
        w[i] = 0.5f - 0.5f * std::cos((float)(2.0 * M_PI * (double)i / (double)n));
    return w;
}
