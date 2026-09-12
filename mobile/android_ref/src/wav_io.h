/* Minimal mono PCM16 WAV reader/writer -- no external dependency, matches the PoC's fixed
 * format (16kHz, mono, PCM16, per SPEC_PLAN input spec). */
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

struct WavData {
    std::vector<float> samples; // normalized to [-1, 1]
    uint32_t sample_rate;
};

inline WavData read_wav_pcm16_mono(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) throw std::runtime_error(std::string("cannot open ") + path);
    char riff[4]; fread(riff, 1, 4, f);
    uint32_t chunk_size; fread(&chunk_size, 4, 1, f);
    char wave[4]; fread(wave, 1, 4, f);
    if (strncmp(riff, "RIFF", 4) != 0 || strncmp(wave, "WAVE", 4) != 0)
        throw std::runtime_error("not a RIFF/WAVE file");

    uint16_t num_channels = 1, bits_per_sample = 16;
    uint32_t sample_rate = 16000, data_size = 0;
    long data_pos = 0;

    while (!feof(f)) {
        char id[4];
        if (fread(id, 1, 4, f) != 4) break;
        uint32_t sz; fread(&sz, 4, 1, f);
        if (strncmp(id, "fmt ", 4) == 0) {
            uint16_t audio_format; fread(&audio_format, 2, 1, f);
            fread(&num_channels, 2, 1, f);
            fread(&sample_rate, 4, 1, f);
            uint32_t byte_rate; fread(&byte_rate, 4, 1, f);
            uint16_t block_align; fread(&block_align, 2, 1, f);
            fread(&bits_per_sample, 2, 1, f);
            if (sz > 16) fseek(f, sz - 16, SEEK_CUR);
        } else if (strncmp(id, "data", 4) == 0) {
            data_size = sz;
            data_pos = ftell(f);
            fseek(f, sz, SEEK_CUR);
        } else {
            fseek(f, sz, SEEK_CUR);
        }
    }
    if (data_pos == 0) throw std::runtime_error("no data chunk found");

    fseek(f, data_pos, SEEK_SET);
    size_t n_samples = data_size / (bits_per_sample / 8) / num_channels;
    WavData out;
    out.sample_rate = sample_rate;
    out.samples.resize(n_samples);
    std::vector<int16_t> raw(n_samples * num_channels);
    fread(raw.data(), sizeof(int16_t), raw.size(), f);
    fclose(f);

    for (size_t i = 0; i < n_samples; ++i) {
        int32_t acc = 0;
        for (uint16_t c = 0; c < num_channels; ++c) acc += raw[i * num_channels + c];
        out.samples[i] = (float)(acc / (double)num_channels) / 32768.0f;
    }
    return out;
}

inline void write_wav_pcm16_mono(const char* path, const std::vector<float>& samples, uint32_t sample_rate) {
    FILE* f = fopen(path, "wb");
    if (!f) throw std::runtime_error(std::string("cannot open for write ") + path);

    uint32_t data_size = (uint32_t)(samples.size() * sizeof(int16_t));
    uint32_t chunk_size = 36 + data_size;
    uint16_t num_channels = 1, bits_per_sample = 16, audio_format = 1;
    uint32_t byte_rate = sample_rate * num_channels * bits_per_sample / 8;
    uint16_t block_align = num_channels * bits_per_sample / 8;

    fwrite("RIFF", 1, 4, f);
    fwrite(&chunk_size, 4, 1, f);
    fwrite("WAVE", 1, 4, f);
    fwrite("fmt ", 1, 4, f);
    uint32_t fmt_size = 16;
    fwrite(&fmt_size, 4, 1, f);
    fwrite(&audio_format, 2, 1, f);
    fwrite(&num_channels, 2, 1, f);
    fwrite(&sample_rate, 4, 1, f);
    fwrite(&byte_rate, 4, 1, f);
    fwrite(&block_align, 2, 1, f);
    fwrite(&bits_per_sample, 2, 1, f);
    fwrite("data", 1, 4, f);
    fwrite(&data_size, 4, 1, f);

    for (float s : samples) {
        float clamped = s < -1.0f ? -1.0f : (s > 1.0f ? 1.0f : s);
        int16_t v = (int16_t)std::lround(clamped * 32767.0f);
        fwrite(&v, sizeof(int16_t), 1, f);
    }
    fclose(f);
}
