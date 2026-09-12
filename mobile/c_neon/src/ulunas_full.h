/* Runtime-free UL-UNAS streaming inference: public C API.
 * No ONNX Runtime / PyTorch / TensorFlow Lite dependency anywhere in this path. */
#pragma once
#include <stddef.h>

#define ULUNAS_HOP 256
#define ULUNAS_WIN 512
#define ULUNAS_FS 16000

typedef struct UlunasState UlunasState;

#ifdef __cplusplus
extern "C" {
#endif

/* Allocates and zero-initializes all state (weights are compiled-in constants from
 * generated/ulunas_weights.h -- no file I/O, no heap use after this call returns).
 * Returns 0 on success, nonzero on failure (e.g. allocation failure). */
int ulunas_init(UlunasState **out_state);

/* Diagnostic only: exact size of the preallocated arena (analysis/OLA buffers + all
 * streaming caches). Not part of the minimal required API but useful for memory reporting. */
size_t ulunas_state_size_bytes(void);

/* Processes exactly ULUNAS_HOP (256) input PCM samples (float, range ~[-1,1], 16kHz mono)
 * and writes ULUNAS_HOP enhanced output samples. Safe to call in a real-time audio
 * callback: no malloc/free, no locks, no logging. Output at call N corresponds to input
 * delayed by the model's algorithmic latency (2 hops = 32ms); the first 2 calls' output
 * is a startup transient (causal zero-history warm-up), not a bug -- see kernels.h notes. */
int ulunas_process_hop(UlunasState *state, const float *input_pcm, float *output_pcm);

/* Exposes the model graph directly on one 257-bin complex STFT frame (mix_spec/enh_spec:
 * [257][2], re/im interleaved) -- the exact same I/O contract as the streaming ONNX graph
 * (mobile/onnx_export/out/ulunas_finetuned_stream_simple.onnx), for direct frame-by-frame
 * parity testing without going through ulunas_process_hop's own STFT/ISTFT. */
void ulunas_process_frame_spec(UlunasState *state, const float *mix_spec, float *enh_spec);

/* Resets all streaming state (caches, OLA buffers) to the same state as a fresh
 * ulunas_init, without reallocating. */
void ulunas_reset(UlunasState *state);

void ulunas_destroy(UlunasState *state);

#ifdef __cplusplus
}
#endif
