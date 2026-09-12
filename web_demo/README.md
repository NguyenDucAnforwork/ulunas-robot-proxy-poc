# web_demo — WASM fallback (Safari-on-iPhone), NOT a native performance demo

## What this is

A browser-based fallback for when no Mac/Xcode/physical iPhone 11 is available to build the
real native app (see `../IOS_BUILD_AND_DEMO.md`). It compiles the same STFT/ISTFT streaming DSP
core used and verified in `../mobile/android_ref/` to WebAssembly via Emscripten, and runs it in
the browser against live microphone input.

**Status of what's actually verified vs not:**

| Item | Status |
|---|---|
| `emcc` install | **PASS** — Emscripten 3.1.6 installed via `apt-get install emscripten` in this container |
| WASM compile | **PASS** — `ulunas_wasm.cpp` compiles to `ulunas_wasm.{js,wasm}` cleanly |
| DSP correctness (Node.js smoke test) | **PASS** — `test_wasm_node.js` verifies the compiled WASM module's streaming STFT/OLA reconstructs an identity (no-model) signal to 1.8e-6 max abs error, once correctly accounting for the pipeline's inherent 1-hop (16ms) causal algorithmic latency. A first attempt at the OLA math (assuming a constant window-sum normalization) was WRONG — verified wrong numerically (the true window-sum-of-squares ranges 0.5–1.0 across a hop, not constant) — and was fixed before this passed. |
| Actual noise-suppression model wired in | **NOT_RUN** — the runtime-free C model implementation was still being completed by a concurrent workstream when this was built. This demo runs a real, verified DSP framing pipeline with an **identity passthrough** standing in for the model (clearly labeled in the page itself). Swapping in the real model once available is a small, mechanical change (see the `TODO` in `ulunas_wasm.cpp`) — the DSP plumbing around it does not need to change. |
| Running in an actual browser (Safari/iPhone or otherwise) | **NOT_RUN** — this container has no browser to test in. Only Node.js-based testing of the compiled WASM module was possible here. |
| AudioWorklet | **NOT ATTEMPTED** — used `ScriptProcessorNode` instead (deprecated but far simpler to get right without a real browser to test against; a raw-WASM-in-AudioWorklet instantiation has its own interop subtleties this environment cannot verify). Documented as a known follow-up, not silently substituted. |
| Resampling quality | Naive nearest-neighbor, adequate only for a demo — not a production resampler. |

## How to serve it (for the person who can test in a real browser)

Safari requires HTTPS (or `localhost`) for `getUserMedia` microphone access. From a machine with the files:

```bash
# quick local HTTPS test (self-signed, browser will warn):
python3 -m http.server 8443 &  # plain HTTP first
# then front it with e.g. ngrok, cloudflared, or any HTTPS tunnel for a real device test:
ngrok http 8443
```
Open the resulting `https://` URL on the iPhone in Safari, tap Start, grant microphone access, and use wired headphones (no AEC in this project — see the in-page warning).

## Rebuilding

```bash
emcc -O2 -std=c++17 ulunas_wasm.cpp \
  -s WASM=1 \
  -s EXPORTED_FUNCTIONS='["_ulunas_wasm_reset","_ulunas_wasm_process_hop","_malloc","_free"]' \
  -s EXPORTED_RUNTIME_METHODS='["ccall","cwrap","HEAPF32"]' \
  -s MODULARIZE=1 -s EXPORT_NAME='UlunasWasmModule' \
  -s ALLOW_MEMORY_GROWTH=1 \
  -o ulunas_wasm.js
node --no-experimental-fetch test_wasm_node.js   # re-verify before shipping
```
(The `--no-experimental-fetch` flag works around a Node 18 quirk where the global `fetch` API
confuses Emscripten's Node-environment file-loading path — a Node/tooling issue, not a bug in
this project's code; not needed when served to an actual browser.)
