# web_demo — WASM fallback (Safari-on-iPhone), NOT a native performance demo

## What this is

A browser-based fallback for when no Mac/Xcode/physical iPhone 11 is available to build the
real native app (see `../IOS_BUILD_AND_DEMO.md`). It compiles the **real** UL-UNAS runtime-free
C engine (`../mobile/c_neon/src/ulunas_full.cpp`, real trained `F_PROXY_ROBOT` weights — the
same engine validated in `../mobile/c_neon/` and benchmarked on real ARM in
`ARM_FULL_GRAPH_RESULTS.md`) to WebAssembly via Emscripten, and runs it in the browser against
live microphone input.

**Status of what's actually verified vs not:**

| Item | Status |
|---|---|
| `emcc` install | **PASS** — Emscripten 3.1.6 installed via `apt-get install emscripten` |
| WASM compile | **PASS** — `ulunas_wasm.cpp` + `ulunas_full.cpp` compile to `ulunas_wasm.{js,wasm}` cleanly |
| Actual noise-suppression model wired in | **PASS (follow-up session)** — `ulunas_wasm.cpp` is now a thin wrapper directly calling the real `ulunas_init`/`ulunas_process_hop` (no DSP of its own; the earlier identity-passthrough placeholder and its `TODO` are gone, see git history for that version). |
| Cross-compiler correctness (Node.js) | **PASS, with an honest caveat** — `test_wasm_node.js` diffs the WASM build's output against a native-x86 build of the *identical* source (`gen_reference.cpp`) on the same input. Post the model's own documented 2-hop causal startup transient, max abs error is **0.025**. The startup transient itself diverges far more (**6.4** max abs error) — a real, measured finding: an ill-conditioned early-frame OLA normalization (dividing by a not-yet-fully-accumulated window-sum) amplifies ordinary cross-compiler (g++ vs emcc/clang) floating-point rounding differences, and that divergence then persists at the smaller ~0.01–0.025 level through the GRU state carried into later hops. This is **not** the ~1e-7 same-platform parity reported elsewhere in this project (that number compares two builds from the *same* compiler); reported honestly rather than picking a threshold that hides it. |
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
# 1. re-export weights if the checkpoint changed (writes ../mobile/c_neon/generated/*.h):
cd ../mobile/c_neon && python3 export_weights.py --checkpoint /path/to/F_PROXY_ROBOT.tar && cd -

# 2. compile the real engine (ulunas_full.cpp) together with the thin WASM wrapper:
emcc -O2 -std=c++17 ulunas_wasm.cpp ../mobile/c_neon/src/ulunas_full.cpp \
  -I../mobile/c_neon/src \
  -s WASM=1 \
  -s EXPORTED_FUNCTIONS='["_ulunas_wasm_reset","_ulunas_wasm_process_hop","_malloc","_free"]' \
  -s EXPORTED_RUNTIME_METHODS='["ccall","cwrap","HEAPF32"]' \
  -s MODULARIZE=1 -s EXPORT_NAME='UlunasWasmModule' \
  -s ALLOW_MEMORY_GROWTH=1 \
  -o ulunas_wasm.js

# 3. regenerate the native-x86 parity reference and re-verify before shipping:
g++ -O2 -std=c++17 -I../mobile/c_neon/src gen_reference.cpp ../mobile/c_neon/src/ulunas_full.cpp -o gen_reference
./gen_reference
node --no-experimental-fetch test_wasm_node.js
```
(The `--no-experimental-fetch` flag works around a Node 18 quirk where the global `fetch` API
confuses Emscripten's Node-environment file-loading path — a Node/tooling issue, not a bug in
this project's code; not needed when served to an actual browser.)
