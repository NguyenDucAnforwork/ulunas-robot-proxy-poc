# THIRD_PARTY_NOTICES.md

Consolidated from `data/manifests/data_manifest.csv` and the license checks performed directly at each source during M1 (see that file for exact URLs, verification dates, and hashes).

## Code / models

- **UL-UNAS** (`github.com/Xiaobin-Rong/ul-unas`) — MIT License, Copyright (c) 2026 Xiaobin Rong.
- **SEtrain** (`github.com/Xiaobin-Rong/SEtrain`) — MIT License, Copyright (c) 2025 Rong Xiaobin. Used as an adapted training framework (model file, dataloader, and config replaced; `HybridLoss` and STFT parameters reused unmodified).
- **facebook/wav2vec2-base-960h** (Hugging Face) — Apache License 2.0. Used frozen, as the ASR loss backend and the primary WER/command-eval backend.
- **ONNX Runtime** (Microsoft) — MIT License. Used as the reference inference runtime (Python `onnxruntime` package and the Android `onnxruntime-android` 1.19.2 prebuilt library) — reference/baseline path only, per project scope; not part of the runtime-free C path.
- **Android NDK r27c** (Google) — Android Software Development Kit License.

## Data

- **LibriSpeech** dev-clean subset (`openslr.org/12`) — CC BY 4.0. Please cite: V. Panayotov, G. Chen, D. Povey, S. Khudanpur, "Librispeech: an ASR corpus based on public domain audio books," ICASSP 2015.
- **Google Speech Commands v0.02** — CC BY 4.0 (Attribution 4.0 International), confirmed directly from the `LICENSE` file packaged in the archive. Please cite: P. Warden, "Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition," arXiv:1804.03209.
- **DEMAND** (Diverse Environments Multichannel Acoustic Noise Database, Zenodo record 1227121) — license listed as CC BY 4.0 in the Zenodo record metadata, but the dataset's own description text states CC BY-SA 3.0; treated here as **CC BY-SA 3.0** (the more restrictive of the two) for attribution purposes. Please cite: J. Thiemann, N. Ito, E. Vincent, "The Diverse Environments Multi-channel Acoustic Noise Database (DEMAND): A database of multichannel environmental noise recordings," ICA 2013.

## Not used (attempted, then dropped)

- **QUT-NOISE** — its CC BY-SA license was confirmed at `research.qut.edu.au`, but the actual audio files could not be downloaded: the host (`data.researchdatafinder.qut.edu.au`) returned `HTTP 403 Forbidden` on direct/scripted requests. This was a server-side access-control block, not a license issue, and no attempt was made to bypass it (no user-agent spoofing or similar). Not used in M1; DEMAND alone served as the generic-noise source.

This project (the fine-tuned checkpoints, generated code, and reports) does not itself carry a declared license here — that is the user's decision to make for the overall PoC.
