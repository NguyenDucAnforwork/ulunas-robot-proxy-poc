# UL-UNAS Robot-Proxy Streaming Noise Suppression — PoC

Fine-tuning and evaluation of [UL-UNAS](https://github.com/Xiaobin-Rong/ul-unas) (ultra-lightweight streaming speech enhancement) on generic and robot-proxy noise domains, plus a runtime-free C99 streaming inference implementation and iOS/WASM demo scaffolding.

**Start here:**
- [`PROJECT_STATUS.md`](PROJECT_STATUS.md) — the master requirement/evidence/status table (PASS/FAIL/BLOCKED_EXTERNAL/NOT_RUN)
- [`EXPERIMENT_REPORT.md`](EXPERIMENT_REPORT.md) — full RQ1/RQ2/RQ3 methodology, metrics, confidence intervals
- [`SPEC_PLAN.md`](SPEC_PLAN.md) / [`EXECUTION_PLAN.md`](EXECUTION_PLAN.md) — original planning documents

**Model checkpoints**: hosted on Hugging Face (not in this repo) — [banhchungtuongot/ulunas-robot-proxy-poc](https://huggingface.co/banhchungtuongot/ulunas-robot-proxy-poc) (private). `F_PROXY_ROBOT*` checkpoints are **CC-BY-NC-SA-4.0 (NonCommercial)** due to UAV training data — see the model card and [`DATA_LICENSES.md`](DATA_LICENSES.md).

**Other key docs**: [`ROBOT_PROXY_DATA.md`](ROBOT_PROXY_DATA.md) (data sourcing/licensing), [`MOBILE_BENCHMARK.md`](MOBILE_BENCHMARK.md) (runtime-free C + Android + iOS status), [`QC_FAILURE_ANALYSIS.md`](QC_FAILURE_ANALYSIS.md), [`IOS_BUILD_AND_DEMO.md`](IOS_BUILD_AND_DEMO.md), [`REPRODUCTION.md`](REPRODUCTION.md) (exact steps to reproduce everything), [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

**Important**: this is a research PoC. "Robot-proxy" means public UAV ego-noise recordings + procedural mechanical-noise simulation — **not a real robot**. Real-device (iPhone 11) benchmarking and demo are `BLOCKED_EXTERNAL` (no physical hardware/Xcode in the environment this was built in) — see `PROJECT_STATUS.md` for exactly what is and isn't verified.
