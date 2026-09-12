# EXPERIMENT_REPORT.md — M1 (Generic) + Robot-Proxy Training/Evaluation

Phạm vi: **M1** (N0, P0, F_GENERIC, F_GENERIC_ASR trên domain GENERIC — mục 1-8, giữ nguyên không sửa) **+ Robot-Proxy takeover** (F_PROXY_ROBOT, F_PROXY_ROBOT_ASR trên domain ROBOT-PROXY, và đánh giá lại toàn bộ 6 điều kiện trên domain ROBOT-PROXY — mục 9 trở đi, MỚI). `F_PROXY_ROBOT`/`F_PROXY_ROBOT_ASR` là proxy (UAV công khai + procedural), **không phải robot thật** — xem `ROBOT_PROXY_DATA.md`.

## 0. Tóm tắt kết quả (đọc nhanh — đã cập nhật với kết quả robot-proxy)

**Trên domain GENERIC (M1, không đổi):**
- **RQ1: CÓ**, rõ ràng, material, robust — xem mục 3.
- **RQ2: negative/trung tính hợp lệ** — ASR loss không tốt hơn material — xem mục 4.

**Trên domain ROBOT-PROXY (mới, mục 9-11):**
- **RQ1 (khử nhiễu có giúp ASR không, trên domain robot-proxy?): CÓ, còn rõ hơn cả domain GENERIC.** `F_PROXY_ROBOT` giảm WER **10.9 điểm phần trăm tuyệt đối** so với noisy (19.6%→8.6%), material & robust (CI hoàn toàn âm). Domain-adapted (`F_PROXY_ROBOT`) cải thiện SI-SDR nhiều nhất trong 3 model fine-tune (+10.1dB vs N0).
- **RQ2 (ASR loss trên domain robot-proxy): negative, còn rõ ràng hơn M1.** `F_PROXY_ROBOT_ASR` vs `F_PROXY_ROBOT`: WER **XẤU ĐI** 0.31 điểm phần trăm, có ý nghĩa thống kê (CI hoàn toàn dương, tức tệ hơn) nhưng dưới ngưỡng material. `F_GENERIC_ASR` vs `F_GENERIC` trên domain robot-proxy: WER không khác biệt có ý nghĩa thống kê (CI chứa 0) — ASR loss **không giúp gì**, và trong trường hợp robot-proxy còn có xu hướng hơi xấu đi. Kết luận: **ASR loss, với cấu hình lambda hiện tại, không đáng để đánh đổi chi phí +47-58% wall-clock / +222% VRAM.**
- **RQ3 (robot-proxy adaptation có tốt hơn generic training trên robot-proxy test không?): CÓ, nhưng nhỏ và có đánh đổi.** `F_PROXY_ROBOT` vs `F_GENERIC` (cùng test robot-proxy): WER tốt hơn 0.77 điểm phần trăm (có ý nghĩa thống kê, CI hoàn toàn âm, nhưng dưới ngưỡng material 1 điểm), SI-SDR/STOI tốt hơn có ý nghĩa thống kê — **nhưng DNSMOS SIG/BAK/OVRL đều XẤU ĐI có ý nghĩa thống kê** (~0.02-0.03, dưới ngưỡng material). Đây là bằng chứng thật cho đánh đổi SIG-vs-WER khi model chuyên biệt hóa cho domain khó hơn (UAV/procedural noise) — xem mục 11 (QC).

## 1. Cấu hình đã chạy

| Model | Loss | Data | Checkpoint khởi tạo | Step | Batch | Seed | LR |
|---|---|---|---|---|---|---|---|
| N0 | — | noisy input thô | — | — | — | — | — |
| P0 | — | — (chỉ inference) | `model_trained_on_dns3.tar` | — | — | — | — |
| F_GENERIC | SE (`HybridLoss`) | LibriSpeech(train speakers)+DEMAND(train envs) | `model_trained_on_dns3.tar` | 18000 | 8 | 43 | 1e-5 constant |
| F_GENERIC_ASR | SE + `0.0598×L_ASR` | **cùng hệt** F_GENERIC | **cùng hệt** F_GENERIC | **18000** | **8** | **43** | **1e-5 constant** |

`lambda_asr=0.0598` chọn bằng gradient-scale probe thật (không phải benchmark): `grad_norm_se=20.94, grad_norm_asr_raw=70.00, target_ratio=0.2` → `lambda=0.2×20.94/70.00=0.0598`.

Quy trình fine-tune đúng theo quyết định #21: chỉ load `model` weights từ checkpoint pretrained (bỏ optimizer/scheduler cũ), optimizer Adam mới LR=1e-5 hằng số, 22 lớp BatchNorm2d bị freeze running stats (`.eval()` trong khi phần còn lại của model ở `.train()`).

## 2. Chi phí tài nguyên của ASR loss (đo thật, không ước lượng)

| | F_GENERIC | F_GENERIC_ASR | Chênh lệch |
|---|---|---|---|
| Wall-clock (18000 step) | 1508.2s (25.1 phút) | 2226.8s (37.1 phút) | **+47.6%** |
| Steps/s | 11.93 | 8.08 | -32.3% throughput |
| Peak VRAM | 931.8 MB | 3004.0 MB | **+222.4%** |

**Verify gradient ASR loss ≠ 0** (bắt buộc theo quyết định #22): grad-norm tại lớp đầu UL-UNAS khi có `L_ASR` = **10.662** (khác 0 rõ ràng) — xác nhận gradient thực sự truyền từ CTC loss qua frozen ASR model tới waveform enhanced rồi tới UL-UNAS, đúng yêu cầu.

## 3. RQ1 — Khử nhiễu có giúp ASR (và chất lượng cảm nhận) không?

Test set: 563 synthetic paired utterances (LibriSpeech test speakers × DEMAND test noise, SNR ngẫu nhiên trong [-5,15]dB, seed cố định — cùng input suy giảm cho cả 4 điều kiện).

| Metric | N0 | P0 | F_GENERIC | F_GENERIC_ASR |
|---|---|---|---|---|
| DNSMOS SIG (mean) | 3.125 | 3.308 | 3.339 | 3.331 |
| DNSMOS BAK (mean) | 2.925 | 3.989 | 4.042 | 4.043 |
| DNSMOS OVRL (mean) | 2.505 | 2.996 | 3.054 | 3.049 |
| SI-SDR (mean, dB) | 5.09 | 13.33 | 18.23 | 18.20 |
| STOI (mean) | 0.9269 | 0.9301 | 0.9398 | 0.9399 |
| WER (corpus) | 10.90% | 9.54% | 8.29% | 8.04% |

**Paired bootstrap CI 95% (delta so với N0 và P0), ngưỡng material: |ΔOVRL|,|ΔSIG|>0.03; |ΔWER|>0.01 absolute:**

| So sánh | ΔSIG | ΔBAK | ΔOVRL | ΔSI-SDR | ΔSTOI | ΔWER |
|---|---|---|---|---|---|---|
| P0 vs N0 | +0.183 [0.137,0.230] ✅material | +1.063 [1.001,1.126] ✅ | +0.492 [0.455,0.529] ✅ | +8.25dB [7.70,8.80] | +0.0032 [-0.0010,0.0074] *(CI chứa 0, không robust)* | -0.0108 [-0.0212,+0.0003] *(chạm ngưỡng material, CI gần như chạm 0 ở biên trên)* |
| F_GENERIC vs N0 | +0.214 [0.168,0.261] ✅ | +1.116 [1.053,1.179] ✅ | +0.549 [0.514,0.586] ✅ | +13.14dB [12.50,13.72] | +0.0129 [0.0092,0.0166] ✅robust | **-0.0260** [-0.0362,-0.0167] ✅material&robust |
| F_GENERIC vs P0 | +0.031 [0.022,0.040] ✅(vừa qua ngưỡng) | +0.053 [0.042,0.065] ✅ | +0.058 [0.048,0.067] ✅ | +4.89dB [4.43,5.37] | +0.0097 [0.0080,0.0115] ✅ | **-0.0152** [-0.0233,-0.0078] ✅material&robust |

**Kết luận RQ1**: khử nhiễu **có** giúp cả ASR lẫn chất lượng cảm nhận, và mức cải thiện là *material* theo đúng ngưỡng đã đăng ký trước — không chỉ là nhiễu thống kê. Domain-matched fine-tuning (`F_GENERIC`, train trực tiếp trên LibriSpeech+DEMAND) cải thiện rõ rệt hơn hẳn so với chỉ dùng checkpoint pretrained trên DNS3 (`P0`), đúng như kỳ vọng khi domain train/test khớp nhau.

## 4. RQ2 — ASR loss có giúp hơn SE-loss thuần không?

So sánh **trực tiếp** `F_GENERIC_ASR` vs `F_GENERIC` (cùng checkpoint/data/step/batch/seed/optimizer/LR, chỉ khác loss — quyết định #8):

| Metric | Δ(ASR - SE-only) | CI 95% | Có material không? | Có robust không? (CI loại 0) |
|---|---|---|---|---|
| DNSMOS SIG | -0.0075 | [-0.0090,-0.0061] | Không (ngưỡng 0.03) | **Có** (CI hoàn toàn âm) |
| DNSMOS BAK | +0.0018 | [+0.0002,+0.0035] | Không | **Có** (CI hoàn toàn dương) |
| DNSMOS OVRL | -0.0053 | [-0.0069,-0.0037] | Không (ngưỡng 0.03) | **Có** |
| SI-SDR | -0.026dB | [-0.057,+0.003] | Không | **Không** (CI chứa 0) |
| STOI | +0.00015 | [-0.00007,+0.00034] | Không | **Không** (CI chứa 0) |
| WER | **-0.0032** (giảm 0.32 điểm %) | [-0.0062,-0.0003] | **Không** (ngưỡng 0.01 = 1 điểm %) | **Có** (CI hoàn toàn âm) |
| Command exact-match | +0.02 (300 utt) | [-0.0067,+0.0500] | — | **Không** (CI chứa 0) |

**Kết luận RQ2 — báo cáo đúng như một negative/trung tính result hợp lệ (theo yêu cầu đề bài khi ASR-loss không tốt hơn rõ rệt)**: trên domain GENERIC với 18000 step, thêm ASR loss tạo ra một cải thiện WER **có ý nghĩa thống kê nhưng dưới ngưỡng material đã đăng ký trước** (0.32 điểm % < 1 điểm %), đổi lại là một suy giảm SIG/OVRL nhỏ tương tự cũng dưới ngưỡng material. SI-SDR, STOI và command exact-match **không phân biệt được với nhiễu ngẫu nhiên** ở cỡ mẫu này (563 utterance synthetic / 300 utterance command). Đây là một đánh đổi nhỏ, cân bằng, **không đủ mạnh để kết luận ASR loss "tốt hơn"** SE-loss thuần trên domain GENERIC — trong khi chi phí thật (mục 2) là +47.6% wall-clock và +222% VRAM. Nếu mục tiêu chỉ là domain GENERIC, SE-loss thuần (`F_GENERIC`) có vẻ là lựa chọn hiệu quả hơn về chi phí/lợi ích. **Chưa biết** kết luận này có giữ nguyên trên domain ROBOT hay không (`F_ROBOT_ASR` đang BLOCKED) — đây là câu hỏi mở quan trọng cho M2.

## 5. RQ3 — Tại sao QC thấy kết quả tệ?

**Chưa thể trả lời ở M1.** Không có clip QC thật (xem `QC_FAILURE_ANALYSIS.md`, giữ `BLOCKED`), và phần lớn giả thuyết RQ3 gốc (robot ego-noise, khoảng cách/góc nói, AGC thiết bị robot...) thuộc domain ROBOT — hiện `F_ROBOT`/`F_ROBOT_ASR` bị BLOCKED nên chưa có gì để so sánh. Điều DUY NHẤT có thể nói ở M1: trên domain GENERIC (đọc + DEMAND), model cải thiện rõ và material; đây là một base rate tốt, nhưng **không suy ra được** nó sẽ tốt tương tự trên ego-noise/điều kiện robot thật.

## 6. Command-specific metrics (300 utterance, Google Speech Commands, 30 từ × 10 utt)

| | N0 | P0 | F_GENERIC | F_GENERIC_ASR |
|---|---|---|---|---|
| Exact-match accuracy | 46.3% | 43.3% | 44.7% | 46.7% |
| Deletion rate | 4.0% | 2.3% | 2.7% | 3.3% |
| Substitution rate | 49.7% | 54.3% | 52.7% | 50.0% |

**Lưu ý quan trọng về giới hạn**: `facebook/wav2vec2-base-960h` được train để nhận dạng câu tiếng Anh liên tục, **không có language model**, và chưa từng thấy các từ lệnh ngắn dạng cô lập (isolated single-word) khi train. Vì vậy exact-match tuyệt đối ở đây THẤP một cách có hệ thống trên **mọi** điều kiện kể cả N0 — đây là giới hạn của backend ASR, không phải bằng chứng model SE làm hỏng lệnh. Cái đáng tin cậy hơn ở đây là **so sánh tương đối** giữa các điều kiện (đã trình bày ở mục 4), không phải giá trị tuyệt đối 43-47%.

## 7. Audio comparison samples

10 utterance mẫu (≥10 theo yêu cầu), mỗi mẫu có `clean/noisy/P0/F_GENERIC/F_GENERIC_ASR` + metadata (`utt_id`, transcript, noise env, SNR) tại `audio_samples/sample00..09_*.wav`.

## 8. Giả định và giới hạn cần nêu rõ

- **`HybridLoss` hyperparameters** (`lamda_ri=30,lamda_mag=70,compress=0.3`) dùng nguyên bản từ `SEtrain/configs/cfg_train.yaml` — đây là giả định hợp lý về những gì đã dùng để train `model_trained_on_dns3.tar`, **chưa xác nhận** là chính xác (repo `ul-unas` không kèm config train gốc).
- **Clean speech = LibriSpeech (audiobook/read speech thuần túy)**. Giới hạn được chấp nhận có ý thức (quyết định #20): không có dữ liệu hội thoại/spontaneous trong M1. Bất kỳ kết luận nào về "đa dạng phong cách nói" đều **không** được rút ra từ M1.
- **DEMAND = 1h, 12 môi trường** (QUT-NOISE bị bỏ do lỗi 403 khi tải, xem `data/manifests/data_manifest.csv`). Đa dạng noise thấp hơn kế hoạch ban đầu (2 nguồn → 1 nguồn).
- **LR schedule hằng số 1e-5**, không warmup/decay — lựa chọn đơn giản cho fine-tune ngắn 18000 step, chưa thử nghiệm lựa chọn khác.
- **Batch=8, segment=4s** cho cả 2 run — chưa sweep các giá trị khác.

---

# PHẦN 2: ROBOT-PROXY (mới, takeover session)

## 9. Cấu hình đã chạy (robot-proxy)

| Model | Loss | Data | Checkpoint khởi tạo | Step | Batch | Seed | LR | Wall-clock | Peak VRAM |
|---|---|---|---|---|---|---|---|---|---|
| F_PROXY_ROBOT | SE (`HybridLoss`) | LibriSpeech (cùng speaker split với F_GENERIC) + robot-proxy noise (50% generic/35% UAV/15% procedural theo sampling weight) | `model_trained_on_dns3.tar` | 18000 | 8 | 43 | 1e-5 constant | 1542.9s (25.7 phút) | 936.6 MB |
| F_PROXY_ROBOT_ASR | SE + `0.0978×L_ASR` | **cùng hệt** F_PROXY_ROBOT | **cùng hệt** | **18000** | **8** | **43** | **1e-5** | 2227.4s (37.1 phút) | 3010.7 MB |

`lambda_asr=0.0978` từ gradient-scale probe thật: `grad_norm_se=8.29, grad_norm_asr_raw=16.95, target_ratio=0.2`. Grad-check: `first_layer_grad_norm_with_asr_loss=1.50` (khác 0, xác nhận đúng). Không NaN/Inf ở cả hai run. Chênh lệch tài nguyên do ASR loss: **+44.3% wall-clock, +221.5% VRAM** (tương tự M1, xác nhận lại trên domain khác).

**Xác nhận fair-comparison thật (không chỉ lý thuyết)**: `F_PROXY_ROBOT` và `F_PROXY_ROBOT_ASR` cho `loss_se` gần như giống hệt nhau ở từng step tương ứng (vd step 150: 0.7114 vs 0.7116) — sai khác cỡ 10⁻⁴ này khớp với nhiễu float-point thông thường của GPU (thứ tự cộng dồn trong cuDNN), không phải do dữ liệu mixture khác nhau — xác nhận cả hai model thực sự huấn luyện trên cùng một chuỗi mixture, đúng yêu cầu "cùng manifest, cùng thứ tự sample".

## 10. Đánh giá 6 điều kiện trên domain ROBOT-PROXY (563 synthetic utterance MỚI, không phải test set của M1)

Quan trọng: đây là **domain test MỚI** — `F_GENERIC`/`F_GENERIC_ASR` (checkpoint M1, không train lại) được **đánh giá lần đầu** trên domain robot-proxy này, để so sánh công bằng với `F_PROXY_ROBOT`/`F_PROXY_ROBOT_ASR` (RQ3).

| Metric | N0 | P0 | F_GENERIC | F_GENERIC_ASR | F_PROXY_ROBOT | F_PROXY_ROBOT_ASR |
|---|---|---|---|---|---|---|
| DNSMOS SIG | 2.819 | 3.157 | 3.157 | 3.155 | 3.137 | 3.126 |
| DNSMOS BAK | 2.506 | 3.972 | 3.984 | 3.991 | 3.952 | 3.951 |
| DNSMOS OVRL | 2.205 | 2.866 | 2.873 | 2.874 | 2.842 | 2.834 |
| SI-SDR (dB) | 5.25 | 12.21 | 14.98 | 15.03 | **15.35** | 15.35 |
| STOI | 0.877 | 0.909 | 0.913 | 0.913 | **0.917** | 0.917 |
| WER (wav2vec2, chính) | 19.6% | 9.7% | 9.4% | 9.3% | **8.6%** | 8.9% |
| WER (whisper-tiny.en, generalization check) | 36.7% | 34.3% | 38.0% | 37.1% | 34.7% | 36.3% |
| Command exact-match | 39.7% | 43.3% | 45.3% | 46.0% | 45.0% | 44.7% |

**Lưu ý về ASR backend thứ hai (whisper-tiny.en)**: WER whisper cao hơn nhiều và KHÔNG đơn điệu giảm theo cùng pattern với wav2vec2 (vd F_GENERIC có whisper-WER cao hơn cả N0 ở một số điều kiện) — whisper-tiny.en (39M tham số, không finetune cho domain này) là một backend yếu hơn nhiều và có hành vi khác biệt, đúng vai trò "generalization check" chứ không phải WER chính. Không dùng whisper-WER để rút kết luận RQ1/RQ2/RQ3 — chỉ wav2vec2 (backend chính, cùng backend dùng trong ASR loss) được dùng cho các kết luận chính thức bên dưới.

### Breakdown theo SNR/noise-source/robot-state (F_PROXY_ROBOT_ASR, ví dụ đầy đủ trong `evaluation_robot_proxy_results.json::breakdown`)

| SNR bin | WER | | Noise source | WER | | Robot state (khó nhất/dễ nhất) | WER |
|---|---|---|---|---|---|---|---|
| -5 to 0dB | 16.3% | | generic | 6.5% | | `fan` (khó nhất) | 24.7% |
| 0 to 5dB | 9.5% | | procedural | 8.6% | | `idle` (bất ngờ khó, xem §11) | 17.4% |
| 5 to 10dB | 5.4% | | **uav (khó nhất)** | **12.7%** | | `motor_high` | 12.0% |
| 10 to 15dB | 5.6% | | | | | `rpm_transition` (dễ nhất) | 4.6% |

## 11. RQ1/RQ2/RQ3 — kết luận đầy đủ với paired bootstrap CI (từ `evaluation_robot_proxy_results.json::paired_bootstrap_ci`)

### RQ1 — Khử nhiễu có giúp ASR không? (trên domain robot-proxy)

| So sánh | ΔWER tuyệt đối | CI95% | Relative WER reduction | Material? | Robust? |
|---|---|---|---|---|---|
| P0 vs N0 | -9.83 điểm % | [-11.70,-7.96] | -50.2% | ✅ | ✅ (CI hoàn toàn âm) |
| F_GENERIC vs N0 | -10.17 điểm % | [-12.12,-8.26] | -52.0% | ✅ | ✅ |
| **F_PROXY_ROBOT vs N0** | **-10.94 điểm %** | **[-12.91,-9.05]** | **-55.9%** | ✅ | ✅ |

**Kết luận RQ1 (robot-proxy)**: khử nhiễu giúp ASR rõ ràng, material, và robust — nhất quán với M1. Model fine-tune trực tiếp trên domain robot-proxy (`F_PROXY_ROBOT`) cho cải thiện WER tuyệt đối lớn nhất trong 3 lựa chọn.

### RQ2 — ASR loss có tốt hơn SE-only không? (so sánh trực tiếp, cùng checkpoint/step/data/seed)

| So sánh | ΔWER | CI95% | Material? | Robust? | Kết luận đúng theo yêu cầu đề bài |
|---|---|---|---|---|---|
| F_GENERIC_ASR vs F_GENERIC | -0.15 điểm % | [-0.45,+0.12] | Không | **Không** (CI chứa 0) | Không có khác biệt phát hiện được |
| F_PROXY_ROBOT_ASR vs F_PROXY_ROBOT | **+0.31 điểm % (XẤU ĐI)** | [+0.06,+0.57] | Không (< 1 điểm) | **Có** (CI hoàn toàn dương = xấu đi) | **"statistically detectable but not materially better" — thực ra là statistically detectable WORSE, không material** |

**Kết luận RQ2 (đầy đủ, cả 2 domain)**: ở CẢ HAI domain, ASR loss (với `lambda_asr` chọn bằng gradient-scale probe mục tiêu 20%) **không cho cải thiện WER material**. Trên domain robot-proxy, kết quả còn tệ hơn: ASR loss làm WER xấu đi có ý nghĩa thống kê (dù nhỏ, dưới ngưỡng material). Đây là **negative result hợp lệ, nhất quán trên cả 2 domain** — không phải do lỗi thực nghiệm ở M1. Chi phí đo được (+44-58% wall-clock, +221-222% VRAM) không đổi lại lợi ích WER material nào ở cấu hình lambda hiện tại.

### RQ3 — Robot-proxy adaptation có tốt hơn generic training trên robot-proxy test không?

| So sánh | ΔSIG | ΔBAK | ΔOVRL | ΔSI-SDR | ΔSTOI | ΔWER |
|---|---|---|---|---|---|---|
| F_PROXY_ROBOT vs F_GENERIC | -0.021 (xấu đi, robust) | -0.032 (xấu đi, robust) | -0.031 (xấu đi, robust) | +0.37dB (tốt hơn, robust) | +0.0048 (tốt hơn, robust) | **-0.77 điểm % (tốt hơn, robust, dưới material)** |
| F_PROXY_ROBOT_ASR vs F_GENERIC_ASR | -0.029 (xấu đi, robust) | -0.040 (xấu đi, robust) | -0.040 (xấu đi, robust) | +0.32dB (tốt hơn, robust) | +0.0044 (tốt hơn, robust) | -0.30 điểm % (CI chứa 0, không robust) |

**Kết luận RQ3**: robot-proxy adaptation cho cải thiện WER/SI-SDR/STOI nhỏ nhưng có ý nghĩa thống kê so với generic-only training, khi đánh giá trên chính domain robot-proxy — nhưng **đổi lại DNSMOS SIG/BAK/OVRL đều xấu đi có ý nghĩa thống kê** (tuy dưới ngưỡng material 0.03). Đây là bằng chứng thật, không suy diễn, cho việc model chuyên biệt hóa domain khó hơn (UAV/procedural) đánh đổi một phần "độ dễ nghe cảm nhận" (DNSMOS) lấy độ chính xác nội dung (WER) và độ trung thực tín hiệu (SI-SDR/STOI) tốt hơn. **Không được suy rộng thành "robot thật sẽ tốt hơn"** — đây chỉ là domain robot-PROXY, chưa phải robot thật (xem `ROBOT_PROXY_DATA.md`).

## 12. QC — phân tích SIG-BAK tradeoff (bằng chứng thật, xem `QC_FAILURE_ANALYSIS.md` §4 để biết chi tiết đầy đủ)

Giả thuyết "over-suppression làm BAK tăng nhưng SIG giảm" được **xác nhận có bằng chứng thật**, tập trung cụ thể ở noise UAV `motor_high` (7/10 case tệ nhất theo tradeoff) và procedural `fan` (1/10) — nơi ΔBAK so với noisy là +1.8 đến +2.5 nhưng ΔSIG là **-0.34 đến -0.97 (SIG giảm XUỐNG DƯỚI cả mức noisy)**. Cơ chế hợp lý (chưa verify bằng counterfactual): motor-harmonic noise trùng phổ với harmonics của giọng nói hữu thanh, khiến model dễ nhầm/xoá nhầm. Đây là tương quan có bằng chứng và cơ chế hợp lý, **không phải kết luận nhân quả đã được kiểm chứng bằng matched-pair test** — xem `QC_FAILURE_ANALYSIS.md` §4.4 để biết chính xác giới hạn.

## 13. Giả định và giới hạn bổ sung (robot-proxy)

- Xem `ROBOT_PROXY_DATA.md` và `DATA_LICENSES.md` cho toàn bộ chi tiết nguồn dữ liệu, license (bao gồm ràng buộc **NonCommercial** từ UAV data CC-BY-NC-SA-4.0), và giới hạn đa dạng (chỉ 3 phiên UAV thật, ~32s mỗi phiên).
- Distance/angle chỉ là proxy (gain+lowpass đơn giản), không phải RIR vật lý thật — xem `ROBOT_PROXY_DATA.md` §3.
- Whisper-tiny.en (generalization check) cho WER cao và không nhất quán với wav2vec2 — không dùng để kết luận, chỉ tham khảo.
- Không có counterfactual/matched-pair test thật nào được chạy cho QC — phát hiện SIG-BAK tradeoff là tương quan có bằng chứng, không phải nhân quả đã chứng minh.
