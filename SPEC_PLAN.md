# SPEC_PLAN.md — Fine-tune UL-UNAS cho triển khai on-device/robot (PoC dữ liệu tiếng Anh)

Trạng thái: **CỔNG LẬP KẾ HOẠCH — chờ phê duyệt bản này**. Toàn bộ nội dung dưới đây dựa trên việc đọc/chạy trực tiếp code và checkpoint của hai repo tham chiếu. Chưa tải dữ liệu, chưa sửa model, chưa chạy training nào.

**Quy ước ngôn ngữ (đã được chỉnh lại theo yêu cầu của bạn)**: `SPEC_PLAN.md`, mọi báo cáo, giải thích, kết luận đều viết bằng **tiếng Việt**. Chỉ có dữ liệu speech, transcript và ASR backend của thí nghiệm là tiếng Anh (thay vì tiếng Việt như bản kế hoạch đầu tiên). Tên biến/code có thể dùng tiếng Anh như thông lệ lập trình.

Repo đã inspect trực tiếp (`git clone --depth 1` vào `/content/repos/`):
- `ul-unas` @ commit `00f7c70` (MIT, Copyright 2026 Xiaobin Rong)
- `SEtrain` @ commit `3ec437d` (MIT, Copyright 2025 Rong Xiaobin) — *SEtrain là tên repo/khung huấn luyện speech-enhancement tổng quát do cùng tác giả UL-UNAS công bố (train.py, loss_factory.py, dataloader.py...), dùng làm bộ khung train chung chứ không đóng gói sẵn cho UL-UNAS — xem gap ở mục 1.4.*

---

## 0. Các quyết định đã chốt trong lượt này

| # | Quyết định | Ảnh hưởng tới kế hoạch |
|---|---|---|
| 1 | Chấp nhận adapt SEtrain để train/fine-tune UL-UNAS | Mục 1.4 là việc phải làm, không còn là blocker chờ quyết định |
| 2 | Speech data + transcript + ASR backend của PoC chuyển sang **tiếng Anh** (không phải toàn bộ SPEC) | Mục 2 viết lại nguồn dữ liệu tiếng Anh; SPEC/báo cáo vẫn tiếng Việt |
| 3 | `READ/DIVERSE` → `GENERIC` (noise/speakerphone công khai) / `ROBOT` (ego-noise + acoustic domain robot) | Mục 2, 3 viết lại theo khung robot |
| 4 | Robot đơn giản hóa: 1 micro trong robot, chỉ single-channel noise suppression, chưa AEC, chưa beamforming, giả định robot không phát loa khi đang nghe | Thu hẹp phạm vi kỹ thuật thực sự cần làm, ghi rõ trong INPUT_SPEC sau này |
| 5 | Ego-noise = noise do chính robot tạo ra (quạt, motor, bánh xe, servo, rung cơ học, tiếng khi di chuyển trên sàn) | Định nghĩa chính thức, dùng xuyên suốt |
| 6 | Nếu chưa có ego-noise thật: vẫn chạy `N0`/`P0`/`F_GENERIC`, gọi là `generic speakerphone baseline`; `F_ROBOT`/`F_ROBOT_ASR` để `BLOCKED/PENDING` | Áp dụng nguyên văn cho toàn bộ file kết quả/tên gọi |
| 7 | Ma trận mới: `N0, P0, F_GENERIC, F_ROBOT (50% generic + 50% ego-noise), F_ROBOT_ASR, F_BEST_OA` | Mục 3.1 |
| 8 | `F_GENERIC/F_ROBOT/F_ROBOT_ASR` cùng checkpoint, clean-speech pool, số optimizer step, batch, seed, optimizer+LR schedule; KHÔNG bắt buộc cùng GPU-time | Mục 3.2 |
| 9 | Nguồn dữ liệu cụ thể: Speech Commands + Common Voice/VCTK/LibriSpeech subset; noise QUT-NOISE/DEMAND/DNS5 speakerphone; ego-noise tự thu 30-60 phút; RIR OpenSLR28/pyroomacoustics (ưu tiên phụ) | Mục 2.3-2.5 |
| 10 | Bắt buộc tự đọc license tại nguồn trước khi tải; ghi URL/license/ngày tải/hash vào manifest | Mục 2.6 |
| 11 | ASR mặc định: frozen Wav2Vec2-base-960h (English CTC) dùng cho cả ASR loss lẫn WER chính; ASR thứ hai (vd Whisper) chỉ là generalization check nếu còn giờ; nếu robot chủ yếu nhận lệnh ngắn, phải báo thêm command exact-match accuracy + false-reject rate | Mục 3.3 |
| 12 | Giữ cùng số step là tiêu chí so sánh chính giữa F_ROBOT/F_ROBOT_ASR; chênh lệch GPU-time là chi phí phải đo và báo cáo | Mục 3.2 |
| 13 | Không cài ESPnet2 để chạy DNSMOS; dùng ONNXRuntime trực tiếp trên DNSMOS onnx có sẵn, nhưng phải kiểm tra preprocessing/output mapping khớp implementation tham chiếu | Mục 3.4 |
| 14 | QC chuyển trọng tâm sang robot state/motor-fan-servo state/surface/distance/angle/environment/SNR/mic-device/AGC/clipping/active speech level; speaking style là factor phụ | Mục 4 |
| 15 | Test set ưu tiên thu qua chính micro robot (phát loa câu clean có transcript, thu lại ở nhiều khoảng cách/góc/trạng thái di chuyển/surface) | Mục 3.5 |
| 16 | QC root-cause chỉ triển khai thật khi có clip+metadata QC thật; hiện tại chỉ xây schema/quy trình, trạng thái `BLOCKED` | Mục 4, 6 |
| 17 | Mobile timeline tách khỏi ngân sách A100 3-4h; A100 chỉ dành cho setup/fine-tune/eval/ONNX export; Android reference + C/NEON là phase engineering riêng, hiện chỉ lập kế hoạch/estimate | Mục 5 |
| 18 | Giải thích thuật ngữ khi xuất hiện lần đầu | Áp dụng xuyên suốt tài liệu (in *nghiêng* ngay sau thuật ngữ) |
| 19 | Thêm `F_GENERIC_ASR` để so sánh SE-only vs SE+ASR ngay cả khi chưa có ego-noise | Mục 3.1 — RQ2 có thể trả lời sớm, không phải chờ ego-noise |
| 20 | Chốt clean training = LibriSpeech subset 4-6h; Speech Commands dành cho command evaluation (không phải training pool); generic noise = DEMAND + QUT-NOISE, mỗi nguồn ~1h | Mục 2.4 viết lại |
| 21 | Quy trình fine-tune: chỉ load model weights (bỏ optimizer/scheduler state của checkpoint pretrained), reset optimizer/scheduler mới, LR khởi đầu 1e-5, freeze BatchNorm running statistics | Mục 3.2 |
| 22 | Frozen ASR: gọi `.eval()` nhưng vẫn cho gradient truyền tới enhanced waveform (không bọc `torch.no_grad()`, chỉ set `requires_grad=False` trên tham số ASR) | Mục 3.3 |
| 23 | SI-SDR/STOI chỉ bắt buộc trên synthetic paired test; real robot recording dùng WER/command exact-match/DNSMOS/listening test; intrusive metric chỉ tính sau khi verify alignment | Mục 3.4 |
| 24 | Bỏ FRR/FAR (cần detector+threshold chưa có); thay bằng command exact-match, deletion, substitution | Mục 3.3 |
| 25 | Chia implementation thành M1-M5, dừng review sau mỗi milestone; RTF<1 là yêu cầu real-time bắt buộc, RTF≤0.25 là mục tiêu tối ưu | Mục 5.4 (mới) |
| 26 | Lượt này: chỉ triển khai **M1** (generic training/evaluation thật, có GPU A100 xác nhận trong môi trường), chưa làm Android/C/NEON | Mục 8 (mới) |

---

## 1. Kiến trúc và luồng xử lý (đã verify trực tiếp, không đổi so với bản inspect gốc)

### 1.1 Thông số xác nhận bằng cách chạy chính code của model

Chạy `python3 ulunas.py` (script tự có trong repo, chỉ inference, không train):

```
The complexity of ULUNAS: MACs=34.35 M, Params=171.33 k
The model is causal, without any look ahead.
```

Đối chiếu checkpoint thật: 431 tensor, tổng 172,312 phần tử (chênh nhỏ so với 171.33k vì gồm cả buffer đóng băng — trọng số ERB filterbank cố định và `num_batches_tracked` của 5 lớp BatchNorm — các tensor này không nằm trong `model.parameters()` train được).

| Thông số | Giá trị | Cách verify |
|---|---|---|
| Sample rate | 16000 Hz, mono | STFT kwargs trong `ulunas.py` + `soundfile.info()` trên `audio/clean/0119.wav` |
| n_fft / hop_len / win_len | 512 / 256 / 512, cửa sổ Hann | mặc định `ULUNAS.__init__`, dùng giống hệt ở cả offline lẫn streaming |
| Algorithmic latency (*độ trễ lý thuyết gây ra bởi chính cách xử lý tín hiệu, không phải do phần cứng chậm*) | 32ms (2×hop=512 mẫu) | test causal có sẵn trong `ulunas.py.__main__`: cắt 2 chuỗi lệch nhau ở giây thứ 2, sai số phần trước mốc `256*2` mẫu đo được `<1e-8` |
| Params thực tế | 171.33k (trainable) / 172,312 (toàn bộ tensor) | chạy trực tiếp `ulunas.py` + `torch.load` checkpoint |
| MACs (*số phép nhân-cộng, thước đo độ nặng tính toán*) | 34.35M/giây audio (63 khung STFT offline, 16kHz) | ptflops, input (16000,) |
| Checkpoint provenance | epoch=191, step=238750, LR cuối≈7.15e-6 (warmup=25000, decay_until=250000, max_lr=1e-3, min_lr=1e-6) | `torch.load(...)['scheduler']` |

### 1.2 Luồng offline (verify bằng forward hook thật, không suy đoán shape)

Input `(1,16000)` → 63 khung STFT:

```
raw wave (1,16000)
  → STFT (n_fft=512,hop=256,win=512,hann,onesided) → log-magnitude (1,1,63,257)
  → ERB.bm: giữ 65 bin thấp + nén 192 bin cao còn 64 bin ERB → (1,1,63,129)
  → Encoder 5 khối (types=[XConv,XMB,XDWS,XMB,XDWS], stride tần số=[2,2,1,1,1]):
      en0 (1,12,63,65) → en1 (1,24,63,33) → en2 (1,24,63,33) → en3 (1,32,63,33) → en4 (1,16,63,33)
  → 2×DPGRNN (dual-path grouped RNN), giữ nguyên shape (1,16,63,33)
  → Decoder 5 khối mirror, cộng skip từ en_outs, deconv upsample:
      de0 (1,32,63,33) → de1 (1,24,63,33) → de2 (1,24,63,33) → de3 (1,12,63,65) → de4 (1,1,63,129)
  → sigmoid mask (1,1,63,129) → ERB.bs mở rộng lại (1,1,63,257)
  → nhân mask phức vào spec gốc → ISTFT → waveform (1,16000)
```

Toàn bộ shape trên lấy từ `forward_hook` chạy thật trên checkpoint đã load.

### 1.3 Luồng streaming ONNX (đã có sẵn trong repo, tái dùng không viết lại)

`ulunas_onnx/stream/ulunas_stream.py` định nghĩa `StreamULUNAS`, load state_dict `strict=True` từ checkpoint offline, export ONNX opset 11.

**I/O tensor (verify bằng `onnx.load` trên file `.onnx` có sẵn):**

| Tensor | Shape | Ý nghĩa |
|---|---|---|
| `mix` (input) | `[1,257,1,2]` | 1 khung STFT phức (re,im), T=1 |
| `conv_cache` (in/out) | `[1,5358]` | 6 ring-buffer conv nhân quả đã pack |
| `tfa_cache` (in/out) | `[1,402]` | 10 hidden-state GRU của cổng attention thời gian (cTFA) |
| `inter_cache` (in/out) | `[1,1056]` | 2 hidden-state GRU inter-path của DPGRNN |
| `enh` (output) | `[1,257,1,2]` | 1 khung STFT phức đã khử nhiễu |

Verify bằng tay khớp 100%: `conv_cache` = (1×2×129)+(24×1×65)+(24×1×33)+(24×1×33)+(12×1×33)+(12×2×65) = **5358** ✓; `tfa_cache` = 24+48+48+64+32+64+48+48+24+2 = **402** ✓; `inter_cache` = 33×16+33×16 = **1056** ✓.

**Quan trọng cho Android**: đồ thị ONNX **không chứa STFT/ISTFT** — app phải tự cài STFT/ISTFT 512-điểm/hop 256/Hann/overlap-add, chỉ đưa từng khung phổ phức vào model.

**Danh sách toán tử ONNX (đồ thị đã simplify, 697 node, đếm bằng `onnx.load`+`op_type`):**

| Op | Số lượng | Op | Số lượng |
|---|---|---|---|
| Transpose | 100 | Sigmoid | 21 |
| Reshape | 90 | Conv | 17 |
| Slice | 77 | Greater | 17 |
| Add | 75 | Where | 17 |
| Mul | 60 | Pow | 14 |
| Unsqueeze | 44 | Squeeze | 14 |
| Concat | 29 | Pad | 13 |
| ReduceMean | 28 | Sqrt | 5 |
| **GRU** | **28** | Div | 5 |
| MatMul | 26 | ConvTranspose | 5 |
| | | BatchNormalization | 5 |
| | | Sub | 4 |
| | | ReduceSum/Clip/Log | 1 mỗi loại |

`GRU` (28 instance) là toán tử khó và tốn công nhất khi viết tay bằng C/NEON — không có kernel NEON GRU chuẩn để tái dùng. Cặp `Greater`/`Where` (17 mỗi loại) tương ứng nhánh PReLU tùy biến (`AffinePReLU`) ở mỗi block.

Opset 11, IR version 6 — cả hai đều khả thi cho `onnx.checker` và cho việc sinh C code tĩnh.

**Chưa tự chạy trong môi trường này**: script tự in sai số streaming-vs-offline và ONNX-vs-offline khi chạy, nhưng tôi mới verify cấu trúc đồ thị tĩnh qua `onnx.load`, chưa cài `onnxsim`/chạy full pipeline. → **PENDING**, sẽ chạy lại ở Phase 1.

### 1.4 Tính tương thích UL-UNAS ↔ SEtrain — gap thật, đã verify, nay được chấp nhận adapt (quyết định #1)

`SEtrain/train.py` dòng 19–21:
```python
from models.gtcrn_end2end import GTCRN as Model
from loss_factory import HybridLoss as Loss
from dataloader_dns3 import DNS3Dataset as Dataset
```

Đã kiểm tra trực tiếp:
- `SEtrain/models/` chỉ có `gtcrn_end2end.py` (kiến trúc SE khác — GTCRN, 23.67k params theo docstring), không có file model UL-UNAS nào.
- Root `SEtrain/` không có `dataloader_dns3.py` — chỉ có `dataloader.py` (chứa class `DNS3Dataset` nhưng khác tên file). Chạy `train.py` nguyên bản sẽ báo `ModuleNotFoundError`.
- `dataloader.py::DNS3Dataset` trỏ cứng đường dẫn tuyệt đối nội bộ tác giả — đúng loại dữ liệu công ty/nội bộ mà đề bài cấm dùng.

**Việc phải làm (đã được duyệt)**: copy `ul-unas/ulunas.py` vào `SEtrain/models/`, sửa import trong `train.py`/`infer.py`; sửa lỗi import `dataloader_dns3`, viết Dataset mới đọc corpus GENERIC/ROBOT tiếng Anh; viết `network_config` khớp constructor của ULUNAS (phần STFT 512/256/512 đã khớp sẵn giữa model và loss, không cần đổi).

**Phần tương thích tốt (verify được, tái dùng nguyên vẹn)**:
- `HybridLoss` (`loss_factory.py`) chỉ nhận waveform `(y_pred,y_true)`, tự tính STFT bên trong với đúng 512/256/512 — khớp chính xác UL-UNAS.
- `configs/cfg_train.yaml` mặc định (`lamda_ri=30, lamda_mag=70, compress_factor=0.3`) là **giả định hợp lý nhưng chưa xác nhận** là đúng thông số đã dùng để train ra `model_trained_on_dns3.tar` (vì `ul-unas` không kèm config train gốc của chính checkpoint). Sẽ ghi rõ đây là giả định trong `EXPERIMENT_REPORT.md`, không khẳng định là sự thật.

---

## 2. Thiết kế dữ liệu (GENERIC / ROBOT, dữ liệu tiếng Anh)

### 2.1 Định nghĩa domain

- **GENERIC**: clean speech (tiếng Anh) trộn với noise/RIR môi trường công khai thông thường (giao thông, café, gia đình, bàn phím) — đại diện cho một thiết bị/speakerphone tổng quát trong môi trường bình thường.
- **ROBOT**: **cùng** clean speech pool với GENERIC (theo quyết định #8), nhưng trộn thêm *ego-noise* — noise do chính robot tự tạo ra khi vận hành (quạt làm mát, motor, bánh xe, servo, rung cơ học, tiếng ma sát khi di chuyển trên các loại mặt sàn khác nhau), theo tỷ lệ 50% generic-noise mixture + 50% ego-noise mixture (quyết định #7) để mô hình vừa giữ khả năng tổng quát vừa thích nghi domain robot.

### 2.2 Đơn giản hóa phạm vi robot (quyết định #4)

Robot trong PoC này được giả định:
- Chỉ có **một** micro đặt trong thân robot (không phải mic array).
- Mô hình chỉ làm **single-channel noise suppression** (khử nhiễu 1 kênh).
- **Chưa** làm echo cancellation (AEC) — vì giả định robot **không phát loa trong lúc đang nghe lệnh** (loại bỏ nhu cầu AEC khỏi phạm vi PoC).
- **Chưa** làm beamforming (cần mic array, ngoài phạm vi 1 micro nói trên).

Đây là giới hạn phạm vi được chấp nhận, phải ghi lại nguyên văn trong `INPUT_SPEC.md` ở phase sau như một phần của "phạm vi hoạt động phù hợp" của mô hình.

### 2.3 Quy tắc đặt tên khi chưa có ego-noise thật (quyết định #6)

Nếu tới lúc chạy vẫn **chưa** có bản ghi ego-noise từ robot thật:
- `N0` và `F_GENERIC` **vẫn được chạy bình thường**.
- Không được gọi bất kỳ kết quả nào là "robot-domain adaptation".
- Mọi kết quả liên quan phải gọi đúng là **`generic speakerphone baseline`** (trong tên file, tên config, và câu chữ trong báo cáo).
- `F_ROBOT` và `F_ROBOT_ASR` giữ trạng thái `BLOCKED/PENDING` cho tới khi có ego-noise thật — không chạy bằng noise proxy rồi gắn nhãn "robot".

### 2.4 Nguồn dữ liệu — đã CHỐT (quyết định #20)

| Vai trò | Nguồn đã chốt | Quy mô mục tiêu | Ghi chú license (verify trực tiếp trước khi tải — quyết định #10) |
|---|---|---|---|
| Clean speech training (English, dùng chung GENERIC+ROBOT) | **LibriSpeech subset** (OpenSLR12) | 4-6h | CC BY 4.0 — cần đọc lại trang openslr.org/12 trước khi tải để xác nhận đúng version |
| Command evaluation (không phải training pool) | **Google Speech Commands** | subset đủ dùng cho command exact-match/deletion/substitution | CC BY 4.0 (theo hiểu biết chung, cần đọc lại license file trong bản tải) |
| Giới hạn đã biết | LibriSpeech = audiobook/read speech thuần túy | — | Đây là giới hạn được chấp nhận có ý thức, không che giấu: clean-training-speech của PoC này **là read speech**, đa dạng phong cách nói (nếu có) sẽ tới từ domain-mismatch giữa GENERIC/ROBOT noise, không phải từ đa dạng phong cách của chính giọng đọc. Phải nêu rõ giới hạn này khi diễn giải RQ3. |
| Generic noise | **DEMAND** + **QUT-NOISE**, mỗi nguồn ~1h | ~2h tổng | DEMAND: CC BY 4.0 (cần verify); QUT-NOISE: cần đọc license tại nguồn QUT research data repository trước khi tải (có thể yêu cầu điều khoản riêng, chưa xác nhận) |
| Robot ego-noise | **tự thu**, mục tiêu 30-60 phút: idle, fan, motor, move, turn, servo, surface khác nhau | 30-60 phút | Cần provenance rõ ràng — không phải data công ty. **Không thuộc phạm vi M1** (xem §8) |
| RIR | OpenSLR28 subset hoặc `pyroomacoustics` tổng hợp | ưu tiên phụ, có thể bỏ ở M1 nếu tốn thời gian | cần verify license OpenSLR28 nếu dùng |
| ASR loss + WER chính (English) | `facebook/wav2vec2-base-960h` (frozen CTC) | — | HuggingFace, Apache-2.0/MIT theo model card — verify lại khi tải |
| ASR thứ hai (generalization check, tùy chọn) | Whisper bản nhỏ (vd `whisper-base.en`) | — | chỉ chạy nếu còn thời gian ở M1, không bắt buộc |

### 2.5 Ngân sách và cách chia (không đổi so với đề bài gốc)

Validation 20-30 phút, test ≥200 utterance, mix on-the-fly (không tạo sẵn hàng trăm giờ mixture). Tách theo nguồn (speaker/recording session cho speech, recording session cho noise/ego-noise) **trước khi cắt chunk**, để chunk từ cùng 1 bản ghi không xuất hiện ở cả train và test.

### 2.6 Data manifest bắt buộc (quyết định #10)

Trước khi tải bất kỳ nguồn nào ở mục 2.4, phải tự đọc trang/license gốc (không dựa vào tóm tắt search), rồi ghi vào `data/manifest.csv`: URL nguồn, tên license, ngày tải, và hash (sha256) của từng file — chưa thực hiện bước này, đây là việc của Phase 1.

---

## 3. Thiết kế thực nghiệm

### 3.1 Ma trận mới (quyết định #7) và ngân sách thời gian A100 3-4h

| Config | Loss | Dữ liệu | Ưu tiên |
|---|---|---|---|
| `N0` | — (không train) | noisy input thô | luôn chạy |
| `P0` | — (không train) | pretrained UL-UNAS, inference only | luôn chạy |
| `F_GENERIC` | SE loss (`HybridLoss`) | 100% generic noise | ưu tiên cao — luôn chạy được kể cả khi chưa có ego-noise |
| `F_GENERIC_ASR` | SE loss + `lambda_asr × L_ASR` | 100% generic noise (giống hệt `F_GENERIC`) | **mới thêm (quyết định #19)** — trả lời RQ2 (SE-only vs SE+ASR) ngay cả khi chưa có ego-noise, không cần chờ `F_ROBOT` |
| `F_ROBOT` | SE loss | 50% generic noise + 50% robot ego-noise | `BLOCKED/PENDING` nếu chưa có ego-noise thật (mục 2.3) |
| `F_ROBOT_ASR` | SE loss + `lambda_asr × L_ASR` | giống hệt `F_ROBOT` | `BLOCKED/PENDING` cùng điều kiện với `F_ROBOT` |
| `F_BEST_OA` | — (chỉ hậu xử lý) | model tốt nhất trong số trên + *observation adding* | chạy sau khi đã có ít nhất 1 model fine-tune |

*Observation adding*: kỹ thuật trộn có kiểm soát giữa đầu ra đã khử nhiễu và tín hiệu nhiễu gốc theo công thức `output=(1-alpha)*enhanced+alpha*noisy`, nhằm giảm hiện tượng méo giọng nói do khử nhiễu quá mạnh, đánh đổi lấy một phần noise còn sót.

**Ước tính thời gian A100** (giữ nguyên logic thời gian như bản trước, chỉ đổi tên):

| Bước | Loại | Ước tính | Ghi chú |
|---|---|---|---|
| Benchmark 300-500 step | timing probe | ~5 phút | ước lượng throughput trước khi cam kết full run |
| N0, P0 eval | inference | ~5-10 phút | không train |
| `F_GENERIC` | train | 35-50 phút (time-box) | luôn chạy được |
| Gradient-scale probe cho `lambda_asr` | probe | ~5-10 phút | chỉ cần nếu `F_ROBOT_ASR` khả thi |
| `F_ROBOT` | train | 35-50 phút | chỉ chạy nếu có ego-noise |
| `F_ROBOT_ASR` | train | 40-50 phút (thêm overhead ASR) | chỉ chạy nếu có ego-noise |
| Eval tất cả model đã có | inference | ~15-20 phút | |
| `F_BEST_OA` (sweep 4 alpha) | inference | ~10 phút | |

Tổng nếu chỉ có `N0/P0/F_GENERIC` (chưa có ego-noise): rất nhẹ, dưới 1h GPU — còn nhiều buffer để làm kỹ evaluation/audio-comparison. Nếu có ego-noise và chạy đủ `F_ROBOT`+`F_ROBOT_ASR`: ~2-2.5h, vẫn nằm gọn trong 3-4h.

### 3.2 Ràng buộc "cùng step" và quy trình fine-tune (quyết định #8, #12, #21)

`F_GENERIC`, `F_GENERIC_ASR`, `F_ROBOT`, `F_ROBOT_ASR` dùng chung: checkpoint khởi tạo, clean-speech pool, số *optimizer step* (*một lần cập nhật trọng số — 1 lần forward+backward+optimizer.step(), khác với epoch là một vòng qua hết dữ liệu*), batch/effective batch, seed, optimizer và learning-rate schedule. **Không bắt buộc cùng GPU-time** — nếu `L_ASR` làm mỗi step chậm hơn (gần như chắc chắn, vì ASR model nặng hơn nhiều so với UL-UNAS 171K tham số), phần chênh lệch % wall-clock và VRAM phải được đo và báo cáo như **chi phí của ASR loss**, không phải một ràng buộc phải triệt tiêu.

**Quy trình fine-tune cụ thể (quyết định #21):**
- Chỉ load **model weights** từ `checkpoints/model_trained_on_dns3.tar` (`state_dict['model']`) — **không** load lại `optimizer`/`scheduler` state của checkpoint pretrained (chúng thuộc về quá trình train gốc trên DNS3, không liên quan tới fine-tune này).
- Khởi tạo optimizer/scheduler **mới hoàn toàn**, LR khởi đầu = **1e-5** (thấp hơn nhiều so với `max_lr=1e-3` lúc pretrain, vì đây là fine-tune từ một checkpoint đã hội tụ, không phải train từ đầu).
- **Freeze BatchNorm running statistics**: trong lúc fine-tune, các lớp `BatchNorm2d` không được cập nhật `running_mean`/`running_var` theo dữ liệu mới (giữ nguyên thống kê đã học từ DNS3) — chỉ tham số affine (`weight`/`bias` của BN, nếu có) và các phần còn lại của model mới được cập nhật bằng gradient. Cách làm: set `module.eval()` cho từng `BatchNorm2d` (để nó dùng running stats cố định thay vì batch stats) trong khi phần còn lại của model ở `train()` mode, hoặc set `momentum=0` — sẽ chọn cách cụ thể khi viết code và ghi lại trong `training_logs`.

### 3.3 Loss và ASR (quyết định #11, #19, #22, #24)

- `F_GENERIC`, `F_ROBOT`: `L_SE = HybridLoss`, tái dùng nguyên vẹn từ `SEtrain/loss_factory.py`.
- `F_GENERIC_ASR`, `F_ROBOT_ASR`: `L_SE + lambda_asr × L_ASR` (*ASR loss — một thành phần loss phụ dùng đầu ra của một mô hình nhận dạng giọng nói đã đóng băng để ép waveform khử nhiễu giữ được nội dung lời nói; khác WER vì WER không khả vi, không thể dùng trực tiếp làm loss*).
- ASR mặc định: `facebook/wav2vec2-base-960h`, English CTC, **đóng băng hoàn toàn** (`requires_grad=False` trên mọi tham số). Model này được dùng **cho cả ASR loss lẫn đo WER chính** (được chấp thuận, vì mục tiêu là cải thiện đúng backend dự kiến triển khai thực tế).
- **Cách freeze đúng (quyết định #22)**: gọi `asr_model.eval()` (để tắt dropout/vô hiệu hóa các layer phụ thuộc train-mode) **nhưng không bọc forward pass của ASR trong `torch.no_grad()`** — gradient vẫn phải chảy được từ `L_ASR` ngược qua ASR model tới waveform `enhanced` rồi tới UL-UNAS. `requires_grad=False` chỉ áp lên tham số ASR (để optimizer không cập nhật ASR), không chặn đường lan truyền gradient qua các phép tính của ASR.
- ASR thứ hai (vd Whisper bản nhỏ) chỉ chạy **nếu còn thời gian**, dùng làm generalization check phụ, không bắt buộc phải khác kiến trúc.
- Phải verify gradient của `L_ASR` khác 0 tại lớp đầu UL-UNAS (so `grad_norm` khi bật/tắt `L_ASR`).
- Ưu tiên CTC loss thật (transcript-supervised); nếu không khả thi kỹ thuật, fallback sang "ASR-representation loss" (đặt tên đúng, không gọi là CTC).
- `lambda_asr` chọn bằng gradient-scale probe (ASR gradient chiếm 10-30% SE gradient ở step đầu), không chọn tùy tiện — **cùng một `lambda_asr` áp dụng cho cả `F_GENERIC_ASR` và `F_ROBOT_ASR`** (probe chạy 1 lần trên domain generic vì đó là domain luôn sẵn có).
- **Bỏ FRR/FAR (quyết định #24)**: đề bài trước yêu cầu false-reject rate, nhưng FRR/FAR đúng nghĩa cần một detector + threshold quyết định "chấp nhận/từ chối" mà PoC này chưa xây (không có wake-word/command detector với ngưỡng tin cậy). Thay vào đó, báo cáo: **command exact-match accuracy**, **deletion rate**, **substitution rate** (tính trực tiếp từ so khớp transcript ASR với nhãn lệnh Speech Commands) — đủ để phản ánh mức độ robot "nghe nhầm/nghe thiếu" lệnh mà không cần dựng thêm một detector mới.

### 3.4 Giao thức đo lường theo loại test set (quyết định #23)

- **Synthetic paired test** (clean+noise mix có sẵn cặp reference, tạo on-the-fly): SI-SDR và STOI **bắt buộc** ở đây (vì cần audio reference căn thời gian chính xác).
- **Real robot recording** (thu qua micro robot thật, xem mục 3.6): dùng **WER, command exact-match/deletion/substitution, DNSMOS (non-intrusive), và listening test chủ quan**. **Không** bắt buộc SI-SDR/STOI trên real recording — các *intrusive metric* (*chỉ số cần audio sạch tham chiếu để so sánh, khác với DNSMOS là non-intrusive không cần reference*) chỉ được tính **sau khi đã verify alignment** giữa bản ghi robot và audio phát gốc (time-offset do phát-thu qua không khí, không tự động khớp mẫu-với-mẫu như synthetic mix) — nếu chưa verify alignment, không báo SI-SDR/STOI trên real recording, tránh số liệu sai lệch do lệch pha.
- Ở phạm vi M1 (mục 8), chỉ có synthetic paired test — real robot recording thuộc M3, chưa thực hiện.

### 3.5 DNSMOS qua ONNXRuntime (quyết định #13)

DNSMOS P.835 (*bộ 3 chỉ số dự đoán chất lượng cảm nhận không cần audio tham chiếu: SIG = chất lượng giọng nói, BAK = mức nhiễu nền còn sót, OVRL = đánh giá tổng thể*) sẽ được tính bằng script gọi `onnxruntime.InferenceSession` trực tiếp trên 4 file `.onnx` có sẵn trong `SEtrain/DNSMOS/DNSMOS/` — **không cài ESPnet2**. Điều kiện bắt buộc đi kèm: phải đối chiếu preprocessing (resample, framing, normalization đầu vào) và cách map output của script tự viết với chính hàm `DNSMOS_local` trong `espnet2.enh.layers.dnsmos` (đọc code tham khảo, không cần cài đặt full espnet2) để đảm bảo không lệch số so với con đường "chuẩn" mà `SEtrain/evaluate.py` dùng.

### 3.6 Test set thu qua robot thật (quyết định #15) — thuộc M3, chưa thực hiện ở M1

Ưu tiên: phát các câu clean (có transcript đã biết) qua loa ngoài, rồi thu lại bằng chính micro robot, lặp lại ở:
- Khoảng cách: 0.5m, 1m, 2m.
- Góc: 0°, 45°, 90°.
- Trạng thái robot: đứng yên và đang di chuyển.
- Nhiều surface/môi trường khác nhau.

Đây là cách tạo test set thực tế nhất phản ánh đúng điều kiện triển khai, thay vì chỉ mix synthetic. Cần lịch/thiết bị thật để thực hiện — hiện tại chỉ là kế hoạch, chưa thu.

---

## 4. Phân tích lỗi QC (chỉ xây quy trình/schema, quyết định #14, #16)

Trục phân tích chính (thay cho đọc/hội thoại trước đây): **robot state, motor/fan/servo state, surface, distance, angle, environment/noise type, SNR, microphone/device, AGC hoặc built-in device processing, clipping, active speech level**. Speaking style vẫn thu thập như metadata phụ, không phải trục chính.

Quy trình root-cause giữ nguyên 8 bước như đề bài gốc (xác nhận tái hiện → kiểm tra operating envelope → bucket theo các trục trên → top-3 factor → counterfactual test giữ 1 biến đổi 1 biến → effect size + CI → phân biệt tương quan/matched-pair/chưa xác định → nếu thiếu metadata thì kết luận "không đủ dữ liệu để xác định", không suy đoán).

**Trạng thái: `BLOCKED`** — chưa có clip và metadata QC thật nào để chạy quy trình này. Mục này trong SPEC chỉ xác nhận schema và quy trình sẽ dùng, không đưa ra bất kỳ kết luận nguyên nhân nào trước khi có dữ liệu thật.

---

## 5. Deployment

### 5.1 ONNX reference path
Tái dùng `ulunas_onnx/stream/ulunas_stream.py` (cấu trúc/cache/operator đã verify ở mục 1.3). Chạy lại full self-test (streaming-vs-offline, ONNX-vs-offline, RTF) sau khi có checkpoint fine-tune mới — hiện **PENDING**.

### 5.2 Runtime-free C/NEON path
Input cho code-gen: đúng 697 node đã liệt kê ở mục 1.3. Rủi ro lớn nhất: 28 node `GRU` phải viết tay (không có kernel NEON GRU chuẩn), cộng 17 cặp `Greater`/`Where` cho AffinePReLU. Quy trình 10 bước như đề bài gốc (scalar C trước, NEON sau, chỉ tối ưu 3 hotspot, preallocate toàn bộ buffer, không heap-alloc trong callback). Chưa bắt đầu, chờ Android reference đạt parity trước.

### 5.3 Timeline mobile — tách khỏi ngân sách A100 (quyết định #17)

| Giai đoạn | Ước tính | Ghi chú |
|---|---|---|
| ONNX export + parity harness (≥100 khung, ≥10 utterance) | 0.5-1 ngày | tái dùng script có sẵn |
| Android reference (NDK/CMake, ONNX Runtime Mobile, streaming state, PCM16, benchmark) | 3-5 ngày | giả định kỹ sư đã quen NDK |
| Runtime-free C99+NEON (sinh graph tĩnh, 28 GRU viết tay, memory planning, soak test 10 phút, benchmark thiết bị thật) | **2-4 tuần** | rủi ro lịch lớn nhất |

Ngân sách A100 3-4h **chỉ dành cho** setup + fine-tune + evaluation + ONNX export (mục 1-4 ở trên). Android reference và runtime-free C/NEON là phase engineering riêng, không phụ thuộc GPU — hiện tại chỉ lập kế hoạch/estimate, chưa code.

Nếu chưa có thiết bị Cortex-A53 thật khi tới lúc benchmark: giữ **PENDING**, không dùng emulator để tuyên bố đạt chỉ tiêu performance.

### 5.4 Yêu cầu RTF (quyết định #25)

- **RTF < 1 là yêu cầu real-time bắt buộc** (*RTF — real-time factor, thời gian xử lý 1 khung / thời lượng thực của khung đó; RTF<1 nghĩa là xử lý nhanh hơn thời gian thực, không bị dồn ứ*) — đây là ngưỡng tối thiểu để hệ thống dùng được streaming trên thiết bị thật.
- **RTF ≤ 0.25 là mục tiêu tối ưu** (như brief gốc), không phải điều kiện bắt buộc để coi là "đạt" — nếu chỉ đạt RTF<1 nhưng chưa tới 0.25, vẫn coi là runtime-free path khả dụng, chỉ chưa tối ưu, không phải FAIL.

---

## 6. Traceability table (done criterion → test → file kết quả → trạng thái)

| Done criterion | Test | File kết quả | Trạng thái |
|---|---|---|---|
| Kiến trúc/param/MACs/latency verified | `python3 ulunas.py` | console log (§1.1) | **PASS** |
| Tensor shape offline verified bằng hook | forward-hook script | console log (§1.2) | **PASS** |
| ONNX operator inventory | `onnx.load`+đếm `op_type` | console log (§1.3) | **PASS** |
| Streaming cache shape khớp code | tính tay | §1.3 | **PASS** |
| Checkpoint ↔ SEtrain compatibility | đọc trực tiếp `train.py`/`models/`/`dataloader.py` | §1.4 | **PASS (kết luận: không tương thích as-is, đã duyệt adapt)** |
| ONNX checker pass trên model export mới (từ checkpoint F_GENERIC thật) | `onnx.checker.check_model` | `mobile/onnx_export/out/parity_results.json` | **PASS** |
| Streaming = offline (100 khung/10 utt, ≤1e-4) | rerun `ulunas_stream.py`-based export | `mobile/onnx_export/out/parity_results.json` | **PASS** (10 utt, 3418 khung, max_abs_err=7.6e-6, wave_rmse=4.8e-5) |
| Data manifest + license + hash | tự đọc license + hash | `data/manifests/data_manifest.csv` | **PASS** (LibriSpeech/SpeechCommands/DEMAND tải+verify; QUT-NOISE BLOCKED do HTTP 403, không phải license) |
| Ego-noise thật từ robot | thu âm thật | `data/ego_noise/manifest.csv` | **BLOCKED** — chưa có, quyết định naming ở §2.3 áp dụng |
| Split không leakage | script kiểm tra overlap | `SEtrain_adapted/split_manifest.py` output | **PASS** (speaker sets + noise-env sets disjoint, verify assert passed) |
| N0/P0/F_GENERIC/F_GENERIC_ASR results | `eval_generic.py` (onnxruntime DNSMOS) | `evaluation/evaluation_results.{json,csv}` | **PASS** |
| F_ROBOT/F_ROBOT_ASR results | cùng script trên | `evaluation_results.{json,csv}` | **BLOCKED/PENDING** cho tới khi có ego-noise thật (M2) |
| Fine-tune đúng quy trình (chỉ load weights, reset optimizer/scheduler, LR=1e-5, freeze BN running stats) | code review + log config | `training_logs/{F_GENERIC,F_GENERIC_ASR}/resource_usage.json` | **PASS** (22 BatchNorm2d frozen, xác nhận trong log) |
| ASR loss gradient ≠ 0 (frozen ASR `.eval()` nhưng gradient vẫn chảy qua) | grad-norm probe | `training_logs/F_GENERIC_ASR/grad_check.json` | **PASS** (grad_norm=10.662, nonzero=true) |
| `lambda_asr` chọn bằng gradient-scale probe | probe script | `training_logs/F_GENERIC_ASR/lambda_probe.json` | **PASS** (lambda=0.0598, ratio=0.2) |
| Peak VRAM/throughput/wall-clock, % chậm thêm do ASR loss | `torch.cuda.max_memory_allocated`, timer | `training_logs/*/resource_usage.json` | **PASS** (+47.6% wall-clock, +222% VRAM, đo thật) |
| Command exact-match accuracy + deletion/substitution rate (thay FRR/FAR) | script so khớp transcript ASR vs nhãn Speech Commands | `evaluation_results.json` | **PASS** (300 utterance, 30 từ) |
| SI-SDR/STOI trên synthetic paired test | script intrusive metric | `evaluation_results.json` | **PASS** (563 utterance) |
| SI-SDR/STOI trên real robot recording | chỉ tính sau khi verify alignment | `evaluation_results.json` | thuộc M3, N/A ở M1 |
| Paired bootstrap CI | bootstrap script trên test set | `evaluation_results.json` + `EXPERIMENT_REPORT.md` §4 | **PASS** (vs N0, vs P0, và trực tiếp F_GENERIC_ASR vs F_GENERIC) |
| ≥10 audio comparison | script chọn mẫu | `audio_samples/` | **PASS** (10 mẫu × 6 file = 60 file) |
| RQ1/RQ2/RQ3 kết luận (kể cả negative result) | tổng hợp | `EXPERIMENT_REPORT.md` | **PASS** — RQ1: có, material+robust; RQ2: **negative/trung tính hợp lệ** (dưới ngưỡng material); RQ3: chưa thể trả lời (BLOCKED, cần M2/M3) |
| `INPUT_SPEC.md` + operating envelope (bao gồm giới hạn 1-mic/no-AEC/no-beamforming) | test grid SNR/distance/angle/level | `INPUT_SPEC.md` | **PASS (DRAFT)** — khung/test-grid đầy đủ, ngưỡng cụ thể còn PENDING validation thiết bị thật |
| Test set thu qua robot thật (0.5/1/2m, 0/45/90°, đứng yên/di chuyển) | thu âm thật | `data/robot_test_set/` | PENDING (cần thiết bị+lịch, thuộc M3) |
| QC root-cause (≥30 clip hoặc toàn bộ nếu ít hơn) | quy trình 8 bước §4 | `QC_FAILURE_ANALYSIS.md` | **BLOCKED** (chưa có clip/metadata QC thật) |
| Android build arm64-v8a chạy được | build NDK r27c thật + link ONNX Runtime Android thật | `mobile/android_ref/build_arm64/ulunas_ref` (ELF aarch64 xác nhận) | **PASS** (build); chạy trên máy thật vẫn PENDING (§5.4) |
| C so ONNX (≤1e-3/RMSE≤1e-4/DNSMOS Δ≤0.02/STOI Δ≤0.005) | parity harness (x86 host build vs PyTorch offline) | `MOBILE_BENCHMARK.md` §1-2 | **PASS một phần** — x86 build: RMSE=9.65e-6, max_err=9.15e-5 (vượt xa yêu cầu); toàn bộ kernel+3 block-pattern+DPGRNN+decoder validated riêng lẻ (xem `MOBILE_BENCHMARK.md`); full-graph wiring **chưa hoàn tất** (~2-5 ngày còn lại, ước tính chi tiết trong `MOBILE_BENCHMARK.md`) |
| RTF Cortex-A53 thật (median≤0.25, P95<hop, RSS≤32MB, binary≤5MB) | benchmark trên thiết bị thật | `MOBILE_BENCHMARK.md` §4 | **BLOCKED** — không có thiết bị Cortex-A53 thật (x86-host/qemu chỉ dùng để verify correctness, không tuyên bố performance, đúng yêu cầu) |

---

## 7. Blocker / quyết định còn mở cần bạn duyệt tiếp

1. **Ego-noise robot thật** — chưa có; quyết định của bạn (#6) đã quy định rõ cách xử lý (chạy N0/P0/F_GENERIC/F_GENERIC_ASR, gọi "generic speakerphone baseline", để F_ROBOT/F_ROBOT_ASR BLOCKED) — không còn là câu hỏi mở về *cách* xử lý, chỉ còn mở về *khi nào* có ego-noise thật để gỡ block ở M2.
2. **License cụ thể của LibriSpeech (OpenSLR12)/Speech Commands/DEMAND/QUT-NOISE/wav2vec2-base-960h** — mới có qua kiến thức/search, chưa tự đọc tận nguồn; bắt buộc verify trước khi tải ở M1 (quyết định #10) — sẽ thực hiện ngay khi bắt đầu M1, trước bất kỳ lệnh download nào.
3. **Thiết bị + lịch thu ego-noise thật và test set qua robot thật** (§3.6, thuộc M3) — cần bạn xác nhận có thiết bị robot thật sẵn sàng và lịch thu âm, nếu không các mục này giữ PENDING vô thời hạn.
4. **Thiết bị Cortex-A53 thật để benchmark mobile** (thuộc M4/M5) — cần xác nhận đã có, nếu chưa thì phần performance giữ PENDING/BLOCKED.
5. **Khung thời gian lịch cho phase mobile** (M4 Android reference 3-5 ngày, M5 C/NEON runtime-free 2-4 tuần) — không nằm trong ngân sách A100, cần bạn xác nhận khung thời gian mong đợi khi tới M4/M5.

---

## 8. Chia nhỏ implementation thành M1-M5 (quyết định #25, #26) — dừng review sau mỗi milestone

| Milestone | Nội dung | Cần GPU A100? | Trạng thái |
|---|---|---|---|
| **M1** | Adapt SEtrain cho ULUNAS; tải+verify license LibriSpeech/Speech Commands/DEMAND/QUT-NOISE; build manifest; fine-tune `F_GENERIC` và `F_GENERIC_ASR` (cùng checkpoint/step/clean-speech, chỉ khác loss); eval `N0/P0/F_GENERIC/F_GENERIC_ASR` (DNSMOS onnxruntime, WER, command exact-match/deletion/substitution trên Speech Commands, SI-SDR/STOI trên synthetic paired test); gradient-scale probe cho `lambda_asr`; báo cáo % chậm thêm + VRAM do ASR loss | Có | **HOÀN TẤT — xem `EXPERIMENT_REPORT.md`.** QUT-NOISE bị bỏ (HTTP 403, không phải license). RQ1: có, material+robust. RQ2: negative/trung tính hợp lệ (dưới ngưỡng material). Ngoài phạm vi M1 ban đầu, cũng đã làm thêm: ONNX export+parity từ checkpoint thật (PASS), Android NDK build thật cho arm64-v8a (PASS), và validate toàn bộ kernel+block-pattern C99 cho runtime-free path (xem `MOBILE_BENCHMARK.md`) — các phần mobile này thuộc M4/M5 nhưng được tranh thủ làm song song, chưa phải hoàn chỉnh M4/M5. |
| M2 | Sourcing ego-noise thật (hoặc quyết định proxy+relabel), train/eval `F_ROBOT`/`F_ROBOT_ASR`, sweep `F_BEST_OA` (observation adding) | Có | Chưa bắt đầu, chờ review M1 |
| M3 | `INPUT_SPEC.md` (test grid SNR/distance/angle/level), thu real robot test set (phát loa qua micro robot ở 0.5/1/2m, 0°/45°/90°, đứng yên/di chuyển), verify alignment cho intrusive metrics trên real recording | Không (thu âm) | Chưa bắt đầu |
| M4 | Export streaming ONNX cho model đã chọn, parity harness (≥100 khung/≥10 utterance), Android reference app (NDK/CMake, ONNX Runtime Mobile, streaming state, PCM16, benchmark instrumentation) | Không (CPU/mobile) | Chưa bắt đầu |
| M5 | Runtime-free C99+NEON (sinh graph tĩnh, 28 GRU viết tay, memory planning, soak test 10 phút), benchmark Cortex-A53 thật, `QC_FAILURE_ANALYSIS.md` thật (khi có clip QC thật) | Không (CPU/mobile) | Chưa bắt đầu |

**Lượt này chỉ triển khai M1.** Sau khi M1 hoàn tất, dừng lại và chờ bạn review kết quả trước khi sang M2. Không làm Android/C/NEON (M4/M5) cho tới khi M1 được duyệt.

---
