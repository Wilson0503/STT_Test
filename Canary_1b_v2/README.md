# Canary-1b-v2 開源碼研究與會議 Real-time 測試

本工作區提供兩部分：

1. `Canary-1b-v2` 的開源碼/架構重點整理。
2. 使用 **NVIDIA NeMo 官方 open-source streaming 腳本** 做「公司會議」即時測試（real-time simulation）的可執行流程。

---

## 1) 我對 Canary-1b-v2 的理解（基於官方文件與原始碼）

### 模型定位

- 模型：`nvidia/canary-1b-v2`
- 任務：
  - 多語 ASR（speech-to-text）
  - AST（English ↔ 24 EU 語言）
- 支援語言：25 個歐洲語言（含 `en`, `de`, `fr`, `es`, `it`, `pl`, `uk`, `ru` 等）。

### 架構

- 類型：AED（Attention Encoder-Decoder）
- Encoder：FastConformer
- Decoder：Transformer Decoder
- Tokenizer：多語 SentencePiece（16384 vocab）
- 參數量：約 978M（~1B）

### 開源碼關鍵位置（NeMo）

- 多任務 AED 模型核心：
  - `nemo/collections/asr/models/aed_multitask_models.py`
  - 類別：`EncDecMultiTaskModel`
- Canary chunked/streaming 文件：
  - `docs/source/asr/streaming_decoding/canary_chunked_and_streaming_decoding.rst`
- 官方 streaming 推論腳本（本次測試直接使用）：
  - `examples/asr/asr_chunked_inference/aed/speech_to_text_aed_streaming_infer.py`
- 官方 chunked 推論腳本：
  - `examples/asr/asr_chunked_inference/aed/speech_to_text_aed_chunked_infer.py`

### 功能能力

- ASR / AST 單一模型多任務。
- 可使用 prompt 控制任務與語言：
  - `+prompt.task=asr|s2t_translation`
  - `+prompt.source_lang=<lang>`
  - `+prompt.target_lang=<lang>`
  - `+prompt.pnc=yes|no`
- Timestamp：支援（依任務輸出 word/segment 或 segment）。
- 長音訊：`transcribe()` 自動 dynamic chunking（長檔或 `batch_size=1`）。
- Streaming：官方支援 `waitk` / `alignatt` 兩種策略，且可計算延遲（LAAL）。

### 重要限制

- NeMo 官方支援矩陣對 Windows 是「No support yet」，建議 Linux / WSL2 / Docker。
- 真正即時品質高度依賴：
  - 音訊前處理（單聲道、16kHz、降噪）
  - chunk/context 設定
  - GPU/VRAM 與推論精度

---

## 2) 會議 real-time 測試方式

本流程採 **官方 streaming 腳本**，屬於「串流解碼模擬真實即時」：

- 輸入會議錄音（或會議錄製中的滾動檔）
- 以 `chunk_secs + context` 逐塊解碼
- 輸出轉寫與延遲指標（LAAL）

> 這是官方可重現、最貼近部署前驗證的方式。

### 步驟 A：準備環境（建議 WSL2 Ubuntu）

1. 安裝 Python 3.10+
2. 安裝 PyTorch（對應 CUDA）
3. 安裝 NeMo ASR

```bash
pip install -U "nemo_toolkit[asr]"
```

4. 下載 NeMo 原始碼（用官方 script）

```bash
git clone https://github.com/NVIDIA-NeMo/NeMo.git
cd NeMo
```

### 步驟 B：準備會議 manifest

使用本倉庫腳本：

```bash
python scripts/build_meeting_manifest.py \
  --audio-dir /path/to/meeting_wavs \
  --output-manifest /path/to/meeting_manifest.jsonl \
  --source-lang en \
  --target-lang en \
  --task asr
```

### 步驟 C：執行官方 streaming 推論

```bash
python /path/to/NeMo/examples/asr/asr_chunked_inference/aed/speech_to_text_aed_streaming_infer.py \
  pretrained_name=nvidia/canary-1b-v2 \
  dataset_manifest=/path/to/meeting_manifest.jsonl \
  output_filename=/path/to/out/meeting_streaming_pred.jsonl \
  left_context_secs=10 \
  chunk_secs=1 \
  right_context_secs=0.5 \
  batch_size=8 \
  decoding.streaming_policy=alignatt \
  decoding.alignatt_thr=8 \
  decoding.waitk_lagging=2 \
  decoding.exclude_sink_frames=8 \
  decoding.xatt_scores_layer=-2 \
  decoding.hallucinations_detector=True \
  +prompt.pnc=yes \
  +prompt.task=asr \
  +prompt.source_lang=en \
  +prompt.target_lang=en
```

### 步驟 D：檢查結果

- 主要輸出：`pred_text`
- 若 manifest 內有 `text`，腳本會計算 WER。
- 會在 log 中得到 LAAL（延遲指標）。

---

## 3) 參數建議（會議情境）

- 低延遲優先：
  - `decoding.streaming_policy=alignatt`
  - `chunk_secs=0.5~1.0`
  - `right_context_secs=0.3~0.8`
- 準確優先：
  - `chunk_secs=1.0~2.0`
  - `left_context_secs>=10`
  - `right_context_secs=0.5~1.0`

---

## 4) 本倉庫附帶腳本

- `scripts/build_meeting_manifest.py`
  - 將會議音檔資料夾轉成 Canary 可用 manifest（含 prompt 欄位）。
- `scripts/run_canary_streaming.ps1`
  - PowerShell 包裝器，直接調用 NeMo 官方 streaming 腳本。
- `server/canary_ws_server.py`
  - Linux GPU 上部署的 WebSocket STT server（可直接對接你的 HTML）。
- `scripts/run_canary_ws_server.sh`
  - Linux 上啟動 WebSocket server 的腳本。

---

## 5) Linux GPU WebSocket 服務（對接你現有 HTML）

你的 HTML 協議可直接使用，不需要改動訊息格式：

- Client -> Server：`start` / `audio` / `stop`
- Server -> Client：`partial` / `final` / `error`

### A. 在 Linux server 安裝

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r server/requirements-linux.txt
```

### B. 啟動服務

```bash
chmod +x scripts/run_canary_ws_server.sh
./scripts/run_canary_ws_server.sh
```

預設會開在：

- `ws://0.0.0.0:8000/ws/stt`
- health check：`http://<server-ip>:8000/healthz`

可透過環境變數調整：

```bash
export STT_HOST=0.0.0.0
export STT_PORT=8000
export CANARY_MODEL_NAME=nvidia/canary-1b-v2
./scripts/run_canary_ws_server.sh
```

### C. 你的 HTML 要填的 WebSocket URL

你的頁面可直接填：

```text
ws://<server-ip>:8000/ws/stt?source_lang=en&target_lang=en&task=asr&pnc=yes
```

若要做語音翻譯（例如英翻中介語轉歐語）：

```text
ws://<server-ip>:8000/ws/stt?source_lang=en&target_lang=de&task=s2t_translation&pnc=yes
```

### D. 目前這版 server 的行為

- 定時回傳 `partial`（即時覆蓋）。
- 偵測短暫靜音時會回 `final`。
- 收到 `stop` 時會 flush 最後一段 `final`。

這樣就能直接用你現在那份 HTML 做本地麥克風 realtime 測試。

---

## 6) 我實際做了什麼

- 已完成官方資料與開源碼關鍵路徑對齊。
- 已建立可直接跑的 real-time 測試流程（使用官方 open-source script）。
- 已把「會議場景所需 manifest 產生與命令模板」整理在本倉庫。
- 已新增可部署在 Linux GPU 的 WebSocket STT server，能直接對接你現有 HTML。
