# Chirp 2 Local Audio Test

這個資料夾提供一支可直接在本機音檔上測試 Google Cloud Speech-to-Text V2 `chirp_2` 模型的 CLI。

涵蓋官方文件「使用 Chirp 2 功能」中的本地音檔情境：

- 同步辨識
- 串流辨識
- 不限語言轉錄
- 語音翻譯
- 字詞層級時間戳記
- 模型 adaptation phrase hints
- 降噪器與 SNR 篩選

另外也補上幾個實務上常一起驗證的選項：

- 自動標點 `--enable-punctuation`
- 字詞信心分數 `--enable-word-confidence`
- 不雅字詞過濾 `--profanity-filter`
- 強制正規化 `--transcript-normalization`

## 1. 先決條件

1. 啟用 Google Cloud Speech-to-Text API V2。
2. 你的專案需能使用 Chirp 2 支援區域，例如 `us-central1`。
3. 設定 Application Default Credentials。

Windows PowerShell 範例：

```powershell
gcloud auth application-default login
$env:GOOGLE_CLOUD_PROJECT = "你的-project-id"
```

如果你是用 service account：

```powershell
$env:GOOGLE_APPLICATION_CREDENTIALS = "C:\path\to\service-account.json"
$env:GOOGLE_CLOUD_PROJECT = "你的-project-id"
```

## 2. 安裝

```powershell
cd c:\Users\iec150094\STT_Test\Chirp_V2
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 3. 基本用法

```powershell
python chirp2_local_test.py sync .\sample.wav --language en-US
```

## 4. 各功能指令

同步辨識：

```powershell
python chirp2_local_test.py sync .\sample.wav --language cmn-Hant-TW --enable-punctuation
```

串流辨識：

```powershell
python chirp2_local_test.py stream .\sample.wav --language en-US --chunk-size 65536
```

不限語言轉錄：

```powershell
python chirp2_local_test.py auto-language .\sample.wav
```

語音翻譯：

```powershell
python chirp2_local_test.py translate .\sample.wav --language ja-JP --target-language en-US
```

字詞層級時間戳記：

```powershell
python chirp2_local_test.py timestamps .\sample.wav --language en-US --json-out .\outputs\timestamps.json
```

模型 adaptation：

```powershell
python chirp2_local_test.py adapt .\sample.wav --language en-US --phrase Contoso --phrase Fabrikam --phrase-boost 20
```

降噪器與 SNR：

```powershell
python chirp2_local_test.py denoise .\sample.wav --language en-US --snr-threshold 20
```

加上強制正規化、標點、信心分數：

```powershell
python chirp2_local_test.py sync .\sample.wav `
  --language en-US `
  --enable-punctuation `
  --enable-word-confidence `
  --transcript-normalization "abc=A.B.C." `
  --transcript-normalization "msft=Microsoft"
```

一次跑完整套：

```powershell
python chirp2_local_test.py full-suite .\sample.wav `
  --language en-US `
  --target-language zh-TW `
  --phrase Contoso `
  --phrase Fabrikam `
  --enable-punctuation `
  --enable-word-confidence `
  --json-out .\outputs
```

`full-suite` 會輸出多個 JSON 檔到指定資料夾。

## 5. 參數說明

- `--project-id`: 不想用環境變數時可直接指定。
- `--region`: 預設 `us-central1`。
- `--recognizer`: 預設 `_`，使用 implicit recognizer。
- `--language`: 來源語言，可重複指定。
- `--target-language`: 翻譯目標語言。
- `--phrase`: adaptation 提示詞。
- `--phrase-boost`: adaptation boost 分數。
- `--snr-threshold`: SNR 門檻。
- `--disable-denoise`: 關閉 denoise_audio。
- `--json-out`: 輸出原始 API 回應 JSON。

## 6. 注意事項

- Chirp 2 只支援 Speech-to-Text V2。
- `BatchRecognize` 官方範例需要 GCS URI，不是純本地檔案流程，所以這個工具主打本地檔案可直接測的功能。
- 翻譯可用語言組合不是完全對稱，若回傳錯誤請改成官方文件支援的 source/target 組合。