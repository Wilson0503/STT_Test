try:
    import audioop
except ModuleNotFoundError:
    import audioop_lts as audioop  # Python 3.13+
import base64, os, tempfile, time, wave
import requests
from dotenv import load_dotenv

load_dotenv()

PROJECT_ID   = os.getenv("project_id", "ebgit-did1c-01")
REGION       = "us-central1"
SA_JSON      = os.getenv("SERVICE_ACCOUNT_JSON", "")
AUDIO_FILE   = os.getenv("audio_file_path", r"C:\Users\iec150094\STT_Test\temp_compressed_new.wav")
INLINE_LIMIT = int(os.getenv("INLINE_LIMIT_BYTES", "9500000"))

MODEL     = "chirp_2"
# chirp_2 在 us-central1，僅支援單語言
LANGUAGES = ["cs-CZ"]


def get_token():
    from google.oauth2 import service_account
    from google.auth.transport.requests import Request as GReq
    creds = service_account.Credentials.from_service_account_file(
        SA_JSON, scopes=["https://www.googleapis.com/auth/cloud-platform"])
    creds.refresh(GReq())
    return creds.token


# 修改後的建議：不進行任何降轉，確保與 Gemini 使用相同原始音質進行公平測試
def prepare_wav(src):
    with wave.open(src, "rb") as wf:
        # 僅讀取長度供 RTFx 計算使用
        duration = wf.getnframes() / wf.getframerate()
    # 直接回傳原始路徑，不產生臨時壓縮檔
    return src, duration


def main():
    print(f"{'='*55}\n Chirp 2 完整功能評估\n{'='*55}")
    print(f"Model: {MODEL} | Region: {REGION} | Project: {PROJECT_ID}")
    print(f"Audio: {AUDIO_FILE}")
    print(f"偵測語言: {', '.join(LANGUAGES)}")

    tmp, duration = prepare_wav(AUDIO_FILE)
    size = os.path.getsize(tmp)
    print(f"原始音檔大小: {size:,} bytes  |  音檔長度: {duration:.2f}s")
    if size > INLINE_LIMIT:
        raise RuntimeError("音檔超限，請換較短音檔")

    with open(tmp, "rb") as f:
        content = base64.b64encode(f.read()).decode()

    payload = {
        "config": {
            "autoDecodingConfig": {},
            "model": MODEL,
            "languageCodes": LANGUAGES,
            "features": {
                "enableWordTimeOffsets":    True,  # 語音切分：逐字時間戳
                "enableWordConfidence":     True,  # 錯字偵測：逐字信心度
                "enableAutomaticPunctuation": True, # 語意連貫：自動標點
                "maxAlternatives":          1,
            },
        },
        "content": content,
    }

    url = (f"https://{REGION}-speech.googleapis.com/v2/projects/{PROJECT_ID}"
           f"/locations/{REGION}/recognizers/_:recognize")
    t0 = time.perf_counter()
    resp = requests.post(url, json=payload,
                         headers={"Authorization": f"Bearer {get_token()}",
                                  "Content-Type": "application/json"},
                         timeout=300)
    elapsed = time.perf_counter() - t0

    print(f"\nHTTP {resp.status_code}  |  推論時間: {elapsed:.2f}s  |  RTF: {elapsed/duration:.4f}  |  RTFx: {duration/elapsed:.2f}x")
    if resp.status_code != 200:
        print(resp.text); return

    results = resp.json().get("results", [])
    if not results:
        print("無辨識結果"); return

    # --- 完整逐字稿 ---
    print("\n--- 完整逐字稿 ---")
    for i, r in enumerate(results, 1):
        alt = (r.get("alternatives") or [{}])[0]
        lang = r.get("languageCode", "?")
        conf = alt.get("confidence", 0)
        print(f"[{i}] 語言={lang}  整體信心度={conf:.3f}")
        print(f"    {alt.get('transcript', '')}")

    # --- 語音切分 (Word Timestamps) + 說話者 ---
    print("\n--- 語音切分 / 說話者分析 ---")
    low_conf = []
    speakers = {}
    for r in results:
        alt = (r.get("alternatives") or [{}])[0]
        for w in alt.get("words", []):
            word   = w.get("word", "")
            start  = float(w.get("startOffset", "0s").rstrip("s"))
            end    = float(w.get("endOffset",   "0s").rstrip("s"))
            wconf  = w.get("confidence", 1.0)
            spk    = w.get("speakerLabel", "")
            spk_s  = f"  [說話者 {spk}]" if spk else ""
            print(f"  {start:>6.2f}s~{end:<6.2f}s  {word:<10} conf={wconf:.3f}{spk_s}")
            if wconf < 0.8:
                low_conf.append((word, wconf, start))
            if spk:
                speakers.setdefault(spk, []).append(word)

    # --- 低信心度詞 (可能錯字) ---
    print("\n--- 低信心度詞 (可能錯字, conf < 0.8) ---")
    if low_conf:
        for w, c, t in sorted(low_conf, key=lambda x: x[1]):
            print(f"  '{w}'  conf={c:.3f}  at {t:.2f}s")
    else:
        print("  無低信心度詞 ")

    # --- 說話者摘要 ---
    if speakers:
        print("\n--- 說話者摘要 ---")
        for spk, words in speakers.items():
            print(f"  說話者 {spk}: {' '.join(words)}")


if __name__ == "__main__":
    main()