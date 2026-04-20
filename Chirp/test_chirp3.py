try:
    import audioop
except ModuleNotFoundError:
    import audioop_lts as audioop  # Python 3.13+
import base64, os, tempfile, time, wave
import requests
from dotenv import load_dotenv
from requests.exceptions import RequestException

load_dotenv()

PROJECT_ID   = os.getenv("project_id", "ebgit-did1c-01")
REGIONS      = ["us", "eu", "asia-southeast1"]
SA_JSON      = os.getenv("SERVICE_ACCOUNT_JSON", "")
AUDIO_FILE   = os.getenv("audio_file_path", r"C:\Users\iec150094\STT_Test\Agentic AI 數位員工_14s-73s.wav")
INLINE_LIMIT = int(os.getenv("INLINE_LIMIT_BYTES", "9500000"))

MODEL     = "chirp_3"
# chirp_3 支援 global region，多語言最多 3 個
LANGUAGES = ["cs-CZ", "en-US"]


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

def post_with_retry(url, payload, headers, retries=3):
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            return requests.post(url, json=payload, headers=headers, timeout=300)
        except RequestException as error:
            last_error = error
            print(f"連線失敗 (attempt {attempt}/{retries}): {error}")
            time.sleep(1.5 * attempt)
    raise RuntimeError(f"連線重試後仍失敗：{last_error}")


def build_url(project_id: str, region: str) -> str:
    host = "speech.googleapis.com" if region == "global" else f"{region}-speech.googleapis.com"
    return f"https://{host}/v2/projects/{project_id}/locations/{region}/recognizers/_:recognize"


def main():
    print(f"{'='*55}\n Chirp 3 完整功能評估\n{'='*55}")
    print(f"Model: {MODEL} | Regions: {', '.join(REGIONS)} | Project: {PROJECT_ID}")
    print(f"Audio: {AUDIO_FILE}")
    print(f"偵測語言 ({len(LANGUAGES)}種): {', '.join(LANGUAGES)}")

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
                "enableWordTimeOffsets":      True,  # 語音切分：逐字時間戳
                "enableAutomaticPunctuation": True,  # 語意連貫：自動標點
                "maxAlternatives":            1,
                #"diarizationConfig": {
                #    "minSpeakerCount": 1,
                #    "maxSpeakerCount": 4,
                #},
            },
        },
        "content": content,
    }

    headers = {
        "Authorization": f"Bearer {get_token()}",
        "Content-Type": "application/json",
    }

    resp = None
    elapsed = 0.0
    used_region = None
    for region in REGIONS:
        url = build_url(PROJECT_ID, region)
        print(f"\n嘗試 Region: {region}")
        t0 = time.perf_counter()
        try:
            resp = post_with_retry(url, payload, headers, retries=3)
        except RuntimeError as error:
            print(error)
            continue
        elapsed = time.perf_counter() - t0
        used_region = region
        if resp.status_code == 400 and "does not exist in the location" in resp.text:
            print(f"Region {region} 不支援 {MODEL}，切換下一個 region...")
            continue
        break

    if resp is None:
        raise RuntimeError("所有 region 嘗試都失敗，請檢查網路、防火牆或公司 Proxy 設定。")

    print(f"\n使用 Region: {used_region}")
    print(f"HTTP {resp.status_code}  |  推論時間: {elapsed:.2f}s  |  RTF: {elapsed/duration:.4f}  |  RTFx: {duration/elapsed:.2f}x")
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
        print(f"[{i}] 語言={lang}")
        print(f"    {alt.get('transcript', '')}")

    # --- 語音切分 (Word Timestamps) + 說話者 ---
    print("\n--- 語音切分 / 說話者分析 ---")
    speakers = {}
    for r in results:
        alt = (r.get("alternatives") or [{}])[0]
        for w in alt.get("words", []):
            word   = w.get("word", "")
            start  = float(w.get("startOffset", "0s").rstrip("s"))
            end    = float(w.get("endOffset",   "0s").rstrip("s"))
            spk    = w.get("speakerLabel", "")
            spk_s  = f"  [說話者 {spk}]" if spk else ""
            print(f"  {start:>6.2f}s~{end:<6.2f}s  {word:<10}{spk_s}")
            if spk:
                speakers.setdefault(spk, []).append(word)

    # --- 說話者摘要 ---
    if speakers:
        print("\n--- 說話者摘要 ---")
        for spk, words in speakers.items():
            print(f"  說話者 {spk}: {' '.join(words)}")


if __name__ == "__main__":
    main()