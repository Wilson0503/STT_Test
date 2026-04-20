import contextlib
import mimetypes
import os
import shutil
import tempfile
import time
import wave
from pathlib import Path

import requests
from dotenv import load_dotenv


SCRIPT_DIR = Path(__file__).resolve().parent
load_dotenv(SCRIPT_DIR / ".env")
load_dotenv()


AZURE_OPENAI_ENDPOINT_RAW = os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("AZURE_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("AZURE_API_KEY")
AZURE_OPENAI_DEPLOYMENT = (
    os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    or os.getenv("GPT4O_TRANSCRIBE_DEPLOYMENT")
    or "gpt-4o-transcribe"
)
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2025-03-01-preview")
MEDIA_FILE_PATH = Path(
    os.getenv("GPT4O_MEDIA_FILE", r"C:\Users\iec150094\STT_Test\instella實戰攻略_27-85s.wav")
)
LANGUAGE = os.getenv("GPT4O_LANGUAGE", "zh-TW")
PROMPT = os.getenv(
    "GPT4O_TRANSCRIBE_PROMPT",
    #"Inventec, Instella, Hayley, 數智開發",
)
SUPPORTED_MEDIA_EXTENSIONS = {".wav", ".mp3", ".m4a", ".mp4", ".mpeg", ".mpga", ".webm"}


def validate_configuration():
    if not AZURE_OPENAI_API_KEY:
        raise RuntimeError("找不到 API Key。請在 .env 設定 AZURE_OPENAI_API_KEY，或至少提供 AZURE_API_KEY。")

    if not AZURE_OPENAI_ENDPOINT_RAW:
        raise RuntimeError(
            "找不到 Azure OpenAI endpoint。請在 .env 設定 AZURE_OPENAI_ENDPOINT，例如 https://<resource>.openai.azure.com/"
        )

    if "openai.azure.com" not in AZURE_OPENAI_ENDPOINT_RAW:
        raise RuntimeError(
            "目前的 endpoint 看起來不是 Azure OpenAI 端點。"
            " GPT-4o-transcribe 需要類似 https://<resource>.openai.azure.com/ 的 endpoint，"
            "不能直接使用 Speech Service 的 cognitive.microsoft.com endpoint。"
        )

    if not MEDIA_FILE_PATH.exists():
        raise RuntimeError(f"找不到音檔: {MEDIA_FILE_PATH}")

    if MEDIA_FILE_PATH.suffix.lower() not in SUPPORTED_MEDIA_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_MEDIA_EXTENSIONS))
        raise RuntimeError(f"不支援副檔名 {MEDIA_FILE_PATH.suffix}，請使用 {supported}")


def get_audio_duration_seconds(file_path: Path):
    if file_path.suffix.lower() != ".wav":
        return None

    try:
        with contextlib.closing(wave.open(str(file_path), "rb")) as wav_file:
            frames = wav_file.getnframes()
            frame_rate = wav_file.getframerate()
            if frame_rate:
                return frames / frame_rate
    except Exception:
        return None

    return None


def ensure_ascii_upload_name(file_path: Path):
    temp_path = None
    upload_path = file_path

    try:
        file_path.name.encode("ascii")
    except UnicodeEncodeError:
        fd, tmp_name = tempfile.mkstemp(prefix="gpt4o_audio_", suffix=file_path.suffix)
        os.close(fd)
        temp_path = Path(tmp_name)
        shutil.copyfile(file_path, temp_path)
        upload_path = temp_path

    return upload_path, temp_path


def build_transcribe_url():
    endpoint = AZURE_OPENAI_ENDPOINT_RAW.rstrip("/")
    
    # Check if endpoint already contains the full path (includes /openai/deployments or /audio/transcriptions)
    if "/openai/deployments/" in endpoint or "/audio/transcriptions" in endpoint:
        # Full URL already provided, just ensure api-version is set
        if "api-version=" not in endpoint:
            separator = "?" if "?" not in endpoint else "&"
            return f"{endpoint}{separator}api-version={AZURE_OPENAI_API_VERSION}"
        return endpoint
    else:
        # Base endpoint provided, build full URL
        return (
            f"{endpoint}/openai/deployments/{AZURE_OPENAI_DEPLOYMENT}/audio/transcriptions"
            f"?api-version={AZURE_OPENAI_API_VERSION}"
        )


def extract_segments(payload):
    segments = payload.get("segments") or payload.get("chunks") or []
    normalized_segments = []

    for segment in segments:
        start = segment.get("start")
        end = segment.get("end")
        text = segment.get("text", "").strip()
        if text:
            normalized_segments.append((start, end, text))

    return normalized_segments


def format_timestamp(seconds):
    if seconds is None:
        return "--:--"

    total_seconds = max(0, int(seconds))
    minutes = total_seconds // 60
    remaining_seconds = total_seconds % 60
    return f"{minutes:02d}:{remaining_seconds:02d}"


def test_gpt4o_transcribe():
    validate_configuration()

    print(f"🚀 啟動 GPT-4o-transcribe 純逐字稿測試: {AZURE_OPENAI_DEPLOYMENT}")
    print(f"Step 1: 準備音檔 ({MEDIA_FILE_PATH.suffix.lower()})...")

    upload_path, temp_upload_path = ensure_ascii_upload_name(MEDIA_FILE_PATH)
    audio_duration = get_audio_duration_seconds(MEDIA_FILE_PATH)
    mime_type = mimetypes.guess_type(upload_path.name)[0] or "application/octet-stream"

    url = build_transcribe_url()
    headers = {"api-key": AZURE_OPENAI_API_KEY}
    data = {
        "language": LANGUAGE,
        "prompt": PROMPT,
        "response_format": "json",
    }

    try:
        print("Step 2: 正在上傳檔案並進行轉錄...")
        start_time = time.time()

        with open(upload_path, "rb") as audio_file:
            files = {"file": (upload_path.name, audio_file, mime_type)}
            response = requests.post(url, headers=headers, data=data, files=files, timeout=300)

        elapsed = time.time() - start_time

        if not response.ok:
            print(f"❌ API 呼叫失敗: HTTP {response.status_code}")
            print(response.text)
            return

        try:
            payload = response.json()
        except ValueError:
            payload = {"text": response.text}

        transcript = payload.get("text", "").strip()
        segments = extract_segments(payload)
        rtfx_str = f"  RTFx: {audio_duration / elapsed:.2f}x" if audio_duration else ""

        print(f"\n{'=' * 50}\n 語音轉文字結果 (耗時: {elapsed:.2f}s{rtfx_str})\n{'=' * 50}")
        print("## 1. 完整逐字稿（含時間戳與說話者）")

        if segments:
            for start, _, text in segments:
                print(f"[{format_timestamp(start)}] {text}")
        elif transcript:
            print(transcript)
        else:
            print("(未取得逐字稿內容)")

        if transcript:
            print("\n## 2. 純文字逐字稿")
            print(transcript)

        if payload.get("language"):
            print("\n## 3. 偵測資訊")
            print(f"偵測語言: {payload['language']}")

        if not segments:
            print("\n(此回應未提供 segment 時間戳；GPT-4o-transcribe 在目前設定下僅回傳全文。)")

    finally:
        if temp_upload_path and temp_upload_path.exists():
            try:
                temp_upload_path.unlink()
            except OSError:
                pass


if __name__ == "__main__":
    try:
        test_gpt4o_transcribe()
    except Exception as exc:
        print(f"❌ 程式發生錯誤: {exc}")