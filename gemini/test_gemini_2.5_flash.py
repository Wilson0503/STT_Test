import os
import shutil
import tempfile
import time
from google import genai  # 官方建議使用 google-genai
from dotenv import load_dotenv

# 1. 載入 .env
load_dotenv()

# ==========================================
# 配置區域
# ==========================================
API_KEY = os.getenv("API_KEY")
MEDIA_FILE_PATH = os.getenv("GEMINI_MEDIA_FILE", r"C:\Users\iec150094\STT_Test\instella實戰攻略_27-85s.wav")
MODEL_NAME = "gemini-2.5-flash"  # 2.5 Flash 版本
SUPPORTED_MEDIA_EXTENSIONS = {".wav", ".mp3", ".m4a", ".mp4"}
# ==========================================

def test_gemini_2_5_flash_audio():
    if not API_KEY:
        print("❌ 錯誤: 找不到 API_KEY")
        return
    client = genai.Client(api_key=API_KEY)
    print(f"🚀 啟動 Gemini 2.5 Flash 純逐字稿測試: {MODEL_NAME}")
    if not os.path.exists(MEDIA_FILE_PATH):
        print(f"❌ 錯誤: 找不到檔案 {MEDIA_FILE_PATH}")
        return
    extension = os.path.splitext(MEDIA_FILE_PATH)[1].lower()
    if extension not in SUPPORTED_MEDIA_EXTENSIONS:
        print(f"❌ 錯誤: 不支援副檔名 {extension}，請使用 {', '.join(sorted(SUPPORTED_MEDIA_EXTENSIONS))}")
        return
    try:
        print(f"Step 1: 正在上傳檔案 ({extension})...")
        upload_path = MEDIA_FILE_PATH
        temp_upload_path = None
        try:
            os.path.basename(MEDIA_FILE_PATH).encode("ascii")
        except UnicodeEncodeError:
            fd, temp_upload_path = tempfile.mkstemp(prefix="gemini_media_", suffix=extension)
            os.close(fd)
            shutil.copyfile(MEDIA_FILE_PATH, temp_upload_path)
            upload_path = temp_upload_path
        audio_file = client.files.upload(file=upload_path)
        print(f"Step 2: 模型正在分析內容", end="")
        while audio_file.state.name == "PROCESSING":
            print(".", end="", flush=True)
            time.sleep(3)
            audio_file = client.files.get(name=audio_file.name)
        print("\n✅ 音檔解析完成。")
        prompt = """
請對演講者Hayley在inventec公司內部對於instella實戰攻略的音檔進行完整的語音轉文字分析，並依照以下格式輸出，不要省略任何段落。

## 1. 完整逐字稿（含時間戳與說話者）
格式：[MM:SS] [說話者A/B/C...] 逐字內容
- 每隔 10~15 秒換一個新行
- 若只有一位說話者，省略說話者標籤
- 自動偵測語言，若有中英混合請保留原語言，不要翻譯

## 2. 低信心度詞彙（可能錯字）
列出你判斷辨識信心較低的詞彙，格式：
- [時間戳] 「詞彙」→ 可能正確寫法（若有的話）

## 3. 說話者摘要
每位說話者各一行，格式：
- 說話者A：主要內容摘要（一句話）

## 4. 整體品質評估
- 語音清晰度：X/10
- 語意連貫度：X/10
- 背景噪音干擾：低/中/高
- 偵測到的語言：列出所有語言

## 5. 翻譯
-不是中文的語言都幫我在最下面新增一段翻譯成中文的完整逐字稿
"""
        import wave
        try:
            with wave.open(MEDIA_FILE_PATH, "rb") as wf:
                audio_duration = wf.getnframes() / wf.getframerate()
        except Exception:
            audio_duration = None
        print(f"Step 3: 正在產生逐字稿...")
        start_time = time.time()
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[prompt, audio_file]
        )
        elapsed = time.time() - start_time
        rtfx_str = f"  RTFx: {audio_duration/elapsed:.2f}x" if audio_duration else ""
        print(f"\n{'='*50}\n 語音轉文字結果 (耗時: {elapsed:.2f}s{rtfx_str})\n{'='*50}")
        print(response.text)
    except Exception as e:
        print(f"\n❌ 程式發生錯誤: {e}")
    finally:
        if 'audio_file' in locals():
            client.files.delete(name=audio_file.name)
            print(f"\n(已清理臨時檔案)")

if __name__ == "__main__":
    test_gemini_2_5_flash_audio()
