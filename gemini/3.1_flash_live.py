import os
import asyncio
from pathlib import Path
from contextlib import suppress

import pyaudio
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).with_name(".env"))

# ==========================================
# 配置區域
# ==========================================
API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("API_KEY")
MODEL_NAME = "gemini-3.1-flash-live-preview"
DEFAULT_LIVE_PROMPT = (
    "Only perform realtime speech transcription to text. "
    "Do not answer the user. Do not summarize. Do not translate. "
    "Do not add commentary. Keep recognized English technical terms in original spelling."
)
LIVE_PROMPT = os.getenv("GEMINI_LIVE_PROMPT", DEFAULT_LIVE_PROMPT).strip()
SHOW_USAGE_TOKENS = os.getenv("GEMINI_SHOW_TOKENS", "0") == "1"
MAX_RETRIES = int(os.getenv("GEMINI_MAX_RETRIES", "5"))
RETRY_DELAY_SEC = float(os.getenv("GEMINI_RETRY_DELAY_SEC", "2.0"))
SILENCE_DURATION_MS = int(os.getenv("GEMINI_SILENCE_DURATION_MS", "800"))

# 音訊設定 (Gemini Live API 建議規格: 16-bit PCM, 16kHz, little-endian)
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 512
# ==========================================


def is_transient_live_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "keepalive ping timeout" in msg or "sent 1011" in msg

async def main():
    if not API_KEY:
        print("❌ 錯誤: 找不到 API_KEY，請確認 .env 設定")
        return

    client = genai.Client(api_key=API_KEY)

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                    input=True, frames_per_buffer=CHUNK)

    print(f"🚀 啟動 {MODEL_NAME} 即時語音辨識...")
    if LIVE_PROMPT:
        print(f"📝 Prompt: {LIVE_PROMPT}")
    print(f"⏱️  Silence split: {SILENCE_DURATION_MS} ms")
    print("🎙️  開始錄音，按 Ctrl+C 停止...\n")

    config = {
        "response_modalities": ["AUDIO"],
        "input_audio_transcription": {},
        "realtime_input_config": {
            "automatic_activity_detection": {
                "disabled": False,
                "start_of_speech_sensitivity": "START_SENSITIVITY_HIGH",
                "end_of_speech_sensitivity": "END_SENSITIVITY_HIGH",
                "prefix_padding_ms": 100,
                "silence_duration_ms": SILENCE_DURATION_MS,
            },
            "activity_handling": "START_OF_ACTIVITY_INTERRUPTS",
            "turn_coverage": "TURN_INCLUDES_ONLY_ACTIVITY",
        },
    }

    if LIVE_PROMPT:
        config["system_instruction"] = LIVE_PROMPT

    retry_count = 0
    try:
        while True:
            try:
                async with client.aio.live.connect(model=MODEL_NAME, config=config) as session:
                    if retry_count > 0:
                        print("✅ 已重新連線，繼續轉錄...", flush=True)

                    # 定義傳送音訊的任務
                    async def send_audio():
                        while True:
                            # stream.read 是阻塞 I/O，改用 to_thread 避免卡住 keepalive
                            data = await asyncio.to_thread(
                                stream.read,
                                CHUNK,
                                exception_on_overflow=False,
                            )
                            await session.send_realtime_input(
                                audio=types.Blob(data=data, mime_type="audio/pcm;rate=16000")
                            )
                            await asyncio.sleep(0)

                    # 定義接收回傳文字的任務
                    async def receive_text():
                        last_text = ""
                        async for response in session.receive():
                            content = response.server_content
                            if content and content.input_transcription and content.input_transcription.text:
                                text = content.input_transcription.text.strip()
                                if text and text != last_text:
                                    print(f"[你說]: {text}", flush=True)
                                    last_text = text

                            if SHOW_USAGE_TOKENS and response.usage_metadata and response.usage_metadata.total_token_count:
                                print(f"[Tokens]: {response.usage_metadata.total_token_count}", flush=True)

                    send_task = asyncio.create_task(send_audio())
                    recv_task = asyncio.create_task(receive_text())
                    done, pending = await asyncio.wait(
                        {send_task, recv_task},
                        return_when=asyncio.FIRST_EXCEPTION,
                    )

                    for task in pending:
                        task.cancel()
                    for task in pending:
                        with suppress(asyncio.CancelledError):
                            await task

                    # 把例外往外拋給重連邏輯
                    for task in done:
                        exc = task.exception()
                        if exc:
                            raise exc

                    # 正常結束（理論上不會到這裡）
                    break

            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if is_transient_live_error(exc) and retry_count < MAX_RETRIES:
                    retry_count += 1
                    print(
                        f"⚠️ 連線暫時中斷（{exc}），{RETRY_DELAY_SEC:.1f}s 後重試 "
                        f"({retry_count}/{MAX_RETRIES})...",
                        flush=True,
                    )
                    await asyncio.sleep(RETRY_DELAY_SEC)
                    continue
                raise
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 即時辨識已停止。")
    except Exception as exc:
        print(f"\n❌ Live API 連線失敗: {exc}")
        print("可能原因:")
        print("1. 模型名稱不是 Live API 支援模型")
        print("2. API key 沒有 Gemini Live API 權限")
        print("3. .env 不在目前腳本目錄，或 GEMINI_API_KEY/API_KEY 未設定")
        print("4. Native audio Live 模型不能把 response_modalities 設成 TEXT，必須是 AUDIO")
        print("5. 網路短暫中斷或事件迴圈被阻塞，導致 keepalive timeout")