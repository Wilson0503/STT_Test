"""Chirp 3 即時語音辨識 (gRPC 串流版)"""

import os
import queue
import sys
import time
import audioop

import pyaudio
from dotenv import load_dotenv
from google.cloud import speech_v2
from google.oauth2 import service_account

load_dotenv()

PROJECT_ID = os.getenv("project_id", "ebgit-did1c-01")
SA_JSON = os.getenv("SERVICE_ACCOUNT_JSON", "")
REGION = "us"

# streaming_recognize 需要至少 1 個語言，最多 3 個
# 先用 us 區域確定支援的語言，避免啟動失敗
LANGUAGES = [lang.strip() for lang in os.getenv("CHIRP_LANGUAGES", "en-US").split(",") if lang.strip()]

RATE = 16000
CHUNK = int(RATE / 10)  # 100ms
AUDIO_DEBUG = os.getenv("AUDIO_DEBUG", "1") == "1"
INPUT_DEVICE_INDEX = int(os.getenv("MIC_DEVICE_INDEX", "-1"))
LIST_MIC_DEVICES = os.getenv("LIST_MIC_DEVICES", "0") == "1"


def list_input_devices():
    pa = pyaudio.PyAudio()
    try:
        print("\n可用麥克風裝置：")
        default_info = pa.get_default_input_device_info()
        default_index = int(default_info.get("index", -1))
        for idx in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(idx)
            if int(info.get("maxInputChannels", 0)) > 0:
                marker = " (default)" if idx == default_index else ""
                print(f"  [{idx}] {info.get('name', 'Unknown')}{marker}")
    finally:
        pa.terminate()


class MicrophoneStream:
    def __init__(self, rate, chunk):
        self._rate = rate
        self._chunk = chunk
        self._buff = queue.Queue()
        self.closed = True
        self._last_level_log = 0.0
        self._chunks_sent = 0

    def __enter__(self):
        self._audio_interface = pyaudio.PyAudio()
        open_kwargs = {
            "format": pyaudio.paInt16,
            "channels": 1,
            "rate": self._rate,
            "input": True,
            "frames_per_buffer": self._chunk,
            "stream_callback": self._fill_buffer,
        }
        if INPUT_DEVICE_INDEX >= 0:
            open_kwargs["input_device_index"] = INPUT_DEVICE_INDEX

        self._audio_stream = self._audio_interface.open(
            **open_kwargs,
        )
        self.closed = False
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        self._buff.put(None)
        self._audio_interface.terminate()

    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        self._buff.put(in_data)
        self._chunks_sent += 1
        if AUDIO_DEBUG:
            now = time.time()
            if now - self._last_level_log >= 1.0:
                # 16-bit PCM (paInt16) 即時音量估計
                rms = audioop.rms(in_data, 2)
                sys.stdout.write(f"\r🎚️ mic level: {rms:5d} | sent_chunks: {self._chunks_sent:5d}   ")
                sys.stdout.flush()
                self._last_level_log = now
        return None, pyaudio.paContinue

    def generator(self):
        while not self.closed:
            chunk = self._buff.get()
            if chunk is None:
                return

            data = [chunk]
            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break

            yield b"".join(data)


def listen_print_loop(responses, start_time):
    last_transcript = ""
    response_count = 0
    for response in responses:
        response_count += 1
        if not response.results:
            if AUDIO_DEBUG and response_count % 10 == 0:
                sys.stdout.write(f"\r📡 server responses: {response_count:5d} (no results yet)   ")
                sys.stdout.flush()
            continue

        result = response.results[0]
        if not result.alternatives:
            continue

        transcript = result.alternatives[0].transcript
        lang_code = result.language_code if result.language_code else "?"
        elapsed = time.time() - start_time

        if result.is_final:
            if transcript and transcript != last_transcript:
                sys.stdout.write(f"\n✅ [{elapsed:06.2f}s] [{lang_code}] {transcript}\n")
                last_transcript = transcript
        else:
            sys.stdout.write(f"\r⏳ [{elapsed:06.2f}s] [{lang_code}] {transcript}")
            sys.stdout.flush()


def request_generator(recognizer_path, streaming_config, audio_generator):
    # 第一包：必須包含 recognizer 和 streaming_config
    yield speech_v2.StreamingRecognizeRequest(
        recognizer=recognizer_path,
        streaming_config=streaming_config,
    )

    # 後續封包：嚴格只送 audio
    for content in audio_generator:
        yield speech_v2.StreamingRecognizeRequest(
            audio=content 
        )

def main():
    if not SA_JSON or not os.path.exists(SA_JSON):
        print("❌ 錯誤: SERVICE_ACCOUNT_JSON 未設定或檔案不存在")
        print(f"   目前值: {SA_JSON}")
        sys.exit(1)

    if LIST_MIC_DEVICES:
        list_input_devices()
        return

    creds = service_account.Credentials.from_service_account_file(SA_JSON)
    client = speech_v2.SpeechClient(
        credentials=creds,
        client_options={"api_endpoint": f"{REGION}-speech.googleapis.com"},
    )

    recognizer_path = f"projects/{PROJECT_ID}/locations/{REGION}/recognizers/_"
    config = speech_v2.RecognitionConfig(
        # 不要用 AutoDetect，直接指定 16bit PCM 16kHz
        explicit_decoding_config=speech_v2.ExplicitDecodingConfig(
            encoding=speech_v2.ExplicitDecodingConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=RATE,
            audio_channel_count=1,
        ),
        model="chirp_3",
        language_codes=LANGUAGES,
        features=speech_v2.RecognitionFeatures(
            enable_automatic_punctuation=True,
        ),
    )
    streaming_config = speech_v2.StreamingRecognitionConfig(
        config=config,
        streaming_features=speech_v2.StreamingRecognitionFeatures(
            interim_results=True,
            # 暫時移除 enable_voice_activity_events 以避免非文字封包干擾
        ),
    )

    print(f"\n{'=' * 60}")
    print("🚀 Chirp 3 即時語音辨識 (gRPC 版)")
    print(f"📍 Project: {PROJECT_ID}")
    print(f"🌐 Region: {REGION}")
    print(f"🗣️  語言: {', '.join(LANGUAGES)}")
    print(f"🎛️  MIC_DEVICE_INDEX: {INPUT_DEVICE_INDEX} (-1 表示系統預設麥克風)")
    print("💡 可用環境變數: CHIRP_LANGUAGES=en-US 或 en-US,zh-CN | MIC_DEVICE_INDEX=裝置索引")
    if AUDIO_DEBUG:
        print("🔎 診斷模式: 顯示 mic level，若數值長期接近 0 代表麥克風沒收音")
    print("🎤 請開始說話 (按 Ctrl+C 結束)...")
    print(f"{'=' * 60}\n")

    start_time = time.time()

    try:
        with MicrophoneStream(RATE, CHUNK) as stream:
            audio_generator = stream.generator()
            requests = request_generator(recognizer_path, streaming_config, audio_generator)
            responses = client.streaming_recognize(requests=requests)
            listen_print_loop(responses, start_time)
    except Exception as e:
        print(f"\n❌ 串流中斷: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n🛑 即時辨識已停止。")
