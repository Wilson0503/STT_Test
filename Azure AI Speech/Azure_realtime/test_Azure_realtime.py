import os
import sys
import time
import threading
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv

try:
    import azure.cognitiveservices.speech as speechsdk
except ImportError:
    print("Missing dependency: azure-cognitiveservices-speech")
    print("Install with: pip install azure-cognitiveservices-speech")
    sys.exit(1)

try:
    import pyaudio
except ImportError:
    pyaudio = None


def parse_region_from_endpoint(endpoint: str) -> str:
    """Best-effort parse: https://eastus.api.cognitive.microsoft.com -> eastus"""
    try:
        host = urlparse(endpoint).hostname or ""
        if not host:
            return ""
        return host.split(".")[0]
    except Exception:
        return ""


def get_env(name: str, default: str = "") -> str:
    value = os.getenv(name, default)
    return value.strip() if isinstance(value, str) else value


def is_placeholder_key(value: str) -> bool:
    return (value or "").strip().lower() in {
        "",
        "your-speech-key",
        "your-subscription-key",
        "<your-key>",
    }


def build_speech_config():
    # Compatible with your existing naming and Speech SDK naming.
    speech_key = get_env("AZURE_SPEECH_KEY")
    api_key = get_env("AZURE_API_KEY")
    if is_placeholder_key(speech_key):
        effective_key = api_key
    else:
        effective_key = speech_key or api_key

    region = get_env("AZURE_SPEECH_REGION")
    endpoint = get_env("AZURE_SPEECH_ENDPOINT") or get_env("AZURE_ENDPOINT")

    if not effective_key:
        raise RuntimeError("Missing AZURE_SPEECH_KEY or AZURE_API_KEY in .env")

    # Prefer region when available. Some generic endpoints are valid for REST calls
    # but can cause realtime session to stop immediately in SDK transcriber flow.
    if region:
        speech_config = speechsdk.SpeechConfig(subscription=effective_key, region=region)
    elif endpoint:
        # Speech SDK supports endpoint + key for custom domain endpoints.
        speech_config = speechsdk.SpeechConfig(subscription=effective_key, endpoint=endpoint)
        region = parse_region_from_endpoint(endpoint)
    else:
        raise RuntimeError("Missing AZURE_SPEECH_REGION (or provide AZURE_SPEECH_ENDPOINT/AZURE_ENDPOINT)")

    speech_config.output_format = speechsdk.OutputFormat.Detailed

    # 對中英夾雜建議使用 Continuous 語言識別模式
    # 可用值: "AtStart" | "Continuous"
    lang_id_mode = get_env("AZURE_LANGUAGE_ID_MODE", "Continuous")
    speech_config.set_property(
        speechsdk.PropertyId.SpeechServiceConnection_LanguageIdMode,
        lang_id_mode,
    )
    return speech_config, region


def main():
    # Load .env from likely locations (workspace root, Azure folder, local folder).
    file_path = Path(__file__).resolve()
    candidate_envs = [
        file_path.parents[2] / ".env",  # workspace root: STT_Test/.env
        file_path.parents[1] / ".env",  # Azure AI Speech/.env
        file_path.parent / ".env",      # Azure_realtime/.env
    ]
    for env_path in candidate_envs:
        if env_path.exists():
            # Let closer .env files override root placeholders.
            load_dotenv(env_path, override=True)

    speech_config, region = build_speech_config()

    languages_raw = get_env("AZURE_AUTO_LANGUAGES", "en-US,zh-TW")
    languages = [x.strip() for x in languages_raw.split(",") if x.strip()]
    if not languages:
        raise RuntimeError("AZURE_AUTO_LANGUAGES cannot be empty")

    # Azure realtime auto-detect requires candidate language list (cannot detect from all languages globally).
    auto_detect = speechsdk.languageconfig.AutoDetectSourceLanguageConfig(languages=languages)
    mic_device_name = get_env("AZURE_MIC_DEVICE_NAME", "")
    use_pyaudio_stream = get_env("AZURE_USE_PYAUDIO_STREAM", "1").lower() not in {"0", "false", "no"}

    pa = None
    pa_stream = None
    push_stream = None
    pump_stop = threading.Event()
    pump_thread = None

    if use_pyaudio_stream and pyaudio is not None:
        # Use PyAudio as mic source, then push PCM chunks to Speech SDK stream.
        pa = pyaudio.PyAudio()
        stream_format = speechsdk.audio.AudioStreamFormat(samples_per_second=16000, bits_per_sample=16, channels=1)
        push_stream = speechsdk.audio.PushAudioInputStream(stream_format=stream_format)
        audio_config = speechsdk.audio.AudioConfig(stream=push_stream)
        pa_stream = pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=1600,
        )

        def _pump_mic_audio():
            read_errors = 0
            while not pump_stop.is_set():
                try:
                    chunk = pa_stream.read(1600, exception_on_overflow=False)
                    push_stream.write(chunk)
                    read_errors = 0
                except Exception as exc:
                    read_errors += 1
                    if read_errors == 1:
                        print(f"[MicReadError] {exc}")
                    if read_errors >= 10:
                        print("[MicReadError] too many consecutive failures, stopping mic pump")
                        break
                    time.sleep(0.05)

        pump_thread = threading.Thread(target=_pump_mic_audio, daemon=True)
    elif mic_device_name:
        audio_config = speechsdk.audio.AudioConfig(device_name=mic_device_name)
    else:
        audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)

    recognizer_type = get_env("AZURE_RECOGNIZER_TYPE", "speech").lower()
    if recognizer_type == "conversation":
        # ConversationTranscriber supports speaker_id, but can be less stable on some setups.
        recognizer = speechsdk.transcription.ConversationTranscriber(
            speech_config=speech_config,
            audio_config=audio_config,
            auto_detect_source_language_config=auto_detect,
        )
        is_conversation = True
    else:
        # SpeechRecognizer is more stable for realtime mic transcription.
        recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_config,
            audio_config=audio_config,
            auto_detect_source_language_config=auto_detect,
        )
        is_conversation = False

    # 可選: Phrase List 強化專有名詞/術語辨識（逗號分隔）
    # 例: AZURE_PHRASE_LIST=Instella,Inventec,interesting
    phrase_list_raw = get_env("AZURE_PHRASE_LIST", "")
    phrase_items = [x.strip() for x in phrase_list_raw.split(",") if x.strip()]
    if phrase_items:
        phrase_grammar = speechsdk.PhraseListGrammar.from_recognizer(recognizer)
        for phrase in phrase_items:
            phrase_grammar.addPhrase(phrase)

    print("=" * 64)
    print("Azure Speech Realtime STT + Auto Language")
    print(f"Region: {region or 'from endpoint'}")
    print(f"Candidate languages: {', '.join(languages)}")
    print(f"Language ID mode: {get_env('AZURE_LANGUAGE_ID_MODE', 'Continuous')}")
    print(f"Recognizer: {'ConversationTranscriber' if is_conversation else 'SpeechRecognizer'}")
    print(f"Phrase list: {', '.join(phrase_items) if phrase_items else '(none)'}")
    if use_pyaudio_stream and pyaudio is not None:
        print("Mic: PyAudio stream (16k PCM mono)")
    elif mic_device_name:
        print(f"Mic: explicit device ({mic_device_name})")
    else:
        print("Mic: default system microphone")
    print("Press Ctrl+C to stop")
    print("=" * 64)

    last_partial = ""
    started_at = [0.0]

    def get_detected_lang(result) -> str:
        try:
            lang_result = speechsdk.AutoDetectSourceLanguageResult(result)
            return lang_result.language or "?"
        except Exception:
            return "?"

    def on_recognizing(evt):
        nonlocal last_partial
        if evt.result.reason == speechsdk.ResultReason.RecognizingSpeech:
            text = (evt.result.text or "").strip()
            if text and text != last_partial:
                lang = get_detected_lang(evt.result)
                elapsed = time.strftime("%H:%M:%S")
                if is_conversation:
                    speaker = getattr(evt.result, "speaker_id", "?") or "?"
                    print(f"[~ {elapsed}] [spk:{speaker}] ({lang}) {text}")
                else:
                    print(f"[~ {elapsed}] ({lang}) {text}")
                last_partial = text

    def on_recognized(evt):
        if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
            text = (evt.result.text or "").strip()
            if text:
                lang = get_detected_lang(evt.result)
                elapsed = time.strftime("%H:%M:%S")
                if is_conversation:
                    speaker = getattr(evt.result, "speaker_id", "?") or "?"
                    print(f"[OK {elapsed}] [spk:{speaker}] ({lang}) {text}")
                else:
                    print(f"[OK {elapsed}] ({lang}) {text}")
        elif evt.result.reason == speechsdk.ResultReason.NoMatch:
            print("[NoMatch] speech not recognized")

    def on_canceled(evt):
        print(f"[Canceled] reason={evt.reason}")
        if evt.reason == speechsdk.CancellationReason.Error:
            print(f"ErrorCode={evt.error_code}")
            print(f"ErrorDetails={evt.error_details}")

    def on_session_started(evt):
        started_at[0] = time.time()
        print("[Session started]")

    def on_session_stopped(evt):
        print("[Session stopped]")
        if started_at[0] > 0 and (time.time() - started_at[0]) < 2.0:
            print("[Hint] Session ended too quickly. Usually this means default microphone is unavailable, busy, or blocked by OS privacy settings.")
            print("[Hint] Try setting AZURE_MIC_DEVICE_NAME, or switch Windows default input to your active microphone.")

    if is_conversation:
        recognizer.transcribing.connect(on_recognizing)
        recognizer.transcribed.connect(on_recognized)
    else:
        recognizer.recognizing.connect(on_recognizing)
        recognizer.recognized.connect(on_recognized)
    recognizer.session_started.connect(on_session_started)
    recognizer.session_stopped.connect(on_session_stopped)
    recognizer.canceled.connect(on_canceled)

    if pump_thread is not None:
        pump_thread.start()

    if is_conversation:
        recognizer.start_transcribing_async().get()
    else:
        recognizer.start_continuous_recognition_async().get()
    try:
        while True:
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        pump_stop.set()
        if is_conversation:
            recognizer.stop_transcribing_async().get()
        else:
            recognizer.stop_continuous_recognition_async().get()
        if push_stream is not None:
            try:
                push_stream.close()
            except Exception:
                pass
        if pa_stream is not None:
            try:
                pa_stream.stop_stream()
                pa_stream.close()
            except Exception:
                pass
        if pa is not None:
            try:
                pa.terminate()
            except Exception:
                pass


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Fatal: {exc}")
        sys.exit(1)
