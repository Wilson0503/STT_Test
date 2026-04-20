#!/usr/bin/env python3
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
    sys.exit(1)


def parse_region_from_endpoint(endpoint: str) -> str:
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


def build_speech_config():
    api_key = get_env("AZURE_SPEECH_KEY") or get_env("AZURE_API_KEY")
    region = get_env("AZURE_SPEECH_REGION")
    endpoint = get_env("AZURE_SPEECH_ENDPOINT") or get_env("AZURE_ENDPOINT")

    if not api_key:
        raise RuntimeError("Missing AZURE_SPEECH_KEY or AZURE_API_KEY in .env")

    if endpoint:
        speech_config = speechsdk.SpeechConfig(subscription=api_key, endpoint=endpoint)
        if not region:
            region = parse_region_from_endpoint(endpoint)
    else:
        if not region:
            raise RuntimeError("Missing AZURE_SPEECH_REGION")
        endpoint = f"https://{region}.stt.speech.microsoft.com/speech/recognition/conversation/cognitiveservices/v1"
        speech_config = speechsdk.SpeechConfig(subscription=api_key, endpoint=endpoint)

    speech_config.output_format = speechsdk.OutputFormat.Detailed

    lang_id_mode = get_env("AZURE_LANGUAGE_ID_MODE", "Continuous")
    speech_config.set_property(
        speechsdk.PropertyId.SpeechServiceConnection_LanguageIdMode,
        lang_id_mode,
    )
    return speech_config, region


def main():
    file_path = Path(__file__).resolve()
    candidate_envs = [
        file_path.parents[2] / ".env",
        file_path.parents[1] / ".env",
        file_path.parent / ".env",
    ]
    for env_path in candidate_envs:
        if env_path.exists():
            load_dotenv(env_path, override=False)

    speech_config, region = build_speech_config()

    languages_raw = get_env("AZURE_AUTO_LANGUAGES", "en-US,zh-TW")
    languages = [x.strip() for x in languages_raw.split(",") if x.strip()]

    auto_detect = speechsdk.languageconfig.AutoDetectSourceLanguageConfig(languages=languages)
    audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)

    recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config,
        audio_config=audio_config,
        auto_detect_source_language_config=auto_detect,
    )

    print("=" * 64)
    print("Azure Speech Realtime STT + Continuous Language Detection")
    print(f"Region: {region or 'from endpoint'}")
    print(f"Candidate languages: {', '.join(languages)}")
    print(f"Language ID mode: {get_env('AZURE_LANGUAGE_ID_MODE', 'Continuous')}")
    print("Mic: default system microphone")
    print("Press Ctrl+C to stop")
    print("=" * 64)

    def get_detected_lang(result) -> str:
        try:
            lang_result = speechsdk.AutoDetectSourceLanguageResult(result)
            return lang_result.language or "?"
        except Exception:
            return "?"

    def on_recognizing(evt):
        if evt.result.reason == speechsdk.ResultReason.RecognizingSpeech:
            text = (evt.result.text or "").strip()
            if text:
                lang = get_detected_lang(evt.result)
                print(f"[~] ({lang}) {text}")

    def on_recognized(evt):
        if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
            text = (evt.result.text or "").strip()
            if text:
                lang = get_detected_lang(evt.result)
                print(f"[✓] ({lang}) {text}")

    def on_session_started(evt):
        print("[✓ Session started - microphone active]")

    def on_session_stopped(evt):
        print("[!] Session stopped")

    def on_canceled(evt):
        print(f"[✗] Canceled: {evt.reason}")
        if evt.reason == speechsdk.CancellationReason.Error:
            print(f"    Error details: {evt.error_details}")

    recognizer.recognizing.connect(on_recognizing)
    recognizer.recognized.connect(on_recognized)
    recognizer.session_started.connect(on_session_started)
    recognizer.session_stopped.connect(on_session_stopped)
    recognizer.canceled.connect(on_canceled)

    print("\n🎤 Listening... (speak now)")
    
    # Use async to allow continuous recognition
    start_result = recognizer.start_continuous_recognition_async().get()
    print(f"[Start result: {start_result}]")
    
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[Stopping...]")
    finally:
        recognizer.stop_continuous_recognition_async().get()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Fatal: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
