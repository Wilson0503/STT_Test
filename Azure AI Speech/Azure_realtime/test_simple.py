#!/usr/bin/env python3
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

try:
    import azure.cognitiveservices.speech as speechsdk
except ImportError:
    print("Missing dependency: azure-cognitiveservices-speech")
    sys.exit(1)


def get_env(name: str, default: str = "") -> str:
    value = os.getenv(name, default)
    return value.strip() if isinstance(value, str) else value


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

    api_key = get_env("AZURE_SPEECH_KEY") or get_env("AZURE_API_KEY")
    region = get_env("AZURE_SPEECH_REGION", "eastus")

    if not api_key:
        raise RuntimeError("Missing AZURE_SPEECH_KEY")

    speech_config = speechsdk.SpeechConfig(subscription=api_key, region=region)
    speech_config.speech_recognition_language = "en-US"

    audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)

    recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config,
        audio_config=audio_config,
    )

    print("=" * 64)
    print("Azure Speech Realtime STT (SIMPLE - no continuous language ID)")
    print(f"Region: {region}")
    print(f"Language: en-US (fixed)")
    print("Mic: default system microphone")
    print("Press Ctrl+C to stop")
    print("=" * 64)

    def on_recognizing(evt):
        if evt.result.reason == speechsdk.ResultReason.RecognizingSpeech:
            text = (evt.result.text or "").strip()
            if text:
                print(f"[~] {text}")

    def on_recognized(evt):
        if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
            text = (evt.result.text or "").strip()
            if text:
                print(f"[✓] {text}")

    def on_session_started(evt):
        print("[✓ Session started - microphone active]")

    def on_session_stopped(evt):
        print("[!] Session stopped")

    def on_canceled(evt):
        print(f"[✗] Canceled: {evt.reason}")
        if evt.reason == speechsdk.CancellationReason.Error:
            print(f"    Error: {evt.error_details}")

    recognizer.recognizing.connect(on_recognizing)
    recognizer.recognized.connect(on_recognized)
    recognizer.session_started.connect(on_session_started)
    recognizer.session_stopped.connect(on_session_stopped)
    recognizer.canceled.connect(on_canceled)

    print("\n🎤 Listening... (speak now)")
    recognizer.start_continuous_recognition_async().get()
    
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
