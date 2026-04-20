#!/usr/bin/env python3
import os
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv

try:
    import azure.cognitiveservices.speech as speechsdk
except ImportError:
    print("Missing dependency: azure-cognitiveservices-speech")
    sys.exit(1)


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


def build_speech_config():
    api_key = get_env("AZURE_SPEECH_KEY") or get_env("AZURE_API_KEY")
    region = get_env("AZURE_SPEECH_REGION")
    endpoint = get_env("AZURE_SPEECH_ENDPOINT") or get_env("AZURE_ENDPOINT")

    if not api_key:
        raise RuntimeError("Missing AZURE_SPEECH_KEY or AZURE_API_KEY in .env")

    # For continuous language identification, must use endpoint-based config
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

    # Continuous language identification
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

    # Find an audio file to test with
    test_dir = Path(".")
    audio_files = list(test_dir.glob("**/*.wav"))
    
    if not audio_files:
        print("❌ No .wav files found in current directory!")
        print(f"Current directory: {test_dir.resolve()}")
        return

    audio_file = audio_files[0]
    print(f"Testing with audio file: {audio_file.name}")

    auto_detect = speechsdk.languageconfig.AutoDetectSourceLanguageConfig(languages=languages)
    audio_config = speechsdk.audio.AudioConfig(filename=str(audio_file))

    recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config,
        audio_config=audio_config,
        auto_detect_source_language_config=auto_detect,
    )

    print("=" * 64)
    print("Azure Speech Continuous Language Detection (File Test)")
    print(f"Region: {region}")
    print(f"Audio file: {audio_file.name}")
    print(f"Candidate languages: {', '.join(languages)}")
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
                print(f"[~ {lang}] {text}")

    def on_recognized(evt):
        if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
            text = (evt.result.text or "").strip()
            if text:
                lang = get_detected_lang(evt.result)
                print(f"[✓ {lang}] {text}")
        elif evt.result.reason == speechsdk.ResultReason.NoMatch:
            print("[✗] No speech recognized")

    def on_canceled(evt):
        if evt.reason == speechsdk.CancellationReason.Error:
            print(f"[Error] {evt.error_details}")

    recognizer.recognizing.connect(on_recognizing)
    recognizer.recognized.connect(on_recognized)
    recognizer.canceled.connect(on_canceled)

    done = False
    
    def on_session_stopped(evt):
        nonlocal done
        done = True
    
    recognizer.session_stopped.connect(on_session_stopped)

    print("\nProcessing audio...")
    recognizer.start_continuous_recognition()
    
    while not done:
        time.sleep(0.5)
    
    recognizer.stop_continuous_recognition()
    print("\n[Done]")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Fatal: {exc}")
        sys.exit(1)
