#!/usr/bin/env python3
"""Test microphone availability and detect audio levels"""
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

try:
    import azure.cognitiveservices.speech as speechsdk
except ImportError:
    print("Missing: azure-cognitiveservices-speech")
    sys.exit(1)

try:
    import pyaudio
    import numpy as np
    HAS_PYAUDIO = True
except ImportError:
    HAS_PYAUDIO = False


def get_env(name: str, default: str = "") -> str:
    value = os.getenv(name, default)
    return value.strip() if isinstance(value, str) else value


def test_pyaudio():
    """Test if microphone can capture audio"""
    if not HAS_PYAUDIO:
        print("⚠ PyAudio not available (skipping)")
        return
        
    print("\n" + "=" * 64)
    print("PyAudio Microphone Test")
    print("=" * 64)
    
    p = pyaudio.PyAudio()
    print(f"Default input device: {p.get_default_input_device_info()['index']}")
    print(f"Device name: {p.get_default_input_device_info()['name']}")
    
    CHUNK = 1024
    FORMAT = pyaudio.paFloat32
    CHANNELS = 1
    RATE = 16000
    
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK,
    )
    
    print(f"\n🎤 Recording 3 seconds... (speak into microphone)")
    
    frames = []
    for i in range(0, int(RATE / CHUNK * 3)):
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)
        
        # Calculate RMS level
        audio_data = np.frombuffer(data, dtype=np.float32)
        rms = np.sqrt(np.mean(audio_data ** 2))
        
        bar = "█" * int(rms * 50)
        print(f"  Level: {bar} {rms:.4f}", end="\r")
    
    print("\n✓ Recording complete")
    
    stream.stop_stream()
    stream.close()
    p.terminate()


def test_azure():
    """Test Azure Speech SDK connection"""
    print("\n" + "=" * 64)
    print("Azure Speech SDK Test")
    print("=" * 64)
    
    file_path = Path(__file__).resolve()
    candidate_envs = [
        file_path.parents[2] / ".env",
        file_path.parents[1] / ".env",
        file_path.parent / ".env",
    ]
    for env_path in candidate_envs:
        if env_path.exists():
            load_dotenv(env_path, override=False)
    
    api_key = get_env("AZURE_SPEECH_KEY")
    region = get_env("AZURE_SPEECH_REGION", "eastus")
    
    if not api_key:
        print("❌ Missing AZURE_SPEECH_KEY")
        return
    
    print(f"✓ API Key found: {api_key[:10]}...")
    print(f"✓ Region: {region}")
    
    # Try to create recognizer
    speech_config = speechsdk.SpeechConfig(subscription=api_key, region=region)
    
    # Test single recognition
    print("\n🎤 Testing single recognition (3 seconds)...")
    audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
    recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
    
    result = recognizer.recognize_once_async().get()
    
    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        print(f"✓ Recognized: {result.text}")
    elif result.reason == speechsdk.ResultReason.NoMatch:
        print("ⓘ No speech recognized (but microphone works)")
    elif result.reason == speechsdk.ResultReason.Canceled:
        print(f"❌ Error: {result.cancellation_details.error_details}")


if __name__ == "__main__":
    try:
        test_pyaudio()
        test_azure()
    except Exception as exc:
        print(f"Error: {exc}")
        import traceback
        traceback.print_exc()
