from __future__ import annotations

import asyncio
import base64
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from nemo.collections.asr.models import EncDecMultiTaskModel


def normalize_text(text: str) -> str:
    return " ".join((text or "").strip().split())


def split_words(text: str) -> list[str]:
    normalized = normalize_text(text)
    return normalized.split(" ") if normalized else []


def delta_from_committed(committed_text: str, candidate_text: str) -> str:
    committed = normalize_text(committed_text)
    candidate = normalize_text(candidate_text)

    if not candidate:
        return ""
    if not committed:
        return candidate
    if candidate == committed:
        return ""
    if candidate.startswith(committed):
        return normalize_text(candidate[len(committed) :])
    if committed.find(candidate) >= 0:
        return ""

    committed_words = split_words(committed)
    candidate_words = split_words(candidate)
    max_overlap = min(len(committed_words), len(candidate_words), 60)
    overlap = 0
    for window in range(max_overlap, 0, -1):
        tail = " ".join(committed_words[-window:])
        head = " ".join(candidate_words[:window])
        if tail == head:
            overlap = window
            break

    return normalize_text(" ".join(candidate_words[overlap:]))


@dataclass
class SessionConfig:
    sample_rate: int = 16000
    channels: int = 1
    source_lang: str = "en"
    target_lang: str = "en"
    task: str = "asr"
    pnc: str = "yes"
    partial_interval_sec: float = 0.8
    min_infer_window_sec: float = 1.0
    max_infer_window_sec: float = 20.0
    silence_energy_threshold: float = 0.008
    silence_finalize_sec: float = 1.0


@dataclass
class SessionState:
    config: SessionConfig
    audio_samples: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    committed_text: str = ""
    latest_partial_text: str = ""
    latest_sent_partial_text: str = ""
    latest_audio_ts: float = field(default_factory=time.time)
    last_voice_ts: float = field(default_factory=time.time)
    running: bool = True
    processing_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def append_pcm16(self, pcm16_bytes: bytes) -> None:
        chunk_int16 = np.frombuffer(pcm16_bytes, dtype=np.int16)
        if chunk_int16.size == 0:
            return
        chunk_float32 = chunk_int16.astype(np.float32) / 32768.0
        if self.config.channels > 1:
            chunk_float32 = chunk_float32.reshape(-1, self.config.channels).mean(axis=1)

        self.audio_samples = np.concatenate([self.audio_samples, chunk_float32])
        self.latest_audio_ts = time.time()

        energy = float(np.sqrt(np.mean(np.square(chunk_float32))))
        if energy >= self.config.silence_energy_threshold:
            self.last_voice_ts = self.latest_audio_ts

    def current_window(self) -> np.ndarray:
        max_samples = int(self.config.max_infer_window_sec * self.config.sample_rate)
        if self.audio_samples.size <= max_samples:
            return self.audio_samples
        return self.audio_samples[-max_samples:]


class CanaryRealtimeService:
    def __init__(self, model_name: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = EncDecMultiTaskModel.from_pretrained(model_name=model_name, map_location=self.device)
        self.model = self.model.to(self.device)
        self.model.eval()

    def transcribe_window(self, audio_samples: np.ndarray, sample_rate: int, source_lang: str, target_lang: str) -> str:
        if audio_samples.size == 0:
            return ""

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
            temp_path = Path(temp_wav.name)

        try:
            sf.write(str(temp_path), audio_samples, sample_rate, subtype="PCM_16")
            output = self.model.transcribe(
                audio=[str(temp_path)],
                batch_size=1,
                source_lang=source_lang,
                target_lang=target_lang,
            )
            if not output:
                return ""
            first = output[0]
            text = first.text if hasattr(first, "text") else str(first)
            return normalize_text(text)
        finally:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass


MODEL_NAME = os.getenv("CANARY_MODEL_NAME", "nvidia/canary-1b-v2")
HOST = os.getenv("STT_HOST", "0.0.0.0")
PORT = int(os.getenv("STT_PORT", "8000"))

app = FastAPI(title="Canary Realtime STT WebSocket Server")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

service: Optional[CanaryRealtimeService] = None


@app.on_event("startup")
async def startup_event() -> None:
    global service
    service = CanaryRealtimeService(model_name=MODEL_NAME)


@app.get("/healthz")
async def healthz() -> dict:
    return {"ok": True, "model": MODEL_NAME, "device": "cuda" if torch.cuda.is_available() else "cpu"}


async def send_json_safe(websocket: WebSocket, payload: dict) -> bool:
    try:
        await websocket.send_json(payload)
        return True
    except Exception:
        return False


async def infer_and_send_partial(websocket: WebSocket, state: SessionState) -> None:
    if service is None:
        return

    min_samples = int(state.config.min_infer_window_sec * state.config.sample_rate)
    if state.audio_samples.size < min_samples:
        return

    async with state.processing_lock:
        window = state.current_window().copy()
        partial_text = await asyncio.to_thread(
            service.transcribe_window,
            window,
            state.config.sample_rate,
            state.config.source_lang,
            state.config.target_lang,
        )
        state.latest_partial_text = partial_text

    if partial_text and partial_text != state.latest_sent_partial_text:
        state.latest_sent_partial_text = partial_text
        await send_json_safe(
            websocket,
            {
                "type": "partial",
                "text": partial_text,
            },
        )


async def maybe_finalize_on_silence(websocket: WebSocket, state: SessionState) -> None:
    silence_duration = time.time() - state.last_voice_ts
    if silence_duration < state.config.silence_finalize_sec:
        return

    delta_text = delta_from_committed(state.committed_text, state.latest_partial_text)
    if not delta_text:
        return

    state.committed_text = normalize_text(f"{state.committed_text} {delta_text}")
    await send_json_safe(
        websocket,
        {
            "type": "final",
            "text": delta_text,
            "speaker": "S0",
        },
    )


async def background_partial_loop(websocket: WebSocket, state: SessionState) -> None:
    while state.running:
        try:
            await infer_and_send_partial(websocket, state)
            await maybe_finalize_on_silence(websocket, state)
        except Exception as error:
            await send_json_safe(websocket, {"type": "error", "message": str(error)})
        await asyncio.sleep(state.config.partial_interval_sec)


def parse_session_config(websocket: WebSocket, start_payload: dict) -> SessionConfig:
    query = websocket.query_params

    source_lang = str(query.get("source_lang", start_payload.get("source_lang", "en")))
    target_lang = str(query.get("target_lang", start_payload.get("target_lang", source_lang)))
    task = str(query.get("task", start_payload.get("task", "asr")))
    pnc = str(query.get("pnc", start_payload.get("pnc", "yes")))

    sample_rate = int(start_payload.get("sample_rate", 16000))
    channels = int(start_payload.get("channels", 1))

    return SessionConfig(
        sample_rate=sample_rate,
        channels=channels,
        source_lang=source_lang,
        target_lang=target_lang,
        task=task,
        pnc=pnc,
    )


@app.websocket("/ws/stt")
async def websocket_stt(websocket: WebSocket) -> None:
    await websocket.accept()

    state: Optional[SessionState] = None
    partial_task: Optional[asyncio.Task] = None

    try:
        while True:
            message = await websocket.receive_json()
            message_type = message.get("type")

            if message_type == "start":
                config = parse_session_config(websocket, message)
                state = SessionState(config=config)
                if partial_task is not None:
                    partial_task.cancel()
                partial_task = asyncio.create_task(background_partial_loop(websocket, state))
                continue

            if message_type == "audio":
                if state is None:
                    await send_json_safe(websocket, {"type": "error", "message": "start not received"})
                    continue
                encoded = message.get("data", "")
                if not encoded:
                    continue
                pcm16_bytes = base64.b64decode(encoded)
                state.append_pcm16(pcm16_bytes)
                continue

            if message_type == "stop":
                if state is None:
                    continue

                await infer_and_send_partial(websocket, state)
                delta_text = delta_from_committed(state.committed_text, state.latest_partial_text)
                if delta_text:
                    state.committed_text = normalize_text(f"{state.committed_text} {delta_text}")
                    await send_json_safe(
                        websocket,
                        {
                            "type": "final",
                            "text": delta_text,
                            "speaker": "S0",
                        },
                    )
                continue

            await send_json_safe(websocket, {"type": "error", "message": f"unknown message type: {message_type}"})

    except WebSocketDisconnect:
        pass
    except Exception as error:
        await send_json_safe(websocket, {"type": "error", "message": str(error)})
    finally:
        if state is not None:
            state.running = False
        if partial_task is not None:
            partial_task.cancel()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server.canary_ws_server:app", host=HOST, port=PORT, reload=False)
