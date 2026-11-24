"""Prove the integration's speech kwargs work against the real gateway.

Uses the shipped helper (speech_request_kwargs) and the same SDK call the TTS
entity makes, so a 200 here means the entity will work unchanged.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "/home/debian/workspace/noris-ai")

from openai import AsyncOpenAI

from custom_components.noris_ai.const import TTS_VOICE
from custom_components.noris_ai.tts import speech_request_kwargs

BASE_URL = "https://ai.noris.de/v1"
MODEL = "qwen3-TTS/qsu/qwen3-tts-voicedesign"
TEXT = "Guten Abend. Der Timer läuft in zehn Minuten ab."


def api_key() -> str:
    return subprocess.run(
        [
            "op",
            "item",
            "get",
            "noris-api-key",
            "--vault",
            "Hermes",
            "--field",
            "credential",
            "--reveal",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


async def main() -> int:
    client = AsyncOpenAI(base_url=BASE_URL, api_key=api_key(), max_retries=0)
    kwargs: dict = {
        "model": MODEL,
        "input": TEXT,
        "voice": TTS_VOICE,
        "response_format": "wav",
    }
    kwargs.update(speech_request_kwargs(MODEL))
    print("request kwargs:", {k: v for k, v in kwargs.items() if k != "input"})

    response = await client.with_options(timeout=90).audio.speech.create(**kwargs)
    audio = response.content

    print(f"http: 200, bytes: {len(audio)}, magic: {audio[:4]!r}")
    if audio[:4] != b"RIFF":
        print("FAIL: not a WAV")
        return 1

    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "voice.wav"
        wav.write_bytes(audio)
        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration:stream=codec_name,sample_rate,channels",
                "-of",
                "default=nw=1",
                str(wav),
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        print("ffprobe:", probe.replace("\n", " "))

        stt = subprocess.run(
            [
                "curl",
                "-sS",
                "-H",
                f"Authorization: Bearer {api_key()}",
                "-F",
                "model=vllm/qsu/nemotron-3.5-transcribe",
                "-F",
                "language=de",
                "-F",
                f"file=@{wav}",
                f"{BASE_URL}/audio/transcriptions",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        print("stt:", stt.stdout.strip()[:300])

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
