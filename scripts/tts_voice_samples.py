"""Generate voice samples through the noris gateway: Kokoro reference vs Qwen3-TTS styles.

Uses the shipped integration helper (speech_request_kwargs) so the samples are
produced by exactly the code path Home Assistant uses. Writes one WAV per
variant into the given output directory and prints size/latency.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from openai import AsyncOpenAI

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from custom_components.noris_ai.tts import speech_request_kwargs

BASE_URL = "https://ai.noris.de/v1"
TEXT = "Guten Abend. Der Timer läuft in zehn Minuten ab. Möchtest du noch etwas?"
KOKORO = "Kokoro-TTS/release/kokoro-tts-german-martin"
QWEN = "qwen3-TTS/qsu/qwen3-tts-voicedesign"

STYLES = {
    "qwen-warm-frau": "Eine warme, freundliche deutsche Frauenstimme, ruhiges Tempo.",
    "qwen-ruhig-mann": "Eine tiefe, ruhige deutsche Männerstimme, sachlich und gelassen.",
    "qwen-hell-frau": "Eine junge, helle deutsche Frauenstimme, lebhaft und freundlich.",
    "qwen-warm-mann": "Eine warme, freundliche deutsche Männerstimme, ruhiges Tempo.",
    "qwen-warm-mann-tief": "Eine warme, tiefe deutsche Männerstimme, ruhig und zugewandt, langsames Tempo.",
    "final-zuegig": "Eine warme, tiefe deutsche Männerstimme, ruhig und zugewandt, zügiges Tempo.",
    "final-sehr-zuegig": "Eine warme, tiefe deutsche Männerstimme, ruhig und zugewandt, zügiges, energisches Tempo.",
}


def api_key() -> str:
    key = os.environ.get("NORIS_API_KEY")
    if key:
        return key
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


async def render(
    client: AsyncOpenAI, name: str, model: str, out_dir: Path, style: str | None
) -> dict:
    kwargs: dict[str, Any] = {
        "model": model,
        "input": TEXT,
        "voice": "default",
        "response_format": "wav",
    }
    kwargs.update(speech_request_kwargs(model))
    if style is not None:
        kwargs["extra_body"] = {**kwargs.get("extra_body", {}), "instructions": style}
    started = time.perf_counter()
    response = await client.with_options(timeout=120).audio.speech.create(**kwargs)
    elapsed = time.perf_counter() - started
    audio = response.content
    path = out_dir / f"{name}.wav"
    path.write_bytes(audio)
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=sample_rate,channels",
            "-of",
            "default=nw=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    return {
        "name": name,
        "model": model,
        "bytes": len(audio),
        "seconds": round(elapsed, 2),
        "path": path,
        "probe": " ".join(probe),
    }


async def main() -> int:
    out_dir = Path(
        sys.argv[1]
        if len(sys.argv) > 1
        else "/home/debian/.hermes/cache/scratch/voices"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    client = AsyncOpenAI(api_key=api_key(), base_url=BASE_URL, max_retries=0)

    results = [await render(client, "kokoro-martin", KOKORO, out_dir, None)]
    for name, style in STYLES.items():
        try:
            results.append(await render(client, name, QWEN, out_dir, style))
        except Exception as err:
            print(f"{name}: FAILED {type(err).__name__}: {err}")

    for r in results:
        print(f"{r['name']:<16} {r['bytes']:>8} B  {r['seconds']:>5} s  {r['probe']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
