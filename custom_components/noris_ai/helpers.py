"""Helper functions for the noris AI integration."""

from __future__ import annotations

import io
import wave

from .const import TTS_DEFAULT_LANGUAGES, TTS_LANGUAGES_BY_PATTERN


def pcm_to_wav(audio: bytes, rate: int, width: int, channels: int) -> bytes:
    """Wrap raw PCM samples in a WAV container.

    The Assist pipeline delivers headerless PCM, which the ai.noris.de
    transcription endpoint rejects with "Invalid or unsupported audio file".

    Args:
        audio: Raw PCM sample data.
        rate: Sample rate in Hz, e.g. 16000.
        width: Sample width in **bytes**, e.g. 2 for 16-bit audio.
        channels: Number of channels, e.g. 1 for mono.

    Returns:
        The same samples wrapped in a RIFF/WAVE container.

    """
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(width)
        wav_file.setframerate(rate)
        wav_file.writeframes(audio)
    return buffer.getvalue()


def tts_languages_for_model(model_id: str) -> list[str]:
    """Return the language tags a speech model can be trusted with.

    The gateway's catalog reports no language information, so coverage is
    derived from the model name — see TTS_LANGUAGES_BY_PATTERN. Returns a copy
    so callers cannot mutate the module-level table.
    """
    for pattern, languages in TTS_LANGUAGES_BY_PATTERN:
        if pattern.search(model_id):
            return list(languages)
    return list(TTS_DEFAULT_LANGUAGES)
