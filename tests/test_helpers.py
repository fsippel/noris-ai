"""Tests for the audio container helper."""

from __future__ import annotations

import io
import wave

import pytest

from custom_components.noris_ai.helpers import pcm_to_wav, tts_languages_for_model

# 100 frames of 16-bit mono silence.
PCM = b"\x00\x00" * 100


def test_pcm_to_wav_writes_a_riff_header() -> None:
    """The result is a RIFF/WAVE container, which the gateway requires."""
    result = pcm_to_wav(PCM, 16000, 2, 1)

    assert result.startswith(b"RIFF")
    assert result[8:12] == b"WAVE"


@pytest.mark.parametrize(
    ("rate", "width", "channels", "payload"),
    [
        (16000, 2, 1, b"\x00\x00" * 100),
        # A second, distinct combination: 8 kHz, 1-byte samples, stereo.
        # Catches a hardcoded setsampwidth(2)/setframerate(16000) that the
        # first case alone (which matches those literals) cannot.
        (8000, 1, 2, bytes(range(200)) * 2),
    ],
)
def test_pcm_to_wav_round_trips_parameters(
    rate: int, width: int, channels: int, payload: bytes
) -> None:
    """Channel count, sample width and frame rate survive the round trip."""
    result = pcm_to_wav(payload, rate, width, channels)

    with wave.open(io.BytesIO(result), "rb") as wav_file:
        assert wav_file.getnchannels() == channels
        assert wav_file.getsampwidth() == width
        assert wav_file.getframerate() == rate
        assert wav_file.getnframes() == len(payload) // (width * channels)


def test_pcm_to_wav_preserves_the_payload() -> None:
    """No sample data is lost or altered."""
    payload = bytes(range(256)) * 4
    result = pcm_to_wav(payload, 16000, 2, 1)

    with wave.open(io.BytesIO(result), "rb") as wav_file:
        assert wav_file.readframes(wav_file.getnframes()) == payload


def test_pcm_to_wav_handles_empty_audio() -> None:
    """An empty payload still produces a valid, zero-frame container."""
    result = pcm_to_wav(b"", 16000, 2, 1)

    with wave.open(io.BytesIO(result), "rb") as wav_file:
        assert wav_file.getnframes() == 0


@pytest.mark.parametrize(
    ("model_id", "expected"),
    [
        (
            "Kokoro-TTS/release/kokoro-tts-german-martin",
            ["de", "de-DE", "de-AT", "de-CH"],
        ),
        (
            "Orpheus-TTS/release/orpheus-tts-english-alex",
            ["en", "en-US", "en-GB", "en-AU", "en-CA"],
        ),
        (
            "Cosyvoice3/release/cosyvoice3-0.5b-rl",
            ["de", "de-DE", "de-AT", "de-CH", "en", "en-US", "en-GB", "en-AU", "en-CA"],
        ),
        (
            "some-vendor/unknown-voice-v2",
            ["de", "de-DE", "de-AT", "de-CH", "en", "en-US", "en-GB", "en-AU", "en-CA"],
        ),
    ],
)
def test_tts_languages_for_model(model_id: str, expected: list[str]) -> None:
    """Languages are derived from the model name, German-only where warranted."""
    assert tts_languages_for_model(model_id) == expected


def test_tts_languages_german_pattern_wins_over_cosyvoice() -> None:
    """An ambiguous name resolves to the safer, narrower claim."""
    assert tts_languages_for_model("Cosyvoice3/release/cosyvoice-german-anna") == [
        "de",
        "de-DE",
        "de-AT",
        "de-CH",
    ]


def test_tts_languages_english_pattern_wins_over_cosyvoice() -> None:
    """The English row is symmetric with the German one: narrower claim wins."""
    assert tts_languages_for_model("Cosyvoice3/release/cosyvoice-english-tom") == [
        "en",
        "en-US",
        "en-GB",
        "en-AU",
        "en-CA",
    ]


def test_tts_languages_returns_a_copy() -> None:
    """Callers must not be able to mutate the module-level table."""
    first = tts_languages_for_model("Cosyvoice3/release/cosyvoice3-0.5b-rl")
    first.append("zz")
    assert "zz" not in tts_languages_for_model("Cosyvoice3/release/cosyvoice3-0.5b-rl")
