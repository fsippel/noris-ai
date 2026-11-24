"""Tests for the noris AI text-to-speech entity."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
)
import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.noris_ai import tts as tts_platform
from custom_components.noris_ai.const import (
    DEFAULT_TTS_NAME,
    DOMAIN,
    TTS_PITCH_SEMITONES,
    TTS_SUBENTRY_TYPE,
    TTS_TIMEOUT,
    TTS_VOICE,
)
from homeassistant.components import tts
from homeassistant.config_entries import ConfigSubentryData
from homeassistant.const import CONF_API_KEY, CONF_MODEL
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import entity_registry as er

from .conftest import setup_integration

WAV = b"RIFF$\x00\x00\x00WAVEfmt " + b"\x00" * 40


@pytest.fixture(autouse=True)
def _skip_pitch_shift(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep ffmpeg out of the unit tests; the filter is tested on its own."""

    async def _identity(audio: bytes) -> bytes:
        return audio

    monkeypatch.setattr(tts_platform, "shift_pitch", _identity)
    monkeypatch.setattr(tts_platform, "postprocess_voicedesign", _identity)


def test_pitch_filter_lowers_pitch_without_slowing() -> None:
    """The filter resamples by the semitone factor and undoes it with atempo."""
    factor = 2 ** (TTS_PITCH_SEMITONES / 12)
    expression = tts_platform.pitch_filter(24000)
    assert f"asetrate=24000*{factor:.6f}" in expression
    assert f"atempo={1 / factor:.6f}" in expression


def _binary(content: bytes) -> SimpleNamespace:
    """Stand in for the SDK's HttpxBinaryResponseContent."""
    return SimpleNamespace(content=content)


def _status_error(status: int) -> APIStatusError:
    """Build an SDK status error."""
    request = httpx.Request("POST", "https://ai.noris.de/v1/audio/speech")
    return APIStatusError(
        "boom", response=httpx.Response(status, request=request), body=None
    )


def _entity(hass: HomeAssistant, entry: MockConfigEntry) -> tts.TextToSpeechEntity:
    """Return the single TTS entity belonging to the config entry."""
    registry = er.async_get(hass)
    entity_id = next(
        e.entity_id
        for e in er.async_entries_for_config_entry(registry, entry.entry_id)
        if e.domain == "tts"
    )
    entity = tts.get_engine_instance(hass, entity_id)
    assert entity is not None
    return entity


async def test_entity_is_created_with_derived_languages(
    hass: HomeAssistant, mock_client: AsyncMock, mock_config_entry: MockConfigEntry
) -> None:
    """The CosyVoice model yields both German and English, with German default."""
    await setup_integration(hass, mock_config_entry)
    entity = _entity(hass, mock_config_entry)

    assert "de" in entity.supported_languages
    assert "de-DE" in entity.supported_languages
    assert "en-US" in entity.supported_languages
    assert entity.default_language == "de"


async def test_synthesis_success(
    hass: HomeAssistant, mock_client: AsyncMock, mock_config_entry: MockConfigEntry
) -> None:
    """The message is sent and WAV bytes come back."""
    mock_client.audio.speech.create = AsyncMock(return_value=_binary(WAV))
    await setup_integration(hass, mock_config_entry)

    extension, audio = await _entity(hass, mock_config_entry).async_get_tts_audio(
        "Das Licht ist an.", "de", {}
    )

    assert extension == "wav"
    assert audio == WAV

    kwargs = mock_client.audio.speech.create.call_args.kwargs
    assert kwargs["model"] == "Cosyvoice3/release/cosyvoice3-0.5b-rl"
    assert kwargs["input"] == "Das Licht ist an."
    assert kwargs["voice"] == TTS_VOICE
    mock_client.with_options.assert_called_with(timeout=TTS_TIMEOUT)


async def test_empty_audio_raises(
    hass: HomeAssistant, mock_client: AsyncMock, mock_config_entry: MockConfigEntry
) -> None:
    """A zero-byte body is never reported as success."""
    mock_client.audio.speech.create = AsyncMock(return_value=_binary(b""))
    await setup_integration(hass, mock_config_entry)

    with pytest.raises(HomeAssistantError):
        await _entity(hass, mock_config_entry).async_get_tts_audio("Hallo.", "de", {})


async def test_api_error_raises(
    hass: HomeAssistant, mock_client: AsyncMock, mock_config_entry: MockConfigEntry
) -> None:
    """Transport failures surface as HomeAssistantError."""
    request = httpx.Request("POST", "https://ai.noris.de/v1/audio/speech")
    mock_client.audio.speech.create = AsyncMock(
        side_effect=APIConnectionError(request=request)
    )
    await setup_integration(hass, mock_config_entry)

    with pytest.raises(HomeAssistantError):
        await _entity(hass, mock_config_entry).async_get_tts_audio("Hallo.", "de", {})


async def test_404_names_the_model(
    hass: HomeAssistant,
    mock_client: AsyncMock,
    mock_config_entry: MockConfigEntry,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A 404 means the chosen model cannot synthesise; say which one."""
    mock_client.audio.speech.create = AsyncMock(side_effect=_status_error(404))
    await setup_integration(hass, mock_config_entry)

    with pytest.raises(HomeAssistantError):
        await _entity(hass, mock_config_entry).async_get_tts_audio("Hallo.", "de", {})

    assert "does not support speech synthesis" in caplog.text
    assert "Cosyvoice3/release/cosyvoice3-0.5b-rl" in caplog.text


async def test_auth_error_starts_reauth(
    hass: HomeAssistant, mock_client: AsyncMock, mock_config_entry: MockConfigEntry
) -> None:
    """An expired key raises the repair flow."""
    request = httpx.Request("POST", "https://ai.noris.de/v1/audio/speech")
    mock_client.audio.speech.create = AsyncMock(
        side_effect=AuthenticationError(
            "nope", response=httpx.Response(401, request=request), body=None
        )
    )
    await setup_integration(hass, mock_config_entry)

    with pytest.raises(HomeAssistantError):
        await _entity(hass, mock_config_entry).async_get_tts_audio("Hallo.", "de", {})
    await hass.async_block_till_done()

    flows = hass.config_entries.flow.async_progress()
    assert any(flow["context"]["source"] == "reauth" for flow in flows)


async def test_timeout_raises(
    hass: HomeAssistant,
    mock_client: AsyncMock,
    mock_config_entry: MockConfigEntry,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A stalled gateway fails rather than hanging the pipeline.

    Asserting on the distinguishing "timed out after" log (the way
    test_404_names_the_model pins its own branch) is what pins this to the
    APITimeoutError clause specifically: without it, deleting that whole
    ``except openai.APITimeoutError`` clause still passes, because
    APITimeoutError subclasses OpenAIError and the broader ``except
    openai.OpenAIError`` clause below already guarantees the error doesn't
    escape as anything but a HomeAssistantError.
    """
    request = httpx.Request("POST", "https://ai.noris.de/v1/audio/speech")
    mock_client.audio.speech.create = AsyncMock(
        side_effect=APITimeoutError(request=request)
    )
    await setup_integration(hass, mock_config_entry)

    with pytest.raises(HomeAssistantError):
        await _entity(hass, mock_config_entry).async_get_tts_audio("Hallo.", "de", {})

    assert "timed out after" in caplog.text


async def test_synthesis_via_component_returns_audio(
    hass: HomeAssistant, mock_client: AsyncMock, mock_config_entry: MockConfigEntry
) -> None:
    """Enter through HA's tts component, not the entity, and get audio back.

    All the other tests in this file call async_get_tts_audio directly on the
    entity object, which never exercises the tts component's own guards —
    notably ``_async_generate_tts_audio``'s check that ``engine_instance.name``
    is neither ``None`` nor ``UNDEFINED``. That check is why
    ``_attr_has_entity_name = True`` + ``_attr_name = None`` made every
    request fail with "TTS engine name is not set." even though the entity's
    own async_get_tts_audio worked fine in isolation. Going through
    generate_media_source_id / async_get_media_source_audio is what actually
    proves the platform can speak.

    preferred_format "wav" is mandatory: without it HA defaults to mp3 and
    shells out to the ffmpeg *binary*, which is not installed in this
    environment (ha-ffmpeg in requirements_test.txt is only the Python
    wrapper), so omitting the hint fails the test for an unrelated reason.
    """
    mock_client.audio.speech.create = AsyncMock(return_value=_binary(WAV))
    await setup_integration(hass, mock_config_entry)

    registry = er.async_get(hass)
    entity_id = next(
        e.entity_id
        for e in er.async_entries_for_config_entry(registry, mock_config_entry.entry_id)
        if e.domain == "tts"
    )

    msg_id = tts.generate_media_source_id(
        hass,
        "Das Licht ist an.",
        engine=entity_id,
        language="de-DE",
        options={"preferred_format": "wav"},
    )
    extension, audio = await tts.async_get_media_source_audio(hass, msg_id)

    assert extension == "wav"
    assert audio == WAV
    mock_client.audio.speech.create.assert_called_once()


async def test_spoken_text_is_never_logged(
    hass: HomeAssistant,
    mock_client: AsyncMock,
    mock_config_entry: MockConfigEntry,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The message is user content and must not reach the log at any level."""
    secret = "Der Tresorcode lautet dreiundvierzig."
    mock_client.audio.speech.create = AsyncMock(return_value=_binary(WAV))
    await setup_integration(hass, mock_config_entry)

    with caplog.at_level("DEBUG"):
        await _entity(hass, mock_config_entry).async_get_tts_audio(secret, "de", {})

    assert secret not in caplog.text


def test_product_names_are_respelled_for_the_german_voice() -> None:
    """Home Assistant must not be read with German phonemes."""
    assert (
        tts_platform.spoken_text("Ich bin der Assistent von Home Assistant.")
        == "Ich bin der Assistent von Houm Ässisstänt."
    )
    assert (
        tts_platform.spoken_text("Kein Produktname hier.") == "Kein Produktname hier."
    )


def test_voicedesign_fields_travel_only_with_the_opt_in() -> None:
    """A VoiceDesign model needs task_type, style, language — and the header."""
    assert tts_platform.speech_request_kwargs("Kokoro-TTS/release/kokoro-tts-de") == {}

    kwargs = tts_platform.speech_request_kwargs("qwen3-TTS/qsu/qwen3-tts-voicedesign")
    assert kwargs["voice"] == "vivian"
    assert kwargs["extra_headers"] == {"x-bf-passthrough-extra-params": "true"}
    assert kwargs["extra_body"] == {
        "task_type": "VoiceDesign",
        "instructions": tts_platform.TTS_VOICEDESIGN_STYLE,
        "language": "German",
    }


async def test_voicedesign_request_reaches_the_gateway_opted_in(
    hass: HomeAssistant, mock_client: AsyncMock
) -> None:
    """Without the header the gateway answers 400; the entity must always send it."""
    mock_client.audio.speech.create = AsyncMock(return_value=_binary(WAV))
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="noris AI",
        data={CONF_API_KEY: "sk-bf-test"},
        subentries_data=[
            ConfigSubentryData(
                data={CONF_MODEL: "qwen3-TTS/qsu/qwen3-tts-voicedesign"},
                subentry_type=TTS_SUBENTRY_TYPE,
                title=DEFAULT_TTS_NAME,
                unique_id=None,
            )
        ],
    )
    entry.add_to_hass(hass)
    await setup_integration(hass, entry)

    extension, audio = await _entity(hass, entry).async_get_tts_audio(
        "Der Timer läuft in zehn Minuten ab.", "de", {}
    )

    assert extension == "wav"
    assert audio == WAV
    kwargs = mock_client.audio.speech.create.call_args.kwargs
    assert kwargs["voice"] == "vivian"
    assert kwargs["extra_headers"] == {"x-bf-passthrough-extra-params": "true"}
    assert kwargs["extra_body"]["task_type"] == "VoiceDesign"
    assert kwargs["extra_body"]["language"] == "German"
    assert "task_type" not in kwargs


def test_voicedesign_filter_speeds_up_and_normalises() -> None:
    """atempo leaves the pitch alone; loudnorm closes the level gap to Kokoro."""
    assert (
        tts_platform.voicedesign_filter(1.10, "loudnorm=x")
        == "atempo=1.100000,loudnorm=x"
    )
    assert tts_platform.voicedesign_filter(1.0, "loudnorm=x") == "loudnorm=x"
    assert tts_platform.voicedesign_filter(1.0, "") is None


async def test_voicedesign_gets_the_tempo_pass(
    hass: HomeAssistant, mock_client: AsyncMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The deeper VoiceDesign voice speaks 10 % faster than the checkpoint does."""
    mock_client.audio.speech.create = AsyncMock(return_value=_binary(WAV))
    speed = AsyncMock(return_value=WAV)
    monkeypatch.setattr(tts_platform, "postprocess_voicedesign", speed)
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="noris AI",
        data={CONF_API_KEY: "sk-bf-test"},
        subentries_data=[
            ConfigSubentryData(
                data={CONF_MODEL: "qwen3-TTS/qsu/qwen3-tts-voicedesign"},
                subentry_type=TTS_SUBENTRY_TYPE,
                title=DEFAULT_TTS_NAME,
                unique_id=None,
            )
        ],
    )
    entry.add_to_hass(hass)
    await setup_integration(hass, entry)

    _, audio = await _entity(hass, entry).async_get_tts_audio("Hallo.", "de", {})

    speed.assert_awaited_once()
    assert speed.await_args.args[0] == WAV
    assert audio == WAV


async def test_cosyvoice_keeps_its_natural_tempo(
    hass: HomeAssistant,
    mock_client: AsyncMock,
    mock_config_entry: MockConfigEntry,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only VoiceDesign is post-processed; every other model is left alone."""
    mock_client.audio.speech.create = AsyncMock(return_value=_binary(WAV))
    speed = AsyncMock(return_value=WAV)
    monkeypatch.setattr(tts_platform, "postprocess_voicedesign", speed)
    await setup_integration(hass, mock_config_entry)

    await _entity(hass, mock_config_entry).async_get_tts_audio("Hallo.", "de", {})

    speed.assert_not_awaited()
