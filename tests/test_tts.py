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

from custom_components.noris_ai.const import TTS_TIMEOUT, TTS_VOICE
from homeassistant.components import tts
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import entity_registry as er

from .conftest import setup_integration

WAV = b"RIFF$\x00\x00\x00WAVEfmt " + b"\x00" * 40


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
