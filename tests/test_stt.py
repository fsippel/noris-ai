"""Tests for the noris AI speech-to-text entity."""

from __future__ import annotations

from collections.abc import AsyncIterable
import io
from types import SimpleNamespace
from unittest.mock import AsyncMock
import wave

import httpx
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
)
import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.noris_ai.const import (
    DEFAULT_STT_NAME,
    DOMAIN,
    STT_SUBENTRY_TYPE,
    STT_TIMEOUT,
)
from homeassistant.components import stt
from homeassistant.config_entries import ConfigSubentryData
from homeassistant.const import CONF_API_KEY, CONF_MODEL
from homeassistant.core import HomeAssistant
from homeassistant.helpers import device_registry as dr, entity_registry as er

from .conftest import setup_integration

PCM = b"\x00\x01" * 800


def _metadata(language: str = "de-DE") -> stt.SpeechMetadata:
    """Build the metadata the Assist pipeline supplies."""
    return stt.SpeechMetadata(
        language=language,
        format=stt.AudioFormats.WAV,
        codec=stt.AudioCodecs.PCM,
        bit_rate=stt.AudioBitRates.BITRATE_16,
        sample_rate=stt.AudioSampleRates.SAMPLERATE_16000,
        channel=stt.AudioChannels.CHANNEL_MONO,
    )


async def _stream(*chunks: bytes) -> AsyncIterable[bytes]:
    """Yield audio chunks the way the pipeline does."""
    for chunk in chunks:
        yield chunk


def _entity(hass: HomeAssistant, entry: MockConfigEntry) -> stt.SpeechToTextEntity:
    """Return the single STT entity belonging to the config entry.

    Resolved through the entity registry and the stt component's public
    lookup, rather than by reaching into hass.data internals.
    """
    registry = er.async_get(hass)
    entity_id = next(
        registry_entry.entity_id
        for registry_entry in er.async_entries_for_config_entry(
            registry, entry.entry_id
        )
        if registry_entry.domain == "stt"
    )
    entity = stt.async_get_speech_to_text_entity(hass, entity_id)
    assert entity is not None
    return entity


def _status_error(status: int) -> APIStatusError:
    """Build an SDK status error."""
    request = httpx.Request("POST", "https://ai.noris.de/v1/audio/transcriptions")
    return APIStatusError(
        "boom", response=httpx.Response(status, request=request), body=None
    )


async def test_entity_is_created(
    hass: HomeAssistant, mock_client: AsyncMock, mock_config_entry: MockConfigEntry
) -> None:
    """An stt subentry produces an STT entity advertising German."""
    await setup_integration(hass, mock_config_entry)
    entity = _entity(hass, mock_config_entry)

    assert "de" in entity.supported_languages
    assert "de-DE" in entity.supported_languages
    assert entity.supported_formats == [stt.AudioFormats.WAV]
    assert entity.supported_codecs == [stt.AudioCodecs.PCM]
    assert entity.supported_bit_rates == [stt.AudioBitRates.BITRATE_16]
    assert entity.supported_sample_rates == [stt.AudioSampleRates.SAMPLERATE_16000]
    assert entity.supported_channels == [stt.AudioChannels.CHANNEL_MONO]


async def test_transcription_success(
    hass: HomeAssistant, mock_client: AsyncMock, mock_config_entry: MockConfigEntry
) -> None:
    """Audio is wrapped as WAV, posted, and the transcript returned."""
    mock_client.audio.transcriptions.create = AsyncMock(
        return_value=SimpleNamespace(text="Schalte das Licht ein.")
    )
    await setup_integration(hass, mock_config_entry)
    # Setup itself calls with_options (to validate the API key); reset so the
    # assertion below only sees the transcription call.
    mock_client.with_options.reset_mock()

    result = await _entity(hass, mock_config_entry).async_process_audio_stream(
        _metadata(), _stream(PCM[:800], PCM[800:])
    )

    assert result.result is stt.SpeechResultState.SUCCESS
    assert result.text == "Schalte das Licht ein."

    mock_client.with_options.assert_called_once_with(timeout=STT_TIMEOUT, max_retries=0)

    kwargs = mock_client.audio.transcriptions.create.call_args.kwargs
    assert kwargs["model"] == "vllm/qsu/voxtral-small-24b-2507"
    assert kwargs["language"] == "de"
    filename, payload, content_type = kwargs["file"]
    assert filename == "audio.wav"
    assert content_type == "audio/wav"
    assert payload.startswith(b"RIFF")
    # Pin the WAV argument mapping: metadata.sample_rate.value and
    # metadata.channel.value must land in the right wave.Wave_write setters,
    # not transposed (both are accepted uncomplaining by `wave`, producing a
    # corrupt-but-valid RIFF file).
    with wave.open(io.BytesIO(payload), "rb") as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getsampwidth() == 2
        assert wav_file.getframerate() == 16000


async def test_empty_stream_skips_the_api(
    hass: HomeAssistant, mock_client: AsyncMock, mock_config_entry: MockConfigEntry
) -> None:
    """Silence is rejected without a pointless round trip."""
    mock_client.audio.transcriptions.create = AsyncMock()
    await setup_integration(hass, mock_config_entry)

    result = await _entity(hass, mock_config_entry).async_process_audio_stream(
        _metadata(), _stream()
    )

    assert result.result is stt.SpeechResultState.ERROR
    assert result.text is None
    mock_client.audio.transcriptions.create.assert_not_called()


async def test_blank_transcript_is_an_error(
    hass: HomeAssistant, mock_client: AsyncMock, mock_config_entry: MockConfigEntry
) -> None:
    """A whitespace-only transcript must not be reported as success."""
    mock_client.audio.transcriptions.create = AsyncMock(
        return_value=SimpleNamespace(text="   ")
    )
    await setup_integration(hass, mock_config_entry)

    result = await _entity(hass, mock_config_entry).async_process_audio_stream(
        _metadata(), _stream(PCM)
    )

    assert result.result is stt.SpeechResultState.ERROR


async def test_api_error_returns_error_result(
    hass: HomeAssistant, mock_client: AsyncMock, mock_config_entry: MockConfigEntry
) -> None:
    """Transport failures fail the pipeline rather than raising."""
    request = httpx.Request("POST", "https://ai.noris.de/v1/audio/transcriptions")
    mock_client.audio.transcriptions.create = AsyncMock(
        side_effect=APIConnectionError(request=request)
    )
    await setup_integration(hass, mock_config_entry)

    result = await _entity(hass, mock_config_entry).async_process_audio_stream(
        _metadata(), _stream(PCM)
    )

    assert result.result is stt.SpeechResultState.ERROR


async def test_timeout_returns_error_result(
    hass: HomeAssistant, mock_client: AsyncMock, mock_config_entry: MockConfigEntry
) -> None:
    """A stalled gateway fails the pipeline rather than hanging it."""
    request = httpx.Request("POST", "https://ai.noris.de/v1/audio/transcriptions")
    mock_client.audio.transcriptions.create = AsyncMock(
        side_effect=APITimeoutError(request=request)
    )
    await setup_integration(hass, mock_config_entry)

    result = await _entity(hass, mock_config_entry).async_process_audio_stream(
        _metadata(), _stream(PCM)
    )

    assert result.result is stt.SpeechResultState.ERROR


async def test_404_names_the_model(
    hass: HomeAssistant,
    mock_client: AsyncMock,
    mock_config_entry: MockConfigEntry,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A 404 means the chosen model cannot transcribe; say which one."""
    mock_client.audio.transcriptions.create = AsyncMock(side_effect=_status_error(404))
    await setup_integration(hass, mock_config_entry)

    result = await _entity(hass, mock_config_entry).async_process_audio_stream(
        _metadata(), _stream(PCM)
    )

    assert result.result is stt.SpeechResultState.ERROR
    assert "vllm/qsu/voxtral-small-24b-2507" in caplog.text


async def test_auth_error_starts_reauth(
    hass: HomeAssistant, mock_client: AsyncMock, mock_config_entry: MockConfigEntry
) -> None:
    """An expired key raises the repair flow instead of failing silently."""
    request = httpx.Request("POST", "https://ai.noris.de/v1/audio/transcriptions")
    mock_client.audio.transcriptions.create = AsyncMock(
        side_effect=AuthenticationError(
            "nope", response=httpx.Response(401, request=request), body=None
        )
    )
    await setup_integration(hass, mock_config_entry)

    result = await _entity(hass, mock_config_entry).async_process_audio_stream(
        _metadata(), _stream(PCM)
    )
    await hass.async_block_till_done()

    assert result.result is stt.SpeechResultState.ERROR
    flows = hass.config_entries.flow.async_progress()
    assert any(flow["context"]["source"] == "reauth" for flow in flows)


async def test_stream_read_failure_returns_error_result(
    hass: HomeAssistant, mock_client: AsyncMock, mock_config_entry: MockConfigEntry
) -> None:
    """A stream that raises mid-read must yield ERROR, never propagate.

    The controller proved this path with a throwaway test and deleted it,
    leaving the method's own docstring promise ("never raises") permanently
    unguarded. This replaces that test.
    """

    async def _broken_stream() -> AsyncIterable[bytes]:
        yield PCM[:800]
        raise ConnectionResetError("connection reset by peer")

    mock_client.audio.transcriptions.create = AsyncMock()
    await setup_integration(hass, mock_config_entry)

    result = await _entity(hass, mock_config_entry).async_process_audio_stream(
        _metadata(), _broken_stream()
    )

    assert result.result is stt.SpeechResultState.ERROR
    assert result.text is None
    mock_client.audio.transcriptions.create.assert_not_called()


async def test_language_without_region(
    hass: HomeAssistant, mock_client: AsyncMock, mock_config_entry: MockConfigEntry
) -> None:
    """A bare language tag is passed through unchanged."""
    mock_client.audio.transcriptions.create = AsyncMock(
        return_value=SimpleNamespace(text="ok")
    )
    await setup_integration(hass, mock_config_entry)

    await _entity(hass, mock_config_entry).async_process_audio_stream(
        _metadata("en"), _stream(PCM)
    )

    assert mock_client.audio.transcriptions.create.call_args.kwargs["language"] == "en"


async def test_device_identity_matches_across_platforms(
    hass: HomeAssistant, mock_client: AsyncMock, mock_config_entry: MockConfigEntry
) -> None:
    """Every platform's device carries the same identity fields.

    The device-info block is shared through NorisAISubentryEntity; this pins the
    fields so the extraction cannot silently change any of them.

    ``identifiers`` and ``unique_id`` get an exact-value assertion, not merely
    "is not None": drift in either is exactly what silently re-registers a
    user's device and orphans their entities, and a reviewer's mutation battery
    showed the previous "is not None" checks did not catch that. The entity
    registry's ``unique_id`` is checked too, since the device identifiers alone
    don't pin what NorisAISubentryEntity.__init__ writes to
    ``self._attr_unique_id``.
    """
    await setup_integration(hass, mock_config_entry)

    device_registry = dr.async_get(hass)
    devices = dr.async_entries_for_config_entry(
        device_registry, mock_config_entry.entry_id
    )

    assert len(devices) == 4
    devices_by_subentry_id = {device.config_subentry_id: device for device in devices}
    assert devices_by_subentry_id.keys() == mock_config_entry.subentries.keys()

    entity_registry = er.async_get(hass)
    entities = er.async_entries_for_config_entry(
        entity_registry, mock_config_entry.entry_id
    )
    entities_by_subentry_id = {e.config_subentry_id: e for e in entities}

    for subentry_id, subentry in mock_config_entry.subentries.items():
        device = devices_by_subentry_id[subentry_id]
        assert device.identifiers == {(DOMAIN, subentry_id)}
        assert device.manufacturer == "noris network AG"
        assert device.entry_type is dr.DeviceEntryType.SERVICE
        assert device.model == subentry.data[CONF_MODEL]
        assert device.name == subentry.title

        entity_entry = entities_by_subentry_id[subentry_id]
        assert entity_entry.unique_id == subentry_id


def test_answer_shaped_transcript_is_discarded() -> None:
    """Voxtral sometimes answers instead of transcribing; that must not pass."""
    from custom_components.noris_ai.stt import _looks_like_an_answer

    assert not _looks_like_an_answer("Wie hoch ist der PV-Ertrag?")
    assert not _looks_like_an_answer("Schalte das Licht in der Küche ein.")
    assert _looks_like_an_answer("### Faktoren, die den Ertrag beeinflussen")
    assert _looks_like_an_answer(
        "Der Ertrag hängt von mehreren Faktoren ab.\n\n1. Lage"
    )
    assert _looks_like_an_answer("x" * 300)


NEMOTRON_MODEL = "vllm/qsu/nemotron-3.5-transcribe"


async def test_unavailable_model_falls_back_to_the_stable_one(
    hass: HomeAssistant,
    mock_client: AsyncMock,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A 503 from the configured model must not kill the voice turn.

    The qsu lane answers 503 with an HTML "Application is not available" page
    while a pod rolls; without the fallback the satellite says "speech-to-text
    failed" and the whole turn is lost.
    """
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="noris AI",
        data={CONF_API_KEY: "sk-bf-test"},
        subentries_data=[
            ConfigSubentryData(
                data={CONF_MODEL: NEMOTRON_MODEL},
                subentry_type=STT_SUBENTRY_TYPE,
                title=DEFAULT_STT_NAME,
                unique_id=None,
            )
        ],
    )
    entry.add_to_hass(hass)

    seen: list[str] = []

    async def create(**kwargs):
        seen.append(kwargs["model"])
        if kwargs["model"] == NEMOTRON_MODEL:
            raise _status_error(503)
        return SimpleNamespace(text="Wie voll ist der Akku?")

    mock_client.audio.transcriptions.create = AsyncMock(side_effect=create)
    await setup_integration(hass, entry)

    result = await _entity(hass, entry).async_process_audio_stream(
        _metadata(), _stream(PCM)
    )

    assert result.result is stt.SpeechResultState.SUCCESS
    assert result.text == "Wie voll ist der Akku?"
    assert seen == [NEMOTRON_MODEL, "vllm/qsu/voxtral-small-24b-2507"]
    assert "fallback" in caplog.text.lower()


async def test_transient_error_retries_the_same_model(
    hass: HomeAssistant,
    mock_client: AsyncMock,
    mock_config_entry: MockConfigEntry,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """No distinct fallback configured: the same model gets a second chance.

    While the STT upstream flaps, a single 503 costs one extra round trip
    instead of the whole voice turn.
    """
    seen: list[str] = []

    async def create(**kwargs):
        seen.append(kwargs["model"])
        if len(seen) == 1:
            raise _status_error(503)
        return SimpleNamespace(text="Wie voll ist der Akku?")

    mock_client.audio.transcriptions.create = AsyncMock(side_effect=create)
    await setup_integration(hass, mock_config_entry)

    result = await _entity(hass, mock_config_entry).async_process_audio_stream(
        _metadata(), _stream(PCM)
    )

    assert result.result is stt.SpeechResultState.SUCCESS
    assert seen == ["vllm/qsu/voxtral-small-24b-2507"] * 2
    assert "retry" in caplog.text.lower()


async def test_breaker_skips_a_dead_primary_between_turns(
    hass: HomeAssistant,
    mock_client: AsyncMock,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Once the primary failed, later turns must not pay its doomed call.

    Turn 1: nemotron 503, voxtral rescues. Turn 2 goes straight to voxtral;
    after the cooldown the primary is probed again (not asserted here — that
    would need a frozen clock).
    """
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="noris AI",
        data={CONF_API_KEY: "sk-bf-test"},
        subentries_data=[
            ConfigSubentryData(
                data={CONF_MODEL: NEMOTRON_MODEL},
                subentry_type=STT_SUBENTRY_TYPE,
                title=DEFAULT_STT_NAME,
                unique_id=None,
            )
        ],
    )
    entry.add_to_hass(hass)

    seen: list[str] = []

    async def create(**kwargs):
        model = kwargs["model"]
        seen.append(model)
        if model == NEMOTRON_MODEL:
            raise _status_error(503)
        return SimpleNamespace(text="Wie voll ist der Akku?")

    mock_client.audio.transcriptions.create = AsyncMock(side_effect=create)
    await setup_integration(hass, entry)
    entity = _entity(hass, entry)

    first = await entity.async_process_audio_stream(_metadata(), _stream(PCM))
    second = await entity.async_process_audio_stream(_metadata(), _stream(PCM))

    assert first.result is stt.SpeechResultState.SUCCESS
    assert second.result is stt.SpeechResultState.SUCCESS
    assert seen.count(NEMOTRON_MODEL) == 1
    assert seen.count("vllm/qsu/voxtral-small-24b-2507") == 2
    assert "then trying the primary again" in caplog.text
