"""Tests for the noris AI config flow."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx
from openai import APIConnectionError, AuthenticationError
import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.noris_ai.config_flow import (
    _is_chat_model,
    _is_stt_model,
    _is_tts_model,
)
from custom_components.noris_ai.const import (
    AI_TASK_SUBENTRY_TYPE,
    DEFAULT_STT_NAME,
    DEFAULT_TTS_NAME,
    DOMAIN,
    STT_SUBENTRY_TYPE,
    TTS_SUBENTRY_TYPE,
)
from homeassistant import config_entries
from homeassistant.const import CONF_API_KEY
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
from homeassistant.setup import async_setup_component

from .conftest import FakeModelList, fake_model, setup_integration


def _auth_error() -> AuthenticationError:
    """Build a 401 from the SDK."""
    request = httpx.Request("GET", "https://ai.noris.de/v1/models")
    return AuthenticationError(
        "nope", response=httpx.Response(401, request=request), body=None
    )


def _connection_error() -> APIConnectionError:
    """Build a transport failure from the SDK."""
    return APIConnectionError(
        request=httpx.Request("GET", "https://ai.noris.de/v1/models")
    )


async def test_user_flow_creates_entry(
    hass: HomeAssistant, mock_client: AsyncMock
) -> None:
    """A valid key creates the config entry."""
    await async_setup_component(hass, "homeassistant", {})
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    assert result["type"] is FlowResultType.FORM

    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_API_KEY: "sk-bf-test"}
    )

    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["title"] == "noris AI"
    assert result["data"] == {CONF_API_KEY: "sk-bf-test"}


@pytest.mark.parametrize(
    ("side_effect", "expected_error"),
    [
        (_auth_error(), "invalid_auth"),
        (_connection_error(), "cannot_connect"),
        (ValueError("boom"), "unknown"),
    ],
)
async def test_user_flow_errors(
    hass: HomeAssistant,
    mock_client: AsyncMock,
    side_effect: Exception,
    expected_error: str,
) -> None:
    """Validation failures are shown on the form rather than aborting."""
    mock_client.models.list = MagicMock(side_effect=side_effect)

    await async_setup_component(hass, "homeassistant", {})
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_API_KEY: "sk-bf-bad"}
    )

    assert result["type"] is FlowResultType.FORM
    assert result["errors"] == {"base": expected_error}


async def test_duplicate_key_aborts(
    hass: HomeAssistant, mock_client: AsyncMock, mock_config_entry: MockConfigEntry
) -> None:
    """Re-adding the same key aborts as already configured."""
    mock_config_entry.add_to_hass(hass)

    await async_setup_component(hass, "homeassistant", {})
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_API_KEY: "sk-bf-test"}
    )

    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "already_configured"


async def test_reauth_updates_key(
    hass: HomeAssistant, mock_client: AsyncMock, mock_config_entry: MockConfigEntry
) -> None:
    """Reauth replaces the stored key."""
    mock_config_entry.add_to_hass(hass)

    await async_setup_component(hass, "homeassistant", {})
    result = await mock_config_entry.start_reauth_flow(hass)
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_API_KEY: "sk-bf-new"}
    )

    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "reauth_successful"
    assert mock_config_entry.data[CONF_API_KEY] == "sk-bf-new"


async def test_reconfigure_updates_key(
    hass: HomeAssistant, mock_client: AsyncMock, mock_config_entry: MockConfigEntry
) -> None:
    """Reconfigure replaces the stored key."""
    mock_config_entry.add_to_hass(hass)

    await async_setup_component(hass, "homeassistant", {})
    result = await mock_config_entry.start_reconfigure_flow(hass)
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_API_KEY: "sk-bf-new"}
    )

    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "reconfigure_successful"
    assert mock_config_entry.data[CONF_API_KEY] == "sk-bf-new"


async def test_subentry_flow_aborts_when_entry_not_loaded(
    hass: HomeAssistant, mock_client: AsyncMock, mock_config_entry: MockConfigEntry
) -> None:
    """A subentry flow against a NOT_LOADED entry aborts rather than erroring.

    Only the conversation handler is exercised here: ConversationFlowHandler,
    AITaskFlowHandler and SttFlowHandler all guard their async_step_init with
    the identical ``if self._get_entry().state is not ConfigEntryState.LOADED``
    check, so one handler is enough to cover the shared guard.
    """
    await async_setup_component(hass, "homeassistant", {})
    mock_config_entry.add_to_hass(hass)
    assert mock_config_entry.state is config_entries.ConfigEntryState.NOT_LOADED

    result = await hass.config_entries.subentries.async_init(
        (mock_config_entry.entry_id, "conversation"),
        context={"source": config_entries.SOURCE_USER},
    )

    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "entry_not_loaded"


def test_chat_filter_accepts_text_models() -> None:
    """Ordinary vLLM chat models remain selectable."""
    assert _is_chat_model(fake_model("vllm/release/gpt-oss-120b")) is True


def test_chat_filter_rejects_rerankers_and_embeddings() -> None:
    """Non-text output modalities are not chat models.

    Uses ids the legacy substring fallback ("reranker" / "harrier") would NOT
    also reject, so this only passes if the output_modalities check itself
    excludes them.
    """
    assert (
        _is_chat_model(fake_model("vllm/release/bge-m3", output_type="rerank")) is False
    )
    assert (
        _is_chat_model(fake_model("vllm/release/e5-large", output_type="embeddings"))
        is False
    )


def test_chat_filter_rejects_audio_models() -> None:
    """Voxtral is an audio model and must not appear in chat dropdowns."""
    model = fake_model(
        "vllm/qsu/voxtral-small-24b-2507",
        hugging_face_id="mistralai/Voxtral-Small-24B-2507",
    )

    assert _is_chat_model(model) is False


def test_chat_filter_rejects_other_providers() -> None:
    """Only vllm/* models are offered, per existing policy."""
    assert _is_chat_model(fake_model("something/else")) is False


def test_chat_filter_falls_back_without_metadata() -> None:
    """With no output_modalities, the old name heuristic still applies."""
    assert (
        _is_chat_model(fake_model("vllm/release/gpt-oss-120b", output_type=None))
        is True
    )
    assert (
        _is_chat_model(fake_model("vllm/release/bge-reranker-v2-m3", output_type=None))
        is False
    )


def test_stt_filter_accepts_voxtral_by_hugging_face_id() -> None:
    """Audio capability is detected from hugging_face_id."""
    model = fake_model(
        "vllm/qsu/some-opaque-name",
        hugging_face_id="mistralai/Voxtral-Small-24B-2507",
    )

    assert _is_stt_model(model) is True


def test_stt_filter_accepts_whisper_by_id() -> None:
    """A future Whisper deployment is picked up from the model id."""
    assert _is_stt_model(fake_model("vllm/release/whisper-large-v3")) is True


def test_stt_filter_tolerates_null_hugging_face_id() -> None:
    """A null hugging_face_id must not raise; many gateway models have none."""
    assert _is_stt_model(fake_model("vllm/release/gpt-oss-120b")) is False


def test_stt_filter_rejects_chat_models() -> None:
    """Text models are not offered as transcription engines."""
    assert _is_stt_model(fake_model("vllm/qsu/glm-5-2")) is False


async def test_stt_subentry_offers_only_audio_models(
    hass: HomeAssistant, mock_client: AsyncMock, mock_config_entry: MockConfigEntry
) -> None:
    """The STT dropdown lists Voxtral and nothing else."""
    await setup_integration(hass, mock_config_entry)

    result = await hass.config_entries.subentries.async_init(
        (mock_config_entry.entry_id, STT_SUBENTRY_TYPE),
        context={"source": config_entries.SOURCE_USER},
    )

    assert result["type"] is FlowResultType.FORM
    options = result["data_schema"].schema["model"].config["options"]
    assert [option["value"] for option in options] == [
        "vllm/qsu/voxtral-small-24b-2507"
    ]


async def test_stt_subentry_creates_entity(
    hass: HomeAssistant, mock_client: AsyncMock, mock_config_entry: MockConfigEntry
) -> None:
    """Choosing a model creates the subentry titled after it."""
    await setup_integration(hass, mock_config_entry)

    result = await hass.config_entries.subentries.async_init(
        (mock_config_entry.entry_id, STT_SUBENTRY_TYPE),
        context={"source": config_entries.SOURCE_USER},
    )
    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], {"model": "vllm/qsu/voxtral-small-24b-2507"}
    )

    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["title"] == DEFAULT_STT_NAME


async def test_stt_subentry_aborts_without_audio_models(
    hass: HomeAssistant,
    mock_client: AsyncMock,
    mock_config_entry: MockConfigEntry,
    default_models: list,
) -> None:
    """With no audio model on the gateway, say so instead of an empty dropdown."""
    await setup_integration(hass, mock_config_entry)
    mock_client.models.list = MagicMock(
        return_value=FakeModelList([m for m in default_models if "voxtral" not in m.id])
    )

    result = await hass.config_entries.subentries.async_init(
        (mock_config_entry.entry_id, STT_SUBENTRY_TYPE),
        context={"source": config_entries.SOURCE_USER},
    )

    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "no_audio_models"


async def test_stt_subentry_reconfigure_changes_model(
    hass: HomeAssistant, mock_client: AsyncMock, mock_config_entry: MockConfigEntry
) -> None:
    """Reconfiguring the STT subentry offers the current model and updates it.

    SttFlowHandler.async_step_reconfigure had zero coverage: a reviewer's
    ``raise RuntimeError`` as its first statement still passed the full suite.
    """
    await setup_integration(hass, mock_config_entry)
    subentry_id = next(
        subentry_id
        for subentry_id, subentry in mock_config_entry.subentries.items()
        if subentry.subentry_type == STT_SUBENTRY_TYPE
    )

    result = await mock_config_entry.start_subentry_reconfigure_flow(hass, subentry_id)

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "reconfigure"
    model_key = next(k for k in result["data_schema"].schema if k == "model")
    assert model_key.description["suggested_value"] == "vllm/qsu/voxtral-small-24b-2507"

    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], {"model": "vllm/qsu/voxtral-small-24b-2507"}
    )

    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "reconfigure_successful"
    updated = mock_config_entry.subentries[subentry_id]
    assert updated.data["model"] == "vllm/qsu/voxtral-small-24b-2507"


async def test_conversation_subentry_excludes_audio_models(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry, mock_client: AsyncMock
) -> None:
    """Voxtral must not be offered as a conversation model."""
    await setup_integration(hass, mock_config_entry)

    result = await hass.config_entries.subentries.async_init(
        (mock_config_entry.entry_id, "conversation"),
        context={"source": config_entries.SOURCE_USER},
    )

    options = result["data_schema"].schema["model"].config["options"]
    values = [option["value"] for option in options]
    assert "vllm/qsu/voxtral-small-24b-2507" not in values
    assert "vllm/release/gpt-oss-120b" in values


def test_tts_filter_accepts_speech_models() -> None:
    """Both speech models on the gateway are offered as TTS engines."""
    assert (
        _is_tts_model(fake_model("Kokoro-TTS/release/kokoro-tts-german-martin")) is True
    )
    assert _is_tts_model(fake_model("Cosyvoice3/release/cosyvoice3-0.5b-rl")) is True


def test_tts_filter_rejects_chat_and_transcription_models() -> None:
    """Chat models and the transcription model are not speech synthesisers."""
    assert _is_tts_model(fake_model("vllm/release/gpt-oss-120b")) is False
    assert (
        _is_tts_model(
            fake_model(
                "vllm/qsu/voxtral-small-24b-2507",
                hugging_face_id="mistralai/Voxtral-Small-24B-2507",
            )
        )
        is False
    )


def test_speech_models_do_not_leak_into_chat_or_stt_filters() -> None:
    """A TTS model must not be selectable as a chat or transcription model."""
    kokoro = fake_model("Kokoro-TTS/release/kokoro-tts-german-martin")
    assert _is_chat_model(kokoro) is False
    assert _is_stt_model(kokoro) is False


async def test_ai_task_subentry_excludes_audio_models(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry, mock_client: AsyncMock
) -> None:
    """Voxtral must not be offered as an AI Task model either.

    Only the conversation half of this fix was covered; the branch's
    headline claim is that Voxtral disappears from BOTH the conversation and
    AI Task dropdowns.
    """
    await setup_integration(hass, mock_config_entry)

    result = await hass.config_entries.subentries.async_init(
        (mock_config_entry.entry_id, AI_TASK_SUBENTRY_TYPE),
        context={"source": config_entries.SOURCE_USER},
    )

    options = result["data_schema"].schema["model"].config["options"]
    values = [option["value"] for option in options]
    assert "vllm/qsu/voxtral-small-24b-2507" not in values
    assert "vllm/release/gpt-oss-120b" in values


async def test_tts_subentry_offers_only_speech_models(
    hass: HomeAssistant, mock_client: AsyncMock, mock_config_entry: MockConfigEntry
) -> None:
    """The TTS dropdown lists the speech models and nothing else."""
    await setup_integration(hass, mock_config_entry)

    result = await hass.config_entries.subentries.async_init(
        (mock_config_entry.entry_id, TTS_SUBENTRY_TYPE),
        context={"source": config_entries.SOURCE_USER},
    )

    assert result["type"] is FlowResultType.FORM
    options = result["data_schema"].schema["model"].config["options"]
    values = [option["value"] for option in options]
    assert "Cosyvoice3/release/cosyvoice3-0.5b-rl" in values
    assert "Kokoro-TTS/release/kokoro-tts-german-martin" in values
    assert "vllm/release/gpt-oss-120b" not in values
    assert "vllm/qsu/voxtral-small-24b-2507" not in values


async def test_tts_subentry_creates_entity(
    hass: HomeAssistant, mock_client: AsyncMock, mock_config_entry: MockConfigEntry
) -> None:
    """Choosing a model creates a subentry with the friendly title."""
    await setup_integration(hass, mock_config_entry)

    result = await hass.config_entries.subentries.async_init(
        (mock_config_entry.entry_id, TTS_SUBENTRY_TYPE),
        context={"source": config_entries.SOURCE_USER},
    )
    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], {"model": "Cosyvoice3/release/cosyvoice3-0.5b-rl"}
    )

    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["title"] == DEFAULT_TTS_NAME
    assert result["data"] == {"model": "Cosyvoice3/release/cosyvoice3-0.5b-rl"}


async def test_tts_subentry_reconfigure_changes_model(
    hass: HomeAssistant, mock_client: AsyncMock, mock_config_entry: MockConfigEntry
) -> None:
    """Reconfiguring the TTS subentry offers the current model and updates it.

    Mirrors test_stt_subentry_reconfigure_changes_model: TtsFlowHandler.
    async_step_reconfigure is the identical defect a reviewer caught in the
    STT work — a ``raise RuntimeError`` as its first statement, or deleting
    its ``no_speech_models`` guard, left the full suite green with zero
    coverage of this method.
    """
    await setup_integration(hass, mock_config_entry)
    subentry_id = next(
        subentry_id
        for subentry_id, subentry in mock_config_entry.subentries.items()
        if subentry.subentry_type == TTS_SUBENTRY_TYPE
    )

    result = await mock_config_entry.start_subentry_reconfigure_flow(hass, subentry_id)

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "reconfigure"
    model_key = next(k for k in result["data_schema"].schema if k == "model")
    assert (
        model_key.description["suggested_value"]
        == "Cosyvoice3/release/cosyvoice3-0.5b-rl"
    )

    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"], {"model": "Kokoro-TTS/release/kokoro-tts-german-martin"}
    )

    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "reconfigure_successful"
    updated = mock_config_entry.subentries[subentry_id]
    assert updated.data["model"] == "Kokoro-TTS/release/kokoro-tts-german-martin"


async def test_tts_subentry_reconfigure_aborts_without_speech_models(
    hass: HomeAssistant,
    mock_client: AsyncMock,
    mock_config_entry: MockConfigEntry,
    default_models: list,
) -> None:
    """Reconfigure also guards against an empty speech-model dropdown.

    Pins the ``no_speech_models`` guard inside async_step_reconfigure, which
    is a separate code path from async_step_init's identical-looking guard.
    """
    await setup_integration(hass, mock_config_entry)
    subentry_id = next(
        subentry_id
        for subentry_id, subentry in mock_config_entry.subentries.items()
        if subentry.subentry_type == TTS_SUBENTRY_TYPE
    )
    mock_client.models.list = MagicMock(
        return_value=FakeModelList([m for m in default_models if not _is_tts_model(m)])
    )

    result = await mock_config_entry.start_subentry_reconfigure_flow(hass, subentry_id)

    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "no_speech_models"


async def test_tts_subentry_aborts_without_speech_models(
    hass: HomeAssistant,
    mock_client: AsyncMock,
    mock_config_entry: MockConfigEntry,
    default_models: list,
) -> None:
    """With no speech model on the gateway, say so instead of an empty dropdown."""
    await setup_integration(hass, mock_config_entry)
    mock_client.models.list = MagicMock(
        return_value=FakeModelList([m for m in default_models if not _is_tts_model(m)])
    )

    result = await hass.config_entries.subentries.async_init(
        (mock_config_entry.entry_id, TTS_SUBENTRY_TYPE),
        context={"source": config_entries.SOURCE_USER},
    )

    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "no_speech_models"
