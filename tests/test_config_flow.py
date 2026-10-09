"""Tests for the noris AI config flow."""

from unittest.mock import MagicMock

from openai import APIConnectionError, AuthenticationError
import pytest

from homeassistant import config_entries
from homeassistant.const import CONF_API_KEY, CONF_LLM_HASS_API, CONF_MODEL, CONF_NAME
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType

from custom_components.noris_ai.const import (
    AI_TASK_SUBENTRY_TYPE,
    CONF_MAX_TOKENS,
    CONF_RECOMMENDED,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    CONVERSATION_SUBENTRY_TYPE,
    DEFAULT_MODEL,
    DOMAIN,
)
from .conftest import setup_integration


async def test_user_flow_happy_path(
    hass: HomeAssistant, mock_openai: MagicMock
) -> None:
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    assert result["type"] is FlowResultType.FORM
    assert result["errors"] == {}

    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_API_KEY: "sk-bf-test"}
    )
    await hass.async_block_till_done()
    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["title"] == "noris AI"
    assert result["data"] == {CONF_API_KEY: "sk-bf-test"}


@pytest.mark.parametrize(
    ("side_effect", "error"),
    [
        (
            AuthenticationError(message="401", response=MagicMock(), body=None),
            "invalid_auth",
        ),
        (APIConnectionError(request=MagicMock()), "cannot_connect"),
    ],
)
async def test_user_flow_errors(
    hass: HomeAssistant, mock_openai: MagicMock, side_effect, error
) -> None:
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    mock_openai.models.list.side_effect = side_effect
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_API_KEY: "sk-bf-bad"}
    )
    assert result["type"] is FlowResultType.FORM
    assert result["errors"] == {"base": error}

    # Recovery: next attempt with a working key succeeds.
    mock_openai.models.list.side_effect = None
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_API_KEY: "sk-bf-good"}
    )
    assert result["type"] is FlowResultType.CREATE_ENTRY


async def test_duplicate_key_aborts(
    hass: HomeAssistant, mock_openai: MagicMock, mock_config_entry
) -> None:
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_API_KEY: "sk-bf-test"}
    )
    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "already_configured"


async def test_reauth_flow(
    hass: HomeAssistant, mock_openai: MagicMock, mock_config_entry
) -> None:
    result = await mock_config_entry.start_reauth_flow(hass)
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "reauth_confirm"

    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_API_KEY: "sk-bf-new"}
    )
    await hass.async_block_till_done()
    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "reauth_successful"
    assert mock_config_entry.data[CONF_API_KEY] == "sk-bf-new"


async def test_conversation_subentry_recommended(
    hass: HomeAssistant, mock_openai: MagicMock, mock_config_entry
) -> None:
    """Create an agent with recommended settings — one step only."""
    await setup_integration(hass, mock_config_entry)

    result = await hass.config_entries.subentries.async_init(
        (mock_config_entry.entry_id, CONVERSATION_SUBENTRY_TYPE),
        context={"source": config_entries.SOURCE_USER},
    )
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "init"

    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"],
        {
            CONF_NAME: "Mein Agent",
            CONF_MODEL: DEFAULT_MODEL,
            CONF_LLM_HASS_API: ["assist"],
            CONF_RECOMMENDED: True,
        },
    )
    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["title"] == "Mein Agent"
    assert result["data"] == {
        CONF_MODEL: DEFAULT_MODEL,
        CONF_LLM_HASS_API: ["assist"],
        CONF_RECOMMENDED: True,
    }


async def test_conversation_subentry_advanced(
    hass: HomeAssistant, mock_openai: MagicMock, mock_config_entry
) -> None:
    """Unchecking recommended leads to the advanced step."""
    await setup_integration(hass, mock_config_entry)

    result = await hass.config_entries.subentries.async_init(
        (mock_config_entry.entry_id, CONVERSATION_SUBENTRY_TYPE),
        context={"source": config_entries.SOURCE_USER},
    )
    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"],
        {
            CONF_NAME: "Tuning-Agent",
            CONF_MODEL: "vllm/release/glm-5-2",
            CONF_RECOMMENDED: False,
        },
    )
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "advanced"

    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"],
        {
            "prompt": "Du bist knapp.",
            CONF_MAX_TOKENS: 5000,
            CONF_TEMPERATURE: 0.5,
            CONF_TOP_P: 0.9,
        },
    )
    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["data"][CONF_MODEL] == "vllm/release/glm-5-2"
    assert result["data"][CONF_MAX_TOKENS] == 5000
    assert result["data"]["prompt"] == "Du bist knapp."


async def test_conversation_subentry_reconfigure(
    hass: HomeAssistant, mock_openai: MagicMock, mock_config_entry
) -> None:
    """Model can be changed on reconfigure."""
    await setup_integration(hass, mock_config_entry)

    result = await hass.config_entries.subentries.async_init(
        (mock_config_entry.entry_id, CONVERSATION_SUBENTRY_TYPE),
        context={"source": config_entries.SOURCE_USER},
    )
    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"],
        {
            CONF_NAME: "Agent",
            CONF_MODEL: DEFAULT_MODEL,
            CONF_RECOMMENDED: True,
        },
    )
    subentry_id = list(mock_config_entry.subentries)[0]

    result = await mock_config_entry.start_subentry_reconfigure_flow(
        hass, subentry_id
    )
    assert result["step_id"] == "init"
    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"],
        {
            CONF_MODEL: "vllm/qwen3.6-27b",
            CONF_RECOMMENDED: True,
        },
    )
    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "reconfigure_successful"
    subentry = mock_config_entry.subentries[subentry_id]
    assert subentry.data[CONF_MODEL] == "vllm/qwen3.6-27b"


async def test_conversation_subentry_reconfigure_keeps_control_disabled(
    hass: HomeAssistant, mock_openai: MagicMock, mock_config_entry
) -> None:
    """Reconfiguring a subentry created without device control must not
    silently re-enable it by defaulting llm_hass_api to Assist again."""
    await setup_integration(hass, mock_config_entry)

    result = await hass.config_entries.subentries.async_init(
        (mock_config_entry.entry_id, CONVERSATION_SUBENTRY_TYPE),
        context={"source": config_entries.SOURCE_USER},
    )
    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"],
        {
            CONF_NAME: "Agent ohne Kontrolle",
            CONF_MODEL: DEFAULT_MODEL,
            CONF_LLM_HASS_API: [],
            CONF_RECOMMENDED: True,
        },
    )
    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert CONF_LLM_HASS_API not in result["data"]
    subentry_id = list(mock_config_entry.subentries)[0]

    result = await mock_config_entry.start_subentry_reconfigure_flow(
        hass, subentry_id
    )
    assert result["step_id"] == "init"
    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"],
        {
            CONF_MODEL: DEFAULT_MODEL,
            CONF_LLM_HASS_API: [],
            CONF_RECOMMENDED: True,
        },
    )
    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "reconfigure_successful"
    subentry = mock_config_entry.subentries[subentry_id]
    assert CONF_LLM_HASS_API not in subentry.data


async def test_conversation_subentry_reconfigure_default_omits_llm_hass_api(
    hass: HomeAssistant, mock_openai: MagicMock, mock_config_entry
) -> None:
    """The reconfigure result must never re-add llm_hass_api unless it was
    explicitly submitted with a non-empty value."""
    await setup_integration(hass, mock_config_entry)

    result = await hass.config_entries.subentries.async_init(
        (mock_config_entry.entry_id, CONVERSATION_SUBENTRY_TYPE),
        context={"source": config_entries.SOURCE_USER},
    )
    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"],
        {
            CONF_NAME: "Agent",
            CONF_MODEL: DEFAULT_MODEL,
            CONF_LLM_HASS_API: [],
            CONF_RECOMMENDED: True,
        },
    )
    subentry_id = list(mock_config_entry.subentries)[0]

    result = await mock_config_entry.start_subentry_reconfigure_flow(
        hass, subentry_id
    )
    assert result["step_id"] == "init"

    # Inspect the form's default for llm_hass_api: it must be empty, not
    # pre-filled with Assist, since the subentry has no stored key.
    schema = result["data_schema"].schema
    llm_hass_api_default = next(
        key.default() for key in schema if str(key) == "llm_hass_api"
    )
    assert llm_hass_api_default == []

    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"],
        {
            CONF_MODEL: DEFAULT_MODEL,
            CONF_RECOMMENDED: True,
        },
    )
    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "reconfigure_successful"
    subentry = mock_config_entry.subentries[subentry_id]
    assert CONF_LLM_HASS_API not in subentry.data


async def test_ai_task_subentry_recommended(
    hass: HomeAssistant, mock_openai: MagicMock, mock_config_entry
) -> None:
    await setup_integration(hass, mock_config_entry)

    result = await hass.config_entries.subentries.async_init(
        (mock_config_entry.entry_id, AI_TASK_SUBENTRY_TYPE),
        context={"source": config_entries.SOURCE_USER},
    )
    assert result["type"] is FlowResultType.FORM

    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"],
        {
            CONF_NAME: "Meine Task",
            CONF_MODEL: DEFAULT_MODEL,
            CONF_RECOMMENDED: True,
        },
    )
    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["title"] == "Meine Task"
    assert result["data"] == {CONF_MODEL: DEFAULT_MODEL, CONF_RECOMMENDED: True}


async def test_ai_task_subentry_advanced(
    hass: HomeAssistant, mock_openai: MagicMock, mock_config_entry
) -> None:
    await setup_integration(hass, mock_config_entry)

    result = await hass.config_entries.subentries.async_init(
        (mock_config_entry.entry_id, AI_TASK_SUBENTRY_TYPE),
        context={"source": config_entries.SOURCE_USER},
    )
    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"],
        {
            CONF_NAME: "Task",
            CONF_MODEL: DEFAULT_MODEL,
            CONF_RECOMMENDED: False,
        },
    )
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "advanced"

    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"],
        {CONF_MAX_TOKENS: 16000, CONF_TEMPERATURE: 0.2, CONF_TOP_P: 1.0},
    )
    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["data"][CONF_MAX_TOKENS] == 16000
    assert "prompt" not in result["data"]
