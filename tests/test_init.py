"""Tests for noris AI setup."""

from unittest.mock import MagicMock

from openai import APIConnectionError, AuthenticationError
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.noris_ai.const import (
    AI_TASK_SUBENTRY_TYPE,
    CONVERSATION_SUBENTRY_TYPE,
    DEFAULT_CONVERSATION_NAME,
    DOMAIN,
)
from homeassistant.config_entries import ConfigEntryState, ConfigSubentryData
from homeassistant.const import CONF_API_KEY, CONF_MODEL
from homeassistant.core import HomeAssistant

from .conftest import setup_integration


async def test_setup_and_unload(
    hass: HomeAssistant, mock_openai: MagicMock, mock_config_entry
) -> None:
    """Entry sets up, validates the key, and unloads cleanly."""
    assert await setup_integration(hass, mock_config_entry)
    assert mock_config_entry.state is ConfigEntryState.LOADED
    mock_openai.models.list.assert_awaited_once()

    await hass.config_entries.async_unload(mock_config_entry.entry_id)
    await hass.async_block_till_done()
    assert mock_config_entry.state is ConfigEntryState.NOT_LOADED


async def test_setup_invalid_auth_sets_setup_error(
    hass: HomeAssistant, mock_openai: MagicMock, mock_config_entry
) -> None:
    """A 401 during setup puts the entry into setup error."""
    mock_openai.models.list.side_effect = AuthenticationError(
        message="401", response=MagicMock(), body=None
    )
    assert not await setup_integration(hass, mock_config_entry)
    assert mock_config_entry.state is ConfigEntryState.SETUP_ERROR


async def test_setup_connection_error_retries(
    hass: HomeAssistant, mock_openai: MagicMock, mock_config_entry
) -> None:
    """A connection problem leads to setup retry."""
    mock_openai.models.list.side_effect = APIConnectionError(request=MagicMock())
    assert not await setup_integration(hass, mock_config_entry)
    assert mock_config_entry.state is ConfigEntryState.SETUP_RETRY


async def test_migrate_entry_retitles_model_id_subentries(
    hass: HomeAssistant, mock_openai: MagicMock
) -> None:
    """1.1 entries get default titles for subentries still named after a model.

    A title the user chose themselves must survive the migration untouched.
    """
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="noris AI",
        data={CONF_API_KEY: "sk-bf-test"},
        version=1,
        minor_version=1,
        subentries_data=[
            ConfigSubentryData(
                data={CONF_MODEL: "vllm/gpt-oss-120b"},
                subentry_type=CONVERSATION_SUBENTRY_TYPE,
                title="vllm/gpt-oss-120b",
                unique_id=None,
            ),
            ConfigSubentryData(
                data={CONF_MODEL: "vllm/gpt-oss-120b"},
                subentry_type=AI_TASK_SUBENTRY_TYPE,
                title="Mein Extraktor",
                unique_id=None,
            ),
        ],
    )
    entry.add_to_hass(hass)

    assert await setup_integration(hass, entry)

    assert entry.minor_version == 2
    titles = {s.subentry_type: s.title for s in entry.subentries.values()}
    assert titles[CONVERSATION_SUBENTRY_TYPE] == DEFAULT_CONVERSATION_NAME
    assert titles[AI_TASK_SUBENTRY_TYPE] == "Mein Extraktor"
