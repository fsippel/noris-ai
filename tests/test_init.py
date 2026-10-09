"""Tests for noris AI setup."""

from unittest.mock import MagicMock

from openai import APIConnectionError, AuthenticationError
import pytest

from homeassistant.config_entries import ConfigEntryState
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
    mock_openai.models.list.side_effect = APIConnectionError(
        request=MagicMock()
    )
    assert not await setup_integration(hass, mock_config_entry)
    assert mock_config_entry.state is ConfigEntryState.SETUP_RETRY
