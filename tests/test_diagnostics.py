"""Tests for noris AI diagnostics."""

from unittest.mock import MagicMock

from custom_components.noris_ai.diagnostics import async_get_config_entry_diagnostics
from homeassistant.const import CONF_API_KEY
from homeassistant.core import HomeAssistant

from .conftest import setup_integration


async def test_api_key_is_redacted(
    hass: HomeAssistant, mock_openai: MagicMock, mock_config_entry_with_conversation
) -> None:
    await setup_integration(hass, mock_config_entry_with_conversation)

    result = await async_get_config_entry_diagnostics(
        hass, mock_config_entry_with_conversation
    )

    assert result["entry_data"][CONF_API_KEY] == "**REDACTED**"
    assert result["subentries"][0]["type"] == "conversation"
