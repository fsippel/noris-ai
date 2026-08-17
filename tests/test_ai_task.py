"""Tests for the noris AI AI Task entity."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry
import voluptuous as vol

from homeassistant.components import ai_task
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import entity_registry as er

from .conftest import setup_integration
from .test_conversation import chat_completion


def _task_entity_id(hass: HomeAssistant, entry: MockConfigEntry) -> str:
    """Return the single AI task entity id belonging to this config entry.

    Resolved through the entity registry rather than ``hass.states``, the
    same pattern used by ``test_conversation.py``'s ``_conversation_entities``
    (isolates the entity noris_ai's platform actually created).
    """
    registry = er.async_get(hass)
    return next(
        entity.entity_id
        for entity in er.async_entries_for_config_entry(registry, entry.entry_id)
        if entity.domain == "ai_task"
    )


async def test_generate_free_text(
    hass: HomeAssistant, mock_client: AsyncMock, mock_config_entry: MockConfigEntry
) -> None:
    """Free-text generation returns the raw model output."""
    mock_client.chat.completions.create = AsyncMock(
        return_value=chat_completion("Es sind 21 Grad.")
    )
    await setup_integration(hass, mock_config_entry)

    result = await ai_task.async_generate_data(
        hass,
        task_name="wetter",
        entity_id=_task_entity_id(hass, mock_config_entry),
        instructions="Wie warm ist es?",
    )

    assert result.data == "Es sind 21 Grad."


async def test_generate_structured_data(
    hass: HomeAssistant, mock_client: AsyncMock, mock_config_entry: MockConfigEntry
) -> None:
    """A structure request parses the JSON response into a dict."""
    mock_client.chat.completions.create = AsyncMock(
        return_value=chat_completion('{"temperatur": 21}')
    )
    await setup_integration(hass, mock_config_entry)

    result = await ai_task.async_generate_data(
        hass,
        task_name="wetter",
        entity_id=_task_entity_id(hass, mock_config_entry),
        instructions="Wie warm ist es?",
        structure=vol.Schema({vol.Required("temperatur"): int}),
    )

    assert result.data == {"temperatur": 21}
    # The mock returns valid JSON regardless of what was requested, so
    # without this the `if structure:` branch that builds response_format
    # in entity.py could be deleted outright and this test would still pass.
    response_format = mock_client.chat.completions.create.call_args.kwargs[
        "response_format"
    ]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["name"] == "wetter"
    assert response_format["json_schema"]["strict"] is True
    assert "temperatur" in response_format["json_schema"]["schema"]["properties"]


async def test_generate_structured_data_invalid_json(
    hass: HomeAssistant, mock_client: AsyncMock, mock_config_entry: MockConfigEntry
) -> None:
    """Unparseable structured output raises a clear error."""
    mock_client.chat.completions.create = AsyncMock(
        return_value=chat_completion("nicht json")
    )
    await setup_integration(hass, mock_config_entry)

    with pytest.raises(HomeAssistantError, match="Error parsing structured response"):
        await ai_task.async_generate_data(
            hass,
            task_name="wetter",
            entity_id=_task_entity_id(hass, mock_config_entry),
            instructions="Wie warm ist es?",
            structure=vol.Schema({vol.Required("temperatur"): int}),
        )
