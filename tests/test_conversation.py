"""Tests for the noris AI conversation agent."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.noris_ai.const import (
    CONVERSATION_SUBENTRY_TYPE,
    DEFAULT_CONVERSATION_NAME,
    DOMAIN,
)
from custom_components.noris_ai.entity import MAX_TOOL_ITERATIONS
from homeassistant.components import conversation
from homeassistant.config_entries import ConfigSubentryData
from homeassistant.const import CONF_API_KEY, CONF_LLM_HASS_API, CONF_MODEL
from homeassistant.core import Context, HomeAssistant
from homeassistant.helpers import entity_registry as er, intent, llm

from .conftest import CHAT_MODEL, setup_integration


def chat_completion(content: str | None, *, tool_calls: list[Any] | None = None) -> Any:
    """Build a minimal chat.completions response."""
    message = SimpleNamespace(
        role="assistant",
        content=content,
        tool_calls=tool_calls,
        model_extra={},
    )
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def tool_call(call_id: str, name: str, arguments: dict[str, Any]) -> Any:
    """Build a minimal Chat Completions tool-call object.

    Mirrors the shape ``_transform_response`` reads from a real
    ``ChatCompletionMessage``: ``id``, ``type`` (must be ``"function"``),
    and ``function.name``/``function.arguments`` (a JSON *string*).
    """
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )


@pytest.fixture
def mock_config_entry_with_llm_api() -> MockConfigEntry:
    """Return a config entry whose conversation subentry enables the Assist LLM API.

    The shared ``mock_config_entry`` fixture's conversation subentry carries
    only ``CONF_MODEL``, so ``chat_log.llm_api`` stays ``None`` and the model
    is never offered tools. This variant adds ``CONF_LLM_HASS_API`` so the
    tool-call loop in ``_async_handle_chat_log`` actually has an LLM API to
    call tools against.
    """
    return MockConfigEntry(
        domain=DOMAIN,
        title="noris AI",
        data={CONF_API_KEY: "sk-bf-test"},
        subentries_data=[
            ConfigSubentryData(
                data={CONF_MODEL: CHAT_MODEL, CONF_LLM_HASS_API: [llm.LLM_API_ASSIST]},
                subentry_type=CONVERSATION_SUBENTRY_TYPE,
                title=DEFAULT_CONVERSATION_NAME,
                unique_id=None,
            ),
        ],
    )


def _conversation_entities(
    hass: HomeAssistant, entry: MockConfigEntry
) -> list[er.RegistryEntry]:
    """Return this config entry's own conversation entities.

    HA's ``conversation`` component always registers a built-in
    ``conversation.home_assistant`` default-agent entity alongside any
    integration's own agent, so filtering ``hass.states`` by domain alone
    also picks that entity up. Filtering by config entry via the entity
    registry isolates the entity noris_ai's platform actually created.
    """
    registry = er.async_get(hass)
    return [
        entity
        for entity in er.async_entries_for_config_entry(registry, entry.entry_id)
        if entity.domain == "conversation"
    ]


async def test_conversation_entity_created(
    hass: HomeAssistant, mock_client: AsyncMock, mock_config_entry: MockConfigEntry
) -> None:
    """A conversation subentry produces a conversation entity."""
    await setup_integration(hass, mock_config_entry)

    assert len(_conversation_entities(hass, mock_config_entry)) == 1


async def test_conversation_answers(
    hass: HomeAssistant, mock_client: AsyncMock, mock_config_entry: MockConfigEntry
) -> None:
    """A plain question returns the model's answer as the response speech."""
    mock_client.chat.completions.create = AsyncMock(
        return_value=chat_completion("Das Licht im Wohnzimmer ist an.")
    )
    await setup_integration(hass, mock_config_entry)
    entity_id = _conversation_entities(hass, mock_config_entry)[0].entity_id

    result = await conversation.async_converse(
        hass, "Ist das Licht an?", None, Context(), agent_id=entity_id
    )

    assert (
        result.response.speech["plain"]["speech"] == "Das Licht im Wohnzimmer ist an."
    )


async def test_conversation_reports_api_error(
    hass: HomeAssistant, mock_client: AsyncMock, mock_config_entry: MockConfigEntry
) -> None:
    """An empty choices list surfaces as an error response, not a crash."""
    mock_client.chat.completions.create = AsyncMock(
        return_value=SimpleNamespace(choices=[])
    )
    await setup_integration(hass, mock_config_entry)
    entity_id = _conversation_entities(hass, mock_config_entry)[0].entity_id

    result = await conversation.async_converse(
        hass, "Ist das Licht an?", None, Context(), agent_id=entity_id
    )

    assert result.response.response_type == intent.IntentResponseType.ERROR


async def test_conversation_tool_call_drives_second_round_trip(
    hass: HomeAssistant,
    mock_client: AsyncMock,
    mock_config_entry_with_llm_api: MockConfigEntry,
) -> None:
    """A tool call in the first response drives a second round trip to the model.

    This is the test that proves the loop loops at all: without the
    ``if not chat_log.unresponded_tool_results: break`` logic re-calling the
    API after a tool result, the model would only ever be called once and
    the final speech would be the (empty) content of the first response
    instead of the second answer.
    """
    mock_client.chat.completions.create = AsyncMock(
        side_effect=[
            chat_completion(
                None,
                tool_calls=[tool_call("call_1", "not_a_real_tool", {"foo": "bar"})],
            ),
            chat_completion("Das Licht im Wohnzimmer ist an."),
        ]
    )
    await setup_integration(hass, mock_config_entry_with_llm_api)
    entity_id = _conversation_entities(hass, mock_config_entry_with_llm_api)[
        0
    ].entity_id

    result = await conversation.async_converse(
        hass, "Schalte das Licht im Wohnzimmer an", None, Context(), agent_id=entity_id
    )

    assert mock_client.chat.completions.create.await_count == 2
    assert (
        result.response.speech["plain"]["speech"] == "Das Licht im Wohnzimmer ist an."
    )


async def test_conversation_tool_call_loop_is_bounded(
    hass: HomeAssistant,
    mock_client: AsyncMock,
    mock_config_entry_with_llm_api: MockConfigEntry,
) -> None:
    """A model that always answers with a tool call stops after MAX_TOOL_ITERATIONS.

    ``MAX_TOOL_ITERATIONS`` is imported from ``entity`` rather than
    hardcoded so this test tracks the constant if it ever changes.
    """
    call_count = 0

    def _always_tool_call(**_kwargs: Any) -> Any:
        """Return a response that always asks for another (bogus) tool call."""
        nonlocal call_count
        call_count += 1
        return chat_completion(
            None,
            tool_calls=[tool_call(f"call_{call_count}", "not_a_real_tool", {})],
        )

    mock_client.chat.completions.create = AsyncMock(side_effect=_always_tool_call)
    await setup_integration(hass, mock_config_entry_with_llm_api)
    entity_id = _conversation_entities(hass, mock_config_entry_with_llm_api)[
        0
    ].entity_id

    await conversation.async_converse(
        hass, "Schalte das Licht im Wohnzimmer an", None, Context(), agent_id=entity_id
    )

    assert mock_client.chat.completions.create.await_count == MAX_TOOL_ITERATIONS
