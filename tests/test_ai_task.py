"""Tests for the noris AI task entity."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock

from openai import BadRequestError
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
import pytest
import voluptuous as vol

from homeassistant.components import ai_task
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError

from .conftest import setup_integration

ENTITY_ID = "ai_task.test_task"

# A structure exactly as an automation hands it to the ai_task.generate_data
# action. Home Assistant's service schema turns every field into a selector
# before the task reaches the entity, so this is the shape real callers send -
# unlike a hand-built schema of plain Python types.
ACTION_STRUCTURE = {
    "comfort_level": {
        "selector": {"select": {"options": ["cold", "warm"]}},
        "required": True,
    },
    "suggestion": {"selector": {"text": {}}},
}


async def _call_generate_data_action(hass: HomeAssistant) -> dict[str, Any]:
    """Run ai_task.generate_data with a structure, the way an automation does."""
    return await hass.services.async_call(
        "ai_task",
        "generate_data",
        {
            "task_name": "comfort",
            "entity_id": ENTITY_ID,
            "instructions": "Bewerte das Raumklima",
            "structure": ACTION_STRUCTURE,
        },
        blocking=True,
        return_response=True,
    )


def _completion(content: str) -> ChatCompletion:
    return ChatCompletion(
        id="c",
        created=0,
        model="test",
        object="chat.completion",
        choices=[
            Choice(
                index=0,
                finish_reason="stop",
                message=ChatCompletionMessage(role="assistant", content=content),
            )
        ],
    )


async def test_generate_text(
    hass: HomeAssistant, mock_openai: MagicMock, mock_config_entry_with_ai_task
) -> None:
    await setup_integration(hass, mock_config_entry_with_ai_task)
    mock_openai.chat.completions.create = AsyncMock(
        return_value=_completion("Freitext-Ergebnis")
    )

    result = await ai_task.async_generate_data(
        hass, task_name="t", entity_id=ENTITY_ID, instructions="Sag was"
    )

    assert result.data == "Freitext-Ergebnis"
    kwargs = mock_openai.chat.completions.create.call_args.kwargs
    assert kwargs["max_tokens"] == 8000
    assert "stream" not in kwargs


async def test_generate_structured_data(
    hass: HomeAssistant, mock_openai: MagicMock, mock_config_entry_with_ai_task
) -> None:
    await setup_integration(hass, mock_config_entry_with_ai_task)
    mock_openai.chat.completions.create = AsyncMock(
        return_value=_completion('```json\n{"name": "Flo"}\n```')
    )

    result = await ai_task.async_generate_data(
        hass,
        task_name="t",
        entity_id=ENTITY_ID,
        instructions="Extrahiere",
        structure=vol.Schema({vol.Required("name"): str}),
    )

    assert result.data == {"name": "Flo"}
    kwargs = mock_openai.chat.completions.create.call_args.kwargs
    assert kwargs["response_format"]["type"] == "json_schema"


async def test_generate_data_action_sends_structure_schema(
    hass: HomeAssistant, mock_openai: MagicMock, mock_config_entry_with_ai_task
) -> None:
    """The fields of an action's structure reach the model as a JSON schema."""
    await setup_integration(hass, mock_config_entry_with_ai_task)
    mock_openai.chat.completions.create = AsyncMock(
        return_value=_completion('{"comfort_level": "cold"}')
    )

    result = await _call_generate_data_action(hass)

    assert result["data"] == {"comfort_level": "cold"}
    kwargs = mock_openai.chat.completions.create.call_args.kwargs
    schema = kwargs["response_format"]["json_schema"]["schema"]
    assert set(schema.get("properties", {})) == {"comfort_level", "suggestion"}
    assert schema["properties"]["comfort_level"]["enum"] == ["cold", "warm"]
    # Optional fields stay optional: only the field marked required is listed.
    assert schema["required"] == ["comfort_level"]


async def test_structured_fallback_without_response_format(
    hass: HomeAssistant, mock_openai: MagicMock, mock_config_entry_with_ai_task
) -> None:
    """BadRequest on response_format triggers a prompt-based retry."""
    await setup_integration(hass, mock_config_entry_with_ai_task)
    response = MagicMock()
    response.status_code = 400
    mock_openai.chat.completions.create = AsyncMock(
        side_effect=[
            BadRequestError(
                message="response_format not supported",
                response=response,
                body=None,
            ),
            _completion('{"name": "Flo"}'),
        ]
    )

    result = await ai_task.async_generate_data(
        hass,
        task_name="t",
        entity_id=ENTITY_ID,
        instructions="Extrahiere",
        structure=vol.Schema({vol.Required("name"): str}),
    )

    assert result.data == {"name": "Flo"}
    assert mock_openai.chat.completions.create.await_count == 2
    retry_kwargs = mock_openai.chat.completions.create.call_args.kwargs
    assert "response_format" not in retry_kwargs
    assert any(
        m["role"] == "system" and "JSON" in str(m["content"])
        for m in retry_kwargs["messages"]
    )


async def test_structured_fallback_prompt_describes_structure_fields(
    hass: HomeAssistant, mock_openai: MagicMock, mock_config_entry_with_ai_task
) -> None:
    """The prompt-based retry names the fields and options of the structure."""
    await setup_integration(hass, mock_config_entry_with_ai_task)
    response = MagicMock()
    response.status_code = 400
    mock_openai.chat.completions.create = AsyncMock(
        side_effect=[
            BadRequestError(
                message="response_format not supported",
                response=response,
                body=None,
            ),
            _completion('{"comfort_level": "cold"}'),
        ]
    )

    result = await _call_generate_data_action(hass)

    assert result["data"] == {"comfort_level": "cold"}
    retry_kwargs = mock_openai.chat.completions.create.call_args.kwargs
    retry_prompt = retry_kwargs["messages"][-1]["content"]
    for expected in ('"comfort_level"', '"suggestion"', '"cold"', '"warm"'):
        assert expected in retry_prompt


async def test_invalid_json_raises(
    hass: HomeAssistant, mock_openai: MagicMock, mock_config_entry_with_ai_task
) -> None:
    await setup_integration(hass, mock_config_entry_with_ai_task)
    mock_openai.chat.completions.create = AsyncMock(
        return_value=_completion("kein json")
    )

    with pytest.raises(HomeAssistantError, match="parsing"):
        await ai_task.async_generate_data(
            hass,
            task_name="t",
            entity_id=ENTITY_ID,
            instructions="Extrahiere",
            structure=vol.Schema({vol.Required("name"): str}),
        )


async def test_length_finish_without_output_raises(
    hass: HomeAssistant, mock_openai: MagicMock, mock_config_entry_with_ai_task
) -> None:
    """A truncated, empty completion must raise a max_tokens hint instead of
    silently returning no data."""
    await setup_integration(hass, mock_config_entry_with_ai_task)
    truncated = ChatCompletion(
        id="c",
        created=0,
        model="test",
        object="chat.completion",
        choices=[
            Choice(
                index=0,
                finish_reason="length",
                message=ChatCompletionMessage(role="assistant", content=None),
            )
        ],
    )
    mock_openai.chat.completions.create = AsyncMock(return_value=truncated)

    with pytest.raises(HomeAssistantError, match="max_tokens"):
        await ai_task.async_generate_data(
            hass, task_name="t", entity_id=ENTITY_ID, instructions="Sag was"
        )


async def test_bad_request_without_structure_does_not_retry(
    hass: HomeAssistant, mock_openai: MagicMock, mock_config_entry_with_ai_task
) -> None:
    """A BadRequestError on a free-text task propagates without fallback retry."""
    await setup_integration(hass, mock_config_entry_with_ai_task)
    response = MagicMock()
    response.status_code = 400
    mock_openai.chat.completions.create = AsyncMock(
        side_effect=BadRequestError(message="bad request", response=response, body=None)
    )

    with pytest.raises(HomeAssistantError):
        await ai_task.async_generate_data(
            hass, task_name="t", entity_id=ENTITY_ID, instructions="Sag was"
        )

    assert mock_openai.chat.completions.create.await_count == 1
