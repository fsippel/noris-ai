"""Tests for the noris AI conversation entity."""

from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

from openai import APIStatusError
import voluptuous as vol

from homeassistant.components import conversation
from homeassistant.core import Context, HomeAssistant
from homeassistant.helpers import intent, llm

from .conftest import setup_integration
from .test_entity_stream import _chunk

AGENT_ID = "conversation.test_agent"


def _stream_mock(chunks):
    async def _gen() -> AsyncGenerator:
        for chunk in chunks:
            yield chunk

    return AsyncMock(return_value=_gen())


async def test_simple_answer(
    hass: HomeAssistant, mock_openai: MagicMock, mock_config_entry_with_conversation
) -> None:
    await setup_integration(hass, mock_config_entry_with_conversation)
    mock_openai.chat.completions.create = _stream_mock(
        [_chunk("<think>kurz nachdenken</think>"), _chunk("Hallo!"), _chunk(None, finish="stop")]
    )

    result = await conversation.async_converse(
        hass, "Hallo", None, Context(), agent_id=AGENT_ID
    )

    assert result.response.response_type == intent.IntentResponseType.ACTION_DONE
    assert result.response.speech["plain"]["speech"] == "Hallo!"
    call_kwargs = mock_openai.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == "vllm/gpt-oss-120b"
    assert call_kwargs["max_tokens"] == 3000
    assert call_kwargs["stream"] is True


async def test_tool_call_round(
    hass: HomeAssistant, mock_openai: MagicMock, mock_config_entry_with_conversation
) -> None:
    """First response calls a tool, second one answers."""
    from openai.types.chat.chat_completion_chunk import (
        ChoiceDeltaToolCall,
        ChoiceDeltaToolCallFunction,
    )

    mock_tool = MagicMock(spec=llm.Tool)
    mock_tool.name = "test_tool"
    mock_tool.description = "Test"
    mock_tool.parameters = vol.Schema({vol.Optional("param"): str})
    mock_tool.async_call = AsyncMock(return_value={"ok": True})

    first = [
        _chunk(
            tool_calls=[
                ChoiceDeltaToolCall(
                    index=0,
                    id="call_1",
                    type="function",
                    function=ChoiceDeltaToolCallFunction(
                        name="test_tool", arguments='{"param": "x"}'
                    ),
                )
            ],
            finish="tool_calls",
        )
    ]
    second = [_chunk("Erledigt"), _chunk(None, finish="stop")]

    async def _gen(chunks):
        for chunk in chunks:
            yield chunk

    async def _fake_get_api(hass_, api_id, llm_context):
        return llm.APIInstance(
            api=MagicMock(id="assist"),
            api_prompt="Call tools when asked.",
            llm_context=llm_context,
            tools=[mock_tool],
        )

    # Patch the public helper instead of AssistAPI internals: those moved
    # between HA 2026.7 (helpers.llm) and 2026.8 (components.llm), while
    # async_get_api/APIInstance are stable across both.
    with patch(
        "homeassistant.helpers.llm.async_get_api",
        side_effect=_fake_get_api,
    ):
        await setup_integration(hass, mock_config_entry_with_conversation)
        mock_openai.chat.completions.create = AsyncMock(
            side_effect=[_gen(first), _gen(second)]
        )
        result = await conversation.async_converse(
            hass, "Mach was", None, Context(), agent_id=AGENT_ID
        )

    assert result.response.speech["plain"]["speech"] == "Erledigt"
    mock_tool.async_call.assert_awaited_once()
    assert mock_openai.chat.completions.create.await_count == 2


async def test_broken_model_reports_friendly_error(
    hass: HomeAssistant, mock_openai: MagicMock, mock_config_entry_with_conversation
) -> None:
    await setup_integration(hass, mock_config_entry_with_conversation)
    response = MagicMock()
    response.status_code = 400
    mock_openai.chat.completions.create = AsyncMock(
        side_effect=APIStatusError(
            message="no keys found that support model: gpt-oss-120b",
            response=response,
            body=None,
        )
    )

    result = await conversation.async_converse(
        hass, "Hallo", None, Context(), agent_id=AGENT_ID
    )

    assert result.response.response_type == intent.IntentResponseType.ERROR
    assert "not available" in result.response.speech["plain"]["speech"]
