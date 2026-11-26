"""Tests for the noris AI conversation entity."""

from collections.abc import AsyncGenerator
import json
from unittest.mock import AsyncMock, MagicMock, patch

from openai import APIStatusError
import pytest
import voluptuous as vol

from custom_components.noris_ai.const import CONF_ENABLE_REASONING, CONF_RECOMMENDED
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
        [
            _chunk("<think>kurz nachdenken</think>"),
            _chunk("Hallo!"),
            _chunk(None, finish="stop"),
        ]
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
    hass: HomeAssistant,
    mock_openai: MagicMock,
    mock_config_entry_with_conversation,
    caplog: pytest.LogCaptureFixture,
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
    mock_tool.async_call = AsyncMock(return_value=llm.ToolResult(data={"ok": True}))

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

    from homeassistant.helpers import llm as llm_helper

    real_async_get_api = llm_helper.async_get_api

    async def _api_with_tool(hass_, api_id, llm_context):
        api = await real_async_get_api(hass_, api_id, llm_context)
        api.tools = [mock_tool]
        return api

    with patch("homeassistant.helpers.llm.async_get_api", side_effect=_api_with_tool):
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
    # The follow-up request hands the tool result back as data plus error flag.
    follow_up = mock_openai.chat.completions.create.call_args.kwargs["messages"]
    tool_message = next(m for m in follow_up if m["role"] == "tool")
    assert tool_message["tool_call_id"] == "call_1"
    assert json.loads(tool_message["content"]) == {"data": {"ok": True}, "error": False}
    # Home Assistant reports the deprecated accessor through its log.
    assert "ToolResultContent.tool_result" not in caplog.text


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


async def test_agent_announces_streaming(
    hass: HomeAssistant, mock_openai: MagicMock, mock_config_entry_with_conversation
) -> None:
    """The pipeline only streams text to TTS when the agent says it can."""
    await setup_integration(hass, mock_config_entry_with_conversation)

    info = conversation.async_get_agent_info(hass, AGENT_ID)

    assert info is not None
    assert info.supports_streaming is True


async def test_reasoning_off_sends_both_switches(
    hass: HomeAssistant, mock_openai: MagicMock, mock_config_entry_with_conversation
) -> None:
    """Qwen obeys enable_thinking, gpt-oss and GLM obey reasoning_effort."""
    await setup_integration(hass, mock_config_entry_with_conversation)
    mock_openai.chat.completions.create = _stream_mock(
        [_chunk("Hallo!"), _chunk(None, finish="stop")]
    )

    await conversation.async_converse(hass, "Hallo", None, Context(), agent_id=AGENT_ID)

    kwargs = mock_openai.chat.completions.create.call_args.kwargs
    assert kwargs["reasoning_effort"] == "low"
    assert kwargs["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}


async def test_reasoning_on_sends_neither_switch(
    hass: HomeAssistant, mock_openai: MagicMock, mock_config_entry_with_conversation
) -> None:
    """With reasoning enabled the model keeps its own default."""
    subentry = next(iter(mock_config_entry_with_conversation.subentries.values()))
    hass.config_entries.async_update_subentry(
        mock_config_entry_with_conversation,
        subentry,
        data={**subentry.data, CONF_RECOMMENDED: False, CONF_ENABLE_REASONING: True},
    )
    await setup_integration(hass, mock_config_entry_with_conversation)
    mock_openai.chat.completions.create = _stream_mock(
        [_chunk("Hallo!"), _chunk(None, finish="stop")]
    )

    await conversation.async_converse(hass, "Hallo", None, Context(), agent_id=AGENT_ID)

    kwargs = mock_openai.chat.completions.create.call_args.kwargs
    assert "reasoning_effort" not in kwargs
    assert "extra_body" not in kwargs
