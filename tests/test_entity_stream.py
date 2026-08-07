"""Tests for the streaming transform."""

from collections.abc import AsyncGenerator

from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import (
    Choice,
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)
import pytest

from homeassistant.exceptions import HomeAssistantError

from custom_components.noris_ai.entity import (
    _decode_tool_arguments,
    _ThinkTagFilter,
    _transform_stream,
)


def _chunk(
    content: str | None = None,
    finish: str | None = None,
    tool_calls: list[ChoiceDeltaToolCall] | None = None,
    **delta_extra,
) -> ChatCompletionChunk:
    return ChatCompletionChunk(
        id="chunk",
        created=0,
        model="test",
        object="chat.completion.chunk",
        choices=[
            Choice(
                index=0,
                delta=ChoiceDelta(
                    content=content, tool_calls=tool_calls, **delta_extra
                ),
                finish_reason=finish,
            )
        ],
    )


async def _stream(chunks) -> AsyncGenerator[ChatCompletionChunk]:
    for chunk in chunks:
        yield chunk


async def _collect(chunks) -> list[dict]:
    return [d async for d in _transform_stream(_stream(chunks))]


def test_think_filter_passthrough() -> None:
    f = _ThinkTagFilter()
    assert f.push("Hallo Welt") == ("Hallo Welt", "")
    assert f.flush() == ("", "")


def test_think_filter_single_chunk() -> None:
    f = _ThinkTagFilter()
    text, think = f.push("<think>grübel</think>Antwort")
    assert think == "grübel"
    assert text == "Antwort"


def test_think_filter_split_across_chunks() -> None:
    f = _ThinkTagFilter()
    collected_text = ""
    collected_think = ""
    for part in ["<thi", "nk>grü", "bel</th", "ink>Ant", "wort"]:
        text, think = f.push(part)
        collected_text += text
        collected_think += think
    text, think = f.flush()
    collected_text += text
    collected_think += think
    assert collected_think == "grübel"
    assert collected_text == "Antwort"


def test_think_filter_unclosed_tag_flushes_as_thinking() -> None:
    f = _ThinkTagFilter()
    f.push("<think>nur gedacht")
    text, think = f.flush()
    assert text == ""
    assert think == "nur gedacht"


async def test_stream_content() -> None:
    deltas = await _collect(
        [_chunk("Hal"), _chunk("lo"), _chunk(None, finish="stop")]
    )
    assert deltas[0] == {"role": "assistant"}
    assert "".join(d.get("content", "") for d in deltas) == "Hallo"


async def test_stream_reasoning_extra_becomes_thinking() -> None:
    deltas = await _collect(
        [
            _chunk(None, **{"reasoning": "hmm"}),
            _chunk("OK", finish="stop"),
        ]
    )
    thinking = "".join(d.get("thinking_content", "") for d in deltas)
    content = "".join(d.get("content", "") for d in deltas)
    assert thinking == "hmm"
    assert content == "OK"


async def test_stream_tool_calls_assembled() -> None:
    deltas = await _collect(
        [
            _chunk(
                tool_calls=[
                    ChoiceDeltaToolCall(
                        index=0,
                        id="call_1",
                        type="function",
                        function=ChoiceDeltaToolCallFunction(
                            name="test_tool", arguments='{"par'
                        ),
                    )
                ]
            ),
            _chunk(
                tool_calls=[
                    ChoiceDeltaToolCall(
                        index=0,
                        function=ChoiceDeltaToolCallFunction(arguments='am": 1}'),
                    )
                ],
                finish="tool_calls",
            ),
        ]
    )
    tool_calls = next(d["tool_calls"] for d in deltas if "tool_calls" in d)
    assert tool_calls[0].tool_name == "test_tool"
    assert tool_calls[0].tool_args == {"param": 1}
    assert tool_calls[0].id == "call_1"


async def test_stream_length_without_output_raises() -> None:
    with pytest.raises(HomeAssistantError, match="max_tokens"):
        await _collect([_chunk(None, finish="length")])


async def test_stream_tool_call_without_arguments_yields_empty_args() -> None:
    """A tool call with a name/id but no argument fragments must decode to
    an empty dict instead of raising a JSON decode error."""
    deltas = await _collect(
        [
            _chunk(
                tool_calls=[
                    ChoiceDeltaToolCall(
                        index=0,
                        id="call_1",
                        type="function",
                        function=ChoiceDeltaToolCallFunction(name="test_tool"),
                    )
                ],
                finish="tool_calls",
            ),
        ]
    )
    tool_calls = next(d["tool_calls"] for d in deltas if "tool_calls" in d)
    assert tool_calls[0].tool_name == "test_tool"
    assert tool_calls[0].tool_args == {}
    assert tool_calls[0].id == "call_1"


def test_decode_tool_arguments_plain() -> None:
    """Plain JSON arguments decode as-is; empty means no arguments."""
    assert _decode_tool_arguments('{"city": "Nürnberg"}') == {"city": "Nürnberg"}
    assert _decode_tool_arguments("") == {}
    assert _decode_tool_arguments("  ") == {}


def test_decode_tool_arguments_strips_code_fence() -> None:
    """Some models wrap tool arguments in Markdown fences (e.g. GLM)."""
    fenced = '```json\n{"city": "Nürnberg"}\n```'
    assert _decode_tool_arguments(fenced) == {"city": "Nürnberg"}
    fenced_no_lang = '```\n{"on": true}\n```'
    assert _decode_tool_arguments(fenced_no_lang) == {"on": True}


def test_decode_tool_arguments_invalid_raises() -> None:
    """Garbage that is not JSON even after stripping raises."""
    with pytest.raises(HomeAssistantError, match="Unexpected tool argument"):
        _decode_tool_arguments("```\nnot json\n```")
    with pytest.raises(HomeAssistantError, match="Unexpected tool argument"):
        _decode_tool_arguments("not json")
