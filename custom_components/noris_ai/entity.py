"""Base entity and streaming helpers for noris AI."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Callable
import json
import re
from typing import Any, Literal, NoReturn

import openai
from openai import AsyncStream
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionChunk,
    ChatCompletionFunctionToolParam,
    ChatCompletionMessage,
    ChatCompletionMessageFunctionToolCallParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionUserMessageParam,
)
from openai.types.chat.chat_completion_message_function_tool_call_param import (
    Function,
)
from openai.types.shared_params import FunctionDefinition
import voluptuous as vol
from voluptuous_openapi import convert

from homeassistant.components import conversation
from homeassistant.config_entries import ConfigSubentry
from homeassistant.const import CONF_MODEL
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import llm
from homeassistant.helpers.entity import Entity
from homeassistant.helpers.json import json_dumps

from . import NorisAIConfigEntry
from .const import (
    CONF_MAX_TOKENS,
    CONF_RECOMMENDED,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    DOMAIN,
    LOGGER,
    RECOMMENDED_TEMPERATURE,
    RECOMMENDED_TOP_P,
)

MAX_TOOL_ITERATIONS = 10

_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


class _ThinkTagFilter:
    """Route ``<think>...</think>`` spans to thinking content.

    Tags may be split across streaming chunks, so a small buffer holds any
    text that could still turn out to be a partial tag.
    """

    def __init__(self) -> None:
        self._buf = ""
        self._in_think = False

    def push(self, text: str) -> tuple[str, str]:
        """Feed text in; return (visible_text, thinking_text)."""
        self._buf += text
        out: list[str] = []
        think: list[str] = []
        while True:
            tag = _THINK_CLOSE if self._in_think else _THINK_OPEN
            target = think if self._in_think else out
            idx = self._buf.find(tag)
            if idx == -1:
                if not self._in_think:
                    keep = self._partial_tag_suffix(tag)
                    cut = len(self._buf) - keep
                    target.append(self._buf[:cut])
                    self._buf = self._buf[cut:]
                break
            target.append(self._buf[:idx])
            self._buf = self._buf[idx + len(tag) :]
            self._in_think = not self._in_think
        return "".join(out), "".join(think)

    def _partial_tag_suffix(self, tag: str) -> int:
        """Length of a buffer suffix that is a prefix of ``tag``."""
        for length in range(min(len(tag) - 1, len(self._buf)), 0, -1):
            if tag.startswith(self._buf[-length:]):
                return length
        return 0

    def flush(self) -> tuple[str, str]:
        """Return any remaining buffered text."""
        buf, self._buf = self._buf, ""
        if self._in_think:
            return "", buf
        return buf, ""


_CODE_FENCE_PATTERN = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


def _strip_code_fences(text: str) -> str:
    """Remove a surrounding markdown code fence, if present."""
    if match := _CODE_FENCE_PATTERN.match(text.strip()):
        return match.group(1)
    return text


def _decode_tool_arguments(arguments: str) -> Any:
    """Decode tool call arguments.

    Tolerant of models (e.g. GLM) that wrap the JSON in a Markdown code
    fence despite the schema constraint.
    """
    if not arguments.strip():
        return {}
    try:
        return json.loads(arguments)
    except json.JSONDecodeError as err:
        cleaned = _strip_code_fences(arguments)
        if cleaned != arguments:
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                pass
        raise HomeAssistantError(
            f"Unexpected tool argument response: {err}"
        ) from err


async def _transform_stream(
    stream: AsyncStream[ChatCompletionChunk] | AsyncGenerator[ChatCompletionChunk],
) -> AsyncGenerator[conversation.AssistantContentDeltaDict]:
    """Convert a chat-completion chunk stream to ChatLog deltas."""
    yield {"role": "assistant"}

    think = _ThinkTagFilter()
    tool_calls: dict[int, dict[str, str]] = {}
    finish_reason: str | None = None
    had_output = False

    try:
        async for chunk in stream:
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            if choice.finish_reason:
                finish_reason = choice.finish_reason
            delta = choice.delta
            if delta is None:
                continue

            extras = delta.model_extra or {}
            for key in ("reasoning", "reasoning_content"):
                value = extras.get(key)
                if isinstance(value, str) and value:
                    yield {"thinking_content": value}

            if delta.content:
                text, thinking = think.push(delta.content)
                if thinking:
                    yield {"thinking_content": thinking}
                if text:
                    had_output = True
                    yield {"content": text}

            for tc in delta.tool_calls or ():
                slot = tool_calls.setdefault(
                    tc.index, {"id": "", "name": "", "args": ""}
                )
                if tc.id:
                    slot["id"] = tc.id
                if tc.function:
                    if tc.function.name:
                        slot["name"] += tc.function.name
                    if tc.function.arguments:
                        slot["args"] += tc.function.arguments
    except openai.OpenAIError as err:
        LOGGER.error("Stream error from API: %s", err)
        raise HomeAssistantError("Error talking to ai.noris.de") from err

    text, thinking = think.flush()
    if thinking:
        yield {"thinking_content": thinking}
    if text:
        had_output = True
        yield {"content": text}

    if tool_calls:
        had_output = True
        yield {
            "tool_calls": [
                llm.ToolInput(
                    id=slot["id"],
                    tool_name=slot["name"],
                    tool_args=_decode_tool_arguments(slot["args"]),
                )
                for _, slot in sorted(tool_calls.items())
            ]
        }

    if finish_reason == "length" and not had_output:
        raise HomeAssistantError(
            "Token limit reached before the model produced an answer; "
            "increase max_tokens"
        )


def _format_tool(
    tool: llm.Tool,
    custom_serializer: Callable[[Any], Any] | None,
) -> ChatCompletionFunctionToolParam:
    """Format tool specification."""
    tool_spec = FunctionDefinition(
        name=tool.name,
        parameters=convert(tool.parameters, custom_serializer=custom_serializer),
    )
    if tool.description:
        tool_spec["description"] = tool.description
    return ChatCompletionFunctionToolParam(type="function", function=tool_spec)


def _convert_content_to_chat_message(
    content: conversation.Content,
) -> ChatCompletionMessageParam | None:
    """Convert a ChatLog message to the Completions API format."""
    if isinstance(content, conversation.ToolResultContent):
        return ChatCompletionToolMessageParam(
            role="tool",
            tool_call_id=content.tool_call_id,
            content=json_dumps(content.tool_result),
        )

    role: Literal["user", "assistant", "system"] = content.role
    if role == "system" and content.content:
        return ChatCompletionSystemMessageParam(role="system", content=content.content)

    if role == "user" and content.content:
        return ChatCompletionUserMessageParam(role="user", content=content.content)

    if role == "assistant":
        param = ChatCompletionAssistantMessageParam(
            role="assistant",
            content=content.content,
        )
        if isinstance(content, conversation.AssistantContent) and content.tool_calls:
            param["tool_calls"] = [
                ChatCompletionMessageFunctionToolCallParam(
                    type="function",
                    id=tool_call.id,
                    function=Function(
                        arguments=json_dumps(tool_call.tool_args),
                        name=tool_call.tool_name,
                    ),
                )
                for tool_call in content.tool_calls
            ]
        return param
    LOGGER.warning("Could not convert message to Completions API: %s", content)
    return None


def _raise_gateway_error(err: openai.OpenAIError, model: str) -> NoReturn:
    """Map gateway errors to user-friendly HomeAssistantErrors."""
    if "no keys found" in str(err):
        raise HomeAssistantError(
            f"The model {model} is currently not available on ai.noris.de"
        ) from err
    LOGGER.error("Error talking to API: %s", err)
    raise HomeAssistantError("Error talking to ai.noris.de") from err


def _format_structured_output(name: str, structure: vol.Schema) -> dict[str, Any]:
    """Build a json_schema response_format (fully wired up in the AI task)."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name or "data",
            "schema": convert(structure),
            "strict": True,
        },
    }


_THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def _split_thinking(content: str | None) -> tuple[str | None, str | None]:
    """Return (cleaned_content, thinking) extracted from ``<think>`` tags."""
    if not content:
        return content, None
    thinking_parts = [m.group(1).strip() for m in _THINK_PATTERN.finditer(content)]
    if not thinking_parts:
        return content, None
    cleaned = _THINK_PATTERN.sub("", content).strip() or None
    thinking = "\n\n".join(part for part in thinking_parts if part) or None
    return cleaned, thinking


def _extract_thinking(
    message: ChatCompletionMessage,
) -> tuple[str | None, str | None]:
    """Split reasoning from an assistant message (non-streaming path)."""
    extras = message.model_extra or {}
    for key in ("reasoning", "reasoning_content"):
        value = extras.get(key)
        if isinstance(value, str) and value.strip():
            return message.content, value.strip()
    return _split_thinking(message.content)


class NorisAIEntity(Entity):
    """Base entity for noris AI."""

    _attr_has_entity_name = True
    _recommended_max_tokens: int

    def __init__(self, entry: NorisAIConfigEntry, subentry: ConfigSubentry) -> None:
        """Initialize the entity."""
        self.entry = entry
        self.subentry = subentry
        self.model = subentry.data[CONF_MODEL]
        self._attr_unique_id = subentry.subentry_id
        self._attr_device_info = dr.DeviceInfo(
            identifiers={(DOMAIN, subentry.subentry_id)},
            name=subentry.title,
            manufacturer="noris network AG",
            model=self.model,
            entry_type=dr.DeviceEntryType.SERVICE,
        )

    def _get_option(self, key: str, default: Any) -> Any:
        """Subentry option, unless recommended settings are active."""
        if self.subentry.data.get(CONF_RECOMMENDED, False):
            return default
        return self.subentry.data.get(key, default)

    async def _async_handle_chat_log(
        self,
        chat_log: conversation.ChatLog,
        *,
        stream: bool = True,
        structure_name: str | None = None,
        structure: vol.Schema | None = None,
    ) -> None:
        """Generate an answer for the chat log."""
        model_args: dict[str, Any] = {
            "model": self.model,
            "max_tokens": int(
                self._get_option(CONF_MAX_TOKENS, self._recommended_max_tokens)
            ),
            "temperature": float(
                self._get_option(CONF_TEMPERATURE, RECOMMENDED_TEMPERATURE)
            ),
            "top_p": float(self._get_option(CONF_TOP_P, RECOMMENDED_TOP_P)),
            "user": chat_log.conversation_id,
        }

        if chat_log.llm_api:
            model_args["tools"] = [
                _format_tool(tool, chat_log.llm_api.custom_serializer)
                for tool in chat_log.llm_api.tools
            ]

        if structure:
            model_args["response_format"] = _format_structured_output(
                structure_name or "data", structure
            )

        model_args["messages"] = [
            m
            for content in chat_log.content
            if (m := _convert_content_to_chat_message(content))
        ]

        client = self.entry.runtime_data

        for _iteration in range(MAX_TOOL_ITERATIONS):
            if stream:
                try:
                    result_stream = await client.chat.completions.create(
                        stream=True, **model_args
                    )
                except openai.OpenAIError as err:
                    _raise_gateway_error(err, self.model)
                deltas = _transform_stream(result_stream)
            else:
                deltas = self._non_streaming_deltas(client, model_args)

            model_args["messages"].extend(
                [
                    msg
                    async for content in chat_log.async_add_delta_content_stream(
                        self.entity_id, deltas
                    )
                    if (msg := _convert_content_to_chat_message(content))
                ]
            )
            if not chat_log.unresponded_tool_results:
                break

    async def _non_streaming_deltas(
        self, client: Any, model_args: dict[str, Any]
    ) -> AsyncGenerator[conversation.AssistantContentDeltaDict]:
        """Non-streaming completion as a delta generator (AI task path)."""
        try:
            result = await client.chat.completions.create(**model_args)
        except openai.BadRequestError as err:
            if "response_format" not in model_args:
                _raise_gateway_error(err, self.model)
            # Gateway/model rejects json_schema output: retry once with the
            # schema embedded in a system message instead.
            LOGGER.warning(
                "response_format rejected by %s, falling back to prompt-based "
                "JSON: %s",
                self.model,
                err,
            )
            retry_args = dict(model_args)
            schema = retry_args.pop("response_format")["json_schema"]["schema"]
            retry_args["messages"] = [
                *retry_args["messages"],
                {
                    "role": "system",
                    "content": (
                        "Respond ONLY with JSON matching this schema: "
                        f"{json.dumps(schema)}"
                    ),
                },
            ]
            try:
                result = await client.chat.completions.create(**retry_args)
            except openai.OpenAIError as retry_err:
                _raise_gateway_error(retry_err, self.model)
        except openai.OpenAIError as err:
            _raise_gateway_error(err, self.model)
        if not result.choices:
            raise HomeAssistantError("API returned empty response")
        choice = result.choices[0]
        message = choice.message

        cleaned, thinking = _extract_thinking(message)
        if (
            choice.finish_reason == "length"
            and not cleaned
            and not message.tool_calls
        ):
            raise HomeAssistantError(
                "Token limit reached before the model produced an answer; "
                "increase max_tokens"
            )
        data: conversation.AssistantContentDeltaDict = {
            "role": message.role,
            "content": cleaned,
        }
        if thinking:
            data["thinking_content"] = thinking
        if message.tool_calls:
            data["tool_calls"] = [
                llm.ToolInput(
                    id=tool_call.id,
                    tool_name=tool_call.function.name,
                    tool_args=_decode_tool_arguments(tool_call.function.arguments),
                )
                for tool_call in message.tool_calls
                if tool_call.type == "function"
            ]
        yield data
