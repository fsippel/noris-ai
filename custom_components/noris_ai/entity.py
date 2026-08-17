"""Base entity for noris AI."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Callable
import json
import re
from typing import Any, Literal

import openai
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionFunctionToolParam,
    ChatCompletionMessage,
    ChatCompletionMessageFunctionToolCallParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionUserMessageParam,
)
from openai.types.chat.chat_completion_message_function_tool_call_param import Function
from openai.types.shared_params import FunctionDefinition
import voluptuous as vol
from voluptuous_openapi import convert

from homeassistant.components import conversation
from homeassistant.config_entries import ConfigSubentry
from homeassistant.const import CONF_MODEL
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import device_registry as dr, llm
from homeassistant.helpers.entity import Entity
from homeassistant.helpers.json import json_dumps

from . import NorisAIConfigEntry
from .const import DOMAIN, LOGGER

MAX_TOOL_ITERATIONS = 10

_THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL)


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


def _format_structure_output(name: str, structure: vol.Schema) -> dict[str, Any]:
    """Build a JSON-schema ``response_format`` for structured AI Task output."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name or "data",
            "schema": convert(structure),
            "strict": True,
        },
    }


def _convert_content_to_chat_message(
    content: conversation.Content,
) -> ChatCompletionMessageParam | None:
    """Convert chat message for this agent to the native format."""
    LOGGER.debug("_convert_content_to_chat_message=%s", content)
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


def _strip_markdown_fence(text: str) -> str:
    """Remove Markdown code fences (```…```) wrapping a JSON payload.

    Some models (notably GLM-5-2) wrap ``response_format: json_object``
    output — and therefore tool-call argument strings — in Markdown fences
    despite the constraint, producing e.g. `````{"city": "Nürnberg"}``````
    which is not directly parseable by ``json.loads``.
    """
    stripped = text.strip()
    # Leading fence: ```json\n… or ```\n…
    if stripped.startswith("```"):
        # Drop the opening fence line (may include a language tag like "json").
        nl = stripped.find("\n")
        stripped = stripped[nl + 1 :] if nl != -1 else stripped[3:]
        # Drop a trailing closing fence.
        if stripped.rstrip().endswith("```"):
            stripped = stripped.rstrip()[:-3]
    return stripped.strip()


def _decode_tool_arguments(arguments: str) -> Any:
    """Decode tool call arguments.

    Tolerant of models that wrap the JSON in Markdown code fences (e.g.
    GLM-5-2) or surround it with stray backticks.  A best-effort second
    parse is attempted after stripping; if that also fails the original
    error is raised so the caller still sees a meaningful message.
    """
    try:
        return json.loads(arguments)
    except json.JSONDecodeError:
        cleaned = _strip_markdown_fence(arguments)
        if cleaned and cleaned != arguments.strip():
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError as err:
                raise HomeAssistantError(
                    f"Unexpected tool argument response: {err}"
                ) from err
        raise HomeAssistantError(
            "Unexpected tool argument response: not valid JSON"
        ) from None


def _split_thinking(content: str | None) -> tuple[str | None, str | None]:
    """Return (cleaned_content, thinking_content) extracted from ``<think>`` tags."""
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
    """Return (cleaned_content, thinking_content) for an assistant message.

    Priority order:
    1. ``message.reasoning`` (vLLM >= 0.16.0 with a ``reasoning_parser``).
    2. ``message.reasoning_content`` (vLLM < 0.16.0 with a ``reasoning_parser``).
    3. Inline ``<think>...</think>`` markup in ``message.content`` (reasoning
       models on vLLM without a ``reasoning_parser`` set).
    """
    extras = message.model_extra or {}
    for key in ("reasoning", "reasoning_content"):
        value = extras.get(key)
        if isinstance(value, str) and value.strip():
            return message.content, value.strip()
    return _split_thinking(message.content)


async def _transform_response(
    message: ChatCompletionMessage,
) -> AsyncGenerator[conversation.AssistantContentDeltaDict]:
    """Transform the noris AI message to a ChatLog format."""
    cleaned_content, thinking_content = _extract_thinking(message)
    data: conversation.AssistantContentDeltaDict = {
        "role": message.role,
        "content": cleaned_content,
    }
    if thinking_content:
        data["thinking_content"] = thinking_content
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


class NorisAISubentryEntity(Entity):
    """Device identity for an entity backed by a config subentry.

    Every platform in this integration maps one subentry to one device, and that
    device's identity is the same regardless of what the entity does. Only the
    identity lives here; behaviour belongs to the subclasses.
    """

    _attr_has_entity_name = True

    def __init__(
        self,
        entry: NorisAIConfigEntry,
        subentry: ConfigSubentry,
    ) -> None:
        """Initialize the entity from its subentry."""
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


class NorisAIEntity(NorisAISubentryEntity):
    """Base entity for the chat-driven noris AI platforms."""

    async def _async_handle_chat_log(
        self,
        chat_log: conversation.ChatLog,
        structure_name: str | None = None,
        structure: vol.Schema | None = None,
    ) -> None:
        """Generate an answer for the chat log."""
        model_args: dict[str, Any] = {
            "model": self.model,
        }

        tools: list[ChatCompletionFunctionToolParam] | None = None
        if chat_log.llm_api:
            tools = [
                _format_tool(tool, chat_log.llm_api.custom_serializer)
                for tool in chat_log.llm_api.tools
            ]

        if tools:
            model_args["tools"] = tools

        if structure:
            model_args["response_format"] = _format_structure_output(
                structure_name or "data", structure
            )

        model_args["messages"] = [
            m
            for content in chat_log.content
            if (m := _convert_content_to_chat_message(content))
        ]

        client = self.entry.runtime_data

        for _iteration in range(MAX_TOOL_ITERATIONS):
            try:
                result = await client.chat.completions.create(**model_args)
            except openai.OpenAIError as err:
                LOGGER.error("Error talking to API: %s", err)
                raise HomeAssistantError("Error talking to API") from err

            if not result.choices:
                LOGGER.error("API returned empty choices")
                raise HomeAssistantError("API returned empty response")

            result_message = result.choices[0].message

            model_args["messages"].extend(
                [
                    msg
                    async for content in chat_log.async_add_delta_content_stream(
                        self.entity_id, _transform_response(result_message)
                    )
                    if (msg := _convert_content_to_chat_message(content))
                ]
            )
            if not chat_log.unresponded_tool_results:
                break
