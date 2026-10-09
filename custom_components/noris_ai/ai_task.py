"""AI Task support for noris AI."""

from __future__ import annotations

from json import JSONDecodeError
import re

from homeassistant.components import ai_task, conversation
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.util.json import json_loads

from . import NorisAIConfigEntry
from .const import AI_TASK_SUBENTRY_TYPE, RECOMMENDED_AI_TASK_MAX_TOKENS
from .entity import NorisAIEntity

PARALLEL_UPDATES = 0

_CODE_FENCE_PATTERN = re.compile(
    r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL
)


def _strip_code_fences(text: str) -> str:
    """Remove a surrounding markdown code fence, if present."""
    if match := _CODE_FENCE_PATTERN.match(text.strip()):
        return match.group(1)
    return text


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: NorisAIConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up AI Task entities."""
    for subentry in config_entry.get_subentries_of_type(AI_TASK_SUBENTRY_TYPE):
        async_add_entities(
            [NorisAITaskEntity(config_entry, subentry)],
            config_subentry_id=subentry.subentry_id,
        )


class NorisAITaskEntity(NorisAIEntity, ai_task.AITaskEntity):
    """noris AI AI Task entity."""

    _attr_name = None
    _attr_supported_features = ai_task.AITaskEntityFeature.GENERATE_DATA
    _recommended_max_tokens = RECOMMENDED_AI_TASK_MAX_TOKENS

    async def _async_generate_data(
        self,
        task: ai_task.GenDataTask,
        chat_log: conversation.ChatLog,
    ) -> ai_task.GenDataTaskResult:
        """Handle a generate data task."""
        await self._async_handle_chat_log(
            chat_log,
            stream=False,
            structure_name=task.name,
            structure=task.structure,
        )

        last = chat_log.content[-1]
        if not isinstance(last, conversation.AssistantContent) or last.content is None:
            raise HomeAssistantError("Unexpected empty response from noris AI")
        text = last.content

        if not task.structure:
            return ai_task.GenDataTaskResult(
                conversation_id=chat_log.conversation_id,
                data=text,
            )

        try:
            data = json_loads(_strip_code_fences(text))
        except JSONDecodeError as err:
            raise HomeAssistantError(
                "Error parsing structured response from noris AI"
            ) from err

        return ai_task.GenDataTaskResult(
            conversation_id=chat_log.conversation_id,
            data=data,
        )
