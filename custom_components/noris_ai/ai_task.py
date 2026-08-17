"""AI Task support for noris AI."""

from __future__ import annotations

from json import JSONDecodeError

from homeassistant.components import ai_task, conversation
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.util.json import json_loads

from . import NorisAIConfigEntry
from .const import AI_TASK_SUBENTRY_TYPE
from .entity import NorisAIEntity

PARALLEL_UPDATES = 0


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

    async def _async_generate_data(
        self,
        task: ai_task.GenDataTask,
        chat_log: conversation.ChatLog,
    ) -> ai_task.GenDataTaskResult:
        """Handle a generate data task."""
        await self._async_handle_chat_log(chat_log, task.name, task.structure)

        # The last assistant message holds the generated answer.
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
            data = json_loads(text)
        except JSONDecodeError as err:
            raise HomeAssistantError(
                "Error parsing structured response from noris AI"
            ) from err

        return ai_task.GenDataTaskResult(
            conversation_id=chat_log.conversation_id,
            data=data,
        )
