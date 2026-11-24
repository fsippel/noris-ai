"""Config flow for the noris AI integration."""

from __future__ import annotations

from collections.abc import Mapping
import logging
from typing import Any

from openai import AuthenticationError, OpenAIError, PermissionDeniedError
import voluptuous as vol

from homeassistant.config_entries import (
    ConfigEntry,
    ConfigEntryState,
    ConfigFlow,
    ConfigFlowResult,
    ConfigSubentryFlow,
    SubentryFlowResult,
)
from homeassistant.const import CONF_API_KEY, CONF_LLM_HASS_API, CONF_MODEL, CONF_NAME
from homeassistant.core import callback
from homeassistant.helpers import llm
from homeassistant.helpers.selector import (
    BooleanSelector,
    NumberSelector,
    NumberSelectorConfig,
    NumberSelectorMode,
    SelectOptionDict,
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
    TemplateSelector,
)

from . import _create_client, _validate_api_key
from .const import (
    AI_TASK_SUBENTRY_TYPE,
    CONF_ENABLE_REASONING,
    CONF_MAX_TOKENS,
    CONF_PROMPT,
    CONF_RECOMMENDED,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    CONVERSATION_SUBENTRY_TYPE,
    DEFAULT_AI_TASK_NAME,
    DEFAULT_CONVERSATION_NAME,
    DEFAULT_STT_NAME,
    DEFAULT_TTS_NAME,
    DOMAIN,
    RECOMMENDED_AI_TASK_MAX_TOKENS,
    RECOMMENDED_CONVERSATION_MAX_TOKENS,
    RECOMMENDED_TEMPERATURE,
    RECOMMENDED_TOP_P,
    STT_SUBENTRY_TYPE,
    TTS_SUBENTRY_TYPE,
)
from .models import (
    async_get_model_options,
    default_model,
    default_stt_model,
    default_tts_model,
    is_chat_model,
    is_stt_model,
    is_tts_model,
)

_LOGGER = logging.getLogger(__name__)

STEP_API_KEY_DATA_SCHEMA = vol.Schema({vol.Required(CONF_API_KEY): str})


class NorisAIConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for noris AI."""

    VERSION = 1
    MINOR_VERSION = 2

    @classmethod
    @callback
    def async_get_supported_subentry_types(
        cls, config_entry: ConfigEntry
    ) -> dict[str, type[ConfigSubentryFlow]]:
        """Return subentries supported by this handler."""
        return {
            CONVERSATION_SUBENTRY_TYPE: ConversationFlowHandler,
            AI_TASK_SUBENTRY_TYPE: AITaskFlowHandler,
            STT_SUBENTRY_TYPE: SttFlowHandler,
            TTS_SUBENTRY_TYPE: TtsFlowHandler,
        }

    async def _async_validate_key(self, api_key: str) -> dict[str, str]:
        """Validate a key, returning a config-flow errors dict."""
        client = _create_client(self.hass, api_key)
        try:
            await _validate_api_key(client)
        except (AuthenticationError, PermissionDeniedError):
            return {"base": "invalid_auth"}
        except OpenAIError:
            return {"base": "cannot_connect"}
        except Exception:
            _LOGGER.exception("Unexpected exception")
            return {"base": "unknown"}
        return {}

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle the initial step."""
        errors: dict[str, str] = {}
        if user_input is not None:
            self._async_abort_entries_match(user_input)
            if not (errors := await self._async_validate_key(user_input[CONF_API_KEY])):
                return self.async_create_entry(title="noris AI", data=user_input)
        return self.async_show_form(
            step_id="user", data_schema=STEP_API_KEY_DATA_SCHEMA, errors=errors
        )

    async def async_step_reauth(
        self, entry_data: Mapping[str, Any]
    ) -> ConfigFlowResult:
        """Perform reauth upon an API authentication error."""
        return await self.async_step_reauth_confirm()

    async def async_step_reauth_confirm(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Confirm reauthentication dialog."""
        errors: dict[str, str] = {}
        if user_input is not None:
            if not (errors := await self._async_validate_key(user_input[CONF_API_KEY])):
                return self.async_update_reload_and_abort(
                    self._get_reauth_entry(), data_updates=user_input
                )
        return self.async_show_form(
            step_id="reauth_confirm",
            data_schema=STEP_API_KEY_DATA_SCHEMA,
            errors=errors,
        )


class NorisAISubentryFlowHandler(ConfigSubentryFlow):
    """Shared logic for every subentry flow of this integration.

    One subentry maps to one device and one platform entity. Conversation and
    AI task entities are chat models with tunable sampling parameters; the
    speech platforms take only a model, so they switch the advanced step off
    instead of duplicating the flow.
    """

    subentry_type: str
    default_name: str
    recommended_max_tokens: int = RECOMMENDED_CONVERSATION_MAX_TOKENS

    # Chat platforms expose the advanced step (token budget, temperature,
    # top_p, reasoning). Speech platforms have nothing to tune there.
    supports_advanced: bool = True

    # Abort reason when the gateway offers no model for this platform.
    no_models_abort_reason: str = "no_models"

    def __init__(self) -> None:
        """Initialize the subentry flow."""
        self._data: dict[str, Any] = {}
        self._name: str | None = None

    @property
    def _is_new(self) -> bool:
        return self.source == "user"

    async def _async_model_options(self) -> list[SelectOptionDict]:
        """Return the models this platform can use."""
        return await async_get_model_options(
            self._get_entry().runtime_data, is_chat_model
        )

    def _default_model(self, options: list[SelectOptionDict]) -> str | None:
        """Return the model preselected in the picker."""
        return default_model(options)

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Create a new subentry."""
        return await self.async_step_init(user_input)

    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Reconfigure an existing subentry."""
        return await self.async_step_init(user_input)

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """First step: name (new only), model, recommended toggle."""
        entry = self._get_entry()
        if entry.state is not ConfigEntryState.LOADED:
            return self.async_abort(reason="entry_not_loaded")

        if user_input is not None:
            self._name = user_input.pop(CONF_NAME, None)
            if not user_input.get(CONF_LLM_HASS_API):
                user_input.pop(CONF_LLM_HASS_API, None)
            self._data = user_input
            if not self.supports_advanced or user_input.get(CONF_RECOMMENDED):
                return await self._async_finish()
            return await self.async_step_advanced()

        try:
            model_options = await self._async_model_options()
        except OpenAIError:
            return self.async_abort(reason="cannot_connect")
        except Exception:
            _LOGGER.exception("Unexpected exception")
            return self.async_abort(reason="unknown")

        if not model_options:
            return self.async_abort(reason=self.no_models_abort_reason)

        existing: Mapping[str, Any] = (
            self._get_reconfigure_subentry().data if not self._is_new else {}
        )

        schema: dict[Any, Any] = {}
        if self._is_new:
            schema[vol.Required(CONF_NAME, default=self.default_name)] = str
        schema[
            vol.Required(
                CONF_MODEL,
                default=existing.get(CONF_MODEL) or self._default_model(model_options),
            )
        ] = SelectSelector(
            SelectSelectorConfig(
                options=model_options,
                mode=SelectSelectorMode.DROPDOWN,
                sort=False,
            )
        )
        self._extend_init_schema(schema, existing)
        if self.supports_advanced:
            schema[
                vol.Required(
                    CONF_RECOMMENDED, default=existing.get(CONF_RECOMMENDED, True)
                )
            ] = BooleanSelector()
        return self.async_show_form(step_id="init", data_schema=vol.Schema(schema))

    def _extend_init_schema(
        self, schema: dict[Any, Any], existing: Mapping[str, Any]
    ) -> None:
        """Hook for subclasses to add fields to the init step."""

    async def async_step_advanced(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Advanced options when recommended settings are disabled."""
        if user_input is not None:
            self._data.update(user_input)
            return await self._async_finish()

        existing: Mapping[str, Any] = (
            self._get_reconfigure_subentry().data if not self._is_new else {}
        )
        schema: dict[Any, Any] = {}
        self._extend_advanced_schema(schema, existing)
        schema.update(
            {
                vol.Required(
                    CONF_MAX_TOKENS,
                    default=existing.get(CONF_MAX_TOKENS, self.recommended_max_tokens),
                ): NumberSelector(
                    NumberSelectorConfig(
                        min=100, max=128000, step=1, mode=NumberSelectorMode.BOX
                    )
                ),
                vol.Required(
                    CONF_TEMPERATURE,
                    default=existing.get(CONF_TEMPERATURE, RECOMMENDED_TEMPERATURE),
                ): NumberSelector(NumberSelectorConfig(min=0, max=2, step=0.05)),
                vol.Required(
                    CONF_TOP_P,
                    default=existing.get(CONF_TOP_P, RECOMMENDED_TOP_P),
                ): NumberSelector(NumberSelectorConfig(min=0, max=1, step=0.05)),
                vol.Required(
                    CONF_ENABLE_REASONING,
                    default=existing.get(CONF_ENABLE_REASONING, False),
                ): BooleanSelector(),
            }
        )
        return self.async_show_form(step_id="advanced", data_schema=vol.Schema(schema))

    def _extend_advanced_schema(
        self, schema: dict[Any, Any], existing: Mapping[str, Any]
    ) -> None:
        """Hook for subclasses to add fields to the advanced step."""

    async def _async_finish(self) -> SubentryFlowResult:
        # NumberSelector returns floats; keep the token budget an int.
        if CONF_MAX_TOKENS in self._data:
            self._data[CONF_MAX_TOKENS] = int(self._data[CONF_MAX_TOKENS])
        if self._is_new:
            return self.async_create_entry(
                title=self._name or self.default_name, data=self._data
            )
        return self.async_update_and_abort(
            self._get_entry(), self._get_reconfigure_subentry(), data=self._data
        )


class ConversationFlowHandler(NorisAISubentryFlowHandler):
    """Subentry flow for conversation agents."""

    subentry_type = CONVERSATION_SUBENTRY_TYPE
    default_name = DEFAULT_CONVERSATION_NAME
    recommended_max_tokens = RECOMMENDED_CONVERSATION_MAX_TOKENS

    def _extend_init_schema(
        self, schema: dict[Any, Any], existing: Mapping[str, Any]
    ) -> None:
        hass_apis: list[SelectOptionDict] = [
            SelectOptionDict(label=api.name, value=api.id)
            for api in llm.async_get_apis(self.hass)
        ]
        default_apis = [llm.LLM_API_ASSIST] if self._is_new else []
        schema[
            vol.Optional(
                CONF_LLM_HASS_API,
                default=existing.get(CONF_LLM_HASS_API, default_apis),
            )
        ] = SelectSelector(SelectSelectorConfig(options=hass_apis, multiple=True))
        schema[
            vol.Optional(
                CONF_PROMPT,
                description={
                    "suggested_value": existing.get(
                        CONF_PROMPT, llm.DEFAULT_INSTRUCTIONS_PROMPT
                    )
                },
            )
        ] = TemplateSelector()

    def _extend_advanced_schema(
        self, schema: dict[Any, Any], existing: Mapping[str, Any]
    ) -> None:
        """Advanced options: token budget, temperature, top_p."""


class AITaskFlowHandler(NorisAISubentryFlowHandler):
    """Subentry flow for AI task entities."""

    subentry_type = AI_TASK_SUBENTRY_TYPE
    default_name = DEFAULT_AI_TASK_NAME
    recommended_max_tokens = RECOMMENDED_AI_TASK_MAX_TOKENS


class SttFlowHandler(NorisAISubentryFlowHandler):
    """Subentry flow for speech-to-text engines.

    Only the transcription model is configurable: the platform reports a fixed
    audio format (16 kHz mono 16-bit PCM in a WAV container), and the language
    comes from the pipeline, so there is nothing else to ask for.
    """

    subentry_type = STT_SUBENTRY_TYPE
    default_name = DEFAULT_STT_NAME
    supports_advanced = False
    no_models_abort_reason = "no_audio_models"

    async def _async_model_options(self) -> list[SelectOptionDict]:
        """Return the models that can transcribe audio."""
        return await async_get_model_options(
            self._get_entry().runtime_data, is_stt_model
        )

    def _default_model(self, options: list[SelectOptionDict]) -> str | None:
        """Preselect the transcription model the gateway actually serves."""
        return default_stt_model(options)


class TtsFlowHandler(NorisAISubentryFlowHandler):
    """Subentry flow for speech synthesis engines.

    Only the speech model is configurable: the gateway requires a ``voice``
    field but ignores its value, and every speech model exposes exactly one
    voice (see TTS_VOICE).
    """

    subentry_type = TTS_SUBENTRY_TYPE
    default_name = DEFAULT_TTS_NAME
    supports_advanced = False
    no_models_abort_reason = "no_speech_models"

    async def _async_model_options(self) -> list[SelectOptionDict]:
        """Return the models that can synthesise speech."""
        return await async_get_model_options(
            self._get_entry().runtime_data, is_tts_model
        )

    def _default_model(self, options: list[SelectOptionDict]) -> str | None:
        """Preselect the German voice."""
        return default_tts_model(options)
