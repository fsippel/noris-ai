"""Config flow for the noris AI integration."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import logging
from typing import Any

from openai import AsyncOpenAI, AuthenticationError, OpenAIError, PermissionDeniedError
import voluptuous as vol

from homeassistant.config_entries import (
    ConfigEntry,
    ConfigEntryState,
    ConfigFlow,
    ConfigFlowResult,
    ConfigSubentryFlow,
    SubentryFlowResult,
)
from homeassistant.const import CONF_API_KEY, CONF_LLM_HASS_API, CONF_MODEL
from homeassistant.core import callback
from homeassistant.helpers import llm
from homeassistant.helpers.selector import (
    SelectOptionDict,
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
    TemplateSelector,
)

from . import _create_client, _validate_api_key
from .const import (
    AI_TASK_SUBENTRY_TYPE,
    AUDIO_MODEL_PATTERN,
    CONF_PROMPT,
    CONVERSATION_SUBENTRY_TYPE,
    DEFAULT_AI_TASK_NAME,
    DEFAULT_CONVERSATION_NAME,
    DEFAULT_STT_NAME,
    DEFAULT_TTS_NAME,
    DOMAIN,
    RECOMMENDED_CONVERSATION_OPTIONS,
    STT_SUBENTRY_TYPE,
    TTS_MODEL_PATTERN,
    TTS_SUBENTRY_TYPE,
)

_LOGGER = logging.getLogger(__name__)

STEP_API_KEY_DATA_SCHEMA = vol.Schema({vol.Required(CONF_API_KEY): str})


def _model_output_types(model: Any) -> set[str]:
    """Return the output modality types the gateway reports for a model.

    These fields are not part of the OpenAI SDK's Model type, so they arrive in
    ``model_extra``. An empty set means the gateway reported nothing usable.
    """
    extra = getattr(model, "model_extra", None) or {}
    modalities = extra.get("output_modalities") or []
    return {m.get("type") for m in modalities if isinstance(m, dict)}


def _is_audio_model(model: Any) -> bool:
    """Return True for models that can transcribe audio.

    See AUDIO_MODEL_PATTERN: the catalog's modality metadata is wrong for
    Voxtral, so known audio families are matched by name. ``hugging_face_id`` is
    the stronger signal but is null for many models, hence the fallback.
    """
    extra = getattr(model, "model_extra", None) or {}
    hugging_face_id = extra.get("hugging_face_id") or ""
    return bool(
        AUDIO_MODEL_PATTERN.search(hugging_face_id)
        or AUDIO_MODEL_PATTERN.search(model.id)
    )


def _is_chat_model(model: Any) -> bool:
    """Return True for vLLM models usable as a chat or agent model."""
    if not model.id.startswith("vllm/"):
        return False
    if _is_audio_model(model):
        return False
    output_types = _model_output_types(model)
    if output_types:
        return "text" in output_types
    # No metadata from the gateway: fall back to the original name heuristic
    # so an upstream catalog change cannot empty the dropdown.
    lowered = model.id.lower()
    return "reranker" not in lowered and "harrier" not in lowered


def _is_stt_model(model: Any) -> bool:
    """Return True for vLLM models usable as a transcription engine."""
    return model.id.startswith("vllm/") and _is_audio_model(model)


def _is_tts_model(model: Any) -> bool:
    """Return True for models that can synthesise speech.

    Unlike _is_chat_model and _is_stt_model, this deliberately does NOT gate on
    a "vllm/" prefix: the speech models live under their own providers
    (Kokoro-TTS/, Cosyvoice3/), so requiring "vllm/" would reject every one of
    them.
    """
    return bool(TTS_MODEL_PATTERN.search(model.id))


async def _fetch_model_options(
    entry: ConfigEntry, predicate: Callable[[Any], bool]
) -> list[SelectOptionDict]:
    """Fetch models from the gateway and keep those matching the predicate."""
    client: AsyncOpenAI = entry.runtime_data
    return [
        SelectOptionDict(value=model.id, label=model.id)
        async for model in client.with_options(timeout=10.0).models.list()
        if predicate(model)
    ]


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

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle the initial step."""
        errors: dict[str, str] = {}
        if user_input is not None:
            self._async_abort_entries_match(user_input)
            client = _create_client(self.hass, user_input[CONF_API_KEY])
            try:
                await _validate_api_key(client)
            except (AuthenticationError, PermissionDeniedError):
                errors["base"] = "invalid_auth"
            except OpenAIError:
                errors["base"] = "cannot_connect"
            except Exception:
                _LOGGER.exception("Unexpected exception")
                errors["base"] = "unknown"
            else:
                return self.async_create_entry(
                    title="noris AI",
                    data=user_input,
                )
        return self.async_show_form(
            step_id="user",
            data_schema=STEP_API_KEY_DATA_SCHEMA,
            errors=errors,
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
            client = _create_client(self.hass, user_input[CONF_API_KEY])
            try:
                await _validate_api_key(client)
            except (AuthenticationError, PermissionDeniedError):
                errors["base"] = "invalid_auth"
            except OpenAIError:
                errors["base"] = "cannot_connect"
            except Exception:
                _LOGGER.exception("Unexpected exception")
                errors["base"] = "unknown"
            else:
                return self.async_update_reload_and_abort(
                    self._get_reauth_entry(),
                    data_updates=user_input,
                )
        return self.async_show_form(
            step_id="reauth_confirm",
            data_schema=STEP_API_KEY_DATA_SCHEMA,
            errors=errors,
        )

    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Reconfigure the API key on an existing entry."""
        errors: dict[str, str] = {}
        entry = self._get_reconfigure_entry()
        if user_input is not None:
            self._async_abort_entries_match(user_input)
            client = _create_client(self.hass, user_input[CONF_API_KEY])
            try:
                await _validate_api_key(client)
            except (AuthenticationError, PermissionDeniedError):
                errors["base"] = "invalid_auth"
            except OpenAIError:
                errors["base"] = "cannot_connect"
            except Exception:
                _LOGGER.exception("Unexpected exception")
                errors["base"] = "unknown"
            else:
                return self.async_update_reload_and_abort(
                    entry, data_updates=user_input
                )
        return self.async_show_form(
            step_id="reconfigure",
            data_schema=self.add_suggested_values_to_schema(
                STEP_API_KEY_DATA_SCHEMA, user_input or entry.data
            ),
            errors=errors,
        )


class ConversationFlowHandler(ConfigSubentryFlow):
    """Handle the conversation agent subentry flow."""

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """User flow to create a conversation agent."""
        return await self.async_step_init(user_input)

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Manage conversation agent configuration."""
        if self._get_entry().state is not ConfigEntryState.LOADED:
            return self.async_abort(reason="entry_not_loaded")

        if user_input is not None:
            if not user_input.get(CONF_LLM_HASS_API):
                user_input.pop(CONF_LLM_HASS_API, None)
            return self.async_create_entry(
                title=DEFAULT_CONVERSATION_NAME, data=user_input
            )

        try:
            model_options = await _fetch_model_options(
                self._get_entry(), _is_chat_model
            )
        except OpenAIError:
            return self.async_abort(reason="cannot_connect")
        except Exception:
            _LOGGER.exception("Unexpected exception")
            return self.async_abort(reason="unknown")

        hass_apis: list[SelectOptionDict] = [
            SelectOptionDict(label=api.name, value=api.id)
            for api in llm.async_get_apis(self.hass)
        ]

        return self.async_show_form(
            step_id="init",
            data_schema=vol.Schema(
                {
                    vol.Required(CONF_MODEL): SelectSelector(
                        SelectSelectorConfig(
                            options=model_options,
                            mode=SelectSelectorMode.DROPDOWN,
                            sort=True,
                        ),
                    ),
                    vol.Optional(
                        CONF_PROMPT,
                        description={
                            "suggested_value": RECOMMENDED_CONVERSATION_OPTIONS[
                                CONF_PROMPT
                            ]
                        },
                    ): TemplateSelector(),
                    vol.Optional(
                        CONF_LLM_HASS_API,
                        default=RECOMMENDED_CONVERSATION_OPTIONS[CONF_LLM_HASS_API],
                    ): SelectSelector(
                        SelectSelectorConfig(options=hass_apis, multiple=True)
                    ),
                }
            ),
        )

    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Reconfigure a conversation agent (prompt + LLM APIs; model is fixed)."""
        subentry = self._get_reconfigure_subentry()
        existing = subentry.data

        if user_input is not None:
            if not user_input.get(CONF_LLM_HASS_API):
                user_input.pop(CONF_LLM_HASS_API, None)
            user_input[CONF_MODEL] = existing[CONF_MODEL]
            return self.async_update_and_abort(
                self._get_entry(), subentry, data=user_input
            )

        hass_apis: list[SelectOptionDict] = [
            SelectOptionDict(label=api.name, value=api.id)
            for api in llm.async_get_apis(self.hass)
        ]
        return self.async_show_form(
            step_id="reconfigure",
            data_schema=vol.Schema(
                {
                    vol.Optional(
                        CONF_PROMPT,
                        description={
                            "suggested_value": existing.get(
                                CONF_PROMPT,
                                RECOMMENDED_CONVERSATION_OPTIONS[CONF_PROMPT],
                            )
                        },
                    ): TemplateSelector(),
                    vol.Optional(
                        CONF_LLM_HASS_API,
                        default=existing.get(
                            CONF_LLM_HASS_API,
                            RECOMMENDED_CONVERSATION_OPTIONS[CONF_LLM_HASS_API],
                        ),
                    ): SelectSelector(
                        SelectSelectorConfig(options=hass_apis, multiple=True)
                    ),
                }
            ),
        )


class AITaskFlowHandler(ConfigSubentryFlow):
    """Handle the AI Task subentry flow."""

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """User flow to create an AI Task entity."""
        return await self.async_step_init(user_input)

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Manage AI Task configuration."""
        if self._get_entry().state is not ConfigEntryState.LOADED:
            return self.async_abort(reason="entry_not_loaded")

        if user_input is not None:
            return self.async_create_entry(title=DEFAULT_AI_TASK_NAME, data=user_input)

        try:
            model_options = await _fetch_model_options(
                self._get_entry(), _is_chat_model
            )
        except OpenAIError:
            return self.async_abort(reason="cannot_connect")
        except Exception:
            _LOGGER.exception("Unexpected exception")
            return self.async_abort(reason="unknown")

        return self.async_show_form(
            step_id="init",
            data_schema=vol.Schema(
                {
                    vol.Required(CONF_MODEL): SelectSelector(
                        SelectSelectorConfig(
                            options=model_options,
                            mode=SelectSelectorMode.DROPDOWN,
                            sort=True,
                        ),
                    ),
                }
            ),
        )


class SttFlowHandler(ConfigSubentryFlow):
    """Handle the speech-to-text subentry flow."""

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """User flow to create a speech-to-text entity."""
        return await self.async_step_init(user_input)

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Manage speech-to-text configuration."""
        if self._get_entry().state is not ConfigEntryState.LOADED:
            return self.async_abort(reason="entry_not_loaded")

        if user_input is not None:
            return self.async_create_entry(title=DEFAULT_STT_NAME, data=user_input)

        try:
            model_options = await _fetch_model_options(self._get_entry(), _is_stt_model)
        except OpenAIError:
            return self.async_abort(reason="cannot_connect")
        except Exception:
            _LOGGER.exception("Unexpected exception")
            return self.async_abort(reason="unknown")

        if not model_options:
            return self.async_abort(reason="no_audio_models")

        return self.async_show_form(
            step_id="init",
            data_schema=vol.Schema(
                {
                    vol.Required(CONF_MODEL): SelectSelector(
                        SelectSelectorConfig(
                            options=model_options,
                            mode=SelectSelectorMode.DROPDOWN,
                            sort=True,
                        ),
                    ),
                }
            ),
        )

    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Change the transcription model of an existing entity."""
        subentry = self._get_reconfigure_subentry()

        if user_input is not None:
            return self.async_update_and_abort(
                self._get_entry(), subentry, data=user_input
            )

        try:
            model_options = await _fetch_model_options(self._get_entry(), _is_stt_model)
        except OpenAIError:
            return self.async_abort(reason="cannot_connect")
        except Exception:
            _LOGGER.exception("Unexpected exception")
            return self.async_abort(reason="unknown")

        if not model_options:
            return self.async_abort(reason="no_audio_models")

        return self.async_show_form(
            step_id="reconfigure",
            data_schema=self.add_suggested_values_to_schema(
                vol.Schema(
                    {
                        vol.Required(CONF_MODEL): SelectSelector(
                            SelectSelectorConfig(
                                options=model_options,
                                mode=SelectSelectorMode.DROPDOWN,
                                sort=True,
                            ),
                        ),
                    }
                ),
                subentry.data,
            ),
        )


class TtsFlowHandler(ConfigSubentryFlow):
    """Handle the text-to-speech subentry flow."""

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """User flow to create a text-to-speech entity."""
        return await self.async_step_init(user_input)

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Manage text-to-speech configuration."""
        if self._get_entry().state is not ConfigEntryState.LOADED:
            return self.async_abort(reason="entry_not_loaded")

        if user_input is not None:
            return self.async_create_entry(title=DEFAULT_TTS_NAME, data=user_input)

        try:
            model_options = await _fetch_model_options(self._get_entry(), _is_tts_model)
        except OpenAIError:
            return self.async_abort(reason="cannot_connect")
        except Exception:
            _LOGGER.exception("Unexpected exception")
            return self.async_abort(reason="unknown")

        if not model_options:
            return self.async_abort(reason="no_speech_models")

        return self.async_show_form(
            step_id="init",
            data_schema=vol.Schema(
                {
                    vol.Required(CONF_MODEL): SelectSelector(
                        SelectSelectorConfig(
                            options=model_options,
                            mode=SelectSelectorMode.DROPDOWN,
                            sort=True,
                        ),
                    ),
                }
            ),
        )

    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Change the speech model of an existing entity."""
        subentry = self._get_reconfigure_subentry()

        if user_input is not None:
            return self.async_update_and_abort(
                self._get_entry(), subentry, data=user_input
            )

        try:
            model_options = await _fetch_model_options(self._get_entry(), _is_tts_model)
        except OpenAIError:
            return self.async_abort(reason="cannot_connect")
        except Exception:
            _LOGGER.exception("Unexpected exception")
            return self.async_abort(reason="unknown")

        if not model_options:
            return self.async_abort(reason="no_speech_models")

        return self.async_show_form(
            step_id="reconfigure",
            data_schema=self.add_suggested_values_to_schema(
                vol.Schema(
                    {
                        vol.Required(CONF_MODEL): SelectSelector(
                            SelectSelectorConfig(
                                options=model_options,
                                mode=SelectSelectorMode.DROPDOWN,
                                sort=True,
                            ),
                        ),
                    }
                ),
                subentry.data,
            ),
        )
