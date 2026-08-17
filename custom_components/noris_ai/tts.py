"""Text-to-speech support for noris AI."""

from __future__ import annotations

from typing import Any

import openai

from homeassistant.components import tts
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from . import NorisAIConfigEntry
from .const import (
    LOGGER,
    TTS_DEFAULT_LANGUAGE,
    TTS_SUBENTRY_TYPE,
    TTS_TIMEOUT,
    TTS_VOICE,
)
from .entity import NorisAISubentryEntity
from .helpers import tts_languages_for_model

PARALLEL_UPDATES = 0


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: NorisAIConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up text-to-speech entities."""
    for subentry in config_entry.get_subentries_of_type(TTS_SUBENTRY_TYPE):
        async_add_entities(
            [NorisAITtsEntity(config_entry, subentry)],
            config_subentry_id=subentry.subentry_id,
        )


class NorisAITtsEntity(tts.TextToSpeechEntity, NorisAISubentryEntity):
    """noris AI speech synthesiser.

    Device identity comes from NorisAISubentryEntity. This does NOT extend
    NorisAIEntity, whose purpose is the chat-log loop, none of which applies to
    synthesis.
    """

    _attr_has_entity_name = False

    @property
    def name(self) -> str:
        """Return the entity name. HA's tts component rejects a None name."""
        return self.subentry.title

    @property
    def default_language(self) -> str:
        """Return the language used when a request names none."""
        return TTS_DEFAULT_LANGUAGE

    @property
    def supported_languages(self) -> list[str]:
        """Return the languages this model can be trusted with.

        Derived from the model name: the catalog reports no language data. See
        TTS_LANGUAGES_BY_PATTERN.
        """
        return tts_languages_for_model(self.model)

    async def async_get_tts_audio(
        self, message: str, language: str, options: dict[str, Any]
    ) -> tts.TtsAudioType:
        """Synthesise speech for a message.

        ``language`` and ``options`` are intentionally unused: the speech
        endpoint takes no language argument, and each model has exactly one
        voice (see TTS_VOICE), so there is nothing to pass through. A request
        naming e.g. "de-AT" is honoured only insofar as the chosen model's
        claimed languages (supported_languages) already cover it.
        """
        client = self.entry.runtime_data

        try:
            # response_format is sent for correctness even though the gateway
            # ignores it and always returns WAV. voice is required by the
            # gateway but its value is ignored — see TTS_VOICE.
            response = await client.with_options(
                timeout=TTS_TIMEOUT
            ).audio.speech.create(
                model=self.model,
                input=message,
                voice=TTS_VOICE,
                response_format="wav",
            )
        # AuthenticationError and PermissionDeniedError subclass APIStatusError,
        # which subclasses OpenAIError, so this ordering (auth -> APIStatusError
        # -> APITimeoutError -> OpenAIError) is load-bearing: reordering it
        # silently breaks the reauth path.
        except (openai.AuthenticationError, openai.PermissionDeniedError) as err:
            LOGGER.error("Authentication failed during synthesis: %s", err)
            self.entry.async_start_reauth(self.hass)
            raise HomeAssistantError("Authentication failed") from err
        except openai.APIStatusError as err:
            if err.status_code == 404:
                LOGGER.error(
                    "Model %s does not support speech synthesis. Reconfigure the "
                    "text-to-speech entity to use a speech model",
                    self.model,
                )
                raise HomeAssistantError(
                    f"Model {self.model} does not support speech synthesis"
                ) from err
            LOGGER.error("Error talking to speech API: %s", err)
            raise HomeAssistantError("Error talking to speech API") from err
        except openai.APITimeoutError as err:
            LOGGER.error("Speech synthesis timed out after %s seconds", TTS_TIMEOUT)
            raise HomeAssistantError("Speech synthesis timed out") from err
        except openai.OpenAIError as err:
            LOGGER.error("Error talking to speech API: %s", err)
            raise HomeAssistantError("Error talking to speech API") from err

        audio: bytes = response.content
        if not audio:
            LOGGER.error("Speech API returned an empty audio body")
            raise HomeAssistantError("Speech API returned no audio")

        # The message is user content and is never logged — only its length.
        LOGGER.debug(
            "Synthesised %d characters into %d bytes", len(message), len(audio)
        )
        return "wav", audio
