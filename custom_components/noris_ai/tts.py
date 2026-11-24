"""Text-to-speech support for noris AI."""

from __future__ import annotations

import asyncio
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
    TTS_PASSTHROUGH_HEADER,
    TTS_PITCH_SEMITONES,
    TTS_PRONUNCIATIONS,
    TTS_SAMPLE_RATE,
    TTS_SUBENTRY_TYPE,
    TTS_TIMEOUT,
    TTS_VOICE,
    TTS_VOICEDESIGN_LANGUAGE,
    TTS_VOICEDESIGN_LOUDNESS,
    TTS_VOICEDESIGN_PATTERN,
    TTS_VOICEDESIGN_SPEED,
    TTS_VOICEDESIGN_STYLE,
    TTS_VOICEDESIGN_TASK_TYPE,
    TTS_VOICEDESIGN_VOICE,
)
from .entity import NorisAISubentryEntity
from .helpers import tts_languages_for_model

PARALLEL_UPDATES = 0


def spoken_text(message: str) -> str:
    """Rewrite product names the German voice would mangle."""
    for written, spoken in TTS_PRONUNCIATIONS:
        message = message.replace(written, spoken)
    return message


def pitch_filter(sample_rate: int = TTS_SAMPLE_RATE) -> str:
    """Return the ffmpeg filter that lowers pitch while keeping the tempo.

    asetrate resamples, which shifts pitch and speed together; atempo then
    undoes the speed change. One semitone is a factor of 2**(1/12).
    """
    factor = 2 ** (TTS_PITCH_SEMITONES / 12)
    return (
        f"asetrate={sample_rate}*{factor:.6f},"
        f"aresample={sample_rate},atempo={1 / factor:.6f}"
    )


def speech_request_kwargs(model: str) -> dict[str, Any]:
    """Return the per-model fields a speech request needs.

    VoiceDesign checkpoints carry no German presets: the timbre comes from the
    style text, so the request has to name task_type, instructions and language.
    The gateway treats all three as unknown fields and forwards them only with
    the passthrough header, hence the header travels with the request.
    """
    if not TTS_VOICEDESIGN_PATTERN.search(model):
        return {}
    return {
        "voice": TTS_VOICEDESIGN_VOICE,
        "extra_headers": {TTS_PASSTHROUGH_HEADER: "true"},
        "extra_body": {
            "task_type": TTS_VOICEDESIGN_TASK_TYPE,
            "instructions": TTS_VOICEDESIGN_STYLE,
            "language": TTS_VOICEDESIGN_LANGUAGE,
        },
    }


def voicedesign_filter(
    speed: float = TTS_VOICEDESIGN_SPEED,
    loudness: str = TTS_VOICEDESIGN_LOUDNESS,
) -> str | None:
    """Return the ffmpeg chain VoiceDesign audio needs, or None to skip.

    atempo changes tempo only (pitch stays untouched, unlike asetrate) and
    loudnorm lifts the checkpoint's low output to Kokoro's level.
    """
    filters: list[str] = []
    if speed > 0 and speed != 1:
        filters.append(f"atempo={speed:.6f}")
    if loudness:
        filters.append(loudness)
    return ",".join(filters) or None


async def _ffmpeg_filter(audio: bytes, audio_filter: str, purpose: str) -> bytes:
    """Return the WAV body with audio_filter applied, or the input on failure.

    Never raises: a missing or unhappy ffmpeg degrades to the original voice
    rather than silencing the assistant.
    """
    args = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-f",
        "wav",
        "-i",
        "pipe:0",
        "-af",
        audio_filter,
        "-ac",
        "1",
        "-ar",
        str(TTS_SAMPLE_RATE),
        "-f",
        "wav",
        "pipe:1",
    ]
    try:
        process = await asyncio.create_subprocess_exec(
            *args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except (FileNotFoundError, NotImplementedError) as err:
        LOGGER.warning("ffmpeg unavailable, keeping the original voice: %s", err)
        return audio

    filtered, errout = await process.communicate(audio)
    if process.returncode != 0 or not filtered:
        LOGGER.warning(
            "%s failed (rc=%s), keeping the original voice: %s",
            purpose,
            process.returncode,
            errout[:200],
        )
        return audio
    return filtered


async def shift_pitch(audio: bytes) -> bytes:
    """Return the WAV body with its pitch lowered, or the input unchanged."""
    if not TTS_PITCH_SEMITONES:
        return audio
    return await _ffmpeg_filter(audio, pitch_filter(), "Pitch shift")


async def postprocess_voicedesign(audio: bytes) -> bytes:
    """Return the WAV body faster and normalised, or the input unchanged."""
    audio_filter = voicedesign_filter()
    if audio_filter is None:
        return audio
    return await _ffmpeg_filter(audio, audio_filter, "VoiceDesign post-processing")


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

        # response_format is sent for correctness even though the gateway
        # ignores it and always returns WAV. voice is required by the gateway;
        # its value is ignored by Kokoro (see TTS_VOICE) but selects the
        # speaker on CosyVoice and VoiceDesign models.
        kwargs: dict[str, Any] = {
            "model": self.model,
            "input": spoken_text(message),
            "voice": TTS_VOICE,
            "response_format": "wav",
        }
        kwargs.update(speech_request_kwargs(self.model))

        try:
            response = await client.with_options(
                timeout=TTS_TIMEOUT
            ).audio.speech.create(**kwargs)
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
        audio = await shift_pitch(audio)
        if TTS_VOICEDESIGN_PATTERN.search(self.model):
            audio = await postprocess_voicedesign(audio)
        return "wav", audio
