"""Speech-to-text support for noris AI."""

from __future__ import annotations

from collections.abc import AsyncIterable
import time

import openai

from homeassistant.components import stt
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from . import NorisAIConfigEntry
from .const import (
    LOGGER,
    STT_BREAKER_COOLDOWN_S,
    STT_FALLBACK_MODEL,
    STT_MAX_TRANSCRIPT_CHARS,
    STT_PROMPT,
    STT_SUBENTRY_TYPE,
    STT_SUPPORTED_LANGUAGES,
    STT_TIMEOUT,
)
from .entity import NorisAISubentryEntity
from .helpers import pcm_to_wav

PARALLEL_UPDATES = 0


def _looks_like_an_answer(text: str) -> bool:
    """Return True for output that cannot be a spoken command.

    Voxtral answers the question instead of transcribing it often enough to
    matter ("Wie hoch ist der PV Ertrag" produced a multi-paragraph essay with
    markdown headings). Such a text is never a thing the user said: it is long,
    structured, or both.
    """
    if len(text) > STT_MAX_TRANSCRIPT_CHARS:
        return True
    markers = ("\n\n", "###", "**", "\n- ", "\n1. ", "\n2. ")
    return any(marker in text for marker in markers)


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: NorisAIConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up speech-to-text entities."""
    for subentry in config_entry.get_subentries_of_type(STT_SUBENTRY_TYPE):
        async_add_entities(
            [NorisAISttEntity(config_entry, subentry)],
            config_subentry_id=subentry.subentry_id,
        )


class NorisAISttEntity(stt.SpeechToTextEntity, NorisAISubentryEntity):
    """noris AI speech-to-text engine.

    Device identity comes from NorisAISubentryEntity. This does NOT extend
    NorisAIEntity, whose purpose is the chat-log loop (tools, thinking
    extraction) — none of which applies to transcription.
    """

    _attr_has_entity_name = True
    _attr_name = None

    @property
    def supported_languages(self) -> list[str]:
        """Return the languages the transcription model supports."""
        return list(STT_SUPPORTED_LANGUAGES)

    @property
    def supported_formats(self) -> list[stt.AudioFormats]:
        """Return the supported audio formats."""
        return [stt.AudioFormats.WAV]

    @property
    def supported_codecs(self) -> list[stt.AudioCodecs]:
        """Return the supported audio codecs."""
        return [stt.AudioCodecs.PCM]

    @property
    def supported_bit_rates(self) -> list[stt.AudioBitRates]:
        """Return the supported bit rates."""
        return [stt.AudioBitRates.BITRATE_16]

    @property
    def supported_sample_rates(self) -> list[stt.AudioSampleRates]:
        """Return the supported sample rates."""
        return [stt.AudioSampleRates.SAMPLERATE_16000]

    @property
    def supported_channels(self) -> list[stt.AudioChannels]:
        """Return the supported channel counts."""
        return [stt.AudioChannels.CHANNEL_MONO]

    async def async_process_audio_stream(
        self, metadata: stt.SpeechMetadata, stream: AsyncIterable[bytes]
    ) -> stt.SpeechResult:
        """Transcribe an audio stream.

        Never raises: a failure is reported as an ERROR result so the pipeline
        can tell the user, rather than propagating into the voice satellite.
        """
        try:
            audio = bytearray()
            async for chunk in stream:
                audio.extend(chunk)

            if not audio:
                LOGGER.debug("Empty audio stream, nothing to transcribe")
                return stt.SpeechResult(None, stt.SpeechResultState.ERROR)

            # The gateway rejects headerless PCM with "Invalid or unsupported
            # audio file", so the samples need a WAV container.
            wav_audio = pcm_to_wav(
                bytes(audio),
                metadata.sample_rate.value,
                metadata.bit_rate.value // 8,
                metadata.channel.value,
            )
        except Exception as err:
            # The pipeline stream and the WAV encoder can both fail in ways
            # this integration cannot enumerate; any of them must still
            # yield ERROR, never a raise.
            LOGGER.error("Error reading or encoding the audio stream: %s", err)
            return stt.SpeechResult(None, stt.SpeechResultState.ERROR)

        client = self.entry.runtime_data

        # The qsu lane flaps: a pod rollout answers 503 with an HTML
        # "Application is not available" page within milliseconds, and that used
        # to kill the whole voice turn. Worse, one model can vanish from the
        # catalog while the other one serves — and then both can sit behind the
        # same upstream. So: try the configured model, then the fallback, and if
        # there is no distinct fallback configured, give the same model one
        # second chance. Every deviation is logged, because a silent model switch
        # would hide a broken deployment.
        #
        # While the primary is down, every turn would pay the doomed call first
        # (~1.5 s of 503 before the fallback even starts). A small circuit
        # breaker stops that: after the primary failed, the fallback serves
        # alone for STT_BREAKER_COOLDOWN_S, then the primary is probed again.
        breaker_open = getattr(self, "_breaker_open_until", 0.0) > time.monotonic()
        has_distinct_fallback = STT_FALLBACK_MODEL != self.model

        if breaker_open and has_distinct_fallback:
            candidates = [STT_FALLBACK_MODEL]
        else:
            candidates = [self.model]
            candidates.append(
                STT_FALLBACK_MODEL if has_distinct_fallback else self.model
            )

        response = None
        successful_model = None
        primary_was_attempted = False
        for attempt, model in enumerate(candidates):
            primary_was_attempted = primary_was_attempted or model == self.model
            try:
                # response_format is left at its default: the gateway mishandles
                # "text" by nesting the whole JSON body inside the text field.
                # max_retries=0 hands recovery to the loop below: the SDK's
                # default (2 retries) amplified every doomed 503 call to three
                # round trips before the fallback even started.
                response = await client.with_options(
                    timeout=STT_TIMEOUT, max_retries=0
                ).audio.transcriptions.create(
                    file=("audio.wav", wav_audio, "audio/wav"),
                    model=model,
                    language=metadata.language.split("-")[0],
                    # temperature 0 keeps the model on the transcription task.
                    # The prompt is sent for forward compatibility only: measured
                    # 2026-11-24, this gateway appears to DROP it — a garbage
                    # prompt produced a byte-identical transcript, and so did
                    # hotwords/word_boosting/context/initial_prompt (all ignored).
                    # Do not rely on prompt-based vocabulary biasing here.
                    temperature=0,
                    prompt=STT_PROMPT,
                )
                successful_model = model
                if attempt and model == self.model:
                    LOGGER.warning(
                        "Transcription succeeded on a retry with %s: the first "
                        "attempt hit a transient upstream error",
                        model,
                    )
                elif attempt:
                    LOGGER.warning(
                        "Transcribed with fallback model %s: configured model %s "
                        "is unavailable. Fix the deployment or reconfigure the "
                        "speech-to-text entity",
                        model,
                        self.model,
                    )
                break
            # AuthenticationError and PermissionDeniedError subclass
            # APIStatusError, which subclasses OpenAIError, so this ordering
            # (auth -> APIStatusError -> APITimeoutError -> OpenAIError) is
            # load-bearing: reordering it silently breaks the reauth path.
            except (openai.AuthenticationError, openai.PermissionDeniedError) as err:
                LOGGER.error("Authentication failed during transcription: %s", err)
                self.entry.async_start_reauth(self.hass)
                return stt.SpeechResult(None, stt.SpeechResultState.ERROR)
            except openai.APIStatusError as err:
                if err.status_code == 404:
                    LOGGER.error(
                        "Model %s does not support transcription. Reconfigure the "
                        "speech-to-text entity to use an audio model",
                        model,
                    )
                else:
                    LOGGER.error(
                        "Error talking to transcription API (%s): %s", model, err
                    )
            except openai.APITimeoutError:
                # No fallback attempt: the pipeline already burned its budget,
                # a second full timeout would double the silence.
                LOGGER.error("Transcription timed out after %s seconds", STT_TIMEOUT)
                return stt.SpeechResult(None, stt.SpeechResultState.ERROR)
            except openai.OpenAIError as err:
                LOGGER.error("Error talking to transcription API (%s): %s", model, err)

        if response is None:
            if primary_was_attempted and has_distinct_fallback:
                # Open the breaker: the next turns go straight to the fallback
                # instead of paying the doomed primary call first.
                self._breaker_open_until = time.monotonic() + STT_BREAKER_COOLDOWN_S
                LOGGER.info(
                    "Primary transcription model %s is unavailable; using "
                    "fallback %s alone for %s s, then trying the primary again",
                    self.model,
                    STT_FALLBACK_MODEL,
                    STT_BREAKER_COOLDOWN_S,
                )
            return stt.SpeechResult(None, stt.SpeechResultState.ERROR)

        if successful_model == self.model:
            self._breaker_open_until = 0.0
        elif primary_was_attempted and has_distinct_fallback:
            self._breaker_open_until = time.monotonic() + STT_BREAKER_COOLDOWN_S
            LOGGER.info(
                "Primary transcription model %s failed once; using fallback %s "
                "alone for %s s, then trying the primary again",
                self.model,
                STT_FALLBACK_MODEL,
                STT_BREAKER_COOLDOWN_S,
            )

        text = (response.text or "").strip()
        if not text:
            LOGGER.warning("Transcription returned an empty result")
            return stt.SpeechResult(None, stt.SpeechResultState.ERROR)

        if _looks_like_an_answer(text):
            # Voxtral occasionally answers the question instead of transcribing
            # it (whole paragraphs, markdown headings). Acting on that would
            # send nonsense into the pipeline, so report a failed transcription
            # and let the satellite say it did not understand.
            LOGGER.warning(
                "Discarded a %d character transcript: it reads like an answer, "
                "not a transcription",
                len(text),
            )
            return stt.SpeechResult(None, stt.SpeechResultState.ERROR)

        # Transcripts are voice content: debug level only, never higher.
        LOGGER.debug("Transcribed %d bytes of audio", len(audio))
        return stt.SpeechResult(text, stt.SpeechResultState.SUCCESS)
