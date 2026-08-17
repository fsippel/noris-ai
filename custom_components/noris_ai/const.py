"""Constants for the noris AI integration."""

import logging
import re

from homeassistant.const import CONF_LLM_HASS_API, CONF_PROMPT
from homeassistant.helpers import llm

DOMAIN = "noris_ai"
LOGGER = logging.getLogger(__package__)

# ai.noris.de is an OpenAI-compatible gateway. Note that authentication does
# NOT use the standard ``Authorization: Bearer`` header but a custom
# ``x-bf-vk`` header (see ``__init__._create_client``).
BASE_URL = "https://ai.noris.de/v1"

# Custom authentication header used by the ai.noris.de gateway.
AUTH_HEADER = "x-bf-vk"

CONF_MAX_TOKENS = "max_tokens"
CONF_TEMPERATURE = "temperature"

CONVERSATION_SUBENTRY_TYPE = "conversation"
AI_TASK_SUBENTRY_TYPE = "ai_task_data"

RECOMMENDED_CONVERSATION_OPTIONS = {
    CONF_LLM_HASS_API: [llm.LLM_API_ASSIST],
    CONF_PROMPT: llm.DEFAULT_INSTRUCTIONS_PROMPT,
}

# AI Task entities only need a model: the ai_task base class supplies the
# system prompt and the user instructions itself, so a prompt option here
# would have no effect.

STT_SUBENTRY_TYPE = "stt"

# Subentry titles. These become the device name, and Home Assistant slugifies
# the device name into the entity id, so they must read well in both places.
# The chosen model is surfaced separately as ``DeviceInfo.model``, which the UI
# renders beneath the name — titling a subentry with the raw model id instead
# leaves the device card with no indication of what the device actually is.
DEFAULT_CONVERSATION_NAME = "noris AI Conversation Agent"
DEFAULT_AI_TASK_NAME = "noris AI Task"
DEFAULT_STT_NAME = "noris AI Speech-to-text"
DEFAULT_TTS_NAME = "noris AI Text-to-speech"

# Seconds to wait for a transcription before failing the pipeline. The gateway
# answers a five-second utterance in well under a second; this bound exists so a
# stalled gateway does not hang the voice satellite.
STT_TIMEOUT = 30.0

# Audio capability CANNOT be read from the model catalog: the gateway reports
# Voxtral's input_modalities as text-only even though transcription works.
# Known audio model families are matched by name instead. Do not "simplify" this
# to the modality metadata without re-probing /v1/models first.
AUDIO_MODEL_PATTERN = re.compile(
    r"voxtral|whisper|qwen.*audio|granite-speech", re.IGNORECASE
)

# Voxtral's documented languages, as BCP-47 tags. Deliberately not the ~130
# languages other cloud engines advertise: claiming unsupported languages
# produces confident, wrong transcripts instead of an honest failure to match.
STT_SUPPORTED_LANGUAGES = [
    "de-DE",
    "de-AT",
    "de-CH",
    "en-US",
    "en-GB",
    "en-AU",
    "en-CA",
    "en-IE",
    "en-IN",
    "en-NZ",
    "en-ZA",
    "es-ES",
    "es-MX",
    "es-AR",
    "es-CO",
    "es-US",
    "fr-FR",
    "fr-BE",
    "fr-CA",
    "fr-CH",
    "it-IT",
    "it-CH",
    "nl-NL",
    "nl-BE",
    "pt-PT",
    "pt-BR",
    "hi-IN",
]

TTS_SUBENTRY_TYPE = "tts"

# Seconds to wait for speech synthesis. Measured on the live gateway: ~1.1 s for
# a short German sentence via Kokoro, ~1.9 s via CosyVoice3. The bound exists so
# a stalled gateway fails promptly instead of hanging a voice satellite.
TTS_TIMEOUT = 30.0

# Speech capability CANNOT be read from the model catalog. As with
# AUDIO_MODEL_PATTERN, every audio model is stamped with the default
# text-in/text-out chat template, so speech models are matched by name.
# Delete this in favour of output_modalities once the catalog reports an
# "audio" output type. Do not "simplify" this to the modality metadata
# without re-probing /v1/models first.
TTS_MODEL_PATTERN = re.compile(
    r"kokoro|cosyvoice|orpheus|xtts|bark|-tts/|tts-", re.IGNORECASE
)

# The gateway REQUIRES a voice field — omitting it returns
# 400 "voice is required for speech completion" — but ignores its value:
# "martin", "alloy" and a nonsense string all returned byte-identical audio.
# Each model has exactly one voice, so no voice picker is offered and this
# placeholder is sent to satisfy the API.
TTS_VOICE = "default"

# Home Assistant matches languages by EXACT string membership, with no region
# fallback, so each claimed language needs its regional variants declared too.
TTS_GERMAN = ["de", "de-DE", "de-AT", "de-CH"]
TTS_ENGLISH = ["en", "en-US", "en-GB", "en-AU", "en-CA"]

# The catalog exposes no language data either, so coverage is derived from the
# model name. Verified by round trip — synthesise, then transcribe back: the
# German Kokoro voice mangles English phonetically ("In turn de on in de de
# ketien in light de pliaze"), while CosyVoice3 handled German and English
# cleanly. Only verified languages are claimed; over-claiming would produce
# confident, wrong pronunciation rather than an honest failure to match.
# First match wins, so an ambiguous name resolves to the narrower claim.
TTS_LANGUAGES_BY_PATTERN = (
    (re.compile(r"german|deutsch", re.IGNORECASE), TTS_GERMAN),
    (re.compile(r"english|englisch", re.IGNORECASE), TTS_ENGLISH),
    (re.compile(r"cosyvoice", re.IGNORECASE), TTS_GERMAN + TTS_ENGLISH),
)
TTS_DEFAULT_LANGUAGES = TTS_GERMAN + TTS_ENGLISH
TTS_DEFAULT_LANGUAGE = "de"
