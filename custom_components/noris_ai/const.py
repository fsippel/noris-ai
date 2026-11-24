"""Constants for the noris AI integration."""

import logging
import re

from homeassistant.const import CONF_LLM_HASS_API, CONF_PROMPT  # noqa: F401

DOMAIN = "noris_ai"
LOGGER = logging.getLogger(__package__)

# ai.noris.de is a Bifrost gateway exposing an OpenAI-compatible API.
# It accepts standard ``Authorization: *** auth (verified live), so the
# openai SDK works out of the box with ``api_key`` + ``base_url``.
BASE_URL = "https://ai.noris.de/v1"

DEFAULT_MODEL = "vllm/gpt-oss-120b"

CONF_RECOMMENDED = "recommended"
CONF_MAX_TOKENS = "max_tokens"
CONF_TEMPERATURE = "temperature"
CONF_TOP_P = "top_p"
CONF_ENABLE_REASONING = "enable_reasoning"

# Reasoning models spend tokens on thinking before answering; too small a
# budget yields empty content with finish_reason == "length".
RECOMMENDED_CONVERSATION_MAX_TOKENS = 3000
RECOMMENDED_AI_TASK_MAX_TOKENS = 8000
RECOMMENDED_TEMPERATURE = 1.0
RECOMMENDED_TOP_P = 1.0

CONVERSATION_SUBENTRY_TYPE = "conversation"
AI_TASK_SUBENTRY_TYPE = "ai_task_data"

STT_SUBENTRY_TYPE = "stt"
TTS_SUBENTRY_TYPE = "tts"

# Subentry titles. These become the device name, and Home Assistant slugifies
# the device name into the entity id, so they must read well in both places.
# The chosen model is surfaced separately as ``DeviceInfo.model``, which the UI
# renders beneath the name — titling a subentry with the raw model id instead
# leaves the device card with no indication of what the device actually is.
DEFAULT_CONVERSATION_NAME = "noris AI Conversation Agent"
DEFAULT_AI_TASK_NAME = "noris AI Task"
DEFAULT_STT_NAME = "noris AI Speech-to-text"
DEFAULT_TTS_NAME = "noris AI Text-to-speech"

# Model ids follow ``<provider>/<channel>/<slug>`` (e.g.
# ``vllm/release/glm-5-3``, ``Kokoro-TTS/release/kokoro-tts-german-martin``).
# The channel separates production from staging deployments of the same model.
# The default picker prefers production and only falls back to staging when a
# capability exists nowhere else, which is currently the case for transcription.
RELEASE_CHANNEL = "release"
STAGING_CHANNEL = "qsu"

# Preferred defaults are matched by id first and by suffix second, because the
# gateway renames models by adding channel prefixes.
PREFERRED_STT_MODEL = "vllm/qsu/voxtral-small-24b-2507"
PREFERRED_TTS_MODEL = "Kokoro-TTS/release/kokoro-tts-german-martin"

# Fallback transcription model. The qsu lane flaps: a pod rollout answers 503
# with an HTML "Application is not available" page in milliseconds, which used to
# fail the entire voice turn ("speech-to-text failed"). One retry on the stable
# model keeps the assistant usable; the switch is logged at warning level so a
# broken deployment never hides behind a working fallback.
STT_FALLBACK_MODEL = "vllm/qsu/voxtral-small-24b-2507"

# Circuit breaker: after the primary transcription model failed, the fallback
# serves ALONE for this long before the primary is probed again. Without it,
# every voice turn pays the doomed primary call first (~1.5 s of 503) while the
# upstream is down. Instance state, so a reload resets it harmlessly.
STT_BREAKER_COOLDOWN_S = 120.0

# Seconds to wait for a transcription before failing the pipeline. The gateway
# answers a five-second utterance in well under a second; this bound exists so a
# stalled gateway does not hang the voice satellite.
STT_TIMEOUT = 30.0

# Audio capability CANNOT be read from the model catalog: the gateway reports
# Voxtral's input_modalities as text-only even though transcription works.
# Known audio model families are matched by name instead. Do not "simplify" this
# to the modality metadata without re-probing /v1/models first.
#
# "transcribe" catches the Nemotron Transcribe family
# (vllm/qsu/nemotron-3.5-transcribe): the gateway never advertises it as an audio
# model, so without the literal match it is not selectable as an STT engine.
AUDIO_MODEL_PATTERN = re.compile(
    r"voxtral|whisper|qwen.*audio|granite-speech|vibevoice|nemotron.*transcribe|transcribe",
    re.IGNORECASE,
)

# Voxtral's documented languages, as BCP-47 tags. Deliberately not the ~130
# languages other cloud engines advertise: claiming unsupported languages
# produces confident, wrong transcripts instead of an honest failure to match.
#
# The bare ISO-639-1 codes come first because an Assist pipeline passes its
# pipeline language ("de"), not a regional tag, as the STT metadata language —
# a region-only list makes every pipeline run fail check_metadata with
# "provider does not support input metadata".
STT_SUPPORTED_LANGUAGES = [
    "de",
    "en",
    "es",
    "fr",
    "it",
    "nl",
    "pt",
    "hi",
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

# Seconds to wait for speech synthesis. Measured on the live gateway: ~1.0 s for
# a short German sentence via Kokoro. The bound exists so a stalled gateway
# fails promptly instead of hanging a voice satellite.
TTS_TIMEOUT = 30.0

# Speech capability CANNOT be read from the model catalog either: every audio
# model is stamped with the default text-in/text-out chat template, so speech
# models are matched by name. Delete this in favour of output_modalities once
# the catalog reports an "audio" output type. Do not "simplify" this to the
# modality metadata without re-probing /v1/models first.
TTS_MODEL_PATTERN = re.compile(
    r"kokoro|cosyvoice|orpheus|xtts|bark|-tts/|tts-", re.IGNORECASE
)

# The gateway REQUIRES a voice field — omitting it returns
# 400 "voice is required for speech completion". Kokoro ignores the value
# ("martin", "alloy" and a nonsense string return byte-identical audio), but
# CosyVoice3 does not: "default" never answers (90 s timeout) while a real
# name answers in ~3 s, and the name selects the speaker. German is what the
# house speaks, so the German speaker is the default.
TTS_VOICE = "default"

# Qwen3-TTS VoiceDesign takes three fields the gateway does not know: task_type,
# the style text (instructions) and language. It treats them as unknown and
# forwards them ONLY when the client opts in with this header (measured
# 2026-11-24: identical payload -> 400 "does not support task_type='CustomVoice'"
# without the header, 200 + WAV with it). The header value must be exactly
# "true". Sending it per request keeps the opt-in local to speech synthesis.
TTS_PASSTHROUGH_HEADER = "x-bf-passthrough-extra-params"
TTS_VOICEDESIGN_PATTERN = re.compile(r"qwen3.*tts.*voicedesign", re.IGNORECASE)
TTS_VOICEDESIGN_VOICE = "vivian"
TTS_VOICEDESIGN_TASK_TYPE = "VoiceDesign"
TTS_VOICEDESIGN_LANGUAGE = "German"
# Style is the only lever for the timbre: the checkpoint ships no German
# presets, so the description is the voice.
TTS_VOICEDESIGN_STYLE = (
    "Eine warme, tiefe deutsche Männerstimme, ruhig und zugewandt, zügiges Tempo."
)
# The style text nudges tempo only weakly and not reproducibly; atempo makes it
# measurable (1.10 ≈ 10 % shorter utterance). VoiceDesign lands 4-6 dB below
# Kokoro, so one loudnorm pass puts it on par (-18.5 vs -18.7 dB mean) without
# a manual media_player gain. Both filters cost ~100 ms together.
TTS_VOICEDESIGN_SPEED = 1.10
TTS_VOICEDESIGN_LOUDNESS = "loudnorm=I=-16:TP=-1.5:LRA=11"

# Optional post-processing with ffmpeg (asetrate → pitch, atempo → tempo
# restored) because the gateway exposes no pitch parameter. 0 disables it —
# and 0 is what a low-latency pipeline wants: every extra ffmpeg pass costs
# time-to-first-audio. Kokoro's German voice needs no correction.
TTS_PITCH_SEMITONES = 0.0

# Native output format of every speech model on the gateway, measured.
TTS_SAMPLE_RATE = 24000

# The German Kokoro voice reads English product names with German phonemes
# ("Home Assistant" becomes "Hohme Assistent"). Rewriting them into German
# phonetics before synthesis is the only lever — the gateway has no language
# switching per request. Order matters: longer phrases first.
TTS_PRONUNCIATIONS: tuple[tuple[str, str], ...] = (
    ("Home Assistant", "Houm Ässisstänt"),
    ("Home-Assistant", "Houm Ässisstänt"),
)

# Hint sent with every transcription. NOTE (measured 2026-11-24): the gateway
# drops this field — a nonsense prompt yields a byte-identical transcript, and
# hotwords/word_boosting/context/initial_prompt are ignored too. Kept only so a
# future model that honours it gets the vocabulary for free.
STT_PROMPT = (
    "Transkribiere wörtlich auf Deutsch. Fachbegriffe: Photovoltaik, PV-Ertrag, "
    "Kilowattstunden, Kilowatt, Speicher, Netzfluss, Küche, Schlafzimmer, "
    "Wohnzimmer, Esszimmer, Rolle."
)

# Anything longer than this is not a spoken command but the model answering the
# question; such transcripts are discarded (see stt.py).
STT_MAX_TRANSCRIPT_CHARS = 160

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

# The Assist pipeline asks for a language even when the caller names none.
# German first, matching the deployments this integration was built for; the
# entity's supported_languages still advertises every language of its model.
TTS_DEFAULT_LANGUAGE = "de"
