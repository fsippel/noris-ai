"""Constants for the noris AI integration."""

import logging

from homeassistant.const import CONF_LLM_HASS_API, CONF_PROMPT  # noqa: F401
from homeassistant.helpers import llm

DOMAIN = "noris_ai"
LOGGER = logging.getLogger(__package__)

# ai.noris.de is an OpenAI-compatible gateway. Note that authentication does
# NOT use the standard ``Authorization: Bearer`` header but a custom
# ``x-bf-vk`` header (see ``__init__._create_client``).
BASE_URL = "https://ai.noris.de/v1"

# Custom authentication header used by the ai.noris.de gateway.
AUTH_HEADER = "x-bf-vk"

# Bounds every request the shared client makes. Handing the SDK Home
# Assistant's shared httpx client is NOT enough on its own: that client carries
# httpx's *default* timeout, and the SDK's check is structural — a client whose
# timeout equals the httpx default counts as "no timeout given" and is replaced
# by the SDK's own 600 s default, with two retries on top. A gateway that
# accepts the connection and then never answers then holds an Assist pipeline
# in "processing" for minutes while the voice satellite refuses to wake. A
# voice turn that takes longer than a minute is useless anyway, so fail it.
REQUEST_TIMEOUT = 60.0
CONNECT_TIMEOUT = 5.0

# One retry, not the SDK's two: a retried voice turn costs another full
# REQUEST_TIMEOUT before the user hears anything.
MAX_RETRIES = 1

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
