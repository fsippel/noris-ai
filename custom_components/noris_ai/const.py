"""Constants for the noris AI integration."""

import logging

from homeassistant.const import CONF_LLM_HASS_API, CONF_PROMPT  # noqa: F401

DOMAIN = "noris_ai"
LOGGER = logging.getLogger(__package__)

# ai.noris.de is a Bifrost gateway exposing an OpenAI-compatible API.
# It accepts standard ``Authorization: Bearer`` auth (verified live), so the
# openai SDK works out of the box with ``api_key`` + ``base_url``.
BASE_URL = "https://ai.noris.de/v1"

# Transitional: releases up to 0.2.x authenticated via this custom header.
# We keep sending it alongside the Bearer header so any gateway config that
# still only reads ``x-bf-vk`` keeps working; drop in a future release.
AUTH_HEADER = "x-bf-vk"

DEFAULT_MODEL = "vllm/gpt-oss-120b"

CONF_RECOMMENDED = "recommended"
CONF_MAX_TOKENS = "max_tokens"
CONF_TEMPERATURE = "temperature"
CONF_TOP_P = "top_p"

# Reasoning models spend tokens on thinking before answering; too small a
# budget yields empty content with finish_reason == "length".
RECOMMENDED_CONVERSATION_MAX_TOKENS = 3000
RECOMMENDED_AI_TASK_MAX_TOKENS = 8000
RECOMMENDED_TEMPERATURE = 1.0
RECOMMENDED_TOP_P = 1.0

CONVERSATION_SUBENTRY_TYPE = "conversation"
AI_TASK_SUBENTRY_TYPE = "ai_task_data"
