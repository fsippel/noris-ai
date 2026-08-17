"""The noris AI integration."""

from __future__ import annotations

from openai import AsyncOpenAI, AuthenticationError, OpenAIError, PermissionDeniedError

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_API_KEY, Platform
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryAuthFailed, ConfigEntryNotReady
from homeassistant.helpers.httpx_client import get_async_client

from .const import (
    AI_TASK_SUBENTRY_TYPE,
    AUTH_HEADER,
    BASE_URL,
    CONVERSATION_SUBENTRY_TYPE,
    DEFAULT_AI_TASK_NAME,
    DEFAULT_CONVERSATION_NAME,
    DEFAULT_STT_NAME,
    DEFAULT_TTS_NAME,
    LOGGER,
    STT_SUBENTRY_TYPE,
    TTS_SUBENTRY_TYPE,
)

PLATFORMS = [Platform.AI_TASK, Platform.CONVERSATION, Platform.STT, Platform.TTS]

# Subentries created before 1.2 were titled with the raw model id, which left
# the device card showing no indication of what the device was.
SUBENTRY_DEFAULT_TITLES = {
    CONVERSATION_SUBENTRY_TYPE: DEFAULT_CONVERSATION_NAME,
    AI_TASK_SUBENTRY_TYPE: DEFAULT_AI_TASK_NAME,
    STT_SUBENTRY_TYPE: DEFAULT_STT_NAME,
    TTS_SUBENTRY_TYPE: DEFAULT_TTS_NAME,
}

type NorisAIConfigEntry = ConfigEntry[AsyncOpenAI]


def _create_client(hass: HomeAssistant, api_key: str) -> AsyncOpenAI:
    """Create the AsyncOpenAI client used by this integration.

    The ai.noris.de gateway authenticates via a custom ``x-bf-vk`` header
    instead of the standard ``Authorization: Bearer`` header, so the key is
    passed through ``default_headers``. ``api_key`` is still required by the
    SDK and carries the same value; the gateway simply reads ``x-bf-vk``.
    TLS certificates are verified through Home Assistant's shared httpx client.
    """
    return AsyncOpenAI(
        base_url=BASE_URL,
        api_key=api_key,
        default_headers={AUTH_HEADER: api_key},
        http_client=get_async_client(hass),
    )


async def _validate_api_key(client: AsyncOpenAI) -> None:
    """Validate the API key against the models endpoint.

    A valid key returns the model list; an invalid key raises
    ``AuthenticationError`` (401) or ``PermissionDeniedError`` (403), which
    propagate to the caller.
    """
    await client.with_options(timeout=10.0).models.list()


async def async_migrate_entry(hass: HomeAssistant, entry: NorisAIConfigEntry) -> bool:
    """Migrate an old config entry.

    1.1 → 1.2 retitles subentries that still carry the raw model id. A title the
    user chose themselves is left alone: only titles that look like a gateway
    model id (``vllm/…``) are replaced. Entity ids are deliberately NOT touched
    — they may already be referenced by pipelines, automations or dashboards,
    and Home Assistant does not follow such references.
    """
    if entry.minor_version >= 2:
        return True

    for subentry in entry.subentries.values():
        default_title = SUBENTRY_DEFAULT_TITLES.get(subentry.subentry_type)
        if default_title is None or not subentry.title.startswith("vllm/"):
            continue
        LOGGER.debug(
            "Retitling %s subentry from %s to %s",
            subentry.subentry_type,
            subentry.title,
            default_title,
        )
        hass.config_entries.async_update_subentry(entry, subentry, title=default_title)

    hass.config_entries.async_update_entry(entry, minor_version=2)
    return True


async def async_setup_entry(hass: HomeAssistant, entry: NorisAIConfigEntry) -> bool:
    """Set up noris AI from a config entry."""
    client = _create_client(hass, entry.data[CONF_API_KEY])

    try:
        await _validate_api_key(client)
    except (AuthenticationError, PermissionDeniedError) as err:
        raise ConfigEntryAuthFailed(err) from err
    except OpenAIError as err:
        raise ConfigEntryNotReady(err) from err

    entry.runtime_data = client

    entry.async_on_unload(entry.add_update_listener(async_update_entry))

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    return True


async def async_update_entry(hass: HomeAssistant, entry: NorisAIConfigEntry) -> None:
    """Reload the entry when its data or subentries change."""
    await hass.config_entries.async_reload(entry.entry_id)


async def async_unload_entry(hass: HomeAssistant, entry: NorisAIConfigEntry) -> bool:
    """Unload noris AI."""
    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
