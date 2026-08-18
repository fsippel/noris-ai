"""The noris AI integration."""

from __future__ import annotations

import httpx
from openai import (
    AsyncOpenAI,
    AuthenticationError,
    OpenAIError,
    PermissionDeniedError,
)

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_API_KEY, Platform
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryAuthFailed, ConfigEntryNotReady
from homeassistant.helpers.httpx_client import get_async_client

from .const import AUTH_HEADER, BASE_URL, CONNECT_TIMEOUT, MAX_RETRIES, REQUEST_TIMEOUT

PLATFORMS = [Platform.AI_TASK, Platform.CONVERSATION]

type NorisAIConfigEntry = ConfigEntry[AsyncOpenAI]


def _create_client(hass: HomeAssistant, api_key: str) -> AsyncOpenAI:
    """Create the AsyncOpenAI client used by this integration.

    The ai.noris.de gateway authenticates via a custom ``x-bf-vk`` header
    instead of the standard ``Authorization: Bearer`` header, so the key is
    passed through ``default_headers``. ``api_key`` is still required by the
    SDK and carries the same value; the gateway simply reads ``x-bf-vk``.
    TLS certificates are verified through Home Assistant's shared httpx client.

    ``timeout`` and ``max_retries`` must be passed explicitly. The shared httpx
    client carries httpx's default timeout, which the SDK reads as "unset" and
    replaces with its own 600 s default — see ``REQUEST_TIMEOUT``.
    """
    return AsyncOpenAI(
        base_url=BASE_URL,
        api_key=api_key,
        default_headers={AUTH_HEADER: api_key},
        http_client=get_async_client(hass),
        timeout=httpx.Timeout(REQUEST_TIMEOUT, connect=CONNECT_TIMEOUT),
        max_retries=MAX_RETRIES,
    )


async def _validate_api_key(client: AsyncOpenAI) -> None:
    """Validate the API key against the models endpoint.

    A valid key returns the model list; an invalid key raises
    ``AuthenticationError`` (401) or ``PermissionDeniedError`` (403), which
    propagate to the caller.
    """
    await client.with_options(timeout=10.0).models.list()


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
