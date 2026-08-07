"""The noris AI integration."""

from __future__ import annotations

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

from .const import AUTH_HEADER, BASE_URL

PLATFORMS = [Platform.AI_TASK, Platform.CONVERSATION]

type NorisAIConfigEntry = ConfigEntry[AsyncOpenAI]


def _create_client(hass: HomeAssistant, api_key: str) -> AsyncOpenAI:
    """Create the AsyncOpenAI client for the ai.noris.de gateway.

    The gateway accepts standard Bearer authentication, so the key goes
    straight into ``api_key``. The legacy ``x-bf-vk`` header is still sent
    alongside during a transition period (see ``const.AUTH_HEADER``). TLS is
    verified through Home Assistant's shared httpx client.
    """
    return AsyncOpenAI(
        base_url=BASE_URL,
        api_key=api_key,
        default_headers={AUTH_HEADER: api_key},
        http_client=get_async_client(hass),
    )


async def _validate_api_key(client: AsyncOpenAI) -> None:
    """Validate the API key against the models endpoint.

    Raises ``AuthenticationError``/``PermissionDeniedError`` on a bad key,
    other ``OpenAIError`` subclasses on connection problems.
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
    entry.async_on_unload(entry.add_update_listener(_async_update_entry))
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    return True


async def _async_update_entry(hass: HomeAssistant, entry: NorisAIConfigEntry) -> None:
    """Reload the entry when its data or subentries change."""
    await hass.config_entries.async_reload(entry.entry_id)


async def async_unload_entry(hass: HomeAssistant, entry: NorisAIConfigEntry) -> bool:
    """Unload noris AI."""
    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
