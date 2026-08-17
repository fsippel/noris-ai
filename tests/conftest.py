"""Fixtures for the noris AI integration tests."""

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.noris_ai.const import (
    AI_TASK_SUBENTRY_TYPE,
    CONVERSATION_SUBENTRY_TYPE,
    DEFAULT_AI_TASK_NAME,
    DEFAULT_CONVERSATION_NAME,
    DEFAULT_STT_NAME,
    DEFAULT_TTS_NAME,
    DOMAIN,
    STT_SUBENTRY_TYPE,
    TTS_SUBENTRY_TYPE,
)
from homeassistant.config_entries import ConfigSubentryData
from homeassistant.const import CONF_API_KEY, CONF_MODEL
from homeassistant.core import HomeAssistant
from homeassistant.setup import async_setup_component

CHAT_MODEL = "vllm/release/gpt-oss-120b"
AUDIO_MODEL = "vllm/qsu/voxtral-small-24b-2507"
SPEECH_MODEL = "Cosyvoice3/release/cosyvoice3-0.5b-rl"


@pytest.fixture(autouse=True)
def auto_enable_custom_integrations(enable_custom_integrations: None) -> None:
    """Enable loading of this custom integration in every test."""
    return


@pytest.fixture
def hass_config_dir(hass_tmp_config_dir: str) -> str:
    """Use a fresh, per-test config directory instead of the plugin's shared one.

    Without this override every test shares
    ``pytest_homeassistant_custom_component/testing_config`` as ``hass.config``,
    including its ``tts/`` disk cache. The tts component's
    ``async_get_media_source_audio`` path checks that disk cache *before*
    calling the entity, keyed only on message+language+options+engine — so a
    synthesis test run once leaves a file behind that lets an identical test
    pass on every subsequent run without the mocked API ever being called
    again, silently defeating the test. ``hass_tmp_config_dir`` (from the
    plugin) copies the shared config into a pytest ``tmp_path`` that is
    discarded after the test.
    """
    return hass_tmp_config_dir


def fake_model(
    model_id: str,
    *,
    output_type: str | None = "text",
    hugging_face_id: str | None = None,
) -> SimpleNamespace:
    """Build a stand-in for an OpenAI SDK Model.

    The gateway returns fields the SDK's Model type does not declare, so they
    live in ``model_extra`` exactly as the SDK exposes them.
    """
    extra: dict[str, Any] = {"hugging_face_id": hugging_face_id}
    if output_type is not None:
        extra["output_modalities"] = [{"type": output_type}]
    return SimpleNamespace(id=model_id, model_extra=extra)


class FakeModelList:
    """Stand-in for the SDK's AsyncPaginator: awaitable and async-iterable.

    ``_validate_api_key`` awaits ``models.list()`` while ``_fetch_model_options``
    iterates it, so the fake has to support both.
    """

    def __init__(self, models: list[Any]) -> None:
        """Store the models this paginator yields."""
        self._models = models

    def __await__(self):
        """Allow ``await client.models.list()``."""

        async def _self() -> FakeModelList:
            return self

        return _self().__await__()

    def __aiter__(self):
        """Allow ``async for model in client.models.list()``."""

        async def _gen():
            for model in self._models:
                yield model

        return _gen()


@pytest.fixture
def default_models() -> list[Any]:
    """Return a model catalog resembling the live gateway."""
    return [
        fake_model(CHAT_MODEL),
        fake_model("vllm/qsu/glm-5-2"),
        fake_model(
            "vllm/qsu/voxtral-small-24b-2507",
            hugging_face_id="mistralai/Voxtral-Small-24B-2507",
        ),
        fake_model(SPEECH_MODEL),
        fake_model("Kokoro-TTS/release/kokoro-tts-german-martin"),
        # Deliberately NOT ids the legacy substring fallback ("reranker" /
        # "harrier") would also catch: these ids must be excluded solely by
        # output_modalities, so a broken metadata check shows up here.
        fake_model("vllm/release/bge-m3", output_type="rerank"),
        fake_model("vllm/release/e5-large", output_type="embeddings"),
    ]


@pytest.fixture
def mock_client(default_models: list[Any]) -> Iterator[AsyncMock]:
    """Patch the AsyncOpenAI client in both modules that reference it.

    ``config_flow`` does ``from . import _create_client``, which binds the name
    into its own namespace at import time, so patching only the package
    attribute would leave the flow using the real client.
    """
    client = AsyncMock()
    client.with_options = MagicMock(return_value=client)
    client.models.list = MagicMock(return_value=FakeModelList(default_models))

    with (
        patch("custom_components.noris_ai._create_client", return_value=client),
        patch(
            "custom_components.noris_ai.config_flow._create_client",
            return_value=client,
        ),
    ):
        yield client


@pytest.fixture
def mock_config_entry() -> MockConfigEntry:
    """Return a config entry with conversation, AI task, STT, and TTS subentries."""
    return MockConfigEntry(
        domain=DOMAIN,
        title="noris AI",
        data={CONF_API_KEY: "sk-bf-test"},
        subentries_data=[
            ConfigSubentryData(
                data={CONF_MODEL: CHAT_MODEL},
                subentry_type=CONVERSATION_SUBENTRY_TYPE,
                title=DEFAULT_CONVERSATION_NAME,
                unique_id=None,
            ),
            ConfigSubentryData(
                data={CONF_MODEL: CHAT_MODEL},
                subentry_type=AI_TASK_SUBENTRY_TYPE,
                title=DEFAULT_AI_TASK_NAME,
                unique_id=None,
            ),
            ConfigSubentryData(
                data={CONF_MODEL: AUDIO_MODEL},
                subentry_type=STT_SUBENTRY_TYPE,
                title=DEFAULT_STT_NAME,
                unique_id=None,
            ),
            ConfigSubentryData(
                data={CONF_MODEL: SPEECH_MODEL},
                subentry_type=TTS_SUBENTRY_TYPE,
                title=DEFAULT_TTS_NAME,
                unique_id=None,
            ),
        ],
    )


async def setup_integration(
    hass: HomeAssistant, entry: MockConfigEntry
) -> MockConfigEntry:
    """Add the config entry to hass and set it up.

    The ``homeassistant`` core component is set up first because
    ``conversation``'s default agent (an ``after_dependencies`` of this
    integration's platforms) reads ``hass.data[DATA_EXPOSED_ENTITIES]``, which
    only exists once ``homeassistant`` has run its own setup. In production
    that component is always loaded first as part of core bootstrap; the test
    ``hass`` fixture starts already in ``CoreState.running`` without it.
    """
    await async_setup_component(hass, "homeassistant", {})
    entry.add_to_hass(hass)
    await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    return entry
