"""Fixtures for noris AI tests."""

from collections.abc import Generator
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from openai.types import Model
import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry

from homeassistant.config_entries import ConfigSubentryData
from homeassistant.const import CONF_API_KEY, CONF_MODEL
from homeassistant.core import HomeAssistant
from homeassistant.setup import async_setup_component

from custom_components.noris_ai.const import (
    AI_TASK_SUBENTRY_TYPE,
    CONF_RECOMMENDED,
    CONVERSATION_SUBENTRY_TYPE,
    DEFAULT_MODEL,
    DOMAIN,
)


@pytest.fixture(autouse=True)
def auto_enable_custom_integrations(enable_custom_integrations):
    """Enable loading custom integrations in all tests."""
    return


@pytest.fixture(autouse=True)
async def setup_homeassistant_component(hass: HomeAssistant) -> None:
    """Set up the homeassistant component.

    The conversation component (a manifest dependency of noris_ai) needs the
    exposed-entities registry that the homeassistant component provides.
    """
    assert await async_setup_component(hass, "homeassistant", {})


@pytest.fixture
def mock_models() -> list[Model]:
    """Gateway model list: usable, external, and filtered-out entries."""
    return [
        Model(id="vllm/gpt-oss-120b", created=0, object="model", owned_by="vllm"),
        Model(id="vllm/qwen3.6-27b", created=0, object="model", owned_by="vllm"),
        Model(id="vllm/release/glm-5-2", created=0, object="model", owned_by="vllm"),
        Model(
            id="vllm/bge-reranker-v2-m3", created=0, object="model", owned_by="vllm"
        ),
        Model(
            id="vllm/harrier-oss-v1-0.6b", created=0, object="model", owned_by="vllm"
        ),
        Model.model_construct(
            id="anthropic/claude-sonnet-5",
            created=0,
            object="model",
            owned_by="anthropic",
            name="Claude Sonnet 5",
        ),
    ]


@pytest.fixture
def mock_client(mock_models: list[Model]) -> MagicMock:
    """A mocked AsyncOpenAI client."""
    client = MagicMock()
    client.with_options = MagicMock(return_value=client)
    client.models.list = AsyncMock(
        return_value=SimpleNamespace(data=mock_models)
    )
    client.chat.completions.create = AsyncMock()
    return client


@pytest.fixture
def mock_openai(mock_client: MagicMock) -> Generator[MagicMock]:
    """Patch the AsyncOpenAI class used by the integration."""
    with patch(
        "custom_components.noris_ai.AsyncOpenAI", return_value=mock_client
    ):
        yield mock_client


@pytest.fixture
def mock_config_entry(hass: HomeAssistant) -> MockConfigEntry:
    """A config entry with only the API key (no subentries)."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="noris AI",
        data={CONF_API_KEY: "sk-bf-test"},
    )
    entry.add_to_hass(hass)
    return entry


@pytest.fixture
def mock_config_entry_with_conversation(hass: HomeAssistant) -> MockConfigEntry:
    """A config entry with one conversation agent (recommended settings)."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="noris AI",
        data={CONF_API_KEY: "sk-bf-test"},
        subentries_data=[
            ConfigSubentryData(
                data={
                    CONF_MODEL: DEFAULT_MODEL,
                    CONF_RECOMMENDED: True,
                    "llm_hass_api": ["assist"],
                },
                subentry_type=CONVERSATION_SUBENTRY_TYPE,
                title="Test Agent",
                unique_id=None,
            )
        ],
    )
    entry.add_to_hass(hass)
    return entry


@pytest.fixture
def mock_config_entry_with_ai_task(hass: HomeAssistant) -> MockConfigEntry:
    """A config entry with one AI task entity (recommended settings)."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="noris AI",
        data={CONF_API_KEY: "sk-bf-test"},
        subentries_data=[
            ConfigSubentryData(
                data={CONF_MODEL: DEFAULT_MODEL, CONF_RECOMMENDED: True},
                subentry_type=AI_TASK_SUBENTRY_TYPE,
                title="Test Task",
                unique_id=None,
            )
        ],
    )
    entry.add_to_hass(hass)
    return entry


async def setup_integration(hass: HomeAssistant, entry: MockConfigEntry) -> bool:
    """Set up the integration and wait for it to settle."""
    result = await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    return result
