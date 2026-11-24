"""Tests for the model helper functions."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from openai.types import Model

from custom_components.noris_ai.const import DEFAULT_MODEL
from custom_components.noris_ai.models import (
    async_get_model_options,
    default_model,
    is_audio_model,
    is_selectable_model,
    is_stt_model,
    model_label,
)
from homeassistant.helpers.selector import SelectOptionDict


def _model(model_id: str, **extra) -> Model:
    payload = {
        "id": model_id,
        "object": "model",
        "created": 0,
        "owned_by": "noris",
        **extra,
    }
    return Model.model_validate(payload)


def test_nemotron_transcribe_counts_as_stt_model() -> None:
    """The gateway does not flag Nemotron Transcribe as an audio model.

    Without the literal name match in AUDIO_MODEL_PATTERN the model never shows
    up in the speech-to-text dropdown and cannot be selected as an engine.
    """
    model = _model("vllm/qsu/nemotron-3.5-transcribe")
    assert is_audio_model(model)
    assert is_stt_model(model)


def test_plain_transcribe_chat_model_is_not_stt() -> None:
    """The name match must not drag text-to-text models into the STT dropdown."""
    assert not is_stt_model(_model("vllm/release/qwen3.8-27b"))


def test_filter_excludes_rerankers_and_drafts() -> None:
    assert is_selectable_model("vllm/gpt-oss-120b")
    assert is_selectable_model("anthropic/claude-sonnet-5")
    assert not is_selectable_model("vllm/bge-reranker-v2-m3")
    assert not is_selectable_model("vllm/prod/jina-reranker-v2-base-multilingual")
    assert not is_selectable_model("vllm/harrier-oss-v1-0.6b")
    assert not is_selectable_model("vllm/prod/harrier-oss-v1-0.6b")
    assert not is_selectable_model("vllm/release/qwen3-embedding-8b")


def _options(*values: str) -> list[SelectOptionDict]:
    return [SelectOptionDict(value=value, label=value) for value in values]


def test_default_model_prefers_exact_match() -> None:
    options = _options(
        "vllm/release/glm-5-2", DEFAULT_MODEL, "anthropic/claude-sonnet-5"
    )
    assert default_model(options) == DEFAULT_MODEL


def test_default_model_matches_channel_prefixed_rename() -> None:
    """Gateway renames like vllm/gpt-oss-120b -> vllm/release/gpt-oss-120b."""
    options = _options(
        "vllm/release/glm-5-2",
        "vllm/release/gpt-oss-120b",
        "anthropic/claude-sonnet-5",
    )
    assert default_model(options) == "vllm/release/gpt-oss-120b"


def test_default_model_falls_back_to_first_onprem() -> None:
    options = _options("vllm/release/glm-5-2", "anthropic/claude-sonnet-5")
    assert default_model(options) == "vllm/release/glm-5-2"


def test_default_model_empty_list() -> None:
    assert default_model([]) is None


def test_model_label_uses_display_name_for_anthropic() -> None:
    anthropic = Model.model_construct(
        id="anthropic/claude-sonnet-5",
        created=0,
        object="model",
        owned_by="anthropic",
        name="Claude Sonnet 5",
    )
    vllm = Model(id="vllm/gpt-oss-120b", created=0, object="model", owned_by="vllm")
    assert model_label(anthropic) == "Claude Sonnet 5"
    assert model_label(vllm) == "vllm/gpt-oss-120b"


async def test_options_sorted_vllm_first(mock_models: list[Model]) -> None:
    client = MagicMock()
    client.with_options = MagicMock(return_value=client)
    client.models.list = AsyncMock(return_value=SimpleNamespace(data=mock_models))

    options = await async_get_model_options(client)

    values = [o["value"] for o in options]
    assert values == [
        "vllm/gpt-oss-120b",
        "vllm/qwen3.6-27b",
        "vllm/release/glm-5-2",
        "anthropic/claude-sonnet-5",
    ]
    labels = [o["label"] for o in options]
    assert labels[-1] == "Claude Sonnet 5"
