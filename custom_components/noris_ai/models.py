"""Model listing helpers for the ai.noris.de gateway."""

from __future__ import annotations

from collections.abc import Callable

from openai import AsyncOpenAI
from openai.types import Model

from homeassistant.helpers.selector import SelectOptionDict

from .const import (
    AUDIO_MODEL_PATTERN,
    DEFAULT_MODEL,
    PREFERRED_STT_MODEL,
    PREFERRED_TTS_MODEL,
    RELEASE_CHANNEL,
    STAGING_CHANNEL,
    TTS_MODEL_PATTERN,
)

# Rerankers, embedding models and tiny draft models cannot act as
# chat/agent models.
_EXCLUDED_SUBSTRINGS = ("reranker", "harrier", "embedding")


def is_selectable_model(model_id: str) -> bool:
    """Return True if the model should be offered in any picker."""
    lowered = model_id.lower()
    return not any(part in lowered for part in _EXCLUDED_SUBSTRINGS)


def _model_extra(model: Model) -> dict:
    """Return the gateway's non-standard model fields."""
    return getattr(model, "model_extra", None) or {}


def _output_types(model: Model) -> set[str]:
    """Return the output modality types the gateway reports for a model.

    These fields are not part of the OpenAI SDK's Model type, so they arrive in
    ``model_extra``. An empty set means the gateway reported nothing usable.
    """
    modalities = _model_extra(model).get("output_modalities") or []
    return {m.get("type") for m in modalities if isinstance(m, dict)}


def is_audio_model(model: Model) -> bool:
    """Return True for models that can transcribe audio.

    See AUDIO_MODEL_PATTERN: the catalog's modality metadata is wrong for
    Voxtral, so known audio families are matched by name. ``hugging_face_id`` is
    the stronger signal but is null for many models, hence the fallback.
    """
    hugging_face_id = _model_extra(model).get("hugging_face_id") or ""
    return bool(
        AUDIO_MODEL_PATTERN.search(hugging_face_id)
        or AUDIO_MODEL_PATTERN.search(model.id)
    )


def is_tts_model(model: Model) -> bool:
    """Return True for models that can synthesise speech.

    Unlike :func:`is_chat_model` and :func:`is_stt_model`, this deliberately
    does NOT gate on a ``vllm/`` prefix: the speech models live under their own
    providers (``Kokoro-TTS/``, ``Cosyvoice3/``), so requiring ``vllm/`` would
    reject every one of them.
    """
    return bool(TTS_MODEL_PATTERN.search(model.id))


def is_stt_model(model: Model) -> bool:
    """Return True for vLLM models usable as a transcription engine."""
    return model.id.startswith("vllm/") and is_audio_model(model)


def is_chat_model(model: Model) -> bool:
    """Return True for vLLM models usable as a chat or agent model.

    Audio and speech models are excluded: the gateway stamps them with the
    default text-in/text-out chat template, so a bare modality check offered
    Kokoro/CosyVoice/Voxtral as chat models and they failed at request time.
    """
    if not model.id.startswith("vllm/"):
        return False
    if is_audio_model(model):
        return False
    output_types = _output_types(model)
    if output_types:
        return "text" in output_types
    # No metadata from the gateway: fall back to the name heuristic so an
    # upstream catalog change cannot empty the dropdown.
    return is_selectable_model(model.id)


def _channel_from_id(model_id: str) -> str | None:
    """Extract the deployment channel from a model id.

    Model ids follow the pattern ``<provider>/<channel>/<slug>`` (e.g.
    ``vllm/release/qwen3.8-27b``, ``vllm/qsu/glm-5-3``). The channel
    distinguishes production (``release``) from staging (``qsu``)
    deployments that share the same display name.
    """
    parts = model_id.split("/")
    if len(parts) >= 3:
        return parts[1]
    return None


def model_label(model: Model) -> str:
    """Human-readable label with channel disambiguation.

    The gateway serves the same model under multiple channels (e.g.
    ``release`` and ``qsu``) with identical display names. Appending the
    channel prevents ambiguous duplicates in the model picker.
    """
    name = _model_extra(model).get("name")
    if isinstance(name, str) and name.strip():
        channel = _channel_from_id(model.id)
        if channel:
            return f"{name} ({channel})"
        return name
    return model.id


def sort_key(model: Model) -> tuple[int, str]:
    """Sort on-prem vllm models first, then by label."""
    group = 0 if model.id.startswith("vllm/") else 1
    return (group, model_label(model).lower())


def _preferred_id(preferred: str, options: list[SelectOptionDict]) -> str | None:
    """Resolve a preferred model id, tolerating channel-prefix changes.

    Exact id first, then the same slug in the production channel, then any
    channel, then the plain suffix match.
    """
    values = [option["value"] for option in options]
    if preferred in values:
        return preferred

    slug = preferred.rsplit("/", 1)[-1]
    for value in values:
        if _channel_from_id(value) == RELEASE_CHANNEL and value.endswith(slug):
            return value
    for value in values:
        if value.endswith(slug):
            return value
    return None


def _channel_rank(model_id: str) -> int:
    """Rank production above staging, everything else last."""
    channel = _channel_from_id(model_id)
    if channel == RELEASE_CHANNEL:
        return 0
    if channel == STAGING_CHANNEL:
        return 1
    return 2


def default_model(options: list[SelectOptionDict]) -> str | None:
    """Pick the best default from the live chat model list.

    The gateway occasionally renames models with channel prefixes
    (e.g. ``vllm/gpt-oss-120b`` -> ``vllm/release/gpt-oss-120b``), so the
    preferred model is matched by exact id first, then by suffix, before
    falling back to the first on-prem model in the list.
    """
    if resolved := _preferred_id(DEFAULT_MODEL, options):
        return resolved
    values = [option["value"] for option in options]
    return next(
        (value for value in values if value.startswith("vllm/")),
        values[0] if values else None,
    )


def default_stt_model(options: list[SelectOptionDict]) -> str | None:
    """Pick the default transcription model.

    Falls back to the production-channel audio model, then to any audio model —
    transcription is currently only served from the staging channel, and an
    empty picker would make the platform unusable.
    """
    return _preferred_id(PREFERRED_STT_MODEL, options) or _default_by_rank(options)


def default_tts_model(options: list[SelectOptionDict]) -> str | None:
    """Pick the default speech model, preferring the German voice."""
    return _preferred_id(PREFERRED_TTS_MODEL, options) or _default_by_rank(options)


def _default_by_rank(options: list[SelectOptionDict]) -> str | None:
    """First option, ordered production channel before staging."""
    if not options:
        return None
    return min(options, key=lambda o: (_channel_rank(o["value"]), o["label"]))["value"]


async def async_get_model_options(
    client: AsyncOpenAI,
    predicate: Callable[[Model], bool] | None = None,
) -> list[SelectOptionDict]:
    """Fetch, filter and sort the selectable models.

    ``predicate`` narrows the list to one capability (chat, transcription,
    speech) so each subentry picker only offers models the platform can use.
    """
    page = await client.with_options(timeout=10.0).models.list()
    models = [
        m
        for m in page.data
        if is_selectable_model(m.id) and (predicate is None or predicate(m))
    ]
    models.sort(key=sort_key)
    return [
        SelectOptionDict(value=model.id, label=model_label(model)) for model in models
    ]
