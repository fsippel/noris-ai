"""Model listing helpers for the ai.noris.de gateway."""

from __future__ import annotations

from openai import AsyncOpenAI
from openai.types import Model

from homeassistant.helpers.selector import SelectOptionDict

from .const import DEFAULT_MODEL

# Rerankers, embedding models and tiny draft models cannot act as
# chat/agent models.
_EXCLUDED_SUBSTRINGS = ("reranker", "harrier", "embedding")


def is_selectable_model(model_id: str) -> bool:
    """Return True if the model should be offered in the picker."""
    lowered = model_id.lower()
    return not any(part in lowered for part in _EXCLUDED_SUBSTRINGS)


def model_label(model: Model) -> str:
    """Human-readable label: display name if the gateway provides one."""
    extras = model.model_extra or {}
    name = extras.get("name")
    if isinstance(name, str) and name.strip():
        return name
    return model.id


def sort_key(model: Model) -> tuple[int, str]:
    """Sort on-prem vllm models first, then by label."""
    group = 0 if model.id.startswith("vllm/") else 1
    return (group, model_label(model).lower())


def default_model(options: list[SelectOptionDict]) -> str | None:
    """Pick the best default from the live model list.

    The gateway occasionally renames models with channel prefixes
    (e.g. ``vllm/gpt-oss-120b`` -> ``vllm/release/gpt-oss-120b``), so the
    preferred model is matched by exact id first, then by suffix, before
    falling back to the first on-prem model in the list.
    """
    values = [option["value"] for option in options]
    if DEFAULT_MODEL in values:
        return DEFAULT_MODEL
    suffix = "/" + DEFAULT_MODEL.removeprefix("vllm/")
    for value in values:
        if value.startswith("vllm/") and value.endswith(suffix):
            return value
    return next(
        (value for value in values if value.startswith("vllm/")),
        values[0] if values else None,
    )


async def async_get_model_options(client: AsyncOpenAI) -> list[SelectOptionDict]:
    """Fetch, filter and sort the selectable models."""
    page = await client.with_options(timeout=10.0).models.list()
    models = [m for m in page.data if is_selectable_model(m.id)]
    models.sort(key=sort_key)
    return [
        SelectOptionDict(value=model.id, label=model_label(model))
        for model in models
    ]
