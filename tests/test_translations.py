"""Ensure strings.json and translations stay in sync."""

import json
import pathlib

BASE = pathlib.Path("custom_components/noris_ai")


def _keys(obj: dict, prefix: str = "") -> set[str]:
    keys: set[str] = set()
    for key, value in obj.items():
        path = f"{prefix}.{key}" if prefix else key
        keys.add(path)
        if isinstance(value, dict):
            keys |= _keys(value, path)
    return keys


def _load(path: pathlib.Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_en_matches_strings() -> None:
    assert _load(BASE / "strings.json") == _load(BASE / "translations/en.json")


def test_de_has_same_keys() -> None:
    strings_keys = _keys(_load(BASE / "strings.json"))
    de_keys = _keys(_load(BASE / "translations/de.json"))
    assert strings_keys == de_keys
