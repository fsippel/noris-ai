#!/usr/bin/env python3
"""Repro: Qwen3-TTS VoiceDesign scheitert am ai.noris.de-Gateway (Stand 2026-11-24).

Zeigt in vier Requests, dass `task_type` das Gateway nicht erreicht:

  1. dokumentierter VoiceDesign-Call  -> 400 "... does not support task_type='CustomVoice'"
  2. derselbe Call als Base           -> 400, Fehlertext nennt trotzdem 'CustomVoice'
  3. derselbe Call ohne task_type     -> 400, erwartungsgemaess 'CustomVoice'
  4. Kokoro-Kontrollcall              -> 200 + WAV (Key/Quota/Route gesund)

Erwartung nach dem Fix (Bifrost reicht task_type durch): Request 1 liefert 200 + WAV.

Key:  op item get 'noris-api-key' --vault Hermes --field credential --reveal
Aufruf: python3 qwen3_tts_repro.py            # nutzt $NORIS_KEY oder /tmp/nk.txt
"""
import json
import os
import sys
import urllib.error
import urllib.request

BASE = os.environ.get("NORIS_BASE", "https://ai.noris.de")
MODEL = "qwen3-TTS/qsu/qwen3-tts-voicedesign"
KOKORO = "Kokoro-TTS/release/kokoro-tts-german-martin"
TEXT = "Der Speicher ist bei siebenundachtzig Prozent."
DESC = "Eine warme, freundliche deutsche Frauenstimme, ruhiges Tempo."


def key() -> str:
    if os.environ.get("NORIS_KEY"):
        return os.environ["NORIS_KEY"].strip()
    with open("/tmp/nk.txt", encoding="utf-8") as fh:
        return fh.read().strip()


def post(path: str, body: dict, timeout: int = 180):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key()}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read(), resp.headers.get("content-type", "")
    except urllib.error.HTTPError as err:
        raw = err.read().decode(errors="replace")
        try:
            return err.code, json.loads(raw)["error"]["message"].encode(), "application/json"
        except Exception:
            return err.code, raw.encode(), "text/plain"


def show(name: str, status: int, payload: bytes) -> None:
    if payload[:4] == b"RIFF":
        print(f"  {name:34s} {status}  {len(payload):7d} B  WAV")
        return
    print(f"  {name:34s} {status}  {len(payload):7d} B  {payload.decode(errors='replace')[:170]}")


def main() -> int:
    print(f"Gateway: {BASE}\n")
    ok = 0

    status, payload, ctype = post("/v1/audio/speech", {
        "model": MODEL, "input": TEXT, "voice": "vivian",
        "task_type": "VoiceDesign", "instructions": DESC,
        "language": "German", "response_format": "wav",
    })
    show("1 VoiceDesign (dokumentiert)", status, payload)
    if status == 200:
        open("qwen3_tts_voicedesign.wav", "wb").write(payload)
        print("    -> OK, Audio gespeichert: qwen3_tts_voicedesign.wav")
        ok += 1

    status, payload, _ = post("/v1/audio/speech", {
        "model": MODEL, "input": TEXT, "voice": "vivian",
        "task_type": "Base", "response_format": "wav",
    })
    show("2 Base (Sonde: kommt Feld an?)", status, payload)

    status, payload, _ = post("/v1/audio/speech", {
        "model": MODEL, "input": TEXT, "voice": "vivian", "response_format": "wav",
    })
    show("3 ohne task_type", status, payload)

    status, payload, _ = post("/v1/audio/speech", {
        "model": KOKORO, "input": TEXT, "voice": "martin", "response_format": "wav",
    })
    show("4 Kokoro-Kontrolle", status, payload)
    if status == 200:
        ok += 1

    print("\nDiagnose:")
    print("  Fehler 2 nennt 'CustomVoice', obwohl 'Base' gesendet wurde ->")
    print("  task_type wird vom Gateway (Bifrost SpeechParameters) verworfen;")
    print("  upstream gilt der Default `request.task_type or 'CustomVoice'`.")
    return 0 if ok == 2 else 1


if __name__ == "__main__":
    sys.exit(main())
