#!/usr/bin/env python3
"""Abnahme: Qwen3-TTS VoiceDesign über das ai.noris.de-Gateway.

Stand 2026-11-24: Das Gateway (Bifrost) verwirft unbekannte Speech-Felder
(`task_type`, `language`) still, solange der Opt-in-Header
`x-bf-passthrough-extra-params: true` fehlt. Mit Header liefert derselbe
Request 200 + WAV. Dieses Skript prüft beide Seiten des Verhaltens und
validiert die durchgereichten Felder über die Fehlermeldungen des Upstream.

  1. VoiceDesign ohne Header      -> 400, Meldung nennt 'CustomVoice'
  2. VoiceDesign mit Header       -> 200 + RIFF/WAV
  3. Base mit Header              -> 400, Meldung nennt 'Base'      (Feld kommt an)
  4. language=Klingon mit Header  -> 400 'Invalid language'         (Feld kommt an)
  5. Kokoro-Kontrolle             -> 200 + WAV                      (Key/Route gesund)
  6. STT-Roundtrip des erzeugten  -> Transkript == Eingabetext
     VoiceDesign-Audios (Nemotron)

Key:  op item get 'noris-api-key' --vault Hermes --field credential --reveal
Aufruf: python3 scripts/qwen3_tts_repro.py     # $NORIS_KEY oder /tmp/nk.txt
Exit:  0 = alle Prüfungen bestanden, 1 = mindestens eine Abweichung
"""
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

BASE = os.environ.get("NORIS_BASE", "https://ai.noris.de")
MODEL = "qwen3-TTS/qsu/qwen3-tts-voicedesign"
KOKORO = "Kokoro-TTS/release/kokoro-tts-german-martin"
STT = "vllm/qsu/nemotron-3.5-transcribe"
TEXT = "Der Speicher ist bei siebenundachtzig Prozent, die Waschmaschine läuft seit zweiundvierzig Minuten."
DESC = "Eine warme, freundliche deutsche Frauenstimme, ruhiges Tempo."
PASSTHROUGH = {"x-bf-passthrough-extra-params": "true"}

failures = []


def key() -> str:
    if os.environ.get("NORIS_KEY"):
        return os.environ["NORIS_KEY"].strip()
    with open("/tmp/nk.txt", encoding="utf-8") as fh:
        return fh.read().strip()


def post(path: str, body: dict, extra_headers: dict | None = None, timeout: int = 180):
    headers = {"Authorization": f"Bearer {key()}", "Content-Type": "application/json"}
    headers.update(extra_headers or {})
    req = urllib.request.Request(
        BASE + path, data=json.dumps(body).encode(), headers=headers, method="POST"
    )
    started = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read(), round(time.time() - started, 2)
    except urllib.error.HTTPError as err:
        raw = err.read().decode(errors="replace")
        try:
            message = json.loads(raw)["error"]["message"]
        except Exception:
            message = raw
        return err.code, message.encode(), round(time.time() - started, 2)


def check(name: str, ok: bool, detail: str) -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name:40s} {detail}")
    if not ok:
        failures.append(name)


def speech_body(**overrides) -> dict:
    body = {
        "model": MODEL,
        "input": TEXT,
        "voice": "vivian",
        "task_type": "VoiceDesign",
        "instructions": DESC,
        "language": "German",
        "response_format": "wav",
    }
    body.update(overrides)
    return body


def transcribe(path: str) -> str:
    boundary = "----q3repro"
    parts = []
    for field, value in (("model", STT), ("language", "de")):
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{field}"\r\n\r\n{value}\r\n'.encode()
        )
    with open(path, "rb") as fh:
        audio = fh.read()
    body = (
        b"".join(parts)
        + f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="q3.wav"\r\n'.encode()
        + b"Content-Type: audio/wav\r\n\r\n"
        + audio
        + f"\r\n--{boundary}--\r\n".encode()
    )
    req = urllib.request.Request(
        BASE + "/v1/audio/transcriptions",
        data=body,
        headers={
            "Authorization": f"Bearer {key()}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read())["text"]


def normalize(text: str) -> str:
    """Vergleich ohne Satzzeichen und Groß-/Kleinschreibung."""
    return re.sub(r"[^\wäöüß]+", " ", text.lower()).strip()


def main() -> int:
    print(f"Gateway: {BASE}\n")

    status, payload, secs = post("/v1/audio/speech", speech_body())
    check(
        "1 VoiceDesign ohne Header -> 400",
        status == 400 and b"CustomVoice" in payload,
        f"{status} in {secs}s  {payload.decode(errors='replace')[:80]}",
    )

    status, payload, secs = post("/v1/audio/speech", speech_body(), PASSTHROUGH)
    is_wav = payload[:4] == b"RIFF"
    check(
        "2 VoiceDesign mit Header -> 200 WAV",
        status == 200 and is_wav and len(payload) > 100_000,
        f"{status} in {secs}s  {len(payload)} B  {'WAV' if is_wav else payload[:80]}",
    )
    if status == 200 and is_wav:
        open("qwen3_tts_voicedesign.wav", "wb").write(payload)
        print("        Audio: qwen3_tts_voicedesign.wav")

    status, payload, secs = post(
        "/v1/audio/speech", speech_body(task_type="Base"), PASSTHROUGH
    )
    check(
        "3 Base mit Header -> Meldung nennt 'Base'",
        status == 400 and b"'Base'" in payload,
        f"{status} in {secs}s  {payload.decode(errors='replace')[:80]}",
    )

    status, payload, secs = post(
        "/v1/audio/speech", speech_body(language="Klingon"), PASSTHROUGH
    )
    check(
        "4 language=Klingon -> 400 'Invalid language'",
        status == 400 and b"Invalid language" in payload,
        f"{status} in {secs}s  {payload.decode(errors='replace')[:60]}",
    )

    status, payload, secs = post(
        "/v1/audio/speech",
        {"model": KOKORO, "input": TEXT, "voice": "martin", "response_format": "wav"},
    )
    check(
        "5 Kokoro-Kontrolle -> 200 WAV",
        status == 200 and payload[:4] == b"RIFF",
        f"{status} in {secs}s  {len(payload)} B",
    )

    if os.path.exists("qwen3_tts_voicedesign.wav"):
        try:
            text = transcribe("qwen3_tts_voicedesign.wav")
            same = normalize(text) == normalize(TEXT)
            check(
                "6 STT-Roundtrip == Eingabetext",
                same,
                "wortgleich (Satzzeichen normalisiert)" if same else f"'{text[:90]}'",
            )
            print(f"        Transkript: {text}")
        except Exception as exc:  # noqa: BLE001
            check("6 STT-Roundtrip", False, f"Fehler: {exc}")

    print("\nErgebnis:")
    if failures:
        print(f"  {len(failures)} Abweichung(en): {', '.join(failures)}")
        return 1
    print("  alle Prüfungen bestanden")
    return 0


if __name__ == "__main__":
    sys.exit(main())
