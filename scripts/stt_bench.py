#!/usr/bin/env python3
"""A/B-Benchmark: Nemotron vs Voxtral unter kuechenaehnlicher Akustik.

TTS (Kokoro, de) -> synthetische Raumimpulsantwort (T60 = 0,5 s) + Rauschen -> STT.
Misst Wortfehlerrate je Modell und Bedingung. Dient als Regressionslatte: nach
jedem Modell-/Gateway-Wechsel erneut laufen lassen und die WER-Zahlen vergleichen.

Nutzung:
    export OP_SERVICE_ACCOUNT_TOKEN=$(cat ~/.op-service-account-token)
    op item get 'noris-api-key' --vault Hermes --field credential --reveal > /tmp/nk.txt
    python3 scripts/stt_bench.py
"""
import io, json, time, uuid, wave, urllib.request
import numpy as np

KEY = open("/tmp/nk.txt").read().strip()
BASE = "https://ai.noris.de/v1"
TTS_MODEL = "Kokoro-TTS/release/kokoro-tts-german-martin"
STT = {
    "nemotron": "vllm/qsu/nemotron-3.5-transcribe",
    "voxtral": "vllm/qsu/voxtral-small-24b-2507",
}
SENTENCES = [
    "Schalte das Licht in der Küche ein.",
    "Wie hoch ist der PV-Ertrag heute?",
    "Stelle einen Timer auf fünfzehn Minuten.",
    "Der Speicher ist bei dreiundneunzig Prozent.",
    "Fahre die Rollladen im Wohnzimmer herunter.",
    "Die Waschmaschine läuft noch zweiundvierzig Minuten.",
]


def tts(text: str) -> tuple[np.ndarray, int]:
    body = json.dumps(
        {"model": TTS_MODEL, "input": text, "voice": "martin", "response_format": "wav"}
    ).encode()
    req = urllib.request.Request(
        BASE + "/audio/speech",
        data=body,
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
    )
    raw = urllib.request.urlopen(req, timeout=120).read()
    w = wave.open(io.BytesIO(raw))
    a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
    return a, w.getframerate()


def rir(sr: int, t60: float, idx: int) -> np.ndarray:
    """Synthetische Raumimpulsantwort: Direktschall + exponentiell abklingende Reflexionen."""
    rng = np.random.default_rng(100 + idx)
    n = int(sr * t60)
    t = np.arange(n) / sr
    h = rng.standard_normal(n).astype(np.float32) * np.exp(-6.9 * t / t60)
    h[: int(0.004 * sr)] = 0  # Praedelay
    h[int(0.004 * sr)] = 1.0  # Direktschall
    return h / np.sqrt((h ** 2).sum())


def convolve(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    return np.convolve(x, h, mode="full")[: len(x)].astype(np.float32)


def add_noise(x: np.ndarray, snr_db: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = rng.standard_normal(len(x)).astype(np.float32)
    n = np.convolve(n, np.ones(31) / 31, mode="same").astype(np.float32)
    rms = float(np.sqrt((x ** 2).mean()))
    n *= rms / (float(np.sqrt((n ** 2).mean())) * 10 ** (snr_db / 20))
    return x + n


def wav_bytes(x: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    w = wave.open(buf, "wb")
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(sr)
    w.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())
    w.close()
    return buf.getvalue()


def stt(audio: bytes, model: str) -> tuple[str, int]:
    b = uuid.uuid4().hex
    fields = {"model": model, "language": "de", "temperature": "0"}
    out = b""
    for k, v in fields.items():
        out += f"--{b}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode()
    out += (
        f"--{b}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\n"
        "Content-Type: audio/wav\r\n\r\n"
    ).encode() + audio + b"\r\n" + f"--{b}--\r\n".encode()
    req = urllib.request.Request(
        BASE + "/audio/transcriptions",
        data=out,
        headers={
            "Authorization": f"Bearer {KEY}",
            "Content-Type": f"multipart/form-data; boundary={b}",
        },
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=90) as r:
        d = json.loads(r.read())
    return (d.get("text") or "").strip(), round((time.time() - t0) * 1000)


def words(s: str) -> list[str]:
    for ch in ".,!?;:\"'":
        s = s.replace(ch, " ")
    return s.lower().split()


def wer(ref: str, hyp: str) -> float:
    r, h = words(ref), words(hyp)
    if not r:
        return 0.0 if not h else 1.0
    prev = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        cur = [i]
        for j, hw in enumerate(h, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rw != hw)))
        prev = cur
    return prev[-1] / len(r)


if __name__ == "__main__":
    acc: dict[str, list[float]] = {f"{m}@{c}": [] for m in STT for c in ("rev", "revnoise")}
    for idx, text in enumerate(SENTENCES):
        a, sr = tts(text)
        h = rir(sr, 0.5, idx)
        r = convolve(a, h)
        r /= max(1e-9, np.abs(r).max() / 0.9)
        conds = {
            "rev": wav_bytes(r, sr),
            "revnoise": wav_bytes(add_noise(r, 10, idx), sr),
        }
        print(f"[{idx}] {text}")
        for cond, data in conds.items():
            for name, model in STT.items():
                try:
                    hyp, ms = stt(data, model)
                    w = wer(text, hyp)
                    acc[f"{name}@{cond}"].append(w)
                    print(f"   {name:9s} {cond:9s} {ms:5d}ms  {w:5.0%}  {hyp!r}")
                except Exception as e:  # noqa: BLE001
                    print(f"   {name:9s} {cond:9s} FEHLER {e}")
    print("\n== WER Mittel (Nachhall T60=0,5 s)")
    for k, v in acc.items():
        print(f"{k:20s} {np.mean(v):.1%}")
