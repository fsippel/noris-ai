#!/usr/bin/env python3
"""Kleiner STT-Benchmark: Nemotron vs Voxtral auf TTS-Sätzen, clean / SNR5 / SNR0.

Misst Wortfehlerrate (WER) und zeigt, welches Modell unter Küchengeräuschen haelt.
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
    with urllib.request.urlopen(req, timeout=120) as r:
        raw = r.read()
    w = wave.open(io.BytesIO(raw))
    sr = w.getframerate()
    a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
    if w.getnchannels() > 1:
        a = a.reshape(-1, w.getnchannels()).mean(axis=1)
    return a, sr


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


def stt(audio: bytes, model: str, prompt: str | None = None) -> tuple[str, int]:
    b = uuid.uuid4().hex
    fields = {"model": model, "language": "de", "temperature": "0"}
    if prompt:
        fields["prompt"] = prompt
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
    results: dict[str, list[float]] = {f"{m}@{c}": [] for m in STT for c in ("clean", "snr5", "snr0")}
    samples = []
    for idx, text in enumerate(SENTENCES):
        audio, sr = tts(text)
        conds = {
            "clean": wav_bytes(audio, sr),
            "snr5": wav_bytes(add_noise(audio, 5, idx), sr),
            "snr0": wav_bytes(add_noise(audio, 0, idx), sr),
        }
        line = {"text": text, "sr": sr, "model_out": {}}
        for cond, data in conds.items():
            for name, model in STT.items():
                try:
                    hyp, ms = stt(data, model)
                    w = wer(text, hyp)
                    results[f"{name}@{cond}"].append(w)
                    line["model_out"].setdefault(name, {})[cond] = (hyp, ms, round(w, 2))
                except Exception as e:  # noqa: BLE001
                    line["model_out"].setdefault(name, {})[cond] = (f"FEHLER {e}", 0, 1.0)
        samples.append(line)
        print(f"[{idx}] {text}")
        for name in STT:
            outs = line["model_out"].get(name, {})
            print(
                f"   {name:9s} clean={outs.get('clean')} snr5={outs.get('snr5')} snr0={outs.get('snr0')}"
            )
    print("\n== Mittelwerte WER")
    for k, v in results.items():
        print(f"{k:16s} {np.mean(v):.1%}  (n={len(v)})")
    # Prompt-Wirkung auf domänenspezifischem Satz
    print("\n== Prompt-Test (Satz 1: PV-Ertrag)")
    audio, sr = tts(SENTENCES[1])
    data = wav_bytes(audio, sr)
    P = (
        "Transkribiere wörtlich auf Deutsch. Fachbegriffe: Photovoltaik, PV-Ertrag, "
        "Kilowattstunden, Speicher, Netzfluss, Küche, Wohnzimmer, Rollladen."
    )
    for name, model in STT.items():
        a, _ = stt(data, model, prompt=P)
        b_, _ = stt(data, model)
        print(f"  {name:9s} mit Prompt: {a!r}")
        print(f"  {'':9s} ohne      : {b_!r}")
