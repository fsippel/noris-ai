# Anforderung: Qwen3-TTS VoiceDesign über ai.noris.de (Option A)

**Datum:** 2026-11-24
**Betroffen:** ai.noris.de (Bifrost-Gateway), Modell `qwen3-TTS/qsu/qwen3-tts-voicedesign`
**Status:** Kern funktioniert bereits (Nachweis unten). Offen sind Dokumentation, Katalog-Hygiene und eine Entscheidung zu einer Komfortänderung.
**Repro:** `scripts/qwen3_tts_repro.py` (Repo `noris-ai`)

---

## 1. Zusammenfassung

Das Modell ist korrekt installiert und zugewiesen. Der dokumentierte VoiceDesign-Call scheitert nur deshalb mit HTTP 400, weil das Gateway unbekannte Request-Felder (`task_type`, `language`) standardmäßig verwirft. Mit dem dokumentierten Opt-in-Header `x-bf-passthrough-extra-params: true` liefert derselbe Call **HTTP 200 und eine abspielbare WAV-Datei**.

Damit ist **keine Codeänderung am Gateway zwingend erforderlich**. Zu tun bleibt:

1. Den Header und seine Semantik dokumentieren (Muss).
2. Den Modell-Katalogeintrag bereinigen und im Admin-Katalog sichtbar machen (Muss).
3. Entscheiden, ob die Speech-Route den Passthrough selbst aktiviert, damit Clients ohne Header arbeiten (Empfehlung, 1 Zeile).
4. Einen Regressionstest ergänzen, der das Durchreichen festschreibt (Muss).

---

## 2. Ausgangsproblem

Der von der vLLM-Omni-Dokumentation vorgesehene Request

```json
{
  "model": "qwen3-TTS/qsu/qwen3-tts-voicedesign",
  "input": "Guten Abend, ich bin die Küchenstimme.",
  "voice": "vivian",
  "task_type": "VoiceDesign",
  "instructions": "Eine warme, freundliche deutsche Frauenstimme, ruhiges Tempo.",
  "language": "German",
  "response_format": "wav"
}
```

antwortet ohne zusätzliche Header mit

```
400 Qwen3-TTS VoiceDesign checkpoint does not support task_type='CustomVoice'.
    Use task_type='VoiceDesign' or load the matching CustomVoice checkpoint.
```

Der Fehlertext ist irreführend: Er empfiehlt genau das Feld, das der Client bereits sendet. Ursache ist nicht das Modell, sondern die Umsetzung am Gateway:

- `transports/bifrost-http/handlers/inference.go:262` listet die bekannten Speech-Felder: `model`, `input`, `fallbacks`, `stream_format`, `voice`, `instructions`, `response_format`, `speed`. **`task_type` und `language` fehlen dort**, gelten also als unbekannt.
- `:789` (`extractExtraParams`) sammelt alle unbekannten Felder korrekt in `ExtraParams`; `:1794` hängt sie an `req.SpeechParameters.ExtraParams`.
- `core/providers/utils/utils.go:2884` mischt `ExtraParams` aber **nur** in den Upstream-Body, wenn im Context `BifrostContextKeyPassthroughExtraParams` gesetzt ist. Ohne Flag wird der Body ohne diese Felder serialisiert.
- Der Upstream (`vllm_omni/.../tts_adapters/qwen3_tts.py`) sieht `task_type = None` und setzt den Default `CustomVoice` — daher die Meldung.

Der Passthrough ist ein bewusstes Opt-in und kein Bug. Er ist über den Header `x-bf-passthrough-extra-params: true` freigeschaltet (`transports/bifrost-http/lib/ctx.go:778`).

---

## 3. Nachweis (Messung 2026-11-24, 10:55–11:15 UTC)

| # | Request | Ergebnis |
|---|---|---|
| 1 | VoiceDesign, ohne Header | **400** in 0,12 s, Fehler nennt `'CustomVoice'` |
| 2 | VoiceDesign, mit `x-bf-passthrough-extra-params: true` | **200**, 126.764 B, 0,41 s, RIFF/WAV |
| 3 | VoiceDesign ohne `task_type`, mit Header | **400**, Fehler nennt `'CustomVoice'` (erwartungsgemäß) |
| 4 | `task_type=Base`, mit Header | **400**, Fehler nennt jetzt `'Base'` → Feld kommt an |
| 5 | `language=Klingon`, mit Header | **400** `Invalid language 'Klingon'. Supported: Auto, Chinese, English, French, German, …` → Feld kommt an und wird validiert |
| 6 | Kokoro-Kontrolle, ohne Header | **200**, WAV in 0,80 s → Key, Quota, Route gesund |

Zusätzlich Roundtrip zur Inhaltsprüfung:

- VoiceDesign-TTS: 4,96 s, 24 kHz, mono, PCM s16le.
- Transkription derselben Datei mit `vllm/qsu/nemotron-3.5-transcribe`: **wortgleich** mit dem Eingabetext, Upstream-Latenz 195 ms.

Damit ist belegt: Das Modell synthetisiert korrekt, das Gateway reicht die Felder durch, sobald das Flag gesetzt ist.

---

## 4. Anforderungen

### M1 — Dokumentation (Muss)

In der API-Dokumentation und im OpenAPI-Schema für `/v1/audio/speech` aufnehmen:

- Header `x-bf-passthrough-extra-params: true` schaltet das Durchreichen unbekannter Request-Felder frei.
- Der Wert muss exakt `true` sein (kein `True`, kein `1`, keine Leerzeichen) — `lib/ctx.go:779` vergleicht per String-Gleichheit.
- Ohne den Header werden unbekannte Felder **still verworfen**. Das ist die eigentliche Falle: Der Client bekommt keinen Hinweis, sondern einen Folgefehler des Upstream.
- Gilt für `/v1/audio/speech` und die Drop-in-Routen (`/openai/v1/audio/speech`, `/litellm/v1/audio/speech`). Auf den Drop-in-Routen kann zusätzlich ein `extra_params`-Objekt gehoben werden (`integrations/router.go:830`).

**Akzeptanz:** Doku-Seite und OpenAPI enthalten Header, Semantik und das Verwerfen ohne Flag.

### M2 — Katalog und Virtual Key (Muss)

- `qwen3-TTS/qsu/qwen3-tts-voicedesign` erscheint im Virtual Key **dreimal identisch**.
- Im Admin-Gesamtkatalog (52 Modelle) fehlt das Modell.
- `is_ready` ist im vKey `true`.

Bitte auf **einen** Eintrag konsolidieren, im Admin-Katalog sichtbar machen und `is_ready`/Provider-Zuordnung bestätigen. Drei identische Einträge erschweren Routing-Diagnose und Governance.

**Akzeptanz:** genau ein Eintrag im vKey, Modell im Admin-Katalog auffindbar.

### M3 — Passthrough für die Speech-Route aktivieren (Empfehlung, Entscheidung erbeten)

Damit Clients ohne Sonderheader funktionieren und sich Speech wie die Bild-Routen verhält:

`transports/bifrost-http/handlers/inference.go`, `func (h *CompletionHandler) speech` (ab Zeile 1808). Nach dem Aufbau von `bifrostCtx` (Zeile 1822), vor dem Streaming-Zweig:

```go
bifrostCtx.SetValue(schemas.BifrostContextKeyPassthroughExtraParams, true)
```

Dieselbe Zeile steht bereits in `imageGeneration` (`:2752`) und `imageEdit` (`:3027`). Der Eingriff ist eine Zeile, reversibel und betrifft nur Speech.

**Trade-off:** Ohne Flag werden Tippfehler in Feldnamen still ignoriert; mit Flag erzeugen sie einen lauten 400 des Upstream. Für unseren Fall ist laut richtig — ein falsch geschriebenes `task_type` soll nicht unbemerkt zu einer anderen Stimme führen.

**Akzeptanz:** Request 1 aus Abschnitt 3 liefert 200 ohne jeden Zusatzheader; Request 3 liefert weiterhin 400.

### M4 — Allowlist statt generischem Passthrough (Optional)

Falls kein generisches Durchreichen gewünscht ist: `TaskType string \`json:"task_type,omitempty"\`` und `Language string \`json:"language,omitempty"\`` in `core/schemas/speech.go` (`SpeechParameters`) ergänzen und in `core/providers/openai/speech.go` bzw. `openai.go:1062` (`Speech`) in den Upstream-Body übernehmen. Alternativ den Merge in `MergeExtraParamsIntoJSON` auf eine Provider-Allowlist beschränken.

Diese Variante ist sauberer in der Typsicherheit, aber modellspezifisch: Das nächste Extension-Feld benötigt wieder einen Patch. M3 ist billiger und generisch.

### M5 — Regressionstest (Muss)

Go-Test in `transports/bifrost-http/handlers/` (z. B. `speech_extra_params_test.go`) gegen einen Mock-Upstream:

1. Request mit `task_type` und Flag → Body enthält `"task_type":"VoiceDesign"`.
2. Request mit `task_type` ohne Flag → Body enthält das Feld nicht (pinnt das Opt-in-Verhalten).
3. Request mit `task_type=Base` → Upstream erhält `Base`, nicht `CustomVoice`.

**Akzeptanz:** Test rot ohne die Merge-Logik, grün mit.

---

## 5. Akzeptanzkriterien (Abnahme)

Alle Prüfungen gegen `https://ai.noris.de`, Key als `<NORIS_KEY>`:

```bash
# A1: VoiceDesign mit Flag -> 200 + WAV
curl -sS -o /tmp/q3.wav -w '%{http_code} %{size_download}\n' \
  -H "Authorization: Bearer $NORIS_KEY" -H 'Content-Type: application/json' \
  -H 'x-bf-passthrough-extra-params: true' \
  -d '{"model":"qwen3-TTS/qsu/qwen3-tts-voicedesign","input":"Guten Abend.","voice":"vivian",
       "task_type":"VoiceDesign","instructions":"Warme deutsche Frauenstimme.","language":"German",
       "response_format":"wav"}' \
  https://ai.noris.de/v1/audio/speech
file /tmp/q3.wav    # RIFF (Microsoft), 24 kHz mono

# A3: Base mit Flag -> 400, Meldung nennt 'Base'
# A4: language=Klingon mit Flag -> 400 mit Sprachliste
# A6: vKey enthält genau einen Eintrag, Admin-Katalog enthält das Modell
```

| ID | Kriterium | Erwartung |
|---|---|---|
| A1 | VoiceDesign mit Flag | 200, WAV > 100 KB, < 2 s |
| A2 | VoiceDesign ohne Flag | 400 (bleibt so, solange M3 nicht umgesetzt ist) |
| A3 | `task_type=Base` mit Flag | 400, Meldung nennt `'Base'` |
| A4 | `language=Klingon` mit Flag | 400 mit unterstützter Sprachliste |
| A5 | Dokumentation | Header, Semantik, Still-Verwerfen beschrieben |
| A6 | Katalog | ein vKey-Eintrag, im Admin sichtbar |
| A7 | Go-Test aus M5 | grün, ohne Merge-Logik rot |

---

## 6. Nicht Teil dieser Anforderung

- **Checkpoint-Wechsel auf CustomVoice (Option B).** Nicht nötig, solange VoiceDesign genutzt wird; ein CustomVoice-Checkpoint liefert keine deutschen Premium-Presets, die Stimmbeschreibung käme dort nicht zum Tragen.
- **Änderungen an vLLM-Omni.** Das Modell verhält sich dokumentationskonform.
- **Änderungen an der Home-Assistant-Integration.** Die Produktion läuft weiter auf Kokoro; VoiceDesign wird erst nach Stimmauswahl und Abnahme umgestellt. Die STT-Kette (Nemotron primär, Voxtral Fallback) bleibt unberührt.

---

## 7. Risiko und Rollback

| Änderung | Risiko | Rollback |
|---|---|---|
| M3 (Flag in Speech) | Unbekannte Felder erreichen den Upstream; falsche Feldnamen erzeugen 400 statt stiller Ignoranz | Zeile entfernen, Deploy |
| M4 (Allowlist) | Modellspezifisch, nächste Erweiterung braucht neuen Patch | Felder entfernen |
| M1/M2/M5 | keins | — |

Der Header aus M1 ist ein Opt-in pro Request und ändert das Verhalten anderer Clients nicht.

---

## 8. Anhang: Code-Referenzen (Bifrost `main`, Commit `36235d0`)

| Datei:Zeile | Inhalt |
|---|---|
| `transports/bifrost-http/handlers/inference.go:262` | `speechParamsKnownFields` (ohne `task_type`, ohne `language`) |
| `transports/bifrost-http/handlers/inference.go:789` | `extractExtraParams` — sammelt unbekannte Felder |
| `transports/bifrost-http/handlers/inference.go:1767` | `prepareSpeechRequest` |
| `transports/bifrost-http/handlers/inference.go:1794` | `req.SpeechParameters.ExtraParams = base.ExtraParams` |
| `transports/bifrost-http/handlers/inference.go:1808` | `speech`-Handler, Einfügepunkt für M3 nach Zeile 1822 |
| `transports/bifrost-http/handlers/inference.go:2752`, `:3027` | Bild-Routen setzen das Flag bereits selbst |
| `transports/bifrost-http/lib/ctx.go:778` | Header-Parsing `x-bf-passthrough-extra-params` (Wert exakt `true`) |
| `transports/bifrost-http/integrations/router.go:830` | Heben von `extra_params` auf Drop-in-Routen |
| `core/providers/utils/utils.go:2873` | `marshalConvertedRequestBody` |
| `core/providers/utils/utils.go:2884` | Merge-Gate auf `BifrostContextKeyPassthroughExtraParams` |
| `core/providers/openai/types.go:1641` | `OpenAISpeechRequest` (ExtraParams `json:"-"`) |
| `core/providers/openai/types.go:1653` | `GetExtraParams()` |
| `core/schemas/speech.go:69` | `SpeechParameters.ExtraParams json:"-"` |
| `core/providers/openai/openai.go:1062` | `OpenAIProvider.Speech`, `ToOpenAISpeechRequest`, Body-Aufbau |

## 9. Was wir auf unserer Seite tun

**Erledigt (noris_ai 0.6.6, Commit `fdb293a`):** Die Integration setzt den Opt-in
jetzt selbst. `speech_request_kwargs()` in `custom_components/noris_ai/tts.py`
hängt `x-bf-passthrough-extra-params: true` plus `task_type`, `instructions` und
`language` an jede Anfrage eines VoiceDesign-Modells; für Kokoro und CosyVoice
bleibt der Request unverändert (leeres Dict). Damit braucht Home Assistant keine
Server-Änderung.

Nachweis mit genau diesen Kwargs gegen `https://ai.noris.de/v1/audio/speech`
(`scripts/tts_voicedesign_smoke.py`, 2026-11-24 11:56 UTC): HTTP 200, 157.484
Byte, `pcm_s16le`/24 kHz/mono, 3,28 s; STT-Roundtrip über Nemotron liefert
„Guten Abend. Der Timer läuft in zehn Minuten ab." wortgleich zurück.

Damit verschiebt sich der Charakter von M3: Es ist keine Voraussetzung mehr für
uns, sondern Komfort für alle anderen Clients (curl, n8n, Drittanbieter), die
sonst denselben Header setzen müssen.

- Repro und Abnahmeskript: `scripts/qwen3_tts_repro.py` prüft A1–A4 und gibt bei Abweichung Exit 1 zurück.
- Bis zur Stimmabnahme bleibt Home Assistant auf Kokoro; keine Umstellung der Produktion auf ein ungeprüftes VoiceDesign-Preset.
- Nach Abnahme: Stimme per `instructions` festlegen, Latenz und Verständlichkeit gegen Kokoro messen, dann erst umstellen.

## 10. Entscheidung erbeten

1. **M3 umsetzen?** Nicht mehr nötig für uns (Header ist im Client), aber es erspart allen anderen Clients den Header. Empfehlung: ja, eine Zeile.
2. Soll M4 (Allowlist) parallel vorbereitet werden, falls generisches Passthrough unerwünscht ist?
3. Welcher Gateway-Build läuft aktuell? Die Zeilennummern oben beziehen sich auf `main` @ `36235d0`; bei einem älteren Build kann das Merge-Gate fehlen — dann ist M3 der kürzeste Weg.
