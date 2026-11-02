# noris AI for Home Assistant

**English Summary**

A custom Home Assistant integration for the **ai.noris.de** Bifrost gateway. Adds a **Conversation agent** (for Assist / voice control), an **AI Task** entity (structured data generation for automations), a **Speech-to-text** entity (transcribe Assist commands) and a **Text-to-speech** entity (speak through the gateway's speech models). Built on the OpenAI-compatible Chat Completions, transcription and speech APIs with standard Bearer authentication. All gateway models are selectable (`vllm/*` listed first); only `vllm/*` models run on-premises. Reranker and draft models are filtered out, and audio/speech models are no longer offered as chat models. Default model: `vllm/gpt-oss-120b`.

---

## Was ist das?

Die **noris AI**-Integration für Home Assistant erweitert dein Smart Home um vier zentrale Funktionen:

1. **Conversation Agent** — ein KI-basierter Sprach- und Textassistent für Home Assistant Assist. Du kannst Fragen stellen, Geräte steuern und Automationen auslösen.
2. **AI Task** — eine Entity für strukturierte und freie Datengenerierung. Nützlich für Automationen, die KI-basierte Eingaben brauchen (z. B. Datenextraktion, Klassenbildung, Synthesis).
3. **Speech-to-text** — Transkription von Assist-Sprachbefehlen über die Audio-Modelle des Gateways (Voxtral).
4. **Text-to-speech** — Sprachausgabe über die Speech-Modelle des Gateways (Kokoro, CosyVoice3).

Beide Funktionen laufen über das **Bifrost-Gateway** (`https://ai.noris.de/v1`) der noris network AG — eine OpenAI-kompatible API mit Zugriff auf lokale (`vllm/*`) und externe Modelle.

**Datenschutz:** Alle Gateway-Modelle sind wählbar, aber nur die **`vllm/*`**-Modelle laufen **on-premises** im noris-Rechenzentrum Nürnberg 6 — deine Daten verlassen dabei das Rechenzentrum nicht. Anthropic-Modelle sind ebenfalls wählbar, werden aber extern geroutet.

## Installation

### Über HACS (empfohlen)

1. Öffne Home Assistant und gehe zu **Settings → Devices & Services → HACS** (oder rufe HACS auf).
2. Oben rechts: **⋮** → **Custom repositories**.
3. Füge die URL dieses Repositorys hinzu: `https://github.com/fsippel/noris-ai`
   - **Kategorie:** *Integration*
4. Klick auf **Create**, dann suche nach **noris AI** in der HACS-Integrationsliste.
5. Klick auf **Install**, wähle eine Version und warte auf die Installation.
6. Starten Sie Home Assistant neu (oder verwenden Sie **Developer Tools → Restart Home Assistant**).

### Manuell

1. Laden Sie den neuesten Release herunter oder klonen Sie das Repository.
2. Kopieren Sie den Ordner `custom_components/noris_ai` in das Verzeichnis `config/custom_components/` Ihrer Home Assistant-Instanz.
3. Starten Sie Home Assistant neu.

## Konfiguration

### Schritt 1: Integration hinzufügen

1. Gehe zu **Settings → Devices & Services → Integrations** (oder **Add Integration**).
2. Suche nach **noris AI** und wähle die Integration aus.
3. Gib deinen **API-Schlüssel** ein (Format: `sk-bf-...`).
   - Der Schlüssel wird beim Anlegen gegen `https://ai.noris.de/v1/models` validiert.
   - Home Assistant speichert den Schlüssel in seiner Konfiguration; in den Diagnostics wird er automatisch geschwärzt.
   - Wird der Schlüssel später ungültig, startet die Integration einen **Reauth-Flow** — du gibst einfach einen neuen Schlüssel ein.

### Schritt 2: Conversation Agent oder AI Task hinzufügen

Nach erfolgreicher Authentifizierung siehe die noris AI-Integration in der Geräteliste.

#### Conversation Agent hinzufügen

1. Klick auf **Add conversation agent**.
2. Wähle ein Modell aus der Liste (Standard: `vllm/gpt-oss-120b`).
3. Optional:
   - **Control Home Assistant:** aktiviere dies, um Geräte per Tool-Call zu steuern (setzt ein Modell mit Function-Calling-Unterstützung voraus).
   - **Empfohlene Einstellungen:** vorgestellt, für die meisten Nutzer ausreichend.
   - **Erweiterte Einstellungen:** Anpassungen für Power User (z. B. `max_tokens`, Temperatur).

#### AI Task hinzufügen

1. Klick auf **Add AI task**.
2. Gib einen **Namen** an (z. B. `data_extractor`).
3. Wähle ein Modell (Standard: `vllm/gpt-oss-120b`).
4. Optional erweiterte Einstellungen.

Ein JSON-Schema für strukturierte Ausgaben wird nicht in diesem Flow konfiguriert, sondern **zur Laufzeit** beim Aufruf des Service `ai_task.generate_data` über den Parameter `structure` übergeben.

#### Speech-to-text hinzufügen

1. Klick auf **Add speech-to-text service**.
2. Gib einen **Namen** an (z. B. `noris_stt`).
3. Wähle ein Audio-Modell — der Picker zeigt nur Modelle, die tatsächlich transkribieren können. Aktuell ist das ausschließlich `vllm/qsu/voxtral-small-24b-2507`.
4. Trage die Entity anschließend in deiner Assist-Pipeline als **Speech-to-text** ein.

#### Text-to-speech hinzufügen

1. Klick auf **Add text-to-speech service**.
2. Gib einen **Namen** an (z. B. `noris_tts`).
3. Wähle ein Speech-Modell. Vorausgewählt ist `Kokoro-TTS/release/kokoro-tts-german-martin` (deutsche Stimme).
4. Trage die Entity in deiner Assist-Pipeline als **Text-to-speech** ein.

## Sprache (STT/TTS)

Beide Sprachplattformen hängen an den Audio-Endpunkten des Gateways (`/v1/audio/transcriptions`, `/v1/audio/speech`) und brauchen keinen eigenen Container — die Transkription und Synthese laufen auf der noris-Infrastruktur.

### Speech-to-text

- Audio wird als **16 kHz, mono, 16 Bit PCM in einem WAV-Container** gesendet. Der Gateway lehnt headerloses PCM mit „Invalid or unsupported audio file" ab, deshalb packt die Integration die Samples vor dem Upload in ein WAV.
- Unterstützte Sprachen sind die von Voxtral dokumentierten: Deutsch, Englisch, Spanisch, Französisch, Italienisch, Niederländisch, Portugiesisch und Hindi (siehe `STT_SUPPORTED_LANGUAGES`).
- **Modellerkennung:** Der Katalog meldet Audio-Fähigkeiten falsch (Voxtral erscheint dort als Text-Modell), deshalb werden Audio-Modelle am Namen erkannt (`AUDIO_MODEL_PATTERN`). Kommt ein neues Audio-Modell hinzu, das nicht erkannt wird, ist das Muster zu erweitern — nicht die Modality-Metadaten zu glauben.

### Text-to-speech

- Die Ausgabe kommt als **24 kHz mono WAV** zurück.
- Das Gateway **verlangt** ein `voice`-Feld (sonst `400 voice is required for speech completion`), **ignoriert** seinen Wert aber: `martin`, `alloy` und eine Fantasie-Zeichenkette liefern byte-identisches Audio. Jedes Modell hat genau eine Stimme, deshalb gibt es keinen Voice-Picker.
- **Sprachabdeckung** wird aus dem Modellnamen abgeleitet (`TTS_LANGUAGES_BY_PATTERN`), weil der Katalog keine Sprachdaten liefert. Beansprucht werden nur verifizierte Sprachen; die deutsche Kokoro-Stimme spricht kein Englisch, CosyVoice3 kann beides.
- **Gemessene Latenz** (live, aus Deutschland): Kokoro ~1,0 s für einen kurzen deutschen Satz. CosyVoice3 antwortete im Test mit `voice=default` **nicht** innerhalb von 60 s — vor dem produktiven Einsatz erneut prüfen, sonst läuft die Pipeline in den `TTS_TIMEOUT` (30 s).

### Standardauswahl

Der Picker wählt vor: die **Release-Variante** eines Modells, wenn es sie gibt, sonst die einzige verfügbare (`qsu`). Für Transkription existiert derzeit nur ein `qsu`-Modell — das ist der einzige Grund, warum hier ein Staging-Kanal zum Einsatz kommt.

## Modelle

Der **Modell-Picker** zeigt alle für dein Konto verfügbaren Modelle auf `ai.noris.de/v1`.

### Verfügbare Modelltypen

- **`vllm/*`** — On-premises-Modelle im Rechenzentrum Nürnberg 6. Sie stehen in der Liste ganz oben und sind die richtige Wahl, wenn Datenschutz eine Rolle spielt (z. B. `vllm/gpt-oss-120b`).
- **Anthropic-Modelle** — ebenfalls wählbar, werden aber über externe APIs geroutet.

### Automatisch gefiltert

- **Reranker-Modelle** — werden nicht angezeigt (nicht für Conversation/Task geeignet).
- **Draft-Modelle** — kleine Hilfsmodelle werden ausgeblendet.
- **Audio- und Speech-Modelle** — Voxtral, Kokoro, CosyVoice3 und Verwandte erscheinen nicht im Chat-Picker, sondern nur im passenden Sprach-Picker. Der Katalog stempelt sie mit dem Standard-Chat-Template, weshalb sie zuvor fälschlich als Chat-Modelle angeboten wurden und zur Laufzeit scheiterten.

### Standard

Das Standardmodell ist **`vllm/gpt-oss-120b`** (ein großes, generalistisches Instruct-Modell mit guter Function-Calling-Unterstützung).

## Hinweise

### Authentifizierung

Die Integration authentifiziert sich per Standard-**`Authorization: Bearer`** mit dem `sk-bf-...`-Schlüssel — genau wie bei jeder OpenAI-kompatiblen API. Es sind **keine benutzerdefinierten Header** nötig. (Das Gateway akzeptiert zusätzlich einen `x-bf-vk`-Header, die Integration verwendet ihn aber nicht.)

### Reasoning-Modelle und „Gedanken"

Manche Modelle unterstützen **Reasoning** — das Modell stellt vor der eigentlichen Antwort interne Überlegungen an.

- Die Integration erfasst `<think>...</think>`-Blöcke und vLLM-Reasoning-Felder.
- Diese werden als separate **"Gedanken"** in der Nachricht angezeigt (getrennt von der Antwort).
- Nützlich zum Debuggen und zum Verständnis der Modelllogik.

### Token-Budget bei Reasoning-Modellen

Reasoning-Modelle benötigen großzügigere Token-Limits:

- **Conversation:** Standard `3000` Token `max_tokens`.
- **AI Task:** Standard `8000` Token `max_tokens`.

Falls dein Modell `max_tokens` überschreitet oder abbricht, erhöhe diese Werte in den erweiterten Einstellungen.

### Defekte Modelle und Fehlerbehandlung

Gelegentlich ist ein Modell im Gateway vorübergehend nicht nutzbar (typische Fehlermeldung des Gateways: "no keys found").

- Die Integration zeigt in diesem Fall eine **verständliche Fehlermeldung** an, anstatt abzustürzen.
- Wechsle dann einfach über die Rekonfiguration des Subentries auf ein anderes Modell.

### TLS und Sicherheit

- Zertifikate werden über Home Assistants globalen HTTP-Client verifiziert.
- Alle Anfragen an `ai.noris.de` werden über HTTPS gemacht.

## License

Apache License 2.0 — see [LICENSE](LICENSE).

## Disclaimer

**Dies ist eine Community-Integration.** Sie ist nicht offiziell von der noris network AG unterstützt oder genehmigt. Verwende sie auf eigenes Risiko. Für Fragen oder Fehler kontaktiere bitte den Integration-Betreuer (GitHub Issues) oder die noris Community.
