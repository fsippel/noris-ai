# noris AI for Home Assistant

**English Summary**

A custom Home Assistant integration for the **ai.noris.de** Bifrost gateway. Adds a **Conversation agent** (for Assist / voice control) and an **AI Task** entity (structured data generation for automations). Built on the OpenAI-compatible Chat Completions API with standard Bearer authentication. All gateway models are selectable (`vllm/*` listed first); only `vllm/*` models run on-premises. Reranker and draft models are filtered out. Default model: `vllm/gpt-oss-120b`.

---

## Was ist das?

Die **noris AI**-Integration für Home Assistant erweitert dein Smart Home um zwei zentrale Funktionen:

1. **Conversation Agent** — ein KI-basierter Sprach- und Textassistent für Home Assistant Assist. Du kannst Fragen stellen, Geräte steuern und Automationen auslösen.
2. **AI Task** — eine Entity für strukturierte und freie Datengenerierung. Nützlich für Automationen, die KI-basierte Eingaben brauchen (z. B. Datenextraktion, Klassenbildung, Synthesis).

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

## Modelle

Der **Modell-Picker** zeigt alle für dein Konto verfügbaren Modelle auf `ai.noris.de/v1`.

### Verfügbare Modelltypen

- **`vllm/*`** — On-premises-Modelle im Rechenzentrum Nürnberg 6. Sie stehen in der Liste ganz oben und sind die richtige Wahl, wenn Datenschutz eine Rolle spielt (z. B. `vllm/gpt-oss-120b`).
- **Anthropic-Modelle** — ebenfalls wählbar, werden aber über externe APIs geroutet.

### Automatisch gefiltert

- **Reranker-Modelle** — werden nicht angezeigt (nicht für Conversation/Task geeignet).
- **Draft-Modelle** — kleine Hilfsmodelle werden ausgeblendet.

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
