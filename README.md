# noris AI for Home Assistant

A custom Home Assistant integration for the **[ai.noris.de](https://ai.noris.de)** LLM
gateway.

It adds a **Conversation agent** (for Assist / voice &amp; text control of your
devices), an **AI Task** entity (structured or free-text data generation for
automations), a **Speech-to-text** entity (transcribe Assist voice commands),
and a **Text-to-speech** entity (speak through the gateway's speech models),
built on the OpenAI-compatible Chat Completions, transcription, and speech APIs
of the gateway.

For the **Conversation agent** and **AI Task** entities, only **`vllm/*`** models
are offered in the model picker; rerankers and small draft models are filtered
out automatically. Speech-to-text and Text-to-speech entities draw models from
other providers available on the gateway.

## Features

- 💬 **Conversation agent** — use it in Assist (voice or text pipelines) to chat
  or control your smart home. Optionally enable *Control Home Assistant* to let
  the model call entities via tool-calling.
- 🧩 **AI Task entity** — generate free text or structured (JSON) data from
  automations and scripts, e.g. summaries, classifications, sensor values.
- 🧠 **Reasoning models** — `thinking` output and vLLM `reasoning` /
  `reasoning_content` fields are surfaced as separate thinking content.
- 🌍 **Translations** — English and German UI.
- 🎙️ **Speech-to-text and Text-to-speech** — transcribe Assist voice commands with an audio model
  (Voxtral), or speak through available speech models. Add *Speech-to-text* or *Text-to-speech*
  entities and select them in your voice pipeline.

## Requirements

- A Home Assistant instance running **2026.7.0** or newer.
- An **API key** for ai.noris.de (see below).
- At least one working **`vllm/*` chat model** available on your gateway (for the
  Conversation agent and AI Task).
- For Speech-to-text: an audio model such as Voxtral or Whisper.
- For Text-to-speech: a speech model such as Cosyvoice3 or Kokoro-TTS.

## Obtaining an API key

The ai.noris.de gateway authenticates requests with a custom `x-bf-vk` header
that carries an API key in the form `sk-bf-…`.

1. Get an API key from your ai.noris.de account / administrator.
2. Copy the full `sk-bf-…` value — you will paste it into the integration.

> **Note:** the key is sent in the `x-bf-vk` header (handled automatically by
> this integration). It is stored encrypted in Home Assistant's credential
> store and never logged.

## Installation

### HACS (recommended)

1. In Home Assistant, open **HACS** → ⋮ (top right) → *Custom repositories*.
2. Add `https://github.com/fsippel/noris-ai`, category **Integration**.
3. Search for **noris AI** and install it.
4. Restart Home Assistant.

### Manual

Copy the `custom_components/noris_ai` folder into your Home Assistant
`config/custom_components/` directory and restart Home Assistant.

## Configuration

1. Go to **Settings → Devices &amp; Services → Add Integration** and search for
   **noris AI**.
2. Paste your API key (the `sk-bf-…` value). The key is validated against
   `https://ai.noris.de/v1/models`.
3. Once added, open the integration card and use **Add conversation agent**,
   **Add AI task**, **Add speech-to-text**, and/or **Add text-to-speech** to create
   entities:
   - For the **Conversation agent** and **AI task**, pick a `vllm/*` model from the
     dropdown (loaded live from the gateway).
   - For the conversation agent, optionally enable **Control Home Assistant**
     to allow device control via tool-calling.
   - For **Speech-to-text**, only models the gateway can transcribe with are listed
     — see [Speech-to-text](#speech-to-text) below.
   - For **Text-to-speech**, select from available speech models — see
     [Text-to-speech](#text-to-speech) below.

## Using the conversation agent

A conversation agent can be used in any **Assist pipeline** (voice or text):

1. **Settings → Assist pipelines** → create or edit a pipeline.
2. Set the *Conversation agent* to your noris AI agent.
3. Talk or type — the agent responds and, if *Control Home Assistant* is
   enabled, can turn devices on/off, read states, etc.

Expose the entities you want the agent to control under
**Settings → Assist → Exposed entities**.

## Using AI Task

The AI Task entity generates data you can use in automations, scripts and
template sensors. Call the `ai_task.generate_data` action with natural-language
instructions and (optionally) a `structure` to get structured JSON back.

### Example 1 — simple text generation

Generate a friendly notification when a window is left open:

```yaml
automation:
  - alias: "Window left open reminder"
    triggers:
      - trigger: state
        entity_id: binary_sensor.living_room_window
        to: "on"
        for:
          minutes: 15
    actions:
      - action: ai_task.generate_data
        data:
          task_name: "window open reminder"
          instructions: >
            Write a short, friendly reminder to close the living room window.
            Mention it has been open for 15 minutes.
        response_variable: generated_text
      - action: notify.mobile_app
        data:
          message: "{{ generated_text.data }}"
```

### Example 2 — structured output (template sensor)

Classify sensor readings into a structured response and expose the result as a
sensor:

```yaml
template:
  - triggers:
      - trigger: time_pattern
        minutes: "/30"
    actions:
      - action: ai_task.generate_data
        data:
          task_name: "{{ this.entity_id }}"
          instructions: >
            Given the outdoor temperature of
            {{ states('sensor.outdoor_temperature') }} °C and the indoor
            temperature of {{ states('sensor.indoor_temperature') }} °C,
            classify the overall comfort level and suggest an action.
          structure:
            comfort_level:
              selector:
                select:
                  options: ["cold", "cool", "comfortable", "warm", "hot"]
            suggestion:
              selector:
                text:
        response_variable: result
    sensor:
      - name: "Comfort level"
        state: "{{ result.data.comfort_level }}"
        attributes:
          suggestion: "{{ result.data.suggestion }}"
```

### Example 3 — summarize a long text

Summarize the day's events into a short digest:

```yaml
script:
  - alias: "Daily digest"
    sequence:
      - action: ai_task.generate_data
        data:
          task_name: "daily digest"
          instructions: >
            Summarize the following events into 3 bullet points:
            {{ states('input_text.todays_events') }}
        response_variable: digest
      - action: notify.persistent_notification
        data:
          title: "📋 Daily digest"
          message: "{{ digest.data }}"
```

> ℹ️ Set a **preferred AI task entity** under **Settings → AI** so automations
> can omit the entity ID. See the
> [AI Task docs](https://www.home-assistant.io/integrations/ai_task/) for the
> full action reference.

## Available models

- **Chat and AI-task dropdowns:** populated **live** from `https://ai.noris.de/v1/models`
  when you add an agent or task. Only `vllm/*` models are listed; models whose ID
  contains `reranker` or `harrier` (draft models) are filtered out.
- **Speech-to-text and Text-to-speech dropdowns:** populated live from available
  models on the gateway. The integration recognizes audio and speech models by name
  (Voxtral, Whisper, Kokoro-TTS, Cosyvoice3, etc.).
- **Tool-calling** (device control) requires a chat model that supports function
  calling. Larger instruct models (e.g. `vllm/gpt-oss-120b`,
  `vllm/gemma-4-31b-it`) work best.
- Even temporarily unavailable models remain selectable; availability changes
  over time and errors are reported cleanly at runtime.

## Notes

- **Authentication:** the gateway uses a custom `x-bf-vk` header instead of the
  standard `Authorization: Bearer` scheme. This is handled by the integration.
- **TLS:** certificates are verified via Home Assistant's shared HTTP client.
- **Reasoning models:** `<think>…</think>` output and vLLM `reasoning` /
  `reasoning_content` fields are surfaced as separate thinking content.

## Speech-to-text

The integration can transcribe voice commands using the gateway's audio models.

1. Go to **Settings → Devices & Services → noris AI** and add a **Speech-to-text** entity.
2. Pick an audio model. Only models that can transcribe are listed — today that is
   `vllm/qsu/voxtral-small-24b-2507` (Mistral's Voxtral Small 24B).
3. Go to **Settings → Voice assistants**, open your pipeline and set **Speech-to-text** to the new
   entity.

Supported languages are the ones Voxtral documents: German, English, Spanish, French, Italian,
Dutch, Portuguese and Hindi. Audio is sent as 16 kHz mono PCM wrapped in a WAV container.

> **Note on model detection:** the gateway's `/v1/models` catalog does not report audio capability
> correctly, so the integration recognises audio models by name (Voxtral, Whisper and similar). If
> your gateway gains an audio model that is not listed, open an issue.

## Text-to-speech

The integration can speak through the gateway's speech models.

1. Go to **Settings → Devices & Services → noris AI** and add a **Text-to-speech** entity.
2. Pick a speech model. Two are available today:
   - `Cosyvoice3/release/cosyvoice3-0.5b-rl` — multilingual, handles German and English.
   - `Kokoro-TTS/release/kokoro-tts-german-martin` — a German voice. It reads English
     phonetically, so pick it only for German.
3. Go to **Settings → Voice assistants**, open your pipeline and set **Text-to-speech** to the new
   entity.

Audio comes back as 24 kHz mono WAV. The languages an entity advertises are derived from its model
name, because the gateway's model catalog does not report language support.

> **Note:** the gateway requires a `voice` parameter but ignores its value — each model has exactly
> one voice — so the integration offers no voice picker.

## Troubleshooting

| Symptom | Likely cause / fix |
|---------|--------------------|
| **Invalid authentication** during setup | Wrong/expired API key. Re-enter the full `sk-bf-…` value. |
| **Failed to connect** | Gateway unreachable or network issue. Check connectivity to `ai.noris.de`. |
| Empty answer from a reasoning model | `max_tokens` too small — the model spent all tokens "thinking". Use a model with a higher token budget or reduce the task complexity. |
| Tool call fails / unexpected JSON | Some models wrap tool arguments in Markdown fences. The integration strips these automatically; if it still fails, try a different model. |
| Model not in the dropdown | Only `vllm/*` chat models are shown; rerankers and draft models are hidden by design. |

## License

Apache License 2.0 — see [LICENSE](LICENSE).

## Disclaimer

Community integration, not affiliated with or supported by noris network AG.
"noris" is a trademark of noris network AG; this project uses the name only to
describe the service it integrates with.
