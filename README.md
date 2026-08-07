# noris AI for Home Assistant

A custom Home Assistant integration for the **[ai.noris.de](https://ai.noris.de)** LLM
gateway.

It adds a **Conversation agent** (for Assist / voice &amp; text control of your
devices) and an **AI Task** entity (structured or free-text data generation for
automations), built on the OpenAI-compatible Chat Completions API of the
gateway.

## Features

- 💬 **Conversation agent** — use it in Assist (voice or text pipelines) to chat
  or control your smart home. Optionally enable *Control Home Assistant* to let
  the model call entities via tool-calling.
- 🧩 **AI Task entity** — generate free text or structured (JSON) data from
  automations and scripts, e.g. summaries, classifications, sensor values.
- ⚡ **Streaming responses** — answers stream into the chat log as they are
  generated instead of arriving in one block.
- 🧠 **Reasoning models** — `<think>…</think>` output and vLLM `reasoning` /
  `reasoning_content` fields are surfaced as separate thinking content, even
  mid-stream.
- 🎛️ **Recommended &amp; advanced settings** — sensible defaults out of the box;
  power users can tune `max_tokens`, `temperature` and `top_p` per agent/task.
- 🩺 **Diagnostics &amp; reauth** — config entry diagnostics (API key redacted)
  and a reauth flow when a key expires.
- 🌍 **Translations** — English and German UI.

## Requirements

- A Home Assistant instance running **2026.7.0** or newer.
- An **API key** for ai.noris.de (see below).
- At least one working chat model available on your gateway.

## Obtaining an API key

1. Get an API key from your ai.noris.de account / administrator.
2. Copy the full `sk-bf-…` value — you will paste it into the integration.

> **Note:** the integration authenticates with the standard
> `Authorization: Bearer` header, like any OpenAI-compatible API. The key is
> stored in Home Assistant's credential store, redacted in diagnostics and
> never logged.

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
3. Once added, open the integration card and use **Add conversation agent**
   and/or **Add AI task** to create entities:
   - Pick a model from the dropdown (loaded live from the gateway,
     default: `vllm/gpt-oss-120b`).
   - For the conversation agent, optionally enable **Control Home Assistant**
     to allow device control via tool-calling.
   - Keep **Recommended settings** enabled for sensible defaults, or disable
     it to tune advanced options.

### Advanced options

With *Recommended settings* disabled you can adjust per agent/task:

| Option | Default (conversation) | Default (AI Task) |
|--------|------------------------|-------------------|
| `max_tokens` | 3000 | 8000 |
| `temperature` | 1.0 | 1.0 |
| `top_p` | 1.0 | 1.0 |

Reasoning models spend part of the token budget on thinking **before** the
visible answer — keep `max_tokens` generous, or answers may come back empty.

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

- The model dropdown is populated **live** from `https://ai.noris.de/v1/models`
  when you add an agent/task. `vllm/*` models are listed first.
- Rerankers, embedding models and small draft models are filtered out — they
  cannot act as chat/agent models.
- **Tool-calling** (device control) requires a model that supports function
  calling. Larger instruct models (e.g. `vllm/gpt-oss-120b`) work best.
- The default model is matched resiliently: if the gateway renames a model
  with a channel prefix (e.g. `vllm/release/gpt-oss-120b`), the integration
  still finds it.

## Notes

- **Authentication:** standard `Authorization: Bearer` with your `sk-bf-…`
  key. For compatibility with older gateway configurations the legacy
  `x-bf-vk` header is currently still sent alongside; it will be removed in a
  future release.
- **TLS:** certificates are verified via Home Assistant's shared HTTP client.
- **Reasoning models:** `<think>…</think>` output and vLLM `reasoning` /
  `reasoning_content` fields are surfaced as separate thinking content.

## Troubleshooting

| Symptom | Likely cause / fix |
|---------|--------------------|
| **Invalid authentication** during setup | Wrong/expired API key. Re-enter the full `sk-bf-…` value. The integration also starts a reauth flow automatically when a key expires. |
| **Failed to connect** | Gateway unreachable or network issue. Check connectivity to `ai.noris.de`. |
| Empty answer / "increase max_tokens" error | The model spent the whole token budget thinking. Disable *Recommended settings* and raise `max_tokens`. |
| Tool call fails / unexpected JSON | Some models wrap tool arguments in Markdown fences. The integration strips these automatically; if it still fails, try a different model. |
| Model not in the dropdown | Rerankers, embedding and draft models are hidden by design. |

## License

Apache License 2.0 — see [LICENSE](LICENSE).

## Disclaimer

Community integration, not affiliated with or supported by noris network AG.
"noris" is a trademark of noris network AG; this project uses the name only to
describe the service it integrates with.
