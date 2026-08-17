# Speech-to-Text for the noris AI integration

**Date:** 2026-08-15
**Status:** Approved design, not yet implemented
**Scope:** Add a Speech-to-Text platform to `custom_components/noris_ai`. Text-to-Speech is
explicitly out of scope — see [Non-goal: TTS](#non-goal-tts).

## 1. Context and goal

The integration today exposes two platforms — `conversation` and `ai_task` — both built on the
OpenAI-compatible Chat Completions API of the `ai.noris.de` gateway. The goal is to let the same
gateway serve the *speech* half of an Assist pipeline, so a voice command can be transcribed by
noris rather than by a local or third-party engine.

The request was modelled on Home Assistant's `google_generative_ai_conversation` integration,
which ships both `tts.py` and `stt.py`. Only the STT half turns out to be achievable, and the
right implementation is materially different from Google's. Both conclusions rest on live probes
of the gateway, recorded below.

## 2. What the gateway actually supports

Probed 2026-08-15 against `https://ai.noris.de/v1` with the API key configured on the production
Home Assistant instance. The gateway is a **Bifrost** router (errors carry `is_bifrost_error`)
fronting exactly one provider: `vllm`.

### 2.1 Transcription works

```
POST /v1/audio/transcriptions
  file=<5 s German WAV, 16 kHz mono 16-bit>
  model=vllm/qsu/voxtral-small-24b-2507
→ HTTP 200 in 675 ms upstream
  {"text":"Schalte bitte das Licht im Wohnzimmer ein und stelle die Markise auf 50%.",
   "usage":{"type":"duration","seconds":5}}
```

`vllm/qsu/voxtral-small-24b-2507` is Mistral's **Voxtral Small 24B**
(`hugging_face_id: mistralai/Voxtral-Small-24B-2507`), an audio-understanding model. It
transcribed a German smart-home utterance verbatim and correctly, including the domain-specific
noun *Markise*.

### 2.2 Constraints discovered

| Probe | Result | Consequence for the design |
|---|---|---|
| Raw PCM posted without a WAV header | `400 Invalid or unsupported audio file` | The entity **must** wrap the pipeline's PCM in a WAV container |
| `response_format=verbose_json` | `400 Currently do not support verbose_json` | No segment/timestamp data available |
| `response_format=text` | `200`, but the gateway double-wraps the JSON body *inside* the `text` field | **Must** use the default `json` format and read `.text` |
| `language=de` + `prompt=…` | Accepted, `200` | `language` is passed; `prompt` showed no observable effect and is omitted (YAGNI) |
| Audio via `chat/completions` `input_audio` | `200`, audio understood, but output rambled and hallucinated: `"... Kunde. Transkribiere wortlich." **Vorgang:** … auf 50% lucht.` | **Do not** copy Google's LLM-with-prompt approach |
| Audio posted with a text model (`vllm/release/gemma-4-31b-it`) | `404` — `provider API error (status 404)`, body `{"detail":"Not Found"}` | A non-audio model yields 404, so the entity can give a model-specific error |

The last row is the key architectural divergence. Google routes STT through `generate_content`
because Gemini exposes no transcription endpoint. noris exposes a real one, and it is both more
accurate and far cheaper in tokens. We use it.

### 2.3 The model catalog misreports capability

`/v1/models` returns rich metadata — `input_modalities`, `output_modalities`, `hugging_face_id`,
`is_ready`, `datacenters`, `compliance`, `pricing`. But Voxtral's entry declares:

```json
"input_modalities": [{"type": "text", ...}]
```

No audio input, despite transcription demonstrably working. **Modality metadata therefore cannot
be used to detect audio-capable models.** See [§6](#6-model-filtering) for the workaround.

The metadata is reliable in the other direction: `output_modalities[].type` correctly distinguishes
`text` from `rerank` and `embeddings`.

## 3. Non-goal at the time of writing: TTS

> **Superseded on 2026-08-17.** The gateway now exposes working speech models and this section's
> conclusion no longer holds. It is kept because the reasoning explains why the STT work shipped
> alone. See `docs/specs/2026-08-17-tts-design.md` for the text-to-speech design.

When this was written, Text-to-Speech was **not implementable** against the gateway, and no amount
of integration code would have changed that:

```
POST /v1/audio/speech
  model=vllm/…  → 400 "speech is not supported by vllm provider"
```

Probes with model ids naming any other speech-capable provider were also rejected, each with
`failed to get config for provider …: not found` — nothing but `vllm` was configured on the
gateway at that point.

The refusal was at **provider** level, not model level. Adding a TTS model to the vLLM fleet would
still have failed, because Bifrost rejected speech for the `vllm` provider outright. Enabling TTS
required noris to configure an additional provider.

**Decision at the time:** ship STT only.

**What changed.** On 2026-08-17 the catalog grew from 25 to 30 models and new providers appeared —
`Kokoro-TTS`, `Cosyvoice3` and `vibevoice-asr`. `/v1/audio/speech` now returns audio:

```
POST /v1/audio/speech  model=Kokoro-TTS/release/kokoro-tts-german-martin  → 200, 1.09 s
POST /v1/audio/speech  model=Cosyvoice3/release/cosyvoice3-0.5b-rl       → 200, 1.95 s
```

Both return WAV / PCM s16le / 24 kHz / mono. The audio was verified as genuine speech by
synthesising a German sentence and transcribing it back through Voxtral, which returned the input
text verbatim.

## 4. Architecture

A third platform alongside the existing two, following the integration's established pattern
exactly: **one `stt` subentry → one `SpeechToTextEntity` → one service device.**

```
ConfigEntry (API key, shared AsyncOpenAI client in runtime_data)
├── subentry "conversation"  → NorisAIConversationEntity   (existing)
├── subentry "ai_task_data"  → NorisAITaskEntity           (existing)
└── subentry "stt"           → NorisAISttEntity            (new)
```

The STT entity reuses the shared `AsyncOpenAI` client from `entry.runtime_data`, so it inherits
the custom `x-bf-vk` header, Home Assistant's shared httpx client and TLS verification. **No new
requirements** — `client.audio.transcriptions.create()` is part of the `openai` SDK already
pinned in the manifest.

The new entity deliberately does **not** inherit `NorisAIEntity`. That base class exists to drive
`_async_handle_chat_log` — tools, thinking extraction, tool-call loops — none of which apply to
transcription. It inherits `stt.SpeechToTextEntity` and duplicates only the handful of lines that
set `unique_id` and `device_info`.

## 5. Components

| File | Change |
|---|---|
| `stt.py` | **new** — `async_setup_entry` over `stt` subentries; `NorisAISttEntity` |
| `helpers.py` | **new** — `pcm_to_wav(audio, rate, width, channels) -> bytes` |
| `const.py` | `STT_SUBENTRY_TYPE`, `AUDIO_MODEL_PATTERN`, `STT_SUPPORTED_LANGUAGES`, `STT_TIMEOUT` |
| `config_flow.py` | `SttFlowHandler`; register the subentry type; `_fetch_model_options` takes a predicate; replace `_is_selectable_model` string hacks with metadata |
| `__init__.py` | add `Platform.STT` to `PLATFORMS` |
| `manifest.json` | add `"stt"` to `dependencies`; bump `0.2.2` → `0.3.0` |
| `strings.json`, `translations/{en,de}.json` | subentry step titles, field labels, error/abort strings |
| `README.md` | STT setup and usage; the TTS limitation with the gateway error quoted |

### 5.1 `helpers.pcm_to_wav`

Google's equivalent parses a MIME string (`audio/L16;rate=24000`) to recover its parameters. We
receive them directly and typed on `stt.SpeechMetadata`, so the helper takes integers instead:

```python
def pcm_to_wav(audio: bytes, rate: int, width: int, channels: int) -> bytes:
    """Wrap raw PCM in a WAV container; the gateway rejects headerless audio."""
```

Implemented with the stdlib `wave` module over `io.BytesIO`. Pure, synchronous, in-memory, and
trivially unit-testable — the whole reason it is a separate module rather than inlined.

### 5.2 Entity contract

```python
supported_formats     = [stt.AudioFormats.WAV]
supported_codecs      = [stt.AudioCodecs.PCM]
supported_bit_rates   = [stt.AudioBitRates.BITRATE_16]
supported_sample_rates= [stt.AudioSampleRates.SAMPLERATE_16000]
supported_channels    = [stt.AudioChannels.CHANNEL_MONO]
```

These are exactly what the Assist pipeline emits. OGG/Opus is *not* declared: it was not probed
against the gateway, and claiming untested support is how silent failures get shipped.

`supported_languages` declares regional variants of **Voxtral's eight documented languages** —
German, English, Spanish, French, Italian, Dutch, Portuguese, Hindi:

```
de-DE de-AT de-CH
en-US en-GB en-AU en-CA en-IE en-IN en-NZ en-ZA
es-ES es-MX es-AR es-CO es-US
fr-FR fr-BE fr-CA fr-CH
it-IT it-CH
nl-NL nl-BE
pt-PT pt-BR
hi-IN
```

We do not mirror Google's ~130-language list. Claiming languages the model does not support
produces confident, wrong transcripts rather than an honest failure to match.

## 6. Data flow

```
Assist pipeline ──raw 16 kHz mono 16-bit PCM chunks──▶ async_process_audio_stream(metadata, stream)
  │
  ├─ buffer all chunks into bytes
  │     HA's STT entity API returns a single SpeechResult; there is no streaming result path,
  │     and the gateway offers no streaming transcription either.
  │
  ├─ guard: empty buffer → SpeechResult(None, ERROR), no API call
  │
  ├─ pcm_to_wav(buf, metadata.sample_rate.value, metadata.bit_rate.value // 8,
  │             metadata.channel.value)
  │     Note: these three are enum members (AudioSampleRates, AudioBitRates,
  │     AudioChannels), not bare ints — `.value` is required.
  │
  ├─ client.with_options(timeout=STT_TIMEOUT).audio.transcriptions.create(
  │       file=("audio.wav", wav_bytes, "audio/wav"),
  │       model=self.model,
  │       language=metadata.language.split("-")[0],   # "de-DE" → "de"
  │   )                                               # default response_format=json
  │
  ├─ text = (response.text or "").strip()
  └─ SpeechResult(text, SUCCESS) if text else SpeechResult(None, ERROR)
```

`STT_TIMEOUT = 30.0` seconds. The observed latency is ~700 ms for a 5-second utterance; the
timeout exists so a stalled gateway fails the pipeline promptly instead of hanging the voice
satellite.

## 7. Model filtering

`_fetch_model_options` gains a predicate parameter so each subentry flow requests the models it
can actually use. Two filters result.

### 7.1 Audio models (new, for the STT flow)

Because the catalog misreports modalities (§2.3), audio capability is detected by matching known
audio model families against **`hugging_face_id`** and the model id:

```
voxtral | whisper | qwen.*audio | granite-speech
```

`hugging_face_id` is the stronger signal (`mistralai/Voxtral-Small-24B-2507`) and is checked
first — but it is **`null` for many models on the gateway** (`gpt-oss-120b`, the `harrier-oss`
draft models, several rerankers), so the predicate must treat a missing or null value as "no
match" and fall through to the id check rather than raising. This is precise today and picks up a future Whisper deployment automatically. The trade-off
— an unrecognised new audio family stays hidden until its name is added — is accepted, and the
constant carries a comment explaining *why* it exists, so a later reader does not "clean up" the
heuristic in favour of the modality metadata that does not work.

### 7.2 Chat models (a fix to existing behaviour)

`_is_selectable_model` currently excludes rerankers and draft models by substring-matching
`"reranker"` and `"harrier"` in the id. The catalog carries the real answer, so this becomes:

```
output_modalities[].type == "text"      # excludes rerank and embeddings models
AND NOT audio-family match              # excludes Voxtral
AND id.startswith("vllm/")              # unchanged: existing data-locality policy
```

This fixes a live bug: **Voxtral currently appears in the conversation and AI Task model
dropdowns**, where it is a poor chat model with no tool-calling support. The `vllm/` prefix rule
is deliberately left alone — it is an existing policy decision and changing it is out of scope.

**Implementation trap:** `hugging_face_id` and `output_modalities` are not fields of the OpenAI
SDK's `Model` type. They must be read via `model.model_extra`, the same mechanism `entity.py`
already uses to recover vLLM's `reasoning` field.

## 8. Error handling

`async_process_audio_stream` never raises. It returns `SpeechResult(None, SpeechResultState.ERROR)`
and lets the pipeline surface the failure to the user, with a distinct log line per cause:

| Condition | Behaviour |
|---|---|
| Empty or silent buffer | ERROR immediately, no API call |
| `AuthenticationError`, `PermissionDeniedError` | log; `entry.async_start_reauth(hass)` so HA raises the repair flow |
| `APIStatusError` with 404 | the configured model is not audio-capable; log naming the model and the remedy |
| `APITimeoutError` | log; ERROR (bounded by `STT_TIMEOUT`) |
| any other `OpenAIError` | log; ERROR |
| Blank or whitespace-only transcript | ERROR — never hand Assist an empty sentence as a success |

**Privacy:** transcripts are voice content. They are logged at `debug` only, never at `info`. The
API key is never logged, and audio bytes are never logged at any level.

## 9. Configuration and UX

A new `SttFlowHandler(ConfigSubentryFlow)` mirroring `AITaskFlowHandler`:

- `async_step_user` → `async_step_init`
- aborts `entry_not_loaded` if the parent entry is not loaded
- aborts `cannot_connect` / `unknown` on model-list failure
- single required field: `CONF_MODEL`, a sorted dropdown of audio-capable models
- subentry title = the chosen model id
- `async_step_reconfigure` allows changing the model on an existing entity

No prompt, temperature, or language field. Language comes from the pipeline at request time;
`prompt` showed no observable effect on the gateway; temperature is meaningless for transcription.

Strings are added to `strings.json` and both translation files. German translations are written
in full — the integration already ships a complete German UI and a half-translated flow would be
a regression.

## 10. Testing and CI

The repository has **no tests, no CI and no dev tooling** today. This work establishes all three
and backfills coverage for the whole integration, to the standard Home Assistant core applies to
its own integrations — the explicit goal being that this fork could be offered upstream.

**Tooling:** `requirements_test.txt` with `pytest-homeassistant-custom-component` (which pins a
matching HA version and pulls in pytest, pytest-asyncio and syrupy) and `ruff`.

**Fixtures** (`tests/conftest.py`): `auto_enable_custom_integrations`; a `mock_config_entry`
carrying conversation, `ai_task_data` and `stt` subentries; a patched `_create_client` returning
an `AsyncMock` whose `models.list()` yields fake models with populated `model_extra`.

| Test module | Covers |
|---|---|
| `test_helpers.py` | `pcm_to_wav` round-tripped back through `wave`: channels, sample width, frame rate, frame payload, 44-byte header. The highest-value test here — fiddly bit-packing that fails silently. |
| `test_stt.py` | happy path asserting the posted file really begins `RIFF` and that `language="de"` was sent; empty stream; `OpenAIError`; 404; auth error triggering reauth; blank transcript; entity-property snapshot |
| `test_config_flow.py` | user / reauth / reconfigure steps × success, `invalid_auth`, `cannot_connect`, `unknown`; STT subentry offers Voxtral but not gpt-oss or rerankers; conversation subentry offers gpt-oss but **not** Voxtral (pins the §7.2 fix); `entry_not_loaded` |
| `test_init.py` | setup; `ConfigEntryAuthFailed` on 401/403; `ConfigEntryNotReady`; unload; reload-on-update |
| `test_conversation.py` | backfill: LLM API wiring, `CONTROL` feature flag, `ConverseError` handling |
| `test_ai_task.py` | backfill: free-text and structured generation, `response_format` construction |
| `test_entity.py` | backfill: tool-call loop and `MAX_TOOL_ITERATIONS`; `<think>` extraction and the `reasoning` / `reasoning_content` precedence; Markdown-fence tool-argument recovery; empty-choices error path |

**CI** (`.github/workflows/`): ruff check and format, `home-assistant/actions/hassfest`,
`hacs/action` with `category: integration`, and pytest with coverage.

## 11. Rollout and verification

1. Back up `/config/custom_components/noris_ai/` on the Home Assistant host.
2. Deploy the updated `custom_components/noris_ai/` to the host.
3. `ha core check`.
4. **`ha core restart`** — a *new platform* in a custom component is not picked up by a reload.
   This interrupts all integrations, so it is confirmed with the user first.
5. Settings → Devices & Services → noris AI → add a **Speech-to-text** subentry; select
   `vllm/qsu/voxtral-small-24b-2507`.
6. Settings → Voice assistants → set the pipeline's STT engine to the new entity.
7. Speak a German command; confirm the transcript in the pipeline debug trace and the entity's
   presence in `/api/states`.

Rollback is restoring the backup directory and restarting.

## 12. Risks and follow-ups

- **Voxtral is only in the `qsu/` namespace.** Every other gateway model exists in both `qsu/` and
  `release/`; Voxtral exists only in `qsu/`, which appears to be a staging tier. It may be less
  stable or may move. Worth raising with noris, and it is why the model is user-selected rather
  than hardcoded.
- **The audio-model heuristic is a workaround** for wrong catalog metadata. If noris fixes
  `input_modalities`, the filter should switch to metadata and the heuristic can be deleted.
- **TTS remains blocked** at the gateway. Revisit if noris configures a speech-capable provider.
- **OGG/Opus support** was never probed. If a future voice satellite sends Opus, test the gateway
  before declaring the format.
- The `vllm/`-only model policy is expressed as a provider-name prefix. Unchanged here, but the
  catalog exposes `datacenters` and `compliance.zdr`, which would express the underlying
  data-locality intent more directly if that rule is ever revisited.

## 13. Decisions

| Decision | Choice | Rationale |
|---|---|---|
| TTS scope | Ship STT only; document TTS as unavailable | Blocked at the gateway's provider layer; unfixable in integration code |
| Model selection | `stt` subentry with a filtered live model picker | Matches the existing subentry pattern; no hardcoded id to rot |
| Audio-model detection | Name match on `hugging_face_id` / id | Catalog modality metadata is wrong for Voxtral |
| Transcription path | `/v1/audio/transcriptions` | Accurate and fast; the Google-style chat path hallucinated |
| Test scope | Full HA-core-style suite, whole integration backfilled | Upstream-contribution quality bar |
