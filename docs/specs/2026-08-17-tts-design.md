# Text-to-Speech for the noris AI integration

**Date:** 2026-08-17
**Status:** Implemented. Live verification against the production instance is still outstanding.
**Supersedes:** §3 of `docs/specs/2026-08-15-stt-design.md`, which concluded that text-to-speech
was not implementable against this gateway. That was true when written and is no longer true.

## 1. Context and goal

The integration exposes three platforms — `conversation`, `ai_task` and `stt`. A voice pipeline
built on it can therefore hear and think, but not speak: the speaking half has to come from a
separate engine such as Home Assistant's local Piper.

On 2026-08-17 the gateway's catalog grew from 25 to 30 models and, decisively, **new providers
appeared**. The previous blocker was that Bifrost refused `/v1/audio/speech` for the `vllm`
provider outright, and no other provider was configured — a refusal no integration code could work
around. Speech models now exist under their own providers and the endpoint answers.

The goal is a fourth platform so the same gateway can serve the whole pipeline.

## 2. What the gateway supports

Probed 2026-08-17 against `https://ai.noris.de/v1` with the API key configured on the production
Home Assistant instance.

### 2.1 Both models work

```
POST /v1/audio/speech  model=Kokoro-TTS/release/kokoro-tts-german-martin  → 200, 1.09 s, 132 876 B
POST /v1/audio/speech  model=Cosyvoice3/release/cosyvoice3-0.5b-rl        → 200, 1.95 s, 134 444 B
```

Both return **WAV / PCM s16le / 24 000 Hz / mono**, parsed from the returned header. Unlike
Voxtral — which exists only under `qsu/` — each is published under both `qsu/` and `release/`, so
the staging-tier concern recorded in the STT spec does not apply here.

The audio is genuine speech, not silence or noise. Verified by round trip: Kokoro synthesised
*"Das Licht im Wohnzimmer ist jetzt eingeschaltet."*, and Voxtral transcribed the result back to
exactly that sentence.

### 2.2 Language coverage differs sharply between the two

Established by the same round trip — synthesise, resample to 16 kHz, transcribe:

| Model | German | English |
|---|---|---|
| `Kokoro-TTS/*/kokoro-tts-german-martin` | clean | **mangled** — *"In turn de on in de de ketien in light de pliaze."* |
| `Cosyvoice3/*/cosyvoice3-0.5b-rl` | clean | clean — *"Turn on the kitchen light, please."* |

Kokoro is a German voice reading foreign text phonetically. CosyVoice3 is genuinely multilingual.
**This drives the language design in [§5](#5-declaring-languages).**

### 2.3 Parameter quirks

| Probe | Result | Consequence |
|---|---|---|
| `voice` omitted | `400 "voice is required for speech completion"` | The field **must** be sent |
| `voice` = `martin` / `alloy` / `zzz_not_a_voice` | all three returned byte-identical audio | The value is **ignored**; each model has one fixed voice |
| `response_format=mp3` | `200`, but the body is RIFF WAV | Ignored; output is always WAV |
| `speed=1.5` | `200`, 48 044 B vs 62 444 B at default | Genuinely works. Range unprobed |
| English text through the German model | `200`, mangled speech | Not an error — silently poor output |

### 2.4 The catalog still misreports capability

Exactly as with Voxtral, every audio model carries the default chat template:

```json
"input_modalities":  [{"type": "text"}],
"output_modalities": [{"type": "text",
                       "supported_parameters": {"stop": …, "temperature": …, "top_p": …},
                       "streaming": true}]
```

No model declares an `audio` modality, `hugging_face_id` is `null` on all four new entries, and the
advertised parameters (`temperature`, `top_p`, `stop`) do not exist on the audio endpoints. **Speech
models therefore have to be detected by name**, and their languages derived by name, exactly like
the existing `AUDIO_MODEL_PATTERN` workaround.

A precise description of what correct catalog entries would look like has been prepared for noris.
When the catalog is fixed, both heuristics in this design should be deleted in favour of metadata.

## 3. Architecture

A fourth platform following the integration's established subentry pattern:

```
ConfigEntry (API key, shared AsyncOpenAI client in runtime_data)
├── subentry "conversation"  → NorisAIConversationEntity   (existing)
├── subentry "ai_task_data"  → NorisAITaskEntity           (existing)
├── subentry "stt"           → NorisAISttEntity            (existing)
└── subentry "tts"           → NorisAITtsEntity            (new)
```

**No new runtime requirements** — `client.audio.speech.create()` is part of the `openai` SDK
already pinned in the manifest, and the entity reuses the shared client from `entry.runtime_data`,
inheriting the `x-bf-vk` header and TLS handling.

The entity is simpler than its STT counterpart: the gateway returns a complete WAV container, so
there is nothing to wrap. `pcm_to_wav` is not involved.

### 3.1 Removing the thrice-duplicated device block

`NorisAISttEntity.__init__` is a character-for-character copy of `NorisAIEntity.__init__` — the
`entry`/`subentry`/`model` assignment, `_attr_unique_id`, and the whole `dr.DeviceInfo` block
including the `"noris network AG"` literal. A reviewer flagged this during the STT work and it was
deliberately deferred, on the grounds that a mixin was a refactor better done when a second
non-chat platform appeared.

That platform is this one. Adding a third copy is the wrong answer, so this design extracts a
`NorisAISubentryEntity` mixin holding exactly that constructor. `NorisAIEntity` and both non-chat
entities inherit it. This is the only refactor in scope; nothing else moves.

## 4. Components

| File | Change |
|---|---|
| `tts.py` | **new** — `async_setup_entry` over `tts` subentries; `NorisAITtsEntity` |
| `entity.py` | extract `NorisAISubentryEntity` (constructor + device info); `NorisAIEntity` inherits it |
| `stt.py` | inherit the mixin instead of duplicating the constructor |
| `const.py` | `TTS_SUBENTRY_TYPE`, `TTS_TIMEOUT`, `TTS_MODEL_PATTERN`, `TTS_VOICE`, `TTS_LANGUAGES_BY_PATTERN`, `TTS_DEFAULT_LANGUAGES`, `DEFAULT_TTS_NAME` |
| `config_flow.py` | `_is_tts_model` predicate; `TtsFlowHandler`; register the subentry type |
| `__init__.py` | `Platform.TTS`; extend the migration's title table |
| `manifest.json` | add `"tts"` to `dependencies`; version `0.4.0` → `0.5.0` |
| `strings.json`, `translations/{en,de}.json` | `tts` subentry block, both languages |
| `requirements_test.txt` | `mutagen==1.48.1`, `ha-ffmpeg==3.2.2` |
| `README.md` | replace the "in progress" note with real setup instructions |

`requirements_test.txt` needs those two because Home Assistant's `tts` component imports `mutagen`
at module scope and depends on `ffmpeg` — the same class of test-harness dependency the suite
already carries for `conversation` and `camera`.

## 5. Declaring languages

Home Assistant validates the requested language by **exact string membership**:

```python
if language not in engine_instance.supported_languages:
    raise HomeAssistantError(f"Language '{language}' not supported")
```

There is no region-tag fallback, so an entity declaring `["de"]` fails for a pipeline configured as
`de-DE`. Each claimed language is therefore declared as the bare tag **and** its common regional
variants.

Because the catalog exposes no language data (§2.4), coverage is derived from the model name:

| Pattern matched against the model id | Languages claimed |
|---|---|
| `german` / `deutsch` | `de`, `de-DE`, `de-AT`, `de-CH` |
| `cosyvoice` | `de`, `de-DE`, `de-AT`, `de-CH`, `en`, `en-US`, `en-GB`, `en-AU`, `en-CA` |
| anything else | the `cosyvoice` set (conservative default) |

The first matching pattern wins, so a hypothetical `cosyvoice-german` would be treated as German
only — the safer reading of an ambiguous name.

Only languages verified by round trip are claimed. Declaring more would not produce an error — it
would produce confident, wrong pronunciation, which is worse, as Kokoro's English demonstrates.

`default_language` is exactly `"de"`, matching the household this is built for and the one language
both models handle well. It is present in every claimed set above, which it must be: Home Assistant
falls back to `default_language` when a request names none, and then applies the same exact-match
check.

## 6. Data flow

```
HA tts component ── async_get_tts_audio(message, language, options)
  │
  ├─ client.with_options(timeout=TTS_TIMEOUT).audio.speech.create(
  │      model=self.model,
  │      input=message,
  │      voice=TTS_VOICE,           # required by the gateway, value ignored (§2.3)
  │      response_format="wav",     # also ignored; sent for correctness
  │  )
  │
  ├─ audio = response.content       # SDK returns HttpxBinaryResponseContent, not a model
  ├─ guard: empty payload → HomeAssistantError
  └─ return ("wav", audio)
```

`TTS_TIMEOUT = 30.0`. Observed latency is one to two seconds; the bound exists so a stalled gateway
fails promptly rather than hanging a voice satellite.

`speed` is **not** exposed. It works, but nobody has asked for it, its valid range is unprobed, and
it is purely additive later. Same for streaming synthesis: Home Assistant supports
`async_stream_tts_audio`, but the gateway returns a complete WAV and its advertised `streaming:
true` is part of the bogus chat template rather than a real capability.

## 7. Error handling

Home Assistant turns a `(None, None)` return into `HomeAssistantError(f"No TTS from {name} for
'{message}'")`. Two consequences shape this design: the entity raises its **own**
`HomeAssistantError` with a specific cause rather than returning `(None, None)` and inheriting that
generic text, and — since HA's own fallback embeds the spoken text in an error string — our
messages deliberately do not.

| Condition | Behaviour |
|---|---|
| `AuthenticationError`, `PermissionDeniedError` | log; `entry.async_start_reauth(hass)`; raise `HomeAssistantError` |
| `APIStatusError` with 404 | the configured model cannot synthesise; log naming the model; raise |
| `APITimeoutError` | log with the timeout value; raise |
| any other `OpenAIError` | log; raise |
| empty audio body | raise — never hand Home Assistant a zero-byte success |

The `except` clauses are ordered auth → `APIStatusError` → `APITimeoutError` → `OpenAIError`, and
carry the comment explaining why: in the SDK `AuthenticationError` and `PermissionDeniedError`
subclass `APIStatusError`, which subclasses `OpenAIError`, so reordering silently breaks the reauth
path with no test failure that names the cause.

**Privacy.** The message being synthesised is user content. It is never logged at any level — only
its length — matching how the STT entity logs a byte count rather than the transcript. The API key
is never logged.

## 8. Configuration and UX

`TtsFlowHandler` mirrors `SttFlowHandler`: one required `CONF_MODEL` field, a dropdown populated
live from the gateway and filtered by `_is_tts_model`, with `entry_not_loaded`, `cannot_connect`,
`unknown` and `no_speech_models` aborts, plus a reconfigure step that pre-fills the current model.

The subentry is titled **"noris AI Text-to-speech"**, consistent with the naming introduced in
0.4.0, which makes the device card readable and yields `tts.noris_ai_text_to_speech` as the entity
id.

No new migration version is needed: no `tts` subentry can exist before this release, so there is
nothing to retitle. The `tts` type is nevertheless added to the 1.1 → 1.2 migration's title table,
purely so the table describes every subentry type rather than three of four — a defensive entry
that will never fire.

No voice picker is offered. The gateway ignores the `voice` value, so a picker would be a lie.

## 9. Testing

Mirrors `tests/test_stt.py`:

- entity properties, including that declared languages contain both bare and regional tags
- happy path: asserts the outgoing `model`, `input` and `voice`, and that the result is `("wav", …)`
- the language-derivation table: Kokoro → German only; CosyVoice → German and English; an
  unrecognised model → the default
- empty body, `OpenAIError`, 404 naming the model, timeout, and auth failure starting reauth
- flow tests: the TTS dropdown offers only speech models; speech models do not leak into the chat,
  AI Task or STT pickers; `no_speech_models` aborts when the gateway offers none
- the mixin refactor: existing conversation, AI-task and STT tests must pass unchanged, which is
  what proves the extraction preserved behaviour

Every new test is mutation-checked — mutate the production line the test names, confirm it goes
red, restore. Four tests in the STT work were caught by exactly this as unable to fail.

## 10. Rollout and verification

1. Back up `/config/custom_components/noris_ai/` **outside** `custom_components/` — a backup
   directory inside it is loaded as an integration and breaks the real one.
2. Deploy, `ha core check`, then `ha core restart` (a new platform is not picked up by a reload),
   confirmed with the user first.
3. Add the Text-to-speech subentry and verify `tts.noris_ai_text_to_speech` exists.
4. **Round-trip verification**: synthesise a German sentence through the new entity, resample to
   16 kHz, transcribe through Voxtral, and assert the text matches. This is a far stronger check
   than confirming bytes were returned, and it is the technique that proved the audio genuine
   during design.

The `HAL` pipeline currently uses `tts.piper`. Switching it to the noris engine changes live voice
behaviour and is **the user's decision**, not part of rollout.

## 11. Risks and follow-ups

- **Two name-based heuristics now**, one for detecting speech models and one for their languages.
  Both carry delete-me comments pointing at the catalog fix. This debt is acknowledged, not hidden.
- **An unrecognised future model** gets the conservative default language set, which may over-claim.
- **`speed` and streaming** are deliberately unimplemented.
- **The gateway's own TTS output cannot be fed to its own STT** without resampling: Voxtral rejects
  the 24 kHz WAV that the speech models produce, with `400 "Invalid or unsupported audio file"`.
  This does not affect the integration — Home Assistant delivers 16 kHz to the STT entity — but it
  is worth reporting to noris.
- **`vibevoice-asr/vibevoice`** appeared in the same catalog update and is unprobed. It may be an
  alternative or complement to Voxtral for transcription.

## 12. Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Scope | `async_get_tts_audio` only | YAGNI; `speed` and streaming are additive later |
| Language declaration | Derived from the model name, bare + regional tags | Catalog exposes none; HA matches by exact string |
| Voice options | None advertised | The gateway ignores the value it insists on receiving |
| Failure contract | Raise `HomeAssistantError` | Specific cause beats HA's generic fallback, which also embeds the spoken text |
| Device-info duplication | Extract a mixin now | A third copy is the wrong answer; deferred once already |
