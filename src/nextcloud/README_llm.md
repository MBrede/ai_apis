# Nextcloud Transcript LLM Post-Processing

A second, independent CronJob (`sync_llm.py`) that runs a hosted KubeAI LLM
over transcripts the whisper job (`sync.py`, see `README_cron.md`) has
already produced. It never transcribes, never calls Whisper, never touches
audio — it only reads `transcriptions/<stem>.txt` and writes to a new
sibling `llm/` folder. The whisper job and its `transcriptions/` output are
completely unaffected.

Opt-in is per file, via a sidecar YAML uploaded **next to the audio file**
(same folder, same stem) — upload the audio and its sidecar together, no
need to wait for transcription to finish first. Files without a sidecar are
never touched by this job. The same sidecar also drives the whisper job's
optional multi-model STT selection (`stt:` field) — see `README_cron.md`.

## How it works

1. Scans the same Nextcloud folder tree(s) as the whisper job (recursively,
   skipping into `transcriptions/`/`llm/`/`prompts/` subfolders — nothing to
   find there) for `<stem>.yaml` or `<stem>.yml` files sitting next to a
   matching audio/video file.
2. For each sidecar found, derives the transcript(s) it expects from the
   sidecar's own `stt:` field — `transcriptions/<stem>.txt` if `stt:` is
   absent, or one `transcriptions/<stem>_<sttmodel>.txt` per listed STT
   model — and checks which of those actually exist yet.
   - **None ready yet** → skipped silently, retried automatically on this
     job's next scheduled run. No action needed from you.
   - **Some ready, some not** → the ready ones are processed now; the rest
     retry on a later run once the whisper job catches up.
   - Each ready transcript is processed **independently** — a sidecar with
     `stt: [turbo, qwen3-asr-1.7b]` runs two full batches (below), one per
     STT model's transcript.
3. For each ready transcript, expands the sidecar's `llm:` and `prompt:`
   fields (each a single name or a list) into the **cartesian product** —
   every listed prompt runs against every listed model. 2 models × 2 prompts
   = 4 jobs per transcript.
4. Renders the chosen prompt template with the transcript text (and the
   sidecar's optional `context:` text, via a `{context}` placeholder) and
   sends it to the corresponding LLM.
5. Uploads the result to `llm/<transcript_stem>_<model>_<prompt>.md`
   (creating the `llm/` folder if needed) — `<transcript_stem>` is
   `<stem>` or `<stem>_<sttmodel>`, so the STT model (if any) is folded into
   the output filename automatically, no separate naming step needed.
6. **Skips work already done**: if that exact filename already exists, that
   combination is not rerun. See "Reprocessing" below.
7. **Jobs run grouped by model**, not crawl-discovery order: every job for
   one model runs before moving to the next model, across the *entire* run
   (all folders, all sidecars, all transcript variants) — not just within
   one file. This keeps consecutive calls to the same KubeAI-hosted model
   inside its `scaleDownDelaySeconds` warm window, avoiding a repeated cold
   start every time the job round-robins between models.

## Sidecar YAML format

`<stem>.yaml`, next to `<stem>.mp3` (or whatever the source extension is):

```yaml
stt: turbo                  # optional — STT model(s) for the WHISPER job (sync.py); str or list; see README_cron.md
llm: glm-4-7-flash          # optional — LLM model(s) for THIS job; str or list; defaults to LLM_DEFAULT_MODEL env
prompt: summary             # optional — prompt name(s), a file in prompts/; str or list; defaults to "summary"
context: ""                 # optional — free text injected into prompt templates via a {context} placeholder
```

All fields are optional — an **empty file** is a valid sidecar and means
"process this with all defaults." The mere presence of a sidecar file is
what opts a file into LLM processing at all (independent of whether it also
carries an `stt:` field — a sidecar with only `stt:` still triggers this
job with default `llm`/`prompt`).

### Worked example — lists for both STT and LLM+prompt

```yaml
stt: [turbo, qwen3-asr-1.7b]
llm: [glm-4-7-flash, qwen3-14b]
prompt: [summary, action-items]
context: "Weekly Benkana project sync — focus on blockers."
```

Produces, for a source file `interview_01.mp3`:

```
transcriptions/interview_01_turbo.txt
transcriptions/interview_01_turbo.srt
transcriptions/interview_01_qwen3-asr-1.7b.txt
transcriptions/interview_01_qwen3-asr-1.7b.srt
llm/interview_01_turbo_glm-4-7-flash_summary.md
llm/interview_01_turbo_glm-4-7-flash_action-items.md
llm/interview_01_turbo_qwen3-14b_summary.md
llm/interview_01_turbo_qwen3-14b_action-items.md
llm/interview_01_qwen3-asr-1.7b_glm-4-7-flash_summary.md
llm/interview_01_qwen3-asr-1.7b_glm-4-7-flash_action-items.md
llm/interview_01_qwen3-asr-1.7b_qwen3-14b_summary.md
llm/interview_01_qwen3-asr-1.7b_qwen3-14b_action-items.md
```
2 STT models × (2 LLM models × 2 prompts = 4 LLM outputs each) = 8 LLM
outputs total — every transcript and every LLM output independently
skippable/rerunnable, and the `context` text is available to all 8 LLM
calls via `{context}`. This lets you directly compare how STT choice
affects downstream summary quality, not just how LLM/prompt choice does.

## Available LLMs

Any model currently `enabled: true` in `llm/models-values.yaml` (KubeAI's
model catalog). As of this writing:

| Model | Notes |
|---|---|
| `glm-4-7-flash` | **Default.** Fast MoE model, good for straightforward summaries. |
| `qwen3-32b` | Largest general chat model — best quality, still one A40. |
| `qwen3-14b` | Mid-large, slower than 8b/flash but stronger reasoning. |
| `qwen3-8b` | Mid-size, good balance of speed/quality. |
| `qwen3-4b` | Small, fast. |
| `qwen3-1b7` | Very small, lowest latency, weakest quality. |
| `qwen3-0b6` | Smallest available. |
| `qwen3-30b-moe` | MoE, fits on one A40 at reduced GPU utilization. |
| `qwen3-6-35b-a3b` | MoE, fits on one A40 at reduced GPU utilization. |

This list can drift — check `llm/models-values.yaml`'s `catalog:` section
for the live set of enabled models before relying on this table.

Models scale to zero when idle (`minReplicas: 0`). Unlike the Whisper
service (which needs a KEDA warm-up step to avoid 502s), KubeAI's own proxy
queues requests during scale-up natively — no warm-up is needed here. A warm
cold-start (weights already on KubeAI's cache PVC) takes ~30–60s, but a true
cold pull (first-ever invocation of that model, weights not cached yet) can
take **up to ~12 minutes** — confirmed in testing: the first `glm-4-7-flash`
call timed out at the old 300s default. `LLM_TIMEOUT` (default 900s) is set
well above that.

## Available STT models

Selected via a sidecar's `stt:` field (see whisper_api.py's `backend`
selection, `_backend_for_stt_model` in `sync.py`):

| Model | Backend | Notes |
|---|---|---|
| `turbo` (and other Whisper size names: `tiny`, `base`, `small`, `medium`, `large-v3`, ...) | `whisperx` | **Default** when `stt:` is omitted entirely — unsuffixed output, today's original behavior. Batched WhisperX transcription + wav2vec2 forced alignment + pyannote diarization. |
| `qwen3-asr-1.7b` (any name starting with `qwen3-asr`) | `qwen3-asr` | Qwen3-ASR-1.7B + Qwen3-ForcedAligner-0.6B (separate alignment pass — Qwen3-ASR itself has no word timestamps) + the same pyannote diarization pipeline. **Caveats**: aligner supports ~5 minutes of audio per call (`sync.py`/`whisper_api.py` chunk longer files automatically); `include_fillers` has no effect with this backend (Whisper-specific prompt-injection mechanism, silently ignored with a warning); observed to sometimes stop generating partway through short clips (model-quality limitation, not a chunking bug — see git history for a concrete example). |
| `ark-asr-3b` (any name starting with `ark-asr`) | `ark-asr` | [Audio8/ARK-ASR-3B](https://huggingface.co/Audio8/ARK-ASR-3B), Whisper-encoder + Qwen decoder. Supports German. **Caveats**: no word-level timestamps — diarizes the audio *first* (pyannote), then transcribes each speaker turn separately (`_diarize_then_transcribe` in `whisper_api.py`), so speaker boundaries come from pyannote's turns rather than word-level alignment; this also sidesteps the model's documented 30s-per-call limit. |
| `hojo-asr-v1` (any name starting with `hojo-asr`) | `hojo-asr` | [HojoAI/Hojo-ASR-V1](https://huggingface.co/HojoAI/Hojo-ASR-V1). **Does not support German** — languages are Mandarin, English, Cantonese, Sichuan dialect only; only useful for non-German/English audio in this deployment. Same diarize-first approach as `ark-asr` (no word-level timestamps). |
| `granite-speech-4.1-2b-plus` (any name starting with `granite-speech`) | `granite-speech` | [ibm-granite/granite-speech-4.1-2b-plus](https://huggingface.co/ibm-granite/granite-speech-4.1-2b-plus). Supports German (English, French, German, Spanish, Portuguese). Has real word-level timestamps (via a dedicated `[T:N]` centisecond-tag prompt) — uses the same transcribe-then-align-then-diarize path as `qwen3-asr` (`GraniteSpeechBuffer`), not the diarize-first approach. **Caveats**: chunked at 3.5 min/call (the model card's own tested limit for the timestamp task); the tag format only carries word *end* times, so per-word `start` is approximated from the previous word's `end`, coarser than WhisperX/Qwen3-ASR's true per-word spans. |
| `nemotron-3.5-asr-streaming-0.6b` (any name starting with `nemotron`) | `nemotron-asr` | [nvidia/nemotron-3.5-asr-streaming-0.6b](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b), FastConformer-RNNT. No word-level timestamps documented — same diarize-first approach as `ark-asr`/`hojo-asr`. Run via `transformers.pipeline("automatic-speech-recognition", ...)` in plain batch mode per diarize turn (its cache-aware streaming mode isn't used here). |

**Dependency note**: `qwen-asr` and `hojo-asr` (PyPI packages) pin
`transformers==4.57.6` and `torch<2.6` respectively — both conflict with
newer floors required elsewhere in this container (`transformers>=5.8` for
Granite-Speech/ARK-ASR, `torch>=2.6.0` project-wide). Both are installed
with `--no-deps` in `Dockerfile.whisper`/`Dockerfile.whisper.hub` so they
don't downgrade shared packages; see those files' comments.

A file with no sidecar `stt:` field is unaffected by any of this — it keeps
transcribing with the `WHISPER_MODEL` env default (`whisperx` backend),
unsuffixed `<stem>.{txt,srt}` output, exactly as before this feature existed.

## Prompts folder

Prompt templates are looked up **in Nextcloud first**, at
`<remote_root>/prompts/<name>.md` — with multiple `NEXTCLOUD_FOLDER` entries
(see `README_cron.md`), each folder's *own* `prompts/` subfolder is checked
independently, using whichever folder that job's sidecar came from (falling
back to the bundled default per-folder, same as single-folder deployments).
Each file's stem is a valid value for a sidecar's `prompt:` field. A
template is plain text with `{transcript}` and (optionally) `{context}`
placeholders, both substituted verbatim (not Python `.format()` — so
arbitrary `{`/`}` elsewhere in the template are safe): `{transcript}` with
the transcript's plain text, `{context}` with the sidecar's `context:`
field (empty string if the sidecar omits it — a template that doesn't
reference `{context}` at all is completely unaffected by this feature).

**To add or edit a prompt: just upload/edit the `.md` file in Nextcloud's
`prompts/` folder.** No image rebuild, no redeploy — the job re-reads it
(with a per-run cache) on every scheduled run.

If a prompt named in a sidecar isn't found in Nextcloud's `prompts/`
folder, the job falls back to the copy bundled in the image
(`ai_apis/src/nextcloud/prompts/*.md`) so a fresh deployment still has a
working default without needing anything uploaded first. Only `summary` is
bundled today.

| Prompt | Purpose |
|---|---|
| `summary` | **Default.** Plain-language summary: topics, decisions, action items, per-speaker points. |

## Output

```
Nextcloud folder/            (one entry of NEXTCLOUD_FOLDER's comma-separated list)
├── prompts/                 # you manage this — one .md per prompt name
│   └── summary.md
├── interview_01.mp3
├── interview_01.yaml            # sidecar, uploaded alongside the audio
├── transcriptions/
│   ├── interview_01.txt         # written by the whisper job (no stt: in sidecar → unsuffixed)
│   └── interview_01.srt
└── llm/
    └── interview_01_glm-4-7-flash_summary.md   # written by this job
```

See the worked example above for the full multi-model/multi-prompt output
shape.

## Reprocessing

The skip check is based on whether `llm/<transcript_stem>_<model>_<prompt>.md`
already exists — the output filename encodes the STT model (if any, via
`<transcript_stem>`), the LLM model, and the prompt, so skip-tracking is per
exact (transcript variant, model, prompt) combination:

- **Changing `prompt:`** produces a different output filename → reprocesses.
- **Changing `llm:`** produces a different output filename too → reprocesses.
  (This used to be a documented limitation — changing only `llm:` silently
  did nothing, since the old filename only encoded the prompt. Fixed
  2026-08-19 by adding the model into the filename.)
- **Changing `stt:`** (adding a new STT model to the list) produces a whole
  new transcript variant once the whisper job catches up, which this job
  then processes as its own independent llm × prompt batch — existing
  outputs for other STT variants are untouched. (Fixed 2026-08-20 — this job
  previously only ever read the bare unsuffixed transcript, ignoring
  STT-model-suffixed variants entirely.)

**One-time migration cost**: every LLM output produced *before* the
2026-08-19 change used the old `<stem>_<prompt>.md` naming (no model in the
filename). On the first run after upgrading, every existing sidecar-driven
file looks "unprocessed" under the new naming scheme and gets reprocessed
once — this is deliberate, not a bug, and is the direct consequence of
making per-model tracking correct. Old `<stem>_<prompt>.md` files are never
deleted automatically; clean them up manually if desired.

## Multi-folder scanning

`NEXTCLOUD_FOLDER` accepts a comma-separated list of folders — each is
scanned independently (see `README_cron.md` for the full details, shared
with the whisper job). Example: `NEXTCLOUD_FOLDER=/Shared/transcription,
/Shared/other-project/transcription` scans both trees in one run.

## Environment variables

In addition to the `NEXTCLOUD_*` variables shared with the whisper job (see
`README_cron.md`):

```env
# KubeAI OpenAI-compatible endpoint (in-cluster, no auth needed —
# same pattern Open WebUI uses; KubeAI doesn't check the key internally)
LLM_URL=http://kubeai.llm.svc.cluster.local/openai/v1

# Default model when a sidecar omits `llm:`
LLM_DEFAULT_MODEL=glm-4-7-flash

# Request timeout in seconds (generous to absorb cold-start queueing)
LLM_TIMEOUT=900
```

## Running

Same mechanics as the whisper job — see `README_cron.md` for Docker Compose
/ manual / Kubernetes CronJob instructions. This job runs as
`python -m src.nextcloud.sync_llm` (vs. `sync` for the whisper job), in the
same `ai-apis-nextcloud` image, on its own CronJob and schedule
(`nextcloud-llm-sync`), independent of the whisper job's schedule.

## Roadmap

This is the base plumbing for LLM post-processing. A later iteration will
add automated **label extraction** via prompt optimization (prompt-opt) —
a new prompt (or prompts) plus a still-to-be-defined structured-output
convention for the extracted labels. Nothing about that is implemented yet;
the Nextcloud `prompts/` folder and sidecar `prompt:` field exist
specifically so that work can be added by uploading a new prompt file, with
no code change to `sync_llm.py`.
