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
ground_truth: ""            # anything else you add becomes a {field_name} placeholder too — see "Judges"
```

All fields are optional — an **empty file** is a valid sidecar and means
"process this with all defaults." The mere presence of a sidecar file is
what opts a file into LLM processing at all (independent of whether it also
carries an `stt:` field — a sidecar with only `stt:` still triggers this
job with default `llm`/`prompt`). Any field beyond `stt`/`llm`/`prompt`/
`context` is not part of the fixed schema — it's an arbitrary extra field,
available as a `{field_name}` placeholder in prompt AND judge templates
(see "Arbitrary extra fields" under "Judges" below).

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

## Batch file (alternative to per-file YAML)

For a folder with many files that each need their own config, a single
`batch.csv` or `batch.xlsx` in that folder can replace hand-authoring one
`<stem>.yaml` per file — same fields, one **row per file** instead of one
file per row:

| filename | stt | llm | prompt | context |
|---|---|---|---|---|
| `interview_01.mp3` | `turbo, qwen3-asr-1.7b` | `glm-4-7-flash` | `summary, action-items` | `Weekly Benkana project sync — focus on blockers.` |
| `interview_02.mp3` | `turbo` | `glm-4-7-flash, qwen3-14b` | `summary` | `Kickoff call, no prior context.` |

- **`filename`** matches against the audio/video file's name in the same
  folder — with or without extension (`interview_01` and `interview_01.mp3`
  both match `interview_01.mp3`).
- `stt`/`llm`/`prompt` are comma-separated lists within a cell (same
  cartesian-product semantics as the YAML `stt:`/`llm:`/`prompt:` lists).
  `context` is free text, not split.
- Column headers are case-insensitive; `file` is also accepted for `filename`.
- Both `.csv` and `.xlsx` are supported. If both exist in the same folder,
  `batch.xlsx` wins and `batch.csv` is ignored.
- **A file's own `<stem>.yaml` always wins over a matching batch row** — the
  batch file only covers files that don't have their own sidecar. This lets
  you hand a customer's spreadsheet-driven folder an individual override for
  one file without touching the batch file.
- A file with neither a sidecar nor a batch row is untouched by this
  section — plain default transcription only (see "How it works" above).
- Any OTHER column beyond `filename`/`stt`/`llm`/`prompt`/`context` (e.g. a
  `ground_truth` column) is passed through per-row too, available as a
  `{field_name}` placeholder in prompt AND judge templates — see "Arbitrary
  extra fields" under "Judges" below.

A batch-covered folder also gets one reviewable `llm/results_table.xlsx`,
rebuilt fresh whenever a job uploads something new: **one sheet per LLM
model**, columns are prompts, rows are transcript stems (STT-model suffix
included when `stt:` was a list), cells are that (stem, model, prompt)
combination's output text (blank if not processed yet) — so you can eyeball
one model's output across all its prompts on a single sheet, then flip
sheets to compare another model. Individual `.md` files under `llm/` still
exist unchanged; this is a convenience view on top of them, not a
replacement.

### Live example

`/Shared/transcription (3)/batch_example/` has a working demo: three
duplicated audio files (`demo_a.m4a`, `demo_b.m4a`, `demo_c.m4a`) and a
`batch.xlsx` with one row each — `demo_c`'s row deliberately omits the
extension in `filename` to show that both forms match. Produces:

```
transcriptions/demo_a_turbo.txt
llm/demo_a_turbo_glm-4-7-flash_summary.md

transcriptions/demo_b_turbo.txt
transcriptions/demo_b_qwen3-asr-1.7b.txt
llm/demo_b_turbo_glm-4-7-flash_summary.md
llm/demo_b_turbo_glm-4-7-flash_rootcause.md
llm/demo_b_qwen3-asr-1.7b_glm-4-7-flash_summary.md
llm/demo_b_qwen3-asr-1.7b_glm-4-7-flash_rootcause.md

transcriptions/demo_c_turbo.txt
llm/demo_c_turbo_glm-4-7-flash_summary.md
llm/demo_c_turbo_qwen3-14b_summary.md
```

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

Selected via a sidecar's `stt:` field (see `_backend_for_stt_model` in
`sync.py`; buffer implementations in `src/audio/stt_backends/`, one file
per backend):

| Model | Backend | Status | Notes |
|---|---|---|---|
| `turbo` (and other Whisper size names: `tiny`, `base`, `small`, `medium`, `large-v3`, ...) | `whisperx` | ✅ verified | **Default** when `stt:` is omitted entirely — unsuffixed output, today's original behavior. Batched WhisperX transcription + wav2vec2 forced alignment + pyannote diarization. |
| `qwen3-asr-1.7b` (any name starting with `qwen3-asr`) | `qwen3-asr` | ✅ verified | Qwen3-ASR-1.7B + Qwen3-ForcedAligner-0.6B (separate alignment pass — Qwen3-ASR itself has no word timestamps) + the same pyannote diarization pipeline. **Caveats**: aligner supports ~5 minutes of audio per call (chunked automatically); `include_fillers` has no effect with this backend (Whisper-specific prompt-injection mechanism, silently ignored with a warning). **Fixed 2026-08-30**: transcripts stopping early was NOT a model-quality limitation as first assumed — `qwen_asr`'s transformers-backend `max_new_tokens` defaults to 512, far too small for a 5-minute chunk; now explicitly set to 8192. See README_cron.md's "Truncation warning" section for the general safety-net check this also motivated. |
| `ark-asr-3b` (any name starting with `ark-asr`) | `ark-asr` | ✅ verified 2026-08-21 | [Audio8/ARK-ASR-3B](https://huggingface.co/Audio8/ARK-ASR-3B), Whisper-encoder + Qwen decoder. Supports German. **Caveats**: no word-level timestamps — diarizes the audio *first* (pyannote), then transcribes each speaker turn separately (`_diarize_then_transcribe` in `stt_backends/base.py`), so speaker boundaries come from pyannote's turns rather than word-level alignment. **Corrected 2026-08-30**: pyannote turns are NOT naturally kept under the model's documented 30s-per-call limit as previously assumed — an uninterrupted monologue turn can run much longer; `_diarize_then_transcribe` now sub-chunks any turn exceeding `MAX_TURN_SECONDS` (25s) before transcribing. `max_new_tokens` also quadrupled (256→1024) as extra headroom, same bug class as the `qwen3-asr` fix above. |
| `hojo-asr-v1` (any name starting with `hojo-asr`) | `hojo-asr` | ❌ broken 2026-08-24 | [HojoAI/Hojo-ASR-V1](https://huggingface.co/HojoAI/Hojo-ASR-V1). **Does not support German** — languages are Mandarin, English, Cantonese, Sichuan dialect only. Same diarize-first approach as `ark-asr` (no word-level timestamps). Runs in its own container (`ai-apis-whisper-hojoasr`) since hojo-asr needs `torch<2.6`. **Currently broken**: that torch pin forces an old pyannote-audio that can't load this deployment's diarization model (needs pyannote-audio 4.x). Not fixable without an upstream hojo-asr release relaxing its torch pin; not prioritized further given no German support either. See `hojo_asr.py`'s docstring for the full chain. |
| `granite-speech-4.1-2b-plus` (any name starting with `granite-speech`) | `granite-speech` | ✅ verified 2026-08-21 | [ibm-granite/granite-speech-4.1-2b-plus](https://huggingface.co/ibm-granite/granite-speech-4.1-2b-plus). Supports German (English, French, German, Spanish, Portuguese). Has real word-level timestamps (via a dedicated `[T:N]` centisecond-tag prompt) — uses the same transcribe-then-align-then-diarize path as `qwen3-asr` (`GraniteSpeechBuffer`), not the diarize-first approach. **Caveats**: chunked at 3.5 min/call (the model card's own tested limit for the timestamp task); the tag format only carries word *end* times, so per-word `start` is approximated from the previous word's `end`, coarser than WhisperX/Qwen3-ASR's true per-word spans. |
| `nemotron-3.5-asr-streaming-0.6b` (any name starting with `nemotron`) | `nemotron-asr` | ✅ verified 2026-08-21 | [nvidia/nemotron-3.5-asr-streaming-0.6b](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b), FastConformer-RNNT. No word-level timestamps documented — same diarize-first approach as `ark-asr`/`hojo-asr`. Run via `transformers.pipeline("automatic-speech-recognition", ...)` in plain batch mode per diarize turn (its cache-aware streaming mode isn't used here). |

**Architecture note (2026-08-24)**: `qwen3-asr` and `hojo-asr` run in their
own separate containers (`ai-apis-whisper-qwen3asr`, `ai-apis-whisper-hojoasr`),
not the primary `ai-apis-whisper` container — `qwen-asr` pins
`transformers==4.57.6` and `hojo-asr` pins `torch<2.6`, both genuinely
incompatible with `transformers>=5.8` (needed by Granite-Speech/ARK-ASR) and
`torch>=2.6.0` (`ml-base`, shared project-wide) respectively — not just
strict metadata pins, their actual code proved incompatible with the newer
versions too (see `qwen3_asr.py`'s and `hojo_asr.py`'s docstrings). The
primary whisper container transparently proxies `backend=qwen3-asr`/
`backend=hojo-asr` requests to the right container over HTTP — see
`whisper_api.py`'s module docstring and `LOCAL_BACKENDS`/`PROXY_URLS`. All
three containers scale to zero via KEDA when idle (`keda.enabled`), so this
split doesn't cost a permanently-reserved GPU slot per backend.

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

## Judges (LLM-as-judge)

Automated scored review on top of the batch-file result tables above — a
`judges/<name>.yaml` (sibling to `prompts/`, one file per top-level
`NEXTCLOUD_FOLDER` entry, not per nested subfolder) scores every existing
output cell of the prompt(s) it names as `jurisdiction:`, producing
`llm/<judge>_scores.xlsx`: **one workbook per judge**, covering every
jurisdiction prompt at once — one sheet per prompt (score matrix + bar
chart) plus an "Overview" sheet comparing average score per prompt, so you
can see how a judge rates `summary` vs. `rootcause` outputs side by side in
one file instead of hunting across several.

**Scope**: judging only applies to **batch-file-driven folders** (folders
with a `batch.csv`/`batch.xlsx`) — same limitation the `results_table.xlsx`
review table above already has, since that's the only place a cell's
existence is already tracked. Plain per-file-sidecar folders get no judge
scores (see Roadmap).

**Only new/unscored cells are ever judged** — a (transcript, model) pair
that already has a score in the workbook is never re-scored, so a judge
doesn't re-spend LLM calls on unchanged data every cron tick (same
discipline the review tables use for their own rebuild gate).

### Judge YAML schema

```yaml
kind: label                    # required: "label" | "logprob"
llm: qwen3-14b                 # required: KubeAI model used as judge
jurisdiction: [summary, action-items]   # required: prompt name(s) to score; scalar or list
scale: [poor, fair, good, excellent]    # required: numeric range/list, OR ascending-quality label list
anchors:                       # optional: scale-value -> description
  poor: "Summary misses major topics or contains factual errors."
  excellent: "Summary is accurate, well-organized, captures every decision."
anti_pattern: "Do not reward summaries that are merely long."   # optional, str or list
few_shot:                      # optional: list of {response, score, rationale}
  - response: "The team discussed the Q3 roadmap and agreed to ship feature X by Friday."
    score: good
    rationale: "Accurate, concise, captures the one decision made."
prompt: |                      # required: judge's own instruction/rubric, self-contained
  You are evaluating the quality of an LLM-generated meeting summary.
  Judge how well the response captures the substance of the conversation.
```

| Field | Type | Required | Notes |
|---|---|---|---|
| `kind` | `label` \| `logprob` | yes | See "label vs. logprob" below. |
| `llm` | string | yes | Any model from "Available LLMs" above. |
| `jurisdiction` | string or list | yes | Prompt name(s) this judge scores. |
| `scale` | string, list, or list of labels | yes | `"1-5"`/`[1,5]` = inclusive numeric range; `[1,3,5]` = explicit discrete numeric set; `["poor","fair","good"]` = labels, **must be given in ascending-quality order** (needed for `kind: logprob`'s scoring). |
| `anchors` | dict or list | no | dict = explicit `{scale_value: description}`, any sparse subset. List = plain description strings, auto-distributed evenly across the scale (endpoints included) — you never have to pick exact positions on a fine-grained scale (e.g. 1-100); 2-4 anchors (worst/mid/best) is the common case regardless of scale granularity. |
| `anti_pattern` | string or list | no | What NOT to reward. |
| `few_shot` | list of `{response, score, rationale}` | no | Worked examples, both good and bad. |
| `prompt` | string | yes | Self-contained rubric text — not a reference to a `prompts/*.md` file. |

`{response}` (the output text being judged) is always available in `prompt:`
via a placeholder; `{transcript}`/`{context}` are also available if
referenced, same plain-substitution mechanism as prompt templates above.

### Arbitrary extra fields (e.g. a ground-truth column)

The sidecar/batch-row schema isn't limited to `stt`/`llm`/`prompt`/`context`
— any OTHER column in `batch.xlsx` (or extra top-level key in a `<stem>.yaml`
sidecar) is passed through automatically and available as a `{field_name}`
placeholder, in **both** regular prompt templates and judge templates. This
is what makes real reference-based judging possible — add a `ground_truth`
column to your batch file with a hand-written reference answer per file,
then reference `{ground_truth}` in a judge's `prompt:` text to have the
judge actually compare the model's output against it, instead of only doing
a reference-free quality check against the transcript.

```
| filename      | llm           | prompt  | ground_truth                                    |
|---------------|---------------|---------|--------------------------------------------------|
| interview_01  | glm-4-7-flash | summary | Root cause: unfamiliar with RFID terminology...   |
```

```yaml
# judges/accuracy-vs-reference.yaml
kind: label
llm: qwen3-32b
jurisdiction: [summary]
scale: [poor, fair, good, excellent]
prompt: |
  Compare the response below against the ground truth reference. Judge how
  well it captures the same key information.

  Ground truth: {ground_truth}
```

No code change needed for a new field name — it's picked up automatically
from whatever's actually in the sheet/sidecar.

### Worked example

The `tone-quality.yaml` above assembles into this judge prompt:

```
You are evaluating the quality of an LLM-generated meeting summary.
Judge how well the response captures the substance of the conversation.

Scale: choose one of: poor, fair, good, excellent.

Anchors:
- poor: Summary misses major topics or contains factual errors.
- excellent: Summary is accurate, well-organized, captures every decision.

Examples:
Response: The team discussed the Q3 roadmap and agreed to ship feature X by Friday.
Score: good
Rationale: Accurate, concise, captures the one decision made.

Do NOT reward:
- Summaries that are merely long or that repeat the transcript verbatim.

Response to evaluate:
{response}

Respond with ONLY the label above, nothing else. No explanation, no punctuation.
```

For each `summary` output that exists in a batch folder's `llm/`, this
produces `llm/tone-quality_scores.xlsx` with a `summary` sheet (columns =
models, rows = transcript stems, cells = the parsed score, plus an embedded
bar chart) — and, if `jurisdiction:` also named `rootcause`, a second
`rootcause` sheet right alongside it in the SAME file, plus an "Overview"
sheet averaging each model's score per prompt so the two are directly
comparable.

### `label` vs. `logprob`

| | `kind: label` | `kind: logprob` |
|---|---|---|
| Mechanism | Judge LLM states its score/label directly; parsed from plain text. | G-Eval style: score derived from probability-weighted token logprobs at the answer position. |
| Output | Discrete (exact scale value). | Continuous (weighted average — useful for fine-grained comparison even on a coarse scale). |
| Dependency | None — works with any model. | Requires the KubeAI/vLLM endpoint to return usable `logprobs`/`top_logprobs`. |
| Status | Always reliable. | **Verified live 2026-08-28 against `glm-4-7-flash`** (vLLM v0.28.0, `--reasoning-parser=glm45`) — a real request's `top_logprobs` at the located answer position produced a weighted score of ≈5.0 for a clearly-correct "5" answer, matching expectations. Not yet exercised against every possible judge `llm:` choice — if a different model's server doesn't return usable logprobs, or no token in the reply matches a scale value, the cell is logged and skipped (silently unscored, not a wrong score), so a bad fit degrades safely rather than silently. |

A reasoning-tuned judge model (e.g. `glm-4-7-flash`, any Qwen3 model) emits
`<think>...</think>` as a mandatory part of its generation — the answer
token is not necessarily the first one generated. Confirmed live: even with
a vLLM `--reasoning-parser` configured (so the message-level `content`/
`reasoning` fields come back cleanly split), the `logprobs` field stays ONE
flat raw token stream covering reasoning + answer together and still
contains a literal `</think>` token — `kind: logprob` locates the answer by
scanning past it. No `max_tokens` cap is set on judge calls, since a small
cap would just truncate mid-thought before the model ever reaches its
answer.

**`kind: label` reliability note (found live, 2026-08-28)**: on a large
production backlog, a `kind: label` judge using a small model
(`qwen3-0b6`) produced ~55 unparseable replies — its `<think>` block
sometimes never closes with `</think>` at all, and the naive
"exactly one label appears as a substring of the reply" fallback has no
way to recover from that the way `kind: logprob`'s targeted post-`</think>`
token scan does. All models in the KubeAI catalog (Qwen3 family + GLM) now
have a `--reasoning-parser` configured, which reduces how often this
happens, but doesn't eliminate the tiny-model verbosity issue. If a
`kind: label` judge produces a lot of skipped cells, prefer `kind: logprob`
or a larger/more instruction-following model.

## Output

```
Nextcloud folder/            (one entry of NEXTCLOUD_FOLDER's comma-separated list)
├── prompts/                 # you manage this — one .md per prompt name
│   └── summary.md
├── judges/                  # optional — one .yaml per judge (batch-file folders only)
│   └── tone-quality.yaml
├── interview_01.mp3
├── interview_01.yaml            # sidecar, uploaded alongside the audio
├── transcriptions/
│   ├── interview_01.txt         # written by the whisper job (no stt: in sidecar → unsuffixed)
│   └── interview_01.srt
└── llm/
    ├── interview_01_glm-4-7-flash_summary.md   # written by this job
    ├── results_table.xlsx                       # one sheet per LLM model, columns = prompts (batch folders only)
    └── tone-quality_scores.xlsx                 # written by the judge pass — one file per judge,
                                                  # one sheet per jurisdiction prompt (batch folders only)
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

**Review table migration (2026-09-02)**: the batch-folder review table moved
from one `<prompt>_table.xlsx` per prompt (columns = models) to a single
`results_table.xlsx` (one sheet per model, columns = prompts). Old
`<prompt>_table.xlsx` files are orphaned, not deleted automatically — clean
them up manually if desired.

**Judge output naming migration (2026-08-28)**: judge score files moved
from `<judge>_<prompt>_scores.xlsx` (one per judge+prompt pair) to
`<judge>_scores.xlsx` (one per judge, all jurisdiction prompts as separate
sheets). Old `<judge>_<prompt>_scores.xlsx` files are orphaned, not deleted
or migrated automatically — the next judge run creates the new
consolidated file fresh, re-scoring nothing that was already scored (the
skip-check reads the NEW file's sheets, not the old one). Clean up old
files manually if desired.

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

# Kill switch for the judge pass — set to "false" to disable judging
# cluster-wide without deleting every judges/*.yaml (e.g. if kind: logprob
# misbehaves in production before it's been verified). Default: enabled.
LLM_JUDGE_ENABLED=true

# Max LLM calls in flight at once in the main jobs loop. Found live
# 2026-09-02: a fully sequential loop took up to 4h to process one prompt
# across many files — each call is dominated by generation time, not I/O,
# and KubeAI/vLLM already serves concurrent requests per model.
LLM_CONCURRENCY=10
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

Judging (see "Judges" above) currently only covers batch-file-driven
folders. Extending it to plain per-file-sidecar folders is a possible future
extension — it would need its own "what output cells exist" enumeration,
since there's no `batch_rows` to drive discovery there.
