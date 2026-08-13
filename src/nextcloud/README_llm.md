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
never touched by this job.

## How it works

1. Scans the same Nextcloud folder tree as the whisper job (recursively,
   skipping into `transcriptions/`/`llm/`/`prompts/` subfolders — nothing to
   find there) for `<stem>.yaml` or `<stem>.yml` files sitting next to a
   matching audio/video file.
2. For each sidecar found, checks whether `transcriptions/<stem>.txt`
   already exists.
   - **Not yet transcribed** → skipped silently, retried automatically on
     this job's next scheduled run. No action needed from you.
   - **Already transcribed** → the sidecar is parsed and the job proceeds.
3. Renders the chosen prompt template with the transcript text and sends it
   to the configured LLM.
4. Uploads the result to `llm/<stem>_<prompt>.md` (creating the `llm/`
   folder if needed).
5. **Skips work already done**: if `llm/<stem>_<prompt>.md` already exists,
   the job is not rerun. See "Reprocessing" below.

## Sidecar YAML format

`<stem>.yaml`, next to `<stem>.mp3` (or whatever the source extension is):

```yaml
llm: glm-4-7-flash   # optional — defaults to LLM_DEFAULT_MODEL env (glm-4-7-flash)
prompt: summary        # optional — name of a file in prompts/, defaults to "summary"
```

Both fields are optional — an **empty file** is a valid sidecar and means
"process this with all defaults." The mere presence of a sidecar file is
what opts a file into LLM processing at all.

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

## Prompts folder

Prompt templates are looked up **in Nextcloud first**, at
`<NEXTCLOUD_FOLDER>/prompts/<name>.md` (a top-level `prompts/` folder,
sibling to the audio subfolders — excluded from the sidecar scan like
`transcriptions/`/`llm/`). Each file's stem is a valid value for a
sidecar's `prompt:` field. A template is plain text with a single
`{transcript}` placeholder, substituted verbatim (not Python `.format()` —
so arbitrary `{`/`}` elsewhere in the template are safe) with the
transcript's plain text before being sent as the LLM's user message.

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
Nextcloud folder/            (NEXTCLOUD_FOLDER root)
├── prompts/                 # you manage this — one .md per prompt name
│   └── summary.md
├── interview_01.mp3
├── interview_01.yaml           # sidecar, uploaded alongside the audio
├── transcriptions/
│   ├── interview_01.txt        # written by the whisper job
│   └── interview_01.srt
└── llm/
    └── interview_01_summary.md # written by this job
```

## Reprocessing

The skip check is based on whether `llm/<stem>_<prompt>.md` already exists.
Since the output filename encodes the prompt name:

- **Changing `prompt:`** in the sidecar naturally reprocesses the file —
  the new prompt name produces a different output filename.
- **Changing only `llm:`** (keeping the same `prompt:`) does **not**
  reprocess — the output filename is unchanged, so the existing file is
  left as-is. To force a rerun with a different model, delete the existing
  `llm/<stem>_<prompt>.md` (or change the prompt name) and let the job pick
  it up on its next run.

This is an intentional first-pass limitation, not a bug — building
per-model output tracking wasn't needed for the current use case.

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
