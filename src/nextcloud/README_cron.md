# Nextcloud Transcription Sync

Scans one or more Nextcloud folders (recursively) — every 30 minutes by
default (Kubernetes; changed 2026-08-19, was daily) — for new audio/video
files, transcribes them with speaker diarization via the local Whisper API,
and uploads `.txt` and `.srt` outputs to a `transcriptions/` subfolder next
to each source file. Already-transcribed files are skipped, so a run with
nothing new to do is cheap — the shorter interval isn't extra load.

By default every file is transcribed once with the `WHISPER_MODEL` env
default. A `<stem>.yaml`/`.yml` sidecar next to the audio file can opt a
file into **multiple STT models** via an `stt:` field (single model name or
a list) — one independent transcription pass per listed model, each with
its own `<stem>_<model>.{txt,srt}` output and skip tracking. This is the
same sidecar file the LLM post-processing job (`sync_llm.py`) uses for its
own `llm:`/`prompt:`/`context:` fields — see `README_llm.md` for the full
schema and a worked multi-model example. Files with no sidecar, or a
sidecar with no `stt:` field, are completely unaffected by this — they keep
today's exact unsuffixed behavior.

## Environment variables

Add these to your `.env` file:

```env
# Nextcloud connection
# NEXTCLOUD_URL must be the server base URL only — no /remote.php/... suffix.
# The sync script constructs the full WebDAV path itself from the username and folder.
NEXTCLOUD_URL=https://cloud.example.com
NEXTCLOUD_USER=myuser
NEXTCLOUD_PASSWORD=mypassword
NEXTCLOUD_FOLDER=path/to/folder   # relative to your Nextcloud home
# Comma-separated for multiple folders, each scanned independently:
# NEXTCLOUD_FOLDER=path/to/folder,/Shared/other-project/transcription

# Speaker count — three options in priority order:
# 1. Encode in the filename (see below) — overrides env vars per file
# 2. Exact global count:
NUM_SPEAKERS=2
# 3. Estimated range (used when NUM_SPEAKERS is unset):
# MIN_SPEAKERS=1
# MAX_SPEAKERS=4

# Optional overrides (defaults shown)
# WHISPER_MODEL=turbo
# WHISPER_TIMEOUT=3600
```

`WHISPER_URL` and `WHISPER_API_KEY` are set automatically by docker-compose
to point at the `whisper` service using the shared `API_KEY`.

## Running via Docker Compose

The `nextcloud_sync` service runs `cron` inside the container and executes
the sync script daily at **02:00**. Start it alongside the other services:

```bash
docker compose up -d nextcloud_sync
```

View logs:

```bash
docker compose exec nextcloud_sync tail -f /var/log/nextcloud_sync.log
```

Run immediately without waiting for the cron trigger:

```bash
docker compose exec nextcloud_sync /app/.venv/bin/python -m src.nextcloud.sync
```

## Running manually (outside Docker)

```bash
pip install -e ".[nextcloud]"
# fill in .env, then:
python -m src.nextcloud.sync
```

## Speaker count in filenames

You can encode the number of speakers directly in the filename instead of (or
to override) the env var. The following patterns are recognised:

| Filename | Detected speakers |
|----------|-------------------|
| `interview_2.mp3` | 2 |
| `session_3spk.wav` | 3 |
| `recording_4speakers.mp4` | 4 |

Per-file values take priority over `NUM_SPEAKERS` / `MIN_SPEAKERS` /
`MAX_SPEAKERS`. If neither the filename nor the env vars provide a speaker
count the file is skipped with an error log entry.

## Output structure

For each source file the script creates a `transcriptions/` subfolder
(if it does not exist) and uploads two files:

```
Nextcloud folder/
├── interview_01.mp3
└── transcriptions/
    ├── interview_01.txt    # timestamped plain-text transcript
    └── interview_01.srt    # subtitle file (importable into video editors)
```

With a sidecar `stt: [turbo, qwen3-asr-1.7b]` next to `interview_01.mp3`,
output is suffixed per model instead:

```
transcriptions/
├── interview_01_turbo.txt
├── interview_01_turbo.srt
├── interview_01_qwen3-asr-1.7b.txt
└── interview_01_qwen3-asr-1.7b.srt
```

`.txt` format:
```
[0:00:00 - 0:00:08] SPEAKER_00: Hello, welcome to the interview.
[0:00:09 - 0:00:14] SPEAKER_01: Thank you for having me.
```

`.srt` format:
```
1
00:00:00,000 --> 00:00:08,320
SPEAKER_00: Hello, welcome to the interview.

2
00:00:09,100 --> 00:00:14,500
SPEAKER_01: Thank you for having me.
```

## Truncation warning (`_OBACHT.txt`)

Found live (2026-08-30): `qwen3-asr`'s underlying library defaults to a
generation-length cap (`max_new_tokens`) far too small for a full audio
chunk, silently producing a transcript that stops well before the audio
actually ends — no error, no visible sign in the output file itself. Fixed
at the source (see `src/audio/stt_backends/qwen3_asr.py`), but as a general
safety net this job now also checks, for every file it transcribes, whether
the last transcript line ends notably earlier than the audio's real
duration:

```
transcriptions/
├── interview_01.txt
├── interview_01.srt
└── interview_01_OBACHT.txt   # only present if truncation looks likely
```

**Presence of `<stem>_OBACHT.txt` means "check this one by hand"** — it's a
blunt, cheap heuristic (compares the transcript's last timestamp against the
audio's real length), not a precise content check, and it can't catch
truncation that happens mid-file rather than right at the end. Absence of
the file doesn't guarantee a perfect transcript, just that this particular
failure mode wasn't detected. The flag is cleared automatically on a
re-run once a transcript no longer looks truncated (e.g. after fixing the
underlying cause and re-processing), so a stale warning doesn't linger next
to a since-corrected file.

A closely related but separate risk: for backends that diarize first and
transcribe each speaker turn separately (`ark-asr`, `hojo-asr`,
`nemotron-asr`), pyannote's speaker turns have no guaranteed maximum
duration — an uninterrupted monologue can produce one very long turn, which
used to get fed to the model whole regardless of that model's own real
per-call audio-length limit. `ark-asr` (documented 30s-per-call limit) now
sub-chunks any turn exceeding that before transcribing (see
`_diarize_then_transcribe` in `stt_backends/base.py`); `nemotron-asr` and
`hojo-asr` have no confirmed documented limit, so they're left as-is rather
than guessing a number.

## Whisper pre-warming

Whisper scales to zero when idle (KEDA scale-from-zero). If the CronJob fires
while Whisper is cold, requests return HTTP 502 until the pod is ready
(typically 30–60 s). Without pre-warming the sync job fails immediately.

The Kubernetes CronJob includes an **init container** (`warm-whisper`) that:
1. Polls `GET /health` on the Whisper service every 15 s
2. Exits once it receives any response other than 502
3. Waits an additional 30 s for the model to fully load into GPU memory

**Known risk with the `qwen3-asr` backend**: `/health` reports whether each
buffer is *accessible*, not whether a specific backend's model is *loaded* —
the fixed 30 s wait was sized for Whisper/WhisperX's typical load time. A
cold Qwen3-ASR load (first request after the pod starts, or after its own
idle-unload timeout) may take longer than 30 s, in which case the sync job's
first `qwen3-asr` request could still race a cold buffer. Not yet fixed —
increase the init container's sleep, or make `/health` report per-backend
model-loaded state, if this turns out to be a real problem in practice.

The main sync container only starts after the init container completes,
guaranteeing Whisper is ready to accept requests. No GPU is held permanently.

This does not apply to the Docker Compose setup (Whisper runs persistently there).

## Cron schedule

**Kubernetes:** The schedule is set via `.env` / `my-values.yaml`:

```bash
NEXTCLOUD_SCHEDULE=*/30 * * * *   # every 30 min (chart default since 2026-08-19; was 0 2 * * * / daily)
```

The value is passed directly to the Kubernetes CronJob — no image rebuild needed. Apply with:

```bash
./scripts/k8s_deploy.sh --values-only
helm upgrade ai-apis helm/ai-apis --namespace ai-apis --values my-values.yaml
```

**Docker Compose:** The default schedule is `0 2 * * *` (02:00 daily).
