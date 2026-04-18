# Nextcloud Transcription Sync

Scans a Nextcloud folder (recursively) once a day for new audio/video files,
transcribes them with speaker diarization via the local Whisper API, and
uploads `.txt` and `.srt` outputs to a `transcriptions/` subfolder next to
each source file. Already-transcribed files are skipped.

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

## Cron schedule

**Kubernetes:** The schedule is set via `.env` / `my-values.yaml`:

```bash
NEXTCLOUD_SCHEDULE=0 2 * * *   # daily at 02:00
```

The value is passed directly to the Kubernetes CronJob — no image rebuild needed. Apply with:

```bash
./scripts/k8s_deploy.sh --values-only
helm upgrade ai-apis helm/ai-apis --namespace ai-apis --values my-values.yaml
```

**Docker Compose:** The default schedule is `0 2 * * *` (02:00 daily).
