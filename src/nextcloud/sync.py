"""
Nextcloud audio/video transcription sync.

Scans one or more configured Nextcloud folders (recursively, comma-separated
in NEXTCLOUD_FOLDER) for new audio/video files, transcribes them with speaker
diarization via the local Whisper API, and uploads .txt and .srt outputs to a
'transcriptions/' subfolder next to each source file.

Every audio/video file is transcribed unconditionally with the
WHISPER_MODEL-env default (unsuffixed output), UNLESS a `<stem>.yaml`/`.yml`
sidecar sits next to it with an `stt:` field (single model name or a list) —
in that case the file is transcribed once per listed model, with output
suffixed `<stem>_<model>.{txt,srt}` and skipped independently per model. See
README_llm.md for the full sidecar schema (shared with sync_llm.py).

A folder-level `batch.csv`/`batch.xlsx` (one row per file, same fields as
the per-file YAML sidecar: `filename`, `stt`, `llm`, `prompt`, `context`,
list-valued fields comma-separated within a cell) covers every file in that
folder that has no `<stem>.yaml` of its own — a customer-facing batch
alternative to hand-authoring one YAML per file. A file's own sidecar
always wins over a matching batch row.

All configuration is read from environment variables (see README_cron.md).

This module is deliberately just the scan/download/transcribe/upload
orchestration (`main()`, plus the two functions most specific to it,
`_collect_new_files` and `_ensure_transcript_folder`) — everything else it
used to contain has moved into sibling modules shared with sync_llm.py:
`constants.py`, `webdav_utils.py`, `batch.py`, `transcription.py`.
"""

import asyncio
import logging
import os
import shutil
import tempfile
from pathlib import Path

import aiohttp
from dotenv import load_dotenv
from tqdm import tqdm
from webdav3.client import Client
from webdav3.exceptions import RemoteResourceNotFound

from src.nextcloud.batch import _load_batch_rows, _parse_sidecar_stt, _stt_models_from_dict
from src.nextcloud.constants import AUDIO_VIDEO_MIME_TYPES, SIDECAR_SUFFIXES, TRANSCRIPT_SUBFOLDER
from src.nextcloud.transcription import (
    _backend_for_stt_model,
    _build_diarize_params,
    _detect_possible_truncation,
    _format_as_srt,
    _format_as_text,
    _transcribe_file,
)
from src.nextcloud.webdav_utils import _download_text, _make_webdav_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Max transcription calls in flight at once, PER target container (the
# primary container and each of its qwen3-asr/hojo-asr sidecar containers
# get their own independent budget). Lower than sync_llm.py's LLM_CONCURRENCY
# on purpose — this hits actual GPU inference directly, no request-queuing
# serving layer like vLLM in front of it. Env-tunable per cluster GPU headroom.
WHISPER_CONCURRENCY = int(os.environ.get("WHISPER_CONCURRENCY", "3"))


def _ensure_transcript_folder(client: Client, remote_dir: str) -> str:
    """Return the transcriptions subfolder path, creating it if needed.

    Args:
        client: WebDAV client.
        remote_dir: Remote directory that contains the source audio file.

    Returns:
        Remote path of the transcriptions subfolder (trailing slash included).
    """
    transcript_dir = remote_dir.rstrip("/") + f"/{TRANSCRIPT_SUBFOLDER}/"
    try:
        client.list(transcript_dir)
    except RemoteResourceNotFound:
        client.mkdir(transcript_dir)
    return transcript_dir


def _collect_new_files(client: Client, remote_roots: list[str], tmp_dir: Path) -> list[dict]:
    """Recursively find audio/video work items that still need transcription.

    Each returned item is one (file, STT model) pair. A file with no sidecar
    and no matching batch.csv/batch.xlsx row, or a sidecar/row with no
    `stt:`/`stt` field, yields exactly one item with `stt_model=None` —
    today's behavior: single WHISPER_MODEL-env-default transcription,
    unsuffixed `<stem>.{txt,srt}` output, skipped if `<stem>` already has a
    transcript. A file with `stt: [m1, m2, ...]` (own sidecar, or a batch
    row if it has no sidecar of its own — see `_load_batch_rows`) yields one
    item per listed model, each independently skippable via its own
    `<stem>_<model>.{txt,srt}` output.

    Args:
        client: WebDAV client.
        remote_roots: Root remote folders to scan (see _make_webdav_client).
        tmp_dir: Local scratch directory for downloading sidecars.

    Returns:
        List of dicts: {"file": <WebDAV file-info dict>, "stt_model": str | None}.
    """
    work_items: list[dict] = []

    def _walk(path: str) -> None:
        try:
            entries = client.list(path, get_info=True)
        except RemoteResourceNotFound:
            logger.warning("Remote path not found, skipping: %s", path)
            return

        subdirs = [e for e in entries if e["isdir"] and e["path"].rstrip("/") != path.rstrip("/")]
        files = [e for e in entries if not e["isdir"]]
        audio_video = [f for f in files if f.get("content_type", "") in AUDIO_VIDEO_MIME_TYPES]

        if audio_video:
            audio_stems = {Path(f["path"]).stem for f in audio_video}
            sidecars_by_stem = {
                Path(f["path"]).stem: f["path"]
                for f in files
                if Path(f["path"]).suffix.lower() in SIDECAR_SUFFIXES
                and Path(f["path"]).stem in audio_stems
            }
            batch_rows = _load_batch_rows(client, files, path, tmp_dir)

            transcript_dir = path.rstrip("/") + f"/{TRANSCRIPT_SUBFOLDER}/"
            try:
                existing_stems = {Path(f).stem for f in client.list(transcript_dir)}
            except RemoteResourceNotFound:
                existing_stems = set()

            for f in audio_video:
                stem = Path(f["path"]).stem
                sidecar_path = sidecars_by_stem.get(stem)
                stt_models: list[str] = []
                if sidecar_path:
                    try:
                        text = _download_text(client, sidecar_path, tmp_dir)
                        stt_models = _parse_sidecar_stt(text)
                    except Exception as exc:
                        logger.error("Failed to read sidecar %s: %s", sidecar_path, exc)
                elif stem in batch_rows:
                    # No per-file sidecar — fall back to this folder's batch.csv/batch.xlsx row.
                    stt_models = _stt_models_from_dict(batch_rows[stem])

                if not stt_models:
                    # No sidecar, or sidecar has no `stt:` — today's default behavior.
                    if stem not in existing_stems:
                        work_items.append({"file": f, "stt_model": None})
                else:
                    for model in stt_models:
                        if f"{stem}_{model}" not in existing_stems:
                            work_items.append({"file": f, "stt_model": model})

        for d in subdirs:
            _walk(d["path"])

    for root in remote_roots:
        _walk(root)
    return work_items


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def main() -> None:
    """Main sync loop: scan → download → transcribe → upload."""
    load_dotenv()

    client, remote_roots = _make_webdav_client()

    tmp_dir = Path(tempfile.mkdtemp(prefix="nc_transcribe_"))
    try:
        logger.info("Scanning %s for new audio/video files...", remote_roots)
        work_items = _collect_new_files(client, remote_roots, tmp_dir)

        if not work_items:
            logger.info("No new files to transcribe.")
            return

        logger.info("Found %d work item(s) to transcribe.", len(work_items))

        timeout = aiohttp.ClientTimeout(total=int(os.environ.get("WHISPER_TIMEOUT", "3600")))
        async with aiohttp.ClientSession(timeout=timeout) as session:
            # Downloads (webdav I/O, cheap) stay sequential; only the actual
            # whisper call is parallelized below, bounded by WHISPER_CONCURRENCY.
            # local_audio is index-prefixed rather than output_stem-derived —
            # two different source folders can share a filename (e.g. two
            # projects both containing "Neue Aufnahme 27.m4a"), which was only
            # safe under the old strictly-one-at-a-time loop; concurrent
            # transcription needs a guaranteed-unique local path per item.
            prepared = []
            for idx, item in enumerate(work_items):
                file_info: dict = item["file"]
                stt_model: str | None = item["stt_model"]
                remote_path: str = file_info["path"]
                mime_type: str = file_info.get("content_type", "audio/wav")
                stem = Path(remote_path).stem
                output_stem = f"{stem}_{stt_model}" if stt_model else stem
                suffix = Path(remote_path).suffix or ".wav"
                local_audio = tmp_dir / f"{idx}_{output_stem}{suffix}"

                try:
                    client.download_sync(remote_path=remote_path, local_path=str(local_audio))
                except Exception as exc:
                    logger.error("Download failed for %s: %s", remote_path, exc)
                    continue

                try:
                    diarize_params = _build_diarize_params(stem, model_override=stt_model)
                except ValueError as exc:
                    logger.error("Skipping %s: %s", remote_path, exc)
                    local_audio.unlink(missing_ok=True)
                    continue

                prepared.append(
                    {
                        "remote_path": remote_path,
                        "output_stem": output_stem,
                        "local_audio": local_audio,
                        "mime_type": mime_type,
                        "diarize_params": diarize_params,
                        # "qwen3-asr" and "hojo-asr" run in their own separate
                        # containers/GPU-scaled deployments (see
                        # whisper_api.py's module docstring) — everything else
                        # (whisperx/ark-asr/granite-speech/nemotron-asr) is
                        # handled by the primary container. Each group gets
                        # its own concurrency budget below so items destined
                        # for one container don't eat into another's queue.
                        "container_group": (
                            _backend_for_stt_model(stt_model) if stt_model else "whisperx"
                        ),
                    }
                )

            proxied_groups = {"qwen3-asr", "hojo-asr"}
            semaphores: dict[str, asyncio.Semaphore] = {}

            def _semaphore_for(group: str) -> asyncio.Semaphore:
                key = group if group in proxied_groups else "primary"
                if key not in semaphores:
                    semaphores[key] = asyncio.Semaphore(WHISPER_CONCURRENCY)
                return semaphores[key]

            async def _run_transcribe(
                p: dict,
            ) -> tuple[dict, tuple[list[dict], float | None] | None]:
                async with _semaphore_for(p["container_group"]):
                    result = await _transcribe_file(
                        str(p["local_audio"]), p["mime_type"], p["diarize_params"], session
                    )
                p["local_audio"].unlink(missing_ok=True)
                return p, result

            tasks = [asyncio.ensure_future(_run_transcribe(p)) for p in prepared]
            for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing"):
                p, result = await coro
                if result is None:
                    continue
                segments, audio_duration = result
                remote_path = p["remote_path"]
                output_stem = p["output_stem"]

                # Write and upload outputs
                remote_dir = str(Path(remote_path).parent)
                transcript_dir = _ensure_transcript_folder(client, remote_dir)

                for content, ext in (
                    (_format_as_text(segments), ".txt"),
                    (_format_as_srt(segments), ".srt"),
                ):
                    local_out = tmp_dir / f"{output_stem}{ext}"
                    local_out.write_text(content, encoding="utf-8")
                    remote_out = transcript_dir + f"{output_stem}{ext}"
                    try:
                        client.upload_sync(local_path=str(local_out), remote_path=remote_out)
                        logger.info("Uploaded %s", remote_out)
                    except Exception as exc:
                        logger.error("Upload failed for %s: %s", remote_out, exc)
                    local_out.unlink(missing_ok=True)

                obacht = _detect_possible_truncation(segments, audio_duration)
                obacht_remote = transcript_dir + f"{output_stem}_OBACHT.txt"
                if obacht:
                    logger.warning("%s: %s", output_stem, obacht)
                    local_obacht = tmp_dir / f"{output_stem}_OBACHT.txt"
                    local_obacht.write_text(obacht, encoding="utf-8")
                    try:
                        client.upload_sync(local_path=str(local_obacht), remote_path=obacht_remote)
                    except Exception as exc:
                        logger.error("Upload failed for %s: %s", obacht_remote, exc)
                    local_obacht.unlink(missing_ok=True)
                else:
                    # Clear a stale flag from a previous run (e.g. re-run
                    # with a fixed backend/config) rather than leaving a
                    # now-inaccurate warning sitting next to a good transcript.
                    try:
                        client.clean(obacht_remote)
                    except RemoteResourceNotFound:
                        pass
                    except Exception as exc:
                        logger.error("Failed to clear stale flag %s: %s", obacht_remote, exc)

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    logger.info("Done.")


if __name__ == "__main__":
    asyncio.run(main())
