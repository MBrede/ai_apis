"""
Nextcloud audio/video transcription sync.

Scans a configured Nextcloud folder (recursively) for new audio/video files,
transcribes them with speaker diarization via the local Whisper API, and
uploads .txt and .srt outputs to a 'transcriptions/' subfolder next to each
source file.

All configuration is read from environment variables (see README_cron.md).
"""

import asyncio
import datetime
import json
import logging
import os
import re
import shutil
import tempfile
import warnings
from pathlib import Path

import aiohttp
from dotenv import load_dotenv
from src.core.auth import build_auth_headers
from tqdm import tqdm
from webdav3.client import Client
from webdav3.exceptions import RemoteResourceNotFound

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

AUDIO_VIDEO_MIME_TYPES: frozenset[str] = frozenset(
    [
        "audio/wav",
        "audio/x-wav",
        "audio/mpeg",
        "audio/mp3",
        "audio/ogg",
        "audio/x-m4a",
        "audio/mp4",
        "audio/flac",
        "audio/aac",
        "audio/webm",
        "video/mp4",
        "video/x-msvideo",
        "video/quicktime",
        "video/x-matroska",
        "video/webm",
        "video/avi",
    ]
)

TRANSCRIPT_SUBFOLDER = "transcriptions"


# ---------------------------------------------------------------------------
# Formatting helpers (mirrors bot.py output)
# ---------------------------------------------------------------------------


def _seconds_to_srt_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp HH:MM:SS,mmm."""
    total_ms = int(seconds * 1000)
    ms = total_ms % 1000
    total_s = total_ms // 1000
    h = total_s // 3600
    m = (total_s % 3600) // 60
    s = total_s % 60
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _format_as_srt(segments: list[dict]) -> str:
    """Format diarization segments as SRT subtitles."""
    blocks = []
    for i, seg in enumerate(segments, start=1):
        start_ts = _seconds_to_srt_timestamp(seg["START"])
        end_ts = _seconds_to_srt_timestamp(seg["START"] + seg["DURATION"])
        blocks.append(
            f"{i}\n{start_ts} --> {end_ts}\n{seg['SPEAKER']}: {seg['TRANSCRIPTION'].strip()}"
        )
    return "\n\n".join(blocks)


def _format_as_text(segments: list[dict]) -> str:
    """Format diarization segments as a readable plain-text transcript."""
    lines = []
    for seg in segments:
        start_str = str(datetime.timedelta(seconds=int(seg["START"])))
        end_str = str(datetime.timedelta(seconds=int(seg["START"] + seg["DURATION"])))
        lines.append(
            f"[{start_str} - {end_str}] {seg['SPEAKER']}: {seg['TRANSCRIPTION'].strip()}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# WebDAV helpers
# ---------------------------------------------------------------------------


def _make_webdav_client() -> tuple[Client, str]:
    """Create a WebDAV client from environment variables.

    Returns:
        Tuple of (client, remote_root_path).

    Raises:
        KeyError: If a required environment variable is missing.
    """
    url = os.environ["NEXTCLOUD_URL"].rstrip("/")
    user = os.environ["NEXTCLOUD_USER"]
    password = os.environ["NEXTCLOUD_PASSWORD"]
    folder = os.environ["NEXTCLOUD_FOLDER"].strip("/")

    client = Client(
        {
            "webdav_hostname": url,
            "webdav_login": user,
            "webdav_password": password,
        }
    )
    remote_root = f"/remote.php/dav/files/{user}/{folder}"
    return client, remote_root


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


def _collect_new_files(client: Client, remote_root: str) -> list[dict]:
    """Recursively find audio/video files that have no transcript yet.

    Args:
        client: WebDAV client.
        remote_root: Root remote folder to scan.

    Returns:
        List of WebDAV file-info dicts for files that still need transcription.
    """
    new_files: list[dict] = []

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
            transcript_dir = path.rstrip("/") + f"/{TRANSCRIPT_SUBFOLDER}/"
            try:
                existing_stems = {Path(f).stem for f in client.list(transcript_dir)}
            except RemoteResourceNotFound:
                existing_stems = set()

            for f in audio_video:
                if Path(f["path"]).stem not in existing_stems:
                    new_files.append(f)

        for d in subdirs:
            _walk(d["path"])

    _walk(remote_root)
    return new_files


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------


def _speakers_from_filename(stem: str) -> int | None:
    """Try to extract a speaker count encoded in a filename stem.

    Recognised patterns (case-insensitive, anywhere in the stem):
        interview_2         → trailing _<digits>
        session_2spk        → _<digits>spk[s]
        recording_2speaker  → _<digits>speaker[s]

    Args:
        stem: Filename without extension, e.g. "interview_2" or "session_2spk".

    Returns:
        Parsed speaker count, or None if no pattern matches.
    """
    patterns = [
        r"_(\d+)speakers?$",
        r"_(\d+)spks?$",
        r"_(\d+)$",
    ]
    for pat in patterns:
        m = re.search(pat, stem, re.IGNORECASE)
        if m:
            return int(m.group(1))
    return None


def _build_diarize_params(stem: str | None = None) -> dict[str, str | int]:
    """Build query parameters for the /transcribe_and_diarize/ endpoint.

    Speaker count priority:
      1. Encoded in the filename stem (e.g. ``interview_2.mp3`` → 2 speakers)
      2. ``NUM_SPEAKERS`` environment variable
      3. ``MIN_SPEAKERS`` + ``MAX_SPEAKERS`` environment variables

    Args:
        stem: Filename stem of the file being transcribed (without extension).

    Returns:
        Query-parameter dict for the diarization endpoint.

    Raises:
        ValueError: If no speaker count can be determined.
    """
    params: dict[str, str | int] = {
        "model_to_use": os.environ.get("WHISPER_MODEL", "turbo"),
    }

    # 1. Filename-encoded speaker count
    if stem is not None:
        n = _speakers_from_filename(stem)
        if n is not None:
            logger.info("Using %d speaker(s) from filename '%s'.", n, stem)
            params["num_speakers"] = n
            return params

    # 2. Environment variables
    num = os.environ.get("NUM_SPEAKERS")
    min_s = os.environ.get("MIN_SPEAKERS")
    max_s = os.environ.get("MAX_SPEAKERS")
    if num:
        params["num_speakers"] = int(num)
    elif min_s and max_s:
        params["min_speakers"] = int(min_s)
        params["max_speakers"] = int(max_s)
    else:
        raise ValueError(
            "Cannot determine speaker count for this file. "
            "Encode it in the filename (e.g. interview_2.mp3) or set "
            "NUM_SPEAKERS / MIN_SPEAKERS + MAX_SPEAKERS in your environment."
        )
    return params


async def _transcribe_file(
    local_path: str,
    mime_type: str,
    params: dict[str, str | int],
    session: aiohttp.ClientSession,
) -> list[dict] | None:
    """POST a file to the Whisper diarization endpoint.

    Args:
        local_path: Path to the downloaded audio/video file.
        mime_type: MIME type reported by Nextcloud.
        params: Query parameters (speaker count, model, api_key).
        session: Shared aiohttp session.

    Returns:
        List of segment dicts on success, None on failure.
    """
    whisper_url = os.environ.get("WHISPER_URL", "http://whisper:8080")
    endpoint = f"{whisper_url.rstrip('/')}/transcribe_and_diarize/"
    filename = Path(local_path).name

    try:
        with open(local_path, "rb") as fh:
            form = aiohttp.FormData()
            form.add_field("file", fh, filename=filename, content_type=mime_type)
            auth_headers = build_auth_headers(os.environ.get("WHISPER_API_KEY"))
            async with session.post(endpoint, params=params, data=form, headers=auth_headers) as resp:
                if not (200 <= resp.status < 400):
                    body = await resp.text()
                    warnings.warn(
                        f"Transcription failed for {filename} (HTTP {resp.status}): {body}"
                    )
                    return None
                return json.loads(await resp.read())["answer"]
    except Exception as exc:
        warnings.warn(f"Error transcribing {filename}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def main() -> None:
    """Main sync loop: scan → download → transcribe → upload."""
    load_dotenv()

    client, remote_root = _make_webdav_client()

    logger.info("Scanning %s for new audio/video files...", remote_root)
    new_files = _collect_new_files(client, remote_root)

    if not new_files:
        logger.info("No new files to transcribe.")
        return

    logger.info("Found %d new file(s) to transcribe.", len(new_files))

    tmp_dir = Path(tempfile.mkdtemp(prefix="nc_transcribe_"))
    try:
        timeout = aiohttp.ClientTimeout(total=int(os.environ.get("WHISPER_TIMEOUT", "3600")))
        async with aiohttp.ClientSession(timeout=timeout) as session:
            for file_info in tqdm(new_files, desc="Processing"):
                remote_path: str = file_info["path"]
                mime_type: str = file_info.get("content_type", "audio/wav")
                stem = Path(remote_path).stem
                suffix = Path(remote_path).suffix or ".wav"
                local_audio = tmp_dir / f"{stem}{suffix}"

                # Download
                try:
                    client.download_sync(remote_path=remote_path, local_path=str(local_audio))
                except Exception as exc:
                    logger.error("Download failed for %s: %s", remote_path, exc)
                    continue

                # Transcribe (speaker count from filename, then env vars)
                try:
                    diarize_params = _build_diarize_params(stem)
                except ValueError as exc:
                    logger.error("Skipping %s: %s", remote_path, exc)
                    local_audio.unlink(missing_ok=True)
                    continue
                segments = await _transcribe_file(str(local_audio), mime_type, diarize_params, session)
                local_audio.unlink(missing_ok=True)
                if segments is None:
                    continue

                # Write and upload outputs
                remote_dir = str(Path(remote_path).parent)
                transcript_dir = _ensure_transcript_folder(client, remote_dir)

                for content, ext in (
                    (_format_as_text(segments), ".txt"),
                    (_format_as_srt(segments), ".srt"),
                ):
                    local_out = tmp_dir / f"{stem}{ext}"
                    local_out.write_text(content, encoding="utf-8")
                    remote_out = transcript_dir + f"{stem}{ext}"
                    try:
                        client.upload_sync(local_path=str(local_out), remote_path=remote_out)
                        logger.info("Uploaded %s", remote_out)
                    except Exception as exc:
                        logger.error("Upload failed for %s: %s", remote_out, exc)
                    local_out.unlink(missing_ok=True)

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    logger.info("Done.")


if __name__ == "__main__":
    asyncio.run(main())
