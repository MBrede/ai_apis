"""
Nextcloud transcript LLM post-processing.

Scans a configured Nextcloud folder (recursively) for audio/video files that
have an optional `<stem>.yaml` sidecar next to them. Once the whisper job
(``sync.py``) has produced a matching ``transcriptions/<stem>.txt``, the
sidecar's chosen prompt (loaded from the Nextcloud ``prompts/`` folder, or
the bundled default if not overridden there) is run against a hosted KubeAI
LLM and the result is uploaded to a sibling ``llm/`` folder as
``<stem>_<prompt>.md``.

Fully independent of the whisper job — no audio download, no diarization, no
Whisper calls. Files without a sidecar are never touched.

All configuration is read from environment variables (see README_llm.md).
"""

import asyncio
import json
import logging
import os
import shutil
import tempfile
from pathlib import Path

import aiohttp
import yaml
from dotenv import load_dotenv
from src.nextcloud.sync import AUDIO_VIDEO_MIME_TYPES, _make_webdav_client
from tqdm import tqdm
from webdav3.client import Client
from webdav3.exceptions import RemoteResourceNotFound

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

TRANSCRIPT_SUBFOLDER = "transcriptions"
LLM_SUBFOLDER = "llm"
PROMPTS_SUBFOLDER = "prompts"
SIDECAR_SUFFIXES = (".yaml", ".yml")
LOCAL_PROMPTS_DIR = Path(__file__).parent / "prompts"
DEFAULT_PROMPT = "summary"


# ---------------------------------------------------------------------------
# Sidecar / job discovery
# ---------------------------------------------------------------------------


def _download_text(client: Client, remote_path: str, tmp_dir: Path) -> str:
    """Download a small remote text file and return its contents.

    Args:
        client: WebDAV client.
        remote_path: Remote file path to download.
        tmp_dir: Local scratch directory for the temporary download.

    Returns:
        Decoded UTF-8 file contents.
    """
    local_path = tmp_dir / Path(remote_path).name
    client.download_sync(remote_path=remote_path, local_path=str(local_path))
    try:
        return local_path.read_text(encoding="utf-8")
    finally:
        local_path.unlink(missing_ok=True)


def _parse_sidecar(text: str) -> tuple[str, str]:
    """Parse a sidecar YAML's content into (llm_model, prompt_name).

    An empty file (or a file with fields omitted) falls back to
    ``LLM_DEFAULT_MODEL`` / ``DEFAULT_PROMPT``.

    Args:
        text: Raw YAML file contents.

    Returns:
        Tuple of (llm_model, prompt_name).
    """
    data = yaml.safe_load(text) or {}
    model = str(data.get("llm") or os.environ.get("LLM_DEFAULT_MODEL", "glm-4-7-flash"))
    prompt = str(data.get("prompt") or DEFAULT_PROMPT)
    return model, prompt


def _collect_llm_jobs(client: Client, remote_root: str, tmp_dir: Path) -> list[dict]:
    """Recursively find sidecar-driven LLM jobs ready to run.

    A job is ready when: a `<stem>.yaml`/`.yml` sidecar sits next to an
    audio/video file, `transcriptions/<stem>.txt` already exists, and
    `llm/<stem>_<prompt>.md` does not exist yet.

    Args:
        client: WebDAV client.
        remote_root: Root remote folder to scan.
        tmp_dir: Local scratch directory for downloading sidecars.

    Returns:
        List of job dicts: transcript_path, llm_dir, output_name, model, prompt.
    """
    jobs: list[dict] = []

    def _walk(path: str) -> None:
        try:
            entries = client.list(path, get_info=True)
        except RemoteResourceNotFound:
            logger.warning("Remote path not found, skipping: %s", path)
            return

        subdirs = [
            e
            for e in entries
            if e["isdir"]
            and e["path"].rstrip("/") != path.rstrip("/")
            and Path(e["path"]).name not in (TRANSCRIPT_SUBFOLDER, LLM_SUBFOLDER, PROMPTS_SUBFOLDER)
        ]
        files = [e for e in entries if not e["isdir"]]
        audio_stems = {
            Path(f["path"]).stem for f in files if f.get("content_type", "") in AUDIO_VIDEO_MIME_TYPES
        }
        sidecars = [
            f
            for f in files
            if Path(f["path"]).suffix.lower() in SIDECAR_SUFFIXES and Path(f["path"]).stem in audio_stems
        ]

        if sidecars:
            transcript_dir = path.rstrip("/") + f"/{TRANSCRIPT_SUBFOLDER}/"
            try:
                existing_transcripts = {Path(f).stem for f in client.list(transcript_dir)}
            except RemoteResourceNotFound:
                existing_transcripts = set()

            llm_dir = path.rstrip("/") + f"/{LLM_SUBFOLDER}/"
            try:
                existing_outputs = set(client.list(llm_dir))
            except RemoteResourceNotFound:
                existing_outputs = set()

            for sidecar in sidecars:
                stem = Path(sidecar["path"]).stem
                if stem not in existing_transcripts:
                    logger.info("Transcript not ready yet, will retry later: %s", stem)
                    continue

                try:
                    text = _download_text(client, sidecar["path"], tmp_dir)
                except Exception as exc:
                    logger.error("Failed to download sidecar %s: %s", sidecar["path"], exc)
                    continue
                model, prompt = _parse_sidecar(text)

                output_name = f"{stem}_{prompt}.md"
                if output_name in existing_outputs:
                    continue

                jobs.append(
                    {
                        "transcript_path": transcript_dir + f"{stem}.txt",
                        "llm_dir": llm_dir,
                        "output_name": output_name,
                        "model": model,
                        "prompt": prompt,
                    }
                )

        for d in subdirs:
            _walk(d["path"])

    _walk(remote_root)
    return jobs


# ---------------------------------------------------------------------------
# Prompting / LLM call
# ---------------------------------------------------------------------------


def _load_prompt_template(
    client: Client,
    remote_root: str,
    prompt_name: str,
    tmp_dir: Path,
    cache: dict[str, str],
) -> str:
    """Load a prompt template, preferring the Nextcloud ``prompts/`` folder.

    Templates are edited by uploading a `.md` file to
    ``<remote_root>/prompts/<prompt_name>.md`` — no image rebuild or redeploy
    needed. Falls back to the template bundled in the image (under the
    script's local ``prompts/`` folder) only if the remote one is missing,
    so a fresh deployment still has a working default. Results are cached
    per run since many jobs typically share the same prompt.

    Args:
        client: WebDAV client.
        remote_root: Root remote folder (prompts live at `<remote_root>/prompts/`).
        prompt_name: Stem of a `.md` template file.
        tmp_dir: Local scratch directory for the temporary download.
        cache: Dict reused across calls within a single run.

    Returns:
        Raw template text (not yet rendered with a transcript).

    Raises:
        FileNotFoundError: If no matching template exists remotely or locally.
    """
    if prompt_name in cache:
        return cache[prompt_name]

    remote_path = f"{remote_root.rstrip('/')}/{PROMPTS_SUBFOLDER}/{prompt_name}.md"
    try:
        template = _download_text(client, remote_path, tmp_dir)
        logger.info("Loaded prompt '%s' from Nextcloud.", prompt_name)
    except RemoteResourceNotFound:
        local_path = LOCAL_PROMPTS_DIR / f"{prompt_name}.md"
        if not local_path.exists():
            raise FileNotFoundError(
                f"Prompt '{prompt_name}' not found in Nextcloud ({remote_path}) "
                f"or bundled defaults ({local_path})."
            )
        logger.info("Prompt '%s' not in Nextcloud, using bundled default.", prompt_name)
        template = local_path.read_text(encoding="utf-8")

    cache[prompt_name] = template
    return template


def _render_prompt(template: str, transcript: str) -> str:
    """Render a prompt template with the transcript text.

    Args:
        template: Raw template text (from `_load_prompt_template`).
        transcript: Plain-text transcript content.

    Returns:
        Rendered prompt text.
    """
    return template.replace("{transcript}", transcript)


async def _call_llm(session: aiohttp.ClientSession, model: str, prompt: str) -> str | None:
    """Send a chat-completion request to the in-cluster KubeAI endpoint.

    Args:
        session: Shared aiohttp session.
        model: KubeAI model name.
        prompt: Fully rendered prompt text.

    Returns:
        The assistant's reply text, or None on failure.
    """
    llm_url = os.environ.get("LLM_URL", "http://kubeai.llm.svc.cluster.local/openai/v1")
    endpoint = f"{llm_url.rstrip('/')}/chat/completions"
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}]}
    headers = {"Authorization": "Bearer not-used"}

    try:
        async with session.post(endpoint, json=payload, headers=headers) as resp:
            if not (200 <= resp.status < 400):
                body = await resp.text()
                logger.error("LLM request failed (HTTP %s): %s", resp.status, body)
                return None
            data = json.loads(await resp.read())
            return data["choices"][0]["message"]["content"]
    except Exception as exc:
        # asyncio.TimeoutError (and a few other exceptions) have an empty
        # str() — include the type name so timeouts are distinguishable
        # from other failures in the logs.
        logger.error("Error calling LLM: %s: %s", type(exc).__name__, exc)
        return None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def main() -> None:
    """Main sync loop: scan for sidecars → wait for transcripts → LLM → upload."""
    load_dotenv()

    client, remote_root = _make_webdav_client()

    tmp_dir = Path(tempfile.mkdtemp(prefix="nc_llm_"))
    try:
        logger.info("Scanning %s for sidecar-driven LLM jobs...", remote_root)
        jobs = _collect_llm_jobs(client, remote_root, tmp_dir)

        if not jobs:
            logger.info("No LLM jobs ready to run.")
            return

        logger.info("Found %d LLM job(s) to run.", len(jobs))

        prompt_cache: dict[str, str] = {}
        timeout = aiohttp.ClientTimeout(total=int(os.environ.get("LLM_TIMEOUT", "900")))
        async with aiohttp.ClientSession(timeout=timeout) as session:
            for job in tqdm(jobs, desc="Processing"):
                try:
                    transcript = _download_text(client, job["transcript_path"], tmp_dir)
                except Exception as exc:
                    logger.error("Failed to download transcript %s: %s", job["transcript_path"], exc)
                    continue

                try:
                    template = _load_prompt_template(client, remote_root, job["prompt"], tmp_dir, prompt_cache)
                except FileNotFoundError as exc:
                    logger.error("%s Skipping %s.", exc, job["transcript_path"])
                    continue
                prompt = _render_prompt(template, transcript)

                answer = await _call_llm(session, job["model"], prompt)
                if answer is None:
                    continue

                try:
                    client.list(job["llm_dir"])
                except RemoteResourceNotFound:
                    client.mkdir(job["llm_dir"])

                local_out = tmp_dir / job["output_name"]
                local_out.write_text(answer, encoding="utf-8")
                remote_out = job["llm_dir"] + job["output_name"]
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
