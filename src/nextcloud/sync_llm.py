"""
Nextcloud transcript LLM post-processing.

Scans one or more configured Nextcloud folders (recursively, comma-separated
in NEXTCLOUD_FOLDER) for audio/video files that have an optional
`<stem>.yaml` sidecar next to them. Once the whisper job (``sync.py``) has
produced a matching transcript, the sidecar's `llm:` and `prompt:` fields
(each a single model/prompt name or a list) are expanded into the CARTESIAN
PRODUCT of every listed prompt against every listed model, each run against
a hosted KubeAI LLM. If the sidecar's `stt:` field requested multiple STT
models, EVERY transcript variant (`transcriptions/<stem>_<sttmodel>.txt`) is
processed independently — the result is uploaded to a sibling ``llm/``
folder as ``<transcript_stem>_<model>_<prompt>.md``, i.e. the STT model (if
any) is folded into the filename automatically since it's already part of
`<transcript_stem>`. An optional `context:` field is injected into prompt
templates via a `{context}` placeholder (same plain-string-replace
mechanism as `{transcript}`).

Jobs are processed grouped by model (not crawl-discovery order) so that
consecutive calls to the same KubeAI-hosted model land within its
scaleDownDelaySeconds warm window, avoiding repeated cold starts.

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
from src.nextcloud.sync import (
    AUDIO_VIDEO_MIME_TYPES,
    SIDECAR_SUFFIXES,
    _download_text,
    _make_webdav_client,
    _normalize_list,
    _parse_sidecar_stt,
)
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
LOCAL_PROMPTS_DIR = Path(__file__).parent / "prompts"
DEFAULT_PROMPT = "summary"


# ---------------------------------------------------------------------------
# Sidecar / job discovery
# ---------------------------------------------------------------------------


def _parse_sidecar(text: str) -> dict:
    """Parse a sidecar YAML's content into normalized model/prompt lists + context.

    An empty file (or a file with fields omitted) falls back to a 1-element
    ``[LLM_DEFAULT_MODEL]`` / ``[DEFAULT_PROMPT]``. `stt:` is intentionally
    ignored here — that field belongs to sync.py's whisper stage.

    Args:
        text: Raw YAML file contents.

    Returns:
        Dict with keys "models" (list[str]), "prompts" (list[str]), and
        "context" (str, empty if absent).
    """
    data = yaml.safe_load(text) or {}
    models = _normalize_list(data.get("llm")) or [os.environ.get("LLM_DEFAULT_MODEL", "glm-4-7-flash")]
    prompts = _normalize_list(data.get("prompt")) or [DEFAULT_PROMPT]
    context = str(data.get("context") or "")
    return {"models": models, "prompts": prompts, "context": context}


def _collect_llm_jobs(client: Client, remote_roots: list[str], tmp_dir: Path) -> list[dict]:
    """Recursively find sidecar-driven LLM jobs ready to run.

    A job is ready when: a `<stem>.yaml`/`.yml` sidecar sits next to an
    audio/video file, at least one `transcriptions/<stem>[_<sttmodel>].txt`
    already exists, and `llm/<transcript_stem>_<model>_<prompt>.md` does not
    exist yet. Every existing transcript variant for the stem is processed
    independently — a sidecar with `stt: [turbo, qwen3-asr-1.7b]` produces
    two full llm x prompt batches, one per STT model's transcript, each
    named after that transcript's own stem (so the STT model is folded into
    the output filename automatically). Each batch is the cartesian product
    of the sidecar's `llm:` models × `prompt:` prompts (each field may be a
    single name or a list — see `_parse_sidecar`).

    Args:
        client: WebDAV client.
        remote_roots: Root remote folders to scan (see _make_webdav_client).
        tmp_dir: Local scratch directory for downloading sidecars.

    Returns:
        List of job dicts: transcript_path, llm_dir, output_name, model,
        prompt, context, remote_root (the folder this job's sidecar came
        from — needed so prompt-template lookups check the right folder's
        own `prompts/` subfolder first).
    """
    jobs: list[dict] = []

    def _walk(path: str, root: str) -> None:
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
                # .txt only — every transcript has a matching .srt with the
                # same stem, and we'd otherwise see each transcript twice.
                existing_transcripts = {
                    Path(f).stem for f in client.list(transcript_dir) if f.lower().endswith(".txt")
                }
            except RemoteResourceNotFound:
                existing_transcripts = set()

            llm_dir = path.rstrip("/") + f"/{LLM_SUBFOLDER}/"
            try:
                existing_outputs = set(client.list(llm_dir))
            except RemoteResourceNotFound:
                existing_outputs = set()

            for sidecar in sidecars:
                stem = Path(sidecar["path"]).stem

                try:
                    text = _download_text(client, sidecar["path"], tmp_dir)
                except Exception as exc:
                    logger.error("Failed to download sidecar %s: %s", sidecar["path"], exc)
                    continue
                parsed = _parse_sidecar(text)

                # A sidecar's `stt:` field can produce several transcripts for
                # one audio file: the bare `<stem>.txt` (no `stt:` — default,
                # unsuffixed) or `<stem>_<model>.txt` per listed STT model.
                # Derive the expected stems from the SIDECAR itself, not by
                # guessing from the folder listing — a naive `startswith`
                # match against every transcript stem in the directory is
                # ambiguous (e.g. "interview_2" is a different audio file's
                # transcript per the speaker-count filename convention in
                # sync.py, not an "interview" file with stt-suffix "2").
                stt_models = _parse_sidecar_stt(text)
                expected_stems = [stem] if not stt_models else [f"{stem}_{m}" for m in stt_models]
                transcript_stems = [s for s in expected_stems if s in existing_transcripts]
                if not transcript_stems:
                    logger.info("Transcript not ready yet, will retry later: %s", stem)
                    continue
                if len(transcript_stems) < len(expected_stems):
                    logger.info(
                        "Some transcripts not ready yet for %s, processing what exists (%d/%d): %s",
                        stem, len(transcript_stems), len(expected_stems), transcript_stems,
                    )

                for transcript_stem in transcript_stems:
                    for model in parsed["models"]:
                        for prompt in parsed["prompts"]:
                            output_name = f"{transcript_stem}_{model}_{prompt}.md"
                            if output_name in existing_outputs:
                                continue

                            jobs.append(
                                {
                                    "transcript_path": transcript_dir + f"{transcript_stem}.txt",
                                    "llm_dir": llm_dir,
                                    "output_name": output_name,
                                    "model": model,
                                    "prompt": prompt,
                                    "context": parsed["context"],
                                    "remote_root": root,
                                }
                            )

        for d in subdirs:
            _walk(d["path"], root)

    for root in remote_roots:
        _walk(root, root)
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
        remote_root: Root remote folder this job's sidecar came from (prompts
            live at `<remote_root>/prompts/`) — with multiple NEXTCLOUD_FOLDER
            entries, each folder's own prompts/ is checked independently, so
            the cache key must include it (see `cache`).
        prompt_name: Stem of a `.md` template file.
        tmp_dir: Local scratch directory for the temporary download.
        cache: Dict reused across calls within a single run, keyed
            `"<remote_root>:<prompt_name>"` — the same prompt name can
            resolve to a different template per folder.

    Returns:
        Raw template text (not yet rendered with a transcript).

    Raises:
        FileNotFoundError: If no matching template exists remotely or locally.
    """
    cache_key = f"{remote_root}:{prompt_name}"
    if cache_key in cache:
        return cache[cache_key]

    remote_path = f"{remote_root.rstrip('/')}/{PROMPTS_SUBFOLDER}/{prompt_name}.md"
    try:
        template = _download_text(client, remote_path, tmp_dir)
        logger.info("Loaded prompt '%s' from Nextcloud (%s).", prompt_name, remote_root)
    except RemoteResourceNotFound:
        local_path = LOCAL_PROMPTS_DIR / f"{prompt_name}.md"
        if not local_path.exists():
            raise FileNotFoundError(
                f"Prompt '{prompt_name}' not found in Nextcloud ({remote_path}) "
                f"or bundled defaults ({local_path})."
            )
        logger.info("Prompt '%s' not in Nextcloud, using bundled default.", prompt_name)
        template = local_path.read_text(encoding="utf-8")

    cache[cache_key] = template
    return template


def _render_prompt(template: str, transcript: str, context: str = "") -> str:
    """Render a prompt template with the transcript text and optional context.

    Args:
        template: Raw template text (from `_load_prompt_template`).
        transcript: Plain-text transcript content.
        context: Sidecar-provided `context:` text, injected via a `{context}`
            placeholder (same plain-string-replace mechanism as
            `{transcript}`). Templates that don't reference `{context}` are
            unaffected — empty string is the default, matching every sidecar
            that omits the field.

    Returns:
        Rendered prompt text.
    """
    return template.replace("{transcript}", transcript).replace("{context}", context)


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
            content = data["choices"][0]["message"]["content"]
            # Reasoning models (e.g. glm-4-7-flash) can emit their chain-of-thought
            # inline as `<think>...</think>` instead of a separate reasoning_content
            # field, depending on vLLM's reasoning-parser config. Strip it.
            if "</think>" in content:
                content = content.rsplit("</think>", 1)[1]
            return content.strip()
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

    client, remote_roots = _make_webdav_client()

    tmp_dir = Path(tempfile.mkdtemp(prefix="nc_llm_"))
    try:
        logger.info("Scanning %s for sidecar-driven LLM jobs...", remote_roots)
        jobs = _collect_llm_jobs(client, remote_roots, tmp_dir)

        if not jobs:
            logger.info("No LLM jobs ready to run.")
            return

        logger.info("Found %d LLM job(s) to run.", len(jobs))

        # Group by model (stable sort keeps each model's jobs in their
        # original crawl-discovery order) so consecutive calls to the same
        # KubeAI-hosted model land within its scaleDownDelaySeconds warm
        # window instead of round-robin-ing across models and forcing a
        # cold start on every single call.
        jobs.sort(key=lambda j: j["model"])
        logger.info("Processing order (grouped by model): %s", [j["model"] for j in jobs][:50])

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
                    template = _load_prompt_template(
                        client, job["remote_root"], job["prompt"], tmp_dir, prompt_cache
                    )
                except FileNotFoundError as exc:
                    logger.error("%s Skipping %s.", exc, job["transcript_path"])
                    continue
                prompt = _render_prompt(template, transcript, job["context"])

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
