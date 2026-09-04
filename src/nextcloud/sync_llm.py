"""
Nextcloud transcript LLM post-processing.

Scans one or more configured Nextcloud folders (recursively, comma-separated
in NEXTCLOUD_FOLDER) for audio/video files that have an optional
`<stem>.yaml` sidecar next to them, OR a matching row in that folder's
`batch.csv`/`batch.xlsx` (same fields, one row per file — see
`_load_batch_rows` in batch.py; a sidecar always wins over a matching batch
row). Once the whisper job (``sync.py``) has produced a matching
transcript, the `llm:` and `prompt:` fields (each a single model/prompt
name or a list) are expanded into the CARTESIAN PRODUCT of every listed
prompt against every listed model, each run against a hosted KubeAI LLM. If
`stt:` requested multiple STT models, EVERY transcript variant
(`transcriptions/<stem>_<sttmodel>.txt`) is processed independently — the
result is uploaded to a sibling ``llm/`` folder as
``<transcript_stem>_<model>_<prompt>.md``, i.e. the STT model (if any) is
folded into the filename automatically since it's already part of
`<transcript_stem>`. An optional `context:` field is injected into prompt
templates via a `{context}` placeholder (same plain-string-replace
mechanism as `{transcript}`). Any OTHER field beyond the fixed
`stt`/`llm`/`prompt`/`context` schema (e.g. a `ground_truth` column added
to a batch.xlsx) is likewise available as a `{field_name}` placeholder —
see `_sidecar_from_dict`'s `extra_fields`.

Jobs are processed grouped by model (not crawl-discovery order) so that
consecutive calls to the same KubeAI-hosted model land within its
scaleDownDelaySeconds warm window, avoiding repeated cold starts.

For batch-file-driven folders, an optional `judges/<name>.yaml` per top-level
NEXTCLOUD_FOLDER entry (sibling to `prompts/`) scores every existing output
cell of its `jurisdiction:` prompt(s) — either `kind: label` (the judge LLM
states its score directly) or `kind: logprob` (G-Eval-style, score derived
from probability-weighted token logprobs). Results land in
`llm/<judge>_scores.xlsx` — one sheet per jurisdiction prompt plus an
"Overview" comparison sheet, so a judge's take across different prompts is
comparable in one file, not scattered across several. Already-scored cells
are never re-scored. Arbitrary extra sidecar/batch-row fields (e.g. a
user-added `ground_truth` column) are available as `{field_name}`
placeholders in both prompt and judge templates. See README_llm.md's
"Judges" section for the full schema.

Fully independent of the whisper job — no audio download, no diarization, no
Whisper calls. Files with neither a sidecar nor a batch row are never touched.

All configuration is read from environment variables (see README_llm.md).

This module is deliberately just the sidecar-driven job discovery
(`_collect_llm_jobs`) and the top-level orchestration (`main()`) —
everything else it used to contain has moved into sibling modules:
`constants.py`, `webdav_utils.py`, `batch.py`, `prompts.py`, `llm_client.py`,
`results_table.py`, `judges.py`.
"""

import asyncio
import logging
import os
import shutil
import tempfile
from pathlib import Path

import aiohttp
import yaml
from dotenv import load_dotenv
from tqdm import tqdm
from webdav3.client import Client
from webdav3.exceptions import RemoteResourceNotFound

from src.nextcloud.batch import _load_batch_rows, _sidecar_from_dict, _stt_models_from_dict
from src.nextcloud.constants import (
    AUDIO_VIDEO_MIME_TYPES,
    JUDGES_SUBFOLDER,
    LLM_SUBFOLDER,
    PROMPTS_SUBFOLDER,
    SIDECAR_SUFFIXES,
    TRANSCRIPT_SUBFOLDER,
)
from src.nextcloud.judges import (
    _collect_judge_scoring_groups,
    _collect_judges,
    _missing_judge_cells,
    _score_one_judge_cell,
    _upload_judge_group_scores,
)
from src.nextcloud.llm_client import _call_llm
from src.nextcloud.prompts import _load_prompt_template, _render_prompt
from src.nextcloud.results_table import _build_prompt_tables
from src.nextcloud.webdav_utils import _download_text, _make_webdav_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Found live 2026-09-02: a fully sequential jobs loop took up to 4h to process
# one prompt across many files — each LLM call is dominated by generation
# time (seconds to low minutes), not I/O, and KubeAI/vLLM already serves
# concurrent requests per model. `LLM_CONCURRENCY` bounds how many _call_llm
# calls run in flight at once (env override for tuning per-cluster GPU
# capacity without a code change) — shared by the main jobs loop and the
# judge-scoring pass below.
LLM_CONCURRENCY = int(os.environ.get("LLM_CONCURRENCY", "10"))


# ---------------------------------------------------------------------------
# Sidecar / job discovery
# ---------------------------------------------------------------------------


def _collect_llm_jobs(
    client: Client, remote_roots: list[str], tmp_dir: Path
) -> tuple[list[dict], dict[str, dict[str, dict]], dict[str, str]]:
    """Recursively find sidecar-driven LLM jobs ready to run.

    A job is ready when: a `<stem>.yaml`/`.yml` sidecar sits next to an
    audio/video file (or, absent that, the file has a matching row in this
    folder's batch.csv/batch.xlsx — see `_load_batch_rows` in sync.py; a
    sidecar always wins over a matching batch row), at least one
    `transcriptions/<stem>[_<sttmodel>].txt` already exists, and
    `llm/<transcript_stem>_<model>_<prompt>.md` does not exist yet. Every
    existing transcript variant for the stem is processed independently —
    `stt: [turbo, qwen3-asr-1.7b]` produces two full llm x prompt batches,
    one per STT model's transcript, each named after that transcript's own
    stem (so the STT model is folded into the output filename
    automatically). Each batch is the cartesian product of `llm:` models ×
    `prompt:` prompts (each field may be a single name or a list — see
    `_sidecar_from_dict`).

    Args:
        client: WebDAV client.
        remote_roots: Root remote folders to scan (see _make_webdav_client).
        tmp_dir: Local scratch directory for downloading sidecars.

    Returns:
        Tuple of (jobs, batch_folders, folder_roots):
        - jobs: list of dicts (transcript_path, llm_dir, output_name, model,
          prompt, context, remote_root — the folder this job's sidecar came
          from, needed so prompt-template lookups check the right folder's
          own `prompts/` subfolder first).
        - batch_folders: {folder_path: batch_rows} for every folder that had
          a batch.csv/batch.xlsx, used by `_build_prompt_tables` to build
          per-prompt review tables after processing.
        - folder_roots: {folder_path: top_level_root} for every folder in
          batch_folders — judges resolve once per top-level root (like
          prompts/), so the judge-scoring pass needs to know which root a
          nested batch folder belongs to.
    """
    jobs: list[dict] = []
    batch_folders: dict[str, dict[str, dict]] = {}
    folder_roots: dict[str, str] = {}

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
            and Path(e["path"]).name
            not in (TRANSCRIPT_SUBFOLDER, LLM_SUBFOLDER, PROMPTS_SUBFOLDER, JUDGES_SUBFOLDER)
        ]
        files = [e for e in entries if not e["isdir"]]
        audio_stems = {
            Path(f["path"]).stem
            for f in files
            if f.get("content_type", "") in AUDIO_VIDEO_MIME_TYPES
        }
        sidecars_by_stem = {
            Path(f["path"]).stem: f["path"]
            for f in files
            if Path(f["path"]).suffix.lower() in SIDECAR_SUFFIXES
            and Path(f["path"]).stem in audio_stems
        }
        batch_rows = _load_batch_rows(client, files, path, tmp_dir)
        if batch_rows:
            batch_folders[path] = batch_rows
            folder_roots[path] = root
        relevant_stems = (set(sidecars_by_stem) | set(batch_rows)) & audio_stems

        if relevant_stems:
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

            for stem in relevant_stems:
                sidecar_path = sidecars_by_stem.get(stem)
                if sidecar_path:
                    try:
                        text = _download_text(client, sidecar_path, tmp_dir)
                    except Exception as exc:
                        logger.error("Failed to download sidecar %s: %s", sidecar_path, exc)
                        continue
                    data = yaml.safe_load(text) or {}
                else:
                    # No per-file sidecar — fall back to this folder's batch.csv/batch.xlsx row.
                    data = batch_rows[stem]

                parsed = _sidecar_from_dict(data)

                # `stt` can produce several transcripts for one audio file:
                # the bare `<stem>.txt` (no `stt` — default, unsuffixed) or
                # `<stem>_<model>.txt` per listed STT model. Derive the
                # expected stems from the sidecar/batch-row data itself, not
                # by guessing from the folder listing — a naive `startswith`
                # match against every transcript stem in the directory is
                # ambiguous (e.g. "interview_2" is a different audio file's
                # transcript per the speaker-count filename convention in
                # sync.py, not an "interview" file with stt-suffix "2").
                stt_models = _stt_models_from_dict(data)
                expected_stems = [stem] if not stt_models else [f"{stem}_{m}" for m in stt_models]
                transcript_stems = [s for s in expected_stems if s in existing_transcripts]
                if not transcript_stems:
                    logger.info("Transcript not ready yet, will retry later: %s", stem)
                    continue
                if len(transcript_stems) < len(expected_stems):
                    logger.info(
                        "Some transcripts not ready yet for %s, processing what exists (%d/%d): %s",
                        stem,
                        len(transcript_stems),
                        len(expected_stems),
                        transcript_stems,
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
                                    "extra_fields": parsed["extra_fields"],
                                    "remote_root": root,
                                }
                            )

        for d in subdirs:
            _walk(d["path"], root)

    for root in remote_roots:
        _walk(root, root)
    return jobs, batch_folders, folder_roots


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
        jobs, batch_folders, folder_roots = _collect_llm_jobs(client, remote_roots, tmp_dir)

        # Defined before the branch below (not inside `else:`) — a true
        # no-op run (no new jobs) still needs this defined for the table-
        # rebuild gate right after, and for the judge pass, which can have
        # work even when this run found zero new LLM jobs.
        updated_llm_dirs: set[str] = set()

        if not jobs:
            logger.info("No LLM jobs ready to run.")
        else:
            logger.info("Found %d LLM job(s) to run.", len(jobs))

            # Group by model (stable sort keeps each model's jobs in their
            # original crawl-discovery order) so consecutive calls to the same
            # KubeAI-hosted model land within its scaleDownDelaySeconds warm
            # window instead of round-robin-ing across models and forcing a
            # cold start on every single call.
            jobs.sort(key=lambda j: j["model"])
            logger.info("Processing order (grouped by model): %s", [j["model"] for j in jobs][:50])

            prompt_cache: dict[str, dict] = {}
            timeout = aiohttp.ClientTimeout(total=int(os.environ.get("LLM_TIMEOUT", "900")))
            async with aiohttp.ClientSession(timeout=timeout) as session:
                # Prompts are prepared up front (webdav downloads are fast,
                # local-network I/O) so the actual bottleneck — each LLM
                # call's own generation time — can run with up to
                # LLM_CONCURRENCY calls in flight at once instead of one at a
                # time. Preparation stays in `jobs`' model-grouped order, and
                # asyncio.gather schedules tasks in that same order, so the
                # warm-model-window benefit of the sort above is preserved
                # even though completion order is no longer sequential.
                prepared = []
                for job in jobs:
                    try:
                        transcript = _download_text(client, job["transcript_path"], tmp_dir)
                    except Exception as exc:
                        logger.error(
                            "Failed to download transcript %s: %s", job["transcript_path"], exc
                        )
                        continue

                    try:
                        loaded = _load_prompt_template(
                            client, job["remote_root"], job["prompt"], tmp_dir, prompt_cache
                        )
                    except (FileNotFoundError, ValueError) as exc:
                        logger.error("%s Skipping %s.", exc, job["transcript_path"])
                        continue
                    prompt = _render_prompt(
                        loaded["template"], transcript, job["context"], job["extra_fields"]
                    )
                    prepared.append((job, prompt, loaded["fields"]))

                semaphore = asyncio.Semaphore(LLM_CONCURRENCY)

                async def _run_call(
                    job: dict, prompt: str, fields: dict[str, str] | None
                ) -> tuple[dict, str | dict[str, str] | None]:
                    async with semaphore:
                        return job, await _call_llm(session, job["model"], prompt, fields)

                tasks = [
                    asyncio.ensure_future(_run_call(job, prompt, fields))
                    for job, prompt, fields in prepared
                ]
                for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing"):
                    job, answer = await coro
                    if answer is None:
                        continue

                    try:
                        client.list(job["llm_dir"])
                    except RemoteResourceNotFound:
                        client.mkdir(job["llm_dir"])

                    # Structured output (from a .yaml prompt) is a dict —
                    # save as YAML so it round-trips losslessly when
                    # _build_prompt_tables re-downloads and re-parses it
                    # later. Filename/extension stay .md either way (skip-
                    # tracking matches on the exact name, not content-type).
                    content = (
                        answer
                        if isinstance(answer, str)
                        else yaml.safe_dump(answer, allow_unicode=True, sort_keys=False)
                    )
                    local_out = tmp_dir / job["output_name"]
                    local_out.write_text(content, encoding="utf-8")
                    remote_out = job["llm_dir"] + job["output_name"]
                    try:
                        client.upload_sync(local_path=str(local_out), remote_path=remote_out)
                        logger.info("Uploaded %s", remote_out)
                        updated_llm_dirs.add(job["llm_dir"])
                    except Exception as exc:
                        logger.error("Upload failed for %s: %s", remote_out, exc)
                    local_out.unlink(missing_ok=True)

        # Only rebuild a folder's per-prompt review tables if a job actually
        # uploaded something new into its llm/ this run — a table is a
        # snapshot of llm/, and llm/ didn't change otherwise. Without this,
        # every cron tick re-downloaded and re-uploaded every table even on
        # a fully-processed, no-op run.
        for folder_path, rows in batch_folders.items():
            llm_dir = folder_path.rstrip("/") + f"/{LLM_SUBFOLDER}/"
            if llm_dir in updated_llm_dirs:
                _build_prompt_tables(client, folder_path, folder_roots[folder_path], rows, tmp_dir)

        # Judge pass — independent of the jobs loop above (judges may have
        # work even on a run with zero new LLM jobs, e.g. outputs from a
        # prior run still awaiting judging). File-driven kill switch: unset
        # or "true" (default) runs it, any other value disables it cluster-
        # wide without needing to delete every judges/*.yaml.
        if os.environ.get("LLM_JUDGE_ENABLED", "true").lower() != "false":
            judges_by_root = _collect_judges(client, remote_roots, tmp_dir)
            if judges_by_root:
                groups = _collect_judge_scoring_groups(
                    client, batch_folders, folder_roots, judges_by_root, tmp_dir
                )
                cell_jobs = [
                    {
                        **group,
                        "prompt": prompt,
                        "stem": stem,
                        "model": model,
                        "output_name": filename,
                        "existing_scores": pdata["existing_scores"],
                    }
                    for group in groups
                    for prompt, pdata in group["prompts"].items()
                    for stem, model, filename in _missing_judge_cells(
                        pdata["cells"], pdata["existing_scores"]
                    )
                ]
                if cell_jobs:
                    # Same warm-model-grouping discipline as the main jobs loop.
                    cell_jobs.sort(key=lambda j: j["judge"]["llm"])
                    logger.info("Found %d judge-scoring cell(s) to run.", len(cell_jobs))

                    # Upload each group's scores.xlsx as soon as ITS cells are
                    # all attempted, not once at the very end — a large
                    # backlog can have hundreds of cells across many groups;
                    # waiting for every single one before any output becomes
                    # visible means nothing shows up for a very long time.
                    remaining: dict[str, int] = {}
                    touched_paths: set[str] = set()
                    for cj in cell_jobs:
                        remaining[cj["scores_remote_path"]] = (
                            remaining.get(cj["scores_remote_path"], 0) + 1
                        )
                    groups_by_path = {g["scores_remote_path"]: g for g in groups}

                    timeout = aiohttp.ClientTimeout(total=int(os.environ.get("LLM_TIMEOUT", "900")))
                    async with aiohttp.ClientSession(timeout=timeout) as session:
                        # Same bounded-concurrency treatment as the main jobs
                        # loop above — up to LLM_CONCURRENCY judge calls in
                        # flight at once (same KubeAI/vLLM backend, same
                        # generation-time-dominated bottleneck). The per-group
                        # "upload once all its cells are attempted" countdown
                        # below is order-independent, so it's unaffected by
                        # cells now completing out of order.
                        semaphore = asyncio.Semaphore(LLM_CONCURRENCY)

                        async def _run_judge_cell(cj: dict) -> tuple[dict, float | str | None]:
                            async with semaphore:
                                return cj, await _score_one_judge_cell(client, session, cj, tmp_dir)

                        tasks = [asyncio.ensure_future(_run_judge_cell(cj)) for cj in cell_jobs]
                        for coro in tqdm(
                            asyncio.as_completed(tasks), total=len(tasks), desc="Judging"
                        ):
                            cj, score = await coro
                            if score is not None:
                                cj["existing_scores"].setdefault(cj["stem"], {})[
                                    cj["model"]
                                ] = score
                                touched_paths.add(cj["scores_remote_path"])

                            remaining[cj["scores_remote_path"]] -= 1
                            if (
                                remaining[cj["scores_remote_path"]] == 0
                                and cj["scores_remote_path"] in touched_paths
                            ):
                                _upload_judge_group_scores(
                                    client, groups_by_path[cj["scores_remote_path"]], tmp_dir
                                )

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    logger.info("Done.")


if __name__ == "__main__":
    asyncio.run(main())
