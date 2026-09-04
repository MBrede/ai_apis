"""
Batch-folder review table: llm/results_table.xlsx, one sheet per LLM model,
columns are prompts (or "<prompt>:<field>" for a structured prompt).
"""

import logging
from pathlib import Path

import yaml
from webdav3.client import Client
from webdav3.exceptions import RemoteResourceNotFound

from src.nextcloud.batch import _sidecar_from_dict, _stt_models_from_dict
from src.nextcloud.constants import LLM_SUBFOLDER, TRANSCRIPT_SUBFOLDER
from src.nextcloud.prompts import _load_prompt_template
from src.nextcloud.webdav_utils import _download_text

logger = logging.getLogger(__name__)


def _scan_prompt_cells(
    client: Client, folder_path: str, batch_rows: dict[str, dict]
) -> dict[str, dict[str, dict[str, str]]]:
    """Map prompt -> transcript_stem -> model -> output filename, for outputs
    that actually exist in this batch folder's llm/ right now.

    Extracted from `_build_prompt_tables` (its original inline
    existing_transcripts/existing_outputs/expected_stems logic) since the
    judge-scoring pass needs the identical "what's actually been produced"
    map — both callers need to know which (stem, model, prompt) cells have a
    real `.md` output before doing anything with them.

    Args:
        client: WebDAV client.
        folder_path: Remote folder that has the batch.csv/batch.xlsx.
        batch_rows: {stem: row_dict}, as returned by `_load_batch_rows`.

    Returns:
        {prompt: {transcript_stem: {model: output_filename}}} — only
        entries whose output file actually exists in llm/.
    """
    transcript_dir = folder_path.rstrip("/") + f"/{TRANSCRIPT_SUBFOLDER}/"
    llm_dir = folder_path.rstrip("/") + f"/{LLM_SUBFOLDER}/"
    try:
        existing_transcripts = {
            Path(f).stem for f in client.list(transcript_dir) if f.lower().endswith(".txt")
        }
    except RemoteResourceNotFound:
        existing_transcripts = set()
    try:
        existing_outputs = set(client.list(llm_dir))
    except RemoteResourceNotFound:
        existing_outputs = set()

    cells: dict[str, dict[str, dict[str, str]]] = {}
    for stem, data in batch_rows.items():
        parsed = _sidecar_from_dict(data)
        stt_models = _stt_models_from_dict(data)
        expected_stems = [stem] if not stt_models else [f"{stem}_{m}" for m in stt_models]
        for transcript_stem in (s for s in expected_stems if s in existing_transcripts):
            for model in parsed["models"]:
                for prompt in parsed["prompts"]:
                    output_name = f"{transcript_stem}_{model}_{prompt}.md"
                    if output_name in existing_outputs:
                        cells.setdefault(prompt, {}).setdefault(transcript_stem, {})[
                            model
                        ] = output_name
    return cells


def _build_prompt_tables(
    client: Client, folder_path: str, remote_root: str, batch_rows: dict[str, dict], tmp_dir: Path
) -> None:
    """Build one llm/results_table.xlsx for this folder's batch.csv/batch.xlsx,
    one sheet per LLM model, columns are prompts (or `<prompt>:<field>` for
    a structured/`.yaml` prompt — see `_load_prompt_template`).

    Batch-input usability feature: a customer who uploaded many files via a
    batch sheet gets one reviewable workbook instead of only scattered
    individual .md files (which still exist, unchanged) — each model's
    sheet has rows = transcript stems (STT-model suffix included when `stt:`
    was a list), columns = prompts, cells = that (stem, model, prompt)
    combination's output text (blank if not processed yet). A structured
    prompt instead contributes one column per field, each cell the parsed
    value for that field (the saved `.md` file is actually YAML for these —
    see the main jobs loop). Fully rebuilt from whatever's actually in
    llm/ right now (not incrementally appended) whenever called — the
    caller in `main()` only calls this for folders where a job actually
    uploaded something this run, so it's a consistent snapshot without
    needless rework on no-op runs. Only applies to batch-covered folders,
    not plain per-file YAML sidecars.

    Args:
        client: WebDAV client.
        folder_path: Remote folder that has the batch.csv/batch.xlsx.
        remote_root: Top-level NEXTCLOUD_FOLDER entry this folder belongs to
            (prompts live at `<remote_root>/prompts/` — see
            `_load_prompt_template`), needed to know which prompts are
            structured and their field lists.
        batch_rows: {stem: row_dict}, as returned by `_load_batch_rows`.
        tmp_dir: Local scratch directory for downloads/uploads.
    """
    llm_dir = folder_path.rstrip("/") + f"/{LLM_SUBFOLDER}/"
    cells = _scan_prompt_cells(client, folder_path, batch_rows)
    if not cells:
        return

    from openpyxl import Workbook

    # Load each distinct prompt's definition once, to know which are
    # structured (have `fields:`) and their field lists/order.
    prompt_cache: dict[str, dict] = {}
    fields_by_prompt: dict[str, dict[str, str] | None] = {}
    for prompt in cells:
        try:
            fields_by_prompt[prompt] = _load_prompt_template(
                client, remote_root, prompt, tmp_dir, prompt_cache
            )["fields"]
        except (FileNotFoundError, ValueError) as exc:
            logger.error("Could not load prompt '%s' for results_table.xlsx: %s", prompt, exc)
            fields_by_prompt[prompt] = None

    # Invert {prompt: {stem: {model: filename}}} -> {model: {stem: {prompt: filename}}}
    by_model: dict[str, dict[str, dict[str, str]]] = {}
    for prompt, rows in cells.items():
        for stem, models in rows.items():
            for model, output_name in models.items():
                by_model.setdefault(model, {}).setdefault(stem, {})[prompt] = output_name

    wb = Workbook()
    wb.remove(wb.active)
    for model in sorted(by_model):
        stems = by_model[model]
        prompts = sorted({prompt for row in stems.values() for prompt in row})
        # Expand a structured prompt into one "<prompt>:<field>" column per
        # field (in the order declared in fields:); a plain prompt stays a
        # single "<prompt>" column, unchanged. Each column remembers which
        # prompt (and which field within it, if structured) it came from.
        columns: list[tuple[str, str, str | None]] = []  # (header, prompt, field_name_or_None)
        for prompt in prompts:
            fields = fields_by_prompt.get(prompt)
            if fields:
                columns.extend((f"{prompt}:{field}", prompt, field) for field in fields)
            else:
                columns.append((prompt, prompt, None))

        ws = wb.create_sheet(model[:31])
        ws.append(["filename"] + [header for header, _, _ in columns])
        for stem in sorted(stems):
            row_values = [stem]
            # Cache one prompt's downloaded/parsed content across its
            # several field-columns in this row, instead of re-downloading
            # the same output file once per field.
            row_cache: dict[str, str | dict] = {}
            for _, prompt, _ in columns:
                if prompt not in row_cache:
                    output_name = stems[stem].get(prompt)
                    row_cache[prompt] = ""
                    if output_name:
                        try:
                            text = _download_text(client, llm_dir + output_name, tmp_dir)
                            row_cache[prompt] = (
                                yaml.safe_load(text) if fields_by_prompt.get(prompt) else text
                            )
                        except Exception as exc:
                            logger.error(
                                "Failed to download %s for results_table.xlsx: %s", output_name, exc
                            )
            for _, prompt, field in columns:
                content = row_cache[prompt]
                if field is None:
                    row_values.append(content if isinstance(content, str) else "")
                else:
                    row_values.append(content.get(field, "") if isinstance(content, dict) else "")
            ws.append(row_values)

    local_path = tmp_dir / "results_table.xlsx"
    wb.save(local_path)
    remote_out = llm_dir + "results_table.xlsx"
    try:
        client.upload_sync(local_path=str(local_path), remote_path=remote_out)
        logger.info("Uploaded %s", remote_out)
    except Exception as exc:
        logger.error("Upload failed for %s: %s", remote_out, exc)
    local_path.unlink(missing_ok=True)
