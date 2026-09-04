"""
Sidecar and batch.csv/batch.xlsx row parsing, shared by sync.py (STT model
selection) and sync_llm.py (LLM model/prompt/context/extra-field selection)
— both read the same underlying per-file YAML sidecar or batch-row shape,
just extracting different fields from it.
"""

import logging
import os
from pathlib import Path

import yaml
from webdav3.client import Client

from src.nextcloud.constants import (
    BATCH_FILENAMES,
    BATCH_ROW_LIST_FIELDS,
    BATCH_ROW_RESERVED_FIELDS,
    DEFAULT_PROMPT,
)

logger = logging.getLogger(__name__)


def _normalize_list(value: object) -> list[str]:
    """Normalize a sidecar field that may be a scalar, a list, or absent.

    Args:
        value: Raw value from a parsed sidecar YAML dict (``data.get(key)``).

    Returns:
        ``[]`` if `value` is ``None``; the list of stringified items if
        `value` is already a list; a 1-element list of `str(value)` otherwise.
    """
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    return [str(value)]


def _stt_models_from_dict(data: dict) -> list[str]:
    """Return the STT models requested by a parsed sidecar/batch-row dict's `stt` field.

    Shared core of `_parse_sidecar_stt` (YAML sidecar) and the batch-file
    path (`_load_batch_rows` already produces this dict shape directly, no
    YAML round-trip needed).

    Returns:
        List of requested model names. Empty means "no opt-in" — callers
        should fall back to the WHISPER_MODEL-env default with unsuffixed
        output (today's behavior).
    """
    return _normalize_list(data.get("stt"))


def _parse_sidecar_stt(text: str) -> list[str]:
    """Return the STT models requested by a sidecar's `stt:` field.

    Args:
        text: Raw sidecar YAML contents.

    Returns:
        List of requested model names. Empty means "no sidecar opt-in" —
        callers should fall back to the WHISPER_MODEL-env default with
        unsuffixed output (today's behavior).
    """
    return _stt_models_from_dict(yaml.safe_load(text) or {})


BATCH_ROW_RESERVED_FIELDS = frozenset({"filename", "file", "context", *BATCH_ROW_LIST_FIELDS})


def _row_to_sidecar_dict(row: dict[str, str]) -> dict:
    """Convert one batch.csv/batch.xlsx row into the same dict shape
    `yaml.safe_load()` produces for an equivalent per-file YAML sidecar.

    Args:
        row: Column name -> cell value (both already lowercased/stripped by
            the caller). `stt`/`llm`/`prompt` are comma-split into lists;
            `context` is passed through as free text. Any OTHER column
            (e.g. a user-added `ground_truth` column) is passed through
            as-is too — `_sidecar_from_dict` exposes these as
            `extra_fields`, usable as `{field_name}` placeholders in both
            prompt and judge templates. Same passthrough happens naturally
            for a YAML sidecar's own extra top-level keys, since
            `yaml.safe_load` already keeps them — this function only needs
            to not drop them for the batch-row path.

    Returns:
        Dict with only the keys that had a non-empty cell — same convention
        `_stt_models_from_dict`/`_sidecar_from_dict` already expect from
        `data.get(key)` returning `None` for an absent field.
    """
    result: dict = {}
    for field in BATCH_ROW_LIST_FIELDS:
        value = (row.get(field) or "").strip()
        if value:
            result[field] = [v.strip() for v in value.split(",") if v.strip()]
    context = (row.get("context") or "").strip()
    if context:
        result["context"] = context
    for key, value in row.items():
        if key in BATCH_ROW_RESERVED_FIELDS:
            continue
        # Always include the column, even blank — `_read_batch_xlsx`/
        # `_read_batch_csv` already give every row a key per header, so
        # this just preserves that. Skipping blanks here would mean a
        # {field_name} placeholder referencing this column stays literally
        # unreplaced for any row/file that happens to leave the cell empty
        # (not substituted with "" the way {context} is), which is a real
        # bug found live — a judge's {ground_truth} placeholder showed up
        # unrendered for the two demo files that had no ground_truth cell.
        result[key] = (value or "").strip()
    return result


def _load_batch_rows(
    client: Client, files: list[dict], remote_dir: str, tmp_dir: Path
) -> dict[str, dict]:
    """Find and parse a folder-level batch.csv/batch.xlsx, if present.

    One row per file (matched by a `filename` column against each audio
    file's stem, extension optional), same fields as the per-file YAML
    sidecar. Lets a customer hand over one spreadsheet instead of
    authoring a `<stem>.yaml` per file — see this module's docstring.

    Args:
        client: WebDAV client.
        files: File-info dicts for this directory (from `client.list(..., get_info=True)`).
        remote_dir: Directory path these files live in (for logging only).
        tmp_dir: Local scratch directory for the temporary download.

    Returns:
        `{stem: row_dict}`, `row_dict` in the same shape `_row_to_sidecar_dict`
        returns. Empty dict if no batch file is present in this folder.
    """
    batch_file = next(
        (f for name in BATCH_FILENAMES for f in files if Path(f["path"]).name.lower() == name),
        None,
    )
    if batch_file is None:
        return {}

    local_path = tmp_dir / Path(batch_file["path"]).name
    client.download_sync(remote_path=batch_file["path"], local_path=str(local_path))
    try:
        suffix = local_path.suffix.lower()
        if suffix == ".xlsx":
            rows = _read_batch_xlsx(local_path)
        else:
            rows = _read_batch_csv(local_path)
    except Exception as exc:
        logger.error("Failed to read batch file %s in %s: %s", batch_file["path"], remote_dir, exc)
        return {}
    finally:
        local_path.unlink(missing_ok=True)

    batch_rows: dict[str, dict] = {}
    for row in rows:
        filename = (row.get("filename") or row.get("file") or "").strip()
        if not filename:
            continue
        batch_rows[Path(filename).stem] = _row_to_sidecar_dict(row)
    return batch_rows


def _read_batch_csv(local_path: Path) -> list[dict[str, str]]:
    """Read a batch.csv into a list of {lowercased column name: cell value} dicts."""
    import csv

    with open(local_path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        return [{(k or "").strip().lower(): (v or "") for k, v in row.items()} for row in reader]


def _read_batch_xlsx(local_path: Path) -> list[dict[str, str]]:
    """Read a batch.xlsx into a list of {lowercased column name: cell value} dicts."""
    from openpyxl import load_workbook

    wb = load_workbook(local_path, read_only=True, data_only=True)
    ws = wb.active
    rows_iter = ws.iter_rows(values_only=True)
    try:
        headers = [str(h or "").strip().lower() for h in next(rows_iter)]
    except StopIteration:
        return []
    return [
        {headers[i]: ("" if v is None else str(v)) for i, v in enumerate(row) if i < len(headers)}
        for row in rows_iter
        if any(v is not None for v in row)
    ]


_RESERVED_SIDECAR_FIELDS = frozenset({"stt", "llm", "prompt", "context"})


def _sidecar_from_dict(data: dict) -> dict:
    """Normalize an already-parsed sidecar/batch-row dict into model/prompt lists + context.

    Shared core of `_parse_sidecar` (YAML sidecar) and the batch-file path
    (`_load_batch_rows` already produces this dict shape directly, no YAML
    round-trip needed).

    An empty dict (or one with fields omitted) falls back to a 1-element
    ``[LLM_DEFAULT_MODEL]`` / ``[DEFAULT_PROMPT]``. `stt` is intentionally
    ignored here — that field belongs to the whisper stage in sync.py (see
    `_stt_models_from_dict`).

    Any field beyond the fixed schema (`stt`/`llm`/`prompt`/`context`) is
    passed through as `extra_fields` — lets a user add an arbitrary column
    (e.g. a `ground_truth` column in batch.xlsx, or an extra key in a YAML
    sidecar) and reference it as a `{field_name}` placeholder in both
    prompt templates (`_render_prompt`) and judge templates
    (`_render_judge_prompt`), same plain-substitution mechanism as
    `{context}`.

    Returns:
        Dict with keys "models" (list[str]), "prompts" (list[str]),
        "context" (str, empty if absent), and "extra_fields"
        (dict[str, str], empty if none present).
    """
    models = _normalize_list(data.get("llm")) or [
        os.environ.get("LLM_DEFAULT_MODEL", "glm-4-7-flash")
    ]
    prompts = _normalize_list(data.get("prompt")) or [DEFAULT_PROMPT]
    context = str(data.get("context") or "")
    extra_fields = {
        str(k): str(v)
        for k, v in data.items()
        if k not in _RESERVED_SIDECAR_FIELDS and v is not None
    }
    return {"models": models, "prompts": prompts, "context": context, "extra_fields": extra_fields}


def _parse_sidecar(text: str) -> dict:
    """Parse a sidecar YAML's content into normalized model/prompt lists + context.

    Args:
        text: Raw YAML file contents.

    Returns:
        See `_sidecar_from_dict`.
    """
    return _sidecar_from_dict(yaml.safe_load(text) or {})
