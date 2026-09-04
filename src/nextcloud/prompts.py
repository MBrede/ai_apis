"""
Prompt template loading/rendering, including structured-output (.yaml/.yml)
support — see README_llm.md's "Structured output" section for the schema.
"""

import logging
from pathlib import Path

import yaml
from webdav3.client import Client
from webdav3.exceptions import RemoteResourceNotFound

from src.nextcloud.constants import LOCAL_PROMPTS_DIR, PROMPTS_SUBFOLDER
from src.nextcloud.webdav_utils import _download_text

logger = logging.getLogger(__name__)


def _load_prompt_template(
    client: Client,
    remote_root: str,
    prompt_name: str,
    tmp_dir: Path,
    cache: dict[str, dict],
) -> dict:
    """Load a prompt template, preferring the Nextcloud ``prompts/`` folder.

    Templates are edited by uploading a `.md` file to
    ``<remote_root>/prompts/<prompt_name>.md`` — no image rebuild or redeploy
    needed. A `.yaml`/`.yml` file instead (checked first) opts the prompt
    into structured output: it must have a `prompt:` key (the template text,
    same placeholders as a plain `.md`) and a `fields:` key (a
    `{field_name: description}` mapping — one output field per entry, each
    a plain string). Falls back to the `.md` template bundled in the image
    (under the script's local ``prompts/`` folder) only if nothing remote
    matches, so a fresh deployment still has a working default — bundled
    defaults are always plain-text, never structured. Results are cached
    per run since many jobs typically share the same prompt.

    Args:
        client: WebDAV client.
        remote_root: Root remote folder this job's sidecar came from (prompts
            live at `<remote_root>/prompts/`) — with multiple NEXTCLOUD_FOLDER
            entries, each folder's own prompts/ is checked independently, so
            the cache key must include it (see `cache`).
        prompt_name: Stem of a `.md`/`.yaml`/`.yml` template file.
        tmp_dir: Local scratch directory for the temporary download.
        cache: Dict reused across calls within a single run, keyed
            `"<remote_root>:<prompt_name>"` — the same prompt name can
            resolve to a different template per folder.

    Returns:
        `{"template": str, "fields": dict[str, str] | None}` — `fields` is
        `None` for a plain `.md` prompt (unstructured, unchanged behavior).

    Raises:
        FileNotFoundError: If no matching template exists remotely or locally.
        ValueError: If a `.yaml`/`.yml` prompt is missing `prompt:` or `fields:`.
    """
    cache_key = f"{remote_root}:{prompt_name}"
    if cache_key in cache:
        return cache[cache_key]

    base = f"{remote_root.rstrip('/')}/{PROMPTS_SUBFOLDER}/{prompt_name}"
    for ext in (".yaml", ".yml"):
        try:
            text = _download_text(client, base + ext, tmp_dir)
        except RemoteResourceNotFound:
            continue
        data = yaml.safe_load(text) or {}
        if "prompt" not in data:
            raise ValueError(f"Prompt '{prompt_name}{ext}' is missing required 'prompt:' key.")
        if "fields" not in data or not isinstance(data["fields"], dict) or not data["fields"]:
            raise ValueError(
                f"Prompt '{prompt_name}{ext}' is missing required non-empty 'fields:' mapping "
                f"(a .yaml/.yml prompt always triggers structured output — use .md for plain text)."
            )
        logger.info("Loaded structured prompt '%s' from Nextcloud (%s).", prompt_name, remote_root)
        result = {
            "template": data["prompt"],
            "fields": {str(k): str(v) for k, v in data["fields"].items()},
        }
        cache[cache_key] = result
        return result

    remote_path = base + ".md"
    try:
        template = _download_text(client, remote_path, tmp_dir)
        logger.info("Loaded prompt '%s' from Nextcloud (%s).", prompt_name, remote_root)
    except RemoteResourceNotFound:
        local_path = LOCAL_PROMPTS_DIR / f"{prompt_name}.md"
        if not local_path.exists():
            raise FileNotFoundError(
                f"Prompt '{prompt_name}' not found in Nextcloud ({remote_path}, "
                f"{base}.yaml, {base}.yml) or bundled defaults ({local_path})."
            )
        logger.info("Prompt '%s' not in Nextcloud, using bundled default.", prompt_name)
        template = local_path.read_text(encoding="utf-8")

    result = {"template": template, "fields": None}
    cache[cache_key] = result
    return result


def _render_prompt(
    template: str, transcript: str, context: str = "", extra_fields: dict[str, str] | None = None
) -> str:
    """Render a prompt template with the transcript text and optional context/extra fields.

    Args:
        template: Raw template text (from `_load_prompt_template`).
        transcript: Plain-text transcript content.
        context: Sidecar-provided `context:` text, injected via a `{context}`
            placeholder (same plain-string-replace mechanism as
            `{transcript}`). Templates that don't reference `{context}` are
            unaffected — empty string is the default, matching every sidecar
            that omits the field.
        extra_fields: Any non-schema sidecar/batch-row fields (see
            `_sidecar_from_dict`), each substituted as its own
            `{field_name}` placeholder. A template that doesn't reference a
            given field name is unaffected.

    Returns:
        Rendered prompt text.
    """
    rendered = template.replace("{transcript}", transcript).replace("{context}", context)
    for key, value in (extra_fields or {}).items():
        rendered = rendered.replace("{" + key + "}", value)
    return rendered


def _json_schema_for_fields(fields: dict[str, str]) -> dict:
    """Build a JSON schema (vLLM/OpenAI `response_format: json_schema` shape)
    from a prompt's `fields:` mapping — every field is a required string
    property, described by its mapping value.

    Args:
        fields: `{field_name: description}`, from `_load_prompt_template`.

    Returns:
        A `response_format` payload value.
    """
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "structured_output",
            "schema": {
                "type": "object",
                "properties": {
                    name: {"type": "string", "description": desc} for name, desc in fields.items()
                },
                "required": list(fields.keys()),
            },
        },
    }


def _render_prompt(
    template: str, transcript: str, context: str = "", extra_fields: dict[str, str] | None = None
) -> str:
    """Render a prompt template with the transcript text and optional context/extra fields.

    Args:
        template: Raw template text (from `_load_prompt_template`).
        transcript: Plain-text transcript content.
        context: Sidecar-provided `context:` text, injected via a `{context}`
            placeholder (same plain-string-replace mechanism as
            `{transcript}`). Templates that don't reference `{context}` are
            unaffected — empty string is the default, matching every sidecar
            that omits the field.
        extra_fields: Any non-schema sidecar/batch-row fields (see
            `_sidecar_from_dict`), each substituted as its own
            `{field_name}` placeholder. A template that doesn't reference a
            given field name is unaffected.

    Returns:
        Rendered prompt text.
    """
    rendered = template.replace("{transcript}", transcript).replace("{context}", context)
    for key, value in (extra_fields or {}).items():
        rendered = rendered.replace("{" + key + "}", value)
    return rendered
