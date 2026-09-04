"""
Chat-completion call to the in-cluster KubeAI endpoint, plain text or
structured (JSON schema) output.
"""

import json
import logging
import os

import aiohttp

logger = logging.getLogger(__name__)


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


async def _call_llm(
    session: aiohttp.ClientSession, model: str, prompt: str, fields: dict[str, str] | None = None
) -> str | dict[str, str] | None:
    """Send a chat-completion request to the in-cluster KubeAI endpoint.

    Args:
        session: Shared aiohttp session.
        model: KubeAI model name.
        prompt: Fully rendered prompt text.
        fields: When given (a structured/`.yaml` prompt — see
            `_load_prompt_template`), requests structured JSON output
            matching these fields via `response_format` and parses the
            reply into a `{field_name: value}` dict instead of raw text.

    Returns:
        The assistant's reply text (or parsed structured dict, if `fields`
        was given), or `None` on failure — including a structured reply
        that wasn't valid JSON, or didn't parse as an object.
    """
    llm_url = os.environ.get("LLM_URL", "http://kubeai.llm.svc.cluster.local/openai/v1")
    endpoint = f"{llm_url.rstrip('/')}/chat/completions"
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}]}
    if fields:
        payload["response_format"] = _json_schema_for_fields(fields)
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
            # inline as `<think>...</think>` instead of a separate "reasoning"
            # field, depending on vLLM's reasoning-parser config. Strip it.
            if "</think>" in content:
                content = content.rsplit("</think>", 1)[1]
            content = content.strip()
            if not fields:
                return content
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                logger.error(
                    "Structured output from '%s' was not valid JSON: %s", model, content[:300]
                )
                return None
            if not isinstance(parsed, dict):
                logger.error(
                    "Structured output from '%s' was not a JSON object: %s", model, content[:300]
                )
                return None
            # Found live 2026-09-03: glm-4-7-flash sometimes returns JSON
            # null for a field with nothing to report, despite the schema
            # declaring it a required string — str(None) would otherwise
            # save the literal text "None" instead of leaving it blank.
            return {
                name: "" if parsed.get(name) is None else str(parsed.get(name)) for name in fields
            }
    except Exception as exc:
        # asyncio.TimeoutError (and a few other exceptions) have an empty
        # str() — include the type name so timeouts are distinguishable
        # from other failures in the logs.
        logger.error("Error calling LLM: %s: %s", type(exc).__name__, exc)
        return None
