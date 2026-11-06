"""Course endpoint settings for command-line tools; TABot still uses harness config."""

from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.parse import urlsplit

import requests

LOCAL_CONFIG = Path(__file__).resolve().parent / "course_endpoint.local.json"


def load_settings(endpoint=None, model=None, path=LOCAL_CONFIG):
    """Resolve CLI > environment > local JSON settings without contacting a server."""
    saved = {}
    if Path(path).exists():
        try:
            saved = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError("Cannot read course endpoint configuration") from exc
        if not isinstance(saved, dict):
            raise ValueError("Course endpoint configuration must be a JSON object")
    endpoint = endpoint or os.environ.get("COURSE_ENDPOINT") or saved.get("endpoint")
    model = model or os.environ.get("COURSE_MODEL") or saved.get("model")
    if not isinstance(endpoint, str) or not endpoint.strip():
        raise ValueError("No course endpoint configured. Supply --endpoint with the staff-provided URL "
                         "or run python scripts/configure_endpoint.py --endpoint <course URL>.")
    endpoint = endpoint.strip().rstrip("/")
    parsed = urlsplit(endpoint)
    if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.query or parsed.fragment:
        raise ValueError("The course endpoint must be an HTTP(S) base URL without a query or fragment")
    if model is not None and (not isinstance(model, str) or not model.strip()):
        raise ValueError("The served model name must be a non-empty string")
    return endpoint, model.strip() if model else None


def list_models(endpoint, timeout=10):
    """Read served model IDs from the configured course server's OpenAI-compatible API."""
    base = endpoint.rstrip("/")
    if not base.endswith("/v1"):
        base += "/v1"
    try:
        response = requests.get(base + "/models", timeout=timeout)
        response.raise_for_status()
        payload = response.json()
        models = list(dict.fromkeys(item["id"] for item in payload["data"]
                                   if isinstance(item["id"], str) and item["id"].strip()))
        if not models:
            raise ValueError("No served model IDs returned")
    except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
        raise TimeoutError("Could not list models from the course endpoint. "
                           "Check the URL and whether the course server is running and reachable.") from exc
    return models


def resolve_settings(endpoint=None, model=None, timeout=10):
    endpoint, model = load_settings(endpoint, model)
    if model is None:
        models = list_models(endpoint, timeout)
        if len(models) != 1:
            raise ValueError("The server has multiple models; select --model from: " + ", ".join(models))
        model = models[0]
    return endpoint, model
