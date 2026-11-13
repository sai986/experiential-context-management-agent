"""
Thin wrapper around the Anthropic API.

Usage:
    client = ModelClient()          # reads ANTHROPIC_API_KEY from environment
    response = client.complete("Summarise this observation: ...")

USE_LLM environment variable controls whether real API calls are made:
    USE_LLM=0  (default) — raises an error if called; prevents accidental billing
    USE_LLM=1            — makes real API calls

temperature is fixed at 0.0 for all calls to ensure reproducibility.
"""
from __future__ import annotations

import os
import time

from src.constants import MODEL_DEFAULT

# How many times to retry a failed API call before giving up.
_MAX_RETRIES = 3

# Starting wait between retries (doubles each attempt: 1s, 2s, 4s).
_RETRY_BASE_SECONDS = 1.0


class ModelClient:
    def __init__(self, model: str = MODEL_DEFAULT) -> None:
        self.model = model
        self._client = self._build_client()

    def _build_client(self):
        if os.environ.get("USE_LLM", "0") != "1":
            return None  # deferred — will raise on first use
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError(
                "anthropic package is not installed. Run: pip install anthropic"
            ) from exc
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY is not set. Add it to your .env file."
            )
        base_url = os.environ.get("ANTHROPIC_BASE_URL")
        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        client = anthropic.Anthropic(**kwargs)
        # Fail fast: send a minimal test call so a bad key/URL surfaces immediately
        # rather than silently falling back to the simulated agent mid-run.
        print(f"[model_client] USE_LLM=1 — testing connection to {base_url or 'api.anthropic.com'} ...")
        try:
            client.messages.create(
                model=self.model,
                max_tokens=5,
                temperature=0.0,
                messages=[{"role": "user", "content": "hi"}],
            )
            print("[model_client] Connection OK — real LLM calls enabled.")
        except Exception as exc:
            raise RuntimeError(
                f"[model_client] USE_LLM=1 but API connection failed: {exc}\n"
                "Check ANTHROPIC_API_KEY and ANTHROPIC_BASE_URL in your .env file.\n"
                "The experiment will NOT run in simulated mode — fix the connection first."
            ) from exc
        return client

    def complete(self, prompt: str) -> str:
        """Send a prompt and return the model's text response.

        Retries up to _MAX_RETRIES times with exponential backoff on API errors.
        """
        if self._client is None:
            raise RuntimeError(
                "ModelClient called with USE_LLM=0. "
                "Set USE_LLM=1 in your environment to enable real API calls."
            )
        last_error: Exception | None = None
        wait = _RETRY_BASE_SECONDS
        for attempt in range(_MAX_RETRIES):
            try:
                response = self._client.messages.create(
                    model=self.model,
                    max_tokens=512,
                    temperature=0.0,
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.content[0].text.strip()
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt < _MAX_RETRIES - 1:
                    time.sleep(wait)
                    wait *= 2
        raise RuntimeError(
            f"[model_client] API call failed after {_MAX_RETRIES} attempts: {last_error}"
        ) from last_error
