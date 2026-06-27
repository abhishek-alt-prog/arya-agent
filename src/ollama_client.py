"""
Ollama client — sends prompts to Gemma 4 running locally via Ollama.
"""
from __future__ import annotations

import json
import logging
from typing import Any

import requests

from .config import OLLAMA_BASE_URL, OLLAMA_MODEL

logger = logging.getLogger(__name__)


class OllamaClient:
    """Communicate with a local Ollama instance running Gemma 4."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
    ):
        self.base_url = (base_url or OLLAMA_BASE_URL).rstrip("/")
        self.model = model or OLLAMA_MODEL

    # ── Core generation ──────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> str:
        """Send a prompt and return the full text response."""
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if system:
            payload["system"] = system

        logger.debug("Ollama request → model=%s  tokens=%d", self.model, max_tokens)

        resp = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        result = resp.json()
        return result.get("response", "")

    def generate_json(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.4,
        max_tokens: int = 4096,
    ) -> dict | list:
        """
        Generate and parse JSON output from the model.
        Retries once if the first response isn't valid JSON.
        """
        for attempt in range(2):
            raw = self.generate(prompt, system, temperature, max_tokens)
            try:
                # Try to extract JSON from the response
                return self._extract_json(raw)
            except (json.JSONDecodeError, ValueError) as exc:
                if attempt == 0:
                    logger.warning(
                        "JSON parse failed (attempt 1), retrying. Error: %s", exc
                    )
                    # Add explicit JSON instruction
                    prompt = (
                        prompt
                        + "\n\nIMPORTANT: Respond ONLY with valid JSON. "
                        "No markdown, no explanation, just the JSON object/array."
                    )
                else:
                    logger.error("JSON parse failed after 2 attempts. Raw: %s", raw[:500])
                    raise

    # ── Health ───────────────────────────────────────────────────────

    def health_check(self) -> bool:
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return resp.status_code == 200
        except requests.ConnectionError:
            return False

    def is_model_available(self) -> bool:
        """Check if the configured model is pulled and available."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if resp.status_code != 200:
                return False
            models = resp.json().get("models", [])
            return any(m.get("name", "").startswith(self.model) for m in models)
        except requests.ConnectionError:
            return False

    # ── Helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _extract_json(text: str) -> dict | list:
        """Extract JSON from a response that may contain markdown fences."""
        text = text.strip()

        # Strip markdown code fences
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first line (```json or ```) and last line (```)
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines).strip()

        # Try parsing directly
        return json.loads(text)
