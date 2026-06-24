"""Cached, concurrent OpenAI chat wrapper shared by the teacher and judges.

Every request is content-addressed by (model, messages, sampling params,
response_format, seed) and persisted to a JSON cache so re-runs are cheap and
reproducible. The OpenAI client is imported lazily so this module (and the
cache-key logic) can be imported and unit-tested without the package or a key.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm


def request_cache_key(
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    response_format: dict[str, Any] | None,
    seed: int | None,
    base_url: str | None = None,
) -> str:
    payload = json.dumps(
        {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": response_format,
            "seed": seed,
            "base_url": base_url,  # so cache doesn't collide across providers
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class OpenAIChat:
    """Thread-safe cached chat client with bounded concurrency and retries."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        cache_path: str | Path | None = None,
        max_concurrency: int = 8,
        max_retries: int = 5,
        request_timeout: float = 120.0,
        base_url: str | None = None,
        api_version: str | None = None,
    ) -> None:
        self.model = model
        # base_url lets you target any OpenAI-compatible endpoint (e.g. Azure
        # OpenAI's /openai/v1 surface or an Azure AI Foundry inference endpoint).
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("AZURE_OPENAI_API_KEY")
        # Some Azure endpoints require an api-version query param; supply via env.
        self.api_version = api_version or os.environ.get("OPENAI_API_VERSION")
        self.cache_path = Path(cache_path) if cache_path else None
        self.max_concurrency = max_concurrency
        self.max_retries = max_retries
        self.request_timeout = request_timeout
        self._client = None
        self._lock = threading.Lock()
        self._cache: dict[str, str] = {}
        if self.cache_path and self.cache_path.exists():
            with self.cache_path.open(encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    self._cache[record["key"]] = record["response"]

    # -- internals -----------------------------------------------------------
    def _ensure_client(self):
        if self._client is None:
            from openai import OpenAI

            if not self.api_key:
                raise RuntimeError(
                    "No API key; set OPENAI_API_KEY / AZURE_OPENAI_API_KEY or pass api_key=."
                )
            kwargs: dict[str, Any] = {"api_key": self.api_key, "timeout": self.request_timeout}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            if self.api_version:
                kwargs["default_query"] = {"api-version": self.api_version}
            self._client = OpenAI(**kwargs)
        return self._client

    def _cache_get(self, key: str) -> str | None:
        with self._lock:
            return self._cache.get(key)

    def _cache_put(self, key: str, response: str) -> None:
        with self._lock:
            if key in self._cache:
                return
            self._cache[key] = response
            if self.cache_path:
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                with self.cache_path.open("a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps({"key": key, "response": response}) + "\n"
                    )

    def _call_with_retries(self, kwargs: dict[str, Any]) -> str:
        client = self._ensure_client()
        kwargs = dict(kwargs)
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                completion = client.chat.completions.create(**kwargs)
                return completion.choices[0].message.content or ""
            except Exception as error:  # noqa: BLE001 - retry on any API error
                last_error = error
                message = str(error).lower()
                # Adapt to model-family parameter differences (e.g. GPT-5 uses
                # max_completion_tokens and rejects a custom temperature).
                if "max_tokens" in message and "max_tokens" in kwargs:
                    kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
                    continue
                if "temperature" in message and "temperature" in kwargs:
                    kwargs.pop("temperature")
                    continue
                if "'seed'" in message and "seed" in kwargs:
                    kwargs.pop("seed")
                    continue
                # Exponential backoff; deterministic (no jitter) for reproducibility.
                time.sleep(min(2**attempt, 30))
        raise RuntimeError(
            f"OpenAI request failed after {self.max_retries} attempts: {last_error}"
        )

    # -- public API ----------------------------------------------------------
    def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 512,
        response_format: dict[str, Any] | None = None,
        seed: int | None = 0,
    ) -> str:
        key = request_cache_key(
            self.model, messages, temperature, max_tokens, response_format, seed, self.base_url
        )
        cached = self._cache_get(key)
        if cached is not None:
            return cached
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format is not None:
            kwargs["response_format"] = response_format
        if seed is not None:
            kwargs["seed"] = seed
        response = self._call_with_retries(kwargs)
        self._cache_put(key, response)
        return response

    def complete_many(
        self,
        message_lists: list[list[dict[str, str]]],
        temperature: float = 0.0,
        max_tokens: int = 512,
        response_format: dict[str, Any] | None = None,
        seed: int | None = 0,
        description: str = "OpenAI",
    ) -> list[str]:
        results: list[str | None] = [None] * len(message_lists)

        def worker(index: int) -> None:
            results[index] = self.complete(
                message_lists[index],
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
                seed=seed,
            )

        with ThreadPoolExecutor(max_workers=self.max_concurrency) as pool:
            list(
                tqdm(
                    pool.map(worker, range(len(message_lists))),
                    total=len(message_lists),
                    desc=description,
                    unit="req",
                )
            )
        return [result or "" for result in results]

    def judge_json(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 400,
        seed: int | None = 0,
    ) -> dict[str, Any]:
        """Return parsed JSON from a judge prompt. Falls back to an empty dict
        on unparseable output so a single bad judge response can't crash a run."""
        raw = self.complete(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
            seed=seed,
        )
        return parse_judge_json(raw)


def parse_judge_json(raw: str) -> dict[str, Any]:
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Tolerate fenced or prefixed output by extracting the first {...} block.
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(raw[start : end + 1])
            except json.JSONDecodeError:
                return {}
        return {}
