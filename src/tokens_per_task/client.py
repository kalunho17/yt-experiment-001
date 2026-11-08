"""Minimal client for an OpenAI-compatible vLLM server (standard library only)."""
from __future__ import annotations

import json
import re
import threading
import time
import urllib.error
import urllib.request

# The server runs on a local network: never route its traffic through a system proxy.
NO_PROXY_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class TransientError(Exception):
    """Server unreachable / 5xx / timeout after retries: the item is not recorded; a rerun retries it."""


class FatalRequestError(Exception):
    """The server rejected the request (4xx): a configuration problem; the run stops."""


class ContextExceeded(Exception):
    """The prompt alone no longer fits into the context window."""


def http_json(method: str, url: str, payload=None, timeout: float = 60, api_key: str = ""):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json", "Accept": "application/json"})
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    with NO_PROXY_OPENER.open(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def root_url(base: str) -> str:
    b = base.rstrip("/")
    return b[:-3] if b.endswith("/v1") else b


_CTX_RE = re.compile(r"context length|max_model_len|maximum context|too large|exceeds|too long", re.I)


class Server:
    """An OpenAI-compatible endpoint, e.g. ``http://localhost:8000/v1``."""

    def __init__(self, base_url: str, api_key: str, timeout: float, max_model_len: int):
        self.base = base_url.rstrip("/")
        self.root = root_url(self.base)
        self.api_key = api_key
        self.timeout = timeout
        self.max_model_len = max_model_len
        self._tok_url: str | None | bool = False   # False = not probed yet, None = unavailable
        self._tok_lock = threading.Lock()
        self.tokenize_warned = False

    def models(self) -> list[dict]:
        return http_json("GET", f"{self.base}/models", timeout=30, api_key=self.api_key).get("data", [])

    def version(self) -> str | None:
        for url in (f"{self.root}/version", f"{self.base}/version"):
            try:
                return str(http_json("GET", url, timeout=15, api_key=self.api_key).get("version"))
            except Exception:  # noqa: BLE001
                continue
        return None

    def count_tokens(self, model: str, text: str) -> int | None:
        """Tokens of ``text`` under the served model's tokenizer (vLLM ``POST /tokenize``).

        Used to count reasoning tokens when the server does not report them in ``usage``.
        """
        if not text:
            return 0
        with self._tok_lock:
            probe = self._tok_url is False
        candidates = [f"{self.root}/tokenize", f"{self.base}/tokenize"] if probe else [self._tok_url]
        for url in candidates:
            if not url:
                continue
            for attempt in range(3):
                try:
                    d = http_json("POST", url, {"model": model, "prompt": text, "add_special_tokens": False},
                                  timeout=60, api_key=self.api_key)
                    n = d.get("count")
                    if n is None and isinstance(d.get("tokens"), list):
                        n = len(d["tokens"])
                    if n is None:
                        break
                    with self._tok_lock:
                        self._tok_url = url
                    return int(n)
                except urllib.error.HTTPError as e:
                    if e.code in (404, 405):
                        break                   # endpoint missing here, try the next one
                    time.sleep(1 + attempt)
                except Exception:  # noqa: BLE001
                    time.sleep(1 + attempt)     # transient; retry
        if probe:
            with self._tok_lock:
                self._tok_url = None
                if not self.tokenize_warned:
                    self.tokenize_warned = True
                    print("  NOTE: the server has no /tokenize endpoint; reasoning_tokens are left empty "
                          "(unless the server reports them in usage).")
        return None

    def chat(self, payload: dict) -> tuple[dict, float, int]:
        """POST chat/completions with retries.

        If prompt + max_tokens exceed the context window, max_tokens is shrunk and the request
        retried. Returns (response, latency in s, max_tokens actually used).
        """
        payload = dict(payload)
        transient_tries, shrink_tries = 0, 0
        while True:
            t0 = time.perf_counter()
            try:
                resp = http_json("POST", f"{self.base}/chat/completions", payload,
                                 timeout=self.timeout, api_key=self.api_key)
                return resp, time.perf_counter() - t0, payload["max_tokens"]
            except urllib.error.HTTPError as e:
                body = ""
                try:
                    body = e.read().decode("utf-8", "replace")
                except Exception:  # noqa: BLE001
                    pass
                if e.code in (400, 413) and _CTX_RE.search(body) and shrink_tries < 6:
                    shrink_tries += 1
                    payload["max_tokens"] = self._shrunk_max_tokens(body, payload["max_tokens"])
                    if payload["max_tokens"] < 256:
                        raise ContextExceeded(body[:300])
                    continue
                if e.code in (408, 429) or e.code >= 500:
                    transient_tries += 1
                    if transient_tries > 3:
                        raise TransientError(f"HTTP {e.code}: {body[:300]}")
                    time.sleep(10 * transient_tries)
                    continue
                raise FatalRequestError(f"HTTP {e.code} from {self.base}/chat/completions: {body[:1500]}")
            except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as e:
                transient_tries += 1
                if transient_tries > 3:
                    raise TransientError(f"{type(e).__name__}: {e}")
                time.sleep(5 * transient_tries)

    def _shrunk_max_tokens(self, body: str, current: int) -> int:
        limit = self.max_model_len
        m = re.search(r"maximum context length is (\d+)", body)
        if m:
            limit = int(m.group(1))
        for pat in (r"(\d+) input tokens", r"\((\d+) in the messages", r"request has (\d+)",
                    r"prompt contains (\d+)", r"(\d+) tokens in the messages", r"input_tokens\D{0,5}(\d+)"):
            m = re.search(pat, body)
            if m:
                return min(current - 1, limit - int(m.group(1)) - 16)
        return current // 2
