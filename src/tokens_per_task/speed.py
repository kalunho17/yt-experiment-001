"""Single-stream speed probe: one streamed request at a time, no competing load.

Splits time to first token (TTFT) from decode speed. Written to results/speed.csv.
"""
from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

from .client import NO_PROXY_OPENER, Server
from .common import now_iso
from .config import CONFIGS, DEFAULT_SEED, request_settings
from .results_io import SPEED_FIELDS, CsvAppender, read_rows

SPEED_RUNS = 3
SPEED_MAX_TOKENS = 800
SPEED_PROMPT = ("Write a plain-English explanation, about 400 words, of why the price of a technology "
                "falls as more of it is produced. Use short paragraphs and no lists.")
# Each profile is probed with its no-thinking condition, so the stream is mostly answer tokens.
SPEED_CONFIG = {"text": "Q-off", "text-qwen35-4b": "S", "text-gemma4": "G-off", "text-gptoss": "O-low"}


def stream_once(server: Server, model: str, cfg: dict) -> dict:
    """One streamed request. Usage comes from the final chunk (stream_options.include_usage)."""
    payload = {"model": model, "messages": [{"role": "user", "content": SPEED_PROMPT}],
               "max_tokens": SPEED_MAX_TOKENS, "seed": DEFAULT_SEED, "stream": True,
               "stream_options": {"include_usage": True}}
    payload.update(request_settings(cfg))
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    if server.api_key:
        headers["Authorization"] = f"Bearer {server.api_key}"
    req = urllib.request.Request(f"{server.base}/chat/completions", data=json.dumps(payload).encode("utf-8"),
                                 headers=headers, method="POST")
    t0 = time.perf_counter()
    t_first = None
    usage, finish, r_chars = {}, "", 0
    with NO_PROXY_OPENER.open(req, timeout=server.timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                break
            try:
                chunk = json.loads(body)
            except ValueError:
                continue
            if chunk.get("usage"):
                usage = chunk["usage"]
            for ch in chunk.get("choices") or []:
                d = ch.get("delta") or {}
                r = d.get("reasoning") or d.get("reasoning_content") or ""
                c = d.get("content") or ""
                r_chars += len(r)
                if (r or c) and t_first is None:
                    t_first = time.perf_counter()
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
    t_end = time.perf_counter()
    comp = int(usage.get("completion_tokens") or 0)
    ttft = (t_first - t0) if t_first else None
    decode = (comp - 1) / (t_end - t_first) if (t_first and comp > 1 and t_end > t_first) else None
    return {"prompt_tokens": usage.get("prompt_tokens", ""), "completion_tokens": comp,
            "reasoning_chars": r_chars, "ttft_s": round(ttft, 3) if ttft is not None else "",
            "total_s": round(t_end - t0, 3), "decode_tok_s": round(decode, 1) if decode else "",
            "end_to_end_tok_s": round(comp / (t_end - t0), 1) if comp else "", "finish_reason": finish}


def speed_probe(server: Server, model: str, profile: str, results: Path):
    """SPEED_RUNS streamed requests into results/speed.csv; skipped if already recorded."""
    cfg_name = SPEED_CONFIG.get(profile)
    if not cfg_name:
        return
    path = results / "speed.csv"
    done = [r for r in read_rows(path) if r.get("model") == model]
    if len(done) >= SPEED_RUNS:
        print(f"  speed probe for {model} already recorded ({len(done)} runs)")
        return
    out = CsvAppender(path, SPEED_FIELDS)
    print(f"  speed probe ({cfg_name}, concurrency 1, {SPEED_RUNS} streamed runs):")
    try:
        for i in range(SPEED_RUNS):
            try:
                r = stream_once(server, model, CONFIGS[cfg_name])
            except Exception as e:  # noqa: BLE001
                print(f"    run {i + 1}: failed ({e}); speed probe skipped")
                return
            out.write([{"timestamp": now_iso(), "profile": profile, "model": model, "config": cfg_name,
                        "run": i + 1, **r}])
            print(f"    run {i + 1}: {r['completion_tokens']} tokens, TTFT {r['ttft_s']} s, "
                  f"decode {r['decode_tok_s']} tok/s, end-to-end {r['end_to_end_tok_s']} tok/s")
    finally:
        out.close()
