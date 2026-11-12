#!/usr/bin/env python3
"""Tiny OpenAI-compatible mock of a vLLM server, for testing the harness without a GPU.

    python mock_server.py --port 8765 [--model qwen3.8-27b-nvfp4]
                          [--reasoning-field reasoning_content|reasoning]
                          [--truncate-every N] [--no-tokenize] [--always-think]

Endpoints: GET /v1/models, GET /version, POST /tokenize, POST /v1/chat/completions (also streamed).
Answers are plausible, not smart: T1 picks a label from the prompt, T2 answers a number,
T3 calls list_suppliers, get_invoices and calculator before answering. Thinking is "on"
unless chat_template_kwargs.enable_thinking is false, and then a reasoning text comes back.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ARGS = None
COUNTER = {"n": 0}
LOCK = threading.Lock()


def n_tokens(text: str) -> int:
    return max(1, len(re.findall(r"\w+|[^\w\s]", text or "")))


def thinking_on(body: dict) -> bool:
    kw = body.get("chat_template_kwargs") or {}
    if ARGS.always_think:
        return True
    if kw.get("enable_thinking") is False:
        return False
    return True


def reply_for(body: dict, rng: random.Random) -> dict:
    msgs = body["messages"]
    first = msgs[0]["content"]
    if body.get("tools"):
        tool_msgs = [m for m in msgs if m.get("role") == "tool"]
        if len(tool_msgs) == 0:
            return {"tool_calls": [{"id": "call_a", "type": "function",
                                    "function": {"name": "list_suppliers", "arguments": "{}"}}]}
        if len(tool_msgs) == 1:
            sups = json.loads(tool_msgs[0]["content"]).get("suppliers", ["?"])
            named = [s for s in sups if s in first] or sups[:1]
            return {"tool_calls": [{"id": f"call_b{i}", "type": "function",
                                    "function": {"name": "get_invoices",
                                                 "arguments": json.dumps({"supplier": s, "quarter": "Q1"})}}
                                   for i, s in enumerate(named[:2])]}
        if not any(m.get("name") == "calculator" for m in tool_msgs):
            amounts = []
            for m in tool_msgs:
                try:
                    amounts += [r[2] for r in json.loads(m["content"]).get("rows", [])]
                except Exception:  # noqa: BLE001
                    pass
            expr = " + ".join(str(a) for a in amounts) or "1 + 1"
            return {"tool_calls": [{"id": "call_c", "type": "function",
                                    "function": {"name": "calculator",
                                                 "arguments": json.dumps({"expression": expr})}}]}
        res = json.loads(tool_msgs[-1]["content"]).get("result", 0)
        return {"content": f"The calculator gives {res}.\nAnswer: {res}"}
    if "intent labels" in first:
        labels = [l for l in first.split("\n\n")[1].splitlines() if l.strip()]
        return {"content": rng.choice(labels)}
    nums = re.findall(r"\d+", first)
    guess = nums[-1] if nums and rng.random() < 0.5 else str(rng.randint(1, 100))
    return {"content": f"Let me work it out.\nAnswer: {guess}"}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def _send(self, code: int, obj: dict):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _stream(self, k: int, msg: dict, finish: str, prompt_tokens: int, completion: int):
        """SSE stream like vLLM: delta chunks, a finish chunk, then usage (stream_options.include_usage)."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()

        def chunk(obj):
            self.wfile.write(b"data: " + json.dumps(obj).encode() + b"\n\n")
            self.wfile.flush()

        base = {"id": f"chatcmpl-{k}", "object": "chat.completion.chunk", "created": int(time.time()),
                "model": ARGS.model}
        if msg.get(ARGS.reasoning_field):
            chunk({**base, "choices": [{"index": 0, "delta": {ARGS.reasoning_field: msg[ARGS.reasoning_field]}}]})
        for i, w in enumerate((msg.get("content") or "").split(" ")):
            time.sleep(0.005)
            chunk({**base, "choices": [{"index": 0, "delta": {"content": w if i == 0 else " " + w}}]})
        chunk({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]})
        chunk({**base, "choices": [], "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion,
                                                 "total_tokens": prompt_tokens + completion}})
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def do_GET(self):
        if self.path.rstrip("/") == "/v1/models":
            return self._send(200, {"object": "list", "data": [
                {"id": ARGS.model, "object": "model", "root": ARGS.model, "max_model_len": 32768}]})
        if self.path.rstrip("/") == "/version":
            return self._send(200, {"version": "0.0.0-mock"})
        self._send(404, {"error": "not found"})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        path = self.path.rstrip("/")
        if path == "/tokenize":
            if ARGS.no_tokenize:
                return self._send(404, {"detail": "Not Found"})
            n = n_tokens(body.get("prompt", ""))
            return self._send(200, {"count": n, "max_model_len": 32768, "tokens": list(range(n))})
        if path != "/v1/chat/completions":
            return self._send(404, {"detail": "Not Found"})
        if body.get("model") != ARGS.model:
            return self._send(404, {"object": "error", "message": f"The model `{body.get('model')}` does not exist."})
        kw = body.get("chat_template_kwargs") or {}
        if "reasoning_effort" in kw and kw["reasoning_effort"] not in ("low", "medium", "xhigh"):
            return self._send(400, {"object": "error", "message": "invalid reasoning_effort"})
        prompt_text = json.dumps(body["messages"]) + (json.dumps(body.get("tools")) if body.get("tools") else "")
        prompt_tokens = n_tokens(prompt_text)
        if prompt_tokens + body.get("max_tokens", 0) > 32768:
            return self._send(400, {"object": "error", "type": "BadRequestError", "message":
                                    f"'max_tokens' or 'max_completion_tokens' is too large: {body['max_tokens']}. "
                                    f"This model's maximum context length is 32768 tokens and your request has "
                                    f"{prompt_tokens} input tokens ({body['max_tokens']} > 32768 - {prompt_tokens})."})
        with LOCK:
            COUNTER["n"] += 1
            k = COUNTER["n"]
        rng = random.Random(k)
        time.sleep(rng.uniform(0.02, 0.1))
        r = reply_for(body, rng)
        reasoning = None
        if thinking_on(body):
            reasoning = "Okay, let me think about this. " * rng.randint(5, 40)
        content = r.get("content", "")
        finish = "tool_calls" if r.get("tool_calls") else "stop"
        if ARGS.truncate_every and k % ARGS.truncate_every == 0:
            finish, content = "length", ""
            r.pop("tool_calls", None)
        completion = n_tokens(content) + n_tokens(reasoning or "") + 10 * len(r.get("tool_calls") or [])
        msg = {"role": "assistant", "content": content or None, "tool_calls": r.get("tool_calls") or []}
        if reasoning is not None:
            msg[ARGS.reasoning_field] = reasoning
        if body.get("stream"):
            return self._stream(k, msg, finish, prompt_tokens, completion)
        self._send(200, {
            "id": f"chatcmpl-{k}", "object": "chat.completion", "created": int(time.time()), "model": ARGS.model,
            "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion,
                      "total_tokens": prompt_tokens + completion},
        })


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--model", default="qwen3.8-27b-nvfp4")
    ap.add_argument("--reasoning-field", default="reasoning_content", choices=["reasoning_content", "reasoning"])
    ap.add_argument("--truncate-every", type=int, default=0)
    ap.add_argument("--no-tokenize", action="store_true")
    ap.add_argument("--always-think", action="store_true", help="ignore enable_thinking=false (to test the CHECKs)")
    ARGS = ap.parse_args()
    srv = ThreadingHTTPServer(("127.0.0.1", ARGS.port), Handler)
    print(f"mock vLLM on http://127.0.0.1:{ARGS.port}/v1 serving {ARGS.model}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
