"""Running items: one chat call, one item (single answer or agent loop), one batch in parallel.

Token accounting: an item's ``total_tokens`` is the sum of ``prompt_tokens + completion_tokens``
over *every* call made for it. An agent re-sends its whole, growing conversation at every
step, so its prompt tokens accumulate; this is the cost a per-token API bills.
"""
from __future__ import annotations

import json
import os
import re
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

from .client import ContextExceeded, FatalRequestError, Server, TransientError
from .common import abort, now_iso
from .config import AGENT_MAX_STEPS, CONFIGS, request_settings
from .ledger import TOOLS, execute_tool
from .scoring import answer_segment, score_numeric, score_t1, score_t3


class RequestContext:
    """Everything needed to send requests for one (condition, task) pair."""

    def __init__(self, server: Server | None, config: str, task: str, model: str, max_tokens: int, seed: int):
        self.server, self.config, self.task, self.model = server, config, task, model
        self.max_tokens, self.seed = max_tokens, seed
        self.cfg = CONFIGS[config]

    def payload(self, messages: list[dict], tools: bool) -> dict:
        p = {"model": self.model, "messages": messages, "max_tokens": self.max_tokens,
             "seed": self.seed, "stream": False}
        p.update(request_settings(self.cfg))
        if tools:
            p["tools"] = TOOLS
            p["tool_choice"] = "auto"
        return p


def split_reasoning(msg: dict) -> tuple[str, str, str]:
    """Returns (answer content, reasoning text, where the reasoning was found)."""
    content = msg.get("content") or ""
    if isinstance(content, list):   # rare: content parts
        content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
    if not reasoning and "</think>" in content:
        reasoning, content = content.split("</think>", 1)
        reasoning = reasoning.replace("<think>", "")
        return content.strip(), reasoning.strip(), "content-split"
    return content, reasoning, ""


def one_call(ctx: RequestContext, messages: list[dict], call_index: int, item_id: str, tools: bool) -> dict:
    payload = ctx.payload(messages, tools)
    try:
        resp, latency, used_max = ctx.server.chat(payload)
    except ContextExceeded:
        return {"context_exceeded": True, "row": None}
    choice = (resp.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    content, reasoning, src = split_reasoning(msg)
    usage = resp.get("usage") or {}
    details = usage.get("completion_tokens_details") or {}
    if not reasoning:
        r_tok, r_src = 0, "empty"
    elif details.get("reasoning_tokens"):
        r_tok, r_src = int(details["reasoning_tokens"]), "usage"
    else:
        r_tok = ctx.server.count_tokens(ctx.model, reasoning)
        r_src = ("tokenize" if r_tok is not None else "unavailable") + (f"+{src}" if src else "")
    tool_calls = msg.get("tool_calls") or []
    row = {"timestamp": now_iso(), "config": ctx.config, "task": ctx.task, "item_id": item_id,
           "seed": ctx.seed, "model": ctx.model, "call_index": call_index,
           "prompt_tokens": int(usage.get("prompt_tokens") or 0),
           "completion_tokens": int(usage.get("completion_tokens") or 0),
           "reasoning_tokens": "" if r_tok is None else r_tok, "reasoning_source": r_src,
           "latency_s": round(latency, 3), "finish_reason": choice.get("finish_reason") or "",
           "n_tool_calls": len(tool_calls), "max_tokens": used_max}
    return {"context_exceeded": False, "row": row, "content": content, "tool_calls": tool_calls,
            "finish_reason": row["finish_reason"]}


def _agent_loop(ctx: RequestContext, item: dict, calls: list[dict]) -> tuple[str, bool, str]:
    """ReAct-style loop with native tool calling. Returns (final content, truncated, note)."""
    ledger = item["ledger"]
    messages: list[dict] = [{"role": "user", "content": item["prompt"]}]
    for step in range(AGENT_MAX_STEPS):
        r = one_call(ctx, messages, step, item["item_id"], tools=True)
        if r["context_exceeded"]:
            return "", True, "context_exceeded"
        calls.append(r["row"])
        if r["finish_reason"] == "length":
            return "", True, "truncated"
        if not r["tool_calls"]:
            return r["content"], False, ""
        tcs = []
        for i, tc in enumerate(r["tool_calls"]):
            fn = tc.get("function") or {}
            args = fn.get("arguments")
            if not isinstance(args, str):
                args = json.dumps(args or {})
            tcs.append({"id": tc.get("id") or f"call_{step}_{i}", "type": "function",
                        "function": {"name": fn.get("name") or "", "arguments": args}})
        messages.append({"role": "assistant", "content": r["content"] or "", "tool_calls": tcs})
        for tc in tcs:
            result = execute_tool(ledger, tc["function"]["name"], tc["function"]["arguments"])
            messages.append({"role": "tool", "tool_call_id": tc["id"], "name": tc["function"]["name"],
                             "content": result})
    return "", False, "step_limit"


def run_item(ctx: RequestContext, item: dict) -> tuple[dict, list[dict]]:
    """Run one item to completion and score it. Returns (item row, call rows)."""
    calls: list[dict] = []
    truncated, note, content = False, "", ""
    if ctx.task in ("T1", "T2", "T4"):
        r = one_call(ctx, [{"role": "user", "content": item["prompt"]}], 0, item["item_id"], tools=False)
        if r["context_exceeded"]:
            truncated, note = True, "context_exceeded"
        else:
            calls.append(r["row"])
            content = r["content"]
            if r["finish_reason"] == "length":
                truncated, note = True, "truncated"
    else:
        content, truncated, note = _agent_loop(ctx, item, calls)

    if truncated or note == "step_limit":
        correct, pred = False, ""
    elif ctx.task == "T1":
        correct, pred = score_t1(content, item["expected"])
    elif ctx.task in ("T2", "T4"):
        correct, pred = score_numeric(content, item["expected"])
    else:
        correct, pred = score_t3(content, item["question"], item["ledger"]["suppliers"])
    if not note:
        if ctx.task == "T3" and ("<tool_call>" in content or "<function=" in content):
            note = "unparsed_tool_call"
        elif ctx.task in ("T2", "T3", "T4") and answer_segment(content) is None:
            note = "no_answer_line"

    def total(field):
        return sum(int(c[field]) for c in calls)

    r_vals = [c["reasoning_tokens"] for c in calls]
    item_row = {
        "timestamp": now_iso(), "config": ctx.config, "task": ctx.task, "item_id": item["item_id"],
        "seed": ctx.seed, "model": ctx.model, "n_calls": len(calls),
        "prompt_tokens": total("prompt_tokens"), "completion_tokens": total("completion_tokens"),
        "total_tokens": total("prompt_tokens") + total("completion_tokens"),
        "reasoning_tokens": "" if any(v == "" for v in r_vals) else sum(int(v) for v in r_vals),
        "n_tool_calls": total("n_tool_calls"), "latency_s": round(sum(float(c["latency_s"]) for c in calls), 3),
        "correct": int(bool(correct)), "truncated": int(truncated),
        "expected": str(item["expected"]), "predicted": re.sub(r"\s+", " ", str(pred))[:200], "note": note,
    }
    return item_row, calls


def run_batch(ctx: RequestContext, todo: list[dict], calls_out, items_out, concurrency: int) -> int:
    """Run items in parallel, writing each finished item at once. Returns the number of items
    that hit server errors and were not recorded (a rerun retries exactly those)."""
    n, n_ok, n_trunc, toks, skipped, consecutive_fail = 0, 0, 0, 0, 0, 0
    t0 = time.time()
    pool = ThreadPoolExecutor(max_workers=max(1, concurrency))
    futures = {pool.submit(run_item, ctx, it): it for it in todo}
    try:
        for fut in as_completed(futures):
            it = futures[fut]
            try:
                item_row, call_rows = fut.result()
            except FatalRequestError as e:
                pool.shutdown(wait=False, cancel_futures=True)
                abort(f"the server rejected the request for {ctx.config}/{ctx.task}. Nothing more is sent.\n{e}\n"
                      "Check that the right model is served (docs/reproduce.md) and the CONFIGS table in "
                      "src/tokens_per_task/config.py.")
            except TransientError as e:
                skipped += 1
                consecutive_fail += 1
                print(f"  server error on {it['item_id']} (not recorded): {e}")
            except Exception:  # noqa: BLE001
                skipped += 1
                consecutive_fail += 1
                print(f"  unexpected error on {it['item_id']} (not recorded):")
                traceback.print_exc()
            else:
                consecutive_fail = 0
                calls_out.write(call_rows)
                items_out.write([item_row])
                n += 1
                n_ok += item_row["correct"]
                n_trunc += item_row["truncated"]
                toks += item_row["total_tokens"]
                print(f"  [{ctx.config} {ctx.task}] {n}/{len(todo)}  {it['item_id']}  "
                      f"{'OK  ' if item_row['correct'] else 'FAIL'}  tokens {item_row['total_tokens']:>6}  "
                      f"calls {item_row['n_calls']}  {item_row['note']}  | acc {n_ok / n:.2f}  "
                      f"mean tokens {toks / n:.0f}  truncated {n_trunc}  {time.time() - t0:.0f}s")
            if consecutive_fail >= 5:
                pool.shutdown(wait=False, cancel_futures=True)
                abort("five items in a row failed with server errors; stopping. Is the server still up? "
                      "Completed items are saved; rerun the same command to continue.")
    except KeyboardInterrupt:
        pool.shutdown(wait=False, cancel_futures=True)
        calls_out.close()
        items_out.close()
        print("\nInterrupted. Completed items are saved; rerun the same command to continue.")
        os._exit(130)
    pool.shutdown(wait=True)
    return skipped
