"""Aggregation of items.csv / calls.csv / speed.csv into summary.json, plus sanity checks."""
from __future__ import annotations

import datetime as dt
import json
import statistics
from pathlib import Path

from .common import now_iso
from .config import CONFIGS
from .results_io import read_rows


def _num(v) -> float | None:
    try:
        return float(v) if v not in ("", None) else None
    except ValueError:
        return None


def summarise_rows(rows: list[dict]) -> list[dict]:
    """Per (condition, task, seed): accuracy and token statistics. Means are per item."""
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault((r["config"], r["task"], r["seed"]), []).append(r)
    order = {c: i for i, c in enumerate(CONFIGS)}
    out = []
    for (config, task, seed), rs in sorted(groups.items(), key=lambda kv: (order.get(kv[0][0], 99), kv[0][1], kv[0][2])):
        n = len(rs)
        tot = [_num(r["total_tokens"]) for r in rs]
        reas = [x for x in (_num(r["reasoning_tokens"]) for r in rs) if x is not None]
        n_ok = sum(int(r["correct"]) for r in rs)
        mean = lambda xs: round(statistics.fmean(xs), 2) if xs else None  # noqa: E731
        out.append({
            "config": config, "task": task, "seed": int(seed) if str(seed).isdigit() else seed,
            "models": sorted({r["model"] for r in rs}),
            "n": n, "n_correct": n_ok, "accuracy": round(n_ok / n, 4),
            "mean_total_tokens": mean(tot), "median_total_tokens": statistics.median(tot),
            "mean_prompt_tokens": mean([_num(r["prompt_tokens"]) for r in rs]),
            "mean_completion_tokens": mean([_num(r["completion_tokens"]) for r in rs]),
            "mean_reasoning_tokens": mean(reas), "n_items_with_reasoning_count": len(reas),
            "mean_calls": mean([_num(r["n_calls"]) for r in rs]),
            "mean_tool_calls": mean([_num(r["n_tool_calls"]) for r in rs]),
            "truncation_rate": round(sum(int(r["truncated"]) for r in rs) / n, 4),
            "mean_latency_s": mean([_num(r["latency_s"]) for r in rs]),
            "total_tokens_per_correct_item": round(sum(tot) / n_ok, 2) if n_ok else None,
        })
    return out


def print_summary(summary: list[dict], only=None, seed=None):
    rows = [s for s in summary if (only is None or (s["config"], s["task"]) in only)
            and (seed is None or str(s["seed"]) == seed)]
    if not rows:
        return
    print("\nconfig    task  seed    n   acc  mean_tok  median_tok  mean_reason  calls  trunc  latency_s  tok/correct")
    for s in rows:
        f = lambda v, w, p=0: (f"{v:>{w}.{p}f}" if isinstance(v, (int, float)) else f"{'-':>{w}}")  # noqa: E731
        print(f"{s['config']:<9} {s['task']:<4} {s['seed']!s:>4} {s['n']:>4} {s['accuracy']:>5.2f} "
              f"{f(s['mean_total_tokens'], 9)} {f(s['median_total_tokens'], 11)} {f(s['mean_reasoning_tokens'], 12)} "
              f"{f(s['mean_calls'], 6, 1)} {s['truncation_rate']:>6.2f} {f(s['mean_latency_s'], 10, 1)} "
              f"{f(s['total_tokens_per_correct_item'], 12)}")


def sanity_warnings(items_path: Path, calls_path: Path, pairs, seed: str) -> list[str]:
    """Checks that the reasoning switch, the reasoning parser and tool calling work as intended."""
    items = [r for r in read_rows(items_path) if (r["config"], r["task"]) in pairs and r["seed"] == seed]
    calls = [r for r in read_rows(calls_path) if (r["config"], r["task"]) in pairs and r["seed"] == seed]
    found = []
    for config, task in dict.fromkeys(pairs):
        rs = [r for r in items if r["config"] == config and r["task"] == task]
        cs = [r for r in calls if r["config"] == config and r["task"] == task]
        if not rs:
            continue
        reas = [int(r["reasoning_tokens"]) for r in cs if r["reasoning_tokens"] not in ("", None)]
        srcs = {r["reasoning_source"] for r in cs}
        msgs = []
        if CONFIGS[config]["thinking"] and reas and max(reas) == 0:
            msgs.append("thinking is ON for this config but no reasoning came back. Either the vLLM service has no "
                        "--reasoning-parser (the thinking would then sit inside the answer text and in "
                        "completion_tokens) or the setting did not reach the chat template.")
        if not CONFIGS[config]["thinking"] and reas and max(reas) > 0:
            msgs.append("thinking is OFF for this config but reasoning tokens came back: the off switch did not "
                        "reach the chat template. Check chat_template_kwargs in CONFIGS.")
        if any("content-split" in s for s in srcs):
            msgs.append("reasoning was found inside the answer text (<think>...</think>): the vLLM service "
                        "needs --reasoning-parser so it is returned separately.")
        if task == "T3" and sum(int(r["n_tool_calls"]) for r in rs) == 0:
            msgs.append("no tool calls at all: the vLLM service needs "
                        "--enable-auto-tool-choice --tool-call-parser <parser for the model>.")
        if task == "T3" and any(r["note"] == "unparsed_tool_call" for r in rs):
            msgs.append("tool calls came back as text: the --tool-call-parser does not match the model.")
        nl = sum(1 for r in rs if r["note"] == "no_answer_line")
        if nl > len(rs) / 2:
            msgs.append(f"{nl} of {len(rs)} replies had no 'Answer:' line; look at the replies before the full run.")
        for m in msgs:
            found.append(f"{config}/{task}: {m}")
            print(f"  CHECK {config}/{task}: {m}")
    if not found:
        print("\nSanity checks passed (reasoning on/off as expected, tool calls present where needed).")
    return found


def speed_from_calls(calls_path: Path) -> list[dict]:
    """In-run speed per condition x task from calls.csv.

    per-request tok/s = completion tokens / request latency (prefill and queueing included);
    aggregate tok/s   = all completion tokens / wall time from the first request's start to the
                        last request's end.
    """
    rows = read_rows(calls_path)
    groups: dict = {}
    for r in rows:
        groups.setdefault((r["config"], r["task"]), []).append(r)
    out = []
    for (cfg, task), rs in sorted(groups.items()):
        per_req, comp_sum, starts, ends = [], 0, [], []
        for r in rs:
            try:
                lat, comp = float(r["latency_s"]), int(r["completion_tokens"])
                end = dt.datetime.fromisoformat(r["timestamp"]).timestamp()
            except (ValueError, KeyError):
                continue
            comp_sum += comp
            if lat > 0:
                per_req.append(comp / lat)
            ends.append(end)
            starts.append(end - lat)
        wall = (max(ends) - min(starts)) if ends else 0
        per_req.sort()
        out.append({"config": cfg, "task": task, "calls": len(rs),
                    "per_request_tok_s_mean": round(sum(per_req) / len(per_req), 1) if per_req else None,
                    "per_request_tok_s_median": round(per_req[len(per_req) // 2], 1) if per_req else None,
                    "aggregate_tok_s": round(comp_sum / wall, 1) if wall > 0 else None,
                    "completion_tokens": comp_sum, "wall_s": round(wall, 1),
                    "note": "timestamps have 1 s resolution; concurrency per PROFILES.max_concurrency"})
    return out


def read_speed(path: Path) -> list[dict]:
    """Mean of the single-stream speed probe per model (speed.csv)."""
    by: dict = {}
    for r in read_rows(path):
        by.setdefault(r["model"], []).append(r)
    out = []
    for model, rs in by.items():
        def mean(k):
            v = [float(r[k]) for r in rs if r.get(k) not in ("", None)]
            return round(sum(v) / len(v), 2) if v else None
        out.append({"model": model, "profile": rs[0]["profile"], "config": rs[0]["config"], "runs": len(rs),
                    "ttft_s": mean("ttft_s"), "decode_tok_s": mean("decode_tok_s"),
                    "end_to_end_tok_s": mean("end_to_end_tok_s"), "completion_tokens": mean("completion_tokens")})
    return out


def write_summary(results: Path) -> Path:
    rows = read_rows(results / "items.csv")
    if not rows:
        raise FileNotFoundError(f"no items in {results / 'items.csv'} yet.")
    summary = summarise_rows(rows)
    out = {"generated": now_iso(), "source": "items.csv", "harness": "tokens_per_task",
           "notes": ["total tokens = prompt + completion tokens summed over all calls of an item",
                     "reasoning tokens are part of completion tokens",
                     "truncated items (finish_reason 'length' or context exceeded) count as failed",
                     "means are per item"],
           "by_config_task": summary,
           "speed_in_run": speed_from_calls(results / "calls.csv"),
           "speed_single_stream": read_speed(results / "speed.csv")}
    path = results / "summary.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print_summary(summary)
    print(f"\nWrote {path}")
    return path
