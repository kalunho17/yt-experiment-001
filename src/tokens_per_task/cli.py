"""Command-line interface.

    python run.py check                                   # no GPU: data, answer keys, tools
    python run.py list-configs
    python run.py run --config Q-off,Q-low --task T1,T2   # against a vLLM you serve yourself
    python run.py summarise                               # results/summary.json from items.csv
    python run.py plan [--smoke-only]                     # whole experiment via a GPU-host conductor

See docs/reproduce.md for the full procedure.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

from . import config as C
from .client import Server
from .common import die, parse_list
from .conductor import GpuSampler, Keepalive, conductor_from_env, ensure_profile, release_gpu
from .datasets import DataError, build_items, gsm8k_key, load_banking77, load_gsm8k, load_math500_hard
from .ledger import QUARTERS, build_ledger, cents_of, execute_tool, tool_get_invoices, verify_question_with_tools
from .provenance import record_setup
from .prompts import t1_prompt
from .results_io import CALL_FIELDS, ITEM_FIELDS, CsvAppender, LayoutError, read_rows
from .runner import RequestContext, run_batch
from .scoring import parse_number, score_numeric
from .speed import speed_probe
from .summary import print_summary, sanity_warnings, summarise_rows, write_summary

# Windows consoles may not print every character of the datasets; never crash on that.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace", line_buffering=True)
    except Exception:  # noqa: BLE001
        pass


# =============================================================================
# Helpers
# =============================================================================
def results_dir(args, smoke: bool = False) -> Path:
    if args.results_dir:
        return Path(args.results_dir)
    return C.DEFAULT_RESULTS / "smoke" if smoke else C.DEFAULT_RESULTS


def resolve_pairs(config_arg, task_arg) -> list[tuple[str, str]]:
    configs = parse_list(config_arg)
    if not configs:
        die("say which config(s) to run, e.g. --config Q-off,Q-low  (python run.py list-configs)")
    unknown = [c for c in configs if c not in C.CONFIGS]
    if unknown:
        die(f"unknown config(s) {unknown}. Known: {', '.join(C.CONFIGS)}")
    tasks = parse_list(task_arg)
    bad = [t for t in tasks if t not in C.N_ITEMS]
    if bad:
        die(f"unknown task(s) {bad}. Known: {', '.join(C.N_ITEMS)}")
    pairs = []
    for c in configs:
        own = C.tasks_of(c)
        for t in (tasks or own):
            if t not in own:
                die(f"config {c} runs {'/'.join(own)}, not {t}. "
                    f"{'T3 needs an agent config (Q-A, Q-A-low).' if t == 'T3' else 'Agent configs run T3 only.'}")
            pairs.append((c, t))
    return pairs


def default_pairs(configs: list[str]) -> list[tuple[str, str]]:
    return [(c, t) for c in configs for t in C.tasks_of(c)]


def make_server(args) -> Server:
    return Server(args.base_url, args.api_key, args.timeout, args.max_model_len)


def pick_model(server: Server, configs: list[str], args) -> str:
    """Find the served model id and check it matches every requested condition."""
    try:
        models = server.models()
    except Exception as e:  # noqa: BLE001
        die(f"no server answering at {server.base} ({e}).\nStart vLLM first (docs/reproduce.md) or pass --base-url.")
    ids = [m.get("id", "") for m in models]
    if not ids:
        die(f"the server at {server.base} lists no models.")
    for m in models:
        if m.get("max_model_len"):
            server.max_model_len = int(m["max_model_len"])
            break
    if args.model:
        if args.model not in ids:
            print(f"  WARNING: --model {args.model} is not in the served list {ids}; using it anyway.")
        return args.model
    if args.any_model:
        return ids[0]
    for c in configs:
        want = C.CONFIGS[c]["model_match"].lower()
        if not any(want in i.lower() for i in ids):
            die(f"config {c} expects a model matching '{C.CONFIGS[c]['model_match']}', but the server serves "
                f"{ids}.\nServe it with --served-model-name {C.CONFIGS[c]['model_match']}, or pass --any-model.")
    want = C.CONFIGS[configs[0]]["model_match"].lower()
    return next(i for i in ids if want in i.lower())


def connect(configs: list[str], args) -> tuple[Server, str]:
    server = make_server(args)
    model = pick_model(server, configs, args)
    # per-model concurrency cap (PROFILES), applied to the user's --concurrency
    if not hasattr(args, "_base_concurrency"):
        args._base_concurrency = args.concurrency
    cap = C.PROFILES.get(C.CONFIGS[configs[0]]["profile"], {}).get("max_concurrency")
    args.concurrency = min(args._base_concurrency, cap) if cap else args._base_concurrency
    # Warm-up: the first request after a start pays a one-off JIT compile (~19 s on Qwen3.5-4B).
    # One throwaway request so no timed item carries it.
    t0 = time.perf_counter()
    try:
        server.chat({"model": model, "messages": [{"role": "user", "content": "Say OK."}],
                     "max_tokens": 16, "temperature": 0.0})
        print(f"  warm-up request: {time.perf_counter() - t0:.1f} s (not recorded)")
    except Exception as e:  # noqa: BLE001
        print(f"  WARNING: warm-up request failed ({e}); the first timed item may be slow.")
    if args.max_tokens > server.max_model_len - 1024:
        new = server.max_model_len - 1024
        print(f"  NOTE: the server's context is {server.max_model_len}; max_tokens lowered from "
              f"{args.max_tokens} to {new}.")
        args.max_tokens = new
    return server, model


def run_pairs(server: Server, model: str, pairs, results: Path, args, smoke: int,
              gpu: GpuSampler | None = None) -> tuple[int, list[str]]:
    """Run (condition, task) pairs into ``results``; resumes by skipping recorded items.
    Returns (items not recorded, CHECK messages)."""
    results.mkdir(parents=True, exist_ok=True)
    mode = f"smoke {smoke} items/task" if smoke else "full run"
    print(f"\nServer {server.base}  model {model}  context {server.max_model_len}  max_tokens {args.max_tokens}  "
          f"seed {args.seed}  concurrency {args.concurrency}  {mode}")
    print(f"Results -> {results}")
    done = {(r["config"], r["task"], r["item_id"], r["seed"]) for r in read_rows(results / "items.csv")}
    try:
        calls_out = CsvAppender(results / "calls.csv", CALL_FIELDS)
        items_out = CsvAppender(results / "items.csv", ITEM_FIELDS)
    except LayoutError as e:
        die(str(e))
    skipped_total = 0
    try:
        for config, task in pairs:
            items = build_items(task)
            if smoke:
                items = items[:smoke]
            todo = [it for it in items if (config, task, it["item_id"], str(args.seed)) not in done]
            print(f"\n== {config} / {task} {C.TASK_NAMES[task]}: {len(items)} items, "
                  f"{len(items) - len(todo)} already done, {len(todo)} to run")
            if not todo:
                continue
            if gpu:
                gpu.tag.update(config=config, task=task)
            ctx = RequestContext(server, config, task, model, args.max_tokens, args.seed)
            skipped_total += run_batch(ctx, todo, calls_out, items_out, args.concurrency)
    finally:
        if gpu:
            gpu.tag.update(config="", task="")
        calls_out.close()
        items_out.close()

    print_summary(summarise_rows(read_rows(results / "items.csv")), only=pairs, seed=str(args.seed))
    checks = sanity_warnings(results / "items.csv", results / "calls.csv", pairs, str(args.seed))
    if skipped_total:
        print(f"\n{skipped_total} item(s) hit server errors and were NOT recorded. "
              f"Run the same command again to retry just those.")
    return skipped_total, checks


# =============================================================================
# Commands
# =============================================================================
def cmd_run(args):
    pairs = resolve_pairs(args.config, args.task)
    configs = list(dict.fromkeys(c for c, _ in pairs))
    results = results_dir(args, smoke=bool(args.smoke))
    cond = conductor_from_env(args.conductor_url, required=bool(args.activate or args.gpu_log))
    keep = gpu = None
    if args.activate:
        profiles = sorted({C.CONFIGS[c]["profile"] for c in configs})
        if len(profiles) != 1:
            die(f"--activate needs configs of one profile (one served model); these need {profiles}.")
        ensure_profile(cond, profiles[0], make_server(args), args.ready_timeout)
    if cond and (args.activate or args.gpu_log):
        keep = Keepalive(cond)
        keep.start()
    try:
        server, model = connect(configs, args)
        if args.gpu_log:
            gpu = GpuSampler(cond, results / "gpu.csv", args.gpu_interval)
            gpu.start()
        run_pairs(server, model, pairs, results, args, smoke=args.smoke, gpu=gpu)
    finally:
        if gpu:
            gpu.stop()
        if keep:
            keep.stop()
    print(f"\nDone. Files in {results}")


def cmd_plan(args):
    """The whole experiment, one served model at a time: activate, smoke test, checks, full runs."""
    cond = conductor_from_env(args.conductor_url, required=True)
    results = results_dir(args)
    smoke_n = args.smoke or 5
    profiles = parse_list(args.profile) or [p for p in C.PLAN_ORDER
                                           if args.include_optional or not C.PROFILES[p]["optional"]]
    bad = [p for p in profiles if p not in C.PROFILES]
    if bad:
        die(f"unknown profile(s) {bad}. Known: {', '.join(C.PROFILES)}")
    print(f"PLAN: profiles {', '.join(profiles)}; smoke {smoke_n} items per task"
          + ("; smoke only" if args.smoke_only else "; then full runs") + f"; results in {results}")
    keep = Keepalive(cond)
    keep.start()
    finished = False
    try:
        for prof in profiles:
            configs = [c for c in C.CONFIGS if C.CONFIGS[c]["profile"] == prof]
            pairs = default_pairs(configs)
            bar = "=" * 78
            print(f"\n{bar}\nPROFILE {prof}: {C.PROFILES[prof]['serve']}  configs {', '.join(configs)}\n{bar}")
            ensure_profile(cond, prof, make_server(args), args.ready_timeout)
            server, model = connect(configs, args)
            record_setup(server, cond, results, args.max_tokens, args.seed, args.concurrency, skip_if_recorded=True)

            smoke_dir = results / "smoke" / prof
            _, checks = run_pairs(server, model, pairs, smoke_dir, args, smoke=smoke_n)
            if checks and not args.ignore_checks:
                failed = results / "smoke" / f"{prof}-failed-{dt.datetime.now():%Y%m%d-%H%M%S}"
                smoke_dir.rename(failed)
                die(f"the smoke test of {prof} raised the CHECK warnings above. The plan stops here.\n"
                    f"Its smoke results were moved to {failed} for inspection. Fix the cause and run "
                    f"`python run.py plan` again (finished work is kept). To go on despite the warnings, "
                    f"add --ignore-checks.", code=3)
            print(f"\nSmoke test of {prof} passed.")
            if args.smoke_only:
                continue
            speed_probe(server, model, prof, results)
            gpu = None if args.no_gpu_log else GpuSampler(cond, results / "gpu.csv", args.gpu_interval)
            if gpu:
                gpu.start()
            try:
                skipped, _ = run_pairs(server, model, pairs, results, args, smoke=0, gpu=gpu)
            finally:
                if gpu:
                    gpu.stop()
            if skipped:
                die(f"{skipped} item(s) of {prof} hit server errors and were not recorded. "
                    f"Run `python run.py plan` again to retry them (finished work is kept).")
        finished = True
    finally:
        keep.stop()
        if not args.keep_running:
            release_gpu(cond)
        else:
            print("\n--keep-running: the profile stays up (the conductor stops it after 30 idle minutes).")
    if args.smoke_only:
        for prof in profiles:
            print(f"\n{prof}:")
            print_summary(summarise_rows(read_rows(results / "smoke" / prof / "items.csv")))
        print(f"\nSMOKE STAGE PASSED for {', '.join(profiles)}. Look at the replies in "
              f"{results / 'smoke'}, then run `python run.py plan` for the full runs.")
    elif finished:
        write_summary(results)
        print("\nPLAN FINISHED.")


def cmd_summarise(args):
    try:
        write_summary(results_dir(args, smoke=bool(args.smoke)))
    except FileNotFoundError as e:
        die(str(e))


def cmd_record_setup(args):
    record_setup(make_server(args), conductor_from_env(args.conductor_url, required=False), results_dir(args),
                 args.max_tokens, args.seed, args.concurrency)


def cmd_release(args):
    release_gpu(conductor_from_env(args.conductor_url, required=True))


def cmd_list_configs(_args):
    print(f"{'config':<9} {'profile':<15} {'tasks':<9} {'model_match':<22} request settings")
    for name, c in C.CONFIGS.items():
        print(f"{name:<9} {c['profile']:<15} {','.join(C.tasks_of(name)):<9} {c['model_match']:<22} "
              f"{json.dumps(C.request_settings(c))}")


def cmd_check(args):
    """No server needed: download data, check every answer key, exercise the tools."""
    print("CHECK: no server needed.\n")
    rows, labels = load_banking77()
    items = build_items("T1")
    assert len(items) == 100 and len({i["item_id"] for i in items}) == 100
    assert all(i["expected"] in labels for i in items)
    print(f"T1 Banking77: {len(rows)} test rows, {len(labels)} labels, sampled {len(items)} "
          f"(seed {C.SAMPLE_SEED}), {len({i['expected'] for i in items})} distinct labels in the sample")
    if args.verbose:
        for k, it in enumerate(items[:3]):
            print(f"\n--- T1 example {k + 1}: {it['item_id']}  expected: {it['expected']}")
            print(it["prompt"] if k == 0 else t1_prompt(it["text"], ["[... the same 77 labels ...]"]))

    gsm = load_gsm8k()
    items = build_items("T2")
    assert len(items) == 100
    bad = [r["idx"] for r in gsm if parse_number(gsm8k_key(r["answer"])) is None]
    assert not bad, f"GSM8K keys that do not parse: {bad[:10]}"
    for it in items:
        assert score_numeric(f"Answer: {it['expected']}", it["expected"])[0]
    print(f"T2 GSM8K: {len(gsm)} test rows, all keys parse as numbers; sampled {len(items)}")
    if args.verbose:
        for k, it in enumerate(items[:2]):
            print(f"\n--- T2 example {k + 1}: {it['item_id']}  expected: {it['expected']}\n{it['prompt']}")

    hard = load_math500_hard()
    items = build_items("T4")
    assert len(items) == 50 and len({i["item_id"] for i in items}) == 50
    for it in items:
        assert score_numeric(f"Answer: {it['expected']}", it["expected"])[0]
    print(f"T4 MATH-500 level 5, integer answers: {len(hard)} problems; sampled {len(items)}")
    if args.verbose:
        for k, it in enumerate(items[:2]):
            print(f"\n--- T4 example {k + 1}: {it['item_id']} ({it['subject']})  expected: {it['expected']}\n{it['prompt']}")

    ledger = build_ledger()
    inv = ledger["invoices"]
    assert len(inv) == 500 and len(ledger["suppliers"]) == 12
    per_sq = {(s, q): len(cents_of(ledger, s, [q])) for s in ledger["suppliers"] for q in QUARTERS}
    print(f"T3 ledger: {len(inv)} invoices, {len(ledger['suppliers'])} suppliers, "
          f"EUR {sum(i['amount_cents'] for i in inv) / 100:,.2f} in total; invoices per supplier and quarter "
          f"{min(per_sq.values())}-{max(per_sq.values())}")
    qs = ledger["questions"]
    assert len(qs) == 20
    print(f"T3 questions ({len(qs)}), each key recomputed from the tool outputs only:")
    for q in qs:
        ok, v, n_calls = verify_question_with_tools(ledger, q)
        lo, hi = q["expected_tool_calls"]
        assert ok, f"{q['id']}: key {q['answer']} but recomputed {v}"
        assert 2 <= n_calls <= 6, f"{q['id']} needs {n_calls} tool calls"
        print(f"  {q['id']} [{q['kind']:<11}] key={q['answer']!s:<32} OK  ({n_calls} tool calls; agent needs ~{lo}-{hi})")
        if args.verbose:
            print(f"        {q['question']}")

    s0 = ledger["suppliers"][0]
    for name, a, should_fail in [
            ("get_invoices", json.dumps({"supplier": s0.upper(), "quarter": "2025-Q1"}), False),
            ("get_invoices", json.dumps({"supplier": "Acme Ltd", "quarter": "Q1"}), True),
            ("calculator", json.dumps({"expression": "__import__('os').system('dir')"}), True),
            ("calculator", json.dumps({"expression": "2 ** 10000000"}), True),
            ("calculator", "not json", True),
            ("delete_everything", "{}", True)]:
        res = execute_tool(ledger, name, a)
        assert ("error" in json.loads(res)) == should_fail, (name, a, res)
    sizes = [len(tool_get_invoices(ledger, s, q)) for s in ledger["suppliers"] for q in QUARTERS]
    print(f"T3 tools: robust to malformed calls; get_invoices result size {min(sizes)}-{max(sizes)} characters")

    C.GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    (C.GENERATED_DIR / "t3_ledger.json").write_text(json.dumps(
        {"suppliers": ledger["suppliers"], "invoices": ledger["invoices"]}, indent=1, ensure_ascii=False),
        encoding="utf-8")
    (C.GENERATED_DIR / "t3_questions.json").write_text(json.dumps(qs, indent=1, ensure_ascii=False),
                                                       encoding="utf-8")
    print(f"Wrote {C.GENERATED_DIR / 't3_ledger.json'} and t3_questions.json")

    print("\nRequest body per config (messages abbreviated):")
    for name in parse_list(args.config) or C.CONFIGS:
        if name not in C.CONFIGS:
            die(f"unknown config {name}")
        ctx = RequestContext(None, name, C.tasks_of(name)[0], "<served model>", args.max_tokens, args.seed)
        p = ctx.payload([{"role": "user", "content": "..."}], tools=C.CONFIGS[name]["agent"])
        if "tools" in p:
            p["tools"] = [t["function"]["name"] for t in p["tools"]]
        print(f"  {name:<8} {json.dumps(p)}")
    print("\nCHECK PASSED.")


# =============================================================================
# Argument parsing
# =============================================================================
def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--base-url", default=os.environ.get("VLLM_BASE_URL", C.DEFAULT_BASE_URL),
                        help=f"OpenAI-compatible endpoint (default {C.DEFAULT_BASE_URL})")
    common.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY", ""), help="if vLLM runs with --api-key")
    common.add_argument("--model", help="served model id (default: detected from /v1/models)")
    common.add_argument("--any-model", action="store_true", help="skip the check that the served model fits the config")
    common.add_argument("--max-tokens", type=int, default=C.DEFAULT_MAX_TOKENS)
    common.add_argument("--max-model-len", type=int, default=C.DEFAULT_MAX_MODEL_LEN,
                        help="context length, if the server does not report it")
    common.add_argument("--seed", type=int, default=C.DEFAULT_SEED,
                        help="request seed (the item sample is always seed 42)")
    common.add_argument("--concurrency", type=int, default=C.DEFAULT_CONCURRENCY)
    common.add_argument("--timeout", type=float, default=3600, help="seconds per request")
    common.add_argument("--smoke", type=int, default=0, metavar="N",
                        help="only the first N items per task (run: results go to results/smoke/)")
    common.add_argument("--results-dir", help="default: results/")
    g = common.add_argument_group("GPU host (optional conductor, see conductor.py)")
    g.add_argument("--conductor-url", help="default: CONDUCTOR_URL from the environment or .env")
    g.add_argument("--ready-timeout", type=float, default=C.CONDUCTOR_READY_TIMEOUT_S,
                   help="seconds to wait for a model to come up")
    g.add_argument("--gpu-interval", type=float, default=2.0, help="seconds between GPU samples")

    ap = argparse.ArgumentParser(prog="run.py", description="Tokens-per-task measurement harness. "
                                 "See docs/reproduce.md.")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("check", parents=[common], help="no server: data, answer keys, tools, request bodies")
    p.add_argument("--config", nargs="+")
    p.add_argument("-v", "--verbose", action="store_true", help="print example prompts")
    p.set_defaults(func=cmd_check)

    sub.add_parser("list-configs", parents=[common], help="conditions and their request settings").set_defaults(
        func=cmd_list_configs)

    p = sub.add_parser("run", parents=[common], help="run conditions against a served model (resumable)")
    p.add_argument("--config", nargs="+", required=True, help="e.g. Q-off,Q-low (see list-configs)")
    p.add_argument("--task", nargs="+", help="e.g. T1,T2 (default: the config's own tasks)")
    p.add_argument("--summarise", action="store_true", help="write summary.json afterwards")
    p.add_argument("--activate", action="store_true", help="start the configs' profile via the conductor first")
    p.add_argument("--gpu-log", action="store_true", help="log GPU util/VRAM from the conductor into gpu.csv")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("plan", parents=[common], help="whole experiment via the conductor: per model "
                       "activate, smoke test, checks, speed probe, full runs; then summary")
    p.add_argument("--smoke-only", action="store_true", help="stop after the smoke stage")
    p.add_argument("--include-optional", action="store_true", help="also run gpt-oss-20b")
    p.add_argument("--profile", nargs="+", help="only these profiles, e.g. text text-gemma4")
    p.add_argument("--ignore-checks", action="store_true", help="go on despite CHECK warnings")
    p.add_argument("--keep-running", action="store_true", help="leave the last model up at the end")
    p.add_argument("--no-gpu-log", action="store_true", help="do not write gpu.csv")
    p.set_defaults(func=cmd_plan)

    sub.add_parser("summarise", aliases=["summarize"], parents=[common],
                   help="write summary.json from items.csv").set_defaults(func=cmd_summarise)
    sub.add_parser("record-setup", parents=[common], help="append the served model's setup to setup.md").set_defaults(
        func=cmd_record_setup)
    sub.add_parser("release", parents=[common], help="free the GPU host (conductor profile none)").set_defaults(
        func=cmd_release)
    return ap


def main(argv=None):
    if sys.version_info < (3, 11):
        die("Python 3.11 or newer is needed.")
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except DataError as e:
        die(str(e))
    if getattr(args, "summarise", False) and args.command == "run":
        cmd_summarise(args)


if __name__ == "__main__":
    main()
