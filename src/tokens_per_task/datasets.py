"""Benchmark data: download (once), load, and draw the fixed item samples.

Sources and licences (see data/README.md):

* Banking77 test split, PolyAI (CC BY 4.0)
* GSM8K test split, OpenAI (MIT)
* MATH-500, HuggingFaceH4 subset of MATH (MIT); level 5 with integer answers only

Every sample is ``random.Random(SAMPLE_SEED).sample(rows, n)``, so all conditions see the
same items.
"""
from __future__ import annotations

import csv
import json
import os
import random
import re
import time
import urllib.request
from pathlib import Path

from .config import DATA_DIR, N_ITEMS, SAMPLE_SEED
from .ledger import build_ledger
from .prompts import t1_prompt, t2_prompt, t3_prompt, t4_prompt

GSM8K_URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
BANKING77_URL = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv"
MATH500_URL = "https://huggingface.co/datasets/HuggingFaceH4/MATH-500/resolve/main/test.jsonl"
GSM8K_FILE = DATA_DIR / "gsm8k_test.jsonl"
BANKING77_FILE = DATA_DIR / "banking77_test.csv"
MATH500_FILE = DATA_DIR / "math500_test.jsonl"


class DataError(RuntimeError):
    pass


def download(url: str, dest: Path, check) -> None:
    """Download ``url`` to ``dest`` unless it exists; ``check(text)`` validates the format."""
    if dest.exists() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    last = None
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "tokens-per-task/1.0"})
            with urllib.request.urlopen(req, timeout=120) as r:
                body = r.read()
            check(body.decode("utf-8"))
            tmp.write_bytes(body)
            os.replace(tmp, dest)
            print(f"  saved {dest.name} ({len(body) // 1024} KB)")
            return
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 * (attempt + 1))
    raise DataError(f"could not download {url}: {last}\nDownload it by hand and save it as {dest}")


def _check_gsm8k(text: str):
    first = json.loads(text.splitlines()[0])
    if "question" not in first or "####" not in first.get("answer", ""):
        raise ValueError("unexpected GSM8K format")


def _check_banking(text: str):
    if not text.lstrip("﻿").startswith("text,category"):
        raise ValueError("unexpected Banking77 header")


def _check_math500(text: str):
    first = json.loads(text.splitlines()[0])
    if not {"problem", "answer", "level"} <= set(first):
        raise ValueError("unexpected MATH-500 format")


def load_gsm8k() -> list[dict]:
    download(GSM8K_URL, GSM8K_FILE, _check_gsm8k)
    rows = []
    with GSM8K_FILE.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if line.strip():
                d = json.loads(line)
                rows.append({"idx": i, "question": d["question"], "answer": d["answer"]})
    return rows


def gsm8k_key(answer: str) -> str:
    """The number after '####' in a GSM8K solution."""
    return answer.split("####")[-1].strip().replace(",", "").replace("$", "")


def load_math500_hard() -> list[dict]:
    """Level-5 MATH-500 problems whose answer is a plain integer (74 problems)."""
    download(MATH500_URL, MATH500_FILE, _check_math500)
    rows = []
    with MATH500_FILE.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            d = json.loads(line)
            ans = str(d["answer"]).strip().replace(",", "")
            if d.get("level") == 5 and re.fullmatch(r"-?\d+", ans):
                rows.append({"idx": i, "problem": d["problem"], "answer": ans,
                             "subject": d.get("subject", ""), "unique_id": d.get("unique_id", "")})
    return rows


def load_banking77() -> tuple[list[dict], list[str]]:
    download(BANKING77_URL, BANKING77_FILE, _check_banking)
    rows = []
    with BANKING77_FILE.open(encoding="utf-8-sig", newline="") as f:
        for i, r in enumerate(csv.DictReader(f)):
            rows.append({"idx": i, "text": r["text"].strip(), "label": r["category"].strip()})
    labels = sorted({r["label"] for r in rows})
    if len(labels) != 77:
        raise DataError(f"Banking77 has {len(labels)} labels, expected 77. Delete {BANKING77_FILE} and retry.")
    return rows, labels


def build_items(task: str) -> list[dict]:
    """Deterministic item sample for a task. Each item has item_id, prompt, expected (+ metadata)."""
    if task == "T1":
        rows, labels = load_banking77()
        sample = random.Random(SAMPLE_SEED).sample(rows, N_ITEMS["T1"])
        return [{"item_id": f"b77-{r['idx']:04d}", "prompt": t1_prompt(r["text"], labels),
                 "expected": r["label"], "labels": labels, "text": r["text"]} for r in sample]
    if task == "T2":
        rows = load_gsm8k()
        sample = random.Random(SAMPLE_SEED).sample(rows, N_ITEMS["T2"])
        return [{"item_id": f"gsm-{r['idx']:04d}", "prompt": t2_prompt(r["question"]),
                 "expected": gsm8k_key(r["answer"])} for r in sample]
    if task == "T4":
        rows = load_math500_hard()
        sample = random.Random(SAMPLE_SEED).sample(rows, N_ITEMS["T4"])
        return [{"item_id": f"math-{r['idx']:03d}", "prompt": t4_prompt(r["problem"]),
                 "expected": r["answer"], "subject": r["subject"]} for r in sample]
    if task == "T3":
        ledger = build_ledger()
        return [{"item_id": q["id"], "prompt": t3_prompt(q["question"]), "expected": q["answer"],
                 "question": q, "ledger": ledger} for q in ledger["questions"]]
    raise ValueError(task)
