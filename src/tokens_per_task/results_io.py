"""Result files: column layouts and an append-only, crash-safe CSV writer.

Column definitions are documented in results/README.md.
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

CALL_FIELDS = ["timestamp", "config", "task", "item_id", "seed", "model", "call_index",
               "prompt_tokens", "completion_tokens", "reasoning_tokens", "reasoning_source",
               "latency_s", "finish_reason", "n_tool_calls", "max_tokens"]
ITEM_FIELDS = ["timestamp", "config", "task", "item_id", "seed", "model", "n_calls",
               "prompt_tokens", "completion_tokens", "total_tokens", "reasoning_tokens",
               "n_tool_calls", "latency_s", "correct", "truncated", "expected", "predicted", "note"]
GPU_FIELDS = ["timestamp", "elapsed_s", "config", "task", "gpu", "util_pct", "vram_used_mb", "power_w"]
SPEED_FIELDS = ["timestamp", "profile", "model", "config", "run", "prompt_tokens", "completion_tokens",
                "reasoning_chars", "ttft_s", "total_s", "decode_tok_s", "end_to_end_tok_s", "finish_reason"]


class LayoutError(RuntimeError):
    pass


class CsvAppender:
    """Appends rows and fsyncs after every write, so a crash never loses a finished item."""

    def __init__(self, path: Path, fields: list[str]):
        self.path, self.fields = path, fields
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.stat().st_size > 0:
            with path.open(encoding="utf-8", newline="") as f:
                header = next(csv.reader(f), [])
            if header != fields:
                raise LayoutError(f"{path} has a different column layout (an older harness version?). "
                                  "Rename or move it away and run again.")
            self.f = path.open("a", encoding="utf-8", newline="")
            self.w = csv.DictWriter(self.f, fieldnames=fields)
        else:
            self.f = path.open("w", encoding="utf-8", newline="")
            self.w = csv.DictWriter(self.f, fieldnames=fields)
            self.w.writeheader()
        self.f.flush()

    def write(self, rows: list[dict]):
        for r in rows:
            self.w.writerow({k: r.get(k, "") for k in self.fields})
        self.f.flush()
        os.fsync(self.f.fileno())

    def close(self):
        self.f.close()


def write_rows(path: Path, fields: list[str], rows: list[dict]):
    """Rewrite a CSV file in full (used when rows are moved, never for appending)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))
