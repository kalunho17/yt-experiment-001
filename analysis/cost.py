"""Cost per task and per *successful* task from the measured token counts, at prices you choose.

    python analysis/cost.py --input-price 0.20 --output-price 0.80
    python analysis/cost.py --prices my_prices.json --currency EUR

Prices are per 1 million tokens. ``--prices`` takes a JSON object mapping a model label
(as in the tables: "Qwen3.5-4B", "Qwen3.8-27B", "Gemma 4 31B", "gpt-oss-20b") or a condition
id ("Q-off", ...) to ``[input_price, output_price]``; unlisted conditions fall back to
``--input-price`` / ``--output-price``. No prices ship with this repository: list prices
change, and this measurement is about token counts.

Note: pricing another provider's model with these counts is a projection; models tokenise
and reason differently.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from tokens_per_task.config import CONFIGS, MODEL_OF, SETTING_OF  # noqa: E402
from tokens_per_task.results_io import read_rows  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-price", type=float, help="price per 1M prompt tokens")
    ap.add_argument("--output-price", type=float, help="price per 1M completion tokens")
    ap.add_argument("--prices", help="JSON file: {model label or condition: [input, output]}")
    ap.add_argument("--currency", default="")
    ap.add_argument("--results", default=str(HERE.parent / "results"))
    a = ap.parse_args()
    table = json.loads(Path(a.prices).read_text(encoding="utf-8")) if a.prices else {}
    if not table and (a.input_price is None or a.output_price is None):
        ap.error("give --input-price and --output-price, or --prices")

    items = [r for r in read_rows(Path(a.results) / "items.csv") if r["seed"] == "42"]
    cur = f" {a.currency}" if a.currency else ""
    print(f"{'Task':<5}{'Model':<13}{'Setting':<24}{'Acc':>6}{'per task':>14}{'per successful task':>22}"
          f"{'per 1,000 successes':>22}")
    for t in ["T1", "T2", "T4", "T3"]:
        for c in CONFIGS:
            rs = [r for r in items if r["config"] == c and r["task"] == t]
            if not rs:
                continue
            price = table.get(c) or table.get(MODEL_OF[c]) or [a.input_price, a.output_price]
            if None in price:
                continue
            p_in, p_out = price
            cost = sum(int(r["prompt_tokens"]) * p_in + int(r["completion_tokens"]) * p_out for r in rs) / 1e6
            ok = sum(int(r["correct"]) for r in rs)
            per_ok = cost / ok if ok else float("nan")
            print(f"{t:<5}{MODEL_OF[c]:<13}{SETTING_OF[c]:<24}{ok / len(rs):>6.0%}"
                  f"{cost / len(rs):>14.6f}{per_ok:>22.6f}{1000 * per_ok:>22.4f}{cur}")
        print()


if __name__ == "__main__":
    main()
