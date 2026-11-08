"""The published results must be consistent with this code: same items, same aggregates."""
import json

import pytest

from tokens_per_task.config import CONFIGS, DEFAULT_RESULTS, N_ITEMS, tasks_of
from tokens_per_task.datasets import build_items
from tokens_per_task.results_io import read_rows
from tokens_per_task.summary import summarise_rows

ITEMS = read_rows(DEFAULT_RESULTS / "items.csv")
CALLS = read_rows(DEFAULT_RESULTS / "calls.csv")


def test_every_condition_task_cell_is_complete():
    cells = {}
    for r in ITEMS:
        cells.setdefault((r["config"], r["task"]), []).append(r["item_id"])
    expected = {(c, t) for c in CONFIGS for t in tasks_of(c)}
    assert set(cells) == expected
    for (c, t), ids in cells.items():
        assert len(ids) == len(set(ids)) == N_ITEMS[t], (c, t)


@pytest.mark.parametrize("task", ["T1", "T2", "T3", "T4"])
def test_recorded_items_are_the_seeded_sample(task):
    sampled = {i["item_id"]: str(i["expected"]) for i in build_items(task)}
    for r in ITEMS:
        if r["task"] == task:
            assert r["item_id"] in sampled
            assert r["expected"] == sampled[r["item_id"]]


def test_item_totals_equal_sum_of_calls():
    by_item = {}
    for c in CALLS:
        k = (c["config"], c["task"], c["item_id"], c["seed"])
        p, q = by_item.get(k, (0, 0))
        by_item[k] = (p + int(c["prompt_tokens"]), q + int(c["completion_tokens"]))
    for r in ITEMS:
        p, q = by_item[(r["config"], r["task"], r["item_id"], r["seed"])]
        assert (int(r["prompt_tokens"]), int(r["completion_tokens"])) == (p, q)
        assert int(r["total_tokens"]) == p + q


def test_summary_json_recomputes_exactly():
    published = json.loads((DEFAULT_RESULTS / "summary.json").read_text(encoding="utf-8"))
    assert summarise_rows(ITEMS) == published["by_config_task"]
