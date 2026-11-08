"""Uncertainty estimates (standard library only).

* Accuracy: Wilson score interval.
* Means and ratios of means: percentile bootstrap over items. All conditions answer the
  *same* items, so comparisons between two conditions resample item ids jointly (paired).
"""
from __future__ import annotations

import math
import random
import statistics

N_BOOT = 10_000
BOOT_SEED = 0


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def _pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    i = q * (len(xs) - 1)
    lo, hi = math.floor(i), math.ceil(i)
    return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)


def boot_mean(x: list[float], n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(x)
    means = [statistics.fmean(x[rng.randrange(n)] for _ in range(n)) for _ in range(n_boot)]
    return _pct(means, 0.025), _pct(means, 0.975)


def paired_boot(a: dict, b: dict, stat, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> tuple[float, float, float]:
    """``a``, ``b``: item_id -> value. ``stat(list_a, list_b)`` -> float. Returns (point, lo, hi)."""
    ids = sorted(set(a) & set(b))
    rng = random.Random(seed)
    point = stat([a[i] for i in ids], [b[i] for i in ids])
    vals = []
    for _ in range(n_boot):
        s = [ids[rng.randrange(len(ids))] for _ in ids]
        vals.append(stat([a[i] for i in s], [b[i] for i in s]))
    return point, _pct(vals, 0.025), _pct(vals, 0.975)


def ratio_of_means(xa: list[float], xb: list[float]) -> float:
    return statistics.fmean(xa) / statistics.fmean(xb)


def diff_of_means(xa: list[float], xb: list[float]) -> float:
    return statistics.fmean(xa) - statistics.fmean(xb)
