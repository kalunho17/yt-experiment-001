"""Figures of the paper, from the raw result files.

    pip install matplotlib
    python analysis/make_figures.py [--results results] [--out figures]

Fig. 1  Accuracy vs tokens per item, one panel per single-answer task. Colour encodes whether
        the model reasons in a separate channel (off / on); model identity is a direct label.
Fig. 2  Where an agent's tokens go: per-step contribution to tokens per item (waterfall),
        next to a single-answer task on the same weights.
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))

from stats import boot_mean, wilson  # noqa: E402
from tokens_per_task.config import CONFIGS  # noqa: E402
from tokens_per_task.results_io import read_rows  # noqa: E402

# Validated reference palette (dataviz skill): slots 1-3 pass all-pairs CVD checks.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8c8b86"
GRID = "#e6e5e0"
BLUE = "#2a78d6"      # slot 1
ORANGE = "#eb6834"    # slot 2
AQUA = "#1baf7a"      # slot 3

FAMILIES = [  # (label, conditions in the order the path is drawn)
    ("Qwen3.5-4B", ["S", "S-on"]),
    ("Qwen3.8-27B", ["Q-off", "Q-low", "Q-med", "Q-xhigh"]),
    ("Gemma 4 31B", ["G-off", "G-on"]),
    ("gpt-oss-20b", ["O-low", "O-high"]),
]
TASK_TITLES = {"T1": "T1  Classify (Banking77, n=100)", "T2": "T2  Calculate (GSM8K, n=100)",
               "T4": "T4  Hard maths (MATH-500 L5, n=50)"}
# label position per (task, family): (anchor condition, dx points, dy points, ha)
LABELS = {
    "T1": {"Qwen3.5-4B": ("S-on", -8, -14, "right"), "Qwen3.8-27B": ("Q-xhigh", 8, -7, "left"),
           "Gemma 4 31B": ("G-on", 8, 5, "left"), "gpt-oss-20b": ("O-low", 0, 11, "center")},
    "T2": {"Qwen3.5-4B": ("S-on", 0, -16, "center"), "Qwen3.8-27B": ("Q-off", 6, -11, "left"),
           "Gemma 4 31B": ("G-on", 8, 6, "left"), "gpt-oss-20b": ("O-low", 0, 11, "center")},
    "T4": {"Qwen3.5-4B": ("S-on", 0, -16, "center"), "Qwen3.8-27B": ("Q-low", -8, -16, "center"),
           "Gemma 4 31B": ("G-on", 8, -2, "left"), "gpt-oss-20b": ("O-low", 0, -16, "center")},
}


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
        "axes.edgecolor": INK_MUTED, "axes.labelcolor": INK_2, "axes.titleweight": "bold",
        "axes.titlecolor": INK, "xtick.color": INK_2, "ytick.color": INK_2, "axes.linewidth": 0.8,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "pdf.fonttype": 42,
    })


def cell(items, c, t):
    return [r for r in items if r["config"] == c and r["task"] == t]


def fig1(items, out: Path):
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.9), sharey=False)
    for ax, t in zip(axes, ["T1", "T2", "T4"]):
        for fam, conds in FAMILIES:
            pts = []
            for c in conds:
                rs = cell(items, c, t)
                tok = [float(r["total_tokens"]) for r in rs]
                k, n = sum(int(r["correct"]) for r in rs), len(rs)
                x, y = statistics.fmean(tok), k / n
                xlo, xhi = boot_mean(tok, n_boot=2000)
                ylo, yhi = wilson(k, n)
                pts.append((c, x, y))
                on = CONFIGS[c]["thinking"]
                ax.plot([xlo, xhi], [y, y], color=ORANGE if on else BLUE, lw=1, alpha=0.35, zorder=2,
                        solid_capstyle="butt")
                ax.plot([x, x], [ylo, yhi], color=ORANGE if on else BLUE, lw=1, alpha=0.35, zorder=2,
                        solid_capstyle="butt")
            ax.plot([p[1] for p in pts], [p[2] for p in pts], color=INK_MUTED, lw=1.5, zorder=3)
            for c, x, y in pts:
                on = CONFIGS[c]["thinking"]
                ax.scatter([x], [y], s=46, color=ORANGE if on else BLUE, edgecolor=SURFACE, linewidth=1.6,
                           zorder=4)
            anchor, dx, dy, ha = LABELS[t][fam]
            ax_x, ax_y = next((x, y) for c, x, y in pts if c == anchor)
            ax.annotate(fam, (ax_x, ax_y), xytext=(dx, dy), textcoords="offset points", ha=ha, va="center",
                        fontsize=8.2, color=INK, zorder=5)
        ax.set_xscale("log")
        ax.set_xlim(180, 25000)
        ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:g}k" if v >= 1000 else f"{v:g}"))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{100 * v:.0f}%"))
        ax.grid(True, which="major", color=GRID, lw=0.7, zorder=0)
        ax.set_axisbelow(True)
        ax.set_title(TASK_TITLES[t], loc="left")
        ax.set_xlabel("Tokens per item (mean, log scale)")
    axes[0].set_ylim(0.58, 0.95)
    axes[1].set_ylim(0.84, 1.005)
    axes[2].set_ylim(0.60, 1.03)
    axes[0].set_ylabel("Accuracy")
    handles = [plt.Line2D([], [], marker="o", ls="", color=BLUE, markersize=7, label="no separate reasoning"),
               plt.Line2D([], [], marker="o", ls="", color=ORANGE, markersize=7,
                          label="reasoning on (effort low → high)"),
               plt.Line2D([], [], color=INK_MUTED, lw=1.5, label="same model weights")]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02), fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, out / "fig1_tokens_vs_accuracy")


def fig2(items, calls, out: Path):
    single = cell(items, "Q-off", "T2")
    n_agent = len(cell(items, "Q-A", "T3"))
    steps: dict[int, list[float]] = {}
    for r in calls:
        if r["config"] == "Q-A" and r["task"] == "T3":
            s = steps.setdefault(int(r["call_index"]), [0.0, 0.0])
            s[0] += int(r["prompt_tokens"])
            s[1] += int(r["completion_tokens"])
    contrib = [(i + 1, p / n_agent, q / n_agent) for i, (p, q) in sorted(steps.items())]  # per-item mean

    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    gap = 0.8  # thin surface edge between stacked segments (keeps small segments visible)
    labels, x = [], 0
    sp = statistics.fmean(int(r["prompt_tokens"]) for r in single)
    sc = statistics.fmean(int(r["completion_tokens"]) for r in single)
    ax.bar(x, sp, width=0.62, color=BLUE, edgecolor=SURFACE, linewidth=gap, zorder=3)
    ax.bar(x, sc, bottom=sp, width=0.62, color=AQUA, edgecolor=SURFACE, linewidth=gap, zorder=3)
    ax.annotate(f"{sp + sc:,.0f}", (x, sp + sc), xytext=(0, 4), textcoords="offset points", ha="center",
                fontsize=8.5, color=INK)
    labels.append("Single answer\n(T2, 1 call)")
    x += 1.4
    base = 0.0
    for step, p, q in contrib:
        ax.bar(x, p, bottom=base, width=0.62, color=BLUE, edgecolor=SURFACE, linewidth=gap, zorder=3)
        ax.bar(x, q, bottom=base + p, width=0.62, color=AQUA, edgecolor=SURFACE, linewidth=gap, zorder=3)
        ax.plot([x + 0.31, x + 0.69], [base + p + q] * 2, color=INK_MUTED, lw=0.8, ls=":", zorder=2)
        base += p + q
        labels.append(f"step {step}")
        x += 1
    tot_p = sum(c[1] for c in contrib)
    tot_q = sum(c[2] for c in contrib)
    ax.bar(x, tot_p, width=0.62, color=BLUE, edgecolor=SURFACE, linewidth=gap, zorder=3)
    ax.bar(x, tot_q, bottom=tot_p, width=0.62, color=AQUA, edgecolor=SURFACE, linewidth=gap, zorder=3)
    ax.annotate(f"{tot_p + tot_q:,.0f}  (×{(tot_p + tot_q) / (sp + sc):.1f})", (x, tot_p + tot_q),
                xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8.5, color=INK)
    labels.append("Agent total\n(T3)")
    ax.set_xticks([0] + [1.4 + i for i in range(len(contrib) + 1)], labels)
    ax.set_xlim(-0.6, x + 0.8)
    ax.set_ylim(0, (tot_p + tot_q) * 1.12)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_ylabel("Tokens per item (mean)")
    ax.grid(True, axis="y", color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0)
    ax.set_title("Qwen3.8-27B, thinking off: an agent re-sends its growing context at every step", loc="left")
    handles = [plt.Rectangle((0, 0), 1, 1, color=BLUE, label="prompt (input) tokens"),
               plt.Rectangle((0, 0), 1, 1, color=AQUA, label="completion (output) tokens")]
    ax.legend(handles=handles, loc="upper left", fontsize=8.5)
    fig.tight_layout()
    save(fig, out / "fig2_agent_tokens")


def save(fig, stem: Path):
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {stem}.png / .pdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(HERE.parent / "results"))
    ap.add_argument("--out", default=str(HERE.parent / "figures"))
    a = ap.parse_args()
    res, out = Path(a.results), Path(a.out)
    items = [r for r in read_rows(res / "items.csv") if r["seed"] == "42"]
    calls = [r for r in read_rows(res / "calls.csv") if r["seed"] == "42"]
    style()
    fig1(items, out)
    fig2(items, calls, out)


if __name__ == "__main__":
    main()
