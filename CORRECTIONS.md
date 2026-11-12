# Corrections

Errors found after publication, with what was measured again and what changed. The results as
published on 2026-11-10 are tagged `v1.0.0`; nothing is deleted, superseded rows move to
`results/superseded/`.

## 2026-11-12: Qwen3.8-27B single-stream speed, 20 → 67 tokens/s

**What was wrong.** The Qwen3.8-27B container was started with `--enforce-eager`, which turns off
CUDA graphs. Without them, every generated token needs hundreds of small GPU kernels, each launched
separately from Python; the GPU finishes each one quickly and then waits for the next launch. That
held a single request to about 20 tokens/s and the GPU at about 40% utilisation. The flag came from
vLLM's set-up guide for this model on an RTX 5090, which recommends eager mode because capturing the
graphs for every batch size up to 512 runs out of memory on a 32 GB card.

**The fix.** CUDA graphs on, captured only up to batch size 64
(`--max-cudagraph-capture-size 64 --max-num-seqs 64`). Throughput levels off at about 64 concurrent
requests, so the larger sizes added nothing; the capture takes 0.67 GiB. Same vLLM build
(`0.1.dev19754+g3a0914114`), same checkpoint (`unsloth/Qwen3.8-27B-NVFP4` @ `7d6f8d4d`), same request
settings.

**What was measured again.** Only the single-stream speed probe (`python run.py speed --profile text
--replace`: 3 streamed runs, one request at a time, 2026-11-12 11:26 UTC) and, as a spot check,
Banking77 with thinking off (Q-off, T1, 100 messages).

| Qwen3.8-27B, thinking off, single stream | Published 2026-11-10 | Corrected 2026-11-12 |
|---|--:|--:|
| Decode, tokens/s (mean of 3 runs) | 20.3 | 67.2 |
| Time to first token (s) | 0.16 | 0.06 |
| Output tokens per run | 545 | 550 |
| gpt-oss-20b (287 tokens/s) ÷ Qwen3.8-27B | ×14 | ×4.3 |

The three published rows are kept in `results/superseded/speed.csv`; the new rows are in
`results/speed.csv`, `results/summary.json` and `results/RESULTS.md`. The setup record of the rerun is
the last section of `results/setup.md`. On the host itself, the same fix moved a single request from
20.6 to 68 tokens/s, eight concurrent requests from 410 to 932 tokens/s in total, and GPU utilisation
from about 44% to about 100% (measured with the host's own tools, outside this harness).

**Spot check.** Banking77, thinking off: 73 of 100 correct (published: 71) at 456 tokens per message
(published: 456). With 100 items, a difference of up to about 9 points is within chance; the token
count is identical. Files: `results/corrections/2026-11-12-qwen38-cudagraphs/`.

**What changes.** Only statements about Qwen3.8-27B's throughput:

- Report §4.4 and the website: 20 tokens/s becomes 67, and "14 times apart" becomes about 4 times.
- `results/RESULTS.md`, "Throughput during the runs": every Qwen3.8-27B row (per-request and
  aggregate tokens/s), the latencies in `items.csv` and `calls.csv` and the wall times of its six
  conditions (Q-off, Q-low, Q-med, Q-xhigh, Q-A, Q-A-low) were measured in eager mode and are
  roughly 3× too slow. They are left as measured and marked; they were not rerun.
- `results/gpu.csv`: Qwen3.8-27B's mean utilisation of about 41% is the eager-mode figure.
- Report §7: the 3.3 hours of wall-clock time include Qwen3.8-27B at eager-mode speed.

**What does not change.** Every token count, accuracy, multiplier and confidence interval. The
checkpoint, build, prompts, sampling settings and seed are the same, and CUDA graphs change when the
kernels run, not what they compute; the spot check agrees. The five key findings stand as published.

**Where it was published.** The video "AI Got 280× Cheaper. So Why Is Your Bill Going Up?" says
"Qwen 3.8: 20. That is 14 times apart" on the slide "Four findings"; the video cannot be changed, so
the correction is in its description and pinned comment. The experiment page on kalunho.com shows
the corrected chart with the published bar kept beside it. The LinkedIn card "Finding 5 · Speed" is
corrected in the same way.
