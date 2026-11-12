# Spot check after the Qwen3.8-27B serving fix (2026-11-12)

Qwen3.8-27B, thinking off (`Q-off`), Banking77 (`T1`), the same 100 seeded messages as the published
run, on the same build and checkpoint with CUDA graphs on (`--max-cudagraph-capture-size 64
--max-num-seqs 64` instead of `--enforce-eager`). Run with:

```bash
python run.py run --config Q-off --task T1 --activate --gpu-log --results-dir results/corrections/2026-11-12-qwen38-cudagraphs
```

| | Published 2026-11-08 (eager mode) | This run |
|---|--:|--:|
| Correct | 71 / 100 | 73 / 100 |
| Mean total tokens per message | 456 | 456 |
| Truncated | 0 | 0 |

The token count is identical and the accuracy difference is within chance for 100 items, so the
published token counts and accuracies were not rerun. The single-stream speed rerun itself is in
`../../speed.csv` (the published rows in `../../superseded/speed.csv`). Context: `../../../CORRECTIONS.md`.
`gpu.csv` has only a handful of samples because the run took eight seconds; T1 is prefill-bound.
