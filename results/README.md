# Results

Measured on 2026-11-08 (runs 18:09–21:29 UTC): one NVIDIA GeForce RTX 5090, vLLM, request seed 42.

| File | What |
|---|---|
| `items.csv` | one row per (condition, task, item); the unit of analysis |
| `calls.csv` | one row per model call; an agent item has several |
| `summary.json` | aggregates per condition × task × seed, in-run throughput, single-stream speed (written by `run.py summarise`) |
| `RESULTS.md` | all tables of the report with 95% CIs (written by `analysis/make_report.py`) |
| `speed.csv` | single-stream speed probe: 3 streamed requests per model, no competing load |
| `gpu.csv` | GPU utilisation and VRAM every 2 s during the full runs (power not available) |
| `setup.md` | per served model: id, root, max length, vLLM version, GPU, request settings |
| `smoke/<profile>/` | the 5-items-per-task smoke stage that gated each model's full run |
| `logs/` | console output of the runs: two smoke passes, the full plan, a resume of the gpt-oss profile |

## `items.csv`

| Column | Meaning |
|---|---|
| `config`, `task`, `item_id`, `seed`, `model` | condition, task (T1–T4), item, request seed, served model id |
| `n_calls` | model calls made for the item (1 for single answer) |
| `prompt_tokens`, `completion_tokens` | summed over all calls (server-reported `usage`) |
| `total_tokens` | prompt + completion |
| `reasoning_tokens` | tokens in the separate reasoning channel (part of completion), counted with the model's tokenizer |
| `n_tool_calls` | tool calls issued (T3) |
| `latency_s` | summed request latency |
| `correct`, `truncated` | 1/0; truncated items count as incorrect |
| `expected`, `predicted` | answer key and parsed answer |
| `note` | `truncated`, `context_exceeded`, `step_limit`, `no_answer_line` (no `Answer:` marker; the last number was scored) or `unparsed_tool_call` |

## `calls.csv`

`call_index` (0-based step), `prompt_tokens`, `completion_tokens`, `reasoning_tokens`, `reasoning_source` (`usage`, `tokenize`, or `empty` when no reasoning came back), `latency_s`, `finish_reason`, `n_tool_calls`, `max_tokens` (the budget actually sent).

## Redactions

The only edits to the raw files: the GPU host's LAN address, both machines' host names and local file paths were replaced by `<gpu-host>`, `gpu-host`, `client` and `<repo>` in `setup.md` and `logs/`. No measured value was changed. A derived report that the earlier harness wrote for one smoke profile was dropped; it can be regenerated from the smoke CSVs.
