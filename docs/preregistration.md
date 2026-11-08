# Pre-registered design and deviations

The design below was written on 2026-11-08, **before any model was run**, and is reproduced
verbatim. It was written as part of a video production workflow, so it refers to video
segments (`s04`, `s06`, ...), to internal files (`rules.md`, `inbox/`) and to a harness
`run.py` that is now `src/tokens_per_task/`. It also plans pricing in EUR, which the video
does; this repository reports price-independent token counts plus a calculator
(`analysis/cost.py`).

## Deviations from the design

| Design | What was done | Why |
|---|---|---|
| Tasks T1, T2, T3 | **T4 added** (MATH-500 level 5, integer answers, n = 50) | added after the first smoke pass (`results/logs/smoke_run_1.log`), before the second smoke pass and all full runs |
| Small model with thinking off only (S) | **S-on added** (Qwen3.5-4B, thinking on) | added at the same point: small model reasoning vs large model answering directly (routing question) |
| gpt-oss-20b optional | run (O-low, O-high) | - |
| Gemma 4: "a 4-bit build" | QAT W4A16 build, served as `gemma4-31b-qat-w4a16`, concurrency 4 | fits 32 GB; KV cache held only ~35k tokens |
| vLLM ≥ 0.26 | 0.26.0 for three models; a development build (`0.1.dev19754+g3a0914114`) for Qwen3.8-27B | Qwen3.8 needed its own image |
| `openai` client, `datasets` library | standard-library HTTP client, raw dataset files | zero dependencies |
| Agent fallback to Qwen3.6-35B-A3B if tool calling fails | not needed: Qwen3.8-27B tool calling passed the smoke checks | - |
| Second seed on T2 | not run | - |
| GPU power logging | utilisation and VRAM only | the host telemetry reports no power draw |
| Record model revision hashes and `vllm serve` flags | revisions filled in after the runs from the GPU host's model registry; Qwen3.8 build is `unsloth/Qwen3.8-27B-NVFP4` (the design named `Inferact/...`); serve flags and driver/CUDA not recorded | limitation, stated in the report |
| - | single-stream speed probe added (`results/speed.csv`) | throughput context |
| - | a warm-up request after each model start (not recorded) | keeps one-off JIT compile time out of the timed items |

Sampling, prompts, item samples (seed 42), scoring rules, `max_tokens` and the context length
are as designed.

---

# Own measurement: tokens per task (ep-001)

> Design, 2026-11-08. Dr. Kalun Ho runs it on his own RTX 5090 with vLLM and reports the
> numbers. It is cited as `type: dataset`, `author: Dr. Kalun Ho`, note "own measurement"
> (`rules.md`). Never run on employer hardware or with employer data.

## The question it answers

How many tokens does **one task** consume, and how often is it done right, when the same
task runs (1) on a small model, (2) on a mid-size model answering directly, (3) on the same
model reasoning first, and (4) as an agent that calls tools? Price per token stays fixed;
only tokens per task and the success rate change. That is the episode's mechanism, measured.

Feeds: **s04** (tokens per task by mode, next to the published multi-agent figures),
**s06** (cost per task in €, priced at API list prices: the user's choice (a)), **s07**
(the routing rule of thumb: where the small model is as good, where it is not), **s08** (the
KPI: cost per *successful* task).

## Hardware and software (record all of it)

| Item | Value to record |
|---|---|
| GPU | RTX 5090, 32 GB (driver version, `nvidia-smi`) |
| OS, CUDA | version strings |
| vLLM | `vllm --version` (Blackwell needs a CUDA 12.8+ build) |
| Python, `openai` client | versions |
| Model | Hugging Face repo id **and revision hash** |
| Server flags | `--max-model-len`, `--gpu-memory-utilization`, parsers |
| Sampling | temperature, top_p, seed, max_tokens (below) |
| Date | of each run |

## Models (all fit in 32 GB; current releases as of November 2026)

| Config | Model | Mode | Why |
|---|---|---|---|
| **S** | `Qwen/Qwen3.5-4B` (Alibaba, Mar 2026; BF16, ~9 GB) | thinking off | the small model a router sends easy tasks to; same vendor as the main model |
| **Q-off** | `Qwen/Qwen3.8-27B` (Alibaba, Aug 2026; NVFP4 build) | thinking off | mid-size, answers directly |
| **Q-low** | `Qwen/Qwen3.8-27B` | `reasoning_effort: low` | **same weights**, the reasoning dial turned up step by step: |
| **Q-med** | `Qwen/Qwen3.8-27B` | `reasoning_effort: medium` | isolates what reasoning alone does to tokens per task |
| **Q-xhigh** | `Qwen/Qwen3.8-27B` | `reasoning_effort: xhigh` (the model's **default**) | what a team gets if nobody touches the setting |
| **G-off / G-on** | `google/gemma-4-31B-it` (Google, 2026; 4-bit build) | thinking off / on | second vendor, so the result is not one family's quirk |
| **O-low / O-high** | `openai/gpt-oss-20b` (OpenAI, Aug 2025; MXFP4, ~14 GB) | effort low / high | optional third vendor; older, but OpenAI's only open-weight model |
| **Q-A** | `Qwen/Qwen3.8-27B` | thinking off, **agent** with tools | the agent multiplier |
| **Q-A-low** | `Qwen/Qwen3.8-27B` | effort low, agent | reasoning × agent |

Chosen 2026-11-08 from the Artificial Analysis open-weights leaderboards (small and tiny
classes), the Qwen3.8-27B model card and the vLLM recipe for it. Qwen3.8-27B is the
strongest open model in the 4–40B class that fits one 5090, and its four settings (off, low,
medium, xhigh) on the same weights are the cleanest test of the mechanism there is.
Record the exact repo id and revision of every build; the 4-bit builds are community
quantisations (no official ones are published).

**Known risks on one RTX 5090** (from the vLLM recipe; check in the dry run):

- Qwen3.8-27B needs an NVFP4 build (`Inferact/Qwen3.8-27B-NVFP4` per the recipe),
  `--enforce-eager` (CUDA-graph capture runs out of memory otherwise) and
  `--max-model-len 32768`. At `xhigh` the model can think for 20,000+ tokens on one prompt,
  so an answer can hit the limit: count that as a failed task and report how often it
  happened (it is part of the story, not a bug).
- The single-card recipe has no tool-calling flags. Add
  `--enable-auto-tool-choice --tool-call-parser qwen3_coder` and test T3 on five items first.
  If it fails, the agent runs move to Qwen3.6-35B-A3B (MoE, 3B active, 4-bit ~23 GB).
- vLLM 0.26 or newer for the NVFP4 kernels on sm_120.

Serve one model at a time:

```bash
# S
vllm serve Qwen/Qwen3.5-4B --max-model-len 32768 --gpu-memory-utilization 0.90 \
  --reasoning-parser qwen3
# Q-off, Q-low, Q-med, Q-xhigh, Q-A, Q-A-low (mode set per request)
vllm serve Inferact/Qwen3.8-27B-NVFP4 --max-model-len 32768 --kv-cache-dtype fp8 \
  --enforce-eager --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder
# G-off, G-on (a 4-bit build of google/gemma-4-31B-it; check the recipe for the parser)
vllm serve <4-bit gemma-4-31B-it build> --max-model-len 32768 --gpu-memory-utilization 0.90
# O-low, O-high (optional)
vllm serve openai/gpt-oss-20b --max-model-len 32768 --gpu-memory-utilization 0.90
```

Per request: thinking off `extra_body={"chat_template_kwargs": {"enable_thinking": false}}`;
Qwen3.8 effort `extra_body={"chat_template_kwargs": {"reasoning_effort": "low"}}`
(`low` | `medium` | `xhigh`; other names raise an error); gpt-oss `reasoning_effort="low"`.

Sampling (model-card defaults): Qwen thinking off T=0.7, top_p=0.8, top_k=20,
presence_penalty=1.5; thinking on T=1.0, top_p=0.95, top_k=20; Gemma 4 and gpt-oss per
their cards. `seed=42`, `max_tokens` = what the context leaves. One run per item; a second
seed on T2 if time allows, to show the spread.

## Tasks (each has an answer key, so success is counted, not judged)

| Task | Data | n | What it stands for | Success |
|---|---|--:|---|---|
| **T1 Classify** | Banking77 test set (CC BY 4.0), random sample, seed 42; prompt lists the 77 intents | 100 | routing a support ticket | exact label match |
| **T2 Calculate** | GSM8K test set (MIT), random sample, seed 42 | 100 | a multi-step business calculation | final number matches |
| **T3 Agent** | synthetic invoice ledger (generated by the harness, seed 42: 500 invoices, 12 suppliers, 4 quarters) with three tools: `list_suppliers`, `get_invoices(supplier, quarter)`, `calculator(expr)`; max 15 steps | 20 | an analyst question answered by looking things up | answer matches the key computed from the ledger |

T1 and T2 run on S, Q-off, Q-low, Q-med, Q-xhigh, G-off, G-on (and O-low, O-high if time
allows). T3 runs on Q-A and Q-A-low. Every call's
`usage.prompt_tokens` and `usage.completion_tokens` are summed over **all calls of a task**:
an agent re-sends its growing context on every step, which is exactly what the episode
explains.

## What to log (the harness writes it)

Per call: config, task, item id, call index, prompt tokens, completion tokens, reasoning
tokens (tokens of `reasoning_content`, counted with the model's tokenizer), latency, tool
calls. Per item: total tokens, number of calls, correct (yes/no). Optional, for idea 2
(rent or own) later: GPU power draw sampled each second with `nvidia-smi`.

## What comes back

`results/calls.csv`, `results/items.csv`, `results/summary.json` and `results/setup.md`
(the table above, filled in). Drop them in `inbox/` with names starting `ep-001-`; they are
routed to `input/experiment/`.

## How it is priced (choice (a))

Two prices, both published list prices on the access date, converted to € at the ECB
reference rate of that day:

1. **The measured models themselves** at a public API provider's list price for
   Qwen3.8-27B, Qwen3.5-4B, Gemma 4 31B (and gpt-oss-20b): the cost per task of exactly what was measured.
2. **A frontier and a small commercial model** (names on the slide, e.g. one OpenAI,
   Anthropic or Google frontier and small tier): the measured token counts priced at their
   list prices, labelled as a projection ("if this task ran at these list prices").
   Frontier models tokenise and reason differently, so this half is `mixed`, and the script
   says so.

## Limits (said on the slide and in the script)

- Token counts are for these open models; commercial models differ, the multipliers are
  the point, not the absolute counts.
- Three task types, 100 + 100 + 20 items: a measurement of this setup, not of "AI".
- Benchmarks stand in for business tasks; the agent task is synthetic.

## Harness

`run.py` (Python, `openai` client against the local vLLM endpoint, `datasets` for Banking77
and GSM8K) is written in step 2c, once the idea goes ahead, together with a dry run that
checks the answer keys without a GPU.
