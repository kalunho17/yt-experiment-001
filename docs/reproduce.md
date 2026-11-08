# Reproducing the measurement

Everything runs with Python ≥ 3.11 and the standard library. `matplotlib` is needed only for figures and `pytest` only for tests.

## 0. Without a GPU

```bash
python run.py check          # downloads the benchmarks if data/raw/ is empty, verifies all keys and tools
python run.py list-configs   # the 12 conditions and exactly what each sends
python -m pytest             # unit, reproducibility and end-to-end tests (mock vLLM + mock conductor)
```

`check` also rewrites `data/generated/t3_ledger.json` and `t3_questions.json`. They must stay byte-identical to the committed files; a test checks this.

## 1. Serve a model

Serve one model at a time with vLLM. The harness checks that the served model id contains the `model_match` string of the condition (see `src/tokens_per_task/config.py`), so pass `--served-model-name` accordingly:

| Profile | Conditions | `--served-model-name` | Weights used here (Hugging Face repo, revision) |
|---|---|---|---|
| `text` | Q-off, Q-low, Q-med, Q-xhigh, Q-A, Q-A-low | `qwen3.8-27b-nvfp4` | `unsloth/Qwen3.8-27B-NVFP4`, `7d6f8d4d72f56b92b3cdbf22f156b90e1bab0108` |
| `text-qwen35-4b` | S, S-on | `qwen3.5-4b` | `Qwen/Qwen3.5-4B` (BF16), `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` |
| `text-gemma4` | G-off, G-on | `gemma4-31b-qat-w4a16` | `google/gemma-4-31B-it-qat-w4a16-ct`, `52f3f65bc7a02d555763bc923bd1d9094898219d` |
| `text-gptoss` | O-low, O-high | `gpt-oss-20b` | `openai/gpt-oss-20b` (MXFP4), `6cee5e81ee83917806bbde320786a8fb61efebee` |

Download a revision with `huggingface-cli download <repo> --revision <hash>`. Reference commands from the design (adapt the paths; on one RTX 5090 the 27B model needs a 4-bit build, `--enforce-eager` and a 32k context):

```bash
# Qwen3.8-27B: reasoning parser + tool calling (T3)
vllm serve unsloth/Qwen3.8-27B-NVFP4 --served-model-name qwen3.8-27b-nvfp4 \
  --max-model-len 32768 --kv-cache-dtype fp8 --enforce-eager --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder

vllm serve Qwen/Qwen3.5-4B --served-model-name qwen3.5-4b \
  --max-model-len 32768 --gpu-memory-utilization 0.90 --reasoning-parser qwen3

vllm serve google/gemma-4-31B-it-qat-w4a16-ct --served-model-name gemma4-31b-qat-w4a16 \
  --max-model-len 32768 --gpu-memory-utilization 0.90 --reasoning-parser <gemma parser>

vllm serve openai/gpt-oss-20b --served-model-name gpt-oss-20b \
  --max-model-len 32768 --gpu-memory-utilization 0.90
```

The exact `vllm serve` flags of the published runs were not recorded. What was recorded (served ids, model roots, repo revisions, vLLM versions, request settings) is in `results/setup.md`.

## 2. Smoke test, then the full run

```bash
# 5 items per task into results/smoke/; prints CHECK lines if something is misconfigured
python run.py run --config S,S-on --smoke 5 --base-url http://localhost:8000/v1

# full run (resumable: rerun the same command after any interruption)
python run.py run --config S,S-on --base-url http://localhost:8000/v1 --results-dir my_results
```

Repeat per served model, then:

```bash
python run.py summarise --results-dir my_results
python analysis/make_report.py --results my_results
python analysis/make_figures.py --results my_results --out my_figures
```

The CHECK lines tell you if a condition's reasoning switch does not reach the chat template, if reasoning leaks into the answer text (missing `--reasoning-parser`), or if tool calls are not parsed (missing `--enable-auto-tool-choice --tool-call-parser`).

| Option | Meaning |
|---|---|
| `--concurrency N` | parallel items (capped per model in `PROFILES`) |
| `--max-tokens N` | answer budget (default 28,000; lowered automatically to fit the context) |
| `--seed N` | request seed; the item sample is always seed 42. Seeds are kept apart in the results. |
| `--task T1,T2` | a subset of tasks |
| `--any-model` | skip the served-model check |

## 3. Optional: the orchestrated plan

The published runs used `python run.py plan --include-optional`, which drives a small control plane on the GPU host (the "conductor") to swap models. For each model in turn, the plan:

1. activates it;
2. records the setup;
3. runs the smoke test and stops on any CHECK;
4. probes single-stream speed;
5. runs all items while logging GPU utilisation;
6. finally releases the GPU.

The conductor's HTTP contract is documented in `src/tokens_per_task/conductor.py`, and `tests/mocks/mock_conductor.py` implements it:

```bash
python tests/mocks/mock_conductor.py --port 8791 --vllm-port 8792 --token t &
CONDUCTOR_URL=http://127.0.0.1:8791 CONDUCTOR_TOKEN=t \
  python run.py plan --smoke-only --base-url http://127.0.0.1:8792/v1 --results-dir /tmp/mock
```

## Costs

`python analysis/cost.py --input-price <per 1M> --output-price <per 1M>` prices the measured token counts per task and per successful task. `--prices file.json` takes per-model prices, e.g. `{"Qwen3.8-27B": [0.3, 1.2], "Qwen3.5-4B": [0.05, 0.2]}`.
