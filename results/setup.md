# Setup of the tokens-per-task measurement (ep-001)

One section per served model (written by `--plan`, or `--record-setup` while it is served).

## Setup recorded 2026-11-08T17:28:44+00:00

- Served model id(s): qwen3.8-27b-nvfp4
  - `qwen3.8-27b-nvfp4`: root `/models/qwen3.8-27b-nvfp4`, max_model_len 32768
- vLLM version (GET /version): 0.1.dev19754+g3a0914114
- vLLM endpoint: http://<gpu-host>:8100/v1
- GPU host: gpu-host (conductor 1.0.0 at http://<gpu-host>:8101)
- GPU: NVIDIA GeForce RTX 5090, 32607 MB VRAM
- Conductor profile: text (phase ready)
  - service `vllm-qwen38`: container `vllm-qwen38`, stack `vllm`, state running, healthy True, VRAM budget 29244 MB, url http://<gpu-host>:8100
- GPU log (gpu.csv): util % and VRAM only; the conductor's /health does not report power draw, so no energy figure comes from this harness
- Client: Python 3.13.15 on Windows-11-10.0.26300-SP0 (client), urllib (standard library, no openai package)
- Harness defaults: max_tokens 28000, seed 42, concurrency 8, item sample seed 42

Configs of this harness that match the served model:

| Config | Request settings |
|---|---|
| Q-off | `{"temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5, "chat_template_kwargs": {"enable_thinking": false}}` |
| Q-low | `{"temperature": 1.0, "top_p": 0.95, "top_k": 20, "chat_template_kwargs": {"reasoning_effort": "low"}}` |
| Q-med | `{"temperature": 1.0, "top_p": 0.95, "top_k": 20, "chat_template_kwargs": {"reasoning_effort": "medium"}}` |
| Q-xhigh | `{"temperature": 1.0, "top_p": 0.95, "top_k": 20, "chat_template_kwargs": {"reasoning_effort": "xhigh"}}` |
| Q-A | `{"temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5, "chat_template_kwargs": {"enable_thinking": false}}` |
| Q-A-low | `{"temperature": 1.0, "top_p": 0.95, "top_k": 20, "chat_template_kwargs": {"reasoning_effort": "low"}}` |

Filled in from the GPU host's model registry after the runs (2026-11-08):

| Item | Value |
|---|---|
| Model repo id and revision hash | `unsloth/Qwen3.8-27B-NVFP4` @ `7d6f8d4d72f56b92b3cdbf22f156b90e1bab0108` (GPU host model registry, verified 2026-08-20) |
| `vllm serve` flags of the service | not recorded |
| NVIDIA driver, CUDA version | not recorded |

## Setup recorded 2026-11-08T17:33:54+00:00

- Served model id(s): qwen3.5-4b
  - `qwen3.5-4b`: root `/models/qwen3.5-4b`, max_model_len 32768
- vLLM version (GET /version): 0.26.0
- vLLM endpoint: http://<gpu-host>:8100/v1
- GPU host: gpu-host (conductor 1.0.0 at http://<gpu-host>:8101)
- GPU: NVIDIA GeForce RTX 5090, 32607 MB VRAM
- Conductor profile: text-qwen35-4b (phase ready)
  - service `vllm-qwen35-4b`: container `vllm-qwen35-4b`, stack `vllm`, state running, healthy True, VRAM budget 28790 MB, url http://<gpu-host>:8100
- GPU log (gpu.csv): util % and VRAM only; the conductor's /health does not report power draw, so no energy figure comes from this harness
- Client: Python 3.13.15 on Windows-11-10.0.26300-SP0 (client), urllib (standard library, no openai package)
- Harness defaults: max_tokens 28000, seed 42, concurrency 8, item sample seed 42

Configs of this harness that match the served model:

| Config | Request settings |
|---|---|
| S | `{"temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5, "chat_template_kwargs": {"enable_thinking": false}}` |

Filled in from the GPU host's model registry after the runs (2026-11-08):

| Item | Value |
|---|---|
| Model repo id and revision hash | `Qwen/Qwen3.5-4B` @ `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` (GPU host model registry, verified 2026-11-08) |
| `vllm serve` flags of the service | not recorded |
| NVIDIA driver, CUDA version | not recorded |

## Setup recorded 2026-11-08T17:35:28+00:00

- Served model id(s): gemma4-31b-qat-w4a16
  - `gemma4-31b-qat-w4a16`: root `/models/gemma4-31b-qat-w4a16`, max_model_len 32768
- vLLM version (GET /version): 0.26.0
- vLLM endpoint: http://<gpu-host>:8100/v1
- GPU host: gpu-host (conductor 1.0.0 at http://<gpu-host>:8101)
- GPU: NVIDIA GeForce RTX 5090, 32607 MB VRAM
- Conductor profile: text-gemma4 (phase ready)
  - service `vllm-gemma4`: container `vllm-gemma4`, stack `vllm`, state running, healthy True, VRAM budget 31130 MB, url http://<gpu-host>:8100
- GPU log (gpu.csv): util % and VRAM only; the conductor's /health does not report power draw, so no energy figure comes from this harness
- Client: Python 3.13.15 on Windows-11-10.0.26300-SP0 (client), urllib (standard library, no openai package)
- Harness defaults: max_tokens 28000, seed 42, concurrency 4, item sample seed 42

Configs of this harness that match the served model:

| Config | Request settings |
|---|---|
| G-off | `{"temperature": 1.0, "top_p": 0.95, "top_k": 64, "chat_template_kwargs": {"enable_thinking": false}}` |
| G-on | `{"temperature": 1.0, "top_p": 0.95, "top_k": 64, "chat_template_kwargs": {"enable_thinking": true}}` |

Filled in from the GPU host's model registry after the runs (2026-11-08):

| Item | Value |
|---|---|
| Model repo id and revision hash | `google/gemma-4-31B-it-qat-w4a16-ct` @ `52f3f65bc7a02d555763bc923bd1d9094898219d` (GPU host model registry, verified 2026-11-08) |
| `vllm serve` flags of the service | not recorded |
| NVIDIA driver, CUDA version | not recorded |

## Setup recorded 2026-11-08T17:36:50+00:00

- Served model id(s): gpt-oss-20b
  - `gpt-oss-20b`: root `/models/gpt-oss-20b`, max_model_len 32768
- vLLM version (GET /version): 0.26.0
- vLLM endpoint: http://<gpu-host>:8100/v1
- GPU host: gpu-host (conductor 1.0.0 at http://<gpu-host>:8101)
- GPU: NVIDIA GeForce RTX 5090, 32607 MB VRAM
- Conductor profile: text-gptoss (phase ready)
  - service `vllm-gptoss`: container `vllm-gptoss`, stack `vllm`, state running, healthy True, VRAM budget 29612 MB, url http://<gpu-host>:8100
- GPU log (gpu.csv): util % and VRAM only; the conductor's /health does not report power draw, so no energy figure comes from this harness
- Client: Python 3.13.15 on Windows-11-10.0.26300-SP0 (client), urllib (standard library, no openai package)
- Harness defaults: max_tokens 28000, seed 42, concurrency 8, item sample seed 42

Configs of this harness that match the served model:

| Config | Request settings |
|---|---|
| O-low | `{"temperature": 1.0, "top_p": 1.0, "reasoning_effort": "low"}` |
| O-high | `{"temperature": 1.0, "top_p": 1.0, "reasoning_effort": "high"}` |

Filled in from the GPU host's model registry after the runs (2026-11-08):

| Item | Value |
|---|---|
| Model repo id and revision hash | `openai/gpt-oss-20b` @ `6cee5e81ee83917806bbde320786a8fb61efebee` (GPU host model registry, verified 2026-11-08) |
| `vllm serve` flags of the service | not recorded |
| NVIDIA driver, CUDA version | not recorded |

## Setup recorded 2026-11-12T11:26:08+00:00

- Served model id(s): qwen3.8-27b-nvfp4
  - `qwen3.8-27b-nvfp4`: root `/models/qwen3.8-27b-nvfp4`, max_model_len 32768
- vLLM version (GET /version): 0.1.dev19754+g3a0914114
- vLLM endpoint: http://<gpu-host>:8100/v1
- GPU host: gpu-host (conductor 1.0.0 at http://<gpu-host>:8101)
- GPU: NVIDIA GeForce RTX 5090, 32607 MB VRAM
- Conductor profile: text (phase ready)
  - service `vllm-qwen38`: container `vllm-qwen38`, stack `vllm`, state running, healthy True, VRAM budget 27766 MB, url http://<gpu-host>:8100
- GPU log (gpu.csv): util % and VRAM only; the conductor's /health does not report power draw, so no energy figure comes from this harness
- Client: Python 3.13.15 on Windows-11-10.0.26300-SP0, urllib (standard library, no openai package)
- Harness defaults: max_tokens 28000, seed 42, concurrency 8, item sample seed 42

Configs of this harness that match the served model:

| Config | Request settings |
|---|---|
| Q-off | `{"temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5, "chat_template_kwargs": {"enable_thinking": false}}` |
| Q-low | `{"temperature": 1.0, "top_p": 0.95, "top_k": 20, "chat_template_kwargs": {"reasoning_effort": "low"}}` |
| Q-med | `{"temperature": 1.0, "top_p": 0.95, "top_k": 20, "chat_template_kwargs": {"reasoning_effort": "medium"}}` |
| Q-xhigh | `{"temperature": 1.0, "top_p": 0.95, "top_k": 20, "chat_template_kwargs": {"reasoning_effort": "xhigh"}}` |
| Q-A | `{"temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5, "chat_template_kwargs": {"enable_thinking": false}}` |
| Q-A-low | `{"temperature": 1.0, "top_p": 0.95, "top_k": 20, "chat_template_kwargs": {"reasoning_effort": "low"}}` |
