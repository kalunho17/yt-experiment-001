# Setup of the tokens-per-task measurement

One section per served model (written by `run.py plan`, or `run.py record-setup` while it is served).

## Setup recorded 2026-11-12T11:26:53+00:00

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
