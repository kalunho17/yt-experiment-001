"""Experimental configuration: served models, request settings per condition, task sizes.

Everything that defines *what was measured* lives in this file. Changing any value here
changes the experiment; the values below are the ones used for the published results
(see ``results/setup.md``).
"""
from __future__ import annotations

from pathlib import Path

# -----------------------------------------------------------------------------
# Paths (repository layout: src/tokens_per_task/config.py -> repo root is parents[2])
# -----------------------------------------------------------------------------
PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parents[1]
DATA_DIR = REPO_ROOT / "data" / "raw"            # downloaded benchmark files
GENERATED_DIR = REPO_ROOT / "data" / "generated"  # the synthetic T3 ledger (written by `check`)
DEFAULT_RESULTS = REPO_ROOT / "results"

# -----------------------------------------------------------------------------
# Served models (Hugging Face repo @ revision, as recorded in the GPU host's model registry).
#
# One profile = one model served by vLLM at a time. ``model_match`` is the served model id
# (or a case-insensitive substring of it): the harness asks the server what it serves
# (GET /v1/models) and refuses to run a condition against the wrong model.
# If you serve the weights yourself, pass ``--served-model-name <model_match>`` to
# ``vllm serve`` (see docs/reproduce.md), or use ``--any-model``.
#
# ``max_concurrency`` caps --concurrency per model: Gemma 4's KV cache on one RTX 5090
# held only 35,246 tokens, so more parallel requests would only queue and distort latency.
# -----------------------------------------------------------------------------
PROFILES: dict[str, dict] = {
    "text": dict(model_match="qwen3.8-27b-nvfp4", serve="unsloth/Qwen3.8-27B-NVFP4@7d6f8d4d72f56b92b3cdbf22f156b90e1bab0108",
                 optional=False, max_concurrency=8),
    "text-qwen35-4b": dict(model_match="qwen3.5-4b", serve="Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
                           optional=False, max_concurrency=8),
    "text-gemma4": dict(model_match="gemma4-31b-qat-w4a16", serve="google/gemma-4-31B-it-qat-w4a16-ct@52f3f65bc7a02d555763bc923bd1d9094898219d",
                        optional=False, max_concurrency=4),
    "text-gptoss": dict(model_match="gpt-oss-20b", serve="openai/gpt-oss-20b@6cee5e81ee83917806bbde320786a8fb61efebee",
                        optional=True, max_concurrency=8),
}
PLAN_ORDER = ["text", "text-qwen35-4b", "text-gemma4", "text-gptoss"]

# -----------------------------------------------------------------------------
# Conditions ("configs"): one model + one reasoning setting.
#
# profile               which served model the condition runs on.
# thinking              whether the model is expected to reason before answering (used only
#                       for post-hoc sanity checks; not sent to the server).
# agent                 True: the condition runs T3 (tool use). False: T1, T2, T4.
# sampling              top-level request fields (model-card defaults).
# chat_template_kwargs  forwarded to the chat template (vLLM extension).
# reasoning_effort      top-level request field (gpt-oss).
# -----------------------------------------------------------------------------
QWEN_THINKING_OFF = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5}
QWEN_THINKING_ON = {"temperature": 1.0, "top_p": 0.95, "top_k": 20}
GEMMA_SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 64}   # Gemma model card
GPT_OSS_SAMPLING = {"temperature": 1.0, "top_p": 1.0}               # gpt-oss model card

CONFIGS: dict[str, dict] = {
    "S": dict(profile="text-qwen35-4b", thinking=False, agent=False,
              sampling=QWEN_THINKING_OFF, chat_template_kwargs={"enable_thinking": False}),
    "S-on": dict(profile="text-qwen35-4b", thinking=True, agent=False,
                 sampling=QWEN_THINKING_ON, chat_template_kwargs={"enable_thinking": True}),
    "Q-off": dict(profile="text", thinking=False, agent=False,
                  sampling=QWEN_THINKING_OFF, chat_template_kwargs={"enable_thinking": False}),
    "Q-low": dict(profile="text", thinking=True, agent=False,
                  sampling=QWEN_THINKING_ON, chat_template_kwargs={"reasoning_effort": "low"}),
    "Q-med": dict(profile="text", thinking=True, agent=False,
                  sampling=QWEN_THINKING_ON, chat_template_kwargs={"reasoning_effort": "medium"}),
    "Q-xhigh": dict(profile="text", thinking=True, agent=False,
                    sampling=QWEN_THINKING_ON, chat_template_kwargs={"reasoning_effort": "xhigh"}),
    "G-off": dict(profile="text-gemma4", thinking=False, agent=False,
                  sampling=GEMMA_SAMPLING, chat_template_kwargs={"enable_thinking": False}),
    "G-on": dict(profile="text-gemma4", thinking=True, agent=False,
                 sampling=GEMMA_SAMPLING, chat_template_kwargs={"enable_thinking": True}),
    "O-low": dict(profile="text-gptoss", thinking=True, agent=False,
                  sampling=GPT_OSS_SAMPLING, reasoning_effort="low"),
    "O-high": dict(profile="text-gptoss", thinking=True, agent=False,
                   sampling=GPT_OSS_SAMPLING, reasoning_effort="high"),
    "Q-A": dict(profile="text", thinking=False, agent=True,
                sampling=QWEN_THINKING_OFF, chat_template_kwargs={"enable_thinking": False}),
    "Q-A-low": dict(profile="text", thinking=True, agent=True,
                    sampling=QWEN_THINKING_ON, chat_template_kwargs={"reasoning_effort": "low"}),
}
for _c in CONFIGS.values():          # model_match / serve are inherited from the profile
    _c["model_match"] = PROFILES[_c["profile"]]["model_match"]
    _c["serve"] = PROFILES[_c["profile"]]["serve"]

# Human-readable labels (used in reports and figures).
MODEL_OF = {"S": "Qwen3.5-4B", "S-on": "Qwen3.5-4B", "Q-off": "Qwen3.8-27B", "Q-low": "Qwen3.8-27B",
            "Q-med": "Qwen3.8-27B", "Q-xhigh": "Qwen3.8-27B", "Q-A": "Qwen3.8-27B", "Q-A-low": "Qwen3.8-27B",
            "G-off": "Gemma 4 31B", "G-on": "Gemma 4 31B", "O-low": "gpt-oss-20b", "O-high": "gpt-oss-20b"}
SETTING_OF = {"S": "thinking off", "S-on": "thinking on", "Q-off": "thinking off", "Q-low": "effort low",
              "Q-med": "effort medium", "Q-xhigh": "effort xhigh (default)",
              "Q-A": "agent, thinking off", "Q-A-low": "agent, effort low", "G-off": "thinking off",
              "G-on": "thinking on", "O-low": "effort low", "O-high": "effort high"}

# -----------------------------------------------------------------------------
# Tasks and run parameters
# -----------------------------------------------------------------------------
N_ITEMS = {"T1": 100, "T2": 100, "T3": 20, "T4": 50}
TASK_NAMES = {"T1": "classify (Banking77)", "T2": "calculate (GSM8K)", "T3": "agent (invoice ledger)",
              "T4": "hard maths (MATH-500 level 5)"}
AGENT_MAX_STEPS = 15
SAMPLE_SEED = 42            # item sampling and the T3 ledger; never changes
DEFAULT_SEED = 42           # request seed (--seed)
DEFAULT_MAX_TOKENS = 28000
DEFAULT_MAX_MODEL_LEN = 32768
DEFAULT_CONCURRENCY = 8
DEFAULT_BASE_URL = "http://localhost:8000/v1"   # vLLM's default; override with --base-url / VLLM_BASE_URL

# Optional orchestration of a remote GPU host (see conductor.py).
CONDUCTOR_READY_TIMEOUT_S = 1800    # model loads can take 10+ minutes
KEEPALIVE_EVERY_S = 240             # the conductor stops an idle profile after 30 min


def request_settings(cfg: dict) -> dict:
    """The fields a condition adds to every chat-completion request."""
    req = dict(cfg.get("sampling", {}))
    if cfg.get("chat_template_kwargs"):
        req["chat_template_kwargs"] = cfg["chat_template_kwargs"]
    if cfg.get("reasoning_effort"):
        req["reasoning_effort"] = cfg["reasoning_effort"]
    return req


def tasks_of(config: str) -> list[str]:
    """The tasks a condition runs by default."""
    return ["T3"] if CONFIGS[config]["agent"] else ["T1", "T2", "T4"]
