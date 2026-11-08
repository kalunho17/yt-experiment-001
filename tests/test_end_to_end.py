"""The full pipeline against mock servers (no GPU): `run` and `plan --smoke-only`."""
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

from tokens_per_task.results_io import read_rows

ROOT = Path(__file__).resolve().parents[1]
MOCKS = ROOT / "tests" / "mocks"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_http(url: str, headers=None, timeout=20):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen(urllib.request.Request(url, headers=headers or {}), timeout=2)
            return
        except Exception:  # noqa: BLE001
            time.sleep(0.2)
    raise TimeoutError(url)


def harness(*args, env=None):
    return subprocess.run([sys.executable, str(ROOT / "run.py"), *args], capture_output=True, text=True,
                          encoding="utf-8", errors="replace", env={**os.environ, **(env or {})}, timeout=300)


@pytest.fixture
def mock_vllm():
    port = free_port()
    p = subprocess.Popen([sys.executable, str(MOCKS / "mock_server.py"), "--port", str(port),
                          "--model", "qwen3.8-27b-nvfp4"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wait_http(f"http://127.0.0.1:{port}/v1/models")
        yield f"http://127.0.0.1:{port}/v1"
    finally:
        p.terminate()
        p.wait(10)


def test_run_single_answer_and_agent(mock_vllm, tmp_path):
    r = harness("run", "--config", "Q-off,Q-low,Q-A", "--smoke", "3", "--base-url", mock_vllm,
                "--results-dir", str(tmp_path), "--summarise")
    assert r.returncode == 0, r.stdout + r.stderr
    items = read_rows(tmp_path / "items.csv")
    assert len(items) == 2 * 3 * 3 + 3          # Q-off, Q-low on T1/T2/T4 + Q-A on T3
    agent = [i for i in items if i["config"] == "Q-A"]
    assert all(int(i["n_calls"]) >= 3 and int(i["n_tool_calls"]) >= 3 for i in agent)
    assert all(int(i["reasoning_tokens"]) > 0 for i in items if i["config"] == "Q-low")
    assert all(int(i["reasoning_tokens"]) == 0 for i in items if i["config"] == "Q-off")
    assert (tmp_path / "summary.json").exists()
    # resumable: a second run sends nothing new
    r2 = harness("run", "--config", "Q-off", "--smoke", "3", "--base-url", mock_vllm, "--results-dir", str(tmp_path))
    assert "0 to run" in r2.stdout and len(read_rows(tmp_path / "items.csv")) == len(items)


def test_wrong_model_is_refused(mock_vllm, tmp_path):
    r = harness("run", "--config", "G-off", "--smoke", "1", "--base-url", mock_vllm, "--results-dir", str(tmp_path))
    assert r.returncode != 0 and "expects a model matching" in r.stderr


def test_plan_smoke_only_with_mock_conductor(tmp_path):
    cport, vport = free_port(), free_port()
    p = subprocess.Popen([sys.executable, str(MOCKS / "mock_conductor.py"), "--port", str(cport),
                          "--vllm-port", str(vport), "--token", "t", "--phase-seconds", "0.3"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wait_http(f"http://127.0.0.1:{cport}/health", headers={"Authorization": "Bearer t"})
        r = harness("plan", "--smoke-only", "--smoke", "2", "--profile", "text-qwen35-4b", "text-gemma4",
                    "--base-url", f"http://127.0.0.1:{vport}/v1", "--results-dir", str(tmp_path),
                    env={"CONDUCTOR_URL": f"http://127.0.0.1:{cport}", "CONDUCTOR_TOKEN": "t"})
        assert r.returncode == 0, r.stdout + r.stderr
        assert "SMOKE STAGE PASSED" in r.stdout
        assert len(read_rows(tmp_path / "smoke" / "text-gemma4" / "items.csv")) == 2 * 3 * 2
        assert "gemma4-31b-qat-w4a16" in (tmp_path / "setup.md").read_text(encoding="utf-8")
    finally:
        p.terminate()
        p.wait(10)
