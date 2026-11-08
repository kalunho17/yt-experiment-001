#!/usr/bin/env python3
"""Mock of the GPU host's conductor, for testing `run.py plan` without the real host.

    python mock_conductor.py --port 8791 --vllm-port 8792 --token test-token
                             [--phase-seconds 2] [--fail-profile text-gemma4]
                             [--always-think-profile text]

On POST /activate it walks the phases (stopping -> starting -> warming -> ready) and starts
mock_server.py on --vllm-port with the profile's model id, like the real conductor starts the
vLLM container. GET /health, GET /profiles, POST /keepalive as in the wire contract
documented in src/tokens_per_task/conductor.py. Every route needs the bearer token.
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PROFILES = {   # profile -> (service, model id the mock vLLM serves)
    "text": ("vllm-qwen38", "qwen3.8-27b-nvfp4"),
    "text-qwen35-4b": ("vllm-qwen35-4b", "qwen3.5-4b"),
    "text-gemma4": ("vllm-gemma4", "gemma4-31b-qat-w4a16"),
    "text-gptoss": ("vllm-gptoss", "gpt-oss-20b"),
    "audio": ("audio-service", None),
}
ARGS = None
STATE = {"active": "none", "target": "none", "phase": "none", "detail": None, "proc": None,
         "keepalives": 0, "activations": [], "gen": 0}
LOCK = threading.Lock()


def log(msg: str):
    print(f"[mock-conductor {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def stop_proc():
    p = STATE["proc"]
    if p and p.poll() is None:
        p.terminate()
        try:
            p.wait(10)
        except subprocess.TimeoutExpired:
            p.kill()
    STATE["proc"] = None


def swap(profile: str, gen: int):
    def set_(**kw):
        with LOCK:
            if STATE["gen"] == gen:
                STATE.update(kw)
    step = ARGS.phase_seconds
    if STATE["proc"]:
        set_(phase="stopping", active=None, detail=f"stopping {STATE['active']}")
        stop_proc()
        time.sleep(step / 2)
    if profile == "none":
        set_(phase="none", active="none", detail=None)
        return
    set_(phase="starting", active=None, detail=f"starting {PROFILES[profile][0]}")
    time.sleep(step)
    if profile == ARGS.fail_profile:
        set_(phase="error", active=None, detail=f"{PROFILES[profile][0]}: container exited (mock failure)")
        return
    model = PROFILES[profile][1]
    if model:
        cmd = [sys.executable, str(Path(__file__).with_name("mock_server.py")), "--port", str(ARGS.vllm_port),
               "--model", model]
        if profile == ARGS.always_think_profile:
            cmd.append("--always-think")
        STATE["proc"] = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    set_(phase="warming", detail="loading weights")
    time.sleep(step)
    set_(phase="ready", active=profile, detail=None)
    log(f"profile {profile} ready (model {model})")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code: int, obj):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _authed(self) -> bool:
        if self.headers.get("Authorization") != f"Bearer {ARGS.token}":
            self._send(401, {"detail": "unauthorized"})
            return False
        return True

    def do_GET(self):
        if not self._authed():
            return
        if self.path == "/health":
            with LOCK:
                busy = STATE["phase"] == "ready" and STATE["active"] not in (None, "none")
                services = {}
                for prof, (svc, _m) in PROFILES.items():
                    running = STATE["active"] == prof and STATE["phase"] == "ready"
                    services[svc] = {"container": svc, "stack": "vllm" if svc.startswith("vllm") else svc,
                                     "state": "running" if running else "absent", "healthy": running,
                                     "vram_mb": 29300,
                                     "url": f"http://127.0.0.1:{ARGS.vllm_port}" if svc.startswith("vllm")
                                     else "http://127.0.0.1:1"}
                return self._send(200, {
                    "conductor": "ok", "service_version": "mock", "hostname": "mock-gpu-host", "uptime_s": 1,
                    "gpu": {"name": "NVIDIA GeForce RTX 5090 (mock)", "vram_total_mb": 32607,
                            "vram_used_mb": 29000 + random.randint(0, 900) if busy else 18,
                            "util_pct": random.randint(60, 99) if busy else 0},
                    "profile": {k: STATE[k] for k in ("active", "target", "phase", "detail")},
                    "idle": {"stop_after_minutes": 30, "seconds_until_stop": 1800 if busy else None,
                             "gpu_busy": busy, "exempt": False},
                    "services": services, "event_seq": STATE["gen"]})
        if self.path == "/profiles":
            return self._send(200, {"profiles": {p: [s] for p, (s, _m) in PROFILES.items()}, "none": [],
                                    "no_idle_stop_profiles": [], "managed_extra": [], "vram_total_mb": 32607})
        if self.path == "/_stats":     # test hook
            with LOCK:
                return self._send(200, {"keepalives": STATE["keepalives"], "activations": STATE["activations"]})
        self._send(404, {"detail": "Not Found"})

    def do_POST(self):
        if not self._authed():
            return
        n = int(self.headers.get("Content-Length", 0) or 0)
        body = json.loads(self.rfile.read(n) or b"{}") if n else {}
        if self.path == "/keepalive":
            with LOCK:
                STATE["keepalives"] += 1
            return self._send(200, {"seconds_until_stop": 1800})
        if self.path == "/activate":
            prof = body.get("profile")
            if prof != "none" and prof not in PROFILES:
                return self._send(400, {"detail": f"unknown profile {prof}"})
            with LOCK:
                STATE["gen"] += 1
                gen = STATE["gen"]
                STATE["target"] = prof
                STATE["activations"].append(prof)
            log(f"activate {prof}")
            threading.Thread(target=swap, args=(prof, gen), daemon=True).start()
            return self._send(202, {"accepted": prof, "generation": gen})
        self._send(404, {"detail": "Not Found"})


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8791)
    ap.add_argument("--vllm-port", type=int, default=8792)
    ap.add_argument("--token", default="test-token")
    ap.add_argument("--phase-seconds", type=float, default=2.0)
    ap.add_argument("--fail-profile")
    ap.add_argument("--always-think-profile")
    ARGS = ap.parse_args()
    srv = ThreadingHTTPServer(("127.0.0.1", ARGS.port), Handler)
    log(f"listening on http://127.0.0.1:{ARGS.port}, vLLM mock on port {ARGS.vllm_port}")
    try:
        srv.serve_forever()
    finally:
        stop_proc()


if __name__ == "__main__":
    main()
