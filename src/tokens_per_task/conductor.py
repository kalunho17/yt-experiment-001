"""Optional: orchestration of a remote GPU host that serves one model at a time.

The published runs used a small HTTP control plane on the GPU host (the "conductor") that
starts and stops vLLM containers ("profiles") and reports GPU utilisation and VRAM. Its wire
contract is the subset used below:

    GET  /health     -> {"hostname", "gpu": {"name", "util_pct", "vram_used_mb", ...},
                         "profile": {"active", "target", "phase", "detail"}, "services": {...}}
    GET  /profiles   -> {"profiles": {profile: [service, ...]}}
    POST /activate   {"profile": "<name>" | "none"}
    POST /keepalive  -> {"seconds_until_stop"}

**You do not need this to reproduce the experiment.** Serve each model yourself with
``vllm serve`` and use ``python run.py run`` (docs/reproduce.md). ``tests/mocks/mock_conductor.py``
implements the contract for testing ``python run.py plan`` without a GPU.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from .client import NO_PROXY_OPENER, Server
from .common import die, now_iso
from .config import KEEPALIVE_EVERY_S, REPO_ROOT
from .results_io import GPU_FIELDS, CsvAppender


def read_dotenv() -> dict:
    """KEY=VALUE pairs from the nearest .env at or above the repository root."""
    for d in [REPO_ROOT, *REPO_ROOT.parents]:
        p = d / ".env"
        if p.is_file():
            out = {}
            for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
                m = re.match(r"\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$", line)
                if m and not line.lstrip().startswith("#"):
                    out[m.group(1)] = m.group(2).strip().strip('"').strip("'")
            return out
    return {}


class ConductorError(Exception):
    pass


class Conductor:
    def __init__(self, base_url: str, token: str, timeout: float = 10):
        self.base = base_url.rstrip("/")
        self._token = token
        self.timeout = timeout

    def redact(self, text) -> str:
        text = str(text)
        return text.replace(self._token, "<token>") if self._token else text

    def _req(self, method: str, path: str, payload=None):
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        req = urllib.request.Request(f"{self.base}{path}", data=data, method=method)
        if self._token:
            req.add_header("Authorization", f"Bearer {self._token}")
        if data is not None:
            req.add_header("Content-Type", "application/json")
        try:
            with NO_PROXY_OPENER.open(req, timeout=self.timeout) as r:
                body = r.read().decode("utf-8")
                return r.status, (json.loads(body) if body.strip() else {})
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")[:300]
            if e.code == 401:
                raise ConductorError("401: CONDUCTOR_TOKEN rejected") from None
            raise ConductorError(self.redact(f"conductor {method} {path} -> {e.code}: {body}")) from None
        except (urllib.error.URLError, OSError) as e:
            raise ConductorError(self.redact(f"conductor at {self.base} unreachable: {e}")) from None

    def health(self) -> dict:
        return self._req("GET", "/health")[1]

    def profiles(self) -> dict:
        return self._req("GET", "/profiles")[1]

    def activate(self, profile: str) -> dict:
        return self._req("POST", "/activate", {"profile": profile})[1]

    def keepalive(self):
        try:
            return self._req("POST", "/keepalive")[1].get("seconds_until_stop")
        except Exception:  # noqa: BLE001
            return None

    def wait_ready(self, profile: str, timeout: float, poll: float = 5, quiet: bool = False) -> dict:
        t0, last = time.time(), ""
        while True:
            try:
                h = self.health()
            except ConductorError as e:
                if time.time() - t0 > timeout:
                    raise
                print(f"  conductor: {e}; retrying")
                time.sleep(poll)
                continue
            p = h.get("profile") or {}
            key = f"{p.get('phase')}|{p.get('active')}|{p.get('detail') or ''}"
            if key != last and not quiet:
                det = f" ({p.get('detail')})" if p.get("detail") else ""
                print(f"  conductor: phase {p.get('phase')}, active {p.get('active')}{det} at {time.time() - t0:.0f} s")
                last = key
            if p.get("phase") == "ready" and p.get("active") == profile:
                return h
            # the "none" profile settles at phase "none", not "ready"
            if profile == "none" and p.get("phase") == "none" and p.get("active") == "none":
                return h
            if p.get("phase") == "error":
                raise ConductorError(f"activating {profile} failed: {p.get('detail') or 'no detail'}")
            if time.time() - t0 > timeout:
                raise ConductorError(f"{profile} not ready after {timeout:.0f} s (phase {p.get('phase')}, "
                                     f"active {p.get('active')}). Raise --ready-timeout if the model is still loading.")
            time.sleep(poll)


def conductor_from_env(conductor_url: str | None, required: bool) -> Conductor | None:
    """CONDUCTOR_URL / CONDUCTOR_TOKEN from the environment or a .env file."""
    dotenv = read_dotenv()
    url = (conductor_url or os.environ.get("CONDUCTOR_URL") or dotenv.get("CONDUCTOR_URL") or "").strip()
    token = (os.environ.get("CONDUCTOR_TOKEN") or dotenv.get("CONDUCTOR_TOKEN") or "").strip()
    if not url:
        if required:
            die("no conductor configured: set CONDUCTOR_URL and CONDUCTOR_TOKEN (environment or .env), "
                "or pass --conductor-url. Without a conductor, serve the model yourself and use `run.py run`.")
        return None
    return Conductor(url, token)


def ensure_profile(cond: Conductor, profile: str, server: Server, ready_timeout: float):
    """Activate a profile (if it is not already up) and wait until vLLM answers."""
    try:
        h = cond.health()
    except ConductorError as e:
        die(str(e))
    p = h.get("profile") or {}
    print(f"\nGPU host {h.get('hostname', '?')} ({(h.get('gpu') or {}).get('name', '?')}): "
          f"active profile {p.get('active')}, phase {p.get('phase')}")
    if p.get("active") == profile and p.get("phase") == "ready":
        print(f"  profile {profile} is already up")
        cond.keepalive()
    else:
        active = p.get("active") or ""
        if active.startswith("train"):
            die(f"the GPU host is running {active} (a training run); not stopping it from here.")
        if p.get("phase") not in ("ready", "none", "error"):
            print(f"  the conductor is mid-swap to {p.get('target')}; waiting for it to settle")
            t0 = time.time()
            while p.get("phase") not in ("ready", "none", "error") and time.time() - t0 < ready_timeout:
                time.sleep(5)
                p = cond.health().get("profile") or {}
        if not (p.get("active") == profile and p.get("phase") == "ready"):
            stops = f" (stops {p.get('active')})" if p.get("active") not in (None, "none") else ""
            print(f"  activating {profile}{stops}; a model load can take 10+ minutes")
            try:
                cond.activate(profile)
                cond.wait_ready(profile, timeout=ready_timeout)
            except ConductorError as e:
                die(f"{e}\nIs the model downloaded and the service configured on the GPU host?")
    try:     # the service URL the conductor reports should be the one we talk to
        services = (cond.profiles().get("profiles") or {}).get(profile, [])
        svc = cond.health().get("services") or {}
        urls = {(svc.get(s) or {}).get("url", "").rstrip("/") for s in services}
        urls.discard("")
        if urls and server.root.rstrip("/") not in urls:
            print(f"  WARNING: the conductor serves {profile} at {', '.join(sorted(urls))}, but --base-url is "
                  f"{server.base}.")
    except Exception:  # noqa: BLE001
        pass
    t0 = time.time()
    while True:                     # vLLM may still be loading after the container is "ready"
        try:
            if server.models():
                print(f"  vLLM answers at {server.base} after {time.time() - t0:.0f} s")
                return
        except Exception:  # noqa: BLE001
            pass
        if time.time() - t0 > ready_timeout:
            die(f"{profile} is up on the conductor but {server.base}/models does not answer.")
        time.sleep(5)


def release_gpu(cond: Conductor):
    print("\nReleasing the GPU (conductor profile none) ...")
    try:
        cond.activate("none")
        cond.wait_ready("none", timeout=600, quiet=True)
        print("  GPU released.")
    except ConductorError as e:
        print(f"  WARNING: could not release the GPU: {e}. The conductor stops an idle profile after 30 min.")


class Keepalive(threading.Thread):
    """POST /keepalive every few minutes so the conductor's idle stop never hits a long run."""

    def __init__(self, cond: Conductor, every: float = KEEPALIVE_EVERY_S):
        super().__init__(daemon=True)
        self.cond, self.every, self.stop_event = cond, every, threading.Event()

    def run(self):
        while not self.stop_event.is_set():
            self.cond.keepalive()
            self.stop_event.wait(self.every)

    def stop(self):
        self.stop_event.set()


class GpuSampler(threading.Thread):
    """Samples the GPU host's /health (util %, VRAM; power only if the conductor reports it)."""

    def __init__(self, cond: Conductor, path: Path, interval: float):
        super().__init__(daemon=True)
        self.cond, self.interval = cond, interval
        self.tag = {"config": "", "task": ""}
        self.stop_event = threading.Event()
        self.out = CsvAppender(path, GPU_FIELDS)
        self.t0 = time.time()

    def run(self):
        while not self.stop_event.is_set():
            t = time.time()
            try:
                g = self.cond.health().get("gpu") or {}
                power = g.get("power_w", g.get("power_draw_w", ""))
                self.out.write([{"timestamp": now_iso(), "elapsed_s": round(t - self.t0, 1),
                                 "config": self.tag["config"], "task": self.tag["task"], "gpu": g.get("name", ""),
                                 "util_pct": g.get("util_pct", ""), "vram_used_mb": g.get("vram_used_mb", ""),
                                 "power_w": "" if power is None else power}])
            except Exception:  # noqa: BLE001
                pass
            self.stop_event.wait(max(0.0, self.interval - (time.time() - t)))

    def stop(self):
        self.stop_event.set()
        self.join(timeout=10)
        self.out.close()
