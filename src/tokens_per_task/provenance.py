"""Records what was served and how (results/setup.md): model id, vLLM version, GPU, client."""
from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

from .client import Server
from .common import now_iso
from .conductor import Conductor, ConductorError
from .config import CONFIGS, SAMPLE_SEED, request_settings


def record_setup(server: Server, cond: Conductor | None, results: Path, max_tokens: int, seed: int,
                 concurrency: int, skip_if_recorded: bool = False):
    """Append one section per served model to results/setup.md."""
    results.mkdir(parents=True, exist_ok=True)
    path = results / "setup.md"
    err = None
    try:
        models = server.models()
    except Exception as e:  # noqa: BLE001
        models, err = [], e
    mids = [str(m.get("id")) for m in models]
    if skip_if_recorded and mids and path.exists():
        if f"- Served model id(s): {', '.join(mids)}\n" in path.read_text(encoding="utf-8"):
            print(f"  setup.md already has a section for {', '.join(mids)}")
            return
    lines = [f"\n## Setup recorded {now_iso()}\n"]
    if models:
        lines += [f"- Served model id(s): {', '.join(mids)}"]
        for m in models:
            lines += [f"  - `{m.get('id')}`: root `{m.get('root')}`, max_model_len {m.get('max_model_len')}"]
    else:
        lines += [f"- Served model: server at {server.base} not reachable ({err})"]
    lines += [f"- vLLM version (GET /version): {server.version() or 'not available'}",
              f"- vLLM endpoint: {server.base}"]
    if cond:
        try:
            h = cond.health()
            g, p = h.get("gpu") or {}, h.get("profile") or {}
            lines += [f"- GPU host: {h.get('hostname', '?')} (conductor {h.get('service_version', '?')} at {cond.base})",
                      f"- GPU: {g.get('name', '?')}, {g.get('vram_total_mb', '?')} MB VRAM",
                      f"- Conductor profile: {p.get('active')} (phase {p.get('phase')})"]
            try:
                services = (cond.profiles().get("profiles") or {}).get(p.get("active") or "", [])
            except ConductorError:
                services = []
            for s in services:
                sv = (h.get("services") or {}).get(s) or {}
                lines += [f"  - service `{s}`: container `{sv.get('container')}`, stack `{sv.get('stack')}`, "
                          f"state {sv.get('state')}, healthy {sv.get('healthy')}, VRAM budget {sv.get('vram_mb')} MB, "
                          f"url {sv.get('url')}"]
            if "power_w" in g or "power_draw_w" in g:
                lines += ["- GPU log (gpu.csv): util %, VRAM and power draw from the conductor's /health"]
            else:
                lines += ["- GPU log (gpu.csv): util % and VRAM only; the conductor's /health does not report power "
                          "draw, so no energy figure comes from this harness"]
        except ConductorError as e:
            lines += [f"- GPU host: conductor not reachable ({e})"]
    else:
        lines += ["- GPU host: no conductor configured"]
    lines += [f"- Client: Python {sys.version.split()[0]} on {platform.platform()}, "
              "urllib (standard library, no openai package)",
              f"- Harness defaults: max_tokens {max_tokens}, seed {seed}, concurrency {concurrency}, "
              f"item sample seed {SAMPLE_SEED}"]
    lines += ["", "Configs of this harness that match the served model:", "",
              "| Config | Request settings |", "|---|---|"]
    for name, c in CONFIGS.items():
        if not mids or any(c["model_match"].lower() in i.lower() for i in mids):
            lines += [f"| {name} | `{json.dumps(request_settings(c))}` |"]
    lines += [""]
    new = not path.exists()
    with path.open("a", encoding="utf-8") as f:
        if new:
            f.write("# Setup of the tokens-per-task measurement\n\n"
                    "One section per served model (written by `run.py plan`, or `run.py record-setup` "
                    "while it is served).\n")
        f.write("\n".join(lines))
    print("\n".join(lines))
    print(f"Appended to {path}")
