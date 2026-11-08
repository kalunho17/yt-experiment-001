"""Small helpers shared by all modules."""
from __future__ import annotations

import datetime as dt
import os
import re
import sys


def now_iso() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def die(msg: str, code: int = 2):
    print(f"\nERROR: {msg}", file=sys.stderr, flush=True)
    sys.exit(code)


def abort(msg: str, code: int = 2):
    """Stop at once, even with requests still running in worker threads. Every result row is
    flushed to disk when written, so nothing completed is lost."""
    sys.stdout.flush()
    print(f"\nERROR: {msg}", file=sys.stderr, flush=True)
    os._exit(code)


def parse_list(s) -> list[str]:
    """``--config Q-off Q-low`` and ``--config Q-off,Q-low`` both work."""
    if isinstance(s, (list, tuple)):
        s = ",".join(s)
    return [x for x in re.split(r"[,\s]+", s or "") if x]
