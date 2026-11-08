#!/usr/bin/env python3
"""Entry point that needs no installation: ``python run.py <command>``.

Equivalent to ``python -m tokens_per_task`` after ``pip install -e .``. Python 3.11+,
standard library only. See docs/reproduce.md.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from tokens_per_task.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
