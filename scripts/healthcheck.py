#!/usr/bin/env python3
"""Wrapper so `python scripts/healthcheck.py` works without installing the package."""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from inference_openpi.healthcheck import main

if __name__ == "__main__":
    main()
