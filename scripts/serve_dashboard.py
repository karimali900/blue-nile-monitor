#!/usr/bin/env python3
"""Serve the monitoring dashboard locally.

Usage:
    streamlit run dashboard/app.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import streamlit.web.cli as stcli

from nilemonitor.config import PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the Nile monitoring dashboard")
    parser.add_argument("--port", default=8501, type=int)
    parser.add_argument("--host", default="localhost")
    args = parser.parse_args()
    sys.argv = [
        "streamlit",
        "run",
        str(PROJECT_ROOT / "dashboard" / "app.py"),
        "--server.port",
        str(args.port),
        "--server.address",
        args.host,
    ]
    stcli.main()


if __name__ == "__main__":
    main()
