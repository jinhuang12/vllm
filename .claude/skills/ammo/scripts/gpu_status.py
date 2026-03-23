#!/usr/bin/env python3
"""GPU reservation status viewer.

Usage:
    python gpu_status.py          # human-readable table
    python gpu_status.py --json   # raw JSON of state.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from gpu_reservation import read_state  # noqa: E402


def _fmt(value: str | None, default: str = "-") -> str:
    return value if value is not None else default


def _iso_to_display(iso: str | None) -> str:
    """Convert ISO timestamp to a more readable local-ish form."""
    if iso is None:
        return "-"
    # Input is always UTC Z-suffix; strip and reformat for display
    try:
        from datetime import datetime, timezone
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        # Display in UTC without timezone suffix for brevity
        return dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return iso


def _truncate(s: str | None, width: int) -> str:
    if s is None:
        return "-"
    return s if len(s) <= width else s[: width - 1] + "…"


def print_table(state: dict) -> None:
    gpus = state.get("gpus", {})
    if not gpus:
        print("No GPUs tracked in state.")
        return

    # Column widths
    COL_GPU = 4
    COL_STATUS = 9
    COL_CMD = 42
    COL_SINCE = 19
    COL_EXPIRES = 19
    COL_SESSION = 16

    header = (
        f"{'GPU':<{COL_GPU}}  "
        f"{'Status':<{COL_STATUS}}  "
        f"{'Command':<{COL_CMD}}  "
        f"{'Since':<{COL_SINCE}}  "
        f"{'Lease Expires':<{COL_EXPIRES}}  "
        f"{'Session':<{COL_SESSION}}"
    )
    separator = "-" * len(header)
    print(header)
    print(separator)

    for gpu_id in sorted(gpus.keys(), key=lambda x: int(x)):
        entry = gpus[gpu_id]
        if entry is None:
            row = (
                f"{gpu_id:<{COL_GPU}}  "
                f"{'FREE':<{COL_STATUS}}  "
                f"{'-':<{COL_CMD}}  "
                f"{'-':<{COL_SINCE}}  "
                f"{'-':<{COL_EXPIRES}}  "
                f"{'-':<{COL_SESSION}}"
            )
        else:
            cmd = _truncate(entry.get("command_snippet"), COL_CMD)
            since = _iso_to_display(entry.get("reserved_at"))
            expires = _iso_to_display(entry.get("lease_expires"))
            session = _truncate(entry.get("session_id"), COL_SESSION)
            row = (
                f"{gpu_id:<{COL_GPU}}  "
                f"{'HELD':<{COL_STATUS}}  "
                f"{cmd:<{COL_CMD}}  "
                f"{since:<{COL_SINCE}}  "
                f"{expires:<{COL_EXPIRES}}  "
                f"{session:<{COL_SESSION}}"
            )
        print(row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Show GPU reservation status."
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON of state.json instead of a formatted table.",
    )
    args = parser.parse_args()

    state = read_state()

    if args.json:
        print(json.dumps(state, indent=2))
    else:
        print_table(state)


if __name__ == "__main__":
    main()
