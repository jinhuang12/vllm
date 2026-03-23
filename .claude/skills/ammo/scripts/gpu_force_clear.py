#!/usr/bin/env python3
"""Force-clear GPU reservations.

Usage:
    python gpu_force_clear.py --gpu-ids 0,1 --session-id <id>
    python gpu_force_clear.py --all --session-id <id>
    python gpu_force_clear.py --all --force-no-session
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from gpu_reservation import force_clear  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Force-clear GPU reservations."
    )

    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument(
        "--gpu-ids",
        metavar="IDS",
        help="Comma-separated GPU IDs to clear (e.g. 0,1).",
    )
    target_group.add_argument(
        "--all",
        action="store_true",
        help="Clear reservations on all GPUs.",
    )

    session_group = parser.add_mutually_exclusive_group(required=True)
    session_group.add_argument(
        "--session-id",
        metavar="ID",
        help="Only clear reservations owned by this session ID.",
    )
    session_group.add_argument(
        "--force-no-session",
        action="store_true",
        help="Emergency override: clear without session check.",
    )

    args = parser.parse_args()

    # Resolve GPU IDs (None means all)
    if args.all:
        gpu_ids = None
    else:
        try:
            gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(",")]
        except ValueError:
            print(
                f"ERROR: --gpu-ids must be comma-separated integers, got: {args.gpu_ids!r}",
                file=sys.stderr,
            )
            sys.exit(1)

    if args.force_no_session:
        print(
            "WARNING: --force-no-session is an emergency override. "
            "All matched reservations will be cleared regardless of session ownership.",
            file=sys.stderr,
        )

    try:
        force_clear(
            gpu_ids=gpu_ids,
            session_id=args.session_id if not args.force_no_session else None,
            force_no_session=args.force_no_session,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    print("GPU reservations cleared.")


if __name__ == "__main__":
    main()
