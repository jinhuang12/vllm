#!/usr/bin/env python3
"""GPU reservation module for AMMO benchmark orchestration.

Provides a file-locked, JSON-backed reservation system for NVIDIA GPUs.
State lives at /tmp/ammo_gpu_res/state.json (overridable via AMMO_GPU_RES_DIR).

Agents dynamically request N GPUs from the pool:
    CVD=$(python gpu_reservation.py reserve --num-gpus 2) && \
        CUDA_VISIBLE_DEVICES=$CVD python benchmark.py

Public API
----------
read_state()                              -> dict
reserve(num_gpus, session_id, ...)        -> list[int]
release_by_session(session_id)            -> list[int]
write_reservation(gpu_ids, session_id, ...) -> None   (internal helper)
force_clear(...)                          -> None
check_and_reclaim_expired()               -> list[int]

No GPU discovery or file creation occurs on import.  All state is initialised
lazily on the first read_state() call.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import IO


# ---------------------------------------------------------------------------
# Configuration (module-level constants — patchable in tests)
# ---------------------------------------------------------------------------

STATE_DIR: Path = Path(os.environ.get("AMMO_GPU_RES_DIR", "/tmp/ammo_gpu_res"))

# Delays between successive flock attempts (seconds).  4 delays → 5 attempts.
BACKOFF_DELAYS: list[float] = [0.1, 0.2, 0.4, 0.8]


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class LockTimeoutError(Exception):
    """Raised when the exclusive file lock cannot be acquired after all retries."""


class ReservationError(Exception):
    """Raised when a GPU reservation cannot be created because the GPU is held."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _lock_path() -> Path:
    return STATE_DIR / "state.lock"


def _state_path() -> Path:
    return STATE_DIR / "state.json"


def _discover_gpu_count() -> int:
    """Return the number of NVIDIA GPUs visible to nvidia-smi.

    Returns 0 on any failure (nvidia-smi not found, no GPUs, etc.).
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        lines = [l for l in result.stdout.splitlines() if l.strip()]
        return len(lines)
    except Exception:
        return 0


def _init_state() -> dict:
    """Create and persist an initial state.json based on discovered GPUs."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    gpu_count = _discover_gpu_count()
    state: dict = {
        "gpus": {str(i): None for i in range(gpu_count)},
        "gpu_count": gpu_count,
        "audit": [],
    }
    _write_state(state)
    return state


def _acquire_flock() -> IO:
    """Acquire an exclusive advisory lock on the lock file.

    Attempts up to len(BACKOFF_DELAYS)+1 times with exponential backoff.
    Returns the open file handle (caller must close / release it).
    Raises LockTimeoutError if all attempts fail.
    """
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    lock_file = _lock_path()
    attempts = len(BACKOFF_DELAYS) + 1  # e.g. 5 with default 4-element list

    fh = None
    for attempt in range(attempts):
        try:
            fh = open(lock_file, "w")
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return fh  # success
        except BlockingIOError:
            if fh is not None:
                fh.close()
                fh = None
            if attempt < len(BACKOFF_DELAYS):
                time.sleep(BACKOFF_DELAYS[attempt])
            # else: last attempt failed — fall through to raise below

    raise LockTimeoutError(
        f"Could not acquire GPU reservation lock after {attempts} attempts"
    )


def _release_flock(fh: IO) -> None:
    """Release and close the lock file handle."""
    try:
        fcntl.flock(fh, fcntl.LOCK_UN)
    finally:
        fh.close()


def _write_state(state: dict) -> None:
    """Atomically write state to state.json via a temp-file + rename."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    target = _state_path()
    fd, tmp_path = tempfile.mkstemp(dir=STATE_DIR, suffix=".json.tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp_path, target)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def read_state() -> dict:
    """Read and return the state dict, initialising state.json if absent."""
    state_file = _state_path()
    if not state_file.exists():
        return _init_state()
    return json.loads(state_file.read_text())


def write_reservation(
    gpu_ids: list[int],
    session_id: str,
    command_snippet: str = "",
    lease_hours: float = 2.0,
) -> None:
    """Reserve the given GPU IDs (internal helper called by reserve()).

    Acquires the exclusive lock, checks that each GPU is free (or expired),
    writes reservation entries, and releases the lock.

    Raises ReservationError if any GPU is currently held by a non-expired lease.
    """
    fh = _acquire_flock()
    try:
        state = read_state()
        now = datetime.now(tz=timezone.utc)

        # Validate that all requested GPU IDs exist in state
        known = set(state["gpus"].keys())
        unknown = [gid for gid in gpu_ids if str(gid) not in known]
        if unknown:
            raise ReservationError(f"GPU IDs not found in state: {unknown}")

        # Check all GPUs are free before writing any
        held: list[str] = []
        for gid in gpu_ids:
            key = str(gid)
            entry = state["gpus"].get(key)
            if entry is not None:
                # Check if the lease has expired
                expires = datetime.fromisoformat(entry["lease_expires"])
                if expires > now:
                    held.append(
                        f"GPU {gid} held by session={entry['session_id']} "
                        f"until {entry['lease_expires']}"
                    )
        if held:
            raise ReservationError(
                "Cannot reserve GPUs — already held:\n" + "\n".join(held)
            )

        # Write entries
        reserved_at = now.strftime("%Y-%m-%dT%H:%M:%SZ")
        lease_expires = (now + timedelta(hours=lease_hours)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        for gid in gpu_ids:
            state["gpus"][str(gid)] = {
                "session_id": session_id,
                "reserved_at": reserved_at,
                "lease_expires": lease_expires,
                "command_snippet": command_snippet[:80],
            }

        _write_state(state)
    finally:
        _release_flock(fh)


def reserve(
    num_gpus: int,
    session_id: str,
    command_snippet: str = "",
    lease_hours: float = 2.0,
) -> list[int]:
    """Dynamically reserve *num_gpus* GPUs from the pool.

    Uses contiguous-first allocation:
      - For num_gpus=1, takes the first free GPU.
      - For num_gpus>1, requires a contiguous block of free GPUs.

    Auto-releases any existing reservations for *session_id* before
    allocating, so retries don't exhaust the pool.

    Returns the list of allocated GPU IDs on success.
    Raises ReservationError if the request cannot be satisfied.
    """
    fh = _acquire_flock()
    try:
        state = read_state()
        now = datetime.now(tz=timezone.utc)
        now_str = now.strftime("%Y-%m-%dT%H:%M:%SZ")

        # --- Reclaim expired leases (inline) ---
        for key, entry in list(state["gpus"].items()):
            if entry is None:
                continue
            expires = datetime.fromisoformat(entry["lease_expires"])
            if expires <= now:
                state["audit"].append(
                    {
                        "action": "reclaim_expired",
                        "gpu_id": key,
                        "reclaimed_at": now_str,
                        "was_session": entry.get("session_id"),
                        "lease_expired_at": entry["lease_expires"],
                    }
                )
                state["gpus"][key] = None

        # --- Auto-release existing reservations for this session_id ---
        for key, entry in state["gpus"].items():
            if entry is not None and entry.get("session_id") == session_id:
                state["gpus"][key] = None

        # --- Find free GPUs ---
        gpu_count = state.get("gpu_count", len(state["gpus"]))
        free_set = set()
        for i in range(gpu_count):
            if state["gpus"].get(str(i)) is None:
                free_set.add(i)

        if len(free_set) < num_gpus:
            _write_state(state)
            raise ReservationError(
                f"Not enough free GPUs: requested {num_gpus}, "
                f"available {len(free_set)}"
            )

        # --- Contiguous-first allocation ---
        allocated: list[int] | None = None

        if num_gpus == 1:
            # Just take the first free GPU
            for i in range(gpu_count):
                if i in free_set:
                    allocated = [i]
                    break
        else:
            # Scan for a contiguous block of num_gpus consecutive free GPUs
            run_start = None
            run_len = 0
            for i in range(gpu_count):
                if i in free_set:
                    if run_start is None:
                        run_start = i
                        run_len = 1
                    else:
                        run_len += 1
                    if run_len >= num_gpus:
                        allocated = list(range(run_start, run_start + num_gpus))
                        break
                else:
                    run_start = None
                    run_len = 0

        if allocated is None:
            _write_state(state)
            raise ReservationError(
                f"No contiguous block of {num_gpus} free GPUs available. "
                f"Free GPUs: {sorted(free_set)}"
            )

        # --- Write reservation entries ---
        reserved_at = now_str
        lease_expires = (now + timedelta(hours=lease_hours)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        for gid in allocated:
            state["gpus"][str(gid)] = {
                "session_id": session_id,
                "reserved_at": reserved_at,
                "lease_expires": lease_expires,
                "command_snippet": command_snippet[:80],
            }

        _write_state(state)
        return allocated
    finally:
        _release_flock(fh)


def release_by_session(session_id: str) -> list[int]:
    """Clear all GPU entries owned by *session_id*.

    Returns the list of released GPU IDs (as integers).
    """
    fh = _acquire_flock()
    try:
        state = read_state()
        released: list[int] = []
        for key, entry in state["gpus"].items():
            if entry is not None and entry.get("session_id") == session_id:
                state["gpus"][key] = None
                released.append(int(key))
        if released:
            _write_state(state)
        return released
    finally:
        _release_flock(fh)


def force_clear(
    gpu_ids: list[int] | None = None,
    session_id: str | None = None,
    force_no_session: bool = False,
) -> None:
    """Forcibly clear reservation entries.

    Must provide either session_id or force_no_session=True.

    If session_id is given, clears only entries owned by that session.
    If gpu_ids is given together with session_id, further restricts to those GPUs.
    If force_no_session=True, clears all entries (or only gpu_ids if given).
    An audit log entry is written for each cleared GPU.

    Raises ValueError if neither session_id nor force_no_session=True is supplied.
    """
    if session_id is None and not force_no_session:
        raise ValueError(
            "force_clear requires either session_id or force_no_session=True"
        )

    fh = _acquire_flock()
    try:
        state = read_state()
        now_str = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        gpu_keys: list[str]
        if gpu_ids is not None:
            gpu_keys = [str(g) for g in gpu_ids]
        else:
            gpu_keys = list(state["gpus"].keys())

        for key in gpu_keys:
            entry = state["gpus"].get(key)
            if entry is None:
                continue
            # Filter by session_id if provided
            if session_id is not None and entry.get("session_id") != session_id:
                continue
            # Log to audit before clearing
            state["audit"].append(
                {
                    "action": "force_clear",
                    "gpu_id": key,
                    "cleared_at": now_str,
                    "was_session": entry.get("session_id"),
                }
            )
            state["gpus"][key] = None

        _write_state(state)
    finally:
        _release_flock(fh)


def check_and_reclaim_expired() -> list[int]:
    """Check for expired leases, clear them, write state, and return reclaimed IDs.

    Acquires the exclusive lock so it is safe to call standalone from hooks.
    Reads fresh state under the lock to avoid stale-state races.
    Returns the list of reclaimed GPU IDs (as integers).
    """
    fh = _acquire_flock()
    try:
        state = read_state()
        now = datetime.now(tz=timezone.utc)
        reclaimed: list[int] = []
        now_str = now.strftime("%Y-%m-%dT%H:%M:%SZ")

        for key, entry in state["gpus"].items():
            if entry is None:
                continue
            expires = datetime.fromisoformat(entry["lease_expires"])
            if expires <= now:
                state["audit"].append(
                    {
                        "action": "reclaim_expired",
                        "gpu_id": key,
                        "reclaimed_at": now_str,
                        "was_session": entry.get("session_id"),
                        "lease_expired_at": entry["lease_expires"],
                    }
                )
                state["gpus"][key] = None
                reclaimed.append(int(key))

        if reclaimed:
            _write_state(state)

        return reclaimed
    finally:
        _release_flock(fh)


# ---------------------------------------------------------------------------
# CLI interface
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AMMO GPU reservation pool manager"
    )
    sub = parser.add_subparsers(dest="command")

    # --- reserve ---
    p_reserve = sub.add_parser("reserve", help="Reserve N GPUs from the pool")
    p_reserve.add_argument(
        "--num-gpus", type=int, required=True, help="Number of GPUs to reserve"
    )
    p_reserve.add_argument(
        "--session-id", type=str, default="cli", help="Session identifier"
    )
    p_reserve.add_argument(
        "--lease-hours", type=float, default=2.0, help="Lease duration in hours"
    )
    p_reserve.add_argument(
        "--command-snippet", type=str, default="", help="Command description"
    )

    # --- release-session ---
    p_release = sub.add_parser(
        "release-session", help="Release all GPUs held by a session"
    )
    p_release.add_argument(
        "--session-id", type=str, required=True, help="Session identifier"
    )

    # --- status ---
    sub.add_parser("status", help="Print current reservation state as JSON")

    args = parser.parse_args()

    if args.command == "reserve":
        try:
            gpu_ids = reserve(
                num_gpus=args.num_gpus,
                session_id=args.session_id,
                command_snippet=args.command_snippet,
                lease_hours=args.lease_hours,
            )
            # Print comma-separated GPU IDs to stdout
            print(",".join(str(g) for g in gpu_ids))
            sys.exit(0)
        except (ReservationError, LockTimeoutError) as exc:
            print(str(exc), file=sys.stderr)
            sys.exit(1)

    elif args.command == "release-session":
        released = release_by_session(args.session_id)
        print(
            f"Released GPUs: {','.join(str(g) for g in released)}",
            file=sys.stderr,
        )

    elif args.command == "status":
        state = read_state()
        print(json.dumps(state, indent=2))

    else:
        parser.print_help()
        sys.exit(1)
