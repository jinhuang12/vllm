#!/usr/bin/env python3
"""GPU reservation module for AMMO benchmark orchestration.

Provides a file-locked, JSON-backed reservation system for NVIDIA GPUs.
State lives at /tmp/ammo_gpu_res/state.json (overridable via AMMO_GPU_RES_DIR).

Public API
----------
read_state()                     -> dict
write_reservation(...)           -> None
release_by_hash(cmd_hash)        -> None
force_clear(...)                 -> None
check_and_reclaim_expired(state) -> list[int]
command_hash(command)            -> str

No GPU discovery or file creation occurs on import.  All state is initialised
lazily on the first read_state() call.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import subprocess
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
    cmd_hash: str,
    session_id: str,
    cvd_requested: str,
    command_snippet: str,
    lease_hours: float = 2.0,
) -> None:
    """Reserve the given GPU IDs.

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
                        f"hash={entry['command_hash']} until {entry['lease_expires']}"
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
                "command_hash": cmd_hash,
                "session_id": session_id,
                "reserved_at": reserved_at,
                "lease_expires": lease_expires,
                "cvd_requested": cvd_requested,
                "command_snippet": command_snippet[:80],
            }

        _write_state(state)
    finally:
        _release_flock(fh)


def release_by_hash(cmd_hash: str) -> None:
    """Clear all GPU entries whose command_hash matches cmd_hash."""
    fh = _acquire_flock()
    try:
        state = read_state()
        for key, entry in state["gpus"].items():
            if entry is not None and entry.get("command_hash") == cmd_hash:
                state["gpus"][key] = None
        _write_state(state)
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
                    "was_hash": entry.get("command_hash"),
                }
            )
            state["gpus"][key] = None

        _write_state(state)
    finally:
        _release_flock(fh)


def check_and_reclaim_expired(state: dict) -> list[int]:
    """Check for expired leases, clear them in-place, write state, and return reclaimed IDs.

    Acquires the exclusive lock so it is safe to call standalone from hooks.
    Modifies *state* in place and persists the changes.
    Returns the list of reclaimed GPU IDs (as integers).
    """
    fh = _acquire_flock()
    try:
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
                        "was_hash": entry.get("command_hash"),
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


def command_hash(command: str) -> str:
    """Return the first 16 hex characters of the SHA-256 hash of *command*."""
    return hashlib.sha256(command.encode()).hexdigest()[:16]
