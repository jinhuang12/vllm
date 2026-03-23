#!/usr/bin/env python3
"""Tests for gpu_reservation.py — the foundation GPU reservation module.

Run with:
    python -m pytest .claude/skills/ammo/tests/test_gpu_reservation.py -v
"""

from __future__ import annotations

import fcntl
import json
import multiprocessing
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

import pytest

# Ensure the scripts directory is importable.
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import gpu_reservation
from gpu_reservation import (
    LockTimeoutError,
    ReservationError,
    check_and_reclaim_expired,
    command_hash,
    force_clear,
    read_state,
    release_by_hash,
    write_reservation,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_temp_dir(tmp_path: Path) -> Path:
    """Return a subdirectory inside tmp_path that does NOT yet exist (so
    _init_state can create it)."""
    d = tmp_path / "ammo_gpu_res"
    return d


# ---------------------------------------------------------------------------
# TestInitState
# ---------------------------------------------------------------------------

class TestInitState:
    def test_creates_state_with_gpu_count(self, tmp_path):
        """_init_state with 4 discovered GPUs creates state with 4 null entries."""
        state_dir = _make_temp_dir(tmp_path)
        with mock.patch.object(gpu_reservation, "STATE_DIR", state_dir), \
             mock.patch("gpu_reservation._discover_gpu_count", return_value=4):
            state = read_state()

        assert state["gpu_count"] == 4
        assert len(state["gpus"]) == 4
        for v in state["gpus"].values():
            assert v is None

    def test_reads_existing_state(self, tmp_path):
        """read_state returns an already-written state without re-initialising."""
        state_dir = _make_temp_dir(tmp_path)
        state_dir.mkdir(parents=True, exist_ok=True)

        existing = {
            "gpus": {"0": None, "1": None},
            "gpu_count": 2,
            "audit": [],
        }
        state_file = state_dir / "state.json"
        state_file.write_text(json.dumps(existing))

        with mock.patch.object(gpu_reservation, "STATE_DIR", state_dir), \
             mock.patch("gpu_reservation._discover_gpu_count", return_value=99):
            # _discover_gpu_count should NOT be called because state already exists
            state = read_state()

        assert state["gpu_count"] == 2
        assert len(state["gpus"]) == 2


# ---------------------------------------------------------------------------
# TestAcquireFlock
# ---------------------------------------------------------------------------

class TestAcquireFlock:
    def test_acquires_lock_on_first_try(self, tmp_path):
        """Basic lock/unlock cycle succeeds without errors."""
        state_dir = _make_temp_dir(tmp_path)
        state_dir.mkdir(parents=True, exist_ok=True)
        lock_file = state_dir / "state.lock"

        with mock.patch.object(gpu_reservation, "STATE_DIR", state_dir):
            fh = gpu_reservation._acquire_flock()
            assert fh is not None
            gpu_reservation._release_flock(fh)

    def test_raises_after_5_retries(self, tmp_path):
        """LockTimeoutError raised when lock cannot be acquired after 5 attempts."""
        state_dir = _make_temp_dir(tmp_path)
        state_dir.mkdir(parents=True, exist_ok=True)
        lock_file = state_dir / "state.lock"

        # Hold the lock in this process
        holder = open(lock_file, "w")
        fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)

        try:
            with mock.patch.object(gpu_reservation, "STATE_DIR", state_dir), \
                 mock.patch.object(gpu_reservation, "BACKOFF_DELAYS", [0.001] * 4):
                with pytest.raises(LockTimeoutError):
                    gpu_reservation._acquire_flock()
        finally:
            fcntl.flock(holder, fcntl.LOCK_UN)
            holder.close()


# ---------------------------------------------------------------------------
# TestWriteReservation
# ---------------------------------------------------------------------------

class TestWriteReservation:
    def test_reserves_free_gpus(self, tmp_path):
        """Reserving free GPUs writes entries into state."""
        state_dir = _make_temp_dir(tmp_path)
        with mock.patch.object(gpu_reservation, "STATE_DIR", state_dir), \
             mock.patch("gpu_reservation._discover_gpu_count", return_value=4):
            write_reservation(
                gpu_ids=[0, 1],
                cmd_hash="aabbccdd11223344",
                session_id="sess001",
                cvd_requested="0,1",
                command_snippet="python bench.py",
                lease_hours=2.0,
            )
            state = read_state()

        assert state["gpus"]["0"] is not None
        assert state["gpus"]["0"]["command_hash"] == "aabbccdd11223344"
        assert state["gpus"]["0"]["session_id"] == "sess001"
        assert state["gpus"]["1"] is not None
        assert state["gpus"]["2"] is None
        assert state["gpus"]["3"] is None

    def test_blocks_on_held_gpu(self, tmp_path):
        """Trying to reserve an already-held GPU raises ReservationError."""
        state_dir = _make_temp_dir(tmp_path)
        with mock.patch.object(gpu_reservation, "STATE_DIR", state_dir), \
             mock.patch("gpu_reservation._discover_gpu_count", return_value=2):
            # First reservation succeeds
            write_reservation(
                gpu_ids=[0],
                cmd_hash="aaaa000011111111",
                session_id="sess_a",
                cvd_requested="0",
                command_snippet="python a.py",
                lease_hours=2.0,
            )
            # Second attempt on same GPU must fail
            with pytest.raises(ReservationError):
                write_reservation(
                    gpu_ids=[0],
                    cmd_hash="bbbb000022222222",
                    session_id="sess_b",
                    cvd_requested="0",
                    command_snippet="python b.py",
                    lease_hours=2.0,
                )


# ---------------------------------------------------------------------------
# TestReleaseByHash
# ---------------------------------------------------------------------------

class TestReleaseByHash:
    def test_releases_matching_hash(self, tmp_path):
        """release_by_hash clears entries whose command_hash matches."""
        state_dir = _make_temp_dir(tmp_path)
        h = "cccc000033333333"
        with mock.patch.object(gpu_reservation, "STATE_DIR", state_dir), \
             mock.patch("gpu_reservation._discover_gpu_count", return_value=2):
            write_reservation(
                gpu_ids=[0, 1],
                cmd_hash=h,
                session_id="sess_c",
                cvd_requested="0,1",
                command_snippet="python c.py",
                lease_hours=2.0,
            )
            release_by_hash(h)
            state = read_state()

        assert state["gpus"]["0"] is None
        assert state["gpus"]["1"] is None


# ---------------------------------------------------------------------------
# TestLeaseExpiry
# ---------------------------------------------------------------------------

class TestLeaseExpiry:
    def test_reclaims_expired_leases(self, tmp_path):
        """check_and_reclaim_expired removes entries with lease_hours=0 (already expired)."""
        state_dir = _make_temp_dir(tmp_path)
        with mock.patch.object(gpu_reservation, "STATE_DIR", state_dir), \
             mock.patch("gpu_reservation._discover_gpu_count", return_value=2):
            # lease_hours=0 means it expires immediately (at the time of writing)
            write_reservation(
                gpu_ids=[0],
                cmd_hash="dddd000044444444",
                session_id="sess_d",
                cvd_requested="0",
                command_snippet="python d.py",
                lease_hours=0.0,
            )
            # Small sleep to ensure expiry has passed
            time.sleep(0.01)
            state = read_state()
            reclaimed = check_and_reclaim_expired(state)

        assert 0 in reclaimed
        assert state["gpus"]["0"] is None


# ---------------------------------------------------------------------------
# TestForceClear
# ---------------------------------------------------------------------------

class TestForceClear:
    def test_clears_by_session_id(self, tmp_path):
        """force_clear(session_id='s1') removes only entries owned by s1."""
        state_dir = _make_temp_dir(tmp_path)
        with mock.patch.object(gpu_reservation, "STATE_DIR", state_dir), \
             mock.patch("gpu_reservation._discover_gpu_count", return_value=2):
            write_reservation(
                gpu_ids=[0],
                cmd_hash="eeee000055555555",
                session_id="s1",
                cvd_requested="0",
                command_snippet="python e.py",
                lease_hours=2.0,
            )
            write_reservation(
                gpu_ids=[1],
                cmd_hash="ffff000066666666",
                session_id="s2",
                cvd_requested="1",
                command_snippet="python f.py",
                lease_hours=2.0,
            )
            force_clear(session_id="s1")
            state = read_state()

        assert state["gpus"]["0"] is None, "GPU 0 (owned by s1) should be cleared"
        assert state["gpus"]["1"] is not None, "GPU 1 (owned by s2) should remain"

    def test_requires_session_or_flag(self, tmp_path):
        """force_clear() with no args raises ValueError."""
        state_dir = _make_temp_dir(tmp_path)
        with mock.patch.object(gpu_reservation, "STATE_DIR", state_dir), \
             mock.patch("gpu_reservation._discover_gpu_count", return_value=1):
            with pytest.raises(ValueError):
                force_clear()


# ---------------------------------------------------------------------------
# TestCommandHash
# ---------------------------------------------------------------------------

class TestCommandHash:
    def test_deterministic(self):
        """Same input always produces the same hash."""
        h1 = command_hash("python bench.py --model foo")
        h2 = command_hash("python bench.py --model foo")
        assert h1 == h2

    def test_16_chars(self):
        """Hash output is exactly 16 hex characters."""
        h = command_hash("anything")
        assert len(h) == 16
        assert all(c in "0123456789abcdef" for c in h)
