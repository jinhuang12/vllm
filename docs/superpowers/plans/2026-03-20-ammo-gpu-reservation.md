# AMMO GPU Reservation System — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent GPU contention during AMMO Stages 4-5 by adding hook-managed per-call GPU reservations and TP-aware orchestrator GPU partitioning.

**Architecture:** PreToolUse hook auto-reserves GPUs when a command includes `CUDA_VISIBLE_DEVICES=X`, PostToolUse auto-releases when the command completes. The orchestrator computes GPU partitions at track spawn time (exclusive if enough GPUs, shared otherwise) and passes assignments in spawn prompts. A shared Python module handles state I/O with flock-based atomicity and lease-based crash recovery.

**Tech Stack:** Python 3.12 (shared module), Bash (hooks), JSON (state), `fcntl.flock` (atomicity), `hashlib.sha256` (command hashing)

**Spec:** `docs/superpowers/specs/2026-03-20-ammo-gpu-reservation-design.md`

---

## File Structure

```
New files:
  .claude/skills/ammo/scripts/gpu_reservation.py   — shared module (state I/O, flock, lease)
  .claude/skills/ammo/scripts/gpu_status.py         — diagnostic CLI
  .claude/skills/ammo/scripts/gpu_force_clear.py    — crash recovery CLI
  .claude/hooks/ammo-gpu-release.sh                 — PostToolUse hook

Modified files:
  .claude/hooks/ammo-pretool-guard.sh               — add GPU auto-reserve section
  .claude/hooks/ammo-postcompact.sh                 — add flag file cleanup on SessionStart
  .claude/settings.local.json                       — register PostToolUse hook
  .claude/skills/ammo/scripts/run_vllm_bench_latency_sweep.py — remove old locking
  .claude/skills/ammo/SKILL.md                      — doc updates
  .claude/skills/ammo/orchestration/parallel-tracks.md — TP-aware partitioning
  .claude/agents/ammo-impl-champion.md              — GPU usage instructions
  .claude/agents/ammo-impl-validator.md             — GPU usage instructions
```

---

### Task 1: Shared Module — `gpu_reservation.py`

The foundation. All hooks and scripts import from this module.

**Files:**
- Create: `.claude/skills/ammo/scripts/gpu_reservation.py`
- Test: `.claude/skills/ammo/tests/test_gpu_reservation.py`

- [ ] **Step 1: Write tests for `_init_state()` and `read_state()`**

```python
# .claude/skills/ammo/tests/test_gpu_reservation.py
"""Tests for gpu_reservation shared module."""
import json
import os
import tempfile
import pytest
from unittest import mock

# The module under test
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))


class TestInitState:
    """Tests for _init_state() — GPU discovery and initial state creation."""

    def test_creates_state_with_gpu_count(self, tmp_path):
        """nvidia-smi reports 4 GPUs → state has 4 null entries."""
        from gpu_reservation import _init_state
        with mock.patch('gpu_reservation.STATE_DIR', tmp_path):
            with mock.patch('gpu_reservation._discover_gpu_count', return_value=4):
                state = _init_state()
        assert state['gpu_count'] == 4
        assert all(state['gpus'][str(i)] is None for i in range(4))
        assert state['audit'] == []

    def test_reads_existing_state(self, tmp_path):
        """If state.json already exists, read it instead of reinitializing."""
        from gpu_reservation import read_state
        state_file = tmp_path / 'state.json'
        existing = {'gpus': {'0': None, '1': None}, 'gpu_count': 2, 'audit': []}
        state_file.write_text(json.dumps(existing))
        with mock.patch('gpu_reservation.STATE_DIR', tmp_path):
            state = read_state()
        assert state['gpu_count'] == 2


class TestAcquireFlock:
    """Tests for _acquire_flock() — exponential backoff + LockTimeoutError."""

    def test_acquires_lock_on_first_try(self, tmp_path):
        from gpu_reservation import _acquire_flock, _release_flock
        with mock.patch('gpu_reservation.STATE_DIR', tmp_path):
            fh = _acquire_flock()
            assert fh is not None
            _release_flock(fh)

    def test_raises_after_5_retries(self, tmp_path):
        """If lock is held, raises LockTimeoutError after retries."""
        import fcntl
        from gpu_reservation import _acquire_flock, LockTimeoutError
        lock_path = tmp_path / '.lock'
        # Hold the lock externally
        blocker = open(lock_path, 'w')
        fcntl.flock(blocker.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            with mock.patch('gpu_reservation.STATE_DIR', tmp_path):
                with mock.patch('gpu_reservation.BACKOFF_DELAYS', [0.001] * 5):
                    with pytest.raises(LockTimeoutError):
                        _acquire_flock()
        finally:
            fcntl.flock(blocker.fileno(), fcntl.LOCK_UN)
            blocker.close()
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
cd /home/jinhun/vllm && source .venv/bin/activate
python -m pytest .claude/skills/ammo/tests/test_gpu_reservation.py -v 2>&1 | head -30
```
Expected: `ModuleNotFoundError: No module named 'gpu_reservation'`

- [ ] **Step 3: Implement `gpu_reservation.py` — core state I/O + flock**

```python
# .claude/skills/ammo/scripts/gpu_reservation.py
"""AMMO GPU reservation shared module.

Provides atomic state I/O for per-call GPU reservations managed by hooks.
State lives at /tmp/ammo_gpu_res/state.json (machine-scoped, local filesystem).
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import IO, Any

STATE_DIR = Path(os.environ.get('AMMO_GPU_RES_DIR', '/tmp/ammo_gpu_res'))
BACKOFF_DELAYS = [0.1, 0.2, 0.4, 0.8]  # 4 delays for 5 attempts


class LockTimeoutError(Exception):
    """Raised when flock cannot be acquired after retries."""
    pass


class ReservationError(Exception):
    """Raised when GPU reservation fails (GPUs held by another command)."""
    pass


def _discover_gpu_count() -> int:
    """Discover GPU count via nvidia-smi."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '-L'],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            return len([l for l in result.stdout.strip().splitlines() if l.strip()])
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return 0


def _init_state() -> dict:
    """Create initial state.json with discovered GPU count."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    gpu_count = _discover_gpu_count()
    state = {
        'gpus': {str(i): None for i in range(gpu_count)},
        'gpu_count': gpu_count,
        'audit': [],
    }
    state_path = STATE_DIR / 'state.json'
    state_path.write_text(json.dumps(state, indent=2) + '\n')
    return state


def read_state() -> dict:
    """Read state.json, initializing if missing."""
    state_path = STATE_DIR / 'state.json'
    if not state_path.exists():
        return _init_state()
    return json.loads(state_path.read_text())


def _write_state(state: dict) -> None:
    """Write state.json atomically (caller must hold flock)."""
    state_path = STATE_DIR / 'state.json'
    tmp_path = state_path.with_suffix('.tmp')
    tmp_path.write_text(json.dumps(state, indent=2) + '\n')
    tmp_path.rename(state_path)


def _acquire_flock() -> IO:
    """Acquire exclusive flock with exponential backoff. Raises LockTimeoutError."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    lock_path = STATE_DIR / '.lock'
    fh = open(lock_path, 'w')
    for attempt in range(len(BACKOFF_DELAYS) + 1):
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return fh
        except BlockingIOError:
            if attempt < len(BACKOFF_DELAYS):
                time.sleep(BACKOFF_DELAYS[attempt])
    fh.close()
    raise LockTimeoutError(
        'GPU reservation lock timeout — system is busy. Retry in a few seconds.'
    )


def _release_flock(fh: IO) -> None:
    """Release flock and close file handle."""
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    finally:
        fh.close()


def command_hash(command: str) -> str:
    """SHA256 hash of command string, first 16 hex chars (64 bits)."""
    return hashlib.sha256(command.encode()).hexdigest()[:16]


def check_and_reclaim_expired(state: dict) -> list[int]:
    """Check all reservations for lease expiry. Returns list of reclaimed GPU IDs."""
    now = datetime.now(timezone.utc)
    reclaimed = []
    for gpu_id, entry in state['gpus'].items():
        if entry is None:
            continue
        expires = datetime.fromisoformat(entry['lease_expires'])
        if now > expires:
            state['audit'].append({
                'timestamp': now.isoformat(),
                'action': 'lease_expired',
                'gpu_id': int(gpu_id),
                'command_hash': entry.get('command_hash', ''),
                'session_id': entry.get('session_id', ''),
            })
            state['gpus'][gpu_id] = None
            reclaimed.append(int(gpu_id))
    return reclaimed


def write_reservation(
    gpu_ids: list[int],
    cmd_hash: str,
    session_id: str,
    cvd_requested: str,
    command_snippet: str,
    lease_hours: float = 2.0,
) -> None:
    """Write reservation for given GPUs. Acquires flock internally."""
    fh = _acquire_flock()
    try:
        state = read_state()
        reclaimed = check_and_reclaim_expired(state)

        now = datetime.now(timezone.utc)
        expires = now + timedelta(hours=lease_hours)

        # Check all requested GPUs are free
        held_by = []
        for gid in gpu_ids:
            entry = state['gpus'].get(str(gid))
            if entry is not None:
                held_by.append((gid, entry))

        if held_by:
            msgs = []
            for gid, entry in held_by:
                msgs.append(
                    f"GPU {gid} held by command: {entry.get('command_snippet', '?')} "
                    f"(since {entry.get('reserved_at', '?')})"
                )
            raise ReservationError('\n'.join(msgs))

        # Write reservations
        entry = {
            'command_hash': cmd_hash,
            'session_id': session_id,
            'reserved_at': now.isoformat(),
            'lease_expires': expires.isoformat(),
            'cvd_requested': cvd_requested,
            'command_snippet': command_snippet[:80],
        }
        for gid in gpu_ids:
            state['gpus'][str(gid)] = dict(entry)

        _write_state(state)
    finally:
        _release_flock(fh)


def release_by_hash(cmd_hash: str) -> None:
    """Release all GPU entries matching the given command hash."""
    fh = _acquire_flock()
    try:
        state = read_state()
        for gpu_id, entry in state['gpus'].items():
            if entry is not None and entry.get('command_hash') == cmd_hash:
                state['gpus'][gpu_id] = None
        _write_state(state)
    finally:
        _release_flock(fh)


def force_clear(
    gpu_ids: list[int] | None = None,
    session_id: str | None = None,
    force_no_session: bool = False,
) -> None:
    """Force-clear reservations. Requires session_id or force_no_session."""
    if not session_id and not force_no_session:
        raise ValueError('Must provide --session-id or --force-no-session')

    fh = _acquire_flock()
    try:
        state = read_state()
        now = datetime.now(timezone.utc)
        cleared = []

        for gid, entry in state['gpus'].items():
            if entry is None:
                continue
            if gpu_ids is not None and int(gid) not in gpu_ids:
                continue
            if session_id and entry.get('session_id') != session_id:
                continue
            cleared.append(int(gid))
            state['gpus'][gid] = None

        for gid in cleared:
            state['audit'].append({
                'timestamp': now.isoformat(),
                'action': 'force_clear',
                'gpu_id': gid,
                'session_id': session_id or 'ALL',
                'force_no_session': force_no_session,
            })

        _write_state(state)
    finally:
        _release_flock(fh)
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
python -m pytest .claude/skills/ammo/tests/test_gpu_reservation.py -v
```
Expected: All tests PASS.

- [ ] **Step 5: Write tests for `write_reservation`, `release_by_hash`, `check_and_reclaim_expired`**

```python
# Append to test_gpu_reservation.py

class TestWriteReservation:
    def test_reserves_free_gpus(self, tmp_path):
        from gpu_reservation import write_reservation, read_state, _init_state
        with mock.patch('gpu_reservation.STATE_DIR', tmp_path):
            with mock.patch('gpu_reservation._discover_gpu_count', return_value=4):
                _init_state()
                write_reservation(
                    gpu_ids=[0, 1], cmd_hash='abc123', session_id='sess1',
                    cvd_requested='0,1', command_snippet='python bench.py'
                )
                state = read_state()
                assert state['gpus']['0'] is not None
                assert state['gpus']['0']['command_hash'] == 'abc123'
                assert state['gpus']['1'] is not None
                assert state['gpus']['2'] is None

    def test_blocks_on_held_gpu(self, tmp_path):
        from gpu_reservation import (
            write_reservation, read_state, _init_state, ReservationError
        )
        with mock.patch('gpu_reservation.STATE_DIR', tmp_path):
            with mock.patch('gpu_reservation._discover_gpu_count', return_value=4):
                _init_state()
                write_reservation(
                    gpu_ids=[0], cmd_hash='first', session_id='s1',
                    cvd_requested='0', command_snippet='cmd1'
                )
                with pytest.raises(ReservationError, match='GPU 0 held'):
                    write_reservation(
                        gpu_ids=[0], cmd_hash='second', session_id='s2',
                        cvd_requested='0', command_snippet='cmd2'
                    )


class TestReleaseByHash:
    def test_releases_matching_hash(self, tmp_path):
        from gpu_reservation import (
            write_reservation, release_by_hash, read_state, _init_state
        )
        with mock.patch('gpu_reservation.STATE_DIR', tmp_path):
            with mock.patch('gpu_reservation._discover_gpu_count', return_value=4):
                _init_state()
                write_reservation(
                    gpu_ids=[0, 1], cmd_hash='abc123', session_id='s1',
                    cvd_requested='0,1', command_snippet='bench'
                )
                release_by_hash('abc123')
                state = read_state()
                assert state['gpus']['0'] is None
                assert state['gpus']['1'] is None


class TestLeaseExpiry:
    def test_reclaims_expired_leases(self, tmp_path):
        from gpu_reservation import (
            write_reservation, check_and_reclaim_expired, read_state, _init_state
        )
        with mock.patch('gpu_reservation.STATE_DIR', tmp_path):
            with mock.patch('gpu_reservation._discover_gpu_count', return_value=2):
                _init_state()
                write_reservation(
                    gpu_ids=[0], cmd_hash='old', session_id='s1',
                    cvd_requested='0', command_snippet='old cmd',
                    lease_hours=0.0  # expires immediately
                )
                state = read_state()
                reclaimed = check_and_reclaim_expired(state)
                assert 0 in reclaimed
                assert state['gpus']['0'] is None
```

- [ ] **Step 6: Run all tests — verify they pass**

```bash
python -m pytest .claude/skills/ammo/tests/test_gpu_reservation.py -v
```

- [ ] **Step 7: Commit**

```bash
git add .claude/skills/ammo/scripts/gpu_reservation.py .claude/skills/ammo/tests/test_gpu_reservation.py
git commit -m "feat(ammo): add gpu_reservation shared module with state I/O, flock, and lease logic"
```

---

### Task 2: PreToolUse Hook — GPU Auto-Reserve

Extends the existing `ammo-pretool-guard.sh` with GPU reservation logic.

**Files:**
- Modify: `.claude/hooks/ammo-pretool-guard.sh`
- Test: Manual testing via `echo '{"tool_input":{"command":"..."}}' | bash .claude/hooks/ammo-pretool-guard.sh`

- [ ] **Step 1: Read the existing hook to understand structure**

```bash
cat -n .claude/hooks/ammo-pretool-guard.sh
```

The hook already has: JSON input parsing (line 15), fast bail for non-AMMO context (line 20), read-only command skip (line 23), and production-parity checks (lines 28-44).

- [ ] **Step 2: Add GPU detection patterns and CVD extraction after the existing checks**

Append to `ammo-pretool-guard.sh` (before the final `exit 0`):

```bash
# ── GPU Reservation Auto-Reserve ──
# Detects GPU commands and manages per-call reservations via state.json

GPU_RES_DIR="${AMMO_GPU_RES_DIR:-/tmp/ammo_gpu_res}"
GPU_STATE="$GPU_RES_DIR/state.json"

# Skip if reservation system not initialized
[ -f "$GPU_STATE" ] || exit 0

# GPU detection patterns (conservative)
IS_GPU_CMD=false

# Pattern 1: python/pytest with GPU keywords
if echo "$COMMAND" | grep -qP '(python|pytest)\b' && \
   echo "$COMMAND" | grep -qiP '(torch|cuda|triton|vllm|benchmark|kernel|gpu)'; then
    IS_GPU_CMD=true
fi

# Pattern 2: inline python with GPU imports
if echo "$COMMAND" | grep -qP 'python\s+-c\b' && \
   echo "$COMMAND" | grep -qiP '(torch|cuda|triton)'; then
    IS_GPU_CMD=true
fi

# Pattern 3: GPU profiling tools
if echo "$COMMAND" | grep -qP '^\s*(nsys|ncu)\b'; then
    IS_GPU_CMD=true
fi
if echo "$COMMAND" | grep -qP 'nvidia-smi\b.*--query-compute'; then
    IS_GPU_CMD=true
fi

# Exemptions: bare import checks (worktree setup)
if echo "$COMMAND" | grep -qP "python\s+-c\s+['\"]import (vllm|torch)['\"]"; then
    IS_GPU_CMD=false
fi

[ "$IS_GPU_CMD" = "false" ] && exit 0

# ── Extract CVD from command ──
CVD_VALUE=""
if echo "$COMMAND" | grep -qP 'CUDA_VISIBLE_DEVICES=([\d,]+)'; then
    CVD_VALUE=$(echo "$COMMAND" | grep -oP 'CUDA_VISIBLE_DEVICES=\K[\d,]+')
fi
CVD_EMPTY=false
if echo "$COMMAND" | grep -qP 'CUDA_VISIBLE_DEVICES=(["\x27]{2}|""|$|\s)'; then
    CVD_EMPTY=true
fi

# ── Case A: CVD with GPU IDs → auto-reserve ──
if [ -n "$CVD_VALUE" ]; then
    SESSION_ID="${CLAUDE_SESSION_ID:-unknown}"
    CMD_HASH=$(echo -n "$COMMAND" | python3 -c "import sys,hashlib; print(hashlib.sha256(sys.stdin.read().encode()).hexdigest()[:16])")
    SNIPPET=$(echo "$COMMAND" | head -c 80)

    # Detect nsys for longer lease
    LEASE_HOURS=2.0
    if echo "$COMMAND" | grep -qP '\bnsys\b'; then
        LEASE_HOURS=4.0
    fi

    # Call shared module to write reservation
    RESULT=$(python3 -c "
import sys, os
sys.path.insert(0, os.path.join('$PROJECT_DIR', '.claude/skills/ammo/scripts'))
os.environ.setdefault('AMMO_GPU_RES_DIR', '$GPU_RES_DIR')
from gpu_reservation import write_reservation, ReservationError, LockTimeoutError
try:
    write_reservation(
        gpu_ids=[int(x) for x in '$CVD_VALUE'.split(',')],
        cmd_hash='$CMD_HASH',
        session_id='$SESSION_ID',
        cvd_requested='$CVD_VALUE',
        command_snippet='''$SNIPPET''',
        lease_hours=$LEASE_HOURS,
    )
    print('OK')
except ReservationError as e:
    print(f'HELD:{e}')
except LockTimeoutError as e:
    print(f'LOCK_TIMEOUT:{e}')
except Exception as e:
    print(f'ERROR:{e}')
" 2>&1)

    case "$RESULT" in
        OK)
            exit 0
            ;;
        HELD:*)
            echo "${RESULT#HELD:}" >&2
            echo "Retry shortly." >&2
            exit 2
            ;;
        LOCK_TIMEOUT:*)
            echo "${RESULT#LOCK_TIMEOUT:}" >&2
            exit 2
            ;;
        *)
            echo "AMMO GPU WARNING: Reservation error: $RESULT" >&2
            exit 0  # fail-open on unexpected errors
            ;;
    esac
fi

# ── Case B: CVD="" → no GPU needed, skip reservation ──
if [ "$CVD_EMPTY" = "true" ]; then
    # Sanity check: warn if command looks GPU-heavy
    if echo "$COMMAND" | grep -qiP '(run_vllm_bench|benchmark_kernel|nsys\s+profile|ncu\b)'; then
        echo "AMMO GPU WARNING: CVD is empty but command looks GPU-intensive. Verify this is intentional." >&2
    fi
    exit 0
fi

# ── Case C: No CVD → one-shot block ──
SESSION_HASH="${CLAUDE_SESSION_ID:-}"
if [ -z "$SESSION_HASH" ]; then
    exit 0  # fail-open if session ID not available
fi
FLAG_FILE="$GPU_RES_DIR/.warned_${SESSION_HASH}"
if [ -f "$FLAG_FILE" ]; then
    exit 0  # already warned this session
fi

# First time — block
mkdir -p "$GPU_RES_DIR"
touch "$FLAG_FILE"
echo "AMMO GPU: Command matches GPU pattern but has no CUDA_VISIBLE_DEVICES prefix." >&2
echo "Add CUDA_VISIBLE_DEVICES=X (from your spawn prompt) for GPU work," >&2
echo "or CUDA_VISIBLE_DEVICES=\"\" if this command does not use GPUs." >&2
exit 2
```

- [ ] **Step 3: Test the hook manually**

```bash
# Case A: CVD with GPU IDs (should reserve and pass)
echo '{"tool_input":{"command":"CUDA_VISIBLE_DEVICES=0 python benchmark.py"}}' | \
  CLAUDE_PROJECT_DIR=/home/jinhun/vllm CLAUDE_SESSION_ID=test123 \
  bash .claude/hooks/ammo-pretool-guard.sh; echo "Exit: $?"

# Case B: CVD empty (should pass immediately)
echo '{"tool_input":{"command":"CUDA_VISIBLE_DEVICES=\"\" python cpu_script.py"}}' | \
  CLAUDE_PROJECT_DIR=/home/jinhun/vllm CLAUDE_SESSION_ID=test123 \
  bash .claude/hooks/ammo-pretool-guard.sh; echo "Exit: $?"

# Case C: No CVD (should block first time, exit 2)
echo '{"tool_input":{"command":"python benchmark.py"}}' | \
  CLAUDE_PROJECT_DIR=/home/jinhun/vllm CLAUDE_SESSION_ID=test456 \
  bash .claude/hooks/ammo-pretool-guard.sh; echo "Exit: $?"
```

- [ ] **Step 4: Commit**

```bash
git add .claude/hooks/ammo-pretool-guard.sh
git commit -m "feat(ammo): add GPU auto-reserve to PreToolUse hook (Cases A/B/C)"
```

---

### Task 3: PostToolUse Hook — GPU Auto-Release

**Files:**
- Create: `.claude/hooks/ammo-gpu-release.sh`

- [ ] **Step 1: Create the PostToolUse hook**

```bash
#!/bin/bash
# PostToolUse hook — AMMO GPU reservation auto-release.
#
# After a Bash command completes, releases any GPU reservation that was
# made by the PreToolUse hook for that command. Matches by command_hash.
set -euo pipefail
if ! command -v jq &>/dev/null; then exit 0; fi

GPU_RES_DIR="${AMMO_GPU_RES_DIR:-/tmp/ammo_gpu_res}"
GPU_STATE="$GPU_RES_DIR/state.json"
[ -f "$GPU_STATE" ] || exit 0

INPUT=$(cat)
# Try both field names for command extraction
COMMAND=$(echo "$INPUT" | jq -r '.tool_input.command // .input.command // empty' 2>/dev/null) || true
PROJECT_DIR="${CLAUDE_PROJECT_DIR:-.}"

if [ -z "$COMMAND" ]; then
    # Fallback: cannot extract command — log warning, let lease handle cleanup
    echo "AMMO GPU PostToolUse: could not extract command — reservation not released. Will expire via lease." >&2
    exit 0
fi

# Only release for commands with CUDA_VISIBLE_DEVICES=<digits>
if ! echo "$COMMAND" | grep -qP 'CUDA_VISIBLE_DEVICES=[\d,]+'; then
    exit 0
fi

# Compute command hash and release
CMD_HASH=$(echo -n "$COMMAND" | python3 -c "import sys,hashlib; print(hashlib.sha256(sys.stdin.read().encode()).hexdigest()[:16])")

python3 -c "
import sys, os
sys.path.insert(0, os.path.join('$PROJECT_DIR', '.claude/skills/ammo/scripts'))
os.environ.setdefault('AMMO_GPU_RES_DIR', '$GPU_RES_DIR')
from gpu_reservation import release_by_hash
release_by_hash('$CMD_HASH')
" 2>&1 || echo "AMMO GPU PostToolUse: release failed — will expire via lease." >&2

exit 0
```

- [ ] **Step 2: Make executable**

```bash
chmod +x .claude/hooks/ammo-gpu-release.sh
```

- [ ] **Step 3: Test manually**

```bash
# Simulate PostToolUse after a GPU command
echo '{"tool_input":{"command":"CUDA_VISIBLE_DEVICES=0 python benchmark.py"}}' | \
  CLAUDE_PROJECT_DIR=/home/jinhun/vllm \
  bash .claude/hooks/ammo-gpu-release.sh; echo "Exit: $?"
```

- [ ] **Step 4: Commit**

```bash
git add .claude/hooks/ammo-gpu-release.sh
git commit -m "feat(ammo): add PostToolUse hook for GPU auto-release"
```

---

### Task 4: Hook Registration + SessionStart Cleanup

**Files:**
- Modify: `.claude/settings.local.json`
- Modify: `.claude/hooks/ammo-postcompact.sh`

- [ ] **Step 1: Read current settings.local.json**

```bash
cat .claude/settings.local.json
```

- [ ] **Step 2: Add PostToolUse hook registration**

Add the `PostToolUse` section to the `hooks` object in `.claude/settings.local.json`:

```json
"PostToolUse": [
  {
    "matcher": "Bash",
    "hooks": [
      {
        "type": "command",
        "command": "$CLAUDE_PROJECT_DIR/.claude/hooks/ammo-gpu-release.sh"
      }
    ]
  }
]
```

- [ ] **Step 3: Add flag file cleanup to SessionStart hook**

Append to `.claude/hooks/ammo-postcompact.sh` — add cleanup of one-shot warning flags before the final `fi`:

```bash
# Clean up stale GPU reservation warning flags from previous sessions
rm -f /tmp/ammo_gpu_res/.warned_* 2>/dev/null || true
```

- [ ] **Step 4: Commit**

```bash
git add .claude/settings.local.json .claude/hooks/ammo-postcompact.sh
git commit -m "feat(ammo): register PostToolUse hook + clean up GPU flags on SessionStart"
```

---

### Task 5: Diagnostic Scripts — `gpu_status.py` + `gpu_force_clear.py`

**Files:**
- Create: `.claude/skills/ammo/scripts/gpu_status.py`
- Create: `.claude/skills/ammo/scripts/gpu_force_clear.py`

- [ ] **Step 1: Create `gpu_status.py`**

```python
#!/usr/bin/env python3
"""Print current AMMO GPU reservation state."""
from __future__ import annotations
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from gpu_reservation import read_state

def main():
    parser = argparse.ArgumentParser(description='AMMO GPU reservation status')
    parser.add_argument('--json', action='store_true', help='Output raw JSON')
    args = parser.parse_args()

    state = read_state()

    if args.json:
        print(json.dumps(state, indent=2))
        return

    print(f"{'GPU':<5} {'Status':<8} {'Command':<40} {'Since':<22} {'Lease Expires':<22} {'Session'}")
    print('-' * 120)
    for gpu_id in sorted(state['gpus'].keys(), key=int):
        entry = state['gpus'][gpu_id]
        if entry is None:
            print(f"{gpu_id:<5} {'FREE':<8} {'-':<40} {'-':<22} {'-':<22} -")
        else:
            snippet = entry.get('command_snippet', '?')[:38]
            since = entry.get('reserved_at', '?')[:19]
            expires = entry.get('lease_expires', '?')[:19]
            sid = entry.get('session_id', '?')[:10]
            print(f"{gpu_id:<5} {'HELD':<8} {snippet:<40} {since:<22} {expires:<22} {sid}")

if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Create `gpu_force_clear.py`**

```python
#!/usr/bin/env python3
"""Force-clear AMMO GPU reservations (orchestrator crash recovery)."""
from __future__ import annotations
import argparse, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from gpu_reservation import force_clear

def main():
    parser = argparse.ArgumentParser(description='Force-clear GPU reservations')
    parser.add_argument('--gpu-ids', type=str, help='Comma-separated GPU IDs (e.g., 0,1)')
    parser.add_argument('--all', action='store_true', help='Clear all GPUs')
    parser.add_argument('--session-id', type=str, help='Session ID to match')
    parser.add_argument('--force-no-session', action='store_true',
                        help='Emergency: clear ALL regardless of session')
    args = parser.parse_args()

    if not args.all and not args.gpu_ids:
        parser.error('Must specify --gpu-ids or --all')
    if not args.session_id and not args.force_no_session:
        parser.error('Must specify --session-id or --force-no-session')

    gpu_ids = None
    if args.gpu_ids:
        gpu_ids = [int(x) for x in args.gpu_ids.split(',')]

    if args.force_no_session:
        print('WARNING: Clearing ALL GPU reservations regardless of session.', file=sys.stderr)

    force_clear(gpu_ids=gpu_ids, session_id=args.session_id,
                force_no_session=args.force_no_session)
    print('GPU reservations cleared.')

if __name__ == '__main__':
    main()
```

- [ ] **Step 3: Test both scripts**

```bash
# Initialize state (if not already)
python .claude/skills/ammo/scripts/gpu_status.py
python .claude/skills/ammo/scripts/gpu_status.py --json
python .claude/skills/ammo/scripts/gpu_force_clear.py --all --force-no-session
```

- [ ] **Step 4: Commit**

```bash
git add .claude/skills/ammo/scripts/gpu_status.py .claude/skills/ammo/scripts/gpu_force_clear.py
git commit -m "feat(ammo): add gpu_status.py and gpu_force_clear.py diagnostic scripts"
```

---

### Task 6: Sweep Script Retrofit — Remove Old Locking

**Files:**
- Modify: `.claude/skills/ammo/scripts/run_vllm_bench_latency_sweep.py`

- [ ] **Step 1: Locate the functions to remove by searching (line numbers may have shifted)**

```bash
grep -n '_acquire_gpu_lock\|_check_gpu_idle\|lock_handle' .claude/skills/ammo/scripts/run_vllm_bench_latency_sweep.py
```

This gives you the actual current line numbers for: `_acquire_gpu_lock()` function definition, `_check_gpu_idle()` function definition, the `lock_handle = _acquire_gpu_lock(...)` call, and the `lock_handle.close()` cleanup.

- [ ] **Step 2: Remove `_acquire_gpu_lock()` function (lines 383-432)**

Delete the entire `_acquire_gpu_lock` function.

- [ ] **Step 3: Remove `_check_gpu_idle()` function (lines 435-461)**

Delete the entire `_check_gpu_idle` function.

- [ ] **Step 4: Remove lock acquisition at line ~1277**

Replace:
```python
lock_handle = _acquire_gpu_lock(artifact_dir=artifact_dir, is_child=is_child)
```
With:
```python
# GPU reservation is now managed by PreToolUse/PostToolUse hooks.
# The hooks auto-reserve when CUDA_VISIBLE_DEVICES=X is in the command
# and auto-release when the command completes. No in-script locking needed.
```

- [ ] **Step 5: Remove lock release at end of main (lines ~1671-1675)**

Delete the `lock_handle.close()` block and any associated `if lock_handle:` checks.

- [ ] **Step 6: Remove `fcntl` import if no longer used elsewhere**

Check if `fcntl` is used anywhere else in the file. If only used for GPU locking, remove the import.

- [ ] **Step 7: Add transition safety check**

At the top of `main()`, after argument parsing, add:

```python
# Transition safety: warn if old locking system files exist
old_lock_dir = Path('/tmp/ammo_gpu_locks')
if old_lock_dir.exists() and any(old_lock_dir.glob('*.lock')):
    print(
        'WARNING: Old GPU lock files found at /tmp/ammo_gpu_locks/. '
        'GPU reservation is now managed by hooks. Old locks can be safely deleted.',
        file=sys.stderr,
    )
```

- [ ] **Step 8: Verify the sweep script still parses correctly**

```bash
python -c "import ast; ast.parse(open('.claude/skills/ammo/scripts/run_vllm_bench_latency_sweep.py').read()); print('Syntax OK')"
```

- [ ] **Step 9: Commit**

```bash
git add .claude/skills/ammo/scripts/run_vllm_bench_latency_sweep.py
git commit -m "refactor(ammo): remove old GPU locking from sweep script (hooks manage it now)"
```

---

### Task 7: Skill Documentation Updates

**Files:**
- Modify: `.claude/skills/ammo/SKILL.md`
- Modify: `.claude/skills/ammo/orchestration/parallel-tracks.md`
- Modify: `.claude/agents/ammo-impl-champion.md`
- Modify: `.claude/agents/ammo-impl-validator.md`

- [ ] **Step 1: Update SKILL.md — add non-negotiable, update hooks table, session_id MUST**

Add after Non-Negotiable 4 (GPU sequencing):

```markdown
5. **GPU isolation**: GPU commands MUST include a `CUDA_VISIBLE_DEVICES=X` prefix (from spawn prompt assignment) for GPU work, or `CUDA_VISIBLE_DEVICES=""` to explicitly signal no GPU is needed. The PreToolUse hook auto-reserves GPUs when CVD contains GPU IDs and blocks on contention. PostToolUse auto-releases. No manual reservation or release needed. *(Enforced by `ammo-pretool-guard.sh` PreToolUse + `ammo-gpu-release.sh` PostToolUse — one-shot block on first missing CVD, then trusts agent judgment)*
```

Update the Hook Enforcement table to add the PostToolUse row and update the PreToolUse description.

Update Helper Scripts table to add `gpu_status.py` and `gpu_force_clear.py`.

Change `"The lead SHOULD record the session ID"` → `"The lead MUST record the session ID"`.

Delete SKILL.md line ~141 containing `"On multi-GPU systems (N >= 3), the last GPU is soft-reserved for debate micro-experiments."`.

Add Resume Protocol step 3b per the spec.

- [ ] **Step 2: Update parallel-tracks.md — TP-aware partitioning**

Locate the sections to modify by searching:

```bash
grep -n 'CUDA_VISIBLE_DEVICES\|flock\|soft-reserved\|GPU Assignment\|GPU Allocation During Overlap' .claude/skills/ammo/orchestration/parallel-tracks.md
```

Replace the GPU Assignment table (the table starting with `| Track |`) with the TP-aware partitioning section from the spec. Delete the "GPU Allocation During Overlap" section that references debate soft-reservation. Update spawn prompt templates to include `kernel_cvd` + `e2e_cvd`.

- [ ] **Step 3: Update agent .md files**

The agent files are at `.claude/agents/ammo-impl-champion.md` and `.claude/agents/ammo-impl-validator.md` (in the repo root `.claude/agents/`, NOT in `.claude/skills/ammo/agents/`).

In `ammo-impl-champion.md`, find the "GPU Coordination" section (grep for `GPU Coordination`) and replace with the simplified GPU Usage instructions from the spec.

In `ammo-impl-validator.md`, find GPU references (grep for `GPU\|CUDA_VISIBLE`) and replace with the same simplified instructions.

- [ ] **Step 4: Commit**

```bash
git add .claude/skills/ammo/SKILL.md .claude/skills/ammo/orchestration/parallel-tracks.md \
  .claude/agents/ammo-impl-champion.md .claude/agents/ammo-impl-validator.md
git commit -m "docs(ammo): update skill docs for GPU reservation system + TP-aware partitioning"
```

---

### Task 8: Integration Verification

**Files:** None (verification only)

- [ ] **Step 1: Verify PostToolUse hook receives command**

```bash
# Create a test PostToolUse invocation to verify the input schema
echo '{"tool_name":"Bash","tool_input":{"command":"CUDA_VISIBLE_DEVICES=0 python -c \"print(1)\""},"tool_response":{"stdout":"1\n","exitCode":0}}' | \
  CLAUDE_PROJECT_DIR=/home/jinhun/vllm bash .claude/hooks/ammo-gpu-release.sh
echo "Exit: $?"
```

- [ ] **Step 2: End-to-end test — reserve + release cycle**

```bash
# Initialize reservation state
python .claude/skills/ammo/scripts/gpu_status.py

# Simulate PreToolUse (reserve)
echo '{"tool_input":{"command":"CUDA_VISIBLE_DEVICES=0 python benchmark.py"}}' | \
  CLAUDE_PROJECT_DIR=/home/jinhun/vllm CLAUDE_SESSION_ID=e2e_test \
  bash .claude/hooks/ammo-pretool-guard.sh
echo "PreToolUse exit: $?"

# Check state — GPU 0 should be reserved
python .claude/skills/ammo/scripts/gpu_status.py

# Simulate PostToolUse (release)
echo '{"tool_input":{"command":"CUDA_VISIBLE_DEVICES=0 python benchmark.py"}}' | \
  CLAUDE_PROJECT_DIR=/home/jinhun/vllm \
  bash .claude/hooks/ammo-gpu-release.sh
echo "PostToolUse exit: $?"

# Check state — GPU 0 should be free
python .claude/skills/ammo/scripts/gpu_status.py
```

- [ ] **Step 3: Test contention blocking**

```bash
# Reserve GPU 0
python3 -c "
import os, sys
sys.path.insert(0, '.claude/skills/ammo/scripts')
from gpu_reservation import write_reservation
write_reservation([0], 'blocker', 'sess1', '0', 'blocking cmd')
"

# Try to reserve GPU 0 again via hook — should block (exit 2)
echo '{"tool_input":{"command":"CUDA_VISIBLE_DEVICES=0 python other.py"}}' | \
  CLAUDE_PROJECT_DIR=/home/jinhun/vllm CLAUDE_SESSION_ID=e2e_test2 \
  bash .claude/hooks/ammo-pretool-guard.sh
echo "Expected exit 2, got: $?"

# Clean up
python .claude/skills/ammo/scripts/gpu_force_clear.py --all --force-no-session
```

- [ ] **Step 4: Test one-shot block (Case C)**

```bash
# Clean flag files
rm -f /tmp/ammo_gpu_res/.warned_*

# First attempt — should block (exit 2)
echo '{"tool_input":{"command":"python benchmark.py"}}' | \
  CLAUDE_PROJECT_DIR=/home/jinhun/vllm CLAUDE_SESSION_ID=oneshot_test \
  bash .claude/hooks/ammo-pretool-guard.sh
echo "First attempt exit (expect 2): $?"

# Second attempt — should pass (exit 0)
echo '{"tool_input":{"command":"python benchmark.py"}}' | \
  CLAUDE_PROJECT_DIR=/home/jinhun/vllm CLAUDE_SESSION_ID=oneshot_test \
  bash .claude/hooks/ammo-pretool-guard.sh
echo "Second attempt exit (expect 0): $?"
```

- [ ] **Step 5: Commit verification results**

```bash
git add -A && git commit -m "test(ammo): verify GPU reservation end-to-end hook cycle"
```
