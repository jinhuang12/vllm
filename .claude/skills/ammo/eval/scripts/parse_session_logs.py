#!/usr/bin/env python3
"""Parse Claude Code session JSONL files to extract ground-truth timing and token cost data.

Replaces manual tracking in state.json with data derived from the actual session logs.

JSONL format (one JSON object per line):
  type: "user", "assistant", "progress", "queue-operation", "system", "file-history-snapshot"
  timestamp: ISO 8601 (e.g., "2026-03-11T18:07:00.123Z")
  sessionId: parent session UUID
  agentId: present on subagent messages
  message.content: array of text/tool_use blocks

Usage:
  python parse_session_logs.py \\
    --session-id <SESSION_UUID> \\
    [--session-dir ~/.claude/projects/-home-jinhun-vllm] \\
    [--artifact-dir <path>] \\
    --output /tmp/ammo_eval_session_data.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_iso(ts: Optional[str]) -> Optional[datetime]:
    """Parse ISO timestamp string to datetime, returning None on failure."""
    if not ts or not isinstance(ts, str):
        return None
    try:
        # Handle Z suffix
        ts_clean = ts.replace("Z", "+00:00")
        return datetime.fromisoformat(ts_clean)
    except (ValueError, TypeError):
        return None


def _iso(dt: Optional[datetime]) -> Optional[str]:
    """Format datetime as ISO 8601 string."""
    if dt is None:
        return None
    return dt.isoformat()


def _infer_role(name: Optional[str], subagent_type: Optional[str], description: Optional[str]) -> str:
    """Infer agent role from name, subagent_type, and description."""
    # Prefer explicit subagent_type
    if subagent_type:
        st = subagent_type.lower()
        if "researcher" in st:
            return "researcher"
        if "champion" in st:
            return "champion"
        if "delegate" in st:
            return "delegate"
        if "implementer" in st:
            return "implementer"

    # Fall back to name pattern
    if name:
        n = name.lower()
        if re.match(r"champion-?\d+", n):
            return "champion"
        if re.match(r"delegate-?\d+\w*", n):
            return "delegate"
        if "implementer" in n:
            return "implementer"
        if "researcher" in n:
            return "researcher"

    # Fall back to description keywords
    if description:
        d = description.lower()
        if "researcher" in d or "baseline" in d or "bottleneck" in d:
            return "researcher"
        if "implementer" in d or "implementation" in d:
            return "implementer"
        if "champion" in d:
            return "champion"
        if "delegate" in d:
            return "delegate"

    return "unknown"


def _extract_task_notification(content_str: str) -> Optional[Dict[str, Any]]:
    """Extract task-notification data from a content string."""
    m = re.search(r"<task-notification>(.*?)</task-notification>", content_str, re.DOTALL)
    if not m:
        return None
    xml = m.group(0)

    def _field(tag: str) -> Optional[str]:
        fm = re.search(rf"<{tag}>(.*?)</{tag}>", xml, re.DOTALL)
        return fm.group(1).strip() if fm else None

    task_id = _field("task-id")
    tool_use_id = _field("tool-use-id")
    status = _field("status")
    summary = _field("summary")

    total_tokens = None
    tool_uses_count = None
    duration_ms = None

    usage_m = re.search(r"<usage>(.*?)</usage>", xml, re.DOTALL)
    if usage_m:
        usage_text = usage_m.group(0)
        tt = re.search(r"<total_tokens>(\d+)</total_tokens>", usage_text)
        tu = re.search(r"<tool_uses>(\d+)</tool_uses>", usage_text)
        dm = re.search(r"<duration_ms>(\d+)</duration_ms>", usage_text)
        if tt:
            total_tokens = int(tt.group(1))
        if tu:
            tool_uses_count = int(tu.group(1))
        if dm:
            duration_ms = int(dm.group(1))

    return {
        "task_id": task_id,
        "tool_use_id": tool_use_id,
        "status": status,
        "summary": summary,
        "total_tokens": total_tokens,
        "tool_uses": tool_uses_count,
        "duration_ms": duration_ms,
    }


def _get_content_text(content: Any) -> str:
    """Extract text from message content (string or list of blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif isinstance(block.get("text"), str):
                    parts.append(block["text"])
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return ""


# ---------------------------------------------------------------------------
# Session JSONL Streaming Parser
# ---------------------------------------------------------------------------

class SessionParser:
    """Streams a session JSONL file and extracts relevant events."""

    def __init__(self, session_path: Path):
        self.session_path = session_path

        # Registry: tool_use_id -> agent spawn info
        self.agent_spawns: Dict[str, Dict[str, Any]] = {}
        # Registry: task_id -> task-notification data
        self.task_notifications: Dict[str, Dict[str, Any]] = {}

        # Team lifecycle events: list of {action, name, timestamp}
        self.team_events: List[Dict[str, Any]] = []

        # All events in order for stage inference
        self.events: List[Dict[str, Any]] = []

        # Session boundaries
        self.session_start: Optional[str] = None
        self.session_end: Optional[str] = None

    def parse(self) -> None:
        """Stream and parse the JSONL file."""
        with open(self.session_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                self._process_line(obj)

    def _process_line(self, obj: Dict[str, Any]) -> None:
        ts = obj.get("timestamp")
        if ts:
            if self.session_start is None:
                self.session_start = ts
            self.session_end = ts

        msg_type = obj.get("type")

        if msg_type == "assistant":
            self._process_assistant(obj)
        elif msg_type == "user":
            self._process_user(obj)

    def _process_assistant(self, obj: Dict[str, Any]) -> None:
        ts = obj.get("timestamp")
        msg = obj.get("message", {})
        content = msg.get("content", [])
        if not isinstance(content, list):
            return

        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_use":
                continue
            tool_name = block.get("name", "")
            tool_use_id = block.get("id")
            inp = block.get("input", {}) or {}

            if tool_name == "Agent":
                agent_name = inp.get("name")
                subagent_type = inp.get("subagent_type")
                description = inp.get("description")
                model = inp.get("model")
                team_name = inp.get("team_name")
                role = _infer_role(agent_name, subagent_type, description)

                spawn_info = {
                    "tool_use_id": tool_use_id,
                    "spawn_timestamp": ts,
                    "name": agent_name or description,
                    "display_name": agent_name,
                    "subagent_type": subagent_type,
                    "description": description,
                    "model": model,
                    "team_name": team_name,
                    "role": role,
                }
                if tool_use_id:
                    self.agent_spawns[tool_use_id] = spawn_info

                self.events.append({
                    "event": "agent_spawn",
                    "timestamp": ts,
                    "tool_use_id": tool_use_id,
                    "role": role,
                    "name": agent_name or description,
                    "team_name": team_name,
                    "subagent_type": subagent_type,
                })

            elif tool_name == "TeamCreate":
                team_name = inp.get("team_name")
                self.team_events.append({
                    "action": "create",
                    "name": team_name,
                    "timestamp": ts,
                    "description": inp.get("description"),
                })
                self.events.append({
                    "event": "team_create",
                    "timestamp": ts,
                    "team_name": team_name,
                })

            elif tool_name == "TeamDelete":
                # TeamDelete input is often empty; infer team from most recent TeamCreate without a delete
                team_name = inp.get("team_name")
                if not team_name:
                    # Find the most recently created team not yet deleted
                    deleted_names = {e.get("name") for e in self.team_events if e["action"] == "delete"}
                    active = [e for e in self.team_events
                              if e["action"] == "create" and e.get("name") not in deleted_names]
                    if active:
                        team_name = active[-1]["name"]
                if team_name is not None:
                    self.team_events.append({
                        "action": "delete",
                        "name": team_name,
                        "timestamp": ts,
                    })
                    self.events.append({
                        "event": "team_delete",
                        "timestamp": ts,
                        "team_name": team_name,
                    })

    def _process_user(self, obj: Dict[str, Any]) -> None:
        ts = obj.get("timestamp")
        msg = obj.get("message", {})
        content = msg.get("content", "")
        content_text = _get_content_text(content)

        if "<task-notification>" not in content_text:
            return

        notif = _extract_task_notification(content_text)
        if not notif:
            return

        notif["completion_timestamp"] = ts
        task_id = notif.get("task_id")
        tool_use_id = notif.get("tool_use_id")

        if task_id:
            self.task_notifications[task_id] = notif
        # Also index by tool_use_id for quick lookup
        if tool_use_id:
            self.task_notifications[f"by_tool_use_id:{tool_use_id}"] = notif

        self.events.append({
            "event": "task_complete",
            "timestamp": ts,
            "task_id": task_id,
            "tool_use_id": tool_use_id,
            "status": notif.get("status"),
            "total_tokens": notif.get("total_tokens"),
            "duration_ms": notif.get("duration_ms"),
        })


# ---------------------------------------------------------------------------
# Stage Timestamp Inference
# ---------------------------------------------------------------------------

def _infer_stage_timestamps(
    events: List[Dict[str, Any]],
    agent_spawns: Dict[str, Dict[str, Any]],
    task_notifications: Dict[str, Dict[str, Any]],
    team_events: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Infer per-round stage timestamps from observable events."""

    # Identify campaign rounds by TeamCreate/TeamDelete pairs
    # Also identify researcher spawns and implementer spawns

    # Build per-round data: team lifecycle defines rounds
    # Round 1: before first TeamCreate (researcher, then TeamCreate, then TeamDelete, then implementers)
    # Round N: second TeamCreate...

    team_creates = [(e["timestamp"], e.get("team_name")) for e in team_events if e["action"] == "create"]
    team_deletes = [(e["timestamp"], e.get("name")) for e in team_events if e["action"] == "delete"]

    # Get researcher spawns and completions
    researcher_spawns = [e for e in events if e.get("event") == "agent_spawn" and e.get("role") == "researcher"]
    researcher_completions = [e for e in events if e.get("event") == "task_complete" and
                               e.get("tool_use_id") in {s.get("tool_use_id") for s in researcher_spawns}]

    # Get implementer spawns
    implementer_spawns = [e for e in events if e.get("event") == "agent_spawn" and e.get("role") == "implementer"]
    implementer_completions = [e for e in events if e.get("event") == "task_complete" and
                                e.get("tool_use_id") in {s.get("tool_use_id") for s in implementer_spawns}]

    # Build rounds
    # Number of rounds = number of TeamCreate calls (each TeamCreate is a debate stage per round)
    # But there can also be implementer rounds between debate rounds

    result = {}

    # Determine how many campaign rounds there are by counting TeamCreate events
    num_debate_rounds = len(team_creates)

    # Round 1 always exists
    # Stage 1: researcher spawn
    # Stage 2: researcher completion
    # Stage 3: TeamCreate for debate
    # Stage 3 end: first TeamDelete
    # Stage 4-5: implementer spawns
    # Stage 4-5 end: last implementer completion
    # Stages 6-7: inferred from remaining

    # For multi-round: between TeamDelete and next TeamCreate = campaign eval (stages 6-7)
    # Then next TeamCreate starts next debate round

    session_start_ts = events[0]["timestamp"] if events else None

    # --- Round 1 ---
    r1: Dict[str, Any] = {}

    # Stage 1 start: first researcher spawn
    r1_researcher_spawns = [s for s in researcher_spawns][:2]  # could be 2 researchers in round 1 (retry)
    if r1_researcher_spawns:
        r1["stage_1_baseline_start"] = r1_researcher_spawns[0]["timestamp"]
        # Stage 2 completion: last researcher completion before TeamCreate
        # Find the completion of the last researcher spawned before TeamCreate
        tc1_ts = team_creates[0][0] if team_creates else None
        pre_tc1_completions = [c for c in researcher_completions
                                if tc1_ts is None or c["timestamp"] <= tc1_ts]
        if pre_tc1_completions:
            r1["stage_2_bottleneck_end"] = pre_tc1_completions[-1]["timestamp"]

    # Stage 3: TeamCreate for debate
    if team_creates:
        r1["stage_3_debate_start"] = team_creates[0][0]
        # Stage 3 end: first TeamDelete after stage_3_debate_start
        post_tc1_deletes = [d for d in team_deletes if d[0] >= team_creates[0][0]]
        if post_tc1_deletes:
            r1["stage_3_debate_end"] = post_tc1_deletes[0][0]

    # Stage 4-5: implementer spawns
    # Find implementers spawned after TeamDelete and before next TeamCreate (if any)
    r1_debate_end = r1.get("stage_3_debate_end")
    tc2_ts = team_creates[1][0] if len(team_creates) > 1 else None

    r1_implementers = [s for s in implementer_spawns
                       if (r1_debate_end is None or s["timestamp"] >= r1_debate_end) and
                          (tc2_ts is None or s["timestamp"] < tc2_ts)]
    if r1_implementers:
        r1["stage_4_5_impl_start"] = r1_implementers[0]["timestamp"]
        # Stage 4-5 end: last completion of those implementers
        r1_impl_ids = {s.get("tool_use_id") for s in r1_implementers}
        r1_impl_completions = [c for c in implementer_completions if c.get("tool_use_id") in r1_impl_ids]
        if r1_impl_completions:
            r1["stage_4_5_impl_end"] = max(c["timestamp"] for c in r1_impl_completions)

    # Stages 6-7: between impl end and next round's researcher spawn (or session end)
    # Note: async pipeline means Round 2 debate may START during Round 1 implementation,
    # so tc2_ts can be BEFORE r1_impl_end. Use the later of tc2_ts and r1_impl_end.
    r1_impl_end = r1.get("stage_4_5_impl_end")
    if r1_impl_end:
        r1["stage_6_7_eval_start"] = r1_impl_end
        # Find the next researcher spawn after impl end (signals next round's re-profiling)
        next_researchers = [e for e in events
                           if e.get("event") == "agent_spawn" and e.get("role") == "researcher"
                           and e["timestamp"] > r1_impl_end]
        if next_researchers:
            r1["stage_6_7_eval_end"] = next_researchers[0]["timestamp"]
        elif events:
            r1["stage_6_7_eval_end"] = events[-1]["timestamp"]

    result["round_1"] = r1

    # --- Round 2+ (if multi-round) ---
    for ri in range(1, len(team_creates)):
        round_num = ri + 1
        rN: Dict[str, Any] = {}
        tc_ts = team_creates[ri][0]
        tc_next_ts = team_creates[ri + 1][0] if ri + 1 < len(team_creates) else None

        # Stage 3: debate for this round
        rN["stage_3_debate_start"] = tc_ts
        # Find corresponding TeamDelete after this TeamCreate
        post_tc_deletes = [d for d in team_deletes if d[0] >= tc_ts and
                           (tc_next_ts is None or d[0] < tc_next_ts)]
        if post_tc_deletes:
            rN["stage_3_debate_end"] = post_tc_deletes[0][0]

        # Stage 4-5: implementers after debate ends
        rN_debate_end = rN.get("stage_3_debate_end")
        rN_implementers = [s for s in implementer_spawns
                           if (rN_debate_end is not None and s["timestamp"] >= rN_debate_end) and
                              (tc_next_ts is None or s["timestamp"] < tc_next_ts)]
        if rN_implementers:
            rN["stage_4_5_impl_start"] = rN_implementers[0]["timestamp"]
            rN_impl_ids = {s.get("tool_use_id") for s in rN_implementers}
            rN_impl_completions = [c for c in implementer_completions if c.get("tool_use_id") in rN_impl_ids]
            if rN_impl_completions:
                rN["stage_4_5_impl_end"] = max(c["timestamp"] for c in rN_impl_completions)

        result[f"round_{round_num}"] = rN

    return result


# ---------------------------------------------------------------------------
# Subagent Discovery
# ---------------------------------------------------------------------------

def _discover_subagents(session_dir: Path, session_id: str) -> List[Dict[str, Any]]:
    """Discover and summarize subagent JSONL files."""
    subagents_dir = session_dir / session_id / "subagents"
    if not subagents_dir.exists():
        return []

    result = []
    for jsonl_path in sorted(subagents_dir.glob("agent-*.jsonl")):
        agent_id = jsonl_path.stem.replace("agent-", "")
        meta_path = subagents_dir / f"agent-{agent_id}.meta.json"
        agent_type = None
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                agent_type = meta.get("agentType")
            except (json.JSONDecodeError, OSError):
                pass

        # Read first and last lines
        first_ts = None
        last_ts = None
        msg_count = 0
        try:
            with open(jsonl_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    ts = obj.get("timestamp")
                    if ts:
                        if first_ts is None:
                            first_ts = ts
                        last_ts = ts
                    msg_count += 1
        except OSError:
            pass

        result.append({
            "agent_id": agent_id,
            "agent_type": agent_type,
            "first_timestamp": first_ts,
            "last_timestamp": last_ts,
            "message_count": msg_count,
        })

    return result


# ---------------------------------------------------------------------------
# Cost Summary
# ---------------------------------------------------------------------------

def _build_agent_costs(
    agent_spawns: Dict[str, Dict[str, Any]],
    task_notifications: Dict[str, Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Build agent_costs array and cost_summary."""
    agent_costs = []

    for tool_use_id, spawn in agent_spawns.items():
        notif_key = f"by_tool_use_id:{tool_use_id}"
        notif = task_notifications.get(notif_key)

        entry: Dict[str, Any] = {
            "name": spawn.get("name"),
            "display_name": spawn.get("display_name"),
            "role": spawn.get("role", "unknown"),
            "subagent_type": spawn.get("subagent_type"),
            "description": spawn.get("description"),
            "team_name": spawn.get("team_name"),
            "spawn_timestamp": spawn.get("spawn_timestamp"),
            "tool_use_id": tool_use_id,
        }

        if notif:
            entry["total_tokens"] = notif.get("total_tokens")
            entry["tool_uses"] = notif.get("tool_uses")
            entry["duration_ms"] = notif.get("duration_ms")
            entry["status"] = notif.get("status")
            entry["completion_timestamp"] = notif.get("completion_timestamp")
        else:
            entry["total_tokens"] = None
            entry["tool_uses"] = None
            entry["duration_ms"] = None
            entry["status"] = "killed_or_no_notification"
            entry["completion_timestamp"] = None

        agent_costs.append(entry)

    # Build cost_summary
    by_role: Dict[str, Any] = {}
    total_tokens = 0
    total_duration_ms = 0
    total_invocations = 0

    for entry in agent_costs:
        role = entry.get("role", "unknown")
        tokens = entry.get("total_tokens") or 0
        duration = entry.get("duration_ms") or 0

        total_tokens += tokens
        total_duration_ms += duration
        total_invocations += 1

        if role not in by_role:
            by_role[role] = {"count": 0, "total_tokens": 0, "total_duration_ms": 0}
        by_role[role]["count"] += 1
        by_role[role]["total_tokens"] += tokens
        by_role[role]["total_duration_ms"] += duration

    cost_summary = {
        "total_agent_invocations": total_invocations,
        "total_tokens": total_tokens,
        "total_duration_ms": total_duration_ms,
        "by_role": by_role,
    }

    return agent_costs, cost_summary


# ---------------------------------------------------------------------------
# Team Lifecycle Summary
# ---------------------------------------------------------------------------

def _build_team_lifecycle(team_events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build a structured team lifecycle from raw events."""
    teams: Dict[str, Dict[str, Any]] = {}
    result = []

    for event in team_events:
        action = event["action"]
        name = event.get("name")
        ts = event["timestamp"]

        if action == "create":
            teams[name or "_unnamed"] = {
                "name": name,
                "create_timestamp": ts,
                "description": event.get("description"),
                "delete_timestamp": None,
            }
        elif action == "delete":
            key = name or "_unnamed"
            if key in teams:
                teams[key]["delete_timestamp"] = ts
                entry = dict(teams[key])
                if entry.get("create_timestamp") and entry.get("delete_timestamp"):
                    c = _parse_iso(entry["create_timestamp"])
                    d = _parse_iso(entry["delete_timestamp"])
                    if c and d:
                        entry["duration_seconds"] = round((d - c).total_seconds(), 1)
                result.append(entry)
            else:
                result.append({
                    "name": name,
                    "create_timestamp": None,
                    "delete_timestamp": ts,
                    "duration_seconds": None,
                })

    # Add any teams without delete
    for key, team in teams.items():
        if team.get("delete_timestamp") is None:
            result.append(team)

    return result


# ---------------------------------------------------------------------------
# Main Parser
# ---------------------------------------------------------------------------

def parse_session(
    session_id: str,
    session_dir: Path,
    artifact_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Parse a session and return structured data."""

    # Find the JSONL file
    session_path = session_dir / f"{session_id}.jsonl"
    if not session_path.exists():
        raise FileNotFoundError(f"Session JSONL not found: {session_path}")

    parser = SessionParser(session_path)
    parser.parse()

    # Build agent costs
    agent_costs, cost_summary = _build_agent_costs(parser.agent_spawns, parser.task_notifications)

    # Build stage timestamps
    stage_timestamps = _infer_stage_timestamps(
        parser.events,
        parser.agent_spawns,
        parser.task_notifications,
        parser.team_events,
    )

    # Build team lifecycle
    team_lifecycle = _build_team_lifecycle(parser.team_events)

    # Discover subagents
    subagents = _discover_subagents(session_dir, session_id)

    return {
        "session_id": session_id,
        "session_start": parser.session_start,
        "session_end": parser.session_end,
        "stage_timestamps": stage_timestamps,
        "agent_costs": agent_costs,
        "team_lifecycle": team_lifecycle,
        "subagents": subagents,
        "cost_summary": cost_summary,
        "parsed_at": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--session-id", required=True, help="Session UUID")
    p.add_argument(
        "--session-dir",
        default="~/.claude/projects/-home-jinhun-vllm",
        help="Directory containing session JSONL files",
    )
    p.add_argument(
        "--artifact-dir",
        default=None,
        help="Optional: artifact directory to cross-reference state.json for role mapping",
    )
    p.add_argument(
        "--output",
        required=True,
        help="Output JSON path",
    )

    args = p.parse_args()
    session_dir = Path(args.session_dir).expanduser().resolve()
    artifact_dir = Path(args.artifact_dir).expanduser().resolve() if args.artifact_dir else None

    if not session_dir.exists():
        print(f"ERROR: Session directory does not exist: {session_dir}", file=sys.stderr)
        return 1

    try:
        data = parse_session(args.session_id, session_dir, artifact_dir)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    out_path = Path(args.output)
    out_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote: {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
