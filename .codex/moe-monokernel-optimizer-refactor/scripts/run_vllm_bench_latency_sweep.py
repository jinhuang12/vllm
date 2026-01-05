#!/usr/bin/env python3
"""Run production-parity E2E latency sweeps via `vllm bench latency`.

This is *measurement plumbing*.
It does NOT decide what to optimize, and it does NOT guess model semantics.

Why this exists:
- Phase 4.3 requires E2E evidence under identical knobs (CUDA graphs / torch.compile / TP/EP / bucketing).
- Re-running ad-hoc commands is a common source of parity drift and missing evidence.

Inputs
------
Reads a JSON target spec (default: {artifact_dir}/target.json) created by scripts/new_target.py.

Minimal expected schema:

{
  "artifact_dir": "...",
  "target": {
    "model_id": "Qwen/...",
    "dtype": "fp8",
    "tp": 1,
    "ep": 1,
    "max_model_len": 4096
  },
  "workload": {
    "input_len": 64,
    "output_len": 512,
    "batch_sizes": [1,4,8],
    "num_iters": 20
  },
  "bench": {
    "runner": "vllm_bench_latency",
    "vllm_cmd": "vllm",
    "extra_args": [],
    "baseline_env": {},
    "opt_env": {"VLLM_USE_MOE_MONOKERNEL": "1"},
    "baseline_label": "baseline",
    "opt_label": "opt",
    "fastpath_evidence": {
      "opt": {"require_patterns": ["..."], "forbid_patterns": ["..."]},
      "baseline": {"require_patterns": [], "forbid_patterns": []}
    }
  }
}

Guardrails / anti-thrash
-----------------------
- Defaults to --dry-run; must pass --run to execute.
- Fails fast if required fields are missing or still placeholders.
- Records full commands + env vars + stdout/stderr logs per bucket.
- Optional fast-path evidence checks: require/forbid regex patterns per run.

Outputs
-------
Writes into:
  {artifact_dir}/e2e_latency/
    - (dry-run) e2e_latency_dry_run.json / e2e_latency_dry_run.md
    - (run)     e2e_latency_results.json / e2e_latency_results.md
    - logs/{label}_bs{BS}.log
    - json/{label}_bs{BS}.json  (raw vllm bench --output-json)

Usage
-----
  python scripts/run_vllm_bench_latency_sweep.py --artifact-dir <dir> --dry-run
  python scripts/run_vllm_bench_latency_sweep.py --artifact-dir <dir> --run

If your vLLM CLI differs, edit target.json: bench.vllm_cmd, bench.extra_args.

"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


PLACEHOLDER = "<FILL_ME>"


@dataclass
class RunSpec:
    label: str
    env: Dict[str, str]
    require_patterns: List[str]
    forbid_patterns: List[str]


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SystemExit(f"Target spec not found: {path}")
    except json.JSONDecodeError as e:
        raise SystemExit(f"Failed to parse JSON {path}: {e}")


def _is_placeholder(v: Any) -> bool:
    return isinstance(v, str) and (v.strip() == PLACEHOLDER or v.strip().startswith("<") and v.strip().endswith(">"))


def _require(obj: Dict[str, Any], key: str, ctx: str) -> Any:
    if key not in obj:
        raise SystemExit(f"Missing required field: {ctx}.{key}")
    val = obj[key]
    if _is_placeholder(val):
        raise SystemExit(f"Field still placeholder: {ctx}.{key}={val!r}")
    return val


def _require_int(obj: Dict[str, Any], key: str, ctx: str) -> int:
    val = _require(obj, key, ctx)
    if not isinstance(val, int):
        raise SystemExit(f"Expected int for {ctx}.{key}, got {type(val).__name__}")
    return val


def _require_list_int(obj: Dict[str, Any], key: str, ctx: str) -> List[int]:
    val = _require(obj, key, ctx)
    if not isinstance(val, list) or not all(isinstance(x, int) for x in val):
        raise SystemExit(f"Expected list[int] for {ctx}.{key}, got {val!r}")
    return val


def _maybe_list_str(obj: Dict[str, Any], key: str) -> List[str]:
    val = obj.get(key, [])
    if val is None:
        return []
    if not isinstance(val, list) or not all(isinstance(x, str) for x in val):
        raise SystemExit(f"Expected list[str] for bench.{key}, got {val!r}")
    return val


def _bench_exe_tokens(vllm_cmd: Any) -> List[str]:
    if isinstance(vllm_cmd, list):
        if not all(isinstance(x, str) for x in vllm_cmd):
            raise SystemExit(f"bench.vllm_cmd list must be list[str], got {vllm_cmd!r}")
        return vllm_cmd
    if isinstance(vllm_cmd, str):
        return shlex.split(vllm_cmd)
    raise SystemExit(f"bench.vllm_cmd must be str or list[str], got {type(vllm_cmd).__name__}")


_LAT_RE = re.compile(r"^\s*(?P<key>Avg latency|\d+% percentile latency):\s*(?P<val>[0-9.eE+-]+)\s*seconds\s*$", re.MULTILINE)


def _parse_latency_metrics(stdout: str) -> Dict[str, float]:
    """Parse vllm bench latency stdout.

    Expected lines resemble:
      Avg latency: 10.9455 seconds
      50% percentile latency: 10.9064 seconds

    If this format changes, do NOT guess; just return empty dict.
    """
    out: Dict[str, float] = {}
    for m in _LAT_RE.finditer(stdout):
        key = m.group("key").strip()
        val_s = m.group("val")
        try:
            val = float(val_s)
        except ValueError:
            continue
        if key == "Avg latency":
            out["avg_s"] = val
        else:
            # "50% percentile latency" -> p50_s
            pct = key.split("%", 1)[0]
            if pct.isdigit():
                out[f"p{pct}_s"] = val
    return out


def _check_patterns(text: str, require: List[str], forbid: List[str]) -> Dict[str, Any]:
    """Return pattern check results without throwing.

    Patterns are treated as regex. We record which matched.
    """
    req_hits = []
    req_miss = []
    for pat in require:
        if re.search(pat, text):
            req_hits.append(pat)
        else:
            req_miss.append(pat)
    forb_hits = []
    for pat in forbid:
        if re.search(pat, text):
            forb_hits.append(pat)

    ok = (len(req_miss) == 0) and (len(forb_hits) == 0)
    return {
        "ok": ok,
        "require_hits": req_hits,
        "require_miss": req_miss,
        "forbid_hits": forb_hits,
    }


def _run_cmd(cmd: List[str], env: Dict[str, str], *, cwd: Optional[Path], timeout_s: int) -> Dict[str, Any]:
    start = datetime.now(timezone.utc)
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        end = datetime.now(timezone.utc)
        return {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "start_time": start.isoformat(),
            "end_time": end.isoformat(),
            "duration_s": (end - start).total_seconds(),
        }
    except FileNotFoundError as e:
        end = datetime.now(timezone.utc)
        return {
            "ok": False,
            "error": f"FileNotFoundError: {e}",
            "stdout": "",
            "stderr": "",
            "start_time": start.isoformat(),
            "end_time": end.isoformat(),
            "duration_s": (end - start).total_seconds(),
        }
    except subprocess.TimeoutExpired as e:
        end = datetime.now(timezone.utc)
        return {
            "ok": False,
            "error": f"TimeoutExpired: {e}",
            "stdout": e.stdout or "",
            "stderr": e.stderr or "",
            "start_time": start.isoformat(),
            "end_time": end.isoformat(),
            "duration_s": (end - start).total_seconds(),
        }


def _format_cmd_for_md(cmd: List[str], env_overrides: Dict[str, str]) -> str:
    # Keep it copy/pasteable.
    env_prefix = " ".join([f"{k}={shlex.quote(v)}" for k, v in env_overrides.items()])
    cmd_str = " ".join([shlex.quote(x) for x in cmd])
    return (env_prefix + " " + cmd_str).strip()


def _is_safe_preexisting_out_root(out_root: Path) -> bool:
    """Return True if out_root contains only dry-run artifacts and/or empty dirs.

    This enables the common workflow:
      1) run --dry-run to review commands
      2) run --run without needing --overwrite
    """
    allowed_files = {
        "e2e_latency_dry_run.json",
        "e2e_latency_dry_run.md",
    }
    allowed_empty_dirs = {
        "logs",
        "json",
    }
    for p in out_root.iterdir():
        if p.is_dir():
            if p.name not in allowed_empty_dirs:
                return False
            # Only safe if empty; otherwise we might clobber real evidence.
            if any(p.iterdir()):
                return False
            continue
        if p.name in allowed_files:
            continue
        return False
    return True


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _build_vllm_bench_cmd(
    *,
    vllm_exe: List[str],
    model_id: str,
    tp: int,
    max_model_len: int,
    input_len: int,
    output_len: int,
    batch_size: int,
    num_iters: int,
    output_json: Path,
    extra_args: List[str],
) -> List[str]:
    # Command template based on validation/E2E_LATENCY_GUIDE.md
    cmd = (
        vllm_exe
        + [
            "bench",
            "latency",
            "--model",
            model_id,
            "--tensor-parallel-size",
            str(tp),
            "--max-model-len",
            str(max_model_len),
            "--input-len",
            str(input_len),
            "--output-len",
            str(output_len),
            "--batch-size",
            str(batch_size),
            "--num-iters",
            str(num_iters),
            "--output-json",
            str(output_json),
        ]
    )
    if extra_args:
        cmd += extra_args
    return cmd


def _render_md_table(rows: List[Dict[str, Any]], baseline_label: str, opt_label: str) -> str:
    header = f"| Batch Size | {baseline_label} avg (s) | {opt_label} avg (s) | Speedup | Improvement | Fast-path evidence |"
    sep = "|---:|---:|---:|---:|---:|---|"
    lines = [header, sep]
    for r in rows:
        bs = r["batch_size"]
        b = r.get(baseline_label, {})
        o = r.get(opt_label, {})
        b_avg = b.get("avg_s")
        o_avg = o.get("avg_s")
        speedup = r.get("speedup")
        improve = r.get("improvement_pct")
        evidence = o.get("fastpath_evidence", {}).get("status", "unknown")

        def fmt(x: Any) -> str:
            if x is None:
                return ""
            if isinstance(x, (int, float)):
                return f"{x:.6g}"
            return str(x)

        lines.append(
            f"| {bs} | {fmt(b_avg)} | {fmt(o_avg)} | {fmt(speedup)}x | {fmt(improve)}% | {evidence} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifact-dir", type=str, required=True)
    p.add_argument("--target-json", type=str, default=None, help="Path to target.json (default: {artifact_dir}/target.json)")

    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Print commands but do not execute (default).")
    mode.add_argument("--run", action="store_true", help="Actually run the commands.")

    p.add_argument("--timeout-s", type=int, default=1800, help="Timeout per vllm bench invocation")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing e2e_latency outputs")
    p.add_argument("--require-fastpath", action="store_true", help="Fail if opt fast-path evidence patterns do not pass")

    args = p.parse_args()

    dry_run = True
    if args.run:
        dry_run = False

    artifact_dir = Path(args.artifact_dir).expanduser().resolve()
    target_path = Path(args.target_json).expanduser().resolve() if args.target_json else (artifact_dir / "target.json")

    spec = _load_json(target_path)

    artifact_dir_spec = Path(_require(spec, "artifact_dir", "root")).expanduser().resolve()
    if artifact_dir_spec != artifact_dir:
        # This mismatch often indicates copy/paste mistakes; fail fast.
        raise SystemExit(
            f"artifact_dir mismatch: CLI={artifact_dir} vs target.json={artifact_dir_spec}. "
            "Fix target.json or pass --artifact-dir to match."
        )

    target = _require(spec, "target", "root")
    if not isinstance(target, dict):
        raise SystemExit("root.target must be an object")

    model_id = _require(target, "model_id", "target")
    tp = _require_int(target, "tp", "target")
    ep = _require_int(target, "ep", "target")
    max_model_len = _require_int(target, "max_model_len", "target")

    workload = _require(spec, "workload", "root")
    if not isinstance(workload, dict):
        raise SystemExit("root.workload must be an object")

    input_len = _require_int(workload, "input_len", "workload")
    output_len = _require_int(workload, "output_len", "workload")
    batch_sizes = _require_list_int(workload, "batch_sizes", "workload")
    num_iters = _require_int(workload, "num_iters", "workload")

    bench = _require(spec, "bench", "root")
    if not isinstance(bench, dict):
        raise SystemExit("root.bench must be an object")

    runner = _require(bench, "runner", "bench")
    if runner != "vllm_bench_latency":
        raise SystemExit(f"Unsupported bench.runner: {runner!r} (expected 'vllm_bench_latency')")

    vllm_cmd = _require(bench, "vllm_cmd", "bench")
    vllm_exe = _bench_exe_tokens(vllm_cmd)

    extra_args = _maybe_list_str(bench, "extra_args")

    baseline_env = bench.get("baseline_env", {})
    opt_env = bench.get("opt_env", {})
    if not isinstance(baseline_env, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in baseline_env.items()):
        raise SystemExit("bench.baseline_env must be a dict[str,str]")
    if not isinstance(opt_env, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in opt_env.items()):
        raise SystemExit("bench.opt_env must be a dict[str,str]")

    baseline_label = bench.get("baseline_label", "baseline")
    opt_label = bench.get("opt_label", "opt")

    fpe = bench.get("fastpath_evidence", {})
    if not isinstance(fpe, dict):
        fpe = {}

    def _read_evidence(label: str) -> Tuple[List[str], List[str]]:
        cfg = fpe.get(label, {})
        if not isinstance(cfg, dict):
            return ([], [])
        req = cfg.get("require_patterns", [])
        forb = cfg.get("forbid_patterns", [])
        if not isinstance(req, list) or not all(isinstance(x, str) for x in req):
            raise SystemExit(f"bench.fastpath_evidence.{label}.require_patterns must be list[str]")
        if not isinstance(forb, list) or not all(isinstance(x, str) for x in forb):
            raise SystemExit(f"bench.fastpath_evidence.{label}.forbid_patterns must be list[str]")
        return (req, forb)

    baseline_req, baseline_forb = _read_evidence("baseline")
    opt_req, opt_forb = _read_evidence("opt")

    out_root = artifact_dir / "e2e_latency"
    logs_dir = out_root / "logs"
    json_dir = out_root / "json"

    if dry_run:
        out_json_path = out_root / "e2e_latency_dry_run.json"
        out_md_path = out_root / "e2e_latency_dry_run.md"
    else:
        out_json_path = out_root / "e2e_latency_results.json"
        out_md_path = out_root / "e2e_latency_results.md"

    if not dry_run and out_root.exists() and any(out_root.iterdir()) and not args.overwrite:
        # Don't accidentally clobber evidence. Exception: if the directory only contains
        # dry-run artifacts (and/or empty dirs), allow the real run.
        if not _is_safe_preexisting_out_root(out_root):
            raise SystemExit(
                f"Output dir {out_root} already exists and is non-empty. "
                "Pass --overwrite to replace, or move it aside."
            )

    # Always create the root dir so we can write JSON/MD plans. In dry-run, we only
    # create empty subdirs so copy/pasted commands are runnable.
    out_root.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)

    # Prepare envs: start from current env, then apply overrides.
    base_env = dict(os.environ)

    baseline_run = RunSpec(
        label=baseline_label,
        env={**base_env, **baseline_env},
        require_patterns=baseline_req,
        forbid_patterns=baseline_forb,
    )
    opt_run = RunSpec(
        label=opt_label,
        env={**base_env, **baseline_env, **opt_env},
        require_patterns=opt_req,
        forbid_patterns=opt_forb,
    )

    print("=== Target ===")
    print(f"artifact_dir: {artifact_dir}")
    print(f"model_id: {model_id}")
    print(f"tp: {tp}, ep: {ep}, max_model_len: {max_model_len}")
    print(f"workload: input_len={input_len}, output_len={output_len}, num_iters={num_iters}")
    print(f"batch_sizes: {batch_sizes}")
    print(f"baseline_label: {baseline_label}, opt_label: {opt_label}")
    print(f"dry_run: {dry_run}")

    if ep != 1:
        print(
            "WARNING: target.ep != 1, but this script does not pass an explicit EP flag to "
            "`vllm bench latency`. Ensure your EP configuration is applied via bench.extra_args "
            "and/or environment, and that baseline/opt runs are truly production-parity.",
            file=sys.stderr,
        )

    all_runs: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dry_run": dry_run,
        "target_json": str(target_path),
        "model_id": model_id,
        "tp": tp,
        "ep": ep,
        "max_model_len": max_model_len,
        "workload": {
            "input_len": input_len,
            "output_len": output_len,
            "num_iters": num_iters,
            "batch_sizes": batch_sizes,
        },
        "bench": {
            "vllm_exe": vllm_exe,
            "extra_args": extra_args,
            "baseline_env": baseline_env,
            "opt_env": opt_env,
            "baseline_label": baseline_label,
            "opt_label": opt_label,
            "fastpath_evidence": fpe,
        },
        "results": [],
    }

    for bs in batch_sizes:
        # Paths
        baseline_json = json_dir / f"{baseline_label}_bs{bs}.json"
        opt_json = json_dir / f"{opt_label}_bs{bs}.json"
        baseline_log = logs_dir / f"{baseline_label}_bs{bs}.log"
        opt_log = logs_dir / f"{opt_label}_bs{bs}.log"

        baseline_cmd = _build_vllm_bench_cmd(
            vllm_exe=vllm_exe,
            model_id=model_id,
            tp=tp,
            max_model_len=max_model_len,
            input_len=input_len,
            output_len=output_len,
            batch_size=bs,
            num_iters=num_iters,
            output_json=baseline_json,
            extra_args=extra_args,
        )
        opt_cmd = _build_vllm_bench_cmd(
            vllm_exe=vllm_exe,
            model_id=model_id,
            tp=tp,
            max_model_len=max_model_len,
            input_len=input_len,
            output_len=output_len,
            batch_size=bs,
            num_iters=num_iters,
            output_json=opt_json,
            extra_args=extra_args,
        )

        print(f"\n=== batch_size={bs} ===")
        print(f"Baseline cmd: {_format_cmd_for_md(baseline_cmd, baseline_env)}")
        print(f"Opt cmd: {_format_cmd_for_md(opt_cmd, {**baseline_env, **opt_env})}")

        if dry_run:
            # Record the planned commands deterministically so users can review/diff.
            row: Dict[str, Any] = {
                "batch_size": bs,
                baseline_label: {
                    "cmd": baseline_cmd,
                    "env_overrides": baseline_env,
                    "log": str(baseline_log.relative_to(out_root)),
                    "output_json": str(baseline_json.relative_to(out_root)),
                    "fastpath_evidence": {"status": "unknown"},
                },
                opt_label: {
                    "cmd": opt_cmd,
                    "env_overrides": {**baseline_env, **opt_env},
                    "log": str(opt_log.relative_to(out_root)),
                    "output_json": str(opt_json.relative_to(out_root)),
                    "fastpath_evidence": {"status": "unknown"},
                },
            }
            all_runs["results"].append(row)
            continue

        # Run baseline.
        baseline_res = _run_cmd(baseline_cmd, baseline_run.env, cwd=None, timeout_s=args.timeout_s)
        baseline_text = (baseline_res.get("stdout", "") or "") + "\n" + (baseline_res.get("stderr", "") or "")
        baseline_metrics = _parse_latency_metrics(baseline_text)
        baseline_evidence = _check_patterns(baseline_text, baseline_run.require_patterns, baseline_run.forbid_patterns)

        _write_text(baseline_log, baseline_text)

        # Run opt.
        opt_res = _run_cmd(opt_cmd, opt_run.env, cwd=None, timeout_s=args.timeout_s)
        opt_text = (opt_res.get("stdout", "") or "") + "\n" + (opt_res.get("stderr", "") or "")
        opt_metrics = _parse_latency_metrics(opt_text)
        opt_evidence = _check_patterns(opt_text, opt_run.require_patterns, opt_run.forbid_patterns)
        _write_text(opt_log, opt_text)

        # Summaries
        def _status_from_evidence(ev: Dict[str, Any], patterns_configured: bool) -> str:
            if not patterns_configured:
                return "unknown"
            return "pass" if ev.get("ok") else "fail"

        baseline_patterns_configured = bool(baseline_run.require_patterns or baseline_run.forbid_patterns)
        opt_patterns_configured = bool(opt_run.require_patterns or opt_run.forbid_patterns)

        baseline_entry: Dict[str, Any] = {
            "ok": baseline_res.get("ok"),
            "returncode": baseline_res.get("returncode"),
            "cmd": baseline_cmd,
            "env_overrides": baseline_env,
            "metrics": baseline_metrics,
            "avg_s": baseline_metrics.get("avg_s"),
            "fastpath_evidence": {
                **baseline_evidence,
                "status": _status_from_evidence(baseline_evidence, baseline_patterns_configured),
            },
            "log": str(baseline_log.relative_to(out_root)),
            "output_json": str(baseline_json.relative_to(out_root)),
            "timing": {
                "start_time": baseline_res.get("start_time"),
                "end_time": baseline_res.get("end_time"),
                "duration_s": baseline_res.get("duration_s"),
            },
        }

        opt_entry: Dict[str, Any] = {
            "ok": opt_res.get("ok"),
            "returncode": opt_res.get("returncode"),
            "cmd": opt_cmd,
            "env_overrides": {**baseline_env, **opt_env},
            "metrics": opt_metrics,
            "avg_s": opt_metrics.get("avg_s"),
            "fastpath_evidence": {
                **opt_evidence,
                "status": _status_from_evidence(opt_evidence, opt_patterns_configured),
            },
            "log": str(opt_log.relative_to(out_root)),
            "output_json": str(opt_json.relative_to(out_root)),
            "timing": {
                "start_time": opt_res.get("start_time"),
                "end_time": opt_res.get("end_time"),
                "duration_s": opt_res.get("duration_s"),
            },
        }

        row: Dict[str, Any] = {
            "batch_size": bs,
            baseline_label: baseline_entry,
            opt_label: opt_entry,
        }

        # Compute deltas.
        b_avg = baseline_entry.get("avg_s")
        o_avg = opt_entry.get("avg_s")
        if isinstance(b_avg, (int, float)) and isinstance(o_avg, (int, float)) and o_avg > 0:
            speedup = b_avg / o_avg
            improvement_pct = (b_avg - o_avg) / b_avg * 100.0 if b_avg != 0 else None
            row["speedup"] = speedup
            row["improvement_pct"] = improvement_pct

        all_runs["results"].append(row)

        if args.require_fastpath and opt_patterns_configured and not opt_evidence.get("ok"):
            raise SystemExit(
                f"Fast-path evidence FAILED for opt at BS={bs}. "
                f"Missing={opt_evidence.get('require_miss')}, forbidden_hits={opt_evidence.get('forbid_hits')}. "
                f"See {opt_log}"
            )

    # Always write outputs on dry-run too (so agent can diff / inspect).
    _write_json(out_json_path, all_runs)

    md_lines: List[str] = []
    md_lines.append("# E2E Latency Sweep (vllm bench latency)")
    md_lines.append("")
    md_lines.append(f"Generated: {all_runs['generated_at']} (UTC)")
    md_lines.append("")
    md_lines.append("## Workload")
    md_lines.append("")
    md_lines.append(f"- model_id: {model_id}")
    md_lines.append(f"- input_len: {input_len}, output_len: {output_len}")
    md_lines.append(f"- tp: {tp}, max_model_len: {max_model_len}")
    md_lines.append(f"- num_iters: {num_iters}")
    md_lines.append("")
    md_lines.append("## Results")
    md_lines.append("")
    md_lines.append(_render_md_table(all_runs["results"], baseline_label, opt_label))

    if dry_run:
        md_lines.append("\n> Dry-run only. Re-run with --run to execute.\n")

    _write_text(out_md_path, "\n".join(md_lines))

    print(f"\nWrote: {out_json_path}")
    print(f"Wrote: {out_md_path}")


if __name__ == "__main__":
    main()
