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
    "baseline_extra_args": [],
    "opt_extra_args": [],
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
- Fails fast if required fields are missing or still placeholders.
- Records full commands + env vars + stdout/stderr logs per bucket.
- Optional fast-path evidence checks: require/forbid regex patterns per run.
- Avoids model reload thrash by default: uses an in-process sweep runner that
  loads the model once per label (baseline/opt) and benchmarks all batch sizes
  in that process. This reduces end-to-end sweep time without changing
  per-iteration latency measurement. Use `--execution-mode cli_per_bs` for
  strict per-bucket process isolation.
- Supports vLLM's dotted "json-style" CLI flags in `inproc_sweep` (e.g.
  `-cc.pass_config.enable_sp=false`), because it uses vLLM's
  `FlexibleArgumentParser` for argument parsing.
- Archives an existing output directory automatically (instead of refusing to
  run again) unless `--overwrite` is passed.
- Emits live progress to logs and `status/*.json` so agents can distinguish
  "slow" from "hung".

Outputs
-------
Writes into:
  {artifact_dir}/{out_name}/
    - e2e_latency_results.json / e2e_latency_results.md
    - logs/{label}_bs{BS}.log
    - json/{label}_bs{BS}.json  (raw vllm bench --output-json)
    - status/{label}.json  (heartbeat + current phase)

Usage
-----
  python scripts/run_vllm_bench_latency_sweep.py --artifact-dir <dir>
  python scripts/run_vllm_bench_latency_sweep.py --artifact-dir <dir> --out-name e2e_latency_combined

If your vLLM CLI differs, edit target.json: bench.vllm_cmd, bench.extra_args,
bench.baseline_extra_args, bench.opt_extra_args.

"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import shlex
import selectors
import socket
import subprocess
import sys
import time
import traceback
import shutil
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


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_json_atomic(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SystemExit(f"Expected JSON output not found: {path}")
    except json.JSONDecodeError as e:
        raise SystemExit(f"Failed to parse JSON {path}: {e}")


def _metrics_from_vllm_latency_json(obj: Dict[str, Any]) -> Dict[str, float]:
    """Convert vllm bench latency JSON into this script's metrics dict.

    vLLM JSON schema (as of vllm/benchmarks/latency.py):
      {"avg_latency": float, "latencies": [...], "percentiles": {"10": float, ...}}
    """
    out: Dict[str, float] = {}
    avg = obj.get("avg_latency")
    if isinstance(avg, (int, float)):
        out["avg_s"] = float(avg)
    percentiles = obj.get("percentiles", {})
    if isinstance(percentiles, dict):
        for k, v in percentiles.items():
            if isinstance(k, str) and k.isdigit() and isinstance(v, (int, float)):
                out[f"p{k}_s"] = float(v)
    return out


def _utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")


def _sanitize_filename(s: str) -> str:
    # Keep it conservative for shared filesystems.
    return re.sub(r"[^A-Za-z0-9._=-]+", "_", s).strip("_") or "default"


def _pick_archive_path(path: Path, run_id: str) -> Path:
    """Pick a unique archive name for an existing output directory."""
    base = path.with_name(f"{path.name}_{run_id}")
    if not base.exists():
        return base
    for i in range(1, 1000):
        cand = path.with_name(f"{path.name}_{run_id}:{i}")
        if not cand.exists():
            return cand
    raise SystemExit(f"Failed to pick unique archive name for {path}")


def _prepare_out_root(
    *,
    artifact_dir: Path,
    out_name: str,
    overwrite: bool,
) -> Path:
    out_root = artifact_dir / out_name
    if out_root.exists() and any(out_root.iterdir()):
        if overwrite:
            # Be explicit: overwrite means discard previous evidence.
            shutil.rmtree(out_root, ignore_errors=True)
        else:
            run_id = _utc_run_id()
            archived = _pick_archive_path(out_root, run_id)
            print(f"Archiving existing output dir: {out_root} -> {archived}")
            out_root.replace(archived)
    out_root.mkdir(parents=True, exist_ok=True)
    return out_root


def _acquire_gpu_lock(*, artifact_dir: Path, enabled: bool) -> Optional[Any]:
    """Best-effort inter-process lock keyed by visible GPUs.

    This prevents running multiple sweeps concurrently on the same GPUs, which
    makes measurements meaningless due to contention.
    """
    if not enabled:
        return None

    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    key = "all" if cuda_visible is None else (cuda_visible.strip() or "empty")
    key = _sanitize_filename(key)

    lock_dir = artifact_dir / ".gpu_locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / f"vllm_bench_latency_{key}.lock"

    fh = open(lock_path, "a+", encoding="utf-8")
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        fh.seek(0)
        existing = fh.read().strip()
        raise SystemExit(
            f"GPU lock is already held for CUDA_VISIBLE_DEVICES={cuda_visible!r}.\n"
            f"Lock file: {lock_path}\n"
            f"Current holder (best-effort):\n{existing}\n"
            "If this is stale, delete the lock file."
        )

    # Record holder info for debugging.
    fh.seek(0)
    fh.truncate(0)
    holder = {
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "cuda_visible_devices": cuda_visible,
        "cmdline": sys.argv,
    }
    fh.write(json.dumps(holder, indent=2, sort_keys=True) + "\n")
    fh.flush()
    return fh


def _run_cmd_streaming(
    cmd: List[str],
    *,
    env: Dict[str, str],
    cwd: Optional[Path],
    timeout_s: int,
    log_path: Path,
    heartbeat_s: int,
    status_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run a command and stream its output to a log (and stdout).

    This is intentionally "agent friendly": it keeps logs hot so a supervisor can
    tell whether we're progressing.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = datetime.now(timezone.utc)

    # Create the log immediately so tailing works even if exec/import is slow.
    with open(log_path, "w", encoding="utf-8", buffering=1) as log_f:
        log_f.write(f"=== cmd ===\n{_format_cmd_for_md(cmd, {})}\n")
        log_f.write(f"=== start ===\n{start.isoformat()}\n")
        log_f.flush()

        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd) if cwd else None,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        sel = selectors.DefaultSelector()
        if proc.stdout is not None:
            sel.register(proc.stdout, selectors.EVENT_READ)

        last_output_t = time.time()
        while True:
            if proc.poll() is not None:
                break

            now = time.time()
            if timeout_s > 0 and (now - start.timestamp()) > timeout_s:
                proc.kill()
                break

            events = sel.select(timeout=0.25)
            if events:
                for key, _mask in events:
                    stream = key.fileobj
                    line = stream.readline()
                    if not line:
                        continue
                    last_output_t = time.time()
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    log_f.write(line)
                    log_f.flush()

            now = time.time()
            if heartbeat_s > 0 and (now - last_output_t) >= heartbeat_s:
                hb = f"[heartbeat] still running; elapsed={(now - start.timestamp()):.0f}s\n"
                if status_path and status_path.exists():
                    try:
                        hb_status = _read_json(status_path)
                        hb = (
                            f"[heartbeat] still running; elapsed={(now - start.timestamp()):.0f}s; "
                            f"status={hb_status.get('phase')} bs={hb_status.get('batch_size')} "
                            f"last_update={hb_status.get('last_update')}\n"
                        )
                    except Exception:
                        pass
                sys.stdout.write(hb)
                sys.stdout.flush()
                log_f.write(hb)
                log_f.flush()
                last_output_t = now

        try:
            sel.close()
        except Exception:
            pass

        rc = proc.wait(timeout=10)
        end = datetime.now(timezone.utc)
        log_f.write(f"\n=== end ===\n{end.isoformat()}\nreturncode={rc}\n")
        log_f.flush()

    return {
        "ok": rc == 0,
        "returncode": rc,
        "start_time": start.isoformat(),
        "end_time": end.isoformat(),
        "duration_s": (end - start).total_seconds(),
    }


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


def _build_cli_equivalent_args_for_inproc(
    *,
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
    # Mirror _build_vllm_bench_cmd, but without the leading `vllm bench latency`.
    base = [
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
    return base + (extra_args or [])


def _run_inproc_latency_sweep_child(
    *,
    label: str,
    model_id: str,
    tp: int,
    max_model_len: int,
    input_len: int,
    output_len: int,
    batch_sizes: List[int],
    num_iters: int,
    extra_args: List[str],
    out_root: Path,
    timeout_s_per_bucket: int,
) -> int:
    """Child-mode runner: load model once, benchmark all batch sizes.

    Writes per-bucket artifacts to the same locations the parent expects:
      logs/{label}_bs{BS}.log
      json/{label}_bs{BS}.json          (raw vllm bench latency format)
      json/{label}_bs{BS}.runner.json   (runner status + timing + errors)
    """
    logs_dir = out_root / "logs"
    json_dir = out_root / "json"
    status_dir = out_root / "status"
    logs_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)
    status_dir.mkdir(parents=True, exist_ok=True)

    status_path = status_dir / f"{label}.json"
    child_log_path = logs_dir / f"{label}_child.log"

    def _update_status(phase: str, *, batch_size: Optional[int] = None, extra: Optional[Dict[str, Any]] = None) -> None:
        payload: Dict[str, Any] = {
            "label": label,
            "phase": phase,
            "batch_size": batch_size,
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "last_update": datetime.now(timezone.utc).isoformat(),
        }
        if extra:
            payload.update(extra)
        _write_json_atomic(status_path, payload)

    # Emit status/logs *before* heavy imports so supervisors can tell we're alive.
    _update_status("starting_import", extra={"model_id": model_id})
    with open(child_log_path, "a", encoding="utf-8", buffering=1) as child_log:
        child_log.write(f"=== child start ({label}) ===\n{datetime.now(timezone.utc).isoformat()}\n")
        child_log.write(f"model_id={model_id} tp={tp} max_model_len={max_model_len}\n")
        child_log.flush()

    # Lazy imports: avoid importing vLLM unless the child is actually executing.
    import dataclasses as _dataclasses
    import time as _time

    try:
        import numpy as np  # type: ignore

        from vllm import LLM, SamplingParams  # type: ignore
        from vllm.benchmarks import latency as vllm_latency  # type: ignore
        from vllm.engine.arg_utils import EngineArgs  # type: ignore
        from vllm.inputs import PromptType  # type: ignore
        from vllm.sampling_params import BeamSearchParams  # type: ignore
        from vllm.utils.argparse_utils import FlexibleArgumentParser  # type: ignore
    except Exception as e:
        # Import errors are common when not running under the vLLM venv; make it explicit.
        err = f"Failed to import vLLM benchmark deps in child runner: {e}"
        _update_status("import_failed", extra={"error": err})
        _write_text(out_root / f"child_{label}_import_error.log", err + "\n" + traceback.format_exc())
        with open(child_log_path, "a", encoding="utf-8", buffering=1) as child_log:
            child_log.write(err + "\n")
            child_log.write(traceback.format_exc() + "\n")
        return 2

    # Parse CLI-equivalent args once (using the first batch size), then override per-bucket.
    # This ensures we honor bench.extra_args using vLLM's own argparse schema.
    # Use vLLM's FlexibleArgumentParser so we support dotted "json-style" flags
    # like `-cc.pass_config.enable_sp=false` (CompilationConfig), etc.
    parser = FlexibleArgumentParser(add_help=False, add_json_tip=False)
    vllm_latency.add_cli_args(parser)
    seed_bs = batch_sizes[0] if batch_sizes else 1
    seed_out = json_dir / f"{label}_bs{seed_bs}.json"
    seed_argv = _build_cli_equivalent_args_for_inproc(
        model_id=model_id,
        tp=tp,
        max_model_len=max_model_len,
        input_len=input_len,
        output_len=output_len,
        batch_size=seed_bs,
        num_iters=num_iters,
        output_json=seed_out,
        extra_args=extra_args,
    )
    args = parser.parse_args(seed_argv)

    if getattr(args, "profile", False):
        # vllm bench latency --profile is a single-run action; sweeping doesn't make sense.
        _write_text(out_root / f"child_{label}_error.log", "Refusing to sweep with --profile enabled.\n")
        _update_status("error", extra={"error": "Refusing to sweep with --profile enabled."})
        return 2

    engine_args = EngineArgs.from_cli_args(args)
    _update_status("loading_model")
    llm = LLM(**_dataclasses.asdict(engine_args))
    _update_status("model_loaded")

    # Same invariant as vllm.benchmarks.latency.main
    if llm.llm_engine.model_config.max_model_len < (input_len + output_len):
        _write_text(
            out_root / f"child_{label}_error.log",
            "max_model_len is smaller than input_len + output_len; adjust target.json.\n",
        )
        _update_status("error", extra={"error": "max_model_len < input_len + output_len"})
        return 2

    sampling_params = SamplingParams(
        n=int(getattr(args, "n", 1)),
        temperature=1.0,
        top_p=1.0,
        ignore_eos=True,
        max_tokens=output_len,
        detokenize=not bool(getattr(args, "disable_detokenize", False)),
    )

    for bs in batch_sizes:
        raw_json = json_dir / f"{label}_bs{bs}.json"
        runner_json = json_dir / f"{label}_bs{bs}.runner.json"
        log_path = logs_dir / f"{label}_bs{bs}.log"

        # Override per-bucket args that should affect only prompt construction / output paths.
        setattr(args, "batch_size", int(bs))
        setattr(args, "input_len", int(input_len))
        setattr(args, "output_len", int(output_len))
        setattr(args, "num_iters", int(num_iters))
        setattr(args, "output_json", str(raw_json))

        start = datetime.now(timezone.utc)
        status: Dict[str, Any] = {
            "ok": False,
            "label": label,
            "batch_size": bs,
            "start_time": start.isoformat(),
        }

        with open(log_path, "w", encoding="utf-8", buffering=1) as bucket_log, open(
            child_log_path, "a", encoding="utf-8", buffering=1
        ) as child_log:

            def _log(msg: str) -> None:
                bucket_log.write(msg + "\n")
                bucket_log.flush()
                child_log.write(msg + "\n")
                child_log.flush()
                print(msg, flush=True)

            try:
                _update_status("bucket_start", batch_size=bs)
                _log(f"=== inproc vllm bench latency sweep ({label}) bs={bs} ===")
                _log(
                    _format_cmd_for_md(
                        ["vllm", "bench", "latency"]
                        + _build_cli_equivalent_args_for_inproc(
                            model_id=model_id,
                            tp=tp,
                            max_model_len=max_model_len,
                            input_len=input_len,
                            output_len=output_len,
                            batch_size=bs,
                            num_iters=num_iters,
                            output_json=raw_json,
                            extra_args=extra_args,
                        ),
                        {},
                    )
                )

                dummy_prompt_token_ids = np.random.randint(10000, size=(bs, input_len))
                dummy_prompts: list[PromptType] = [
                    {"prompt_token_ids": batch} for batch in dummy_prompt_token_ids.tolist()
                ]

                def llm_generate() -> None:
                    if not bool(getattr(args, "use_beam_search", False)):
                        llm.generate(dummy_prompts, sampling_params=sampling_params, use_tqdm=False)
                    else:
                        llm.beam_search(
                            dummy_prompts,
                            BeamSearchParams(
                                beam_width=int(getattr(args, "n", 1)),
                                max_tokens=output_len,
                                ignore_eos=True,
                            ),
                        )

                def run_to_completion() -> float:
                    t0 = _time.perf_counter()
                    llm_generate()
                    t1 = _time.perf_counter()
                    return t1 - t0

                warmup = int(getattr(args, "num_iters_warmup", 10))
                deadline = _time.monotonic() + float(timeout_s_per_bucket)
                _update_status("warmup", batch_size=bs, extra={"num_iters_warmup": warmup})
                _log("Warming up...")
                last_status_t = _time.monotonic()
                for i in range(warmup):
                    if _time.monotonic() > deadline:
                        raise TimeoutError(f"Bucket timeout exceeded ({timeout_s_per_bucket}s)")
                    llm_generate()
                    if (_time.monotonic() - last_status_t) > 5.0:
                        _update_status("warmup", batch_size=bs, extra={"warmup_iter": i + 1})
                        last_status_t = _time.monotonic()

                # Benchmark.
                _update_status("benchmark", batch_size=bs, extra={"num_iters": int(getattr(args, "num_iters", 30))})
                latencies: List[float] = []
                last_status_t = _time.monotonic()
                num_iters_eff = int(getattr(args, "num_iters", 30))
                for i in range(num_iters_eff):
                    if _time.monotonic() > deadline:
                        raise TimeoutError(f"Bucket timeout exceeded ({timeout_s_per_bucket}s)")
                    latencies.append(run_to_completion())
                    if (_time.monotonic() - last_status_t) > 5.0:
                        _update_status("benchmark", batch_size=bs, extra={"iter": i + 1})
                        last_status_t = _time.monotonic()

                # Match vLLM bench JSON schema.
                arr = np.array(latencies, dtype=np.float64)
                percentages = [10, 25, 50, 75, 90, 99]
                percentiles = np.percentile(arr, percentages)
                avg_latency = float(np.mean(arr))

                _log(f"Avg latency: {avg_latency} seconds")
                for percentage, percentile in zip(percentages, percentiles):
                    _log(f"{percentage}% percentile latency: {float(percentile)} seconds")

                raw = {
                    "avg_latency": avg_latency,
                    "latencies": [float(x) for x in latencies],
                    "percentiles": {str(k): float(v) for k, v in zip(percentages, percentiles)},
                }
                _write_json(raw_json, raw)

                # Optional: keep parity with vLLM bench output sidecar format.
                try:
                    vllm_latency.save_to_pytorch_benchmark_format(args, raw)
                except Exception:
                    _log("WARNING: failed to write pytorch benchmark sidecar JSON.")
                    _log(traceback.format_exc())

                end = datetime.now(timezone.utc)
                status.update({
                    "ok": True,
                    "end_time": end.isoformat(),
                    "duration_s": (end - start).total_seconds(),
                })
                _update_status("bucket_done", batch_size=bs, extra={"ok": True})
            except Exception as e:
                end = datetime.now(timezone.utc)
                status.update({
                    "ok": False,
                    "error": f"{type(e).__name__}: {e}",
                    "end_time": end.isoformat(),
                    "duration_s": (end - start).total_seconds(),
                    "traceback": traceback.format_exc(),
                })
                _update_status("bucket_failed", batch_size=bs, extra={"ok": False, "error": status.get("error")})
                # Ensure the raw JSON exists even on failure (helps parent remain deterministic).
                if not raw_json.exists():
                    _write_json(raw_json, {"error": status["error"]})

            _write_json(runner_json, status)

    return 0


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
    p.add_argument("--out-name", type=str, default="e2e_latency", help="Output directory name under {artifact_dir}")
    p.add_argument("--timeout-s", type=int, default=1800, help="Timeout per bucket (seconds)")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing output dir instead of archiving it")
    p.add_argument("--require-fastpath", action="store_true", help="Fail if opt fast-path evidence patterns do not pass")
    p.add_argument(
        "--execution-mode",
        type=str,
        default="inproc_sweep",
        choices=["inproc_sweep", "cli_per_bs"],
        help=(
            "How to execute the sweep. "
            "'inproc_sweep' loads the model once per label and benchmarks all batch sizes (faster). "
            "'cli_per_bs' shells out to `vllm bench latency` per batch size (slower, legacy)."
        ),
    )
    p.add_argument("--gpu-lock", action="store_true", default=True, help="Prevent concurrent sweeps on same GPUs (default)")
    p.add_argument("--no-gpu-lock", action="store_false", dest="gpu_lock", help="Disable GPU lock (not recommended)")
    p.add_argument("--heartbeat-s", type=int, default=30, help="Heartbeat interval while a subprocess is quiet")
    p.add_argument("--_child-label", type=str, default=None, help=argparse.SUPPRESS)
    p.add_argument("--_out-root", type=str, default=None, help=argparse.SUPPRESS)

    args = p.parse_args()

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
    # Agent-friendly fallback: if `vllm` isn't on PATH, run via `python -m ...` in
    # the current interpreter env so CLI mode still works under managed shells.
    if (
        vllm_exe
        and len(vllm_exe) >= 1
        and vllm_exe[0] == "vllm"
        and shutil.which("vllm") is None
    ):
        vllm_exe = [sys.executable, "-m", "vllm.entrypoints.cli.main"]

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

    baseline_extra_args = _maybe_list_str(bench, "baseline_extra_args")
    opt_extra_args = _maybe_list_str(bench, "opt_extra_args")
    baseline_args = extra_args + baseline_extra_args
    opt_args = extra_args + opt_extra_args

    # Output directory:
    # - Parent run: choose/create and archive existing outputs unless --overwrite.
    # - Child run: parent passes the resolved out_root explicitly via --_out-root.
    if args._out_root:
        out_root = Path(args._out_root).expanduser().resolve()
        out_root.mkdir(parents=True, exist_ok=True)
    else:
        out_root = _prepare_out_root(artifact_dir=artifact_dir, out_name=args.out_name, overwrite=args.overwrite)

    logs_dir = out_root / "logs"
    json_dir = out_root / "json"
    status_dir = out_root / "status"
    logs_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)
    status_dir.mkdir(parents=True, exist_ok=True)

    out_json_path = out_root / "e2e_latency_results.json"
    out_md_path = out_root / "e2e_latency_results.md"

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
    print(f"out_dir: {out_root}")
    print(f"model_id: {model_id}")
    print(f"tp: {tp}, ep: {ep}, max_model_len: {max_model_len}")
    print(f"workload: input_len={input_len}, output_len={output_len}, num_iters={num_iters}")
    print(f"batch_sizes: {batch_sizes}")
    print(f"baseline_label: {baseline_label}, opt_label: {opt_label}")
    print(f"execution_mode: {args.execution_mode}")
    print(f"gpu_lock: {args.gpu_lock}")

    if ep != 1:
        print(
            "WARNING: target.ep != 1, but this script does not pass an explicit EP flag to "
            "`vllm bench latency`. Ensure your EP configuration is applied via bench.extra_args "
            "and/or environment, and that baseline/opt runs are truly production-parity.",
            file=sys.stderr,
        )

    all_runs: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target_json": str(target_path),
        "out_dir": str(out_root),
        "out_name": args.out_name,
        "execution_mode": args.execution_mode,
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
            "baseline_extra_args": baseline_extra_args,
            "opt_extra_args": opt_extra_args,
            "baseline_env": baseline_env,
            "opt_env": opt_env,
            "baseline_label": baseline_label,
            "opt_label": opt_label,
            "fastpath_evidence": fpe,
        },
        "results": [],
    }

    # Acquire GPU lock for the entire run (prevents meaningless contention).
    lock_handle = _acquire_gpu_lock(artifact_dir=artifact_dir, enabled=bool(args.gpu_lock))

    # Record the per-bucket plan deterministically (for stable paths + repro commands).
    planned: List[Dict[str, Any]] = []
    for bs in batch_sizes:
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
            extra_args=baseline_args,
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
            extra_args=opt_args,
        )

        print(f"\n=== batch_size={bs} ===")
        print(f"Baseline cmd: {_format_cmd_for_md(baseline_cmd, baseline_env)}")
        print(f"Opt cmd: {_format_cmd_for_md(opt_cmd, {**baseline_env, **opt_env})}")

        planned.append({
            "batch_size": bs,
            "baseline_cmd": baseline_cmd,
            "opt_cmd": opt_cmd,
            "baseline_log": baseline_log,
            "opt_log": opt_log,
            "baseline_json": baseline_json,
            "opt_json": opt_json,
        })

        # Populate a minimal row deterministically (stable paths + repro commands).
        all_runs["results"].append({
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
        })

    # Ensure per-bucket log files exist early (so tailing works even if model load is slow).
    for p in planned:
        for log_path in (Path(p["baseline_log"]), Path(p["opt_log"])):
            if not log_path.exists():
                _write_text(
                    log_path,
                    "PENDING: benchmark has not started for this bucket yet.\n"
                    "If this stays unchanged for a long time, check logs/*_child.log and status/*.json.\n",
                )

    # Seed status files so supervisors can poll immediately.
    for label in (baseline_label, opt_label):
        st = status_dir / f"{label}.json"
        if not st.exists():
            _write_json_atomic(
                st,
                {
                    "label": label,
                    "phase": "queued",
                    "batch_size": None,
                    "pid": os.getpid(),
                    "host": socket.gethostname(),
                    "last_update": datetime.now(timezone.utc).isoformat(),
                },
            )

    if args._child_label:
        # Child mode: just run a single label sweep into the provided out_root.
        label = args._child_label
        if label not in (baseline_label, opt_label):
            raise SystemExit(f"Invalid --_child-label: {label!r} (expected {baseline_label!r} or {opt_label!r})")
        label_args = baseline_args if label == baseline_label else opt_args
        returncode = _run_inproc_latency_sweep_child(
            label=label,
            model_id=model_id,
            tp=tp,
            max_model_len=max_model_len,
            input_len=input_len,
            output_len=output_len,
            batch_sizes=batch_sizes,
            num_iters=num_iters,
            extra_args=label_args,
            out_root=out_root,
            timeout_s_per_bucket=args.timeout_s,
        )
        raise SystemExit(returncode)

    # Summaries
    def _status_from_evidence(ev: Dict[str, Any], patterns_configured: bool) -> str:
        if not patterns_configured:
            return "unknown"
        return "pass" if ev.get("ok") else "fail"

    baseline_patterns_configured = bool(baseline_run.require_patterns or baseline_run.forbid_patterns)
    opt_patterns_configured = bool(opt_run.require_patterns or opt_run.forbid_patterns)

    if args.execution_mode == "inproc_sweep":
        script_path = Path(__file__).resolve()
        child_timeout = int(args.timeout_s) * max(1, len(batch_sizes)) + 1800

        for run in (baseline_run, opt_run):
            child_cmd = [
                sys.executable,
                str(script_path),
                "--artifact-dir",
                str(artifact_dir),
                "--target-json",
                str(target_path),
                "--execution-mode",
                "inproc_sweep",
                "--timeout-s",
                str(args.timeout_s),
                "--out-name",
                args.out_name,
                "--_out-root",
                str(out_root),
                "--no-gpu-lock",
                "--_child-label",
                run.label,
            ]
            print(f"\n=== Running inproc sweep for {run.label} ===")
            print(f"Child cmd: {_format_cmd_for_md(child_cmd, {})}")
            child_log = logs_dir / f"{run.label}_supervisor.log"
            child_status = status_dir / f"{run.label}.json"
            child_res = _run_cmd_streaming(
                child_cmd,
                env=run.env,
                cwd=None,
                timeout_s=child_timeout,
                log_path=child_log,
                heartbeat_s=int(args.heartbeat_s),
                status_path=child_status,
            )
            if not child_res.get("ok"):
                raise SystemExit(
                    f"Inproc sweep child failed for {run.label}: returncode={child_res.get('returncode')}. "
                    f"See {child_log} and {logs_dir / f'{run.label}_child.log'}"
                )

        # Populate per-bucket entries from artifacts written by children.
        new_rows: List[Dict[str, Any]] = []
        for p, row in zip(planned, all_runs["results"]):
            bs = int(p["batch_size"])
            baseline_log = Path(p["baseline_log"])
            opt_log = Path(p["opt_log"])
            baseline_json = Path(p["baseline_json"])
            opt_json = Path(p["opt_json"])

            baseline_runner_json = json_dir / f"{baseline_label}_bs{bs}.runner.json"
            opt_runner_json = json_dir / f"{opt_label}_bs{bs}.runner.json"

            baseline_text = ""
            opt_text = ""
            if baseline_log.exists():
                baseline_text = baseline_log.read_text(encoding="utf-8", errors="replace")
            if opt_log.exists():
                opt_text = opt_log.read_text(encoding="utf-8", errors="replace")

            baseline_raw = _read_json(baseline_json) if baseline_json.exists() else {}
            opt_raw = _read_json(opt_json) if opt_json.exists() else {}

            baseline_metrics = _metrics_from_vllm_latency_json(baseline_raw) or _parse_latency_metrics(baseline_text)
            opt_metrics = _metrics_from_vllm_latency_json(opt_raw) or _parse_latency_metrics(opt_text)

            baseline_evidence = _check_patterns(baseline_text, baseline_run.require_patterns, baseline_run.forbid_patterns)
            opt_evidence = _check_patterns(opt_text, opt_run.require_patterns, opt_run.forbid_patterns)

            baseline_status = _read_json(baseline_runner_json) if baseline_runner_json.exists() else {"ok": None}
            opt_status = _read_json(opt_runner_json) if opt_runner_json.exists() else {"ok": None}

            baseline_entry = {
                "ok": baseline_status.get("ok"),
                "returncode": 0 if baseline_status.get("ok") else 1,
                "cmd": p["baseline_cmd"],
                "env_overrides": baseline_env,
                "metrics": baseline_metrics,
                "avg_s": baseline_metrics.get("avg_s"),
                "fastpath_evidence": {
                    **baseline_evidence,
                    "status": _status_from_evidence(baseline_evidence, baseline_patterns_configured),
                },
                "log": str(baseline_log.relative_to(out_root)),
                "output_json": str(baseline_json.relative_to(out_root)),
                "runner_json": str(baseline_runner_json.relative_to(out_root)),
                "timing": {
                    "start_time": baseline_status.get("start_time"),
                    "end_time": baseline_status.get("end_time"),
                    "duration_s": baseline_status.get("duration_s"),
                },
            }
            opt_entry = {
                "ok": opt_status.get("ok"),
                "returncode": 0 if opt_status.get("ok") else 1,
                "cmd": p["opt_cmd"],
                "env_overrides": {**baseline_env, **opt_env},
                "metrics": opt_metrics,
                "avg_s": opt_metrics.get("avg_s"),
                "fastpath_evidence": {
                    **opt_evidence,
                    "status": _status_from_evidence(opt_evidence, opt_patterns_configured),
                },
                "log": str(opt_log.relative_to(out_root)),
                "output_json": str(opt_json.relative_to(out_root)),
                "runner_json": str(opt_runner_json.relative_to(out_root)),
                "timing": {
                    "start_time": opt_status.get("start_time"),
                    "end_time": opt_status.get("end_time"),
                    "duration_s": opt_status.get("duration_s"),
                },
            }

            new_row = {"batch_size": bs, baseline_label: baseline_entry, opt_label: opt_entry}

            b_avg = baseline_entry.get("avg_s")
            o_avg = opt_entry.get("avg_s")
            if isinstance(b_avg, (int, float)) and isinstance(o_avg, (int, float)) and o_avg > 0:
                speedup = b_avg / o_avg
                improvement_pct = (b_avg - o_avg) / b_avg * 100.0 if b_avg != 0 else None
                new_row["speedup"] = speedup
                new_row["improvement_pct"] = improvement_pct

            new_rows.append(new_row)

            if args.require_fastpath and opt_patterns_configured and not opt_evidence.get("ok"):
                raise SystemExit(
                    f"Fast-path evidence FAILED for opt at BS={bs}. "
                    f"Missing={opt_evidence.get('require_miss')}, forbidden_hits={opt_evidence.get('forbid_hits')}. "
                    f"See {opt_log}"
                )

        all_runs["results"] = new_rows

    else:
        # Legacy mode: shell out per batch size.
        new_rows: List[Dict[str, Any]] = []
        for p in planned:
            bs = int(p["batch_size"])
            baseline_cmd = p["baseline_cmd"]
            opt_cmd = p["opt_cmd"]
            baseline_log = Path(p["baseline_log"])
            opt_log = Path(p["opt_log"])
            baseline_json = Path(p["baseline_json"])
            opt_json = Path(p["opt_json"])

            # Run baseline.
            baseline_res = _run_cmd_streaming(
                baseline_cmd,
                env=baseline_run.env,
                cwd=None,
                timeout_s=args.timeout_s,
                log_path=baseline_log,
                heartbeat_s=int(args.heartbeat_s),
            )

            # Run opt.
            opt_res = _run_cmd_streaming(
                opt_cmd,
                env=opt_run.env,
                cwd=None,
                timeout_s=args.timeout_s,
                log_path=opt_log,
                heartbeat_s=int(args.heartbeat_s),
            )

            baseline_text = baseline_log.read_text(encoding="utf-8", errors="replace") if baseline_log.exists() else ""
            opt_text = opt_log.read_text(encoding="utf-8", errors="replace") if opt_log.exists() else ""

            baseline_raw = _read_json(baseline_json) if baseline_json.exists() else {}
            opt_raw = _read_json(opt_json) if opt_json.exists() else {}

            baseline_metrics = _metrics_from_vllm_latency_json(baseline_raw) or _parse_latency_metrics(baseline_text)
            opt_metrics = _metrics_from_vllm_latency_json(opt_raw) or _parse_latency_metrics(opt_text)

            baseline_evidence = _check_patterns(baseline_text, baseline_run.require_patterns, baseline_run.forbid_patterns)
            opt_evidence = _check_patterns(opt_text, opt_run.require_patterns, opt_run.forbid_patterns)

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

            new_rows.append(row)

            if args.require_fastpath and opt_patterns_configured and not opt_evidence.get("ok"):
                raise SystemExit(
                    f"Fast-path evidence FAILED for opt at BS={bs}. "
                    f"Missing={opt_evidence.get('require_miss')}, forbidden_hits={opt_evidence.get('forbid_hits')}. "
                    f"See {opt_log}"
                )
        all_runs["results"] = new_rows

    # Write outputs.
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

    _write_text(out_md_path, "\n".join(md_lines))

    print(f"\nWrote: {out_json_path}")
    print(f"Wrote: {out_md_path}")

    if lock_handle is not None:
        try:
            lock_handle.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
