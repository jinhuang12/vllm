#!/usr/bin/env python3
"""Run production-parity E2E latency sweeps via `vllm bench latency`.

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
    "num_iters": 5
  },
  "bench": {
    "runner": "vllm_bench_latency",
    "vllm_cmd": "vllm",
    "extra_args": [],
    "baseline_extra_args": [],
    "opt_extra_args": [],
    "baseline_env": {},
    "opt_env": {"<ENABLE_FLAG>": "1"},
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
- Avoids model reload thrash: uses an in-process sweep runner that loads the
  model once per label (baseline/opt) and benchmarks all batch sizes in that
  process. This reduces end-to-end sweep time without changing per-iteration
  latency measurement.
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
import ast


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


def _ensure_worktree_pythonpath(env: Dict[str, str]) -> Dict[str, str]:
    """Prepend CWD to PYTHONPATH so worktree's ``vllm`` package is imported first.

    When the sweep script runs inside a git worktree, the editable install
    resolves to the main repo.  By prepending CWD (the worktree root) to
    PYTHONPATH, child subprocesses will ``import vllm`` from the worktree's
    modified code rather than the main repo's.
    """
    env = dict(env)  # shallow copy — don't mutate caller's dict
    cwd = os.getcwd()
    existing = env.get("PYTHONPATH", "")
    if existing:
        # Avoid duplicating if CWD is already the first entry.
        parts = existing.split(":")
        if parts[0] != cwd:
            env["PYTHONPATH"] = f"{cwd}:{existing}"
    else:
        env["PYTHONPATH"] = cwd
    return env


def _sanitize_vllm_op_env(env: Dict[str, str]) -> Dict[str, str]:
    """Strip all VLLM_OP* env vars from *env* to prevent cross-track contamination.

    Prior-round gating flags (e.g. VLLM_OP001=1) may be present in the inherited
    process environment because ``vllm/envs.py`` was edited on the worktree branch.
    If left in place they silently activate stale optimizations during baseline and
    opt sweeps, producing false negatives (as observed in the op003/op004 audit).

    Non-optimization VLLM vars (VLLM_ATTENTION_BACKEND, VLLM_USE_V1, etc.) are
    preserved — only keys matching ``VLLM_OP\\d+`` are removed.
    """
    import re as _re
    _vllm_op_re = _re.compile(r"^VLLM_OP\d+$")
    return {k: v for k, v in env.items() if not _vllm_op_re.match(k)}


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


def _expand_workload_to_buckets(workload: Dict[str, Any]) -> List[Dict[str, int]]:
    """Normalize workload spec into a list of {input_len, output_len, batch_size} dicts.

    Supports two formats:
    1. New ``workload_matrix`` (list of dicts) — used when present, flat fields ignored.
    2. Legacy flat ``{input_len, output_len, batch_sizes}`` — expanded into matrix internally.
    """
    matrix = workload.get("workload_matrix")
    if matrix is not None:
        if not isinstance(matrix, list) or len(matrix) == 0:
            raise SystemExit("workload.workload_matrix must be a non-empty list")
        required_keys = {"input_len", "output_len", "batch_size"}
        buckets: List[Dict[str, int]] = []
        seen: set = set()
        for i, entry in enumerate(matrix):
            if not isinstance(entry, dict):
                raise SystemExit(f"workload.workload_matrix[{i}] must be a dict, got {type(entry).__name__}")
            missing = required_keys - set(entry.keys())
            if missing:
                raise SystemExit(f"workload.workload_matrix[{i}] missing keys: {missing}")
            bucket = {
                "input_len": int(entry["input_len"]),
                "output_len": int(entry["output_len"]),
                "batch_size": int(entry["batch_size"]),
            }
            key = (bucket["input_len"], bucket["output_len"], bucket["batch_size"])
            if key in seen:
                raise SystemExit(
                    f"Duplicate bucket in workload.workload_matrix: "
                    f"input_len={bucket['input_len']}, output_len={bucket['output_len']}, batch_size={bucket['batch_size']}"
                )
            seen.add(key)
            buckets.append(bucket)
        return buckets
    else:
        # Legacy flat format.
        input_len = workload.get("input_len")
        output_len = workload.get("output_len")
        batch_sizes = workload.get("batch_sizes")
        if input_len is None or output_len is None or batch_sizes is None:
            raise SystemExit("workload must have input_len, output_len, and batch_sizes (or workload_matrix)")
        return [
            {"input_len": int(input_len), "output_len": int(output_len), "batch_size": int(bs)}
            for bs in batch_sizes
        ]


def _bucket_file_tag(bucket: Dict[str, int], all_buckets: List[Dict[str, int]]) -> str:
    """Return a file-name tag for *bucket*.

    Returns ``bs{BS}`` when all buckets share the same (input_len, output_len)
    (homogeneous), otherwise ``il{IL}_ol{OL}_bs{BS}`` (heterogeneous).
    """
    il_ol_set = {(b["input_len"], b["output_len"]) for b in all_buckets}
    if len(il_ol_set) <= 1:
        return f"bs{bucket['batch_size']}"
    return f"il{bucket['input_len']}_ol{bucket['output_len']}_bs{bucket['batch_size']}"


def _validate_buckets_model_len(buckets: List[Dict[str, int]], max_model_len: int) -> None:
    """Raise SystemExit if any bucket exceeds max_model_len."""
    for b in buckets:
        total = b["input_len"] + b["output_len"]
        if total > max_model_len:
            raise SystemExit(
                f"Bucket (input_len={b['input_len']}, output_len={b['output_len']}, batch_size={b['batch_size']}) "
                f"requires {total} tokens but max_model_len={max_model_len}"
            )


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


# ---- GSM8K helpers (adapted from tests/evals/gsm8k/gsm8k_eval.py) ----

_GSM8K_SUBSET_PATH = Path(__file__).parent.parent / "data" / "gsm8k_subset.json"
_INVALID_ANSWER = -9999999


def _download_and_cache_file(url: str, filename: str | None = None) -> str:
    import requests as _requests_mod  # Lazy: only needed for >200 question fallback
    if filename is None:
        filename = os.path.join("/tmp", url.split("/")[-1])
    if os.path.exists(filename):
        return filename
    print(f"Downloading from {url} to {filename}")
    response = _requests_mod.get(url, stream=True)
    response.raise_for_status()
    with open(filename, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024):
            f.write(chunk)
    return filename


def _read_jsonl(filename: str):
    with open(filename) as fin:
        for line in fin:
            if not line.startswith("#"):
                yield json.loads(line)


def _load_gsm8k_data(num_questions: int) -> Tuple[List[dict], List[dict]]:
    """Load GSM8K data — bundled subset first, GitHub fallback if needed."""
    if num_questions <= 200 and _GSM8K_SUBSET_PATH.exists():
        with open(_GSM8K_SUBSET_PATH) as f:
            data = json.load(f)
        return data["train"], data["test"][:num_questions]
    # Fallback to full download
    train_url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl"
    test_url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
    train_file = _download_and_cache_file(train_url)
    test_file = _download_and_cache_file(test_url)
    return list(_read_jsonl(train_file)), list(_read_jsonl(test_file))


def _get_answer_value(answer_str: str) -> int:
    answer_str = answer_str.replace(",", "")
    numbers = re.findall(r"\d+", answer_str)
    if len(numbers) < 1:
        return _INVALID_ANSWER
    try:
        return ast.literal_eval(numbers[-1])
    except SyntaxError:
        return _INVALID_ANSWER


def _build_gsm8k_prompts(
    num_questions: int = 200, num_shots: int = 5
) -> Tuple[List[str], List[int]]:
    """Build few-shot GSM8K prompts and ground-truth labels."""
    if num_questions == 0:
        return [], []
    train_data, test_data = _load_gsm8k_data(num_questions)
    num_questions = min(num_questions, len(test_data))
    few_shot_examples = ""
    for i in range(min(num_shots, len(train_data))):
        few_shot_examples += (
            f"Question: {train_data[i]['question']}\n"
            f"Answer: {train_data[i]['answer']}\n\n"
        )
    prompts, labels = [], []
    for i in range(num_questions):
        prompts.append(few_shot_examples + f"Question: {test_data[i]['question']}\nAnswer:")
        labels.append(_get_answer_value(test_data[i]["answer"]))
    assert all(label != _INVALID_ANSWER for label in labels), "Some labels are invalid"
    return prompts, labels


def _serialize_correctness_outputs(outputs) -> List[Dict[str, Any]]:
    """Serialize vLLM RequestOutput objects to JSON-safe dicts."""
    serialized = []
    for i, req_output in enumerate(outputs):
        comp = req_output.outputs[0]
        token_ids = list(comp.token_ids)
        logprobs_list = []
        if comp.logprobs is not None:
            for pos_logprobs in comp.logprobs:
                top_lps = {str(tid): lp.logprob for tid, lp in pos_logprobs.items()}
                logprobs_list.append({"top_logprobs": top_lps})
        serialized.append({
            "prompt_index": i,
            "token_ids": token_ids,
            "text": comp.text,
            "logprobs": logprobs_list,
            "num_tokens": len(token_ids),
        })
    return serialized


def _score_gsm8k_predictions(outputs, labels: List[int]) -> Tuple[List[int], float]:
    """Score GSM8K outputs. Returns (predictions_list, accuracy)."""
    preds = []
    for req_output in outputs:
        text = req_output.outputs[0].text
        preds.append(_get_answer_value(text))
    correct = sum(1 for p, l in zip(preds, labels) if p == l)
    accuracy = correct / len(labels) if labels else 0.0
    return preds, accuracy


def _compare_correctness(
    *,
    golden_refs: List[Dict[str, Any]],
    opt_outputs: List[Dict[str, Any]],
    labels: Optional[List[int]],
    baseline_preds: Optional[List[int]],
    opt_preds: Optional[List[int]],
) -> Dict[str, Any]:
    """Compare optimized outputs against golden references (Gate 5.1b v2).

    Single accuracy gate: ``opt_accuracy >= baseline_accuracy``.
    Token-level data is computed as diagnostics only (never affects verdict).
    """
    num_questions = min(len(golden_refs), len(opt_outputs))

    # ---- Edge case: empty question set ----
    if num_questions == 0:
        return {
            "gate": "5.1b", "verdict": "FAIL",
            "num_questions": 0,
            "baseline_accuracy": 0.0, "optimized_accuracy": 0.0,
            "accuracy_delta": 0.0,
            "baseline_correct_count": 0, "optimized_correct_count": 0,
            "questions_lost": [], "questions_gained": [],
            "diagnostics": {
                "divergent_questions": 0,
                "first_divergence_positions_p50": -1,
                "first_divergence_positions_p95": -1,
                "churn_rate": 0.0,
                "note": "Token-level data is informational only and does not affect the verdict.",
            },
        }

    # ---- Edge case: accuracy gate requires labels + preds ----
    if labels is None or baseline_preds is None or opt_preds is None:
        return {
            "gate": "5.1b", "verdict": "FAIL",
            "num_questions": num_questions,
            "baseline_accuracy": 0.0, "optimized_accuracy": 0.0,
            "accuracy_delta": 0.0,
            "baseline_correct_count": 0, "optimized_correct_count": 0,
            "questions_lost": [], "questions_gained": [],
            "diagnostics": {
                "divergent_questions": 0,
                "first_divergence_positions_p50": -1,
                "first_divergence_positions_p95": -1,
                "churn_rate": 0.0,
                "note": "Token-level data is informational only and does not affect the verdict.",
            },
            "_error": "Accuracy gate requires labels, baseline_preds, and opt_preds.",
        }

    # ---- Accuracy computation ----
    n = min(len(labels), len(baseline_preds), len(opt_preds), num_questions)
    labels_t = labels[:n]
    baseline_preds_t = baseline_preds[:n]
    opt_preds_t = opt_preds[:n]

    baseline_correct = {i for i, (p, l) in enumerate(zip(baseline_preds_t, labels_t)) if p == l}
    opt_correct = {i for i, (p, l) in enumerate(zip(opt_preds_t, labels_t)) if p == l}
    questions_lost = sorted(baseline_correct - opt_correct)
    questions_gained = sorted(opt_correct - baseline_correct)

    baseline_acc = len(baseline_correct) / n
    opt_acc = len(opt_correct) / n

    # ---- Baseline accuracy floor ----
    if len(baseline_correct) == 0:
        return {
            "gate": "5.1b", "verdict": "FAIL",
            "num_questions": n,
            "baseline_accuracy": 0.0, "optimized_accuracy": round(opt_acc, 4),
            "accuracy_delta": round(opt_acc, 4),
            "baseline_correct_count": 0,
            "optimized_correct_count": len(opt_correct),
            "questions_lost": [], "questions_gained": questions_gained,
            "infrastructure_error": True,
            "infrastructure_message": "Baseline accuracy is 0% — model cannot solve any GSM8K questions; environment suspect.",
            "diagnostics": {
                "divergent_questions": 0,
                "first_divergence_positions_p50": -1,
                "first_divergence_positions_p95": -1,
                "churn_rate": 0.0,
                "note": "Token-level data is informational only and does not affect the verdict.",
            },
        }

    # ---- Verdict: opt_accuracy >= baseline_accuracy ----
    verdict = "PASS" if opt_acc >= baseline_acc else "FAIL"

    # ---- Token-level diagnostics (informational only) ----
    first_divergence_positions: List[int] = []
    divergent_questions = 0
    all_empty = True

    for q in range(n):
        b_ids = golden_refs[q]["token_ids"]
        o_ids = opt_outputs[q]["token_ids"]
        if b_ids or o_ids:
            all_empty = False
        min_len = min(len(b_ids), len(o_ids))
        first_div = next((p for p in range(min_len) if b_ids[p] != o_ids[p]), -1)
        if first_div == -1 and len(b_ids) != len(o_ids):
            first_div = min_len
        if first_div >= 0:
            divergent_questions += 1
            first_divergence_positions.append(first_div)

    # All outputs empty → override to FAIL regardless of accuracy
    if all_empty and n > 0:
        verdict = "FAIL"

    # p50/p95 of first_divergence_positions (across divergent questions only)
    if first_divergence_positions:
        sorted_pos = sorted(first_divergence_positions)
        p50_idx = max(0, int(len(sorted_pos) * 0.50) - 1)
        p95_idx = max(0, int(len(sorted_pos) * 0.95) - 1)
        fdp_p50 = sorted_pos[p50_idx]
        fdp_p95 = sorted_pos[p95_idx]
    else:
        fdp_p50 = -1
        fdp_p95 = -1

    churn_rate = round((len(questions_lost) + len(questions_gained)) / n, 4) if n else 0.0

    return {
        "gate": "5.1b",
        "verdict": verdict,
        "num_questions": n,
        "baseline_accuracy": round(baseline_acc, 4),
        "optimized_accuracy": round(opt_acc, 4),
        "accuracy_delta": round(opt_acc - baseline_acc, 4),
        "baseline_correct_count": len(baseline_correct),
        "optimized_correct_count": len(opt_correct),
        "questions_lost": questions_lost,
        "questions_gained": questions_gained,
        "diagnostics": {
            "divergent_questions": divergent_questions,
            "first_divergence_positions_p50": fdp_p50,
            "first_divergence_positions_p95": fdp_p95,
            "churn_rate": churn_rate,
            "note": "Token-level data is informational only and does not affect the verdict.",
        },
        "_diagnostic_notes": "p50/p95 computed across questions where first_divergence_pos >= 0; set to -1 if no questions diverge. churn_rate = (questions_lost + questions_gained) / num_questions.",
    }


def _run_inproc_latency_sweep_child(
    *,
    label: str,
    model_id: str,
    tp: int,
    max_model_len: int,
    buckets: List[Dict[str, int]],
    num_iters: int,
    extra_args: List[str],
    out_root: Path,
    timeout_s_per_bucket: int,
    nsys_profile: bool = False,
    capture_golden_refs: bool = False,
    verify_correctness: bool = False,
    correctness_num_questions: int = 200,
) -> int:
    """Child-mode runner: load model once, benchmark all buckets.

    Each bucket is a dict with ``input_len``, ``output_len``, ``batch_size``.
    SamplingParams are reconstructed per-bucket (output_len may vary).

    Writes per-bucket artifacts using ``_bucket_file_tag`` for naming:
      logs/{label}_{tag}.log
      json/{label}_{tag}.json          (raw vllm bench latency format)
      json/{label}_{tag}.runner.json   (runner status + timing + errors)
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

    # Parse CLI-equivalent args once (using the first bucket), then override per-bucket.
    # This ensures we honor bench.extra_args using vLLM's own argparse schema.
    # Use vLLM's FlexibleArgumentParser so we support dotted "json-style" flags
    # like `-cc.pass_config.enable_sp=false` (CompilationConfig), etc.
    seed = buckets[0] if buckets else {"input_len": 64, "output_len": 512, "batch_size": 1}
    parser = FlexibleArgumentParser(add_help=False, add_json_tip=False)
    vllm_latency.add_cli_args(parser)
    seed_tag = _bucket_file_tag(seed, buckets)
    seed_out = json_dir / f"{label}_{seed_tag}.json"
    seed_argv = _build_cli_equivalent_args_for_inproc(
        model_id=model_id,
        tp=tp,
        max_model_len=max_model_len,
        input_len=seed["input_len"],
        output_len=seed["output_len"],
        batch_size=seed["batch_size"],
        num_iters=num_iters,
        output_json=seed_out,
        extra_args=extra_args,
    )
    args = parser.parse_args(seed_argv)

    if getattr(args, "profile", False) and not nsys_profile:
        # vllm bench latency --profile is a single-run action; sweeping doesn't make sense.
        # (When nsys_profile is active, we handle profiler start/stop per-bucket ourselves.)
        _write_text(out_root / f"child_{label}_error.log", "Refusing to sweep with --profile enabled.\n")
        _update_status("error", extra={"error": "Refusing to sweep with --profile enabled."})
        return 2

    engine_args = EngineArgs.from_cli_args(args)
    _update_status("loading_model")
    # Work around pydantic validation: _dataclasses.asdict may produce None
    # values inside nested config dicts (e.g. compilation_config.cudagraph_capture_sizes)
    # which CompilationConfig rejects. Filter them out.
    ea_dict = _dataclasses.asdict(engine_args)
    for _cfg_key in ("compilation_config", "profiler_config", "attention_config",
                      "structured_outputs_config"):
        if isinstance(ea_dict.get(_cfg_key), dict):
            ea_dict[_cfg_key] = {k: v for k, v in ea_dict[_cfg_key].items() if v is not None}
    llm = LLM(**ea_dict)
    _update_status("model_loaded")

    # ---- Phase 1: Correctness (GSM8K greedy decode) ----
    if capture_golden_refs or verify_correctness:
        import torch
        from vllm import SamplingParams as _CorrectnessSP
        _update_status("correctness_phase_start")
        json_dir = out_root / "json"
        json_dir.mkdir(parents=True, exist_ok=True)
        try:
            prompts, gsm8k_labels = _build_gsm8k_prompts(num_questions=correctness_num_questions)
            correctness_sp = _CorrectnessSP(
                temperature=0.0, max_tokens=1024,
                stop=["Question", "Assistant:", "<|separator|>"],
                seed=42, logprobs=5,
            )
            print(f"[correctness] Running GSM8K greedy decode: {len(prompts)} questions")
            t0 = time.time()
            outputs = llm.generate(prompts, sampling_params=correctness_sp, use_tqdm=False)
            duration = time.time() - t0
            print(f"[correctness] Generation done in {duration:.1f}s")

            serialized = _serialize_correctness_outputs(outputs)
            preds, accuracy = _score_gsm8k_predictions(outputs, gsm8k_labels)
            gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unknown"

            if capture_golden_refs:
                # Self-consistency check: run prompts a second time
                print("[correctness] Running self-consistency check...")
                outputs2 = llm.generate(prompts, sampling_params=correctness_sp, use_tqdm=False)
                serialized2 = _serialize_correctness_outputs(outputs2)
                deterministic = all(
                    s1["token_ids"] == s2["token_ids"]
                    for s1, s2 in zip(serialized, serialized2)
                )
                if not deterministic:
                    print("[correctness] WARNING: Self-consistency check FAILED — environment is non-deterministic.")
                else:
                    print("[correctness] Self-consistency check PASSED — greedy decode is deterministic.")

                golden_data = {
                    "metadata": {
                        "num_questions": len(prompts), "num_shots": 5, "max_tokens": 1024,
                        "seed": 42, "logprobs_k": 5, "gsm8k_accuracy": round(accuracy, 4),
                        "capture_duration_s": round(duration, 2), "gpu_name": gpu_name,
                        "deterministic": deterministic,
                        "baseline_preds": preds, "labels": gsm8k_labels,
                    },
                    "outputs": serialized,
                }
                golden_path = json_dir / "golden_refs.json"
                _write_json(golden_path, golden_data)
                print(f"[correctness] Golden refs saved to {golden_path}")
                _update_status("correctness_done", extra={"accuracy": accuracy, "deterministic": deterministic})

            if verify_correctness:
                golden_path = json_dir / "golden_refs.json"
                if not golden_path.exists():
                    print(f"[correctness] ERROR: golden_refs.json not found at {golden_path}")
                    _update_status("correctness_error", extra={"error": "golden_refs.json not found"})
                    return 4
                golden_data = json.loads(golden_path.read_text(encoding="utf-8"))
                golden_refs = golden_data["outputs"]
                golden_meta = golden_data.get("metadata", {})
                baseline_preds = golden_meta.get("baseline_preds")
                golden_labels = golden_meta.get("labels")

                # Metadata mismatch detection (exit code 4)
                golden_nq = golden_meta.get("num_questions")
                if golden_nq is not None and golden_nq != correctness_num_questions:
                    msg = (f"[correctness] ERROR: golden refs num_questions={golden_nq} "
                           f"!= current num_questions={correctness_num_questions}. "
                           "Re-capture golden refs with matching --correctness-num-questions.")
                    print(msg)
                    _update_status("correctness_error", extra={"error": msg})
                    return 4
                golden_mt = golden_meta.get("max_tokens")
                if golden_mt is not None and golden_mt != 1024:
                    msg = (f"[correctness] ERROR: golden refs max_tokens={golden_mt} "
                           f"!= current max_tokens=1024. "
                           "Re-capture golden refs with v2 settings.")
                    print(msg)
                    _update_status("correctness_error", extra={"error": msg})
                    return 4

                # GPU name mismatch warning
                golden_gpu = golden_meta.get("gpu_name", "")
                if golden_gpu and golden_gpu != gpu_name:
                    print(f"[correctness] WARNING: GPU mismatch — golden refs captured on '{golden_gpu}', current GPU is '{gpu_name}'")

                # Save opt outputs
                opt_path = json_dir / "opt_outputs.json"
                _write_json(opt_path, {"outputs": serialized})

                # Run comparator
                verdict = _compare_correctness(
                    golden_refs=golden_refs, opt_outputs=serialized,
                    labels=golden_labels, baseline_preds=baseline_preds, opt_preds=preds,
                )
                verdict["duration_s"] = round(duration, 2)

                verdict_path = json_dir / "correctness_verdict.json"
                _write_json(verdict_path, verdict)
                print(f"[correctness] Verdict: {verdict['verdict']}")
                print(f"[correctness] Written to {verdict_path}")

                # Infrastructure error (baseline_accuracy=0) → exit code 4
                if verdict.get("infrastructure_error"):
                    print(f"[correctness] {verdict.get('infrastructure_message', 'Infrastructure error')}")
                    _update_status("correctness_error", extra=verdict)
                    return 4

                if verdict["verdict"] != "PASS":
                    _update_status("correctness_failed", extra=verdict)
                    return 3  # Correctness FAIL — don't proceed to latency
                _update_status("correctness_done", extra=verdict)

        except Exception as e:
            print(f"[correctness] ERROR: {type(e).__name__}: {e}")
            traceback.print_exc()
            _update_status("correctness_error", extra={"error": str(e)})
            return 4  # Infrastructure error
    # ---- End Phase 1 ----

    for bucket in buckets:
        b_input_len = bucket["input_len"]
        b_output_len = bucket["output_len"]
        bs = bucket["batch_size"]
        tag = _bucket_file_tag(bucket, buckets)

        # Validate per-bucket model len constraint.
        if llm.llm_engine.model_config.max_model_len < (b_input_len + b_output_len):
            _write_text(
                out_root / f"child_{label}_error.log",
                f"max_model_len is smaller than input_len + output_len for bucket {tag}; adjust target.json.\n",
            )
            _update_status("error", extra={"error": f"max_model_len < input_len + output_len for bucket {tag}"})
            return 2

        # Reconstruct SamplingParams per-bucket (output_len may vary).
        sampling_params = SamplingParams(
            n=int(getattr(args, "n", 1)),
            temperature=1.0,
            top_p=1.0,
            ignore_eos=True,
            max_tokens=b_output_len,
            detokenize=not bool(getattr(args, "disable_detokenize", False)),
        )

        raw_json = json_dir / f"{label}_{tag}.json"
        runner_json = json_dir / f"{label}_{tag}.runner.json"
        log_path = logs_dir / f"{label}_{tag}.log"

        # Override per-bucket args that should affect only prompt construction / output paths.
        setattr(args, "batch_size", int(bs))
        setattr(args, "input_len", int(b_input_len))
        setattr(args, "output_len", int(b_output_len))
        setattr(args, "num_iters", int(num_iters))
        setattr(args, "output_json", str(raw_json))

        start = datetime.now(timezone.utc)
        status: Dict[str, Any] = {
            "ok": False,
            "label": label,
            "batch_size": bs,
            "input_len": b_input_len,
            "output_len": b_output_len,
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
                _log(f"=== inproc vllm bench latency sweep ({label}) {tag} ===")
                _log(
                    _format_cmd_for_md(
                        ["vllm", "bench", "latency"]
                        + _build_cli_equivalent_args_for_inproc(
                            model_id=model_id,
                            tp=tp,
                            max_model_len=max_model_len,
                            input_len=b_input_len,
                            output_len=b_output_len,
                            batch_size=bs,
                            num_iters=num_iters,
                            output_json=raw_json,
                            extra_args=extra_args,
                        ),
                        {},
                    )
                )

                dummy_prompt_token_ids = np.random.randint(10000, size=(bs, b_input_len))
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
                                max_tokens=b_output_len,
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

                # Start nsys capture for this bucket (if enabled).
                # NOTE: We call cudaProfilerStart/Stop directly instead of
                # llm.start_profile() because the latter injects --profile
                # semantics (single-iteration capture) and propagates via
                # vLLM's executor RPC.  Direct cudaProfilerStart() works with
                # nsys because nsys's --capture-range=cudaProfilerApi captures
                # ALL traced processes (including TP workers followed via
                # --trace-fork-before-exec) when ANY process triggers the
                # capture range — this is an nsys-level mechanism, not CUDA
                # profiler propagation.
                _torch_prof = None
                if nsys_profile:
                    import torch as _torch_prof
                    _torch_prof.cuda.synchronize()
                    _torch_prof.cuda.cudart().cudaProfilerStart()
                    _log(f"[nsys] cudaProfilerStart for {tag}")

                try:
                    for i in range(num_iters_eff):
                        if _time.monotonic() > deadline:
                            raise TimeoutError(f"Bucket timeout exceeded ({timeout_s_per_bucket}s)")
                        latencies.append(run_to_completion())
                        if (_time.monotonic() - last_status_t) > 5.0:
                            _update_status("benchmark", batch_size=bs, extra={"iter": i + 1})
                            last_status_t = _time.monotonic()
                finally:
                    # Stop nsys capture for this bucket. Must happen even on
                    # exception to avoid corrupting the repeat:N capture count
                    # and leaving nsys waiting indefinitely.
                    if nsys_profile and _torch_prof is not None:
                        _torch_prof.cuda.synchronize()
                        _torch_prof.cuda.cudart().cudaProfilerStop()
                        _log(f"[nsys] cudaProfilerStop for {tag}")

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
    # Determine if rows are heterogeneous (mixed IL/OL values).
    il_ol_set: set = set()
    for r in rows:
        il = r.get("input_len")
        ol = r.get("output_len")
        if il is not None and ol is not None:
            il_ol_set.add((il, ol))
    heterogeneous = len(il_ol_set) > 1

    if heterogeneous:
        header = f"| Input Len | Output Len | Batch Size | {baseline_label} avg (s) | {opt_label} avg (s) | Speedup | Improvement | Fast-path evidence |"
        sep = "|---:|---:|---:|---:|---:|---:|---:|---|"
    else:
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

        if heterogeneous:
            il = r.get("input_len", "")
            ol = r.get("output_len", "")
            lines.append(
                f"| {il} | {ol} | {bs} | {fmt(b_avg)} | {fmt(o_avg)} | {fmt(speedup)}x | {fmt(improve)}% | {evidence} |"
            )
        else:
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
    p.add_argument("--heartbeat-s", type=int, default=30, help="Heartbeat interval while a subprocess is quiet")
    p.add_argument(
        "--labels",
        type=str,
        default="baseline",
        help=(
            "Comma-separated labels to run: 'baseline' (default), 'opt', or 'baseline,opt'. "
            "Use '--labels baseline' for Stage 1 baseline-only sweeps (no opt). "
            "Use '--labels opt' for Stage 5 validation (opt only, compare against Stage 1 baseline)."
        ),
    )
    p.add_argument(
        "--nsys-profile",
        action="store_true",
        default=False,
        help=(
            "Capture per-bucket nsys profiles during inproc_sweep. "
            "Produces one .nsys-rep per bucket in {out}/nsys/. "
            "Uses --capture-range=cudaProfilerApi with repeat mode."
        ),
    )
    p.add_argument(
        "--nsys-extra-flags",
        type=str,
        default="",
        help="Extra flags to pass to nsys profile (e.g. '--stats=true')",
    )
    p.add_argument(
        "--nsys-timeout-s",
        type=int,
        default=600,
        help=(
            "Per-bucket timeout in seconds when --nsys-profile is active (default: 600). "
            "Total timeout = nsys_timeout_s * num_buckets."
        ),
    )
    p.add_argument(
        "--nsys-output-len",
        type=int,
        default=None,
        help=(
            "Override output_len for nsys profiling only (default: use workload output_len). "
            "Decouples profiling sequence length from benchmark sequence length to avoid "
            "superlinear nsys overhead on models with many kernels/step. "
            "Requires --nsys-profile."
        ),
    )
    p.add_argument(
        "--nsys-num-iters",
        type=int,
        default=None,
        help=(
            "Override num_iters for nsys profiling only. Defaults to 1 when "
            "--nsys-output-len is set. Requires --nsys-profile."
        ),
    )
    p.add_argument(
        "--baseline-from",
        type=str,
        default=None,
        help=(
            "Path to a Stage 1 baseline output directory (e.g., {artifact_dir}/e2e_baseline/) "
            "containing json/{baseline_label}_{tag}.json files.  Used with '--labels opt' to "
            "import existing baseline data so gate results include speedup calculations. "
            "When omitted with '--labels opt', results are flagged with a warning."
        ),
    )
    p.add_argument("--capture-golden-refs", action="store_true", default=False,
                   help="Stage 1: run GSM8K greedy decode and save golden refs to json/golden_refs.json")
    p.add_argument("--verify-correctness", action="store_true", default=False,
                   help="Stage 5: compare GSM8K outputs against golden refs; exit nonzero on mismatch")
    p.add_argument("--correctness-num-questions", type=int, default=200,
                   help="Number of GSM8K questions for correctness phase (default: 200)")
    p.add_argument("--_child-label", type=str, default=None, help=argparse.SUPPRESS)
    p.add_argument("--_out-root", type=str, default=None, help=argparse.SUPPRESS)
    p.add_argument("--_nsys-profile", action="store_true", default=False, help=argparse.SUPPRESS)
    p.add_argument("--_cudagraph-capture-sizes", nargs="+", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--_nsys-output-len", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--_nsys-num-iters", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--_capture-golden-refs", action="store_true", default=False, help=argparse.SUPPRESS)
    p.add_argument("--_verify-correctness", action="store_true", default=False, help=argparse.SUPPRESS)
    p.add_argument("--_correctness-num-questions", type=int, default=200, help=argparse.SUPPRESS)

    args = p.parse_args()

    # Transition safety: warn if old locking system files exist
    old_lock_dir = Path('/tmp/ammo_gpu_locks')
    if old_lock_dir.exists() and any(old_lock_dir.glob('*.lock')):
        print(
            'WARNING: Old GPU lock files found at /tmp/ammo_gpu_locks/. '
            'GPU reservation is now managed by hooks. Old locks can be safely deleted.',
            file=sys.stderr,
        )

    # Validate --nsys-profile constraints early.
    if args.nsys_profile:
        if not shutil.which("nsys"):
            raise SystemExit(
                "nsys not found on PATH. Install Nsight Systems CLI or remove --nsys-profile."
            )
    if (args.nsys_output_len is not None or args.nsys_num_iters is not None) and not args.nsys_profile:
        raise SystemExit(
            "--nsys-output-len and --nsys-num-iters require --nsys-profile."
        )
    if args.nsys_output_len is not None and args.nsys_output_len <= 0:
        raise SystemExit(
            f"--nsys-output-len must be positive, got {args.nsys_output_len}"
        )
    if args.nsys_num_iters is not None and args.nsys_num_iters <= 0:
        raise SystemExit(
            f"--nsys-num-iters must be positive, got {args.nsys_num_iters}"
        )

    # Validate correctness flag constraints.
    if args.capture_golden_refs and args.verify_correctness:
        raise SystemExit("--capture-golden-refs and --verify-correctness are mutually exclusive")
    if args.verify_correctness and not args.baseline_from:
        raise SystemExit("--verify-correctness requires --baseline-from (to import golden_refs.json)")

    # Parse and validate --labels.
    selected_labels = {s.strip() for s in args.labels.split(",")}
    invalid_labels = selected_labels - {"baseline", "opt"}
    if invalid_labels:
        raise SystemExit(
            f"Invalid --labels value(s): {invalid_labels}. "
            "Accepted: 'baseline', 'opt', or 'baseline,opt'."
        )
    if not selected_labels:
        raise SystemExit("--labels must specify at least one label.")

    # Validate --baseline-from early.
    baseline_from: Optional[Path] = None
    if args.baseline_from:
        baseline_from = Path(args.baseline_from).expanduser().resolve()
        if not baseline_from.is_dir():
            raise SystemExit(
                f"--baseline-from path does not exist or is not a directory: {baseline_from}"
            )

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

    # Expand workload into buckets (supports both legacy flat and new matrix format).
    # Legacy flat fields are still required when workload_matrix is absent.
    has_matrix = "workload_matrix" in workload
    if not has_matrix:
        # Validate legacy required fields.
        _require_int(workload, "input_len", "workload")
        _require_int(workload, "output_len", "workload")
        _require_list_int(workload, "batch_sizes", "workload")

    buckets = _expand_workload_to_buckets(workload)
    num_iters = _require_int(workload, "num_iters", "workload")

    # Legacy convenience vars (for backward-compat in console output and workload JSON).
    input_len = workload.get("input_len")
    output_len = workload.get("output_len")
    batch_sizes = workload.get("batch_sizes")

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

    # Validate env dict keys and values are not placeholders.
    for env_name, env_dict in [("baseline_env", baseline_env), ("opt_env", opt_env)]:
        for k, v in env_dict.items():
            if _is_placeholder(k):
                raise SystemExit(
                    f"bench.{env_name} key is still a placeholder: {k!r}. "
                    "Replace it with the actual environment variable name (e.g., VLLM_MY_OPT=1)."
                )
            if _is_placeholder(v):
                raise SystemExit(f"bench.{env_name}[{k!r}] value is still a placeholder: {v!r}")

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

    # Prepare envs: start from current env, then strip stale VLLM_OP* vars.
    # Cross-track contamination fix: prior-round gating flags (VLLM_OP001, etc.)
    # may be set in the inherited environment. Stripping them ensures each sweep
    # measures only the target track's optimization.  The target track's env var
    # is re-added explicitly via baseline_env / opt_env from target.json.
    base_env = _sanitize_vllm_op_env(dict(os.environ))

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

    # Config identity check: warn or fail when baseline and opt are identical.
    if "baseline" in selected_labels and "opt" in selected_labels:
        if not opt_env and opt_extra_args == baseline_extra_args:
            raise SystemExit(
                "baseline and opt configs are identical — opt would produce the "
                "same results as baseline.\n"
                "Update opt_env in target.json, or use --labels baseline to skip opt."
            )

    # Validate all buckets fit within max_model_len at plan time.
    _validate_buckets_model_len(buckets, max_model_len)
    if args.nsys_output_len is not None:
        _validate_buckets_model_len(
            [{**b, "output_len": args.nsys_output_len} for b in buckets],
            max_model_len,
        )

    print("=== Target ===")
    print(f"artifact_dir: {artifact_dir}")
    print(f"out_dir: {out_root}")
    print(f"model_id: {model_id}")
    print(f"tp: {tp}, ep: {ep}, max_model_len: {max_model_len}")
    if has_matrix:
        print(f"workload: {len(buckets)} buckets (workload_matrix), num_iters={num_iters}")
        for i, b in enumerate(buckets):
            print(f"  [{i}] input_len={b['input_len']}, output_len={b['output_len']}, batch_size={b['batch_size']}")
    else:
        print(f"workload: input_len={input_len}, output_len={output_len}, num_iters={num_iters}")
        print(f"batch_sizes: {batch_sizes}")
    print(f"baseline_label: {baseline_label}, opt_label: {opt_label}")
    if args.nsys_profile:
        nsys_ol_eff = args.nsys_output_len if args.nsys_output_len is not None else "(workload)"
        nsys_ni_eff = args.nsys_num_iters
        if nsys_ni_eff is None and args.nsys_output_len is not None:
            nsys_ni_eff = "1 (auto)"
        elif nsys_ni_eff is None:
            nsys_ni_eff = "(workload)"
        print(f"nsys_profile: enabled, nsys_output_len={nsys_ol_eff}, nsys_num_iters={nsys_ni_eff}")
        print(
            "WARNING: nsys profiling adds overhead — latency results from this run "
            "should not be used for performance comparison.",
            file=sys.stderr,
        )

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
        "execution_mode": "inproc_sweep",
        "model_id": model_id,
        "tp": tp,
        "ep": ep,
        "max_model_len": max_model_len,
        "workload": {
            "input_len": input_len,
            "output_len": output_len,
            "num_iters": num_iters,
            "batch_sizes": batch_sizes,
            "buckets": buckets,
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

    # Track baseline source and warnings for gate runs (--labels opt).
    run_warnings: List[str] = []
    if "opt" in selected_labels and "baseline" not in selected_labels:
        if baseline_from:
            all_runs["baseline_source"] = str(baseline_from)
        else:
            all_runs["baseline_source"] = "none"
            run_warnings.append(
                "No baseline reference provided (--baseline-from not set); "
                "speedup calculations unavailable."
            )
            print(
                "WARNING: --labels opt without --baseline-from; baseline data "
                "will be null in results. Pass --baseline-from <stage1_dir> for "
                "gate results with speedup calculations.",
                file=sys.stderr,
            )

    # GPU reservation is now managed by PreToolUse/PostToolUse hooks.
    # The hooks auto-reserve when CUDA_VISIBLE_DEVICES=X is in the command
    # and auto-release when the command completes. No in-script locking needed.

    is_child = bool(args._child_label)

    # Record the per-bucket plan deterministically (for stable paths + repro commands).
    planned: List[Dict[str, Any]] = []
    for bucket in buckets:
        b_input_len = bucket["input_len"]
        b_output_len = bucket["output_len"]
        bs = bucket["batch_size"]
        tag = _bucket_file_tag(bucket, buckets)

        baseline_json_p = json_dir / f"{baseline_label}_{tag}.json"
        opt_json_p = json_dir / f"{opt_label}_{tag}.json"
        baseline_log_p = logs_dir / f"{baseline_label}_{tag}.log"
        opt_log_p = logs_dir / f"{opt_label}_{tag}.log"

        baseline_cmd = _build_vllm_bench_cmd(
            vllm_exe=vllm_exe,
            model_id=model_id,
            tp=tp,
            max_model_len=max_model_len,
            input_len=b_input_len,
            output_len=b_output_len,
            batch_size=bs,
            num_iters=num_iters,
            output_json=baseline_json_p,
            extra_args=baseline_args,
        )
        opt_cmd = _build_vllm_bench_cmd(
            vllm_exe=vllm_exe,
            model_id=model_id,
            tp=tp,
            max_model_len=max_model_len,
            input_len=b_input_len,
            output_len=b_output_len,
            batch_size=bs,
            num_iters=num_iters,
            output_json=opt_json_p,
            extra_args=opt_args,
        )

        print(f"\n=== {tag} (il={b_input_len}, ol={b_output_len}, bs={bs}) ===")
        print(f"Baseline cmd: {_format_cmd_for_md(baseline_cmd, baseline_env)}")
        print(f"Opt cmd: {_format_cmd_for_md(opt_cmd, {**baseline_env, **opt_env})}")

        planned.append({
            "batch_size": bs,
            "input_len": b_input_len,
            "output_len": b_output_len,
            "tag": tag,
            "baseline_cmd": baseline_cmd,
            "opt_cmd": opt_cmd,
            "baseline_log": baseline_log_p,
            "opt_log": opt_log_p,
            "baseline_json": baseline_json_p,
            "opt_json": opt_json_p,
        })

        # Populate a minimal row deterministically (stable paths + repro commands).
        all_runs["results"].append({
            "batch_size": bs,
            "input_len": b_input_len,
            "output_len": b_output_len,
            baseline_label: {
                "cmd": baseline_cmd,
                "env_overrides": baseline_env,
                "log": str(baseline_log_p.relative_to(out_root)),
                "output_json": str(baseline_json_p.relative_to(out_root)),
                "fastpath_evidence": {"status": "unknown"},
            },
            opt_label: {
                "cmd": opt_cmd,
                "env_overrides": {**baseline_env, **opt_env},
                "log": str(opt_log_p.relative_to(out_root)),
                "output_json": str(opt_json_p.relative_to(out_root)),
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
    for label in [l for l in (baseline_label, opt_label)
                  if ("baseline" if l == baseline_label else "opt") in selected_labels]:
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
        # Inject --cudagraph-capture-sizes if the parent passed it (nsys mode).
        if args._cudagraph_capture_sizes:
            label_args = (
                label_args
                + ["--cudagraph-capture-sizes"]
                + [str(x) for x in args._cudagraph_capture_sizes]
            )
        # Apply nsys overrides when present (decouples profiling OL from benchmark OL).
        child_buckets = buckets
        child_num_iters = num_iters
        nsys_ol = getattr(args, "_nsys_output_len", None)
        nsys_ni = getattr(args, "_nsys_num_iters", None)
        if nsys_ol is not None:
            child_buckets = [{**b, "output_len": nsys_ol} for b in buckets]
        if nsys_ni is not None:
            child_num_iters = nsys_ni
        returncode = _run_inproc_latency_sweep_child(
            label=label,
            model_id=model_id,
            tp=tp,
            max_model_len=max_model_len,
            buckets=child_buckets,
            num_iters=child_num_iters,
            extra_args=label_args,
            out_root=out_root,
            timeout_s_per_bucket=args.timeout_s,
            nsys_profile=getattr(args, "_nsys_profile", False),
            capture_golden_refs=getattr(args, "_capture_golden_refs", False),
            verify_correctness=getattr(args, "_verify_correctness", False),
            correctness_num_questions=getattr(args, "_correctness_num_questions", 200),
        )
        raise SystemExit(returncode)

    # Summaries
    def _status_from_evidence(ev: Dict[str, Any], patterns_configured: bool) -> str:
        if not patterns_configured:
            return "unknown"
        return "pass" if ev.get("ok") else "fail"

    baseline_patterns_configured = bool(baseline_run.require_patterns or baseline_run.forbid_patterns)
    opt_patterns_configured = bool(opt_run.require_patterns or opt_run.forbid_patterns)

    script_path = Path(__file__).resolve()
    child_timeout = int(args.timeout_s) * max(1, len(buckets)) + 1800
    if args.nsys_profile:
        child_timeout = args.nsys_timeout_s * max(1, len(buckets))

    # Set up nsys output directory if profiling.
    nsys_dir: Optional[Path] = None
    nsys_capture_sizes: Optional[List[int]] = None
    if args.nsys_profile:
        nsys_dir = out_root / "nsys"
        nsys_dir.mkdir(parents=True, exist_ok=True)
        # Restrict CUDA graph capture to only the batch sizes being profiled.
        # Default vLLM captures ~50 sizes (~2,142 CUDAGraph objects); restricting
        # to workload batch sizes reduces this to ~len(buckets), mitigating the
        # memory pressure that causes nsys --cuda-graph-trace=node replay hangs.
        nsys_capture_sizes = sorted(set(b["batch_size"] for b in buckets))
        print(
            f"nsys mode: restricting cudagraph_capture_sizes to {nsys_capture_sizes} "
            f"(from workload batch_sizes)"
        )

    runs_to_execute = []
    if "baseline" in selected_labels:
        runs_to_execute.append(baseline_run)
    if "opt" in selected_labels:
        runs_to_execute.append(opt_run)

    # Copy golden refs BEFORE spawning children — children need it during Phase 1.
    if baseline_from and args.verify_correctness:
        golden_src = baseline_from / "json" / "golden_refs.json"
        golden_dst = json_dir / "golden_refs.json"
        if golden_src.exists():
            shutil.copy2(str(golden_src), str(golden_dst))
            print(f"Imported golden references from {golden_src}")
        else:
            raise SystemExit(
                f"--verify-correctness requires golden_refs.json but "
                f"--baseline-from has none at {golden_src}"
            )

    for run in runs_to_execute:
        child_cmd = [
            sys.executable,
            str(script_path),
            "--artifact-dir",
            str(artifact_dir),
            "--target-json",
            str(target_path),
            "--timeout-s",
            str(args.timeout_s),
            "--out-name",
            args.out_name,
            "--_out-root",
            str(out_root),
            "--_child-label",
            run.label,
        ]
        if args.nsys_profile:
            child_cmd.append("--_nsys-profile")
            if nsys_capture_sizes:
                child_cmd.extend(
                    ["--_cudagraph-capture-sizes"] + [str(bs) for bs in nsys_capture_sizes]
                )
            # Pass nsys overrides to child (decouples profiling OL from benchmark OL).
            nsys_ol = args.nsys_output_len
            nsys_ni = args.nsys_num_iters
            if nsys_ni is None and nsys_ol is not None:
                nsys_ni = 1  # Default to 1 iter when profiling OL is decoupled.
            if nsys_ol is not None:
                child_cmd.extend(["--_nsys-output-len", str(nsys_ol)])
            if nsys_ni is not None:
                child_cmd.extend(["--_nsys-num-iters", str(nsys_ni)])

        # Forward correctness flags to child.
        if args.capture_golden_refs:
            child_cmd.append("--_capture-golden-refs")
        if args.verify_correctness:
            child_cmd.append("--_verify-correctness")
        if args.capture_golden_refs or args.verify_correctness:
            child_cmd.extend(["--_correctness-num-questions", str(args.correctness_num_questions)])

        # Prepend nsys wrapper when profiling.
        child_env = dict(run.env) if run.env else dict(os.environ)
        child_env = _ensure_worktree_pythonpath(child_env)
        if args.nsys_profile:
            assert nsys_dir is not None
            child_env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
            nsys_prefix = [
                "nsys", "profile",
                "--trace=cuda,nvtx",
                "--sample=none",
                "--capture-range=cudaProfilerApi",
                f"--capture-range-end=repeat:{len(buckets)}",
                "--cuda-graph-trace=node",
                "--trace-fork-before-exec=true",
                "--force-overwrite=true",
                "-o", str(nsys_dir / f"{run.label}_profile"),
            ]
            if args.nsys_extra_flags:
                import shlex as _shlex
                nsys_prefix.extend(_shlex.split(args.nsys_extra_flags))
            child_cmd = nsys_prefix + child_cmd

        print(f"\n=== Running inproc sweep for {run.label} ===")
        print(f"Child cmd: {_format_cmd_for_md(child_cmd, {})}")
        child_log = logs_dir / f"{run.label}_supervisor.log"
        child_status = status_dir / f"{run.label}.json"
        child_res = _run_cmd_streaming(
            child_cmd,
            env=child_env,
            cwd=None,
            timeout_s=child_timeout,
            log_path=child_log,
            heartbeat_s=int(args.heartbeat_s),
            status_path=child_status,
        )
        if not child_res.get("ok"):
            hint = ""
            if args.nsys_profile and child_res.get("returncode") in (-9, 137, None):
                hint = (
                    " This is likely an nsys --cuda-graph-trace=node replay hang. "
                    "Try adding --nsys-output-len 32 to decouple profiling OL, "
                    "reducing workload batch_sizes, or increasing --nsys-timeout-s "
                    f"(current: {args.nsys_timeout_s}s per bucket)."
                )
            raise SystemExit(
                f"Inproc sweep child failed for {run.label}: returncode={child_res.get('returncode')}. "
                f"See {child_log} and {logs_dir / f'{run.label}_child.log'}"
                f"{hint}"
            )

        # Rename nsys output files to match bucket tags.
        if nsys_dir is not None:
            # Use nsys-overridden buckets for tags if OL was overridden.
            nsys_tag_buckets = buckets
            if args.nsys_output_len is not None:
                nsys_tag_buckets = [{**b, "output_len": args.nsys_output_len} for b in buckets]
                # Detect tag collisions: heterogeneous workload_matrix buckets
                # that differ only in output_len would collide after OL override.
                tags = [_bucket_file_tag(b, nsys_tag_buckets) for b in nsys_tag_buckets]
                if len(tags) != len(set(tags)):
                    dupes = [t for t in tags if tags.count(t) > 1]
                    print(
                        f"WARNING: --nsys-output-len creates duplicate file tags {set(dupes)}. "
                        f"Nsys profiles for colliding buckets will be overwritten. "
                        f"Consider using distinct batch_sizes or input_lens.",
                        file=sys.stderr,
                    )
            renamed = 0
            for i, bucket in enumerate(nsys_tag_buckets, 1):
                src = nsys_dir / f"{run.label}_profile.{i}.nsys-rep"
                tag = _bucket_file_tag(bucket, nsys_tag_buckets)
                dst = nsys_dir / f"{run.label}_{tag}.nsys-rep"
                if src.exists():
                    src.rename(dst)
                    print(f"  nsys: {src.name} -> {dst.name}")
                    renamed += 1
            if renamed < len(buckets):
                print(
                    f"  WARNING: nsys produced {renamed} of {len(buckets)} "
                    f"expected profile files for {run.label}"
                )

    # Import Stage 1 baseline artifacts when --baseline-from is set.
    # This copies baseline JSON + runner JSON into the gate run's json/ dir
    # so the results collection below picks them up transparently.
    if baseline_from and "opt" in selected_labels and "baseline" not in selected_labels:
        baseline_from_json = baseline_from / "json"
        if not baseline_from_json.is_dir():
            raise SystemExit(
                f"--baseline-from has no json/ subdirectory: {baseline_from}. "
                "Expected a Stage 1 output directory with json/{baseline_label}_*.json files."
            )
        imported = 0
        missing_tags: List[str] = []
        for p in planned:
            tag = p["tag"]
            for suffix in (".json", ".runner.json"):
                src = baseline_from_json / f"{baseline_label}_{tag}{suffix}"
                dst = json_dir / f"{baseline_label}_{tag}{suffix}"
                if src.exists():
                    import shutil as _shutil_copy
                    _shutil_copy.copy2(str(src), str(dst))
                    if suffix == ".json":
                        imported += 1
                else:
                    if suffix == ".json":
                        missing_tags.append(tag)
        if missing_tags:
            raise SystemExit(
                f"--baseline-from is missing baseline data for {len(missing_tags)} "
                f"bucket(s): {missing_tags}. "
                f"Looked in {baseline_from_json} for {baseline_label}_<tag>.json files."
            )
        print(f"Imported {imported} baseline artifact(s) from {baseline_from}")

    # Populate per-bucket entries from artifacts written by children.
    new_rows: List[Dict[str, Any]] = []
    for p in planned:
        bs = int(p["batch_size"])
        b_il = int(p["input_len"])
        b_ol = int(p["output_len"])
        tag = p["tag"]
        baseline_log = Path(p["baseline_log"])
        opt_log = Path(p["opt_log"])
        baseline_json = Path(p["baseline_json"])
        opt_json = Path(p["opt_json"])

        baseline_runner_json = json_dir / f"{baseline_label}_{tag}.runner.json"
        opt_runner_json = json_dir / f"{opt_label}_{tag}.runner.json"

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

        new_row: Dict[str, Any] = {
            "batch_size": bs,
            "input_len": b_il,
            "output_len": b_ol,
            baseline_label: baseline_entry,
            opt_label: opt_entry,
        }

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
                f"Fast-path evidence FAILED for opt at {tag}. "
                f"Missing={opt_evidence.get('require_miss')}, forbidden_hits={opt_evidence.get('forbid_hits')}. "
                f"See {opt_log}"
            )

    all_runs["results"] = new_rows

    # Append warnings accumulated during the run.
    if run_warnings:
        all_runs["warnings"] = run_warnings

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
    if has_matrix:
        md_lines.append(f"- buckets: {len(buckets)} (workload_matrix)")
        for i, b in enumerate(buckets):
            md_lines.append(f"  - [{i}] input_len={b['input_len']}, output_len={b['output_len']}, batch_size={b['batch_size']}")
    else:
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


if __name__ == "__main__":
    main()
