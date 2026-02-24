#!/usr/bin/env python3
"""Generate `{artifact_dir}/validation_results.md` from recorded evidence.

This is deliberately conservative: it only reports what it can *prove* from
existing files. It will not invent kernel timings or correctness tolerances.

Inputs (all optional, but the report is more useful with them):
- {artifact_dir}/env.json (from scripts/collect_env.py)
- {artifact_dir}/e2e_latency/e2e_latency_results.json (from scripts/run_vllm_bench_latency_sweep.py)

Outputs:
- {artifact_dir}/validation_results.md (overwritten)
- {artifact_dir}/validation_summary.json (machine-readable)

Guardrails:
- If evidence is missing, the report includes TODO blocks rather than guessing.

Example:
  python scripts/generate_validation_report.py --artifact-dir artifacts/qwen3_l40s_fp8_tp1
"""

from __future__ import annotations

import argparse
import json
import shlex
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except json.JSONDecodeError as e:
        raise SystemExit(f"Failed to parse JSON {path}: {e}")


def _fmt_cmd(cmd: List[str], env_overrides: Dict[str, str]) -> str:
    env_prefix = " ".join([f"{k}={shlex.quote(v)}" for k, v in env_overrides.items()])
    cmd_str = " ".join([shlex.quote(x) for x in cmd])
    return (env_prefix + " " + cmd_str).strip()


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _render_e2e_section(e2e: Dict[str, Any]) -> str:
    baseline_label = e2e.get("bench", {}).get("baseline_label", "baseline")
    opt_label = e2e.get("bench", {}).get("opt_label", "opt")

    lines: List[str] = []
    lines.append("## E2E latency (vllm bench latency)")
    lines.append("")

    wl = e2e.get("workload", {})
    lines.append("Workload:")
    lines.append(f"- model_id: {e2e.get('model_id')}")
    lines.append(f"- input_len: {wl.get('input_len')}, output_len: {wl.get('output_len')}")
    lines.append(f"- tp: {e2e.get('tp')}, max_model_len: {e2e.get('max_model_len')}")
    lines.append(f"- num_iters: {wl.get('num_iters')}")
    lines.append("")

    # Table
    header = f"| Batch Size | {baseline_label} avg (s) | {opt_label} avg (s) | Speedup | Improvement | Fast-path evidence |"
    sep = "|---:|---:|---:|---:|---:|---|"
    lines.append(header)
    lines.append(sep)

    results = e2e.get("results", [])
    if not isinstance(results, list):
        results = []

    for row in results:
        if not isinstance(row, dict):
            continue
        bs = row.get("batch_size")
        b = row.get(baseline_label, {}) if isinstance(row.get(baseline_label), dict) else {}
        o = row.get(opt_label, {}) if isinstance(row.get(opt_label), dict) else {}

        b_avg = b.get("avg_s")
        o_avg = o.get("avg_s")
        speedup = row.get("speedup")
        improve = row.get("improvement_pct")
        evidence = o.get("fastpath_evidence", {}).get("status", "unknown")

        def fmt(x: Any) -> str:
            if x is None:
                return ""
            if isinstance(x, (int, float)):
                return f"{x:.6g}"
            return str(x)

        lines.append(f"| {bs} | {fmt(b_avg)} | {fmt(o_avg)} | {fmt(speedup)}x | {fmt(improve)}% | {evidence} |")

    lines.append("")

    # Repro commands (first bucket only, for brevity)
    if results:
        first = results[0]
        if isinstance(first, dict):
            b0 = first.get(baseline_label, {}) if isinstance(first.get(baseline_label), dict) else {}
            o0 = first.get(opt_label, {}) if isinstance(first.get(opt_label), dict) else {}

            b_cmd = b0.get("cmd")
            o_cmd = o0.get("cmd")
            b_env = b0.get("env_overrides", {}) if isinstance(b0.get("env_overrides"), dict) else {}
            o_env = o0.get("env_overrides", {}) if isinstance(o0.get("env_overrides"), dict) else {}

            if isinstance(b_cmd, list) and all(isinstance(x, str) for x in b_cmd):
                lines.append("Repro (baseline example):")
                lines.append("```bash")
                lines.append(_fmt_cmd(b_cmd, b_env))
                lines.append("```")
                lines.append("")

            if isinstance(o_cmd, list) and all(isinstance(x, str) for x in o_cmd):
                lines.append("Repro (optimized example):")
                lines.append("```bash")
                lines.append(_fmt_cmd(o_cmd, o_env))
                lines.append("```")
                lines.append("")

    lines.append("> Note: ensure CUDA graphs / torch.compile settings match production parity per `validation/E2E_LATENCY_GUIDE.md`.")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifact-dir", type=str, required=True)
    p.add_argument("--env-json", type=str, default=None)
    p.add_argument("--e2e-json", type=str, default=None)

    args = p.parse_args()

    artifact_dir = Path(args.artifact_dir).expanduser().resolve()
    env_path = Path(args.env_json).expanduser().resolve() if args.env_json else (artifact_dir / "env.json")
    e2e_path = Path(args.e2e_json).expanduser().resolve() if args.e2e_json else (artifact_dir / "e2e_latency" / "e2e_latency_results.json")

    env = _load_json(env_path)
    e2e = _load_json(e2e_path)

    summary: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "artifact_dir": str(artifact_dir),
        "inputs": {
            "env_json": str(env_path) if env is not None else None,
            "e2e_json": str(e2e_path) if e2e is not None else None,
        },
        "status": {
            "has_env": env is not None,
            "has_e2e": e2e is not None,
        },
    }

    lines: List[str] = []
    lines.append("# Validation Results (Phase 4)")
    lines.append("")
    lines.append(f"Generated: {summary['generated_at']} (UTC)")
    lines.append("")

    lines.append("> Default gates + required reporting checklist: `references/validation-defaults.md`.")
    lines.append("")

    # Environment
    lines.append("## Environment")
    lines.append("")
    if env is None:
        lines.append("- TODO: run `python scripts/collect_env.py --artifact-dir {artifact_dir}` to capture env.json")
    else:
        nvsmi = env.get("nvidia_smi", {})
        if isinstance(nvsmi, dict) and nvsmi.get("ok"):
            gpus = nvsmi.get("gpus", [])
            if isinstance(gpus, list) and gpus:
                # Record first GPU entry as representative.
                gpu0 = gpus[0]
                lines.append(f"- GPU: {gpu0.get('name')} (CC {gpu0.get('compute_cap')}, {gpu0.get('memory.total')} MiB)")
                lines.append(f"- Driver: {gpu0.get('driver_version')}")
        torch_info = env.get("torch", {})
        if isinstance(torch_info, dict):
            lines.append(f"- Torch: {torch_info.get('torch_version')} (CUDA {torch_info.get('cuda_version')})")
        vllm_info = env.get("vllm", {})
        if isinstance(vllm_info, dict):
            lines.append(f"- vLLM: {vllm_info.get('version')}")
        git_info = env.get("git", {})
        if isinstance(git_info, dict) and git_info.get("is_git_repo"):
            lines.append(f"- Git: {git_info.get('head')} (branch {git_info.get('branch')}, dirty={git_info.get('dirty')})")
    lines.append("")

    # Correctness placeholder
    lines.append("## Correctness")
    lines.append("")
    lines.append("- TODO: run model-appropriate correctness tests (prefer existing vLLM tests for the model).")
    lines.append("")

    # Kernel perf placeholder
    lines.append("## Kernel perf (CUDA graphs)")
    lines.append("")
    lines.append("- TODO: add GPU kernel-time table (baseline vs optimized) under CUDA graphs for the validated bucket set.")
    lines.append("")

    # E2E section
    if e2e is None:
        lines.append("## E2E latency (vllm bench latency)")
        lines.append("")
        lines.append("- TODO: run `python scripts/run_vllm_bench_latency_sweep.py --artifact-dir {artifact_dir} --run`.")
        lines.append("")
    else:
        lines.append(_render_e2e_section(e2e))

        # Summarize pass/fail vs default gates (if we have numbers)
        baseline_label = e2e.get("bench", {}).get("baseline_label", "baseline")
        opt_label = e2e.get("bench", {}).get("opt_label", "opt")
        buckets = e2e.get("results", [])
        gate = {
            "small_bs": {"buckets": [1, 4, 8], "min_improvement_pct": 5.0},
            "large_bs": {"buckets": [16, 32, 64], "min_improvement_pct": 0.0},
        }

        failing: List[str] = []
        if isinstance(buckets, list):
            for row in buckets:
                if not isinstance(row, dict):
                    continue
                bs = row.get("batch_size")
                imp = row.get("improvement_pct")
                if not isinstance(bs, int) or not isinstance(imp, (int, float)):
                    continue

                if bs in gate["small_bs"]["buckets"] and imp < gate["small_bs"]["min_improvement_pct"]:
                    failing.append(f"BS={bs}: {imp:.2f}% < 5%")
                if bs in gate["large_bs"]["buckets"] and imp <= gate["large_bs"]["min_improvement_pct"]:
                    failing.append(f"BS={bs}: {imp:.2f}% <= 0%")

        summary["e2e_gate"] = {
            "default_gate": gate,
            "failing": failing,
            "pass": len(failing) == 0,
            "note": "Default gates from references/validation-defaults.md; adjust if component share is small (see references/e2e-delta-math.md).",
        }

    # Decision placeholder
    lines.append("## Decision")
    lines.append("")
    lines.append("- TODO: Ship / restrict envelope / pivot route / stop (justify using kernel + E2E evidence).")
    lines.append("")

    out_md = artifact_dir / "validation_results.md"
    out_summary = artifact_dir / "validation_summary.json"

    _write_text(out_md, "\n".join(lines))
    _write_json(out_summary, summary)

    print(f"Wrote: {out_md}")
    print(f"Wrote: {out_summary}")


if __name__ == "__main__":
    main()
