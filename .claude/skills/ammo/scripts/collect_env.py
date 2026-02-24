#!/usr/bin/env python3
"""ammo: collect environment + repo metadata.

Goal: produce reproducible, comparable benchmark evidence without guessing.

Writes (by default) into {artifact_dir}:
  - env.json  (machine-readable)
  - env.md    (human summary)

Guardrails:
  - If a tool is missing (nvidia-smi, git), record that fact rather than failing.
  - Never mutates the repo.

Example:
  python scripts/collect_env.py --artifact-dir artifacts/qwen3_l40s_fp8_tp1
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _run(cmd: List[str], *, cwd: Optional[Path] = None, timeout_s: int = 20) -> Dict[str, Any]:
    """Run a command and return a structured result."""
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        return {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "cmd": cmd,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    except FileNotFoundError as e:
        return {
            "ok": False,
            "error": f"FileNotFoundError: {e}",
            "cmd": cmd,
            "stdout": "",
            "stderr": "",
        }
    except subprocess.TimeoutExpired as e:
        return {
            "ok": False,
            "error": f"TimeoutExpired: {e}",
            "cmd": cmd,
            "stdout": e.stdout or "",
            "stderr": e.stderr or "",
        }


def _maybe_import_version(mod_name: str) -> Optional[str]:
    try:
        mod = __import__(mod_name)
        return getattr(mod, "__version__", None)
    except Exception:
        return None


def _git_info(repo_root: Path) -> Dict[str, Any]:
    # Detect if inside git repo.
    is_git = _run(["git", "rev-parse", "--is-inside-work-tree"], cwd=repo_root)
    if not is_git.get("ok"):
        return {"is_git_repo": False, "detail": is_git}

    head = _run(["git", "rev-parse", "HEAD"], cwd=repo_root)
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_root)
    status = _run(["git", "status", "--porcelain"], cwd=repo_root)

    dirty = bool(status.get("stdout", "").strip()) if status.get("ok") else None

    return {
        "is_git_repo": True,
        "head": head.get("stdout", "").strip() if head.get("ok") else None,
        "branch": branch.get("stdout", "").strip() if branch.get("ok") else None,
        "dirty": dirty,
        "status_porcelain": status.get("stdout", "") if status.get("ok") else None,
        "raw": {
            "rev_parse": head,
            "branch": branch,
            "status": status,
        },
    }


def _nvidia_smi() -> Dict[str, Any]:
    # Prefer query-based output for deterministic parsing.
    query_fields = [
        "name",
        "uuid",
        "compute_cap",
        "driver_version",
        "memory.total",
        "clocks.max.sm",
        "pci.bus_id",
    ]
    cmd = [
        "nvidia-smi",
        f"--query-gpu={','.join(query_fields)}",
        "--format=csv,noheader,nounits",
    ]
    res = _run(cmd)
    if not res.get("ok"):
        # Fallback to plain nvidia-smi.
        res2 = _run(["nvidia-smi"])
        return {
            "ok": False,
            "query": res,
            "fallback": res2,
        }

    gpus: List[Dict[str, str]] = []
    for line in res.get("stdout", "").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != len(query_fields):
            continue
        gpus.append({k: v for k, v in zip(query_fields, parts)})

    return {
        "ok": True,
        "gpus": gpus,
        "raw": res,
    }


def _torch_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    try:
        import torch  # type: ignore

        info["torch_version"] = getattr(torch, "__version__", None)
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_version"] = getattr(torch.version, "cuda", None)
        info["cudnn_version"] = getattr(torch.backends.cudnn, "version", lambda: None)()

        if torch.cuda.is_available():
            info["device_count"] = torch.cuda.device_count()
            # Record each visible device name + capability.
            devs = []
            for i in range(torch.cuda.device_count()):
                cap = torch.cuda.get_device_capability(i)
                devs.append(
                    {
                        "index": i,
                        "name": torch.cuda.get_device_name(i),
                        "capability": f"{cap[0]}.{cap[1]}",
                        "total_memory_bytes": int(torch.cuda.get_device_properties(i).total_memory),
                    }
                )
            info["devices"] = devs
    except Exception as e:
        info["error"] = str(e)
    return info


def _selected_env() -> Dict[str, str]:
    # Only capture environment variables that commonly affect vLLM / CUDA perf.
    allow_prefixes = (
        "CUDA",
        "NCCL",
        "VLLM",
        "TORCH",
        "PYTORCH",
        "HF_",
        "TRANSFORMERS",
    )
    allow_exact = {
        "PATH",
        "LD_LIBRARY_PATH",
        "CONDA_PREFIX",
        "VIRTUAL_ENV",
    }
    out: Dict[str, str] = {}
    for k, v in os.environ.items():
        if k in allow_exact or k.startswith(allow_prefixes):
            # Avoid dumping extremely long PATH-like vars in full.
            if k in ("PATH", "LD_LIBRARY_PATH") and len(v) > 2000:
                out[k] = v[:2000] + "...<truncated>"
            else:
                out[k] = v
    return dict(sorted(out.items(), key=lambda kv: kv[0]))


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_md(path: Path, env: Dict[str, Any]) -> None:
    def g(path_keys: List[str]) -> Any:
        cur: Any = env
        for k in path_keys:
            if not isinstance(cur, dict) or k not in cur:
                return None
            cur = cur[k]
        return cur

    lines: List[str] = []
    lines.append("# Environment Snapshot")
    lines.append("")
    lines.append(f"Generated: {env.get('generated_at')} (UTC)")
    lines.append("")

    # GPU
    nvsmi = env.get("nvidia_smi", {})
    if nvsmi.get("ok"):
        lines.append("## GPU (nvidia-smi)")
        lines.append("")
        for gpu in nvsmi.get("gpus", []):
            lines.append(f"- {gpu.get('name')} (CC {gpu.get('compute_cap')}), {gpu.get('memory.total')} MiB, driver {gpu.get('driver_version')}, bus {gpu.get('pci.bus_id')}")
        lines.append("")
    else:
        lines.append("## GPU (nvidia-smi)")
        lines.append("")
        lines.append("- nvidia-smi not available or query failed.")
        lines.append("")

    # Torch
    torch_info = env.get("torch", {})
    lines.append("## Python / Torch")
    lines.append("")
    lines.append(f"- Python: {env.get('python', {}).get('version')}")
    lines.append(f"- Torch: {torch_info.get('torch_version')}")
    lines.append(f"- Torch CUDA: {torch_info.get('cuda_version')} (available={torch_info.get('cuda_available')})")
    lines.append("")

    # vLLM
    lines.append("## vLLM")
    lines.append("")
    lines.append(f"- vLLM version: {env.get('vllm', {}).get('version')}")
    git = env.get("git", {})
    if git.get("is_git_repo"):
        lines.append(f"- Git: {git.get('head')} (branch {git.get('branch')}, dirty={git.get('dirty')})")
    else:
        lines.append("- Git: not a git repo (or git not available)")
    lines.append("")

    lines.append("## Selected environment variables")
    lines.append("")
    sel = env.get("env", {})
    if not sel:
        lines.append("- (none captured)")
    else:
        for k, v in sel.items():
            lines.append(f"- {k}={v}")
    lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifact-dir", type=str, required=True)
    p.add_argument("--repo-root", type=str, default=None, help="Optional repo root for git metadata; defaults to CWD")
    p.add_argument("--out-json", type=str, default=None)
    p.add_argument("--out-md", type=str, default=None)

    args = p.parse_args()

    artifact_dir = Path(args.artifact_dir).expanduser().resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)

    repo_root = Path(args.repo_root).expanduser().resolve() if args.repo_root else Path.cwd().resolve()

    env: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "python": {
            "version": sys.version.replace("\n", " "),
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "nvidia_smi": _nvidia_smi(),
        "torch": _torch_info(),
        "vllm": {
            "version": _maybe_import_version("vllm"),
        },
        "git": _git_info(repo_root),
        "env": _selected_env(),
    }

    out_json = Path(args.out_json).expanduser().resolve() if args.out_json else (artifact_dir / "env.json")
    out_md = Path(args.out_md).expanduser().resolve() if args.out_md else (artifact_dir / "env.md")

    _write_json(out_json, env)
    _write_md(out_md, env)

    print(f"Wrote: {out_json}")
    print(f"Wrote: {out_md}")


if __name__ == "__main__":
    main()
