#!/usr/bin/env python3
"""Run adapter command templates from AMMO artifacts.

This script is runtime-agnostic. Project-specific behavior comes from adapter
manifests and optional key-value overrides.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _format_template(template: str, values: Dict[str, Any]) -> str:
    return template.format(**values)


def _run_cmd(cmd: str) -> int:
    proc = subprocess.run(cmd, shell=True, check=False)
    return proc.returncode


def _extract_values(bundle: Dict[str, Any], artifact_dir: Path) -> Dict[str, Any]:
    framework = bundle.get("framework", {})
    parity = bundle.get("parity", {})
    workload = bundle.get("workload", {})
    return {
        "artifact_dir": str(artifact_dir),
        "framework": framework.get("name", ""),
        "adapter": framework.get("adapter", ""),
        "model_id": workload.get("model_id", ""),
        "dtype": workload.get("dtype", ""),
        "tp": workload.get("tp", 1),
        "ep": workload.get("ep", 1),
        "buckets": ",".join(workload.get("buckets", [])) if isinstance(workload.get("buckets"), list) else "",
        "cuda_graphs": parity.get("cuda_graphs", ""),
        "torch_compile": parity.get("torch_compile", ""),
        "batch_size": 1,
        "input_len": 1024,
        "output_len": 32,
        "max_model_len": 4096,
        "warmup_iters": 3,
        "num_iters": 20,
        "label": "baseline",
        "bench_cmd": "",
    }


def _parse_set_args(raw_items: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for item in raw_items:
        if "=" not in item:
            raise SystemExit(f"Invalid --set entry: {item!r} (expected key=value)")
        k, v = item.split("=", 1)
        k = k.strip()
        if not k:
            raise SystemExit(f"Invalid --set entry: {item!r} (empty key)")
        out[k] = v
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifact-dir", required=True)
    p.add_argument("--manifest", required=True, help="Path to adapter manifest JSON")
    p.add_argument("--phase", choices=["baseline", "profile", "e2e"], default="baseline")
    p.add_argument(
        "--set",
        dest="set_values",
        action="append",
        default=[],
        help="Template override key=value (repeatable)",
    )
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    artifact_dir = Path(args.artifact_dir).expanduser().resolve()
    bundle_path = artifact_dir / "artifact_bundle.json"
    if not bundle_path.exists():
        raise SystemExit(f"artifact_bundle.json not found at {bundle_path}")

    bundle = _load_json(bundle_path)
    manifest = _load_json(Path(args.manifest).expanduser().resolve())

    phase_key = {
        "baseline": "baseline_cmd_templates",
        "profile": "profile_cmd_templates",
        "e2e": "e2e_cmd_templates",
    }[args.phase]

    templates = manifest.get(phase_key, [])
    adapter = manifest.get("name", "<unknown>")
    maturity = manifest.get("maturity", "beta")

    if not templates:
        if maturity == "beta":
            raise SystemExit(
                f"{adapter} adapter is beta and has no {phase_key}; mark run blocked and provide environment-specific templates"
            )
        raise SystemExit(f"No templates found for {adapter}:{phase_key}")

    values = _extract_values(bundle, artifact_dir)
    values.update(_parse_set_args(args.set_values))
    rc = 0
    for i, template in enumerate(templates, start=1):
        try:
            cmd = _format_template(template, values)
        except KeyError as exc:
            raise SystemExit(
                f"Template variable missing for adapter={adapter} phase={args.phase}: {exc}. "
                "Provide with --set key=value or add it to artifact_bundle.json."
            )
        print(f"[{i}/{len(templates)}] {cmd}")
        if not args.dry_run:
            rc = _run_cmd(cmd)
            if rc != 0:
                raise SystemExit(rc)

    raise SystemExit(rc)


if __name__ == "__main__":
    main()
