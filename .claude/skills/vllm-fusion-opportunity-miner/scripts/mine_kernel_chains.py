#!/usr/bin/env python3
"""
Mine:
- kernel_ranking.csv from cuda_gpu_kern_sum.csv
- kernel_chains.csv from cuda_gpu_trace.csv (adjacent micro-kernel soup patterns)

This script is heuristic. It is used to prioritize investigation, not to prove
correctness or performance.
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional


def _read_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _pick_col(cols: Iterable[str], candidates: List[str]) -> Optional[str]:
    cols_lc = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in cols_lc:
            return cols_lc[cand.lower()]
    return None


def _parse_float(x: str) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def _infer_unit(col_name: str) -> str:
    name = col_name.lower()
    if "(ns)" in name or name.endswith("_ns"):
        return "ns"
    if "(us)" in name or name.endswith("_us"):
        return "us"
    if "(ms)" in name or name.endswith("_ms"):
        return "ms"
    if "(s)" in name or name.endswith("_s"):
        return "s"
    return "unknown"


def _to_us(value: float, unit: str) -> float:
    if unit == "ns":
        return value / 1e3
    if unit == "us":
        return value
    if unit == "ms":
        return value * 1e3
    if unit == "s":
        return value * 1e6
    # Unknown: assume microseconds (common in nsys stats CSV) and let users override by selecting different columns.
    return value


@dataclass(frozen=True)
class KernelEvent:
    name: str
    dur_us: float


def _write_csv(path: str, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def build_kernel_ranking(kern_sum_csv: str) -> List[Dict[str, Any]]:
    rows = _read_csv_rows(kern_sum_csv)
    if not rows:
        return []

    cols = rows[0].keys()

    name_col = _pick_col(cols, ["Kernel Name", "Name", "Kernel", "KernelName"])
    total_col = _pick_col(
        cols, ["Total Time (ns)", "Total Time (us)", "Total Time (ms)", "Total Time", "Time", "Total"]
    )
    calls_col = _pick_col(cols, ["Calls", "Num Calls", "Count"])

    if name_col is None or total_col is None:
        raise RuntimeError(
            f"Could not find required columns in {kern_sum_csv}. Columns={list(cols)}"
        )

    total_unit = _infer_unit(total_col)

    agg_us: Dict[str, float] = defaultdict(float)
    calls: Dict[str, int] = defaultdict(int)

    for r in rows:
        name = (r.get(name_col) or "").strip()
        if not name:
            continue
        raw_total = r.get(total_col, "") or ""
        total_value = _parse_float(raw_total)
        if total_value is None:
            continue
        agg_us[name] += _to_us(total_value, total_unit)

        if calls_col:
            raw_calls = r.get(calls_col, "") or ""
            calls_value = _parse_float(raw_calls)
            if calls_value is not None:
                calls[name] += int(calls_value)

    total_us_all = sum(agg_us.values()) or 1.0
    out: List[Dict[str, Any]] = []
    for name, total_us in sorted(agg_us.items(), key=lambda kv: kv[1], reverse=True):
        out.append(
            {
                "kernel": name,
                "total_us": f"{total_us:.3f}",
                "share_pct": f"{(100.0 * total_us / total_us_all):.3f}",
                "calls": calls.get(name, 0),
            }
        )
    return out


def build_kernel_chain_candidates(
    gpu_trace_csv: str,
    *,
    max_kernel_us: float,
    min_chain_len: int,
    max_chain_len: int,
    min_chain_total_us: float,
) -> List[Dict[str, Any]]:
    rows = _read_csv_rows(gpu_trace_csv)
    if not rows:
        return []

    cols = rows[0].keys()
    name_col = _pick_col(cols, ["Name", "Kernel Name", "Kernel", "KernelName"])
    dur_col = _pick_col(
        cols, ["Duration (ns)", "Duration (us)", "Duration (ms)", "Duration", "Time (ns)", "Time (us)", "Time"]
    )
    if name_col is None or dur_col is None:
        raise RuntimeError(
            f"Could not find required columns in {gpu_trace_csv}. Columns={list(cols)}"
        )

    dur_unit = _infer_unit(dur_col)

    events: List[KernelEvent] = []
    for r in rows:
        name = (r.get(name_col) or "").strip()
        if not name:
            continue
        raw_duration = r.get(dur_col, "") or ""
        duration_value = _parse_float(raw_duration)
        if duration_value is None:
            continue
        events.append(KernelEvent(name=name, dur_us=_to_us(duration_value, dur_unit)))

    # Sliding-window over events. Only consider windows where all kernels are <= max_kernel_us.
    chains: Counter[str] = Counter()
    chain_total_us: Dict[str, float] = defaultdict(float)

    n = len(events)
    for i in range(n):
        if events[i].dur_us > max_kernel_us:
            continue

        seq: List[KernelEvent] = []
        total_us = 0.0
        for j in range(i, min(n, i + max_chain_len)):
            if events[j].dur_us > max_kernel_us:
                break
            seq.append(events[j])
            total_us += events[j].dur_us
            if len(seq) >= min_chain_len and total_us >= min_chain_total_us:
                sig = " -> ".join(e.name for e in seq)
                chains[sig] += 1
                chain_total_us[sig] += total_us

    out: List[Dict[str, Any]] = []
    for sig, cnt in chains.most_common():
        out.append(
            {
                "chain": sig,
                "occurrences": cnt,
                "total_us": f"{chain_total_us[sig]:.3f}",
                "mean_us_per_occurrence": f"{(chain_total_us[sig] / max(cnt, 1)):.3f}",
                "len": sig.count("->") + 1,
            }
        )

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu-trace-csv", required=True, help="Path to cuda_gpu_trace.csv")
    ap.add_argument("--kern-sum-csv", required=True, help="Path to cuda_gpu_kern_sum.csv")
    ap.add_argument("--out-dir", required=True, help="Output directory for analysis CSVs")

    ap.add_argument(
        "--max-kernel-us",
        type=float,
        default=20.0,
        help="Per-kernel duration threshold to consider it a micro-kernel.",
    )
    ap.add_argument("--min-chain-len", type=int, default=2)
    ap.add_argument("--max-chain-len", type=int, default=6)
    ap.add_argument(
        "--min-chain-total-us",
        type=float,
        default=10.0,
        help="Minimum total duration (us) for a chain occurrence to be recorded.",
    )

    args = ap.parse_args()

    ranking = build_kernel_ranking(args.kern_sum_csv)
    _write_csv(
        os.path.join(args.out_dir, "kernel_ranking.csv"),
        ["kernel", "total_us", "share_pct", "calls"],
        ranking,
    )

    chains = build_kernel_chain_candidates(
        args.gpu_trace_csv,
        max_kernel_us=args.max_kernel_us,
        min_chain_len=args.min_chain_len,
        max_chain_len=args.max_chain_len,
        min_chain_total_us=args.min_chain_total_us,
    )
    _write_csv(
        os.path.join(args.out_dir, "kernel_chains.csv"),
        ["chain", "occurrences", "total_us", "mean_us_per_occurrence", "len"],
        chains[:200],
    )

    print(
        "Wrote:\n"
        f"  {os.path.join(args.out_dir, 'kernel_ranking.csv')}\n"
        f"  {os.path.join(args.out_dir, 'kernel_chains.csv')}"
    )


if __name__ == "__main__":
    main()

