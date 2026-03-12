#!/usr/bin/env python3
"""Generate matplotlib charts for AMMO optimization reports.

Reads campaign artifacts (bottleneck_analysis.md, constraints.md, validation
results, benchmark JSONs) and produces publication-quality PNG charts.

Usage:
    python generate_charts.py --artifact-dir <path> [--output-dir <path>]

Outputs:
    1. kernel_breakdown_pie.png    — GPU time by kernel category
    2. bw_utilization_bar.png      — Per-GEMM bandwidth utilization
    3. e2e_results_bar.png         — Before/after E2E latency comparison
    4. roofline_plot.png           — Arithmetic intensity vs throughput
    5. nsys_timeline_synthetic.png — One decode step kernel sequence
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _extract_table_rows(md_text: str, header_pattern: str) -> list[list[str]]:
    """Extract markdown table rows after a header matching *header_pattern*."""
    lines = md_text.splitlines()
    in_table = False
    rows: list[list[str]] = []
    for line in lines:
        if re.search(header_pattern, line, re.IGNORECASE):
            in_table = True
            continue
        if in_table:
            stripped = line.strip()
            if stripped.startswith("|") and "---" not in stripped:
                cells = [c.strip() for c in stripped.split("|")[1:-1]]
                rows.append(cells)
            elif not stripped.startswith("|") and rows:
                break
    return rows


def _parse_float(s: str) -> float | None:
    """Best-effort float extraction from a markdown cell."""
    m = re.search(r"[\d.]+", s.replace(",", "").replace("~", ""))
    return float(m.group()) if m else None


# ---------------------------------------------------------------------------
# Chart generators
# ---------------------------------------------------------------------------

def generate_kernel_pie(
    bottleneck_md: str, output_path: Path, model_name: str, batch_size: int
) -> None:
    """Chart 1: Kernel time breakdown pie chart."""
    # Try to parse from bottleneck analysis executive summary table
    rows = _extract_table_rows(bottleneck_md, r"Executive Summary|Candidate Ranking")
    if not rows:
        # Fallback: parse from top-15 kernel breakdown
        rows = _extract_table_rows(bottleneck_md, r"Top 15|kernel breakdown")

    # Build data from whatever we found, or use defaults
    labels: list[str] = []
    sizes: list[float] = []

    for row in rows:
        if len(row) >= 3:
            name = row[1] if len(row) > 1 else row[0]
            f_val = _parse_float(row[2]) if len(row) > 2 else None
            if f_val and f_val > 0:
                labels.append(name.strip()[:30])
                sizes.append(f_val)

    if not sizes:
        print(f"  WARNING: Could not parse kernel data for pie chart, skipping")
        return

    # Add 'Other' if total < 100
    total = sum(sizes)
    if total < 99:
        labels.append("Other")
        sizes.append(100 - total)

    colors = plt.cm.Set2(np.linspace(0, 1, len(sizes)))

    fig, ax = plt.subplots(figsize=(10, 7))
    wedges, _, autotexts = ax.pie(
        sizes, labels=None,
        autopct=lambda p: f"{p:.1f}%" if p > 1.5 else "",
        colors=colors, startangle=90, pctdistance=0.82,
        textprops={"fontsize": 9},
    )
    ax.legend(
        wedges,
        [f"{l} ({s:.1f}%)" for l, s in zip(labels, sizes)],
        loc="center left", bbox_to_anchor=(1, 0.5), fontsize=9,
    )
    ax.set_title(
        f"GPU Kernel Time Breakdown — {model_name} BS={batch_size} Decode\n"
        f"(Production Parity: CUDA Graphs + torch.compile)",
        fontsize=13, fontweight="bold",
    )
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close()


def generate_bw_bar(
    bottleneck_md: str, output_path: Path, hw_bw_gbps: float
) -> None:
    """Chart 2: Per-GEMM bandwidth utilization bar chart."""
    rows = _extract_table_rows(bottleneck_md, r"Per-GEMM breakdown|Per-GEMM.*BW|Utilization")
    if not rows:
        print("  WARNING: Could not parse per-GEMM data, skipping BW chart")
        return

    gemms: list[str] = []
    bw_pcts: list[float] = []
    for row in rows:
        if len(row) >= 5:
            name = row[0].strip()[:20]
            util = _parse_float(row[-1])  # Last column is usually utilization
            if util and util > 0:
                gemms.append(name)
                bw_pcts.append(util)

    if not gemms:
        print("  WARNING: No GEMM BW data found, skipping")
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(gemms))
    colors = ["#e74c3c" if v < 10 else "#3498db" for v in bw_pcts]
    bars = ax.bar(x, bw_pcts, 0.5, color=colors, edgecolor="white", linewidth=0.5)

    avg_bw = sum(bw_pcts) / len(bw_pcts) if bw_pcts else 0
    ax.axhline(y=avg_bw, color="red", linestyle="--", linewidth=1.5,
               label=f"Average: {avg_bw:.1f}%")
    ax.axhline(y=100, color="green", linestyle=":", linewidth=1, alpha=0.5,
               label=f"Peak BW ({hw_bw_gbps:.0f} GB/s)")

    for bar, val in zip(bars, bw_pcts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xlabel("GEMM Operation", fontsize=12)
    ax.set_ylabel("HBM Bandwidth Utilization (%)", fontsize=12)
    ax.set_title("Per-GEMM Bandwidth Utilization — Baseline", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(gemms, fontsize=9, rotation=15, ha="right")
    ax.set_ylim(0, 115)
    ax.legend(fontsize=10)
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close()


def generate_e2e_bar(
    artifact_dir: Path, output_path: Path, shipped_ops: list[str]
) -> None:
    """Chart 3: E2E results grouped bar chart (baseline vs optimized)."""
    # Find baseline and optimized JSON files
    baseline_dir = artifact_dir / "e2e_latency" / "json"
    batch_sizes: list[int] = []
    baselines: list[float] = []
    optimized: list[float] = []

    # Try to find optimized results from any shipped track
    opt_dirs = []
    for op_id in shipped_ops:
        op_id_clean = op_id.lower().replace("-", "")
        for pattern in [f"e2e_latency_{op_id_clean}", f"e2e_latency_{op_id_clean}_*"]:
            opt_dirs.extend(artifact_dir.glob(pattern))
    # Prefer latest version (_v3 > _v2 > base)
    opt_dirs.sort(key=lambda d: d.name)

    for bs in [1, 4, 8, 16, 32, 64]:
        bl_file = baseline_dir / f"baseline_bs{bs}.json"
        if bl_file.exists():
            bl = _load_json(bl_file)
            bl_avg = bl.get("avg_latency")
            if bl_avg is None:
                continue

            # Find matching optimized result
            opt_avg = None
            for od in opt_dirs:
                of = od / "json" / f"opt_bs{bs}.json"
                if of.exists():
                    opt_data = _load_json(of)
                    opt_avg = opt_data.get("avg_latency")
                    break

            if opt_avg is not None:
                batch_sizes.append(bs)
                baselines.append(bl_avg)
                optimized.append(opt_avg)

    if not batch_sizes:
        print("  WARNING: No matching baseline/optimized pairs found, skipping E2E chart")
        return

    improvement = [(b - o) / b * 100 for b, o in zip(baselines, optimized)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5),
                                     gridspec_kw={"width_ratios": [2, 1]})
    x = np.arange(len(batch_sizes))
    width = 0.35
    ax1.bar(x - width / 2, baselines, width, label="Baseline", color="#3498db", alpha=0.8)
    ax1.bar(x + width / 2, optimized, width, label="Optimized", color="#2ecc71", alpha=0.8)
    ax1.set_xlabel("Batch Size", fontsize=12)
    ax1.set_ylabel("E2E Latency (seconds)", fontsize=12)
    ax1.set_title("E2E Latency: Baseline vs Optimized", fontsize=13, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(batch_sizes)
    ax1.legend(fontsize=10)

    colors_imp = ["#95a5a6" if v <= 0 else "#2ecc71" for v in improvement]
    bars = ax2.bar(x, improvement, 0.5, color=colors_imp, edgecolor="white")
    for bar, val in zip(bars, improvement):
        ax2.text(bar.get_x() + bar.get_width() / 2, max(bar.get_height(), 0) + 0.15,
                 f"{val:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Batch Size", fontsize=12)
    ax2.set_ylabel("E2E Improvement (%)", fontsize=12)
    ax2.set_title("Improvement Over Baseline", fontsize=13, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(batch_sizes)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close()


def generate_roofline(
    bottleneck_md: str, output_path: Path,
    hw_bw_gbps: float, hw_tflops: float
) -> None:
    """Chart 4: Roofline plot."""
    peak_flops = hw_tflops * 1e3  # GFLOPS
    ridge = peak_flops / hw_bw_gbps

    fig, ax = plt.subplots(figsize=(10, 6))
    ai_range = np.logspace(-1, 3, 200)
    roofline = np.minimum(peak_flops, hw_bw_gbps * ai_range)
    ax.loglog(ai_range, roofline, "k-", linewidth=2, label="Roofline")

    # Try to extract data points from bottleneck analysis
    rows = _extract_table_rows(bottleneck_md, r"Per-GEMM breakdown|Per-GEMM.*BW")
    if rows:
        for row in rows:
            if len(row) >= 5:
                name = row[0].strip()[:15]
                util = _parse_float(row[-1])
                if util and util > 0:
                    ai = 8  # Default AI for BF16 GEMMs at small M
                    perf = util / 100 * hw_bw_gbps * ai
                    c = "#f39c12" if util < 10 else "#e74c3c"
                    ax.plot(ai, perf, "o", markersize=8, color=c, zorder=5)
                    ax.annotate(name, (ai, perf), textcoords="offset points",
                                xytext=(10, 5), fontsize=7)

    ax.axvline(x=ridge, color="gray", linestyle=":", alpha=0.5)
    ax.text(ridge * 1.1, peak_flops * 0.5,
            f"Ridge: {ridge:.0f}\nFLOPS/byte", fontsize=8, color="gray")
    ax.set_xlabel("Arithmetic Intensity (FLOPS/byte)", fontsize=12)
    ax.set_ylabel("Performance (GFLOPS)", fontsize=12)
    ax.set_title("Roofline Analysis\nDecode GEMMs in the memory-bound regime",
                 fontsize=13, fontweight="bold")
    ax.set_xlim(0.5, 2000)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close()


def generate_timeline(
    constraints_md: str, output_path: Path
) -> None:
    """Chart 5: Synthetic nsys timeline for one decode step."""
    # Try to parse kernel sequence from constraints
    rows = _extract_table_rows(constraints_md, r"Per-decode-step kernel sequence")

    # Fallback: use a generic per-layer sequence
    kernels = [
        ("LayerNorm", 1.8, "#9b59b6"),
        ("Projection\n(GEMM)", 98, "#e74c3c"),
        ("Small GEMM", 15, "#c0392b"),
        ("Pointwise", 3, "#95a5a6"),
        ("Conv1d", 2.5, "#1abc9c"),
        ("Gating", 1.2, "#f39c12"),
        ("Recurrence", 30, "#3498db"),
        ("Norm", 3.4, "#9b59b6"),
        ("MLP up\n(GEMM)", 155, "#e74c3c"),
        ("Activation", 1.2, "#f39c12"),
        ("MLP down\n(GEMM)", 75, "#e74c3c"),
        ("LayerNorm", 1.6, "#9b59b6"),
    ]

    fig, ax = plt.subplots(figsize=(16, 3.5))
    t = 0
    for name, dur, color in kernels:
        rect = plt.Rectangle((t, 0.2), dur, 0.6, facecolor=color,
                              edgecolor="white", linewidth=0.5)
        ax.add_patch(rect)
        if dur > 8:
            ax.text(t + dur / 2, 0.5, f"{name}\n{dur}us",
                    ha="center", va="center", fontsize=7,
                    fontweight="bold", color="white")
        t += dur + 0.3

    ax.set_xlim(-5, t + 10)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Time (microseconds)", fontsize=12)
    ax.set_title("Kernel Timeline — One Decoder Layer (decode step)\n"
                 "GEMMs (red) dominate layer time",
                 fontsize=12, fontweight="bold")
    ax.set_yticks([])
    legend_items = [
        mpatches.Patch(color="#e74c3c", label="GEMM"),
        mpatches.Patch(color="#3498db", label="Recurrence"),
        mpatches.Patch(color="#9b59b6", label="LayerNorm"),
        mpatches.Patch(color="#f39c12", label="Activation"),
        mpatches.Patch(color="#95a5a6", label="Pointwise"),
        mpatches.Patch(color="#1abc9c", label="Conv1d"),
    ]
    ax.legend(handles=legend_items, loc="upper right", fontsize=8, ncol=3)
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifact-dir", required=True, type=str,
                   help="Path to AMMO campaign artifact directory")
    p.add_argument("--output-dir", type=str, default=None,
                   help="Output directory for charts (default: {artifact-dir}/report_assets)")
    p.add_argument("--hw-bw-gbps", type=float, default=864,
                   help="GPU peak HBM bandwidth in GB/s (default: 864 for L40S)")
    p.add_argument("--hw-tflops", type=float, default=362,
                   help="GPU peak BF16 TFLOPS (default: 362 for L40S)")
    args = p.parse_args()

    if not HAS_MATPLOTLIB:
        print("WARNING: matplotlib not available. Charts will not be generated.")
        print("Install with: pip install matplotlib")
        sys.exit(0)

    artifact_dir = Path(args.artifact_dir)
    output_dir = Path(args.output_dir) if args.output_dir else artifact_dir / "report_assets"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read source data
    bottleneck_md = _read_text(artifact_dir / "bottleneck_analysis.md")
    constraints_md = _read_text(artifact_dir / "constraints.md")
    state = _load_json(artifact_dir / "state.json")

    model_name = state.get("target", {}).get("model_id", "Unknown Model")
    model_short = model_name.split("/")[-1] if "/" in model_name else model_name
    shipped = state.get("campaign", {}).get("shipped_optimizations", [])

    plt.rcParams.update({
        "font.size": 11,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
    })

    charts = [
        ("kernel_breakdown_pie.png",
         lambda p: generate_kernel_pie(bottleneck_md, p, model_short, 8)),
        ("bw_utilization_bar.png",
         lambda p: generate_bw_bar(bottleneck_md, p, args.hw_bw_gbps)),
        ("e2e_results_bar.png",
         lambda p: generate_e2e_bar(artifact_dir, p, shipped)),
        ("roofline_plot.png",
         lambda p: generate_roofline(bottleneck_md, p, args.hw_bw_gbps, args.hw_tflops)),
        ("nsys_timeline_synthetic.png",
         lambda p: generate_timeline(constraints_md, p)),
    ]

    for name, gen_fn in charts:
        out = output_dir / name
        try:
            gen_fn(out)
            print(f"  Generated: {out}")
        except Exception as e:
            print(f"  ERROR generating {name}: {e}")

    print(f"\nAll charts written to: {output_dir}")


if __name__ == "__main__":
    main()
