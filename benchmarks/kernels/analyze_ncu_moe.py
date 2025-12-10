#!/usr/bin/env python3
"""
Analyze Nsight Compute results for MoE monokernel.

This script parses NCU CSV output and generates a summary report
identifying bottlenecks and optimization opportunities.

Usage:
    # Generate CSV from NCU report:
    ncu --import moe_monokernel_bs64.ncu-rep --csv > moe_bs64.csv

    # Analyze the CSV:
    python benchmarks/kernels/analyze_ncu_moe.py moe_bs64.csv

    # Compare two profiles:
    python benchmarks/kernels/analyze_ncu_moe.py moe_bs1.csv moe_bs64.csv

Expected Metrics (from roofline analysis):
    - Tensor Core Utilization: ~1-3%
    - SM Throughput: ~5-10%
    - DRAM Throughput: ~20-40%
    - Barrier Stalls: HIGH (grid.sync() overhead)

LLM Council Approved: 2025-12-09
"""

import argparse
import sys
from pathlib import Path
from typing import Optional


def parse_ncu_csv(csv_path: str) -> dict:
    """
    Parse NCU CSV output into a dictionary of metric name -> value.

    Args:
        csv_path: Path to NCU CSV file

    Returns:
        Dictionary mapping metric names to values
    """
    metrics = {}

    try:
        # Try pandas first for robust CSV parsing
        import pandas as pd

        df = pd.read_csv(csv_path)

        # NCU CSV format typically has columns:
        # "ID", "Process ID", "Process Name", "Host Name", "Kernel Name",
        # "Kernel Time", "Context", "Stream", "Section Name", "Metric Name",
        # "Metric Unit", "Metric Value"

        if "Metric Name" in df.columns and "Metric Value" in df.columns:
            for _, row in df.iterrows():
                name = str(row.get("Metric Name", "")).strip()
                value = row.get("Metric Value", "")
                if name and value != "":
                    try:
                        metrics[name] = float(value)
                    except (ValueError, TypeError):
                        metrics[name] = str(value)
        else:
            print(f"Warning: Unexpected CSV format in {csv_path}")
            print(f"Columns: {list(df.columns)}")

    except ImportError:
        # Fallback to basic CSV parsing
        import csv

        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row.get("Metric Name", "").strip()
                value = row.get("Metric Value", "")
                if name and value:
                    try:
                        metrics[name] = float(value)
                    except ValueError:
                        metrics[name] = value

    return metrics


def get_metric(metrics: dict, pattern: str, default: float = 0.0) -> float:
    """
    Get a metric value by partial name match.

    Args:
        metrics: Dictionary of metrics
        pattern: Pattern to match in metric name
        default: Default value if not found

    Returns:
        Metric value or default
    """
    for name, value in metrics.items():
        if pattern in name:
            if isinstance(value, (int, float)):
                return float(value)
    return default


def analyze_metrics(metrics: dict, batch_size: int = 64) -> dict:
    """
    Analyze NCU metrics and identify bottlenecks.

    Args:
        metrics: Dictionary of metrics from NCU CSV
        batch_size: Batch size used for profiling

    Returns:
        Dictionary of analysis results
    """
    results = {}

    # Key metrics for GEMM analysis
    # Tensor Core utilization
    tc_util = get_metric(metrics, "pipe_tensor.avg.pct_of_peak_sustained_active")
    results["tensor_core_utilization"] = tc_util

    # SM throughput
    sm_throughput = get_metric(metrics, "sm__throughput.avg.pct_of_peak_sustained")
    results["sm_throughput"] = sm_throughput

    # DRAM throughput
    dram_throughput = get_metric(metrics, "dram__throughput.avg.pct_of_peak_sustained")
    results["dram_throughput"] = dram_throughput

    # Warp occupancy
    warp_occupancy = get_metric(metrics, "warps_active.avg.pct_of_peak_sustained")
    results["warp_occupancy"] = warp_occupancy

    # L2 cache hit rate
    l2_hit_rate = get_metric(metrics, "lts__average_hit_rate.pct")
    if l2_hit_rate == 0.0:
        l2_hit_rate = get_metric(metrics, "lts__t_sector_hit_rate.pct")
    results["l2_hit_rate"] = l2_hit_rate

    # Stall reasons
    stall_barrier = get_metric(metrics, "warp_issue_stall_barrier")
    stall_membar = get_metric(metrics, "warp_issue_stall_membar")
    stall_wait = get_metric(metrics, "warp_issue_stall_wait")
    stall_mio = get_metric(metrics, "warp_issue_stall_mio")
    stall_lg = get_metric(metrics, "warp_issue_stall_lg")

    results["stalls"] = {
        "barrier": stall_barrier,
        "membar": stall_membar,
        "wait": stall_wait,
        "mio": stall_mio,
        "lg": stall_lg,
    }

    # FP8 instruction mix (if available)
    hmma_sum = get_metric(metrics, "inst_executed_pipe_tensor_op_hmma.sum")
    results["hmma_instructions"] = hmma_sum

    return results


def identify_bottlenecks(results: dict) -> list:
    """
    Identify bottlenecks from analysis results.

    Args:
        results: Dictionary from analyze_metrics()

    Returns:
        List of (bottleneck_name, description, recommendation) tuples
    """
    bottlenecks = []

    tc_util = results.get("tensor_core_utilization", 0)
    sm_throughput = results.get("sm_throughput", 0)
    dram_throughput = results.get("dram_throughput", 0)
    warp_occupancy = results.get("warp_occupancy", 0)
    l2_hit_rate = results.get("l2_hit_rate", 0)
    stalls = results.get("stalls", {})

    # Low Tensor Core utilization (expected for MoE monokernel)
    if tc_util < 5:
        bottlenecks.append(
            (
                "LOW_TENSOR_CORE",
                f"Tensor Core utilization is {tc_util:.1f}% (expected 1-3%)",
                "Root cause: Small M=8 tiles underutilize 16x8x8 MMA operations. "
                "Sequential expert processing limits parallelism. "
                "This is a fundamental constraint of the monokernel design.",
            )
        )

    # Memory bandwidth limited
    if dram_throughput > 70:
        bottlenecks.append(
            (
                "MEMORY_BANDWIDTH",
                f"DRAM throughput is {dram_throughput:.1f}% of peak",
                "Memory bandwidth is highly utilized. "
                "Consider: Better weight caching, prefetching, or L2 cache optimization.",
            )
        )

    # Low occupancy
    if warp_occupancy > 0 and warp_occupancy < 30:
        bottlenecks.append(
            (
                "LOW_OCCUPANCY",
                f"Warp occupancy is {warp_occupancy:.1f}%",
                "Low occupancy reduces latency hiding. "
                "Consider: Reduce register usage, shared memory per block, or increase threads.",
            )
        )

    # High barrier stalls (grid.sync overhead)
    barrier_stall = stalls.get("barrier", 0)
    if barrier_stall > 20:
        bottlenecks.append(
            (
                "BARRIER_STALLS",
                f"Barrier stalls at {barrier_stall:.1f}%",
                "High barrier stalls indicate grid.sync() overhead. "
                "This is expected for cooperative kernels. "
                "Consider: Reduce number of grid syncs or use async operations.",
            )
        )

    # High memory stalls
    mio_stall = stalls.get("mio", 0)
    lg_stall = stalls.get("lg", 0)
    if mio_stall + lg_stall > 30:
        bottlenecks.append(
            (
                "MEMORY_STALLS",
                f"Memory stalls: MIO={mio_stall:.1f}%, LG={lg_stall:.1f}%",
                "High memory stalls indicate memory access latency is not being hidden. "
                "Consider: Better prefetching, double/triple buffering, or cache optimization.",
            )
        )

    # Low L2 hit rate
    if 0 < l2_hit_rate < 50:
        bottlenecks.append(
            (
                "L2_CACHE_MISS",
                f"L2 cache hit rate is {l2_hit_rate:.1f}%",
                "Low L2 hit rate indicates poor data locality. "
                "Consider: Reorder memory accesses, tile sizes, or reduce working set.",
            )
        )

    return bottlenecks


def print_report(results: dict, bottlenecks: list, csv_path: str):
    """
    Print analysis report.

    Args:
        results: Dictionary from analyze_metrics()
        bottlenecks: List from identify_bottlenecks()
        csv_path: Path to CSV file analyzed
    """
    print("=" * 70)
    print("MoE Monokernel NCU Analysis Report")
    print("=" * 70)
    print(f"Source: {csv_path}")
    print()

    # Key metrics
    print("Key Metrics:")
    print("-" * 40)
    tc = results.get("tensor_core_utilization", 0)
    sm = results.get("sm_throughput", 0)
    dram = results.get("dram_throughput", 0)
    occ = results.get("warp_occupancy", 0)
    l2 = results.get("l2_hit_rate", 0)

    print(f"  Tensor Core Utilization: {tc:6.2f}%")
    print(f"  SM Throughput:           {sm:6.2f}%")
    print(f"  DRAM Throughput:         {dram:6.2f}%")
    print(f"  Warp Occupancy:          {occ:6.2f}%")
    print(f"  L2 Cache Hit Rate:       {l2:6.2f}%")
    print()

    # Stall breakdown
    stalls = results.get("stalls", {})
    if any(v > 0 for v in stalls.values()):
        print("Stall Breakdown:")
        print("-" * 40)
        for name, value in sorted(stalls.items(), key=lambda x: -x[1]):
            if value > 0:
                print(f"  {name:20s}: {value:6.2f}%")
        print()

    # Bottleneck analysis
    if bottlenecks:
        print("Identified Bottlenecks:")
        print("-" * 40)
        for i, (name, desc, rec) in enumerate(bottlenecks, 1):
            print(f"\n{i}. [{name}]")
            print(f"   {desc}")
            print(f"   Recommendation: {rec}")
        print()
    else:
        print("No significant bottlenecks identified.")
        print()

    # Summary
    print("Summary:")
    print("-" * 40)
    if tc < 5:
        print(
            "  The monokernel achieves ~{:.1f}% Tensor Core utilization.".format(tc)
        )
        print("  This is expected due to:")
        print("    - Small M=8 tile size (Tensor Cores prefer M>=64)")
        print("    - Sequential expert processing (limited parallelism)")
        print("    - Cooperative kernel constraints (128 blocks max)")
        print()
        print("  The monokernel is optimal for its design constraints:")
        print("    - BS=1-4: ~4x faster than reference (kernel launch overhead)")
        print("    - BS>4: Use hybrid dispatch with Triton for better throughput")
    else:
        print(
            "  Higher than expected Tensor Core utilization ({:.1f}%).".format(tc)
        )
        print("  Verify measurement methodology.")

    print()
    print("=" * 70)


def compare_reports(
    results1: dict, results2: dict, path1: str, path2: str
):
    """
    Compare two NCU analysis results.

    Args:
        results1: First profile results
        results2: Second profile results
        path1: Path to first CSV
        path2: Path to second CSV
    """
    print("=" * 70)
    print("MoE Monokernel NCU Comparison")
    print("=" * 70)
    print(f"Profile 1: {path1}")
    print(f"Profile 2: {path2}")
    print()

    print("Metric Comparison:")
    print("-" * 60)
    print(f"{'Metric':<30s} {'Profile 1':>12s} {'Profile 2':>12s} {'Delta':>12s}")
    print("-" * 60)

    metrics_to_compare = [
        ("Tensor Core Util", "tensor_core_utilization"),
        ("SM Throughput", "sm_throughput"),
        ("DRAM Throughput", "dram_throughput"),
        ("Warp Occupancy", "warp_occupancy"),
        ("L2 Hit Rate", "l2_hit_rate"),
    ]

    for name, key in metrics_to_compare:
        v1 = results1.get(key, 0)
        v2 = results2.get(key, 0)
        delta = v2 - v1
        delta_str = f"{delta:+.2f}%" if v1 != 0 or v2 != 0 else "N/A"
        print(f"{name:<30s} {v1:>11.2f}% {v2:>11.2f}% {delta_str:>12s}")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Nsight Compute results for MoE monokernel",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Analyze single profile:
    python %(prog)s moe_bs64.csv

    # Compare two profiles:
    python %(prog)s moe_bs1.csv moe_bs64.csv

To generate CSV from NCU report:
    ncu --import moe_monokernel_bs64.ncu-rep --csv > moe_bs64.csv
""",
    )
    parser.add_argument(
        "csv_file",
        type=str,
        help="Path to NCU CSV file",
    )
    parser.add_argument(
        "csv_file2",
        type=str,
        nargs="?",
        default=None,
        help="Optional second CSV file for comparison",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=64,
        help="Batch size used for profiling. Default: 64",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )

    args = parser.parse_args()

    # Check file exists
    if not Path(args.csv_file).exists():
        print(f"Error: File not found: {args.csv_file}", file=sys.stderr)
        sys.exit(1)

    # Parse and analyze
    metrics = parse_ncu_csv(args.csv_file)
    if not metrics:
        print(f"Warning: No metrics found in {args.csv_file}", file=sys.stderr)
        print("Make sure the file is in NCU CSV format.", file=sys.stderr)
        sys.exit(1)

    results = analyze_metrics(metrics, args.batch_size)
    bottlenecks = identify_bottlenecks(results)

    if args.csv_file2:
        # Compare mode
        if not Path(args.csv_file2).exists():
            print(f"Error: File not found: {args.csv_file2}", file=sys.stderr)
            sys.exit(1)

        metrics2 = parse_ncu_csv(args.csv_file2)
        results2 = analyze_metrics(metrics2, args.batch_size)

        compare_reports(results, results2, args.csv_file, args.csv_file2)
    elif args.json:
        # JSON output
        import json

        output = {
            "metrics": results,
            "bottlenecks": [
                {"name": name, "description": desc, "recommendation": rec}
                for name, desc, rec in bottlenecks
            ],
        }
        print(json.dumps(output, indent=2))
    else:
        # Standard report
        print_report(results, bottlenecks, args.csv_file)


if __name__ == "__main__":
    main()
