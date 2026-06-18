#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sqlite3
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class KernelEvent:
    start: int
    duration_ns: int
    name: str


def _die(message: str, exit_code: int = 2) -> None:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(exit_code)


def _read_table_names(conn: sqlite3.Connection) -> set[str]:
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    )
    return {row[0] for row in cursor.fetchall()}


def _read_table_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    cursor = conn.execute(f"PRAGMA table_info({table})")
    return [row[1] for row in cursor.fetchall()]


def _read_table_column_types(conn: sqlite3.Connection, table: str) -> dict[str, str]:
    cursor = conn.execute(f"PRAGMA table_info({table})")
    return {row[1]: (row[2] or "") for row in cursor.fetchall()}


def _pick_column(columns: set[str], candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def _detect_kernel_table(conn: sqlite3.Connection) -> tuple[str, str, str, str, str]:
    """
    Return (table, start_col, end_or_duration_col, name_expr, join_clause).

    Nsight Systems sqlite schemas can vary across versions; this tries to
    discover a usable table/column set without hardcoding a single schema.
    """
    tables = _read_table_names(conn)

    preferred_tables = [
        "CUPTI_ACTIVITY_KIND_KERNEL",
        "CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL",
        "CUDA_GPU_TRACE",
    ]
    candidate_tables = [t for t in preferred_tables if t in tables]
    if not candidate_tables:
        kernelish = [t for t in tables if "KERNEL" in t.upper()]
        if kernelish:
            candidate_tables = [kernelish[0]]

    if not candidate_tables:
        _die(
            "Could not find a kernel table in sqlite. "
            "Try exporting with: nsys export --type sqlite -o <out> <report.nsys-rep>"
        )

    for table in candidate_tables:
        columns = set(_read_table_columns(conn, table))
        column_types = _read_table_column_types(conn, table)
        start_col = _pick_column(columns, ["start", "startTime", "timestamp", "ts"])
        end_col = _pick_column(columns, ["end", "endTime"])
        duration_col = _pick_column(columns, ["duration", "dur"])
        name_col = _pick_column(
            columns,
            [
                "demangledName",
                "name",
                "shortName",
                "kernelName",
                "mangledName",
                "symbolName",
            ],
        )

        if not (start_col and name_col and (end_col or duration_col)):
            continue

        # Newer Nsight Systems exports store kernel names as integer IDs into
        # StringIds.value. Detect and join when needed.
        join_clause = ""
        name_expr = name_col
        name_type = (column_types.get(name_col) or "").upper()
        if name_type == "INTEGER" and "StringIds" in tables:
            join_clause = f" JOIN StringIds ON {table}.{name_col} = StringIds.id"
            name_expr = "StringIds.value"

        return table, start_col, (end_col or duration_col), name_expr, join_clause

    _die(
        f"Found candidate kernel tables {candidate_tables}, but could not identify "
        "start/end/duration/name columns. Inspect schema with sqlite3 and PRAGMA."
    )
    raise AssertionError("unreachable")


def _export_nsys_to_sqlite(nsys_rep: Path, out_sqlite: Path) -> None:
    nsys = shutil.which("nsys")
    if not nsys:
        _die("`nsys` not found in PATH; provide --sqlite instead of --nsys-rep")
    out_sqlite.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        nsys,
        "export",
        "--force-overwrite=true",
        "--type",
        "sqlite",
        "-o",
        str(out_sqlite),
        str(nsys_rep),
    ]
    subprocess.run(cmd, check=True)


def _is_sqlite_file(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            header = f.read(16)
        return header.startswith(b"SQLite format 3")
    except OSError:
        return False


def _resolve_exported_sqlite(out_prefix: Path) -> Path:
    # Nsight Systems output naming has changed across versions. Depending on the
    # installed version, `nsys export --type sqlite -o <prefix>` may produce:
    #   - <prefix>
    #   - <prefix>.sqlite
    #   - <prefix>.sqlite.db (less common)
    candidates = [
        out_prefix,
        Path(str(out_prefix) + ".sqlite"),
        Path(str(out_prefix) + ".sqlite.db"),
        Path(str(out_prefix) + ".db"),
    ]
    for candidate in candidates:
        if candidate.exists() and _is_sqlite_file(candidate):
            return candidate

    # Fallback: pick any sqlite-looking file that shares the prefix.
    for candidate in sorted(out_prefix.parent.glob(out_prefix.name + "*")):
        if candidate.exists() and _is_sqlite_file(candidate):
            return candidate

    _die(
        f"nsys export did not produce a readable sqlite database for prefix: {out_prefix}"
    )
    raise AssertionError("unreachable")


def _iter_kernel_events(
    conn: sqlite3.Connection,
    table: str,
    start_col: str,
    end_or_duration_col: str,
    name_expr: str,
    join_clause: str,
    max_events: int | None,
) -> Iterable[KernelEvent]:
    columns = set(_read_table_columns(conn, table))
    has_end = end_or_duration_col in {"end", "endTime"} or end_or_duration_col.startswith(
        "end"
    )

    select_cols = f"{start_col}, {end_or_duration_col}, {name_expr}"
    query = f"SELECT {select_cols} FROM {table}{join_clause} ORDER BY {start_col}"
    if max_events is not None:
        query += f" LIMIT {int(max_events)}"

    cursor = conn.execute(query)
    for start, end_or_duration, name in cursor:
        if name is None:
            continue
        start_int = int(start)
        value_int = int(end_or_duration)
        if has_end:
            duration_ns = max(0, value_int - start_int)
        else:
            duration_ns = max(0, value_int)
        yield KernelEvent(start=start_int, duration_ns=duration_ns, name=str(name))


def _normalize_kernel_name(name: str, max_len: int = 180) -> str:
    normalized = " ".join(name.split())
    if len(normalized) <= max_len:
        return normalized
    return normalized[: max_len - 1] + "…"


def _format_float(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="nsys_mine.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """
            Mine and rank GPU kernel fusion opportunities from a Nsight Systems sqlite export.

            Typical workflow:
              1) nsys profile -o report -- <command...>
              2) nsys export --type sqlite -o report_sqlite report.nsys-rep
              3) python scripts/nsys_mine.py --sqlite report_sqlite.sqlite --out-dir artifacts/
            """
        ).strip(),
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--sqlite", type=Path, help="Nsight Systems sqlite export.")
    input_group.add_argument("--nsys-rep", type=Path, help="Nsight Systems .nsys-rep file.")

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("."),
        help="Output directory (default: current directory).",
    )
    parser.add_argument(
        "--top-kernels",
        type=int,
        default=50,
        help="Top kernels to include by total GPU time.",
    )
    parser.add_argument(
        "--chain-lens",
        type=str,
        default="2,3,4,5",
        help="Comma-separated chain lengths to mine (default: 2,3,4,5).",
    )
    parser.add_argument(
        "--min-chain-count",
        type=int,
        default=50,
        help="Minimum occurrences for a chain to be reported.",
    )
    parser.add_argument(
        "--small-kernel-us",
        type=float,
        default=30.0,
        help="Threshold for 'small kernel' heuristic (default: 30us).",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=500_000,
        help="Maximum kernel events to load (default: 500000).",
    )
    parser.add_argument(
        "--no-normalize-names",
        action="store_true",
        help="Do not shorten long kernel names.",
    )

    args = parser.parse_args(argv)
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_md = out_dir / "nsys_mining.md"
    out_json = out_dir / "nsys_mining.json"

    sqlite_path: Path
    if args.sqlite:
        sqlite_path = args.sqlite
    else:
        nsys_rep: Path = args.nsys_rep
        if not nsys_rep.exists():
            _die(f"nsys report not found: {nsys_rep}")
        sqlite_prefix = out_dir / nsys_rep.stem
        _export_nsys_to_sqlite(nsys_rep, sqlite_prefix)
        sqlite_path = _resolve_exported_sqlite(sqlite_prefix)

    if not sqlite_path.exists():
        _die(f"sqlite not found: {sqlite_path}")

    conn = sqlite3.connect(str(sqlite_path))
    try:
        table, start_col, end_or_duration_col, name_expr, join_clause = _detect_kernel_table(conn)
        kernel_events = list(
            _iter_kernel_events(
                conn,
                table=table,
                start_col=start_col,
                end_or_duration_col=end_or_duration_col,
                name_expr=name_expr,
                join_clause=join_clause,
                max_events=args.max_events,
            )
        )
    finally:
        conn.close()

    if not kernel_events:
        _die("No kernel events found in sqlite export.")

    normalize_names = not args.no_normalize_names
    events: list[KernelEvent] = []
    for event in kernel_events:
        name = _normalize_kernel_name(event.name) if normalize_names else event.name
        events.append(KernelEvent(start=event.start, duration_ns=event.duration_ns, name=name))

    # Per-kernel stats.
    kernel_total_ns: dict[str, int] = {}
    kernel_count: dict[str, int] = {}
    kernel_max_ns: dict[str, int] = {}
    for event in events:
        kernel_total_ns[event.name] = kernel_total_ns.get(event.name, 0) + event.duration_ns
        kernel_count[event.name] = kernel_count.get(event.name, 0) + 1
        kernel_max_ns[event.name] = max(kernel_max_ns.get(event.name, 0), event.duration_ns)

    total_gpu_ns = sum(kernel_total_ns.values())

    top_by_total = sorted(
        kernel_total_ns.items(), key=lambda kv: kv[1], reverse=True
    )[: max(1, args.top_kernels)]
    top_by_count = sorted(kernel_count.items(), key=lambda kv: kv[1], reverse=True)[
        : max(1, args.top_kernels)
    ]

    # Encode kernel names for n-gram mining.
    name_to_id: dict[str, int] = {}
    id_to_name: list[str] = []
    kernel_ids: list[int] = []
    durations_ns: list[int] = []

    for event in events:
        kernel_name = event.name
        kernel_id = name_to_id.get(kernel_name)
        if kernel_id is None:
            kernel_id = len(id_to_name)
            name_to_id[kernel_name] = kernel_id
            id_to_name.append(kernel_name)
        kernel_ids.append(kernel_id)
        durations_ns.append(event.duration_ns)

    prefix_ns = [0]
    for duration in durations_ns:
        prefix_ns.append(prefix_ns[-1] + duration)

    chain_lens = []
    for part in args.chain_lens.split(","):
        part = part.strip()
        if not part:
            continue
        chain_lens.append(int(part))
    chain_lens = sorted({l for l in chain_lens if l >= 2})
    if not chain_lens:
        _die("--chain-lens must include at least one length >= 2")

    @dataclass
    class ChainStats:
        chain: tuple[int, ...]
        count: int = 0
        total_ns: int = 0

        @property
        def avg_us(self) -> float:
            return (self.total_ns / max(1, self.count)) / 1_000.0

        @property
        def total_ms(self) -> float:
            return self.total_ns / 1_000_000.0

    chains_by_len: dict[int, dict[tuple[int, ...], ChainStats]] = {}
    num_events = len(kernel_ids)

    for chain_len in chain_lens:
        stats: dict[tuple[int, ...], ChainStats] = {}
        limit = num_events - chain_len + 1
        for i in range(limit):
            window = tuple(kernel_ids[i : i + chain_len])
            total_ns_window = prefix_ns[i + chain_len] - prefix_ns[i]
            chain_stat = stats.get(window)
            if chain_stat is None:
                chain_stat = ChainStats(chain=window)
                stats[window] = chain_stat
            chain_stat.count += 1
            chain_stat.total_ns += total_ns_window
        chains_by_len[chain_len] = stats

    def chain_to_names(chain: tuple[int, ...]) -> list[str]:
        return [id_to_name[kernel_id] for kernel_id in chain]

    # Candidate selection and scoring.
    kernel_avg_us: dict[str, float] = {}
    for name, total_ns in kernel_total_ns.items():
        kernel_avg_us[name] = (total_ns / max(1, kernel_count[name])) / 1_000.0

    @dataclass(frozen=True)
    class Candidate:
        chain_len: int
        chain: tuple[int, ...]
        count: int
        total_ms: float
        avg_us: float
        score: float
        all_small: bool

    candidates: list[Candidate] = []
    for chain_len, chain_stats in chains_by_len.items():
        for stats in chain_stats.values():
            if stats.count < args.min_chain_count:
                continue
            names = chain_to_names(stats.chain)
            all_small = all(kernel_avg_us[n] <= args.small_kernel_us for n in names)
            if not all_small:
                continue
            score = stats.total_ms * (chain_len - 1) * math.log1p(stats.count)
            candidates.append(
                Candidate(
                    chain_len=chain_len,
                    chain=stats.chain,
                    count=stats.count,
                    total_ms=stats.total_ms,
                    avg_us=stats.avg_us,
                    score=score,
                    all_small=all_small,
                )
            )

    candidates.sort(key=lambda c: c.score, reverse=True)
    top_candidates = candidates[:50]

    # Materialize top chains for reporting.
    def top_chains(chain_len: int, top_n: int = 30) -> list[ChainStats]:
        stats = list(chains_by_len.get(chain_len, {}).values())
        stats = [s for s in stats if s.count >= args.min_chain_count]
        stats.sort(key=lambda s: (s.total_ns, s.count), reverse=True)
        return stats[:top_n]

    report: dict[str, Any] = {
        "input_sqlite": str(sqlite_path),
        "kernel_table": {
            "table": table,
            "start_col": start_col,
            "end_or_duration_col": end_or_duration_col,
            "name_expr": name_expr,
            "join_clause": join_clause,
        },
        "num_events": len(events),
        "total_gpu_time_ms": total_gpu_ns / 1_000_000.0,
        "top_kernels_by_total": [
            {
                "name": name,
                "total_ms": total_ns / 1_000_000.0,
                "count": kernel_count[name],
                "avg_us": kernel_avg_us[name],
                "max_us": kernel_max_ns[name] / 1_000.0,
            }
            for name, total_ns in top_by_total
        ],
        "top_kernels_by_count": [
            {
                "name": name,
                "count": count,
                "total_ms": kernel_total_ns[name] / 1_000_000.0,
                "avg_us": kernel_avg_us[name],
            }
            for name, count in top_by_count
        ],
        "top_small_kernel_chain_candidates": [
            {
                "chain_len": c.chain_len,
                "count": c.count,
                "total_ms": c.total_ms,
                "avg_us": c.avg_us,
                "score": c.score,
                "chain": chain_to_names(c.chain),
            }
            for c in top_candidates
        ],
    }

    out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    def md_table(rows: list[list[str]], headers: list[str]) -> str:
        lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
        for row in rows:
            lines.append("| " + " | ".join(row) + " |")
        return "\n".join(lines)

    top_kernel_rows = []
    for item in report["top_kernels_by_total"]:
        top_kernel_rows.append(
            [
                item["name"],
                _format_float(item["total_ms"]),
                str(item["count"]),
                _format_float(item["avg_us"]),
                _format_float(item["max_us"]),
                _format_float(100.0 * item["total_ms"] / (report["total_gpu_time_ms"] or 1.0)),
            ]
        )

    candidate_rows = []
    for item in report["top_small_kernel_chain_candidates"][:20]:
        chain_str = "  →  ".join(item["chain"])
        candidate_rows.append(
            [
                str(item["chain_len"]),
                str(item["count"]),
                _format_float(item["total_ms"]),
                _format_float(item["avg_us"]),
                _format_float(item["score"]),
                chain_str,
            ]
        )

    mined_md = []
    mined_md.append("# vLLM fusion opportunity mining (Nsight Systems)\n")
    mined_md.append("## Summary")
    mined_md.append(
        f"- Input sqlite: `{sqlite_path}`\n"
        f"- Kernel table: `{table}` ({start_col}, {end_or_duration_col}, {name_expr})\n"
        f"- Kernel events: `{report['num_events']}`\n"
        f"- Total GPU kernel time (sum): `{_format_float(report['total_gpu_time_ms'])} ms`"
    )
    if join_clause:
        mined_md.append(f"- Name join: `{join_clause.strip()}`")
    mined_md.append("\n## Top kernels by total GPU time")
    mined_md.append(
        md_table(
            top_kernel_rows,
            headers=[
                "Kernel",
                "Total (ms)",
                "Count",
                "Avg (us)",
                "Max (us)",
                "Share (%)",
            ],
        )
    )

    mined_md.append("\n## Candidate kernel-chain fusions (heuristic)")
    mined_md.append(
        "Heuristic: chains that repeat frequently and consist entirely of kernels whose per-kernel avg runtime "
        f"is <= `{args.small_kernel_us} us`.\n"
    )
    if candidate_rows:
        mined_md.append(
            md_table(
                candidate_rows,
                headers=["Len", "Count", "Total (ms)", "Avg (us)", "Score", "Chain"],
            )
        )
    else:
        mined_md.append(
            "- No candidates matched the heuristic thresholds. "
            "Try lowering `--min-chain-count` or increasing `--small-kernel-us`."
        )

    mined_md.append("\n## Copy/paste patch plan (fill-in)")
    mined_md.append(
        "Use this section as a prioritized checklist. For each item, link evidence back to this file and the "
        "raw `.nsys-rep` / `.sqlite` artifacts.\n"
    )
    if report["top_small_kernel_chain_candidates"]:
        for idx, item in enumerate(report["top_small_kernel_chain_candidates"][:10], start=1):
            chain_str = " → ".join(item["chain"])
            mined_md.append(
                textwrap.dedent(
                    f"""
                    ### Opportunity {idx}: fuse/tune repeated kernel chain

                    - Evidence: chain_len={item['chain_len']}, count={item['count']}, total_ms={_format_float(item['total_ms'])}, avg_us={_format_float(item['avg_us'])}, score={_format_float(item['score'])}
                    - Kernel chain: `{chain_str}`
                    - Suspected vLLM code path(s):
                      - TODO: search `csrc/` and `vllm/` for kernel/op names; add file+symbol pointers
                    - Fusion/tuning hypothesis:
                      - TODO: (e.g., fuse pointwise ops, reduce launch count, hoist casts, use persistent kernel, use epilogue fusion, use grouped GEMM, etc.)
                    - Feasibility (H/M/L): TODO
                    - Risk (H/M/L): TODO (numerics, nondeterminism, alignment, shape dynamism, graph capture constraints)
                    - Validation plan:
                      - TODO: correctness tests + golden outputs
                      - TODO: benchmark command + expected win (prefill vs decode)
                    """
                ).strip()
            )
            mined_md.append("")
    else:
        mined_md.append("- No candidates to generate a patch plan from.")

    mined_md.append("\n## Reproducibility checklist")
    mined_md.append(
        "- Store the exact benchmark command, model, dtype, batch/seq lengths, and vLLM git SHA.\n"
        "- Capture both `.nsys-rep` and exported `.sqlite`.\n"
        "- Record key findings and commands in `validation_results.md` (repo convention)."
    )
    out_md.write_text("\n".join(mined_md).rstrip() + "\n")

    print(f"Wrote: {out_md}")
    print(f"Wrote: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
