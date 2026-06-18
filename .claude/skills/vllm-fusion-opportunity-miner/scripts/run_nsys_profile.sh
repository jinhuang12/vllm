#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_nsys_profile.sh --out-dir <dir> [--name <prefix>] [--] <command...>

Examples:
  run_nsys_profile.sh --out-dir artifacts/2026-01-02 --name decode -- \
    python benchmarks/benchmark_throughput.py --help

Notes:
  - Requires `nsys` in PATH.
  - Add extra Nsight Systems options via NSYS_ARGS, e.g.:
      NSYS_ARGS="--trace=cuda,nvtx,osrt --sample=none"
  - Enable CUDA graph node tracing (if supported) via:
      NSYS_ENABLE_CUDA_GRAPH_TRACE=1
EOF
}

out_dir=""
name="nsys_report"

if [[ $# -eq 0 ]]; then
  usage
  exit 2
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out-dir)
      out_dir="${2:-}"
      shift 2
      ;;
    --name)
      name="${2:-}"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    *)
      break
      ;;
  esac
done

if [[ -z "${out_dir}" ]]; then
  echo "error: --out-dir is required" >&2
  exit 2
fi
if [[ $# -lt 1 ]]; then
  echo "error: command is required (use -- <command...>)" >&2
  exit 2
fi

mkdir -p "${out_dir}"

{
  echo "date_utc: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  echo "pwd: $(pwd)"
  echo "nsys: $(command -v nsys || true)"
  echo "NSYS_ARGS: ${NSYS_ARGS:-}"
  echo "NSYS_ENABLE_CUDA_GRAPH_TRACE: ${NSYS_ENABLE_CUDA_GRAPH_TRACE:-0}"
  printf "command:"
  printf " %q" "$@"
  echo
} > "${out_dir}/command.txt"

default_args=(--trace=cuda,nvtx --sample=none)
nsys_help="$(nsys profile --help 2>&1 || true)"
if [[ "${nsys_help}" == *"--cuda-event-trace"* ]]; then
  default_args+=(--cuda-event-trace=false)
fi
if [[ "${NSYS_ENABLE_CUDA_GRAPH_TRACE:-0}" == "1" ]] && [[ "${nsys_help}" == *"--cuda-graph-trace"* ]]; then
  default_args+=(--cuda-graph-trace=node)
fi
extra_args=()
if [[ -n "${NSYS_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  extra_args=(${NSYS_ARGS})
fi

nsys profile \
  --force-overwrite=true \
  -o "${out_dir}/${name}" \
  "${default_args[@]}" \
  "${extra_args[@]}" \
  -- "$@"

echo "Wrote:"
echo "  ${out_dir}/${name}.nsys-rep"
echo "  ${out_dir}/${name}.qdstrm (if enabled)"
