#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  nsys_export_reports.sh <trace.nsys-rep> <out_dir>

Exports a small set of CSV reports that are useful for:
  - kernel ranking by total time
  - repeated kernel-chain mining
  - NVTX attribution (if present)

Outputs (best-effort; some may be unavailable depending on trace/settings):
  - cuda_gpu_kern_sum.csv
  - cuda_gpu_trace.csv
  - nvtx_sum.csv
EOF
}

if [[ $# -ne 2 ]]; then
  usage
  exit 2
fi

rep="$1"
out_dir="$2"

mkdir -p "${out_dir}"

emit_clean_csv() {
  local report="$1"
  local out_file="$2"

  # `nsys stats` may print informational preamble lines (e.g., "Generating SQLite file...")
  # before the CSV header. Strip everything until the first line containing a comma.
  nsys stats --force-overwrite=true --report "${report}" --format csv "${rep}" \
    | awk 'BEGIN{started=0} { if (!started && index($0, ",")>0) started=1; if (started) print }' \
    > "${out_file}"
}

emit_clean_csv cuda_gpu_kern_sum "${out_dir}/cuda_gpu_kern_sum.csv"
emit_clean_csv cuda_gpu_trace "${out_dir}/cuda_gpu_trace.csv" || true
emit_clean_csv nvtx_sum "${out_dir}/nvtx_sum.csv" || true

echo "Wrote:"
echo "  ${out_dir}/cuda_gpu_kern_sum.csv"
echo "  ${out_dir}/cuda_gpu_trace.csv"
echo "  ${out_dir}/nvtx_sum.csv (if available)"
