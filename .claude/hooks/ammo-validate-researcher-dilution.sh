#!/bin/bash
# PostToolUse hook — researcher dilution sanity check (Phase 1B / spec §3.3).
#
# Fires on Write|Edit. Path filter: only validate when the written file matches
# the canonical V2 path
#   */kernel_opt_artifacts/*/rounds/*/mining/bottleneck_analysis.md
# or the legacy V1 path
#   */kernel_opt_artifacts/*/bottleneck_analysis.md
#
# Schema-version legacy guard: silent exit 0 when state.json has
# campaign.schema_version absent or < "4.1".
#
# Validates (schema_version >= 4.1):
#   1. decode_busy in [0.20, 1.0]
#   2. decode_share_of_e2e in [0.0, 1.0]
#   3. decode_kernel_s / decode_wall_s ≈ decode_busy (±0.05)
#   4. f_e2e ≤ f_decode (decode-graph %) for non-prefill-active, non-infra rows
#      (skip rows where decode-graph %=n/a OR prefill-active?=Yes)
#   5. sum(f_e2e for kernel rows) ≤ decode_busy × decode_share_of_e2e + 0.01
#
# Failure mode: ALWAYS exit 0 (fail-open). Errors surface as a stderr WARN
# line beginning with "WARN: ammo-validate-researcher-dilution:". Missing jq
# emits one WARN and exits.
set -uo pipefail
trap 'exit 0' ERR

WARN_PREFIX="WARN: ammo-validate-researcher-dilution"

emit_warn() {
    echo "${WARN_PREFIX}: $*, skipping check" >&2
}

# ── Dependency check ───────────────────────────────────────────────────────
if ! command -v jq >/dev/null 2>&1; then
    emit_warn "jq not available"
    exit 0
fi

INPUT=$(cat)

FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // ""' 2>/dev/null) || true
[ -z "$FILE_PATH" ] && exit 0

# ── Path filter: V2 canonical path or legacy V1 path ──────────────────────
# V2: */kernel_opt_artifacts/*/rounds/*/mining/bottleneck_analysis.md
# V1: */kernel_opt_artifacts/*/bottleneck_analysis.md
case "$FILE_PATH" in
    */kernel_opt_artifacts/*/rounds/*/mining/bottleneck_analysis.md) ;;
    */kernel_opt_artifacts/*/bottleneck_analysis.md) ;;
    *) exit 0;;
esac

[ -f "$FILE_PATH" ] || exit 0

# ── Locate state.json. In V1, bottleneck_analysis.md sits at the artifact
# root. In V2, it lives at rounds/{N}/mining/bottleneck_analysis.md, so we
# walk up until we find state.json (capped at 5 levels for safety).
ARTIFACT_DIR=$(dirname "$FILE_PATH")
STATE_FILE=""
_search_dir="$ARTIFACT_DIR"
for _ in 1 2 3 4 5; do
    if [ -f "$_search_dir/state.json" ]; then
        STATE_FILE="$_search_dir/state.json"
        ARTIFACT_DIR="$_search_dir"
        break
    fi
    _parent=$(dirname "$_search_dir")
    [ "$_parent" = "$_search_dir" ] && break
    _search_dir="$_parent"
done
# Preserve legacy semantics: if no state.json found at all, point at the
# original sibling location (will trigger the missing-file silent skip).
[ -z "$STATE_FILE" ] && STATE_FILE="$ARTIFACT_DIR/state.json"

# ── Schema-version legacy guard ────────────────────────────────────────────
if [ ! -f "$STATE_FILE" ]; then
    # Spec: missing state.json → fail-open silent (no WARN required).
    exit 0
fi

SCHEMA_VERSION=$(jq -r '.campaign.schema_version // ""' "$STATE_FILE" 2>/dev/null)
JQ_RC=$?
if [ $JQ_RC -ne 0 ]; then
    emit_warn "jq failed to parse $STATE_FILE (exit $JQ_RC)"
    exit 0
fi
if [ -z "$SCHEMA_VERSION" ]; then
    # schema_version absent → silent skip (legacy)
    exit 0
fi

# Semver-ish comparison: split on '.', compare first two as ints.
SV_MAJOR=$(echo "$SCHEMA_VERSION" | cut -d. -f1)
SV_MINOR=$(echo "$SCHEMA_VERSION" | cut -d. -f2)
case "$SV_MAJOR$SV_MINOR" in
    ''|*[!0-9]*)
        # Non-numeric — skip silently (legacy).
        exit 0
        ;;
esac

if [ "$SV_MAJOR" -lt 4 ] || { [ "$SV_MAJOR" -eq 4 ] && [ "$SV_MINOR" -lt 1 ]; }; then
    # < 4.1 → silent skip.
    exit 0
fi

# ── Run python parser/validator ────────────────────────────────────────────
WARNINGS=$(BOTTLENECK_FILE="$FILE_PATH" python3 <<'PY' 2>/dev/null
import os
import re
import sys


def _try_float(s):
    if s is None:
        return None
    s = s.strip()
    if not s or s.lower() in ("n/a", "na", "unknown", "—", "-"):
        return None
    # Strip trailing % or ×, $ etc.
    s2 = re.sub(r"[%×x*]+\s*$", "", s).strip()
    try:
        return float(s2)
    except ValueError:
        return None


def _parse_pct_as_fraction(s):
    """Parse "70.9%" → 0.709, "0.5" → 0.5. Returns None if 'n/a' etc."""
    if s is None:
        return None
    s = s.strip()
    if not s or s.lower() in ("n/a", "na", "unknown", "—", "-"):
        return None
    has_pct = s.endswith("%")
    s2 = re.sub(r"[%×x*]+\s*$", "", s).strip()
    try:
        v = float(s2)
    except ValueError:
        return None
    if has_pct:
        v = v / 100.0
    return v


def _parse_md_table(lines, start_idx):
    """Parse a markdown table starting at lines[start_idx] (header row).

    Returns (headers, rows_as_dicts, end_idx).
    """
    header_line = lines[start_idx].strip()
    if not header_line.startswith("|"):
        return None, [], start_idx
    # Skip the separator line at start_idx + 1
    headers = [h.strip() for h in header_line.strip("|").split("|")]
    rows = []
    i = start_idx + 2
    while i < len(lines):
        line = lines[i].strip()
        if not line.startswith("|"):
            break
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < len(headers):
            cells += [""] * (len(headers) - len(cells))
        rows.append({headers[j]: cells[j] for j in range(len(headers))})
        i += 1
    return headers, rows, i


def main():
    path = os.environ["BOTTLENECK_FILE"]
    try:
        with open(path) as f:
            text = f.read()
    except OSError:
        return
    lines = text.splitlines()

    warnings = []

    # ── Find Workload Dilution table ─────────────────────────────────────────
    dilution_rows = []
    top_components_rows = []
    for i, line in enumerate(lines):
        s = line.strip().lower()
        if s.startswith("## workload dilution"):
            # Find the first table header that follows
            for j in range(i + 1, min(i + 12, len(lines))):
                if lines[j].lstrip().startswith("|"):
                    _, rows, _ = _parse_md_table(lines, j)
                    dilution_rows = rows
                    break
        elif s.startswith("## top components"):
            for j in range(i + 1, min(i + 12, len(lines))):
                if lines[j].lstrip().startswith("|"):
                    _, rows, _ = _parse_md_table(lines, j)
                    top_components_rows = rows
                    break

    if not dilution_rows:
        warnings.append("missing '## Workload Dilution' table")
    if not top_components_rows:
        warnings.append("missing '## Top Components' table")

    # If we didn't find required tables, emit warnings and stop.
    if warnings:
        for w in warnings:
            print(w)
        return

    # Use first dilution row (per-BS validation; one row per BS — we sanity check each)
    for d in dilution_rows:
        # Headers may carry stray spaces; build a normalized lookup
        norm = {k.strip().lower(): v for k, v in d.items()}
        decode_busy = _try_float(norm.get("decode_busy"))
        decode_share = _try_float(norm.get("decode_share_of_e2e"))
        decode_wall = _try_float(norm.get("decode_wall_s"))
        decode_kernel = _try_float(norm.get("decode_kernel_s"))
        bs_label = norm.get("bs", "?")

        # ─ Bounds ────────────────────────────────────────────
        if decode_busy is None:
            warnings.append(f"BS={bs_label}: decode_busy missing/non-numeric")
        else:
            if decode_busy < 0.20 or decode_busy > 1.0:
                warnings.append(
                    f"BS={bs_label}: decode_busy={decode_busy:.3f} outside [0.20, 1.0]"
                )

        if decode_share is None:
            warnings.append(f"BS={bs_label}: decode_share_of_e2e missing/non-numeric")
        else:
            if decode_share < 0.0 or decode_share > 1.0:
                warnings.append(
                    f"BS={bs_label}: decode_share_of_e2e={decode_share:.3f} outside [0.0, 1.0]"
                )

        # ─ Cross-check: decode_kernel_s / decode_wall_s ≈ decode_busy ────────
        if decode_busy is not None and decode_wall and decode_wall > 0 and decode_kernel is not None:
            ratio = decode_kernel / decode_wall
            if abs(ratio - decode_busy) > 0.05:
                warnings.append(
                    f"BS={bs_label}: decode_kernel_s/decode_wall_s={ratio:.3f} "
                    f"vs decode_busy={decode_busy:.3f} — cross-check ratio mismatch >0.05"
                )

        # ─ Budget: Σ f_e2e (kernel rows) ≤ decode_busy × decode_share + 0.01 ─
        if decode_busy is not None and decode_share is not None:
            budget = decode_busy * decode_share + 0.01
            kernel_sum = 0.0
            kernel_rows_seen = 0
            for r in top_components_rows:
                rn = {k.strip().lower(): v for k, v in r.items()}
                # Match BS if present
                row_bs = rn.get("bs", "")
                if bs_label and row_bs and row_bs != bs_label:
                    continue
                graph_pct_raw = rn.get("decode-graph %", "")
                if graph_pct_raw.strip().lower() in ("n/a", "na", "—", "-", ""):
                    # Infra row — excluded from kernel-budget sum
                    continue
                f_e2e = _try_float(rn.get("f_e2e"))
                if f_e2e is None:
                    continue
                kernel_sum += f_e2e
                kernel_rows_seen += 1
            if kernel_rows_seen > 0 and kernel_sum > budget:
                warnings.append(
                    f"BS={bs_label}: sum(f_e2e kernel rows)={kernel_sum:.3f} "
                    f"exceeds decode_busy×decode_share={budget - 0.01:.3f} (budget+0.01) — "
                    f"f_e2e values exceed total decode-kernel budget"
                )

    # ─ f_e2e ≤ f_decode (decode-graph %) per kernel row ──────────────────────
    for r in top_components_rows:
        rn = {k.strip().lower(): v for k, v in r.items()}
        graph_raw = rn.get("decode-graph %", "")
        graph_low = graph_raw.strip().lower()
        if graph_low in ("n/a", "na", "—", "-", ""):
            continue  # infra row
        prefill_active = rn.get("prefill-active?", "").strip().lower()
        if prefill_active in ("yes", "y", "true"):
            continue  # mixed-phase: skip per spec
        f_decode = _parse_pct_as_fraction(graph_raw)
        f_e2e = _try_float(rn.get("f_e2e"))
        if f_decode is None or f_e2e is None:
            continue
        # Allow tiny float wiggle (rounding from displayed percentages).
        if f_e2e > f_decode + 0.005:
            comp = rn.get("component", "?")
            warnings.append(
                f"row '{comp}': f_e2e={f_e2e:.3f} > f_decode={f_decode:.3f} "
                f"(decode-graph %={graph_raw}) — f_e2e cannot exceed f_decode for "
                "decode-only kernels"
            )

    for w in warnings:
        print(w)


try:
    main()
except Exception as e:  # pragma: no cover - fail-open guard
    print(f"internal parser error: {type(e).__name__}: {e}")
PY
)

if [ -n "$WARNINGS" ]; then
    while IFS= read -r line; do
        [ -z "$line" ] && continue
        emit_warn "$line"
    done <<< "$WARNINGS"
fi

exit 0
