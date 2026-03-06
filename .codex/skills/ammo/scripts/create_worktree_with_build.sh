#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 3 ]]; then
  echo "Usage: $0 <worktree-name> [branch-name] [main-repo]" >&2
  exit 2
fi

WORKTREE_NAME="$1"
BRANCH_NAME="${2:-ammo/${WORKTREE_NAME}}"
MAIN_REPO="${3:-$(git rev-parse --show-toplevel)}"
MAIN_REPO="$(cd "$MAIN_REPO" && pwd)"
WORKTREE_ROOT="$MAIN_REPO/.codex/worktrees"
WORKTREE_DIR="$WORKTREE_ROOT/$WORKTREE_NAME"
LOCKFILE="$WORKTREE_ROOT/.create-lock"
MAIN_VENV="$MAIN_REPO/.venv"

if [[ ! -x "$MAIN_VENV/bin/python" ]]; then
  echo "Missing main repo Python environment: $MAIN_VENV/bin/python" >&2
  exit 1
fi

if [[ -d "$WORKTREE_DIR" ]]; then
  echo "$WORKTREE_DIR"
  exit 0
fi

mkdir -p "$WORKTREE_ROOT"

(
  flock -x 200
  git -C "$MAIN_REPO" worktree add -b "$BRANCH_NAME" "$WORKTREE_DIR" HEAD >/dev/null 2>&1 || \
  git -C "$MAIN_REPO" worktree add "$WORKTREE_DIR" "$BRANCH_NAME" >/dev/null 2>&1 || {
    echo "Failed to create worktree $WORKTREE_NAME" >&2
    exit 1
  }
) 200>"$LOCKFILE"

if [[ ! -d "$WORKTREE_DIR" ]]; then
  echo "Worktree creation failed: $WORKTREE_DIR" >&2
  exit 1
fi

if [[ -f "$MAIN_REPO/CMakeUserPresets.json" ]]; then
  cp "$MAIN_REPO/CMakeUserPresets.json" "$WORKTREE_DIR/"
  if command -v jq >/dev/null 2>&1; then
    jq --arg pm "$WORKTREE_DIR:$MAIN_REPO" \
      '.configurePresets[0].environment.CCACHE_PATH_MAP = $pm' \
      "$WORKTREE_DIR/CMakeUserPresets.json" > "$WORKTREE_DIR/CMakeUserPresets.json.tmp"
    mv "$WORKTREE_DIR/CMakeUserPresets.json.tmp" "$WORKTREE_DIR/CMakeUserPresets.json"
  fi
fi

while IFS= read -r so_path; do
  rel_path="${so_path#$MAIN_REPO/}"
  mkdir -p "$(dirname "$WORKTREE_DIR/$rel_path")"
  cp "$so_path" "$WORKTREE_DIR/$rel_path"
done < <(find "$MAIN_REPO/vllm" -name '*.so' -type f 2>/dev/null | sort)

if [[ -f "$MAIN_REPO/vllm/_version.py" ]]; then
  mkdir -p "$WORKTREE_DIR/vllm"
  cp "$MAIN_REPO/vllm/_version.py" "$WORKTREE_DIR/vllm/_version.py"
fi

while IFS= read -r ignored_file; do
  [[ -z "$ignored_file" ]] && continue
  mkdir -p "$(dirname "$WORKTREE_DIR/$ignored_file")"
  cp "$MAIN_REPO/$ignored_file" "$WORKTREE_DIR/$ignored_file"
done < <(
  git -C "$MAIN_REPO" ls-files --others --ignored --exclude-standard -- \
    'vllm/vllm_flash_attn' 'vllm/third_party' 'vllm/grpc' 2>/dev/null | \
    grep -v '__pycache__' || true
)

PY_VERSION="$($MAIN_VENV/bin/python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
"$MAIN_VENV/bin/python" -m venv --without-pip "$WORKTREE_DIR/.venv" >/dev/null 2>&1
SITE_PACKAGES="$WORKTREE_DIR/.venv/lib/python${PY_VERSION}/site-packages"
mkdir -p "$SITE_PACKAGES"
printf '%s\n' "$MAIN_REPO/.venv/lib/python${PY_VERSION}/site-packages" > "$SITE_PACKAGES/main-venv.pth"
printf '%s\n' "$WORKTREE_DIR" > "$SITE_PACKAGES/worktree.pth"

for cmd in pytest vllm; do
  cat > "$WORKTREE_DIR/.venv/bin/$cmd" <<'WRAPPER'
#!/usr/bin/env bash
exec "$(dirname "$0")/python" -m CMD_PLACEHOLDER "$@"
WRAPPER
  sed -i "s|CMD_PLACEHOLDER|$cmd|g" "$WORKTREE_DIR/.venv/bin/$cmd"
  chmod +x "$WORKTREE_DIR/.venv/bin/$cmd"
done

echo "$WORKTREE_DIR"
