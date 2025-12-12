#!/usr/bin/env python3
"""
Render a Codex custom-prompt markdown file into a plain prompt string for `codex exec`.

- Strips YAML frontmatter (`--- ... ---`) if present at the top of the file.
- Replaces `$KEY` placeholders with values passed as `KEY=VALUE` CLI args.

Example:
  python render_prompt.py --prompt-file .codex/prompts/moe-monokernel-optimizer.md \
      MODEL_ID="Qwen/..." HARDWARE="L40S" DTYPE=fp8 TP=1 TOPK=8
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


_FRONTMATTER_RE = re.compile(r"\A---\s*\n.*?\n---\s*\n", re.DOTALL)


def strip_frontmatter(text: str) -> str:
    m = _FRONTMATTER_RE.match(text)
    return text[m.end():] if m else text


def parse_kv(args: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for a in args:
        if "=" not in a:
            raise ValueError(f"Expected KEY=VALUE, got: {a}")
        k, v = a.split("=", 1)
        k = k.strip()
        if not k:
            raise ValueError(f"Empty KEY in: {a}")
        out[k] = v
    return out


def substitute(text: str, mapping: dict[str, str]) -> str:
    # Replace $KEY occurrences. We intentionally do not try to be too clever:
    # this is a simple literal substitution that mirrors how prompt placeholders behave.
    for k, v in mapping.items():
        text = text.replace(f"${k}", v)
    return text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-file", required=True, help="Path to the codex prompt markdown file")
    ap.add_argument("kv", nargs="*", help="KEY=VALUE replacements")
    ns = ap.parse_args()

    prompt_path = Path(ns.prompt_file)
    if not prompt_path.exists():
        print(f"error: prompt file not found: {prompt_path}", file=sys.stderr)
        return 2

    raw = prompt_path.read_text(encoding="utf-8")
    body = strip_frontmatter(raw)

    mapping = {}
    try:
        mapping = parse_kv(ns.kv)
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    rendered = substitute(body, mapping).strip() + "\n"
    sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
