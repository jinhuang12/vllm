"""CuTeDSL CUDA-graph capture self-test scaffold.

Template used by AMMO champions proposing a CuTeDSL kernel. Fills in four checks
required by `references/technology-selection.md` § CuTeDSL caveats #1:

  1. Capture succeeds (no exception).
  2. Replay output matches an eager reference within tolerance (correctness).
  3. Replay output is deterministic across >= 3 iterations.
  4. Replay does not trigger JIT recompilation / cache misses.

The champion must replace the three TODO blocks (build_inputs, eager_reference,
graph_capture_launch) with calls against their actual CuTeDSL kernel. The rest
of the harness is fixed so reviewers can diff the self-test log against the
acceptance criteria mechanically.

Exit code 0 means all four checks passed. Non-zero means the proposal must
pivot off CuTeDSL for this target (or switch to an AOT cute.compile path and
re-run the self-test against the AOT kernel).
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

import torch


# ---------------------------------------------------------------------------
# TODO (champion): replace these three functions with your kernel's plumbing.
# ---------------------------------------------------------------------------

def build_inputs(device: torch.device) -> dict[str, Any]:
    """Return the input tensors / state needed by both eager and graphed paths.

    Use production-representative shapes and dtypes (from target.json + nsys
    traces). Do not shrink shapes "for speed" — the self-test is cheap.
    """
    raise NotImplementedError("champion: build production-shape inputs here")


def eager_reference(inputs: dict[str, Any]) -> torch.Tensor:
    """Run the proposed CuTeDSL kernel (or its reference) in eager mode.

    This is the ground truth for correctness comparison. If the CuTeDSL kernel
    itself is the reference, run it once outside any graph context here.
    """
    raise NotImplementedError("champion: run the eager reference path here")


def graph_capture_launch(inputs: dict[str, Any]) -> torch.Tensor:
    """Launch the CuTeDSL kernel inside the currently active capture context.

    Called by the capture helper below while a CUDA graph is being recorded.
    Must return the kernel's output tensor (so we can read it post-replay).
    """
    raise NotImplementedError("champion: launch the CuTeDSL kernel here")


# ---------------------------------------------------------------------------
# Fixed harness — do not modify below this line.
# ---------------------------------------------------------------------------


def _count_cute_jit_events() -> int:
    """Best-effort JIT cache-miss counter.

    CuTeDSL's JIT cache surface depends on the DSL version; if the interface is
    not importable we return -1 and the caller treats check #4 as NOT VERIFIED
    (self-test fails). Champions on DSL versions where this is unreliable must
    add an nsys-based fallback in a follow-up check.
    """
    try:
        from cutlass import cute  # type: ignore
        stats = getattr(cute, "compile_stats", None)
        if stats is None:
            return -1
        return int(stats().misses)
    except Exception:
        return -1


def run_selftest(tolerance: float, replays: int) -> int:
    device = torch.device("cuda")
    torch.cuda.synchronize()

    inputs = build_inputs(device)
    reference_out = eager_reference(inputs).detach().clone()

    # Check 1: capture
    graph = torch.cuda.CUDAGraph()
    jit_before_capture = _count_cute_jit_events()
    try:
        with torch.cuda.graph(graph):
            captured_out = graph_capture_launch(inputs)
    except Exception as exc:
        print(f"[FAIL] check 1 (capture): {type(exc).__name__}: {exc}")
        return 1
    print("[PASS] check 1 (capture): CUDA graph recorded without exception")

    # Check 2 + 3: replay correctness + determinism
    replay_outputs: list[torch.Tensor] = []
    jit_before_replay = _count_cute_jit_events()
    for i in range(replays):
        graph.replay()
        torch.cuda.synchronize()
        replay_outputs.append(captured_out.detach().clone())

    for i, out in enumerate(replay_outputs):
        if not torch.allclose(out, reference_out, atol=tolerance, rtol=tolerance):
            abs_err = (out - reference_out).abs().max().item()
            print(
                f"[FAIL] check 2 (correctness): replay {i} diverges from eager "
                f"reference (max abs err {abs_err:.3e} > tol {tolerance:.1e})"
            )
            return 2
    print(f"[PASS] check 2 (correctness): {replays} replays match eager reference within tol {tolerance:.1e}")

    for i in range(1, len(replay_outputs)):
        if not torch.equal(replay_outputs[0], replay_outputs[i]):
            print(f"[FAIL] check 3 (determinism): replay 0 != replay {i} bitwise")
            return 3
    print(f"[PASS] check 3 (determinism): all {replays} replays match bitwise")

    # Check 4: no JIT recompilation during replay
    jit_after_replay = _count_cute_jit_events()
    if jit_before_capture == -1 or jit_before_replay == -1 or jit_after_replay == -1:
        print(
            "[FAIL] check 4 (no re-capture): cannot read cutlass.cute JIT cache "
            "stats on this DSL version. Re-run with an nsys fallback to "
            "confirm zero recompile events during replay."
        )
        return 4
    if jit_after_replay != jit_before_replay:
        print(
            f"[FAIL] check 4 (no re-capture): {jit_after_replay - jit_before_replay} "
            f"CuTe JIT compile events fired during replay (must be zero)"
        )
        return 4
    print("[PASS] check 4 (no re-capture): zero CuTe JIT compile events during replay")

    print("\nAll four acceptance criteria passed. CuTeDSL CUDA-graph self-check: OK.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tol", type=float, default=1e-3, help="allclose atol/rtol (default 1e-3)")
    parser.add_argument("--replays", type=int, default=3, help="replay iterations (default 3)")
    args = parser.parse_args()
    return run_selftest(tolerance=args.tol, replays=args.replays)


if __name__ == "__main__":
    sys.exit(main())
