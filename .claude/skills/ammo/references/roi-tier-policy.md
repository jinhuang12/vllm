# ROI Tier Policy (AMMO Stage 6)

Use this policy to decide whether an optimization is worth long-term maintenance.

## Preconditions

Do not apply ROI tiers unless validation gates are complete:

1. Correctness gate: pass (or explicit waiver)
2. Kernel-time gate: pass (or explicit waiver)
3. E2E gate: pass (or explicit waiver)

If any gate is `fail`, recommendation is `reject` regardless of ROI.

## Definitions

- `I_primary`: weighted mean E2E improvement (%) on primary buckets.
- `I_worst`: worst per-bucket E2E improvement (%) in validated envelope.
- `K_worst`: worst per-bucket kernel-time improvement (%) vs incumbent in validated envelope.
- `C`: complexity class (`low`, `medium`, `high`) from rubric below.

Improvement convention:
- Positive is better (`baseline_time > candidate_time`).
- Negative is regression.

## Tier thresholds

- Tier S:
  - `I_primary >= 8.0%`
  - `I_worst >= 0.0%`
  - `K_worst >= 0.0%`
  - no unwaived correctness risk
- Tier A:
  - `4.0% <= I_primary < 8.0%`
  - `I_worst >= -1.0%` (small, justified regression allowed only with envelope restriction)
  - `K_worst >= 0.0%` unless explicitly re-scoped
  - bounded enablement + rollback required
- Tier B:
  - `1.0% <= I_primary < 4.0%`
  - `I_worst >= -1.0%`
  - no correctness regression
  - only acceptable when `C=low` or `C=medium` with explicit justification
- Tier C:
  - `I_primary < 1.0%`, or
  - unwaived correctness/regression risk, or
  - complexity is high relative to benefit

## Complexity rubric

Score each axis `0..2`, sum to `C_score`.

Axes:
1. Code surface:
   - 0: small localized edits
   - 1: moderate multi-file changes
   - 2: large or cross-subsystem changes
2. Runtime coupling:
   - 0: no new dispatch/config coupling
   - 1: one additional coupling point
   - 2: multiple sensitive coupling points
3. Adapter divergence:
   - 0: no adapter-specific forks
   - 1: one adapter-specific path
   - 2: multiple adapter-specific forks
4. Test burden:
   - 0: existing tests cover behavior
   - 1: moderate new tests required
   - 2: large ongoing test burden

Map score to class:
- `0..2 => low`
- `3..5 => medium`
- `6..8 => high`

## Decision matrix

- Ship:
  - Tier S, or
  - Tier A with rollback and bounded envelope
- Ship restricted:
  - Tier A/B with controlled bucket/model envelope and explicit ops sign-off
- Reject:
  - Tier C, or any failing mandatory gate

## Required reporting in `maintenance_decision.md`

Include:
- `I_primary`, `I_worst`, `K_worst`
- tier assignment and why
- complexity score and class with axis breakdown
- ship decision (`ship|ship_restricted|reject`)
- exact envelope and rollback switch

## Guardrails

- Do not hide regressions in averages; always report worst-bucket deltas.
- Do not promote tier based on synthetic or non-parity benchmarks.
- If benefit is narrow, prefer `ship_restricted` with explicit dispatch guards.
