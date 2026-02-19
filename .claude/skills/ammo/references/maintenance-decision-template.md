# Maintenance Decision Template (AMMO Stage 6)

Use this template for `{artifact_dir}/maintenance_decision.md`.

## 0) Decision context

- Target envelope (framework/model/hardware/dtype/TP/EP/buckets):
- Candidate variant ID:
- Incumbent variant ID:
- Candidate set (if parallel): primary / secondary / tertiary IDs
- Autonomy mode (`single_wave|bounded_exhaustion`):
- Bounded-exhaustion terminate reason:
- Production parity knobs:
- Evidence artifacts used:

## 1) Gate outcomes (from Stage 5)

- Correctness gate: pass/fail/waived/blocked
- Kernel-time gate: pass/fail/waived/blocked
- E2E gate: pass/fail/waived/blocked
- Waiver details (if any):

## 2) Quantitative ROI inputs

- `I_primary` (weighted primary-bucket E2E improvement %):
- `I_worst` (worst validated-bucket E2E improvement %):
- `K_worst` (worst validated-bucket kernel-time improvement % vs incumbent):
- Measurement variance notes:

## 3) Complexity evaluation

Axis scores (`0..2` each):

- Code surface:
- Runtime coupling:
- Adapter divergence:
- Test burden:
- Total `C_score`:
- Complexity class (`low|medium|high`):

## 4) Tier assignment

- Assigned ROI tier (`S|A|B|C`):
- Why this tier applies:
- Policy reference: `references/roi-tier-policy.md`

## 5) Recommendation

- Decision: `ship | ship_restricted | reject`
- Rationale:
- Enablement envelope (exact):
- Rollback switch and fallback behavior:
- Operational risks and mitigations:

## 5.1 Multi-candidate promotion record (parallel runs)

| Order | Candidate ID | Local gates | Acceptance gates | Stack result | Notes |
|---|---|---|---|---|---|
| 1 | OP-001 | pass/fail/blocked | pass/fail/blocked | kept/dropped |  |
| 2 | OP-002 | pass/fail/blocked | pass/fail/blocked | kept/dropped |  |
| 3 | OP-003 | pass/fail/blocked | pass/fail/blocked | kept/dropped |  |

- Final promoted set:
- Regression cut-point (if stacking stopped):
- Final shipped mode: single candidate or stacked candidates
- `promotion_history` acceptance hash chain:

## 6) Follow-up actions

1. Implement decision action: `ship | ship_restricted | reject` in deployment config and release notes.
2. Add monitoring/rollback checks tied to validated envelope and worst-bucket risk.
3. Schedule re-validation trigger (runtime upgrade, model change, topology change, or workload drift).
