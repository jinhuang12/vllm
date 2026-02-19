# Constraints Template + Quality Bar (AMMO Stage 1)

Use this template to write `{artifact_dir}/constraints.md`.

This is the mandatory Stage 1 artifact. Stage 3 planning is blocked until this file is complete and reflected in `artifact_bundle.json.constraints`.

## Non-negotiable quality bar

Constraints are incomplete if any of these are true:
- target envelope is ambiguous (missing framework/adapter/hardware/model/dtype/topology)
- parity knobs are not explicitly listed
- baseline evidence links are missing or non-reproducible
- hotspot claims are not backed by profiling artifacts
- incumbent path and activation conditions are unknown
- adapter-required constraints fields are missing
- artifact still contains scaffold-only instructions instead of concrete evidence and decisions

## Copy/paste template

```markdown
# Constraints

## 0) Snapshot metadata

- Constraints snapshot ID: `CS-YYYYMMDD-01`
- Stage status: `pending|complete|blocked`
- Updated by:
- Date:

## 1) Target envelope

- Framework:
- Adapter:
- Model/workload target:
- Hardware (GPU/driver/CUDA):
- Dtype / quant format:
- Topology (TP/EP/other partitioning):
- Primary buckets:

## 2) Production parity signature (blocking)

- CUDA graphs mode:
- Compile/runtime mode:
- Serving/scheduler knobs affecting dispatch:
- Other parity-critical flags:
- Parity signature string/hash:

## 3) Baseline evidence manifest

- Baseline command(s):
- Profiling command(s):
- Artifacts:
  - nsys:
  - ncu (optional):
  - benchmark logs/json:
- Repro notes:

## 4) Baseline truth snapshot

- Per-bucket baseline metrics:
- Dominant hotspot groups:
- Launch/API vs kernel split:
- Top kernel/path evidence links:

## 5) Incumbent optimization map

| Incumbent optimization | Activation condition | Current evidence | Interaction risk |
|---|---|---|---|
|  |  |  | preserve/modify/disable |

## 6) Candidate feasibility constraints

- Correctness constraints (numerics/ordering/shape rules):
- Runtime constraints (graphs/allocator/stream/capture restrictions):
- Deployment constraints (envelope, rollback, compatibility):

## 7) Adapter-required appendix (blocking)

Document adapter-specific required fields here.

- Appendix source: `references/adapters/<adapter>.md`
- Required fields checklist:
  - [ ]
  - [ ]
  - [ ]

## 8) Open risks and blockers

- Known unknowns:
- Current blockers:
- Required waivers (if any):

## 9) Stage 1 completion record

- `artifact_bundle.json.constraints.status`: `complete|blocked`
- `artifact_bundle.json.constraints.snapshot_id`:
- `artifact_bundle.json.constraints.evidence_links`:
- `artifact_bundle.json.constraints.adapter_required_fields_complete`: `true|false`
- Reviewer notes on Stage 1 quality:
  - baseline parity evidence sufficient: `yes|no`
  - hotspot and incumbent sections complete: `yes|no`
  - adapter appendix complete: `yes|no`
```
