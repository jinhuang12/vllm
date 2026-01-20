# State File Schema

Documentation for the `{artifact_dir}/state.json` file structure and required fields.

## Search Anchors

state.json, verification_run, phase completion, gate status, schema

---

## Base Schema

Every state.json MUST contain:

```json
{
  "target": {
    "model_id": "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8",
    "hardware": "L40S",
    "dtype": "fp8",
    "tp": 1,
    "ep": 1
  },
  "phase": "2_planning",
  "status": "in_progress",
  "last_update": "2026-01-07"
}
```

---

## Phase Completion Requirements

### Phase 1 Completion

state.json MUST contain:
```json
{
  "phase1_completed": {
    "date": "2026-01-07",
    "verification": "PASS",
    "profile_files": ["baseline_bs8.nsys-rep"],
    "key_findings": {
      "fused_moe_kernel_bs8": "XXX µs avg"
    }
  }
}
```

### Phase 4 (Validation) Completion

**BLOCKING**: state.json MUST contain `verification_run.phase4` with `status: "PASS"`.

```json
{
  "verification_run": {
    "phase4": {
      "script": "verify_phase4_gates.py",
      "status": "PASS",
      "date": "2026-01-07",
      "gates": {
        "baseline_is_vllm": "PASS",
        "numerical_comparison": "PASS",
        "production_parity": "PASS",
        "kill_criteria_complete": "PASS",
        "state_json_gates": "PASS"
      }
    }
  },
  "phase4_completed": {
    "date": "2026-01-07",
    "status": "VALIDATED",
    "correctness": {
      "status": "PASS",
      "max_abs_diff": 0.00002,
      "tolerance": "atol=0.02, rtol=0.02"
    },
    "performance": {
      "status": "PASS",
      "methodology": "CUDA graph capture",
      "bs1": {"baseline_us": 183.1, "kernel_us": XXX, "improvement_pct": X.X}
    }
  }
}
```

**CRITICAL**: If `verification_run.phase4.status` is not "PASS", Phase 4 is INCOMPLETE.

---

## Verification Fields

### Why verification_run is Required

1. **Enforcement**: Proves that `verify_phase4_gates.py` was actually run
2. **Traceability**: Records which gates passed/failed and when
3. **Blocking**: Orchestrator MUST check this before proceeding to Phase 5

### Schema

```json
{
  "verification_run": {
    "phase1": {
      "script": "verify_phase1_baseline.py",
      "status": "PASS",
      "date": "2026-01-07"
    },
    "phase4": {
      "script": "verify_phase4_gates.py",
      "status": "PASS",
      "date": "2026-01-07",
      "gates": {
        "baseline_is_vllm": "PASS|FAIL|WARN",
        "numerical_comparison": "PASS|FAIL|WARN",
        "production_parity": "PASS|FAIL|WARN",
        "kill_criteria_complete": "PASS|FAIL|WARN",
        "state_json_gates": "PASS|FAIL|WARN"
      }
    }
  }
}
```

---

## Status Values

| Status | Meaning |
|--------|---------|
| `in_progress` | Phase currently being worked on |
| `pending` | Phase not yet started |
| `completed` | Phase finished successfully |
| `blocked` | Phase failed, cannot proceed |
| `INVALID` | Validation was rejected (requires re-run) |
| `revalidating` | Re-running validation after finding issues |

---

## Blocker Schema

When a phase is blocked:

```json
{
  "status": "blocked",
  "blocker": {
    "description": "Production parity gate failed - TORCH_COMPILE_DISABLE=1 found",
    "severity": "critical",
    "gate": "production_parity",
    "attempts": 1,
    "escalation_needed": false
  }
}
```

---

## Validation Checklist

Before marking Phase 4 complete, verify state.json contains:

- [ ] `verification_run.phase4.status == "PASS"`
- [ ] `verification_run.phase4.gates.production_parity == "PASS"`
- [ ] `phase4_completed.performance.methodology == "CUDA graph capture"` (or similar)
- [ ] No `status: "blocked"` or `status: "INVALID"`
