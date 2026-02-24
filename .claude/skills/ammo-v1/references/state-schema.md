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
  "stage": "2_bottleneck_mining",
  "status": "in_progress",
  "last_update": "2026-01-07",
  "max_attempts": 3,
  "current_opportunity_id": null,
  "opportunity_attempts": [],
  "route_decision": {}
}
```

### Iteration Loop Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_attempts` | int | 3 | Maximum optimization opportunities to try before declaring exhausted |
| `current_opportunity_id` | string\|null | null | ID of the opportunity currently being pursued (e.g., `"OP-003"`) |
| `opportunity_attempts` | array | [] | History of all attempted opportunities (see schema below) |
| `route_decision` | object | {} | Current routing decision including kill criteria results |

#### `opportunity_attempts` Entry Schema

Each entry in the `opportunity_attempts` array:

```json
{
  "attempt": 1,
  "opportunity_id": "OP-003",
  "status": "KILLED",
  "kill_criteria_results": {
    "criterion_1": "FAIL: 0% improvement — identical kernel (flash_fwd_splitkv_kernel)",
    "criterion_2": "PASS: no correctness regression"
  },
  "kill_reason": "FlashInfer uses identical flash_fwd_splitkv_kernel — 0% improvement possible",
  "date": "2026-02-15"
}
```

Valid `status` values: `"KILLED"`, `"SHIPPED"`, `"BLOCKED"`, `"IN_PROGRESS"`

#### `route_decision` Schema

```json
{
  "route_decision": {
    "decision": "KILL",
    "opportunity_id": "OP-003",
    "kill_criteria_results": {
      "criterion_1": "FAIL: ...",
      "criterion_2": "PASS: ..."
    },
    "next_action": "pivot_to_OP-002",
    "date": "2026-02-15"
  }
}
```

---

## Stage Completion Requirements

### Stage 1 Completion

state.json MUST contain:
```json
{
  "stage1_completed": {
    "date": "2026-01-07",
    "verification": "PASS",
    "profile_files": ["baseline_bs8.nsys-rep"],
    "key_findings": {
      "target_kernel_bs8": "XXX us avg"
    }
  }
}
```

### Validation Stage (Stage 5) Completion

**BLOCKING**: state.json MUST contain `verification_run.validation` with `status: "PASS"`.

```json
{
  "verification_run": {
    "validation": {
      "script": "verify_validation_gates.py",
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
  "validation_completed": {
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

**CRITICAL**: If `verification_run.validation.status` is not "PASS", the Validation Stage is INCOMPLETE.

---

## Verification Fields

### Why verification_run is Required

1. **Enforcement**: Proves that `verify_validation_gates.py` was actually run
2. **Traceability**: Records which gates passed/failed and when
3. **Blocking**: Orchestrator MUST check this before proceeding to Stage 6

### Schema

```json
{
  "verification_run": {
    "stage1": {
      "script": "verify_phase1_baseline.py",
      "status": "PASS",
      "date": "2026-01-07"
    },
    "validation": {
      "script": "verify_validation_gates.py",
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

## Team Field (Optional)

When using agent teams, state.json MAY include a `team` field for tracking:

```json
{
  "team": {
    "name": "ammo-qwen3-30b-a3b-l40s-fp8-tp1",
    "members": ["lead", "verifier", "planner", "implementer"]
  }
}
```

This field is **informational only**. Verification scripts ignore it, preserving backward compatibility with artifact directories created before team support.

---

## Path Compatibility Note

Verification scripts accept multiple JSON paths for backward compatibility. When writing to `state.json`, use the **preferred path** (first listed). Scripts will also check alternate locations.

### `kill_criteria_results`

| Priority | Path | Notes |
|----------|------|-------|
| Preferred | `route_decision.kill_criteria_results` | Canonical location for current attempt |
| Fallback 1 | `kill_criteria_results` (top-level) | Legacy; accepted for backward compat |
| Fallback 2 | `opportunity_attempts[-1].kill_criteria_results` | Extracted from latest attempt record |

### Validation gates

| Priority | Path | Notes |
|----------|------|-------|
| Preferred | `phase_4_validation.gates` | Legacy name, still accepted |
| Fallback | `verification_run.validation.gates` | Canonical per Stage 5 schema above |

---

## Validation Checklist

Before marking the Validation Stage complete, verify state.json contains:

- [ ] `verification_run.validation.status == "PASS"`
- [ ] `verification_run.validation.gates.production_parity == "PASS"`
- [ ] `validation_completed.performance.methodology == "CUDA graph capture"` (or similar)
- [ ] No `status: "blocked"` or `status: "INVALID"`
