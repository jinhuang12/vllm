# GPU Configuration Guardrails

Use this as a generic hardware guardrail reference when evaluating optimization feasibility.

## Baseline facts to record

- GPU model and compute capability
- driver and CUDA versions
- memory capacity and major bandwidth limits
- relevant runtime limits (shared memory, registers, occupancy constraints)

## Feasibility checks

Before implementing a candidate, verify:
- expected resource usage fits hardware limits,
- launch geometry is compatible with workload bucket sizes,
- changes do not create obvious occupancy collapse,
- fallback path exists for unsupported envelopes.

## Reporting

For each validated candidate, include:
- target hardware envelope
- known unsupported shapes/modes
- risk notes tied to hardware constraints
