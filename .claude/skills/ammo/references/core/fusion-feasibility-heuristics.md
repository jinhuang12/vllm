# Fusion Feasibility Heuristics

Use these heuristics to decide whether fusion-style changes are worth implementation effort.

## Inputs

- hotspot time share and call frequency
- memory traffic and intermediate materialization cost
- expected synchronization/barrier overhead
- implementation complexity and rollback burden

## Quick triage

A candidate is low feasibility if:
- expected removable overhead is tiny relative to total time,
- it increases synchronization complexity significantly,
- it risks disabling a larger incumbent optimization.

A candidate is high feasibility if:
- it removes repeated expensive intermediates,
- it preserves occupancy and incumbent fast paths,
- expected gains survive parity constraints.

## Output

Assign per candidate:
- impact score (0-5)
- feasibility score (0-5)
- risk score (0-5)
- rationale with evidence links
