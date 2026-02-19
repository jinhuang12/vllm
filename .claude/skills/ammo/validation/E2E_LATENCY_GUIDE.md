# E2E Validation Guide (Generic)

Use this guide for AMMO Stage 5.3 end-to-end validation.

## Objective

Confirm that a locally validated optimization yields meaningful end-to-end improvement in realistic workloads.

## Requirements

- baseline and candidate runs must use identical parity knobs
- workloads must represent production-relevant scenarios
- enough iterations must be used to reduce noise

## Recommended process

1. Define primary workload buckets.
2. Run baseline and candidate under parity settings.
3. Collect per-bucket latency/throughput metrics.
4. Compute primary aggregate and worst-bucket deltas.
5. Compare observed E2E gain to expected bound from hotspot-share math.

## Minimum report fields

- exact commands
- environment and parity knobs
- per-bucket baseline vs candidate metrics
- aggregate primary-bucket summary
- variance notes
- pass/fail conclusion

## Decision guidance

- if E2E gain is significant and correctness is clean, proceed to ROI decision
- if E2E gain is negligible despite large local speedups, reject or classify as infrastructure investment
