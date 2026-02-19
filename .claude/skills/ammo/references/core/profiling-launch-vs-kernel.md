# Launch Overhead vs Kernel-Time

Performance regressions/wins can come from different sources. Separate them before deciding what to optimize.

## Signal types

- Host/API launch overhead: command submission, synchronization, orchestration cost.
- Device kernel execution time: actual GPU compute/memory work.

## Why separation matters

- Reducing kernel count helps only if launch overhead is significant.
- Kernel-level optimization is required when device time dominates.
- Mixed bottlenecks require combined strategy.

## Required reporting

For each primary bucket include:
- total GPU kernel-time
- top-kernel contributions
- host/API overhead indicators
- interpretation of dominant bottleneck class
