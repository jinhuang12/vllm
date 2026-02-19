# Tiling and Occupancy Tuning (Generic)

Use this framework for tuning tile sizes and launch shapes in GPU kernels.

## Objective

Choose tile and launch configurations that improve effective throughput while preserving correctness and stability.

## Inputs

- problem dimensions per workload bucket
- dtype and precision behavior
- memory hierarchy limits (registers/shared/global)
- occupancy and wave quantization behavior

## Process

1. Start from conservative baseline tile config.
2. Sweep one parameter axis at a time (tile M/N/K, stages, warps/CTA).
3. Record occupancy, stalls, and kernel-time for each candidate.
4. Reject configurations that trade tiny speedups for high fragility.
5. Keep only configurations that are stable across target buckets.

## Acceptance rule

A tiling change is viable only if:
- correctness passes,
- kernel-time improves in validated envelope,
- no major regression appears in neighboring bucket sizes,
- fallback exists outside tuned envelope.
