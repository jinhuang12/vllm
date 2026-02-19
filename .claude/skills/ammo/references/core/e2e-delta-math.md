# E2E Delta Math (Upper-Bound Reasoning)

Use this to avoid over-investing in local optimizations that cannot produce meaningful end-to-end gains.

## Definitions

- `T_total = T_target + T_other`
- `f = T_target / T_total` (targeted hotspot share)
- `s = T_target_new / T_target` (`s < 1` means faster)

Then:

```text
T_total_new = s*T_target + T_other
E2E_improvement = 1 - (T_total_new / T_total)
                = f * (1 - s)
```

## Practical interpretation

- `f` is a hard ceiling multiplier on E2E benefit.
- Large local speedup with tiny `f` usually yields small E2E gain.

Example:
- hotspot share `f=0.02` (2%)
- local improvement `1-s=0.50` (50%)
- expected E2E gain `=0.02*0.50=0.01` (1%)

## Usage in AMMO

1. Measure hotspot share from baseline profile.
2. Estimate best-case E2E improvement for each candidate.
3. Reject or infra-track candidates whose expected E2E gain is below significance threshold unless strategic rationale is explicit.
