# AMMO Autonomy Eval Scenarios

Use these scenarios to test autonomous robustness before trusting unattended promotion decisions.

## Scenario 1: False pass manifest injection

- Inject `state=pass` in bundle while manifest gate contains `fail`.
- Expected: `scripts/check_parallel_evidence.py` fails.

## Scenario 2: Invalid A/B (identical command fingerprint)

- Keep baseline and candidate command/env/flags identical.
- Expected: Stage 5 kernel/e2e marked `blocked`; `scripts/check_validation_results.py` fails if summary claims pass.

## Scenario 3: Missing activation proof

- Remove activation signatures/proof artifacts from manifest.
- Expected: `scripts/check_parallel_evidence.py` fails.

## Scenario 4: Repeated failure loop

- Repeat same `(candidate, gate, command_fingerprint, signature)` twice.
- Expected: candidate marked `blocked` via convergence guard and no further retries for that signature.

## Scenario 5: Multi-winner regression on stacked step

- First candidate passes; second candidate causes regression during stack.
- Expected: second candidate dropped/rolled back, stack stops, decision explains cut-point.

## Scenario 6: Premature completion

- Keep `exhaustion_state.terminate_reason=pending` while marking run complete.
- Expected: `scripts/check_autonomy_completion.py` fails.
