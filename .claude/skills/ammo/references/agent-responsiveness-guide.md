# Agent Responsiveness Guide

Shared patterns for handling teammate messages in team-based AMMO agents. Role-specific patterns (e.g., GPU conflict resolution, post-validation waiting) stay inline in agent files.

## How Message Delivery Works

Teammate messages are delivered as new conversation turns. A new turn can only start when your current response ends — i.e., when you stop making tool calls. If you run blocking Bash commands without ending your response, queued messages from your teammate are deferred until you pause. This applies even after a blocking command finishes: if you immediately start another tool call, the message is deferred again.

## Never Block With Sleep Loops

Never use sleep loops to wait for teammates or monitor processes. This blocks message delivery for the entire duration AND prevents delivery even afterward if you immediately chain more commands.

**Wrong:**
```bash
for i in 1 2 3 4 5; do sleep 30; check_status; done
```

**Right:** Check once, send a message, then end your turn:
```bash
check_status_once
```
Then use SendMessage to ask your teammate, and **stop making tool calls** so your turn ends and their response can be delivered.

## Long-Running Commands: Background + End Turn

For benchmarks, sweeps, ncu runs — anything >30 seconds where you don't need the result for your next immediate decision:

```json
{"command": "source .venv/bin/activate && python run_sweep.py ...",
 "run_in_background": true, "timeout": 1800000}
```

After starting the background command, **stop making tool calls** so your turn ends and queued messages can be delivered. You'll be notified when it completes.

## Foreground vs Background

| Use foreground | Use background |
|---------------|----------------|
| Need result before next step | Just monitoring progress |
| <30 seconds | >30 seconds |
| `cmake --build`, `pytest`, quick `nvidia-smi` | E2E sweeps, ncu profiling, model benchmarks |

## Status Checks While Waiting

One status check message per 10 minutes of silence. After sending, end your turn to receive the response. While waiting, do useful non-blocking work (review code, draft reports, pre-compute Amdahl's numbers, scaffold test files).

## References

- `gpu-pool.md` — GPU reservation pattern for commands referenced in examples
