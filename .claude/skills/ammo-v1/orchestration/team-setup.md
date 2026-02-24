# AMMO Team Setup

How the lead creates and manages the AMMO agent team.

## Team Name Convention

```
ammo-{component}-{model_short}-{hardware}
```

Examples:
- `ammo-moe-qwen3-30b-a3b-l40s`
- `ammo-attention-llama3-70b-h100`
- `ammo-sampling-mistral-a100`

## TeamCreate

```
TeamCreate:
  team_name: "ammo-{component}-{model_short}-{hardware}"
  description: "Kernel optimization for {model_id} on {hardware} ({dtype}, TP={tp})"
```

## Teammate Spawn Commands

Spawn each teammate using the Task tool with `team_name` and `name` parameters. Use **Role Briefings** from `orchestration/task-prompts.md` as the prompt.

### verifier

```
Task:
  name: "verifier"
  team_name: "ammo-{component}-{model_short}-{hardware}"
  subagent_type: "general-purpose"
  description: "AMMO verifier agent"
  prompt: [Role Briefing from task-prompts.md § Verifier Role Briefing]
```

### planner

```
Task:
  name: "planner"
  team_name: "ammo-{component}-{model_short}-{hardware}"
  subagent_type: "general-purpose"
  description: "AMMO planner agent"
  prompt: [Role Briefing from task-prompts.md § Planner Role Briefing]
```

### implementer

```
Task:
  name: "implementer"
  team_name: "ammo-{component}-{model_short}-{hardware}"
  subagent_type: "general-purpose"
  description: "AMMO implementer agent"
  prompt: [Role Briefing from task-prompts.md § Implementer Role Briefing]
```

## Startup Checklist

The lead follows this sequence at the start of every run:

1. **Scaffold artifact directory**:
   ```bash
   python .claude/skills/ammo/scripts/new_target.py \
     --artifact-dir kernel_opt_artifacts/{component}_{model_short}_{hardware}_{dtype}_tp{tp} \
     --model-id "{model_id}" --hardware "{hardware}" --dtype "{dtype}" --tp {tp}
   ```

2. **Discover GPU resources**:
   ```bash
   nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
   ```
   Record GPU info in state.json under `gpu_resources`:
   ```json
   {
     "gpu_resources": {
       "gpu_count": 1,
       "gpu_model": "NVIDIA L40S",
       "memory_total_gib": 44,
       "cuda_visible_devices": "0"
     }
   }
   ```
   Set `CUDA_VISIBLE_DEVICES` consistently for all benchmark tasks to avoid
   accidental cross-GPU scheduling.

3. **Create team**:
   ```
   TeamCreate:
     team_name: "ammo-{component}-{model_short}-{hardware}"
     description: "Kernel optimization for {model_id} on {hardware}"
   ```

4. **Spawn teammates** (all three in parallel using a single message with multiple Task calls):
   - verifier, planner, implementer

5. **Create tasks** using templates from `orchestration/task-dependency-graph.md`:
   - Create T1 first (scaffold, lead-owned)
   - Create T2-T22 with correct `blockedBy` relationships
   - Assign owners per the dependency graph

6. **Update state.json** with team info:
   ```json
   {
     "team": {
       "name": "ammo-{component}-{model_short}-{hardware}",
       "members": ["lead", "verifier", "planner", "implementer"]
     }
   }
   ```

7. **Complete T1** and begin work — teammates pick up unblocked tasks.

## Lightweight Team Option

For simple targets where component semantics are well-understood:

**2-agent team**: lead + worker

The worker handles profiling, planning, and implementation. Use when:
- Model is a known architecture
- Hardware is well-characterized (standard H100/L40S)
- Optimization approach is likely straightforward

The lead still owns gates and state management. The task dependency graph is the same, but all non-lead tasks are assigned to the single worker.

Note: `moe_monokernel_artifacts` is deprecated. Use `kernel_opt_artifacts` for new runs.

## Resume Protocol

When resuming an existing team:

1. **Read team config**: `~/.claude/teams/{team-name}/config.json`
2. **Read task list**: `TaskList` to see current state
3. **Read state.json**: `{artifact_dir}/state.json`
4. **Identify blocked/pending tasks** and take appropriate action
5. **Send messages** to idle teammates to resume work if needed
