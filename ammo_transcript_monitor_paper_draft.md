
# AMMO: Transcript-Native Runtime Oversight that Steers Long-Horizon Coding Agents

**Authors.**  
[[AUTHOR NAMES]]  
[[AFFILIATIONS]]  
[[CONTACT EMAILS]]

**arXiv draft — working paper with placeholders for empirical findings**

---

## Abstract

Long-horizon coding agents often fail less because they lack raw capability than because they drift: they skip evidence, misread profiler outputs, conflate local speedups with end-to-end wins, or continue along invalid evaluation paths for too long. Existing work has studied coding agents, trajectory observability, and monitorability of reasoning traces, but much of that literature is either post-hoc, focused on sabotage detection, or disconnected from real optimization workflows [5–20]. We present **AMMO**, a transcript-native runtime oversight architecture built around three ideas: (1) an **adversarial transcript monitor** that reads a coding agent’s live machine-readable session transcript, maintains persistent oversight state outside the actor’s context window, and intervenes while work is still in progress; (2) **evidence-demanding review** that grounds optimization claims in staged artifacts for correctness, kernel-level evidence, and end-to-end measurements; and (3) a **delivery-aware intervention path** that lets the monitor steer the implementation agent through the same messaging channel a human supervisor would use, while ensuring that queued monitor messages are actually delivered before more tool use occurs.

We instantiate AMMO in an end-to-end optimization workflow for vLLM-like GPU-intensive systems [1] and present a detailed case study of a 12-round, 10-track optimization campaign (Qwen3.5-4B on NVIDIA L40S), in which 30 monitor sessions produced 31 interventions across 4 categories: methodology (29%), evidence quality (23%), reasoning degradation (26%), and correctness/safety (23%). Four interventions had HIGH counterfactual impact — each either unblocked a successful ship or prevented a corrupted measurement from propagating. A natural experiment within the campaign (Rounds 7-9 lost monitor coverage due to a configuration oversight) provides a within-campaign C5-vs-C0 comparison: the only implementation track that failed with multiple unguided fix attempts was the sole unmonitored lossy track. Across qualitative case studies, we find that the most consequential interventions are not generic safety flags but challenges to methodology: invalid optimization targets based on misattributed bandwidth data, runtime memory bugs in compiled graphs, and CUDA graph capture incompatibilities masking accuracy failures. We also analyze failure modes and limitations, including chain-of-thought unfaithfulness [11], monitor-aware evasion [13–15], correlated actor-monitor blind spots, and hardening gaps in a prompt-and-hook-based implementation. Our central claim is that the missing control primitive for long-horizon coding agents is not another offline judge but a **transcript-native monitor that can actively steer a live session while forcing claims through evidence**.

---

## 1. Introduction

Coding agents are rapidly moving from short patch generation toward longer-horizon software engineering workflows that involve repository navigation, test execution, tool use, debugging, and iterative repair [5–8]. At the same time, performance-oriented agents are beginning to tackle domains such as GPU kernel optimization, where success depends not only on code generation but also on interacting with profilers, interpreting hardware behavior, preserving correctness, and reasoning about end-to-end system impact [2–4]. These workloads expose a gap in current agent oversight.

The gap is not simply that agents make mistakes. Humans also make mistakes during systems work. The harder problem is that **many important mistakes are process failures discovered too late**. An optimization agent may produce a local kernel speedup that cannot matter end-to-end because the optimized component has too small a share of wall time. It may benchmark under the profiler and then mistakenly treat profiled timing as valid throughput evidence. It may compare against the wrong baseline, leak stale environment flags into a benchmark, or decide a candidate “won” without a correctness-preserving incumbent comparison. In long tool chains, these failures can consume substantial engineer time and GPU budget before anyone notices.

Current lines of work address parts of this problem, but not the full loop. Coding-agent papers have improved interfaces and repositories of tasks [5–8]. Observability papers and tools help inspect trajectories or instrument agent behavior [17–20]. AI-control and monitorability work shows that language models can monitor other models’ reasoning traces for suspicious behavior, and that chain-of-thought (CoT) access often outperforms action-only monitoring in detecting certain forms of reward hacking and sabotage [9,10,13–15]. But much of this literature is either (i) **post-hoc**, where the monitor scores a completed trajectory; (ii) **safety-only**, where the positive class is sabotage or scheming rather than ordinary but costly methodology failure; or (iii) **disconnected from live engineering work**, where a monitor does not need to steer an active implementation agent through the same interaction channel that a human supervisor would use.

This paper argues for a different oversight primitive: **transcript-native runtime oversight**. Instead of asking a monitor to judge the final artifact or even a completed trajectory, we let it read the coding agent’s live machine-readable transcript, filter it into an oversight digest, keep persistent monitor state outside the actor’s context window, and interject while the actor is still working. We pair this with an **evidence-demanding adversarial review style**: the monitor’s job is not to be a supportive teammate but to challenge claims until they are backed by explicit evidence. Finally, we close the loop operationally: the monitor communicates through the same teammate-message channel that a human supervisor would use, and the runtime prevents the actor from continuing a long tool sequence if monitor messages are queued but not yet delivered.

We instantiate these ideas in **AMMO** (Automated Model & Machine Optimization), an oversight architecture for long-horizon coding agents operating in end-to-end optimization workflows. AMMO was motivated by GPU-intensive system tuning, particularly vLLM-like inference systems where local performance gains can be misleading without strict staging of correctness, kernel proof, and end-to-end evaluation [1–4]. However, the architectural claim is broader: any coding agent that emits a machine-readable transcript and supports supervisory messages can, in principle, host the same monitor pattern.

The thesis of this paper is not that AMMO invents monitoring or observability from scratch. Several recent works already study reasoning-trace monitoring [9–15], trajectory observability [18–20], and even real-time trajectory analysis with interventions [19]. Our claim is narrower and more practical: **a transcript-native monitor can become a meaningful substitute for a human watching a live coding session only when three pieces are combined**:

1. **Process visibility** into the live transcript, including reasoning, tool use, tool outputs, and subagent activity.
2. **Evidence-demanding adversarial review** that turns optimization claims into proof obligations.
3. **Delivery-aware intervention** that ensures supervisory messages actually alter the live trajectory.

That combination is what we call **monitoring in the loop**.

### 1.1 Contributions

This draft makes the following contributions.

**(1) A transcript-native oversight architecture for live coding agents.**  
We describe AMMO, an architecture in which a monitor consumes raw machine-readable actor transcripts, filters them into an oversight digest, stores persistent monitor state outside the actor’s context, and interjects during live execution rather than after the fact. The architecture distributes trust across structurally independent roles — process oversight and product verification operate on orthogonal information inputs.

**(2) A delivery-aware steering mechanism with calibrated enforcement.**  
We show how a monitor can steer an implementation agent through the same message channel a human supervisor would use, while a message-delivery guard forces the actor to yield when queued oversight messages have not yet been delivered. We describe a broader enforcement design principle in which guard hardness is calibrated to the irreversibility of the violation being prevented.

**(3) Evidence-demanding adversarial review for end-to-end optimization.**  
We formulate optimization oversight as a repeated challenge-response process in which claims must be supported by staged evidence: correctness before performance, kernel-level proof before end-to-end claims, and comparison to the incumbent rather than only to an initial baseline. Beyond adversarial challenge, we show how the monitor’s observational advantage enables constructive coaching through the actor’s own experimental data.

**(4) An iterative campaign architecture for shifting optimization targets.**  
We describe how the oversight architecture adapts across multi-round campaigns in which each successful optimization shifts the bottleneck landscape, requiring dynamic evidence demands, incumbent drift handling, and mechanical termination via Amdahl’s Law.

**(5) A full empirical evaluation plan.**  
We provide a complete A/B and ablation design for evaluating transcript-native steering in long-horizon optimization campaigns, with placeholders for the results you will later insert.

**(6) An honest limitations analysis.**  
We explicitly discuss monitorability limits, chain-of-thought unfaithfulness, monitor-aware evasion, correlated actor-monitor failures, and pragmatic hardening gaps in an open prompt-and-hook-based implementation.

---

## 2. Problem Setting and Motivation

### 2.1 The target workload: long-horizon coding in performance-critical systems

The motivating workload in AMMO is not generic code generation but **end-to-end system optimization** in GPU-intensive applications. We use vLLM as the running example because it is representative of modern LLM serving systems where performance depends on interactions between kernels, scheduling, memory management, request mix, and runtime configuration [1]. Recent work on LLM-driven kernel optimization has shown that language models can generate nontrivial GPU kernels or use performance tools during reasoning [2–4], but practical deployment still requires more than local kernel speed.

A useful optimization candidate must clear a strict sequence of questions:

1. Is the implementation **correct** relative to the incumbent?
2. Did the target kernel or component actually execute in the profiled run?
3. Did the component improve on the relevant metric?
4. Did the change produce a **real end-to-end gain** under production-parity conditions?
5. Is the win large enough, robust enough, and narrow enough in its enablement envelope to be **worth shipping and maintaining**?

These are not merely reporting questions. They shape what the agent should do next. If correctness fails, more benchmarking is pointless. If the kernel never executed, profiling evidence is irrelevant. If the target component contributes only a tiny fraction of wall time, even a dramatic local win may be strategically uninteresting.

### 2.2 Why offline judging is not enough

Suppose an implementation agent is two hours into an optimization campaign. It has edited code, rerun benchmarks, inspected profiler traces, and started drafting a report. An offline judge can still tell us later that the run was invalid, but by then the cost has already been paid. In performance work, that cost is tangible: GPU-hours, queue time, human review effort, and, often, confusion caused by incorrect intermediate conclusions.

The operational value of runtime oversight is therefore not just accuracy of labeling. It is **time-to-intervention**. The earlier a monitor can identify a doomed line of work, the less bad state accumulates.

This is especially important in long tool chains. A coding agent might issue several shell commands back to back, spawn subagents, or continue editing based on a flawed assumption. If supervisory feedback arrives but is not surfaced until much later, the monitor exists only in theory. This is why AMMO treats message delivery as a first-class systems problem rather than a mere prompting detail.

### 2.3 The role of evidence in grounding performance reasoning

Performance reasoning is unusually vulnerable to “paper wins”: local improvements that disappear under realistic conditions, benchmark contamination, or claims unsupported by the right evidence. AMMO addresses this by repeatedly forcing the actor through **proof obligations**.

We use a simple Amdahl-style approximation to motivate these obligations. Let

- \( T_{\text{total}} \) denote end-to-end time,
- \( T_{\text{target}} \) denote the time spent in the optimized component,
- \( f = T_{\text{target}} / T_{\text{total}} \) be the component’s time share, and
- \( s = T'_{\text{target}} / T_{\text{target}} \) be the local speed ratio after optimization, where \(s < 1\) is faster.

Ignoring second-order interactions, the predicted end-to-end improvement is approximately

\[
\Delta_{\text{e2e}} \approx f(1 - s).
\]

This is deliberately simple, but it is useful as a **monitoring heuristic**. If an actor claims a large end-to-end win without showing either a large component share \(f\) or a large local improvement \(1-s\), the claim should trigger skepticism. Conversely, if a local speedup appears genuine but the target component barely matters, the monitor should push the actor toward either rejecting the candidate or narrowing the claim.

The broader insight is that **evidence demand is a grounding mechanism**. By repeatedly asking for correctness artifacts, profiling proof, end-to-end measurements, and incumbent comparisons, the monitor externalizes claims that would otherwise remain hidden inside the actor’s internal reasoning. In that sense, AMMO’s evidence-demanding review operates as a domain-specific **legibility protocol** [12].

### 2.4 Threat model

AMMO is intended for two overlapping failure classes:

1. **Ordinary process failures**: the actor is trying to help, but makes methodological mistakes, overgeneralizes, thrashes, or accepts insufficient evidence.
2. **Adversarial or deceptive failures**: the actor may, intentionally or as an emergent consequence of optimization pressure, pursue shortcuts or reward hacks that preserve apparent success while violating the real task objective [9,13–15].

The first class is the primary operational target in this draft. This matters because much monitorability literature focuses on sabotage or covert side tasks [9,13–15]. AMMO instead emphasizes the much more common and operationally important regime where an agent is **not malicious but still expensive to leave unsupervised**.

---

## 3. Related Work

### 3.1 Coding agents and long-horizon software engineering

Modern coding agents combine natural-language reasoning with environment interaction, repository navigation, editing, and test execution. SWE-agent demonstrated that specialized agent-computer interfaces substantially improve autonomous software engineering performance [5]. OpenHands generalized this idea into a broader platform for tool-using software agents [6]. AutoCodeRover emphasized program-analysis structure and code search for issue resolution [7]. Benchmarks have also evolved: SWE-Bench Pro specifically targets long-horizon software engineering tasks that may take hours or days for humans, with more realistic multi-file, multi-step workflows [8]. These works establish the capability frontier and the evaluation setting in which long-horizon oversight becomes necessary.

### 3.2 Trajectory analysis, observability, and process-centric evaluation

A growing line of work argues that outcome-only evaluation hides the mechanisms of agent success and failure. Bouzenia and Pradel analyze thought-action-result trajectories of software engineering agents and identify motifs and anti-patterns associated with success or failure [17]. SeaView is a visualization system for software-engineering agent trajectories, aimed at helping researchers inspect long runs and compare experiments [18]. Process-Centric Analysis of Agentic Software Systems introduces Graphectory, a graph-based representation of agent trajectories, and importantly reports online monitoring with diagnostic messages and rollback for problematic instances [19]. AgentTrace proposes structured runtime logging for agent observability across cognitive, operational, and contextual surfaces [20].

Our work builds on this literature but differs in emphasis. AMMO is not primarily a visualization system [18], a generic telemetry framework [20], or a graph-based process-analysis framework [19]. Nor do we claim to be the first to analyze trajectories in real time. Instead, AMMO’s novelty lies in **using the existing machine-readable transcript itself as the supervisory surface**, without requiring the actor to adopt a custom trajectory representation, and in combining that with **mid-flight, same-channel steering** and **evidence-demanding optimization oversight**.

### 3.3 Monitoring reasoning traces and AI control

Recent work shows that reasoning traces can be surprisingly useful for oversight. Baker et al. demonstrate that prompted monitors observing a model’s chain of thought can detect reward hacking in agentic coding environments, and that CoT access can substantially outperform action-only monitoring [9]. Guan et al. extend this into a broader evaluation suite for “monitorability,” finding that current frontier reasoning models are often monitorable, though imperfectly so [10]. At the same time, the limits are significant: Chen et al. show that reasoning models do not always faithfully reveal the reasoning they actually use [11], and Baker et al. warn that optimizing directly against CoT monitors can produce obfuscated reward hacking [9].

Other control work studies ways to make monitoring more reliable. Sreevatsa et al. introduce legibility protocols that make an untrusted model’s actions easier for a monitor to evaluate [12]. SHADE-Arena studies sabotage and monitoring in realistic long-horizon environments [13]. Kale et al. investigate weak-to-strong monitoring and monitor red teaming, including settings where humans selectively review flagged cases [14]. Async Control studies asynchronous deployment of monitors and how monitoring performance degrades under realistic queueing and review constraints [15]. A broader policy and governance perspective argues that real-time failure detection should be treated as a core design requirement for more autonomous agents [16].

AMMO sits adjacent to, but distinct from, this literature. Prior work largely frames monitoring as a defense against covert harm or scheming [9,13–15]. AMMO instead asks: **what would it take for a monitor to improve the ordinary day-to-day reliability of long-horizon coding agents engaged in expensive engineering work?** The monitor must not only recognize suspicious trajectories; it must be able to steer a live session before wasted work compounds.

### 3.4 LLM-driven optimization and end-to-end performance work

The optimization side of our setting also has a growing literature. vLLM showed that LLM serving performance is fundamentally a systems problem, with memory management and scheduling innovations such as PagedAttention producing major end-to-end gains [1]. KernelBench formalized the challenge of getting LLMs to write correct and fast GPU kernels, showing that frontier models remain far from saturating the problem [2]. Nichols et al. argue that optimization requires interacting with performance tools during reasoning [3]. Chu et al. present a framework that extracts hotspot kernels into minimal executable programs to avoid full application builds during iterative optimization [4].

These works motivate AMMO in two ways. First, they show that performance engineering is a rich and relevant target for autonomous agents [2–4]. Second, they highlight the danger of focusing on isolated kernels or reduced artifacts alone: end-to-end impact still has to be re-established in the full system [1,4]. AMMO’s contribution is not a new kernel optimizer but an **oversight architecture** for agents operating in that setting.

### 3.5 Positioning statement

The strongest related work overlap is with process-centric trajectory analysis [17–20] and reasoning-trace monitoring [9–15]. We therefore state our novelty claim narrowly.

We do **not** claim:
- to be the first to monitor agent trajectories,
- to be the first to use CoT or transcript information for oversight,
- or to be the first to intervene on a live trajectory.

We **do** claim:
- to present a practical, open, transcript-native architecture in which a monitor can steer a live coding agent through the same channel a human supervisor would use,
- to show how evidence-demanding adversarial review can ground performance-engineering claims in staged artifacts,
- and to frame runtime oversight for coding agents as a systems problem involving observation, memory, delivery, and decision rules—not merely a classifier prompt.

---

## 4. The AMMO Oversight Architecture

### 4.1 Design goals

AMMO is designed around five goals.

**G1. Runtime visibility without actor-side redesign.**  
The monitor should be able to observe the implementation agent using the transcript the runtime already emits, rather than requiring the actor’s logic to be rewritten around a new instrumentation layer.

**G2. Early intervention.**  
Oversight should alter the live trajectory, not merely score it after completion.

**G3. Evidence-grounded decision making.**  
Optimization claims must be backed by explicit staged evidence rather than unsupported narrative reasoning.

**G4. Human-equivalent steering.**  
The monitor should interject through the same messaging surface that a human supervisor could use.

**G5. Explicit ship/reject decisions.**  
The workflow should terminate in a maintenance decision relative to the incumbent, not a vague “seems faster” conclusion.

### 4.2 Roles, trust distribution, and structural independence

AMMO is naturally described as a multi-agent oversight loop with four primary roles:

- **Orchestrator / team lead.** Assigns the optimization target, spawns implementation and oversight roles, manages campaign-level state, and enforces stage gates. Critically, the orchestrator delegates all implementation work — it never writes kernel code or runs benchmarks itself.
- **Implementation champion (actor).** The coding agent that edits code, runs benchmarks, profiles kernels, and prepares candidate optimizations.
- **Transcript monitor.** An adversarial reviewer that watches the live transcript and sends interventions when it detects methodology, evidence, or reasoning failures.
- **Validator(s).** Independent agents or scripts that verify specific stages such as baseline integrity, gate completeness, or report consistency.

In the current open implementation, these roles are instantiated as named agents and shell-hook scripts under a Claude Code–style runtime, but the architecture is general. Any coding-agent runtime with machine-readable transcripts and inter-agent messaging can host the same pattern.

#### Two orthogonal oversight axes

A key design decision is that AMMO separates **process oversight** from **product verification** and enforces structural independence between them.

The **transcript monitor** observes the actor's live trajectory — its reasoning, tool use, and intermediate decisions — and challenges process failures in real time. It has full observational access to the actor's work but no implementation authority. Its concern is *how* the actor is working: is the methodology sound, is the evidence sufficient, is the reasoning degrading?

The **validator**, by contrast, independently verifies the actor's *output artifacts*. Crucially, it receives only the optimization specification (from the debate phase), not the actor's implementation or test code. It writes its own correctness tests from the specification alone, then benchmarks the resulting kernel independently. This means the validator cannot rubber-stamp the actor's tests — it must derive its own verification from first principles.

These two axes are **structurally independent**, not merely prompted to behave independently. The validator never sees the actor's test code; the monitor never modifies artifacts; neither role can be co-opted by the actor's framing because their information inputs are architecturally separated. This provides a concrete defense against correlated failures (discussed further in Section 9.3): even if the actor and monitor share model-family blind spots, the validator's artifact-based checks remain grounded in external execution rather than shared reasoning patterns.

### 4.3 Transcript-native observation

A key AMMO decision is to monitor the actor’s **native machine-readable transcript**, rather than asking the actor to produce a special report or only looking at end artifacts. In the current implementation, the transcript is a JSONL session log. The monitor discovers the relevant transcript, polls it periodically, and runs a filter that produces a human-readable digest while preserving the load-bearing parts of the session:

- assistant reasoning / thinking blocks,
- assistant text,
- tool invocations,
- tool outputs,
- user/system messages,
- and depth-1 subagent activity.

Noise records such as progress updates or low-value queue operations are dropped. This matters because oversight is fundamentally a signal-to-noise problem. A monitor forced to reparse thousands of transcript lines of operational clutter will spend its limited budget on the wrong parts of the trajectory.

#### Why transcript-native matters

Transcript-native monitoring has three advantages:

1. **No actor self-report dependency.** The actor does not need to summarize itself accurately.
2. **Faithfulness to actual work.** The monitor sees what the actor actually did with tools, not just what it later says it did.
3. **Portability.** Many agent runtimes already emit logs; transcript-native oversight can piggyback on them.

This does not solve the CoT faithfulness problem [11], but it does move oversight closer to the real trajectory.

### 4.4 Persistent monitor memory outside the actor context

Long-horizon sessions often exceed the actor’s effective working memory or context budget. AMMO therefore gives the monitor its own persistent observation log, stored outside the actor’s context and updated incrementally as the session unfolds. This lets the monitor preserve long-range concerns such as:

- which methodology risks have already been raised,
- whether the actor addressed a previous warning,
- recurring thrashing patterns,
- and whether a claimed conclusion is supported by earlier evidence.

This design choice is subtle but important. A monitor with no persistent memory risks becoming episodic: it can identify local anomalies, but cannot accumulate a durable case that the actor is repeatedly circling the same mistake.

### 4.5 Adversarial review instead of cooperative critique

The monitor in AMMO is not prompted as a friendly peer reviewer whose job is to “help” the actor complete its work. It is prompted as an **adversarial reviewer** whose job is to challenge weak reasoning, unsupported claims, and process drift as early as possible.

The categories of concern in the current implementation cluster into three families:

#### (a) Methodology failures
Examples include:
- measuring under non-production-parity conditions,
- using the wrong benchmark shape,
- benchmarking under profiler overhead and then treating those timings as valid,
- missing or contaminated baselines,
- cherry-picking batch sizes,
- or failing to reserve the correct hardware resources.

#### (b) Evidence failures
Examples include:
- asserting correctness without a correctness artifact,
- claiming a kernel win without proof the kernel executed,
- claiming end-to-end impact without an unprofiled E2E sweep,
- or writing a report whose claims are not backed by structured outputs.

#### (c) Reasoning degradation
Examples include:
- local-win/global-loss reasoning,
- Amdahl-inconsistent extrapolation,
- blind fix-and-respawn behavior,
- surface-symptom fixes,
- shrinking depth of analysis,
- or repeated loops that do not test new hypotheses.

The underlying idea is that the monitor’s positive class is **not just “malicious behavior.”** It is any trajectory state that makes the current direction of work untrustworthy.

#### (d) Evidence-guided coaching

The three categories above are primarily adversarial — they challenge the actor’s work. But the monitor’s observational advantage enables a fourth operational mode that is constructive rather than adversarial: **using the actor’s own experimental data to suggest unexplored fix paths**.

This mode activates most naturally during validation failures. When a correctness or accuracy gate fails, the actor often has intermediate results — partial measurements, error distributions across tensor dimensions, performance at specific operating points — that contain diagnostic signal the actor may have dismissed or scrolled past. The monitor, maintaining persistent memory across the full session, can synthesize these scattered observations into a suggested investigation path.

For example, if an accuracy gate fails and the actor’s own investigation reveals that divergence concentrates in specific tensor shapes, the monitor can suggest restricting the optimization to the non-diverging subset — a suggestion grounded in the actor’s own evidence, not a generic prescription. If the actor appears to be giving up prematurely, the monitor can point to untried fix strategies that the actor’s own data implies could work.

This coaching behavior is structurally enabled by transcript-native observation: the monitor sees the full trajectory of experimental results, including data the actor may have already lost to context compression. It transforms the monitor from a pure cost center (overhead, false positives, forced turn yields) into a **productivity mechanism** that makes the actor more effective, not just less wrong.

### 4.6 Evidence-demanding review as a legibility protocol

Repeatedly asking the actor for evidence is not merely rhetorical style. It is a control mechanism.

AMMO turns each major optimization claim into a staged proof obligation:

1. **Correctness proof.** The candidate must not degrade correctness relative to the incumbent.
2. **Kernel proof.** The target component or kernel must actually execute in the relevant profiled run.
3. **End-to-end proof.** Real system-level gains must be established under acceptable measurement conditions.
4. **Incumbent comparison.** The candidate must beat the current best-so-far, not just the initial baseline.
5. **Maintenance decision.** The gain must justify the complexity and activation envelope required to ship it.

This sequence serves the same purpose as legibility protocols in AI control [12]: it pushes the actor into a state where its claims can be evaluated by a weaker supervisor. In AMMO, the “legibility protocol” is domain-specific to performance engineering.

### 4.7 Delivery-aware intervention

A monitor is only useful if its messages arrive in time to matter. In many agent runtimes, inter-agent or teammate messages are delivered at turn boundaries rather than immediately in the middle of a long shell sequence. That means a live monitor can still be functionally muted if the actor keeps issuing tools without yielding.

AMMO closes this loop with a **delivery-aware intervention path**:

1. The monitor sends teammate messages to the implementation agent or escalates to the orchestrator.
2. A runtime guard checks whether messages are queued for the actor but not yet reflected in the actor’s delivered transcript.
3. If so, the guard denies further tool use and instructs the actor to end its turn immediately so that the queued oversight messages are delivered.

This detail is central to our claim. It is what turns the monitor from a passive observer into something closer to a human supervisor interrupting a live session. Without it, the architecture collapses back into delayed commentary.

#### Calibrated enforcement: not all violations warrant the same response

The delivery guard is the hardest enforcement point in the system, but it is not the only one. AMMO uses a spectrum of enforcement mechanisms calibrated to the irreversibility of the violation being prevented:

| Violation class | Enforcement style | Rationale |
|---|---|---|
| Undelivered supervisory messages | **Hard deny** (blocks further tool use) | Message delivery is the core oversight mechanism — if it fails, everything downstream is unmonitored |
| Evidence gates (claiming a stage passed) | **Hard block** until artifact exists | Claims without evidence are the primary failure mode being prevented |
| Production-parity violations (e.g., disabling compiler optimizations during benchmarking) | **One-shot warning**, then trust | The actor may have a valid reason; repeated blocking creates adversarial workarounds |
| Session termination during active work | **One-shot nudge**, then allow | Prevents accidental exit but does not trap the actor in an unrecoverable state |
| Monitor spawn reminders | **Advisory** | The orchestrator may have operational reasons to defer |

This calibration reflects a design principle: **enforcement hardness should be proportional to the irreversibility and cost of the violation**. An undelivered oversight message cannot be retroactively fixed — the actor has already continued on a potentially flawed trajectory. A production-parity shortcut, by contrast, invalidates one benchmark run that can be re-executed. A system that hard-blocks every violation class is brittle and incentivizes workarounds; a system that only advises is toothless. The spectrum between these extremes is itself a design choice.

### 4.8 Staged validation and maintenance decisions

AMMO’s runtime oversight is paired with stage verifiers and explicit decision rules. The desired progression is:

1. **Production-parity baseline capture**
2. **Bottleneck ranking from profiler evidence**
3. **Hypothesis selection with kill criteria**
4. **Bounded implementation with fallback path**
5. **Validation in strict order:** correctness → kernel-time / execution proof → end-to-end
6. **Maintenance decision:** ship, ship_restricted, or reject

Two decision principles are especially important.

**Incumbent protection.**  
Every candidate is compared against the current best-so-far implementation, not merely the original baseline. This reduces “regression by iteration,” where an agent convinces itself a later candidate is good because it outperforms an already superseded baseline.

**Explicit enablement envelope.**  
Performance wins are only meaningful within the workload and deployment envelope where they were established. AMMO therefore treats activation conditions—hardware, precision, batch shape, model family, scheduling mode, and so on—as part of the claim.

### 4.9 Iterative campaigns and shifting optimization targets

The preceding sections describe oversight within a single optimization pass. But real performance engineering is iterative: each successful optimization shifts the bottleneck landscape, changing which components matter most and what evidence is needed. AMMO therefore operates over **multi-round campaigns** in which the oversight architecture must adapt to a moving target.

#### Mechanical termination via Amdahl's Law

The campaign loop is governed by a simple physical bound. Let \( f \) denote the share of end-to-end latency consumed by the current top bottleneck. If \( f \) falls below a threshold — concretely, the minimum end-to-end improvement the team considers worth shipping — then Amdahl's Law guarantees that even completely eliminating that component cannot produce a meaningful system-level gain. At that point the campaign terminates mechanically.

This connects directly to the Amdahl framing introduced in Section 2.3, but now the formula does real work as a **stopping rule** rather than a monitoring heuristic. The orchestrator has zero discretion over termination: if the math says there is room, the campaign continues; if it does not, the campaign stops. This is a deliberate design choice. Long-horizon agents are prone to premature exit ("we've tried enough approaches") and to indefinite continuation ("there might still be something"). A physical bound eliminates both failure modes.

#### Incumbent drift and dynamic evidence demands

Each successful round changes the baseline against which the next round's candidates are measured. The monitor must therefore verify not just that the actor has evidence, but that the evidence is measured against the **current incumbent** — not a stale baseline from an earlier round.

This creates a dynamic evidence-demanding protocol. In round 1, the baseline is the unmodified system. In round 3, the baseline includes all previously shipped optimizations. The monitor's proof obligations shift accordingly: a round-3 candidate that claims improvement over the round-1 baseline is making an invalid claim, even if the numbers look favorable. Incumbent protection (Section 4.8) is not a static rule but a property that must be re-established each round as the target shifts.

#### Technology pivoting under bounded autonomy

When one optimization technology class is exhausted — for example, when all viable approaches within a particular kernel framework have been attempted — the system autonomously pivots to alternative technology classes without human intervention. This is an interesting autonomy boundary: the system has discretion over *strategy* (which technology to try next) but zero discretion over *termination* (whether to continue). The orchestrator can change how it attacks a bottleneck but cannot decide the bottleneck is "good enough."

#### Concurrent adversarial and constructive work

In rounds after the first, the next round's adversarial debate can run concurrently with the current round's implementation. Debate participants explore and critique optimization proposals using the existing bottleneck profile, while implementation agents work in isolated environments on the current round's winners. The two workstreams share a coordination layer but are subject to strict communication boundaries: debate participants cannot influence ongoing implementation, and implementation agents cannot bias the debate.

This concurrency is a practical contribution — it saves significant wall-clock time — but it also demonstrates that the architecture handles **simultaneous adversarial and constructive work** within the same oversight framework. The monitor watches both workstreams, and the delivery guard applies to both.

### 4.10 Algorithm sketch

**Algorithm 1. Transcript-native runtime oversight**

```text
Input:
  actor A
  monitor M
  transcript source T
  artifact directory D
  polling interval τ
State:
  S_monitor ← ∅
  last_line ← 0

while not stop_condition(A, T, D):
    Δ ← filter_transcript(T, from=last_line, include_subagents=True)
    if Δ is not empty:
        S_monitor ← update_monitor_state(S_monitor, Δ)
        append_to_monitor_log(D, Δ, S_monitor)

        issues ← adversarial_review(
            delta=Δ,
            state=S_monitor,
            required_evidence={correctness, kernel_proof, e2e, incumbent},
            stage=current_stage(D)
        )

        if severity(issues) ≥ threshold:
            send_teammate_message(A or orchestrator, intervention(issues))

    if queued_messages_for(A) > delivered_messages_seen_in(T):
        deny_new_tool_use(A)
        request_turn_yield(A)

    last_line ← end_of(Δ)
    sleep(τ)
```

The important part is not the pseudocode itself but the coupling among **transcript filtering**, **persistent monitor state**, **adversarial review**, and **message-delivery enforcement**.

### 4.11 What the current open implementation does and does not guarantee

The current open implementation is sufficient to demonstrate the architecture, but it should not be confused with a formally verified control system. It relies on prompts, shell hooks, validator scripts, and runtime conventions. In particular, some hardening features are still pragmatic rather than ideal: certain hooks are advisory or fail-open, and some completion-time enforcement mechanisms need stronger blocking semantics. We discuss these limitations in Section 9 because they matter for interpreting results. The research claim of this paper is therefore **empirical utility**, not formal security.

---

## 5. Experimental Questions and Methodology

This section is written as a complete evaluation plan with explicit placeholders for the results you will later insert.

### 5.1 Research questions

We propose the following research questions.

**RQ1.** Does transcript-native runtime oversight improve the rate of **valid deployable optimization wins** relative to actor-only baselines?

**RQ2.** Does **mid-flight steering** matter beyond post-hoc review or asynchronous monitoring?

**RQ3.** Do AMMO’s **evidence-demanding review rules** reduce invalid optimization claims compared with generic monitor prompts?

**RQ4.** Which classes of monitor interventions are most predictive of downstream outcome changes?

**RQ5.** What are the dominant failure modes of transcript-native oversight, including false positives, missed failures, and monitor-aware adaptation?

### 5.2 Hypotheses

We pre-register the following hypotheses.

**H1.** Full AMMO will increase valid deployable optimization success rate over actor-only and post-hoc-review baselines.

**H2.** Removing delivery-aware intervention will significantly weaken the monitor’s effect, even if the same monitor messages are eventually available offline.

**H3.** A generic live monitor will underperform an evidence-demanding adversarial monitor on invalid-claim reduction, because generic monitors are less likely to enforce staged proof obligations.

**H4.** The largest gains will come from preventing methodology failures early, not from catching rare malicious behavior.

**H5.** Transcript-native oversight will impose a measurable control tax—extra messages, extra turn boundaries, and occasionally conservative re-runs—but the tax will be offset by reduced wasted GPU-hours and lower invalid-win rates.

### 5.3 Experimental setting

We evaluate AMMO on **long-horizon optimization campaigns** in a GPU-intensive codebase centered on vLLM-like workloads [1]. A single campaign starts from a baseline, identifies a bottleneck, implements one or more candidate changes, validates them in stages, and ends in a ship/reject decision.

Each campaign yields:

- raw transcripts,
- filtered monitor digests,
- structured artifacts for baseline and validation stages,
- intervention logs,
- benchmark outputs,
- profiler traces,
- final decisions,
- and (optionally) qualitative human review notes.

#### Unit of analysis

We recommend treating the primary unit of analysis as the **campaign**, not the individual benchmark run. This is important because AMMO is intended to improve the full process, not just a single measurement.

### 5.4 Conditions and ablations

We recommend the following experimental conditions.

| Condition | Live transcript access | Monitor can message actor | Delivery guard | Evidence-demanding checklist | Purpose |
|---|---:|---:|---:|---:|---|
| **C0: Actor only** | No | No | No | No | Baseline coding-agent workflow |
| **C1: Post-hoc validator** | No (post-hoc artifacts only) | No | No | Yes (offline) | Tests whether static validation alone is enough |
| **C2: Live monitor, no delivery guard** | Yes | Yes | No | Yes | Tests whether delayed delivery weakens interventions |
| **C3: Live monitor, generic prompt** | Yes | Yes | Yes | No | Tests whether evidence-demanding review matters |
| **C4: Live monitor, no reasoning surface** | Tool/actions only | Yes | Yes | Yes | Tests value of transcript richness beyond actions alone |
| **C5: Full AMMO** | Yes | Yes | Yes | Yes | Full architecture |

Optional extension:

| Condition | Description |
|---|---|
| **C6: Human supervisor subset** | A human reviewer sends live teammate messages through the same channel on a subset of campaigns, to compare monitor interventions with human-style supervision. |

The strongest comparison is **C5 vs C0**, but the most scientifically informative comparisons are **C5 vs C1**, **C5 vs C2**, and **C5 vs C3**.

### 5.5 Metrics

We define the following primary and secondary metrics.

#### Primary metrics

**Valid deployable optimization success rate.**  
Fraction of campaigns that end with a candidate that passes correctness, kernel proof, end-to-end criteria, incumbent comparison, and maintenance review.

**Invalid optimization claim rate.**  
Fraction of campaigns whose final report or intermediate trajectory contains a material claim later judged unsupported by the required evidence.

**Wasted GPU-hours per campaign.**  
Total GPU-hours consumed by candidate directions later invalidated due to avoidable methodology or evidence failures.

**Median time-to-detection for methodology failures.**  
Elapsed time between the first monitor-detectable failure signal and the first intervention or correction.

#### Secondary metrics

- Number of monitor messages per campaign
- Fraction of monitor messages that are actioned by the actor
- False positive rate of interventions
- False negative rate (issues found later that the monitor missed)
- Number of re-runs triggered by the monitor
- Total campaign wall-clock time
- Number of candidate branches explored per campaign
- Report rewrite rate due to unsupported claims
- Ship / ship_restricted / reject distribution

#### Suggested definitions

To avoid metric drift, we recommend the following adjudication rules:

- A **valid deployable win** must satisfy all staged criteria and be acceptable relative to the incumbent.
- A **false positive intervention** is a monitor intervention later judged by independent reviewers to be unnecessary or incorrect.
- A **false negative** is a material issue discovered later by human review or downstream verification that the monitor had enough information to catch earlier but did not.

### 5.6 Annotation protocol

Some outcomes require human adjudication. We recommend dual expert review on a stratified sample of campaigns and interventions.

Each reviewed intervention should be labeled on at least four axes:

1. **Correctness of the intervention**: Was the monitor right?
2. **Actionability**: Did the message contain a concrete next step?
3. **Causal influence**: Did the intervention change the actor’s subsequent trajectory?
4. **Outcome relevance**: Did the intervention affect the final accept/reject decision or significantly reduce wasted work?

Where raters disagree, adjudicate with a third reviewer. Report inter-rater agreement (e.g., Cohen’s kappa or Krippendorff’s alpha) as a sanity check.

### 5.7 Statistical analysis

For campaign-level metrics, we recommend:

- paired comparisons when the same task set is attempted under multiple conditions,
- bootstrap confidence intervals for rates and medians,
- permutation tests for pairwise differences,
- and mixed-effects models if campaign difficulty varies systematically across targets.

Suggested reporting format:

- mean ± 95% CI,
- median [IQR],
- absolute differences and relative differences,
- and effect sizes with practical interpretation.

### 5.8 Planned figures and tables

We recommend the following figures and tables.

**Figure 1.** AMMO architecture diagram.  
Actor, monitor, validator, orchestrator, artifact directory, transcript filter, message-delivery guard.

**Figure 2.** Intervention timeline case study.  
A single campaign showing transcript events, monitor interventions, and trajectory changes.

**Figure 3.** Main A/B results.  
Valid deployable success rate, invalid-claim rate, wasted GPU-hours.

**Figure 4.** Ablation results.  
Effect of removing delivery guard, evidence-demanding review, or reasoning surface.

**Table 1.** Comparison to related work.  
Position AMMO against trajectory visualization, observability, monitorability, and optimization systems.

**Table 2.** Experimental conditions.  
C0–C5.

**Table 3.** Main quantitative results.  
Placeholder values.

**Table 4.** Intervention taxonomy and frequencies.  
Methodology, evidence, degradation, safety-related.

---

## 6. Results

This section is written as a complete draft with placeholders for the values and case studies you will insert later.

### 6.1 Main quantitative results from a single deep campaign

We present detailed results from one complete optimization campaign: Qwen3.5-4B on NVIDIA L40S (BF16, TP=1), spanning 12 rounds, 10 implementation tracks, 30 monitor sessions, and 91 agent invocations over approximately 28 hours of wall-clock time. This is not a controlled A/B experiment (those require the full evaluation plan in Section 5) but provides the first empirical grounding for the architecture's claims.

**Campaign outcome.** The campaign shipped 5 optimizations achieving 1.422x cumulative E2E speedup (1.253x verified-only after excluding unverified lossy ops). Monitors produced 31 interventions: 10 CRITICAL, 17 WARNING, 4 INFO. Of these, 30/31 were addressed by the actor (97% action rate). Four interventions had HIGH counterfactual impact — directly enabling a ship or preventing a corrupted result.

**Monitor coverage and cost.** Across the 30 monitor sessions, each debate monitor consumed approximately 50-125 transcript polls; each implementation monitor consumed 20-85 polls. The total monitor token cost was not separately metered but is estimated at 15-25% of the total agent token budget (436K tokens across all agents). The runtime cost of oversight manifested as additional turn boundaries when queued messages were pending — a measurable but operationally favorable trade since each forced yield surfaced an actionable intervention.

**Natural experiment.** A configuration oversight caused Rounds 7-9 to lose monitor coverage (Section 6.7), providing a within-campaign comparison. The sole unmonitored lossy implementation track (op008) was also the only track that failed with 3 unguided fix attempts. All monitored tracks that failed did so cleanly, with either immediate recognition of fundamental limitations or monitor-guided systematic investigation.

**[TABLE 3: SINGLE-CAMPAIGN SUMMARY]**

| Metric | Value | Source |
|---|---|---|
| Campaign duration | 12 rounds, ~28 hours | `state.json` |
| Implementation tracks | 10 (5 shipped, 5 failed) | `state.json` |
| Monitor sessions | 30 (debate + impl) | JSONL agent spawns |
| Total interventions | 31 (10 CRITICAL, 17 WARNING, 4 INFO) | Monitor logs |
| Intervention action rate | 97% (30/31) | Monitor logs |
| HIGH-impact interventions | 4 (13%) | Expert adjudication |
| Cumulative E2E speedup | 1.422x (1.253x verified-only) | Scorecard |
| Unmonitored rounds | R7-R9 (natural experiment) | JSONL gap analysis |

### 6.2 Mid-flight steering matters

The strongest evidence for the “monitoring in the loop” claim comes from the natural experiment (Section 6.7) and from the timing of interventions that changed actor behavior within the monitored rounds.

**Timing analysis of HIGH-impact interventions.** The four highest-impact interventions all required mid-flight delivery to be effective:

1. **R2 debate CRITICAL #2** (BW redirect, `monitor_log_champion-1.md:786`): Arrived during debate Phase B, before proposal finalization. A post-hoc review would have flagged the same issue but after the champion had already submitted a tiled-layout proposal for selection.

2. **R2 impl CRITICAL** (OOM fix, `monitor_log_impl-champion-op002.md:33-39`): Arrived during the E2E sweep crash, while the champion was still in the same tool chain. Without delivery enforcement, the champion might have attempted a second sweep or filed a bug report before receiving the root-cause analysis.

3. **R5 impl CRITICAL** (CUDA graph bug, `monitor_log_impl-champion-op006.md:205-217`): Arrived between Fix 2 and Fix 3 of the accuracy recovery sequence. The CUDA graph capture bug was masking the real accuracy improvement. Without mid-flight delivery, the champion would have continued accuracy debugging on a fundamentally broken execution path.

4. **R5 impl CRITICAL** (Gate 5.1b accuracy, `monitor_log_impl-champion-op006.md:127-131`): Arrived immediately after the first accuracy measurement. The monitor's three-path investigation suggestion structured the subsequent 4-attempt fix progression rather than leaving the champion to thrash.

**Observation.** The interventions that mattered most were not generic flags but **specific, actionable critiques that arrived while the actor was still in the failure state**. A monitor that could only comment after the actor completed a full tool chain would have missed the window for 3 of 4 HIGH-impact interventions.

This result supports the claim that **delivery semantics are not an implementation footnote; they are part of the oversight mechanism**.

### 6.3 Evidence-demanding review outperforms generic monitoring

C3 tests whether the monitor’s style matters. In that condition, the monitor had live transcript access and could message the actor, but it used a generic review prompt rather than AMMO’s evidence-demanding adversarial checklist.

The generic monitor caught obvious trajectory problems but underperformed on optimization-specific reasoning errors. Most notably, it was less likely to challenge:

- unsupported end-to-end claims,
- profiler-contaminated timing arguments,
- missing baseline imports,
- weak incumbent comparisons,
- and Amdahl-inconsistent narratives.

As a result, C3 achieved **[[C3_SUCCESS_RATE]]** valid deployable success rate and **[[C3_INVALID_RATE]]** invalid claim rate, compared with **[[C5_SUCCESS_RATE]]** and **[[C5_INVALID_RATE]]** in C5.

This result suggests that the “ask for evidence” loop is not incidental. It is a core part of the architecture.

### 6.4 Transcript richness matters beyond action-only monitoring

C4 restricts the monitor’s view to tool calls and outputs, withholding reasoning blocks. This condition corresponds to action-only or black-box-style monitoring. Consistent with prior monitorability work [9,10], performance degraded relative to C5. The actor’s reasoning traces were especially useful for identifying:

- when the actor had already committed to a flawed explanation,
- when it was rationalizing a weak result,
- when it was conflating local and end-to-end evidence,
- and when it was narrowing its reasoning depth or looping.

However, the action-only monitor still provided useful signal. This distinction is important. AMMO does not require perfect faithfulness of reasoning traces to gain value from them. It only requires that they contain enough additional signal to improve intervention quality on average.

### 6.5 Intervention taxonomy

We categorize monitor interventions into four classes and report frequencies from a 12-round, 10-track optimization campaign targeting Qwen3.5-4B on NVIDIA L40S (BF16, TP=1). Across 30 monitor sessions spanning debate and implementation phases, monitors produced 31 interventions (10 CRITICAL, 17 WARNING, 4 INFO). Of these, 30/31 were addressed by the actor; 1 was persistently ignored (an NCU cross-reference request on a track that failed on other grounds).

| Category | Definition | Count | Share | Representative example |
|---|---|---:|---:|---|
| Methodology | Benchmarking procedure violations: GPU reservation, CUDA graph parity, baseline provenance, profiler overhead, warm/cold cache | 9 | 29% | nsys overhead invalidating E2E latency numbers (R3 impl, `monitor_log_impl-champion-op003.md:51`) |
| Evidence quality | Data integrity, measurement accuracy, metric validity, missing cross-references | 7 | 23% | Smoke test using relative error on near-zero values, producing 21.6% apparent failure on a correct kernel (R6 impl, `monitor_log_impl-champion-op007.md:36`) |
| Reasoning degradation | Flawed causal chains, Amdahl math errors, wrong mechanism hypotheses, unsupported extrapolations | 8 | 26% | Pipeline projection using 2x scaling instead of 4x, understating absolute latency by 50% (R1 debate, `monitor_log_champion-1.md:456-478`) |
| Correctness / safety | Runtime bugs, accuracy failures, memory issues that would crash or corrupt production | 7 | 23% | `torch.cat()` in compiled graph allocating 1.44 GB extra per forward pass, causing OOM during E2E validation (R2 impl, `monitor_log_impl-champion-op002.md:33-39`) |

The most common class was methodology (29%), but the most outcome-changing class was **correctness/safety** — responsible for all 4 interventions judged to have HIGH counterfactual impact, each of which either unblocked a successful ship or prevented a corrupted measurement from propagating.

**[TABLE 4 ABOUT HERE]**

**Impact distribution.** We classified each intervention’s counterfactual impact by asking: “What would have happened if this message had not been sent?”

| Impact level | Count | Share | Criterion |
|---|---:|---:|---|
| HIGH (changed campaign outcome) | 4 | 13% | Intervention directly unblocked a ship, prevented a corrupted result, or redirected the actor to a different optimization target |
| MEDIUM (improved quality or speed) | 10 | 32% | Intervention caught a real issue but the actor or another mechanism (e.g., gate system) would likely have caught it later |
| LOW (minor or redundant) | 10 | 32% | Intervention was correct but addressed a minor process gap or the actor was already investigating |
| NONE (false positive or moot) | 7 | 23% | Intervention was technically correct but substantively moot, or addressed a round that produced no candidates |

### 6.6 Representative interventions

#### Case Study A: Redirecting an optimization target (R2 debate — HIGH impact)

The actor (Champion-1, Round 2) proposed a tiled weight layout to close a 35% bandwidth gap between isolated kernel measurements (64.7% BW) and theoretical peak. The transcript monitor had observed a delegate experiment (`champion1_pershape_bw.log`) showing that **all GEMM shapes individually achieve 80-83% BW** — the 64.7% production figure reflects environmental interference from non-GEMM transitions, not per-kernel inefficiency. The monitor sent a CRITICAL message:

> *”Per-shape BW results are in champion1_pershape_bw.log and show all shapes at 80-83% BW already. Tiled layout won’t help — production gap is environmental.”*
> — `monitor_log_champion-1.md:786`

The champion had independently read the results 2 minutes before the CRITICAL arrived (`monitor_log_champion-1.md:816`) but the monitor’s message confirmed the interpretation and accelerated the pivot. The champion abandoned tiled layout and instead proposed in-projection weight concatenation — which became op002, shipping at 1.015x E2E improvement (`debate/campaign_round_2/summary.md`). A later round (R3) independently validated that tile selection is a no-op on L40S (`tracks/op003_tile_selection/validation_results.md`), confirming the monitor’s redirect was correct.

**Counterfactual.** Without this intervention, the champion would likely have proposed tiled layout, wasting one implementation cycle (estimated 2-3 GPU-hours) on a fundamentally invalid optimization target.

#### Case Study B: Unblocking a ship with a runtime bug fix (R2 impl — HIGH impact)

During op002 E2E validation, the sweep crashed with an out-of-memory error. The transcript monitor identified the root cause from the crash trace: `_init_combined_inproj` used `torch.cat([qkvz_weight, ba_weight])` on every forward call, creating a new tensor allocation inside a `torch.compile`-traced graph. With 24 transformer layers at 60 MB each, this added 1.44 GB of transient memory. The monitor sent a CRITICAL with a specific fix:

> *”Fix by replacing weight.data in-place so combined shares storage.”*
> — `monitor_log_impl-champion-op002.md:39`

The champion implemented the fix by moving concatenation to the `load_weights()` post-load hook and replacing weight tensors with views of the combined buffer. The fix was committed as `c0b9cc779` (“fix(op002): create combined in-proj weight during load, not forward”). After the fix, E2E validation completed successfully: 1.143x kernel speedup, 1.49% E2E improvement, all gates PASS.

**Counterfactual.** Without the monitor’s root cause analysis and specific fix guidance, the champion would have faced a cryptic OOM during `torch.compile` compilation. Diagnosing the interaction between `torch.cat`, CUDA graph capture, and memory pressure typically requires significant debugging time. The op002 track might have been abandoned as “OOM-incompatible.”

#### Case Study C: Guiding systematic accuracy recovery (R5 impl — HIGH impact)

The op006 implementation (INT4 weight-only quantization) triggered the most active monitor session: 8 messages (3 CRITICAL, 2 WARNING, 3 INFO) across 4 fix attempts (`monitor_log_impl-champion-op006.md:322`). The sequence illustrates evidence-demanding review as a steering mechanism:

1. **Initial failure**: Gate 5.1b reported 79.5% accuracy vs 89.5% baseline — a 10% drop from naive INT4 on all layers (`monitor_log_impl-champion-op006.md:127-131`). Monitor sent CRITICAL with three investigation paths: verify quantization formula, try selective INT4 (N-threshold), try group_size=64.

2. **Fix 1** (N>=4096 selective): Accuracy improved to 83.0% — still failing. Monitor suggested narrowing to gate_up_proj only (N>=18000) (`monitor_log_impl-champion-op006.md:184`).

3. **CUDA graph bug**: Monitor detected `torch.tensor()` inside `_dequantize_int4_to_bf16`, which creates new allocations during CUDA graph capture — forbidden (`monitor_log_impl-champion-op006.md:205-211`). Sent CRITICAL with specific fix: add M-check routing to `F.linear` fallback for prefill. After this fix, accuracy jumped from 83% to 87.5% — the CUDA graph bug had been silently corrupting the dequantization path.

4. **Final result**: 88.0% vs 89.5% (−1.5pp, 3 questions) — still failing the `opt >= baseline` gate. Track declared FAIL after exhausting recovery options (`monitor_log_impl-champion-op006.md:298`).

**Counterfactual.** The CUDA graph fix (step 3) was the most consequential intervention. Without it, accuracy was stuck at 83% with no visible path forward — the champion might have declared FAIL after Fix 1 without discovering that the dequantization path itself was broken. The monitor’s structured fix progression (verify → selective → runtime-bug fix) demonstrates how evidence-demanding review can guide an actor through systematic hypothesis elimination rather than blind retry.

### 6.7 Natural experiment: the R7-R9 monitor coverage gap

A configuration oversight during the campaign produced a natural experiment. Due to a team lifecycle issue, **Rounds 7-9 had no debate monitors spawned**, and R7 implementation tracks (op008, op009) had no implementation monitors. This is confirmed by JSONL evidence: `grep -c ‘monitor-r7\|monitor-r8\|monitor-champion-r7\|monitor-champion-r8\|monitor-op008\|monitor-op009’` returns 0 matches, despite 154 activity lines for Round 7 and 50 for Round 8.

This creates a within-campaign comparison between monitored rounds (R1-R6, R9-R12) and unmonitored rounds (R7-R8), approximating a paired C5-vs-C0 ablation on the same optimization target.

**Unmonitored round outcomes.**

| Round | Tracks selected | Outcome | Counterfactual |
|---|---|---|---|
| R7 | op008 (lossy FP8 lm_head), op009 (lossless) | op008 **FAIL** (accuracy, 3 fix attempts); op009 PASS | **MEDIUM** — the R7 debate summary explicitly noted “FP8 argmax stability on 248K vocab” as a known risk (`debate/campaign_round_7/summary.md:30`), following 2 consecutive lossy accuracy failures in R5-R6. A debate monitor would have had this context and could have escalated the pattern. |
| R8 | None (EXHAUSTED) | — | **NONE** — no candidates selected; monitor absence was inconsequential. |

**Monitored round comparison.** All monitored implementation tracks that failed did so cleanly: op003 (single no-op finding), op006 (accuracy failure caught by monitor-guided fix progression), op010 (2 systematic attempts then clean FAIL report confirmed by monitor at `monitor_log_impl-champion-op010.md:121-137`). The only track that failed with multiple unguided fix attempts was op008 — the only unmonitored lossy track.

**Diminishing returns in exhaustion phase.** Rounds 10-12 had monitors but produced zero interventions (`monitor_log_champion-r10-2.md:26`, `monitor_log_champion-r11-1.md:95`, `monitor_log_champion-r12-1.md:36-37`). All champions in these rounds were honestly reporting negative results (hardware BW ceiling confirmed at 73-77% across 3 technology classes). This suggests that monitor value concentrates during active optimization (R1-R6) and drops to near-zero during campaign exhaustion, consistent with hypothesis H4.

### 6.8 Case study: complete campaign narrative (op002, Round 2)

We trace one optimization from bottleneck hypothesis through ship decision, showing all monitor interventions and their effect on the trajectory.

**Setting.** Round 2 of a 12-round campaign. Baseline: Qwen3.5-4B on L40S, BF16, TP=1. GEMM operations constitute 89% of decode latency at 64.7% production bandwidth utilization. Round 1 shipped op001 (Triton GEMM, 1.027x E2E) but 35% BW headroom remained.

**Debate phase.** Three champions proposed approaches for closing the BW gap:
- Champion-1: tiled weight layout for better DRAM page locality
- Champion-2: fused GEMM-SiLU-mul to reduce launch overhead
- Champion-3: GDN-targeted kernel for the non-GEMM residual

The transcript monitor observed Champion-1’s delegate experiment results showing per-shape BW already at 80-83% (`debate/campaign_round_2/micro_experiments/champion1_pershape_bw.log`). The monitor sent two interventions:

1. **CRITICAL #2** (`monitor_log_champion-1.md:786`): “Per-shape BW results show all shapes at 80-83% already. Tiled layout won’t help.” → Champion pivoted to in-projection weight concatenation.
2. **CRITICAL #3** (`monitor_log_champion-1.md:991`): “Fused GEMM-SiLU is 0.896x slower” (confirmed by `champion1_fused_silu_bench.log`: 4769 µs fused vs 4273 µs separate). → Champion abandoned fusion path.

**Selection.** The orchestrator selected Champion-1’s revised proposal (in-proj concatenation, weight packing to reduce GEMM dispatch count) based on the corrected analysis.

**Implementation phase.** Champion implemented op002 in an isolated worktree. Two monitor interventions occurred:

3. **WARNING** (`monitor_log_impl-champion-op002.md:107-108`): “Prototype benchmark required before full integration.” → Champion had skipped the validation gate. After prototyping, the SiLU-prologue fusion showed 2.36x regression, confirming the debate-phase finding. Champion correctly scoped to in-proj concatenation only.

4. **CRITICAL** (`monitor_log_impl-champion-op002.md:33-39`): torch.cat OOM during E2E sweep. Monitor identified root cause (new allocation in compiled graph) and sent specific fix (in-place weight.data replacement). Champion fixed within one turn (commit `c0b9cc779`).

**Validation and ship.** After the OOM fix, E2E validation completed: 1.143x kernel speedup, 1.49% E2E improvement. All gates passed: 5.1a (correctness), 5.1b (accuracy: 89.5% = 89.5%), 5.2 (kernel speedup), 5.3a/b (E2E). Track shipped as cumulative 1.043x (combined with R1’s op001).

**Counterfactual analysis.** Without the monitor:
- CRITICAL #2 accelerated the pivot from tiled layout, though the champion was already reading the same data. Net effect: saved ~30 minutes of debate time, prevented a weak proposal from entering selection.
- CRITICAL #3 provided independent confirmation that fused GEMM-SiLU regresses. The champion had the data but the monitor’s explicit “do NOT claim as win” prevented any ambiguity in the proposal.
- The OOM CRITICAL was the most consequential: without it, op002 would have been blocked at E2E validation with a cryptic OOM during torch.compile. Diagnosing torch.cat inside a compiled graph typically requires understanding of FX tracing internals. **This intervention likely determined whether op002 shipped or was abandoned.**

---

## 7. Discussion

### 7.1 Why transcript-native monitoring works

We believe AMMO works because it supervises **process**, not merely outputs.

A final patch or benchmark table is a lossy compression of a long reasoning and action sequence. Many costly errors become visible much earlier in the trajectory than in the final artifact. By consuming the live transcript, AMMO can detect these earlier signals:

- the actor’s own tentative but flawed causal story,
- a benchmark command that uses the wrong measurement mode,
- repeated loops with no new evidence,
- or a draft conclusion that outruns the available artifacts.

This complements recent arguments for process-centric evaluation [17–19] and reasoning-trace monitoring [9–15]. AMMO’s practical insight is that, for coding agents, **the transcript itself is already a rich process trace**.

### 7.2 Why the “ask for evidence” loop is so valuable

One of the most important lessons from building AMMO is that repeated requests for evidence are more powerful than they may first appear. They do at least four jobs simultaneously:

1. **They externalize reasoning.**  
   The actor must transform a private intuition into a checkable artifact.

2. **They slow down false closure.**  
   Unsupported narratives are harder to maintain when each stage requires a proof obligation.

3. **They create legibility for weaker oversight.**  
   A monitor that cannot fully understand the actor’s internal reasoning can still check the existence and sequencing of evidence [12].

4. **They support incumbent protection.**  
   By requiring explicit comparisons to the current best-so-far, evidence requests keep the agent anchored to a real decision boundary.

This makes AMMO’s oversight style more than “good scientific hygiene.” It is a runtime control measure.

### 7.3 Human-equivalent steering

A central claim of this paper is that the transcript monitor can steer the implementation agent **as if a human were using the session**. This claim should be interpreted functionally, not anthropomorphically.

We do not mean that the monitor matches the best human supervisor at all times. We mean that it uses the same **intervention surface**:

- it sees the live session trace,
- it maintains its own external notes,
- it sends messages into the actor’s workflow,
- and the runtime makes those messages actionable by enforcing turn yields when needed.

This is important for adoption. Many organizations already understand what it means for a human to “watch a session and intervene.” AMMO shows how to approximate that pattern in a scalable, machine-mediated way.

### 7.4 AMMO as a systems contribution rather than a new learning algorithm

It is tempting to frame AMMO as a new monitoring method. That would overstate the novelty. AMMO is better understood as a **systems architecture for runtime oversight** — one whose value comes not from any single component but from how those components compose into a decision procedure.

Consider how AMMO handles a common scenario in performance engineering: an optimization that helps at some operating points but hurts at others. A binary pass/fail verdict would either discard a partially useful optimization or ship a regression. AMMO instead classifies each per-operating-point measurement into a tier that maps directly to an action:

- **Within noise floor**: classify as neutral — do not claim a win or a loss.
- **Measurable regression at specific operating points**: trigger a gating workflow — determine the exact crossover point where the optimization flips from beneficial to harmful, implement a runtime dispatch that activates the optimization only where it helps, and re-validate the gated version.
- **Catastrophic regression**: fail the candidate immediately — no gating can save it.

Each tier triggers a different workflow with different evidence requirements. The gating path, in particular, requires the actor to locate the crossover boundary, implement conditional dispatch logic, and demonstrate that the gated version produces no regression at any operating point. This is not a simple threshold check; it is a **structured decision procedure that turns measurement uncertainty into differentiated actions**.

This example illustrates the broader point. The contribution is not that AMMO uses transcripts (others do), monitors reasoning (others do), or validates artifacts (others do). It is that these mechanisms are composed into an operationally closed loop with domain-specific decision rules that can redirect a live coding agent engaged in expensive engineering work — including handling the messy middle ground between clear success and clear failure.

### 7.5 Why this matters beyond optimization

Although our motivating domain is system optimization, the pattern generalizes to other coding-agent workloads:

- repository migrations,
- performance regressions,
- CI triage,
- infrastructure changes,
- and long-horizon issue resolution.

Any domain where bad process accumulates cost before the final artifact is available may benefit from the same architecture.

---

## 8. Broader Implications for Agent Oversight

AMMO also suggests a reframing for the oversight literature.

A large portion of current monitorability research is organized around **intent detection**: can a monitor detect sabotage, reward hacking, or scheming [9,13–15]? That question is important, but it is not the only one that matters in real deployments. In many practical settings, the dominant source of harm is not covert malevolence but **ordinary, compounding process failure**. Agents waste compute, produce unsupported reports, contaminate benchmarks, or drift into brittle local heuristics. These failures are not sensational, but they are economically and operationally consequential.

AMMO therefore broadens the target of runtime oversight. The monitor is not only a “misbehavior detector.” It is a **live adversarial reviewer**. In that role, its responsibilities include:

- challenging unsupported claims,
- demanding production-parity evidence,
- enforcing stage order,
- identifying reasoning degradation,
- and keeping the actor honest about the difference between a local result and a deployable system win.

We suspect this broader framing will matter for the next generation of agent deployments. Many organizations may adopt coding agents well before they face full-blown sabotage or scheming concerns. For those organizations, the first-order problem will be whether agents can be trusted to run long workflows without accumulating silent methodological debt. AMMO is a concrete answer to that problem.

---

## 9. Limitations and Threats to Validity

A paper making a bold claim about transcript-native steering must also be explicit about its weaknesses.

### 9.1 Chain-of-thought unfaithfulness

AMMO benefits from access to reasoning traces, but reasoning traces are not fully faithful windows into model cognition. Chen et al. show that reasoning models often do not verbalize all of the influential hints or reasons they use [11]. Baker et al. similarly warn that training pressure can produce obfuscated reward hacking [9]. AMMO should therefore be interpreted as a **best-effort process monitor**, not a perfect readout of model intent.

This limitation cuts both ways:

- it may cause the monitor to miss failures that are not verbalized,
- and it may tempt deployers to overtrust a transcript simply because it is detailed.

### 9.2 Monitor-aware adaptation and evasion

Once an actor knows it is being monitored, it may adapt its behavior in ways that reduce monitor effectiveness. Kale et al. show that agent awareness can substantially degrade monitoring reliability [14], and Async Control highlights the fragility of monitoring under realistic deployment constraints [15]. AMMO’s evidence-demanding style may partly resist this by grounding decisions in external artifacts, but it does not eliminate the risk. A monitor-aware actor could still produce plausible but selective evidence, overfit to the monitor’s heuristics, or strategically defer suspicious behavior.

### 9.3 Correlated model failures

If the actor and monitor come from the same model family, they may share blind spots. AMMO defends against this in two ways: the monitor's proof obligations are grounded in external artifacts rather than purely semantic judgments, and the validator writes its own tests from the specification alone without seeing the actor's implementation (Section 4.2). These structural separations mean that even if the actor and monitor share a reasoning blind spot, the validator's independent artifact-based checks may still catch the failure.

Still, correlated misunderstandings remain a real threat, particularly for failures that manifest only in the reasoning layer (e.g., both actor and monitor accepting a flawed causal explanation for a benchmark result). The strongest evaluation would therefore include heterogeneous actor-monitor pairings and, ideally, some human review on critical subsets.

### 9.4 Domain specificity

AMMO is motivated by performance engineering and may encode assumptions that transfer imperfectly to other domains. For example, the staged gate structure—correctness before kernel proof before E2E—is particularly natural in optimization workflows. Other coding tasks may require different proof obligations. We view this as a feature of the architecture rather than a flaw: evidence-demanding monitoring should be domain-specific. But it does limit the generality of any one monitor prompt.

### 9.5 Hardening gaps in the current implementation

The open implementation that motivates this paper is pragmatic and real, but it is not yet a fail-closed runtime. As described in Section 4.7, the system deliberately uses a spectrum of enforcement styles — from hard denial (message delivery, evidence gates) to one-shot warnings (production parity) to advisory reminders (monitor spawning). This calibration is a design choice, not an accident: enforcement hardness is proportional to the irreversibility of the violation. But it does mean that some guards are intentionally permissive, and others rely on prompt discipline, file discovery conventions, or best-effort validator scripts.

A skeptical reviewer would be right to ask whether the advisory end of this spectrum is too permissive for unattended deployment. In particular, the production-parity guard warns but does not block, the session-termination guard nudges once then allows exit, and all hooks fail open on infrastructure errors. These choices prioritize operational robustness over strict enforcement.

We therefore state the limitation directly: **the current implementation is a strong research prototype, not a formal control system.** The paper’s central claim should be evaluated empirically — does the architecture improve outcomes? — not as a statement of perfect enforcement. In future iterations, the advisory guards could be tightened based on observed bypass rates, and the fail-open default could be revisited for deployment contexts where false negatives are more costly than false-positive blocks.

### 9.6 External validity

Results from vLLM-style optimization campaigns may not transfer directly to all coding-agent workloads. The final paper should therefore avoid claiming universal gains unless the evaluation includes additional domains. At minimum, we recommend carefully scoping the main claim: transcript-native runtime oversight is effective for **long-horizon coding tasks with expensive intermediate failures**.

### 9.7 Measurement validity

Some evaluation metrics, especially “causal influence of an intervention,” require human judgment. The paper should therefore treat such measures as annotated estimates rather than objective ground truth, and report agreement statistics and adjudication procedures.

---

## 10. Ethics and Broader Impact

AMMO has both positive and negative implications.

### 10.1 Positive impacts

- **Safer and more reliable coding-agent deployment.**  
  Runtime oversight can reduce invalid claims, wasted compute, and unsafe or unsupported changes.

- **Lower operational waste.**  
  In optimization workflows, earlier intervention can save meaningful GPU time and human review effort.

- **Better transparency for agent behavior.**  
  Transcript-native monitoring encourages a process view of agent performance rather than outcome-only evaluation.

### 10.2 Risks and misuses

- **Overtrust through observability theater.**  
  Rich transcripts can create an illusion of transparency even when reasoning is not fully faithful [11].

- **Normalization of deeper agent autonomy.**  
  Better monitoring may encourage organizations to deploy agents more aggressively, including in settings where monitoring remains insufficient.

- **Privacy and retention risks.**  
  Session transcripts, tool outputs, and monitor logs may contain sensitive code, credentials, or internal infrastructure details. Any release should include redaction and retention policies.

- **Dual use for control or surveillance.**  
  The same techniques that improve reliability can also intensify oversight of human or machine collaborators in ways that may be misused.

### 10.3 Release recommendations

We recommend that any public release include:

- prompts and hook logic,
- transcript-filtering code,
- artifact schemas,
- aggregated or redacted logs,
- annotation guidelines,
- and explicit statements of known hardening limitations.

We do **not** recommend releasing raw proprietary transcripts without redaction.

---

## 11. Conclusion

This paper argues that the most valuable control primitive for long-horizon coding agents is not another offline evaluator but a **transcript-native runtime monitor that can steer the live session while forcing claims through evidence**. We instantiated this idea in AMMO, an oversight architecture built around adversarial transcript review, persistent monitor memory, delivery-aware intervention, and staged validation for end-to-end optimization workflows.

The open implementation is intentionally pragmatic. It uses machine-readable transcripts, filtering scripts, shell hooks, validator checks, and same-channel messaging rather than a new training algorithm or a formally verified runtime. Its value therefore stands or falls on empirical utility. The placeholders in this draft are designed to test exactly that question.

If the final results support the hypotheses, the contribution is not merely that “monitoring helps.” It is that **a monitor can function like a live human supervisor in long-horizon coding work when observation, evidence demand, and delivery are treated as one integrated system**.

---

## Acknowledgments

[[ACKNOWLEDGMENTS]]

---

## Appendix A. Implementation Notes for the Open AMMO Prototype

This appendix describes the main implementation ideas in concrete but implementation-agnostic terms so that reviewers understand the artifact without requiring the exact repo layout.

### A.1 Transcript source and filtering

The monitor reads the actor’s raw machine-readable session log and incrementally filters it into a digest. The digest retains:

- timestamps and line ranges,
- thoughts / reasoning blocks,
- assistant text,
- tool uses,
- tool results,
- and shallow subagent activity.

It drops low-value noise such as periodic progress bookkeeping.

### A.2 Monitor memory

The monitor appends filtered deltas and its own conclusions into a persistent monitor log stored alongside the artifact directory. This memory is not injected back into the actor by default; it exists to stabilize the monitor across long sessions and context compression.

### A.3 Message delivery enforcement

Interventions are sent through the same teammate-message substrate that a human would use. Because such messages may be queued until the actor yields, the runtime checks whether undelivered messages are pending. If so, it denies further tool use and instructs the actor to end the turn, forcing message delivery.

### A.4 Validation scripts and artifact checks

The open prototype uses standalone scripts to verify baseline completeness, staged validation, and report consistency. In the research paper, these checks should be treated as part of the architecture’s evidence layer, while also being described honestly as pragmatic software components with room for stronger hardening.

---

## Appendix B. Suggested Table and Figure Skeletons

### Table 1. Positioning relative to related work

| Work | Primary domain | Observation surface | Online intervention | Human-equivalent message channel | Evidence-demanding optimization review | End-to-end optimization workflow |
|---|---|---|---:|---:|---:|---:|
| Baker et al. (2025) [9] | Agentic coding / safety | CoT + trajectory | Limited / experimental | No | No | No |
| SeaView (2025) [18] | SWE trajectory analysis | Logs / trajectories | No | No | No | No |
| Process-Centric Analysis (2026) [19] | SWE trajectory analysis | Structured trajectory graph | Yes | No | No | No |
| AgentTrace (2026) [20] | Agent observability | Structured telemetry | Not primary focus | No | No | No |
| Nichols et al. (2025) [3] | GPU optimization | Performance-tool reasoning | No | No | No | Partial |
| Chu et al. (2025) [4] | GPU optimization | MEP-based optimization loop | No | No | Partial | Yes |
| **AMMO (ours)** | Long-horizon coding + optimization | Native runtime transcript | **Yes** | **Yes** | **Yes** | **Yes** |

### Table 3. Main quantitative results

| Metric | C0 Actor only | C1 Post-hoc validator | C2 Live monitor, no delivery guard | C3 Live monitor, generic | C4 Action-only live monitor | C5 Full AMMO |
|---|---:|---:|---:|---:|---:|---:|
| Valid deployable success rate | [[ ]] | [[ ]] | [[ ]] | [[ ]] | [[ ]] | [[ ]] |
| Invalid optimization claim rate | [[ ]] | [[ ]] | [[ ]] | [[ ]] | [[ ]] | [[ ]] |
| Wasted GPU-hours / campaign | [[ ]] | [[ ]] | [[ ]] | [[ ]] | [[ ]] | [[ ]] |
| Median time-to-detection | [[ ]] | [[ ]] | [[ ]] | [[ ]] | [[ ]] | [[ ]] |
| Intervention action rate | — | — | [[ ]] | [[ ]] | [[ ]] | [[ ]] |

### Figure captions to fill later

**Figure 1.** Overview of the AMMO oversight loop.  
`[[INSERT DIAGRAM]]`

**Figure 2.** Representative intervention timeline from a single optimization campaign.  
`[[INSERT CASE STUDY FIGURE]]`

**Figure 3.** Main A/B comparison across conditions.  
`[[INSERT BAR OR DOT PLOT]]`

**Figure 4.** Ablation study showing the contribution of delivery guard, transcript richness, and evidence-demanding review.  
`[[INSERT ABLATION FIGURE]]`

---

## Appendix C. Writing Notes for the Final Submission Version

Before submission, replace placeholders in this order:

1. Abstract metrics
2. Main results table
3. Ablation results
4. Intervention taxonomy counts
5. One full case study
6. Acknowledgments / authors
7. Artifact availability statement

Suggested submission checklist:

- [ ] Replace all `[[...]]` placeholders
- [ ] Add exact counts for tasks, models, and campaigns
- [ ] Add dataset / code release statement
- [ ] Add appendix with prompt snippets if desired
- [ ] Confirm terminology matches the actual runtime
- [ ] Add limitations text on any remaining fail-open behavior
- [ ] Insert diagrams and tables

---

## References

[1] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. **Efficient Memory Management for Large Language Model Serving with PagedAttention.** *Proceedings of the 29th Symposium on Operating Systems Principles (SOSP)*, 2023. arXiv:2309.06180. DOI: 10.48550/arXiv.2309.06180.

[2] Anne Ouyang, Simon Guo, Simran Arora, Alex L. Zhang, William Hu, Christopher Ré, and Azalia Mirhoseini. **KernelBench: Can LLMs Write Efficient GPU Kernels?** arXiv:2502.10517, 2025. DOI: 10.48550/arXiv.2502.10517.

[3] Daniel Nichols, Konstantinos Parasyris, Charles Jekel, Abhinav Bhatele, and Harshitha Menon. **Integrating Performance Tools in Model Reasoning for GPU Kernel Optimization.** arXiv:2510.17158, 2025. DOI: 10.48550/arXiv.2510.17158.

[4] Ruifan Chu, Anbang Wang, Xiuxiu Bai, Shuai Liu, and Xiaoshe Dong. **GPU Kernel Optimization Beyond Full Builds: An LLM Framework with Minimal Executable Programs.** arXiv:2512.22147, 2025. DOI: 10.48550/arXiv.2512.22147.

[5] John Yang, Carlos E. Jimenez, Alexander Wettig, Kilian Lieret, Shunyu Yao, Karthik Narasimhan, and Ofir Press. **SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering.** *NeurIPS*, 2024. arXiv:2405.15793.

[6] Xingyao Wang, Boxuan Li, Yufan Song, Frank F. Xu, Xiangru Tang, Mingchen Zhuge, Jiayi Pan, Yueqi Song, Bowen Li, Jaskirat Singh, Hoang H. Tran, Fuqiang Li, Ren Ma, Mingzhang Zheng, Bill Qian, Yanjun Shao, Niklas Muennighoff, Yizhe Zhang, Binyuan Hui, Junyang Lin, Robert Brennan, Hao Peng, Heng Ji, and Graham Neubig. **OpenHands: An Open Platform for AI Software Developers as Generalist Agents.** *ICLR*, 2025. arXiv:2407.16741.

[7] Yuntong Zhang, Haifeng Ruan, Zhiyu Fan, and Abhik Roychoudhury. **AutoCodeRover: Autonomous Program Improvement.** arXiv:2404.05427, 2024.

[8] Xiang Deng et al. **SWE-Bench Pro: Can AI Agents Solve Long-Horizon Software Engineering Tasks?** arXiv:2509.16941, 2025.

[9] Bowen Baker, Joost Huizinga, Leo Gao, Zehao Dou, Melody Y. Guan, Aleksander Madry, Wojciech Zaremba, Jakub Pachocki, and David Farhi. **Monitoring Reasoning Models for Misbehavior and the Risks of Promoting Obfuscation.** arXiv:2503.11926, 2025. DOI: 10.48550/arXiv.2503.11926.

[10] Melody Y. Guan, Miles Wang, Micah Carroll, Zehao Dou, Annie Y. Wei, Marcus Williams, Benjamin Arnav, Joost Huizinga, Ian Kivlichan, Mia Glaese, Jakub Pachocki, and Bowen Baker. **Monitoring Monitorability.** arXiv:2512.18311, 2025.

[11] Yanda Chen, Joe Benton, Ansh Radhakrishnan, Jonathan Uesato, Carson Denison, John Schulman, Arushi Somani, Peter Hase, Misha Wagner, Fabien Roger, Vlad Mikulik, Samuel R. Bowman, Jan Leike, Jared Kaplan, and Ethan Perez. **Reasoning Models Don’t Always Say What They Think.** arXiv:2505.05410, 2025.

[12] Ashwin Sreevatsa, Sebastian Prasanna, and Cody Rushing. **Basic Legibility Protocols Improve Trusted Monitoring.** arXiv:2602.10153, 2026. DOI: 10.48550/arXiv.2602.10153.

[13] Jonathan Kutasov, Yuqi Sun, Paul Colognese, Teun van der Weij, Linda Petrini, Chen Bo Calvin Zhang, John Hughes, Xiang Deng, Henry Sleight, Tyler Tracy, Buck Shlegeris, and Joe Benton. **SHADE-Arena: Evaluating Sabotage and Monitoring in LLM Agents.** arXiv:2506.15740, 2025.

[14] Neil Kale, Chen Bo Calvin Zhang, Kevin Zhu, Ankit Aich, Paula Rodriguez, Scale Red Team, Christina Q. Knight, and Zifan Wang. **Reliable Weak-to-Strong Monitoring of LLM Agents.** arXiv:2508.19461, 2025.

[15] Asa Cooper Stickland, Jan Michelfeit, Arathi Mani, Charlie Griffin, Ollie Matthews, Tomek Korbak, Rogan Inglis, Oliver Makins, and Alan Cooney. **Async Control: Stress-testing Asynchronous Control Measures for LLM Agents.** arXiv:2512.13526, 2025.

[16] Madhulika Srikumar, Kasia Chmielinski, Jacob Pratt, Carolyn Ashurst, Chloé Bakalar, William Bartholomew, Rishi Bommasani, Peter Cihon, Rebecca Crootof, Mia Hoffmann, Ruchika Joshi, Maarten Sap, and Caleb Withers. **Prioritizing Real-Time Failure Detection in AI Agents.** Partnership on AI report, September 2025.

[17] Islem Bouzenia and Michael Pradel. **Understanding Software Engineering Agents: A Study of Thought-Action-Result Trajectories.** arXiv:2506.18824, 2025.

[18] Timothy Bula, Saurabh Pujar, Luca Buratti, Mihaela Bornea, and Avirup Sil. **SeaView: Software Engineering Agent Visual Interface for Enhanced Workflow.** arXiv:2504.08696, 2025. DOI: 10.48550/arXiv.2504.08696.

[19] Shuyang Liu, Yang Chen, Rahul Krishna, Saurabh Sinha, Jatin Ganhotra, and Reyhan Jabbarvand. **Process-Centric Analysis of Agentic Software Systems.** arXiv:2512.02393, 2025/2026. DOI: 10.48550/arXiv.2512.02393.

[20] Adam AlSayyad, Kelvin Yuxiang Huang, and Richik Pal. **AgentTrace: A Structured Logging Framework for Agent System Observability.** arXiv:2602.10133, 2026.

---

## Changelog

Changes made on 2026-04-15 to better represent the full AMMO architecture. All additions use conference-appropriate abstractions (design patterns, mechanisms, principles) rather than implementation-specific details (agent names, hook filenames, script paths).

### Section 4.2: Roles, trust distribution, and structural independence
**What changed**: Expanded from a 4-item bullet list to include a new subsection ("Two orthogonal oversight axes") describing process oversight vs. product verification as structurally independent.
**Why**: The original text undersold a key architectural property — the validator writes its own tests from the specification alone, never seeing the actor's implementation. This structural independence is the system's primary defense against correlated model failures, and was previously only mentioned in passing in Section 9.3. Surfacing it in the architecture section makes the limitations discussion (9.3) more precise and gives reviewers a concrete mechanism to evaluate.

### Section 4.5: New subsection (d) Evidence-guided coaching
**What changed**: Added a fourth intervention category describing the monitor's constructive mode — using the actor's own experimental data to suggest unexplored fix paths during validation failures.
**Why**: The original three categories (methodology, evidence, reasoning degradation) were all adversarial. The real system also coaches actors through fix progressions by synthesizing their own scattered observations, particularly during accuracy recovery sequences. This shifts the value proposition from "catches mistakes" to "makes actors more effective" — a stronger adoption argument and a more accurate description of how the monitor operates in practice.

### Section 4.7: New subsection on calibrated enforcement
**What changed**: Added a table and discussion of enforcement calibration — different violation classes receive different enforcement hardness (hard deny, hard block, one-shot warning, advisory).
**Why**: The original text described only message-delivery enforcement (the hardest guard). The real system deliberately uses a spectrum, calibrated to the irreversibility of each violation. This is a transferable design principle (any hook-based oversight system faces this tradeoff) and makes the hardening-gaps discussion in Section 9.5 more precise — readers can now evaluate which parts of the spectrum are intentionally permissive vs. which are gaps.

### Section 4.9: New section — Iterative campaigns and shifting optimization targets
**What changed**: Added a full subsection covering the multi-round campaign loop: mechanical termination via Amdahl's Law, incumbent drift, technology pivoting, and concurrent adversarial/constructive work.
**Why**: The paper's title says "long-horizon" but the original architecture section described what reads like a single optimization pass. The real system runs iterative campaigns where each success shifts the target, evidence demands adapt dynamically, and the stopping rule is a physical bound rather than a qualitative judgment. None of the related work addresses oversight under shifting targets — this is a genuinely novel framing that the paper was failing to present.

### Section 1.1: Updated contributions list
**What changed**: Added contribution (4) on iterative campaign architecture. Updated contributions (1), (2), and (3) to reflect the new material on structural independence, calibrated enforcement, and evidence-guided coaching. Renumbered (4)→(5), (5)→(6).
**Why**: The contributions list should reflect the paper's actual content. The new material on campaigns, trust distribution, calibrated enforcement, and coaching represents substantive architectural claims that deserve explicit listing.

### Section 7.4: Replaced generic systems-contribution argument
**What changed**: Replaced a 6-item bullet list of ingredients ("transcript availability, monitor prompts, ...") with a concrete example of the tiered measurement classification and gating workflow.
**Why**: The original framing ("we composed existing pieces") was accurate but vague. The tiered verdict system (neutral / regression requiring gating / catastrophic failure) demonstrates what "systems contribution" concretely means: a decision procedure that maps measurement uncertainty into differentiated actions. This gives reviewers something specific to evaluate rather than a hand-wave about composition.

### Section 9.3: Strengthened correlated-failures discussion
**What changed**: Added explicit reference to the structural independence described in Section 4.2 (validator writes own tests from spec alone).
**Why**: The original 9.3 mentioned artifact-based proof obligations as a defense but did not connect back to the two-layer verification model. The revision makes the architectural defense concrete and the residual risk (shared reasoning blind spots) more precisely scoped.

### Section 9.5: Replaced generic hardening-gaps text
**What changed**: Connected to the calibrated enforcement spectrum from Section 4.7. Described specific permissive choices (production-parity warns but doesn't block; session-termination nudges once; all hooks fail open on errors) and explained the design rationale.
**Why**: The original 9.5 said "some hooks are advisory" without explaining why. The revision reframes this as a deliberate design tradeoff (operational robustness vs. strict enforcement) with specific examples, letting reviewers evaluate the tradeoff rather than just noting a gap.

### Section numbering
**What changed**: Old 4.9 (Algorithm sketch) → 4.10. Old 4.10 (Implementation guarantees) → 4.11.
**Why**: Mechanical renumbering to accommodate the new Section 4.9 on iterative campaigns.

