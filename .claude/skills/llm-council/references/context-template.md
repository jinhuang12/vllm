# Context Template for LLM Council

Use this template to build `.llm-council/context.md`. **Rich context is essential** - critics can only review what they see.

## Required Sections

```markdown
# Deliberation Context

## Original User Request (Verbatim)

> Copy-paste the EXACT user request, including any @file references.
> Do not paraphrase or summarize.

"[Paste exact user message here, preserving all details]"

---

## Session History

### Summary of Work Done

Brief timeline of what Claude has done since session start:
1. [First major action/investigation]
2. [Second major action]
3. [etc.]

### Explore Agent Findings

For EACH Explore agent invoked, include its complete findings:

#### Explore Agent 1: [Topic]
**Query:** [What was explored]
**Key Findings:**
- [Finding 1 with specific details]
- [Finding 2 with numbers/evidence]
- [Finding 3]

**Evidence/Code Found:**
```[language]
[Relevant code snippets discovered]
```

#### Explore Agent 2: [Topic]
[Same structure...]

### Plan Mode Analysis

If Plan mode was used, include:
- Original plan outline
- Key decisions made
- Constraints identified
- Trade-offs considered

### Profiling/Benchmark Data

If any measurements were taken:

| Metric | Baseline | Current | Target |
|--------|----------|---------|--------|
| [name] | [value]  | [value] | [value]|

---

## Repository Context

### Project Overview
- **Project:** [Name and brief description]
- **Language/Framework:** [e.g., Python/FastAPI, CUDA/C++]
- **Key directories:**
  - `src/` - [description]
  - `tests/` - [description]

### Configuration Files (if relevant)
AGENTS.md content (if exists):
```
[paste contents or summarize key points]
```

GEMINI.md content (if exists):
```
[paste contents or summarize key points]
```

### Key Constraints
- Hardware: [e.g., L40S GPU, 100KB shared memory]
- Dependencies: [e.g., must use existing API]
- Performance targets: [e.g., must be faster at BS>=8]

---

## Current Implementation

### File: [path/to/file.ext] (lines X-Y)

```[language]
// ACTUAL CODE - not description
// Include enough context for critics to understand
[paste real code here]
```

### File: [path/to/another/file.ext]

```[language]
[paste real code here]
```

---

## Current Proposal

### Overview
[1-2 sentence summary of what you're proposing]

### Phase 1: [Name]

**Problem:** [What issue this addresses]

**Current behavior:**
```[language]
[existing code]
```

**Proposed change:**
```[language]
[new code]
```

**Rationale:** [Why this change helps]

**Expected impact:** [Quantified if possible]

### Phase 2: [Name]
[Same structure...]

---

## Implementation Plan

1. [Step 1] - [which file(s)]
2. [Step 2] - [which file(s)]
3. [Validation step]

---

## Questions for Critics

Be specific about what you want reviewed:

1. [Specific question about correctness]
2. [Question about edge cases]
3. [Question about alternative approaches]
4. [Question about risks]

---

## Files for Critics to Review

List all files critics should examine:

- `path/to/main/file.ext` - [why relevant]
- `path/to/related/file.ext` - [why relevant]
- `path/to/test/file.ext` - [if tests exist]
```

## Anti-Patterns to Avoid

### ❌ Too Sparse
```markdown
## Original Request
Optimize the kernel.

## Proposal
Make it faster using shuffle reduction.

## Questions
Is this correct?
```

### ❌ Description Instead of Code
```markdown
## Current Implementation
The function uses warp 0 for reduction and atomicAdd for output.
```

### ❌ Missing Session History
```markdown
## Proposal
[Jumps straight to proposal without explaining how you got here]
```

## Good Context Indicators

✅ User request is copy-pasted verbatim
✅ Each Explore agent's findings are documented
✅ Actual code snippets included (not descriptions)
✅ File paths are specified for all code
✅ Profiling data includes specific numbers
✅ Questions are specific and actionable
✅ Constraints are explicitly stated