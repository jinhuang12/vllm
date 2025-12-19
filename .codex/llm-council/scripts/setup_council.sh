#!/bin/bash
# Setup LLM Council directory in a project
# Usage: bash setup_council.sh [project_path] [--fingerprint <string>]
#
# Options:
#   --fingerprint <string>  Store a topic fingerprint for context staleness detection
#
# Example:
#   bash setup_council.sh . --fingerprint "moe_kernel_optimization_csrc/moe"

set -e

# Parse arguments
PROJECT_PATH="."
TOPIC_FINGERPRINT=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --fingerprint)
            TOPIC_FINGERPRINT="$2"
            shift 2
            ;;
        *)
            PROJECT_PATH="$1"
            shift
            ;;
    esac
done

COUNCIL_DIR="$PROJECT_PATH/.llm-council"

echo "🏛️  Setting up LLM Council in $PROJECT_PATH"

# Create directory and tmp subdirectory
mkdir -p "$COUNCIL_DIR"
mkdir -p "$COUNCIL_DIR/tmp"

# Add to .gitignore if not already present
GITIGNORE="$PROJECT_PATH/.gitignore"
if [ -f "$GITIGNORE" ]; then
    if ! grep -q "^\.llm-council/$" "$GITIGNORE" 2>/dev/null; then
        echo ".llm-council/" >> "$GITIGNORE"
        echo "✅ Added .llm-council/ to .gitignore"
    fi
else
    echo ".llm-council/" > "$GITIGNORE"
    echo "✅ Created .gitignore with .llm-council/"
fi

# Create critic_prompt.md
cat > "$COUNCIL_DIR/critic_prompt.md" << 'CRITIC_EOF'
# Critic Role Instructions

You are a CRITIC: an objective, thorough reviewer evaluating a proposal. Your job is to protect correctness and feasibility. Be direct and constructive.

## Your Task

Review the proposal against these criteria:
- **Correctness**: Will the proposed changes work as intended?
- **Completeness**: Are there missing steps, edge cases, or error handling?
- **Assumptions**: What assumptions might be incorrect?
- **Risks**: What could go wrong? What are the failure modes?
- **Alternatives**: Are there better approaches?

## Before You Review

1. **Understand the context**: Read the session history to understand HOW the proposer arrived at this plan
2. **Check the constraints**: Note hardware limits, API requirements, existing architecture
3. **Examine the actual code**: Don't just read descriptions - look at the real implementation

## Decision Rule

- **ACCEPT**: Proposal is sound. Minor improvements can be suggested.
- **REJECT**: Blocking issues exist that would cause failure or incorrect results.

## Response Format

```
VOTE: [ACCEPT or REJECT]

## Blocking Issues
[Only for REJECT]
1. [Issue] - [Why blocking] - [Suggested fix]

## Non-Blocking Improvements
- [Suggestion]

## Verification Notes
[Things you couldn't verify]

## What Looks Good
[Acknowledge sound decisions]
```

## Guidelines

- Be specific: cite code sections, line numbers, concrete values
- Distinguish blocking vs non-blocking issues
- Consider context: the proposer may have constraints you don't know
- For multi-round: acknowledge addressed concerns, don't repeat resolved issues
- Avoid: rejecting for style, demanding scope changes, hallucinating issues
CRITIC_EOF

# Create context.md template
cat > "$COUNCIL_DIR/context.md" << 'CONTEXT_EOF'
# Deliberation Context

## Original User Request (Verbatim)

> Paste the EXACT user request here, including any @file references.
> Do not paraphrase.

"[Paste exact user message here]"

---

## Session History

### Work Done So Far
1. [First major action/investigation]
2. [Second major action]

### Explore Agent Findings

#### Explore Agent 1: [Topic]
**Key Findings:**
- [Finding with specific details]

#### Explore Agent 2: [Topic]
[Same structure...]

### Profiling/Benchmark Data (if any)

| Metric | Baseline | Current |
|--------|----------|---------|
| [name] | [value]  | [value] |

---

## Repository Context

- **Project:** [Name and description]
- **Key constraints:** [Hardware limits, dependencies, etc.]

---

## Current Implementation

### File: [path/to/file.ext] (lines X-Y)

```[language]
// ACTUAL CODE - not description
[paste real code here]
```

---

## Current Proposal

### Phase 1: [Name]

**Problem:** [What issue this addresses]

**Proposed change:**
```[language]
[new code]
```

**Expected impact:** [Quantified if possible]

---

## Questions for Critics

1. [Specific question about correctness]
2. [Question about edge cases]
3. [Question about alternatives]
CONTEXT_EOF

# Create empty history.md
cat > "$COUNCIL_DIR/history.md" << 'HISTORY_EOF'
# Deliberation History

<!-- Append each round's results here -->
HISTORY_EOF

# Clear any existing session ID files (fresh start)
rm -f "$COUNCIL_DIR/tmp/session_gemini.txt" 2>/dev/null || true
rm -f "$COUNCIL_DIR/tmp/session_claude.txt" 2>/dev/null || true

# Store topic fingerprint if provided
if [ -n "$TOPIC_FINGERPRINT" ]; then
    echo "$TOPIC_FINGERPRINT" > "$COUNCIL_DIR/topic_fingerprint.txt"
    echo "✅ Stored topic fingerprint for context staleness detection"
fi

echo "✅ LLM Council initialized at $COUNCIL_DIR"
echo ""
echo "Files created:"
echo "  - critic_prompt.md  (critic instructions)"
echo "  - context.md        (deliberation context - FILL THIS IN)"
echo "  - history.md        (multi-round history)"
echo "  - tmp/              (temporary outputs & session IDs)"
if [ -n "$TOPIC_FINGERPRINT" ]; then
echo "  - topic_fingerprint.txt (for staleness detection)"
fi
echo ""
echo "Features enabled:"
echo "  - Full access (YOLO): critics can run commands and inspect/edit files"
echo "  - Gemini web search: critics can verify claims online (Gemini CLI)"
echo "  - Session tracking: robust resume by session ID (best-effort)"
echo "  - Parallel-by-default deliberation: use --sequential to enable critic chaining"
echo "  - Graceful degradation: continues with available critics"
echo ""
echo "⚠️  IMPORTANT: Edit context.md with FULL session history before running critics"
echo "   See references/context-template.md for complete structure"
