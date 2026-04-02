# Gate 5.1b Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the 81%-bypassed component-level tensor capture/compare Gate 5.1b with E2E greedy decode correctness via GSM8K prompts, piggybacked on the sweep script's already-loaded LLM.

**Architecture:** The sweep script gains a Phase 1 (correctness) before its existing Phase 2 (latency). Phase 1 runs 30 GSM8K prompts via greedy decode, captures token_ids + top-5 logprobs, and either saves golden refs (Stage 1) or compares against them (Stage 5). Two comparator modes: `exact_greedy` (BF16 tracks) and `topk_relaxed` (FP8 tracks) with a "zero questions lost" GSM8K accuracy hard gate for quantization tracks.

**Tech Stack:** Python 3.12, vLLM LLM class, argparse, JSON serialization. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-04-01-gate-5-1b-redesign-design.md`

---

## File Structure

| File | Responsibility | Action |
|------|---------------|--------|
| `.claude/skills/ammo/scripts/run_vllm_bench_latency_sweep.py` | Sweep script — gains Phase 1 correctness | Modify |
| `.claude/skills/ammo/scripts/test_correctness_comparator.py` | Unit tests for comparator logic | Create |
| `.claude/skills/ammo/data/gsm8k_subset.json` | Bundled 30+5 GSM8K questions | Create |
| `.claude/skills/ammo/references/validation-defaults.md` | Gate definitions — rewrite 5.1b section | Modify |
| `.claude/skills/ammo/orchestration/parallel-tracks.md` | Pass criteria — update 5.1b ownership | Modify |
| `.claude/skills/ammo/orchestration/integration-logic.md` | Stage 6 — add integration correctness check | Modify |
| `.claude/agents/ammo-impl-validator.md` | Validator agent — narrow to 5.1a only | Modify |
| `.claude/agents/ammo-impl-champion.md` | Champion agent — add sweep correctness flags | Modify |
| `.claude/skills/ammo/references/tensor-capture-template.py` | Old 5.1b capture template | Delete |
| `.claude/skills/ammo/references/tensor-compare-template.py` | Old 5.1b compare template | Delete |

---

### Task 1: Bundle GSM8K Subset Data

**Files:**
- Create: `.claude/skills/ammo/data/gsm8k_subset.json`

- [ ] **Step 1: Download GSM8K data and extract subset**

```bash
source .venv/bin/activate
python3 -c "
import json, requests

train_url = 'https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl'
test_url = 'https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl'

train_data = [json.loads(line) for line in requests.get(train_url).text.strip().split('\n') if not line.startswith('#')]
test_data = [json.loads(line) for line in requests.get(test_url).text.strip().split('\n') if not line.startswith('#')]

subset = {
    'metadata': {
        'source': 'https://github.com/openai/grade-school-math',
        'train_count': 5,
        'test_count': 30,
        'purpose': 'Bundled GSM8K subset for AMMO Gate 5.1b correctness checks'
    },
    'train': train_data[:5],
    'test': test_data[:30]
}

with open('.claude/skills/ammo/data/gsm8k_subset.json', 'w') as f:
    json.dump(subset, f, indent=2)
print(f'Written: {len(subset[\"train\"])} train, {len(subset[\"test\"])} test examples')
"
```

Expected: `Written: 5 train, 30 test examples`

- [ ] **Step 2: Verify the file is valid and small**

```bash
python3 -c "
import json
with open('.claude/skills/ammo/data/gsm8k_subset.json') as f:
    d = json.load(f)
assert len(d['train']) == 5
assert len(d['test']) == 30
assert all('question' in ex and 'answer' in ex for ex in d['train'] + d['test'])
print('OK')
" && wc -c .claude/skills/ammo/data/gsm8k_subset.json
```

Expected: `OK` and file size ~15-30KB.

- [ ] **Step 3: Commit**

```bash
git add .claude/skills/ammo/data/gsm8k_subset.json
git commit -m "feat(ammo): bundle GSM8K subset for Gate 5.1b correctness checks"
```

---

### Task 2: Write Comparator Unit Tests (TDD — Tests First)

**Files:**
- Create: `.claude/skills/ammo/scripts/test_correctness_comparator.py`

- [ ] **Step 1: Write failing tests for `_compare_exact_greedy`**

```python
"""Unit tests for Gate 5.1b correctness comparator functions.

Tests use synthetic golden/opt data (JSON-safe dicts, not vLLM objects).
No model or GPU required.
"""
import pytest


# ---- exact_greedy tests ----

def test_exact_greedy_identical_pass():
    """Identical token sequences → PASS."""
    from run_vllm_bench_latency_sweep import _compare_correctness
    golden = [{"token_ids": [10, 20, 30], "logprobs": [], "text": "a b c", "num_tokens": 3, "prompt_index": 0}]
    opt = [{"token_ids": [10, 20, 30], "logprobs": [], "text": "a b c", "num_tokens": 3, "prompt_index": 0}]
    result = _compare_correctness(golden_refs=golden, opt_outputs=opt, mode="exact_greedy",
                                   max_divergent_positions=0, max_topk_failures_pct=5.0,
                                   labels=None, baseline_preds=None, opt_preds=None)
    assert result["verdict"] == "PASS"
    assert result["divergent_positions"] == 0


def test_exact_greedy_one_divergent_fail():
    """One divergent position with threshold=0 → FAIL."""
    from run_vllm_bench_latency_sweep import _compare_correctness
    golden = [{"token_ids": [10, 20, 30], "logprobs": [], "text": "", "num_tokens": 3, "prompt_index": 0}]
    opt = [{"token_ids": [10, 99, 30], "logprobs": [], "text": "", "num_tokens": 3, "prompt_index": 0}]
    result = _compare_correctness(golden_refs=golden, opt_outputs=opt, mode="exact_greedy",
                                   max_divergent_positions=0, max_topk_failures_pct=5.0,
                                   labels=None, baseline_preds=None, opt_preds=None)
    assert result["verdict"] == "FAIL"
    assert result["divergent_positions"] == 1
    assert "diagnostics" in result
    assert "first_divergence" in result["diagnostics"]
    assert result["diagnostics"]["first_divergence"]["question_idx"] == 0
    assert result["diagnostics"]["first_divergence"]["position"] == 1


def test_exact_greedy_one_divergent_with_tolerance_pass():
    """One divergent position with threshold=1 → PASS."""
    from run_vllm_bench_latency_sweep import _compare_correctness
    golden = [{"token_ids": [10, 20, 30], "logprobs": [], "text": "", "num_tokens": 3, "prompt_index": 0}]
    opt = [{"token_ids": [10, 99, 30], "logprobs": [], "text": "", "num_tokens": 3, "prompt_index": 0}]
    result = _compare_correctness(golden_refs=golden, opt_outputs=opt, mode="exact_greedy",
                                   max_divergent_positions=1, max_topk_failures_pct=5.0,
                                   labels=None, baseline_preds=None, opt_preds=None)
    assert result["verdict"] == "PASS"


def test_exact_greedy_length_mismatch_counts_as_divergence():
    """Opt shorter than baseline → extra baseline positions count as divergent."""
    from run_vllm_bench_latency_sweep import _compare_correctness
    golden = [{"token_ids": [10, 20, 30, 40, 50], "logprobs": [], "text": "", "num_tokens": 5, "prompt_index": 0}]
    opt = [{"token_ids": [10, 20, 30], "logprobs": [], "text": "", "num_tokens": 3, "prompt_index": 0}]
    result = _compare_correctness(golden_refs=golden, opt_outputs=opt, mode="exact_greedy",
                                   max_divergent_positions=0, max_topk_failures_pct=5.0,
                                   labels=None, baseline_preds=None, opt_preds=None)
    assert result["verdict"] == "FAIL"
    assert result["divergent_positions"] == 2  # 2 extra positions in baseline


# ---- topk_relaxed tests ----

def test_topk_relaxed_token_in_topk_pass():
    """Bidirectional containment succeeds → PASS."""
    from run_vllm_bench_latency_sweep import _compare_correctness
    golden = [{"token_ids": [10], "logprobs": [{"top_logprobs": {"10": -0.1, "20": -2.0, "30": -3.0, "40": -4.0, "50": -5.0}}],
               "text": "", "num_tokens": 1, "prompt_index": 0}]
    opt = [{"token_ids": [20], "logprobs": [{"top_logprobs": {"20": -0.1, "10": -1.5, "30": -3.0, "40": -4.0, "50": -5.0}}],
            "text": "", "num_tokens": 1, "prompt_index": 0}]
    result = _compare_correctness(golden_refs=golden, opt_outputs=opt, mode="topk_relaxed",
                                   max_divergent_positions=0, max_topk_failures_pct=5.0,
                                   labels=None, baseline_preds=None, opt_preds=None)
    assert result["verdict"] == "PASS"
    assert result["containment_failures"] == 0


def test_topk_relaxed_token_not_in_topk_fail():
    """Token NOT in other's top-K → containment failure. With 1 position, 100% failure → FAIL."""
    from run_vllm_bench_latency_sweep import _compare_correctness
    golden = [{"token_ids": [10], "logprobs": [{"top_logprobs": {"10": -0.1, "20": -2.0, "30": -3.0, "40": -4.0, "50": -5.0}}],
               "text": "", "num_tokens": 1, "prompt_index": 0}]
    opt = [{"token_ids": [99], "logprobs": [{"top_logprobs": {"99": -0.1, "88": -1.5, "77": -3.0, "66": -4.0, "55": -5.0}}],
            "text": "", "num_tokens": 1, "prompt_index": 0}]
    result = _compare_correctness(golden_refs=golden, opt_outputs=opt, mode="topk_relaxed",
                                   max_divergent_positions=0, max_topk_failures_pct=5.0,
                                   labels=None, baseline_preds=None, opt_preds=None)
    assert result["verdict"] == "FAIL"
    assert result["containment_failures"] == 1


def test_topk_relaxed_length_mismatch_penalty():
    """Opt shorter → extra positions count as containment failures."""
    from run_vllm_bench_latency_sweep import _compare_correctness
    golden = [{"token_ids": [10, 20, 30], "logprobs": [
        {"top_logprobs": {"10": -0.1, "20": -2.0, "30": -3.0, "40": -4.0, "50": -5.0}},
        {"top_logprobs": {"20": -0.1, "10": -2.0, "30": -3.0, "40": -4.0, "50": -5.0}},
        {"top_logprobs": {"30": -0.1, "10": -2.0, "20": -3.0, "40": -4.0, "50": -5.0}},
    ], "text": "", "num_tokens": 3, "prompt_index": 0}]
    opt = [{"token_ids": [10], "logprobs": [
        {"top_logprobs": {"10": -0.1, "20": -2.0, "30": -3.0, "40": -4.0, "50": -5.0}},
    ], "text": "", "num_tokens": 1, "prompt_index": 0}]
    result = _compare_correctness(golden_refs=golden, opt_outputs=opt, mode="topk_relaxed",
                                   max_divergent_positions=0, max_topk_failures_pct=50.0,
                                   labels=None, baseline_preds=None, opt_preds=None)
    # 1 matched position + 2 length penalty = 3 total, 2 failures = 66.7%
    assert result["total_positions"] == 3
    assert result["containment_failures"] == 2
    assert result["verdict"] == "FAIL"  # 66.7% > 50%


def test_topk_relaxed_empty_outputs_all_fail():
    """All questions produce 0 tokens → total_positions=0 → FAIL."""
    from run_vllm_bench_latency_sweep import _compare_correctness
    golden = [{"token_ids": [], "logprobs": [], "text": "", "num_tokens": 0, "prompt_index": 0}]
    opt = [{"token_ids": [], "logprobs": [], "text": "", "num_tokens": 0, "prompt_index": 0}]
    result = _compare_correctness(golden_refs=golden, opt_outputs=opt, mode="topk_relaxed",
                                   max_divergent_positions=0, max_topk_failures_pct=5.0,
                                   labels=None, baseline_preds=None, opt_preds=None)
    assert result["verdict"] == "FAIL"


# ---- accuracy gate tests ----

def test_accuracy_gate_zero_lost_pass():
    """Optimized gets all baseline-correct questions right → PASS."""
    from run_vllm_bench_latency_sweep import _compare_correctness
    golden = [{"token_ids": [10], "logprobs": [{"top_logprobs": {"10": -0.1, "20": -2.0, "30": -3.0, "40": -4.0, "50": -5.0}}],
               "text": "", "num_tokens": 1, "prompt_index": 0}]
    opt = [{"token_ids": [10], "logprobs": [{"top_logprobs": {"10": -0.1, "20": -2.0, "30": -3.0, "40": -4.0, "50": -5.0}}],
            "text": "", "num_tokens": 1, "prompt_index": 0}]
    labels = [42]
    baseline_preds = [42]  # correct
    opt_preds = [42]  # also correct
    result = _compare_correctness(golden_refs=golden, opt_outputs=opt, mode="topk_relaxed",
                                   max_divergent_positions=0, max_topk_failures_pct=100.0,
                                   labels=labels, baseline_preds=baseline_preds, opt_preds=opt_preds)
    assert result["gsm8k_accuracy_gate"]["verdict"] == "PASS"
    assert result["gsm8k_accuracy_gate"]["lost_questions"] == []


def test_accuracy_gate_lost_question_fail():
    """Baseline got Q0 right, optimized got it wrong → FAIL."""
    from run_vllm_bench_latency_sweep import _compare_correctness
    golden = [{"token_ids": [10], "logprobs": [{"top_logprobs": {"10": -0.1, "20": -2.0, "30": -3.0, "40": -4.0, "50": -5.0}}],
               "text": "", "num_tokens": 1, "prompt_index": 0}]
    opt = [{"token_ids": [10], "logprobs": [{"top_logprobs": {"10": -0.1, "20": -2.0, "30": -3.0, "40": -4.0, "50": -5.0}}],
            "text": "", "num_tokens": 1, "prompt_index": 0}]
    labels = [42]
    baseline_preds = [42]  # correct
    opt_preds = [99]  # wrong
    result = _compare_correctness(golden_refs=golden, opt_outputs=opt, mode="topk_relaxed",
                                   max_divergent_positions=0, max_topk_failures_pct=100.0,
                                   labels=labels, baseline_preds=baseline_preds, opt_preds=opt_preds)
    assert result["gsm8k_accuracy_gate"]["verdict"] == "FAIL"
    assert result["gsm8k_accuracy_gate"]["lost_questions"] == [0]


def test_accuracy_gate_not_applied_exact_greedy():
    """Accuracy gate is informational-only for exact_greedy mode."""
    from run_vllm_bench_latency_sweep import _compare_correctness
    golden = [{"token_ids": [10], "logprobs": [], "text": "", "num_tokens": 1, "prompt_index": 0}]
    opt = [{"token_ids": [10], "logprobs": [], "text": "", "num_tokens": 1, "prompt_index": 0}]
    labels = [42]
    baseline_preds = [42]
    opt_preds = [99]  # wrong, but exact_greedy doesn't gate on accuracy
    result = _compare_correctness(golden_refs=golden, opt_outputs=opt, mode="exact_greedy",
                                   max_divergent_positions=0, max_topk_failures_pct=5.0,
                                   labels=labels, baseline_preds=baseline_preds, opt_preds=opt_preds)
    assert result["verdict"] == "PASS"  # token match passes
    assert result["gsm8k_accuracy_gate"]["verdict"] == "INFO"  # informational, not gating
```

- [ ] **Step 2: Run tests — verify they all fail (functions don't exist yet)**

```bash
cd /home/jinhun/vllm
source .venv/bin/activate
PYTHONPATH=.claude/skills/ammo/scripts pytest .claude/skills/ammo/scripts/test_correctness_comparator.py -v 2>&1 | head -40
```

Expected: All tests FAIL with `ImportError: cannot import name '_compare_correctness'`

- [ ] **Step 3: Commit test file**

```bash
git add .claude/skills/ammo/scripts/test_correctness_comparator.py
git commit -m "test(ammo): add failing tests for Gate 5.1b correctness comparator"
```

---

### Task 3: Implement GSM8K Prompt Builder in Sweep Script

**Files:**
- Modify: `.claude/skills/ammo/scripts/run_vllm_bench_latency_sweep.py:95-99` (imports) and `:570` (new functions before `_run_inproc_latency_sweep_child`)

- [ ] **Step 1: Add imports at the top of the sweep script**

After line 94 (`from typing import Any, Dict, List, Optional, Tuple`), add:

```python
import ast
```

Note: `requests` is NOT imported at the top level — it is lazy-imported inside `_download_and_cache_file` only (needed for the >30 question GitHub fallback path, which is rare). This avoids breaking the sweep script in environments where `requests` is not installed.

- [ ] **Step 2: Add GSM8K helper functions before `_run_inproc_latency_sweep_child` (~line 570)**

Insert before line 573 (`def _run_inproc_latency_sweep_child(`):

```python
# ---- GSM8K helpers (adapted from tests/evals/gsm8k/gsm8k_eval.py) ----

_GSM8K_SUBSET_PATH = Path(__file__).parent.parent / "data" / "gsm8k_subset.json"
_INVALID_ANSWER = -9999999


def _download_and_cache_file(url: str, filename: str | None = None) -> str:
    import requests as _requests_mod  # Lazy: only needed for >30 question fallback
    if filename is None:
        filename = os.path.join("/tmp", url.split("/")[-1])
    if os.path.exists(filename):
        return filename
    print(f"Downloading from {url} to {filename}")
    response = _requests_mod.get(url, stream=True)
    response.raise_for_status()
    with open(filename, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024):
            f.write(chunk)
    return filename


def _read_jsonl(filename: str):
    with open(filename) as fin:
        for line in fin:
            if not line.startswith("#"):
                yield json.loads(line)


def _load_gsm8k_data(num_questions: int) -> Tuple[List[dict], List[dict]]:
    """Load GSM8K data — bundled subset first, GitHub fallback if needed."""
    if num_questions <= 30 and _GSM8K_SUBSET_PATH.exists():
        with open(_GSM8K_SUBSET_PATH) as f:
            data = json.load(f)
        return data["train"], data["test"][:num_questions]
    # Fallback to full download
    train_url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl"
    test_url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
    train_file = _download_and_cache_file(train_url)
    test_file = _download_and_cache_file(test_url)
    return list(_read_jsonl(train_file)), list(_read_jsonl(test_file))


def _get_answer_value(answer_str: str) -> int:
    answer_str = answer_str.replace(",", "")
    numbers = re.findall(r"\d+", answer_str)
    if len(numbers) < 1:
        return _INVALID_ANSWER
    try:
        return ast.literal_eval(numbers[-1])
    except SyntaxError:
        return _INVALID_ANSWER


def _build_gsm8k_prompts(
    num_questions: int = 30, num_shots: int = 5
) -> Tuple[List[str], List[int]]:
    """Build few-shot GSM8K prompts and ground-truth labels."""
    if num_questions == 0:
        return [], []
    train_data, test_data = _load_gsm8k_data(num_questions)
    num_questions = min(num_questions, len(test_data))
    few_shot_examples = ""
    for i in range(min(num_shots, len(train_data))):
        few_shot_examples += (
            f"Question: {train_data[i]['question']}\n"
            f"Answer: {train_data[i]['answer']}\n\n"
        )
    prompts, labels = [], []
    for i in range(num_questions):
        prompts.append(few_shot_examples + f"Question: {test_data[i]['question']}\nAnswer:")
        labels.append(_get_answer_value(test_data[i]["answer"]))
    assert all(label != _INVALID_ANSWER for label in labels), "Some labels are invalid"
    return prompts, labels
```

- [ ] **Step 3: Verify prompt builder works**

```bash
cd /home/jinhun/vllm
source .venv/bin/activate
PYTHONPATH=.claude/skills/ammo/scripts python3 -c "
from run_vllm_bench_latency_sweep import _build_gsm8k_prompts
prompts, labels = _build_gsm8k_prompts(num_questions=5)
print(f'{len(prompts)} prompts, {len(labels)} labels')
print(f'First label: {labels[0]}')
print(f'Prompt prefix length: {len(prompts[0])} chars')
assert len(prompts) == 5
assert all(isinstance(l, int) for l in labels)
print('OK')
"
```

Expected: `5 prompts, 5 labels`, `OK`

- [ ] **Step 4: Commit**

```bash
git add .claude/skills/ammo/scripts/run_vllm_bench_latency_sweep.py
git commit -m "feat(ammo): add GSM8K prompt builder to sweep script for Gate 5.1b"
```

---

### Task 4: Implement Serialization and Comparator Functions

**Files:**
- Modify: `.claude/skills/ammo/scripts/run_vllm_bench_latency_sweep.py` (add after GSM8K helpers, before `_run_inproc_latency_sweep_child`)

- [ ] **Step 1: Add output serialization function**

Insert after `_build_gsm8k_prompts`:

```python
def _serialize_correctness_outputs(outputs) -> List[Dict[str, Any]]:
    """Serialize vLLM RequestOutput objects to JSON-safe dicts."""
    serialized = []
    for i, req_output in enumerate(outputs):
        comp = req_output.outputs[0]
        token_ids = list(comp.token_ids)
        logprobs_list = []
        if comp.logprobs is not None:
            for pos_logprobs in comp.logprobs:
                top_lps = {str(tid): lp.logprob for tid, lp in pos_logprobs.items()}
                logprobs_list.append({"top_logprobs": top_lps})
        serialized.append({
            "prompt_index": i,
            "token_ids": token_ids,
            "text": comp.text,
            "logprobs": logprobs_list,
            "num_tokens": len(token_ids),
        })
    return serialized


def _score_gsm8k_predictions(outputs, labels: List[int]) -> Tuple[List[int], float]:
    """Score GSM8K outputs. Returns (predictions_list, accuracy)."""
    preds = []
    for req_output in outputs:
        text = req_output.outputs[0].text
        preds.append(_get_answer_value(text))
    correct = sum(1 for p, l in zip(preds, labels) if p == l)
    accuracy = correct / len(labels) if labels else 0.0
    return preds, accuracy
```

- [ ] **Step 2: Add comparator function**

Insert after `_score_gsm8k_predictions`:

```python
def _compare_correctness(
    *,
    golden_refs: List[Dict[str, Any]],
    opt_outputs: List[Dict[str, Any]],
    mode: str,
    max_divergent_positions: int,
    max_topk_failures_pct: float,
    labels: Optional[List[int]],
    baseline_preds: Optional[List[int]],
    opt_preds: Optional[List[int]],
) -> Dict[str, Any]:
    """Compare optimized outputs against golden references."""
    num_questions = min(len(golden_refs), len(opt_outputs))
    per_question = []

    # ---- Accuracy gate (computed for both modes) ----
    accuracy_gate: Dict[str, Any] = {"enabled": False}
    if labels is not None and baseline_preds is not None and opt_preds is not None:
        baseline_correct = {i for i, (p, l) in enumerate(zip(baseline_preds, labels)) if p == l}
        opt_correct = {i for i, (p, l) in enumerate(zip(opt_preds, labels)) if p == l}
        lost = sorted(baseline_correct - opt_correct)
        gained = sorted(opt_correct - baseline_correct)
        bl_acc = len(baseline_correct) / len(labels) if labels else 0.0
        op_acc = len(opt_correct) / len(labels) if labels else 0.0
        is_hard_gate = mode == "topk_relaxed"
        accuracy_gate = {
            "enabled": True,
            "baseline_accuracy": round(bl_acc, 4),
            "baseline_correct_indices": sorted(baseline_correct),
            "optimized_accuracy": round(op_acc, 4),
            "optimized_correct_indices": sorted(opt_correct),
            "lost_questions": lost,
            "gained_questions": gained,
            "verdict": "FAIL" if (is_hard_gate and len(lost) > 0) else ("PASS" if is_hard_gate else "INFO"),
        }

    if mode == "exact_greedy":
        total_divergent = 0
        total_positions = 0
        first_divergence = None
        for q in range(num_questions):
            b_ids = golden_refs[q]["token_ids"]
            o_ids = opt_outputs[q]["token_ids"]
            min_len = min(len(b_ids), len(o_ids))
            divergent = sum(1 for p in range(min_len) if b_ids[p] != o_ids[p])
            first_div = next((p for p in range(min_len) if b_ids[p] != o_ids[p]), -1)
            divergent += abs(len(b_ids) - len(o_ids))
            total_divergent += divergent
            total_positions += max(len(b_ids), len(o_ids))
            per_question.append({
                "idx": q, "baseline_tokens": len(b_ids), "opt_tokens": len(o_ids),
                "divergent": divergent, "first_divergence_pos": first_div,
            })
            if first_divergence is None and first_div >= 0:
                first_divergence = {"question_idx": q, "position": first_div,
                                     "baseline_token": b_ids[first_div], "opt_token": o_ids[first_div]}
        if total_positions == 0:
            verdict = "FAIL"  # All questions produced empty output
        else:
            verdict = "PASS" if total_divergent <= max_divergent_positions else "FAIL"
        result = {
            "gate": "5.1b", "verdict": verdict, "mode": "exact_greedy",
            "num_questions": num_questions, "total_positions": total_positions,
            "divergent_positions": total_divergent, "threshold": max_divergent_positions,
            "per_question_summary": per_question, "gsm8k_accuracy_gate": accuracy_gate,
        }
        if verdict == "FAIL" and first_divergence:
            result["diagnostics"] = {"first_divergence": first_divergence,
                                      "divergence_summary": f"{sum(1 for pq in per_question if pq['divergent'] > 0)} of {num_questions} questions diverged"}
        return result

    elif mode == "topk_relaxed":
        total_positions = 0
        containment_failures = 0
        empty_output_count = 0
        first_failure = None
        for q in range(num_questions):
            b = golden_refs[q]
            o = opt_outputs[q]
            b_len = len(b["token_ids"])
            o_len = len(o["token_ids"])
            if b_len == 0 and o_len == 0:
                empty_output_count += 1
                per_question.append({"idx": q, "baseline_tokens": 0, "opt_tokens": 0, "containment_failures": 0})
                continue
            min_len = min(b_len, o_len)
            max_len = max(b_len, o_len)
            q_failures = 0
            for pos in range(min_len):
                total_positions += 1
                b_topk = {int(k) for k in b["logprobs"][pos]["top_logprobs"]}
                o_topk = {int(k) for k in o["logprobs"][pos]["top_logprobs"]}
                b_token = b["token_ids"][pos]
                o_token = o["token_ids"][pos]
                if o_token not in b_topk or b_token not in o_topk:
                    containment_failures += 1
                    q_failures += 1
                    if first_failure is None:
                        first_failure = {"question_idx": q, "position": pos,
                                          "baseline_token": b_token, "opt_token": o_token,
                                          "baseline_top5": dict(b["logprobs"][pos]["top_logprobs"]),
                                          "opt_top5": dict(o["logprobs"][pos]["top_logprobs"])}
            length_penalty = max_len - min_len
            total_positions += length_penalty
            containment_failures += length_penalty
            per_question.append({"idx": q, "baseline_tokens": b_len, "opt_tokens": o_len, "containment_failures": q_failures + length_penalty})

        if empty_output_count > num_questions * 0.1:
            print(f"[correctness] WARNING: {empty_output_count}/{num_questions} questions produced empty output")
        if total_positions == 0:
            verdict = "FAIL"
            failure_rate = 0.0
        else:
            failure_rate = containment_failures / total_positions * 100
            verdict = "PASS" if failure_rate <= max_topk_failures_pct else "FAIL"

        # Accuracy gate can override to FAIL even if token-level passes
        if verdict == "PASS" and accuracy_gate.get("verdict") == "FAIL":
            verdict = "FAIL"

        result = {
            "gate": "5.1b", "verdict": verdict, "mode": "topk_relaxed",
            "num_questions": num_questions, "total_positions": total_positions,
            "containment_failures": containment_failures,
            "failure_rate_pct": round(failure_rate, 4), "threshold_pct": max_topk_failures_pct,
            "empty_output_count": empty_output_count,
            "per_question_summary": per_question, "gsm8k_accuracy_gate": accuracy_gate,
        }
        if verdict == "FAIL" and first_failure:
            result["diagnostics"] = {"first_divergence": first_failure,
                                      "divergence_summary": f"{containment_failures} containment failures across {total_positions} positions ({failure_rate:.1f}%)",
                                      "length_mismatches": sum(1 for pq in per_question if pq.get("baseline_tokens", 0) != pq.get("opt_tokens", 0)),
                                      "empty_outputs": empty_output_count}
        return result
    else:
        raise ValueError(f"Unknown correctness mode: {mode}")
```

- [ ] **Step 3: Run the unit tests — verify they pass**

```bash
cd /home/jinhun/vllm
source .venv/bin/activate
PYTHONPATH=.claude/skills/ammo/scripts pytest .claude/skills/ammo/scripts/test_correctness_comparator.py -v
```

Expected: All tests PASS.

- [ ] **Step 4: Commit**

```bash
git add .claude/skills/ammo/scripts/run_vllm_bench_latency_sweep.py
git commit -m "feat(ammo): implement correctness comparator for Gate 5.1b (exact_greedy + topk_relaxed)"
```

---

### Task 5: Add CLI Flags and Child Forwarding

**Files:**
- Modify: `.claude/skills/ammo/scripts/run_vllm_bench_latency_sweep.py:1008-1024` (argparse), `:1037` (validation), `:1538` (child cmd forwarding)

- [ ] **Step 1: Add 6 public CLI flags after `--baseline-from` (line 1018)**

Insert after line 1018 (after the `--baseline-from` argument block, before `--_child-label`):

```python
    p.add_argument("--capture-golden-refs", action="store_true", default=False,
                   help="Stage 1: run GSM8K greedy decode and save golden refs to json/golden_refs.json")
    p.add_argument("--verify-correctness", action="store_true", default=False,
                   help="Stage 5: compare GSM8K outputs against golden refs; exit nonzero on mismatch")
    p.add_argument("--correctness-mode", type=str, default="exact_greedy",
                   choices=["exact_greedy", "topk_relaxed"],
                   help="Comparator mode for --verify-correctness (default: exact_greedy)")
    p.add_argument("--correctness-num-questions", type=int, default=30,
                   help="Number of GSM8K questions for correctness phase (default: 30)")
    p.add_argument("--max-divergent-positions", type=int, default=0,
                   help="(exact_greedy) max allowed token mismatches (default: 0)")
    p.add_argument("--max-topk-failures-pct", type=float, default=5.0,
                   help="(topk_relaxed) max pct of positions failing top-K containment (default: 5.0)")
```

- [ ] **Step 2: Add 6 hidden child-forwarding flags after existing hidden flags (line 1024)**

Insert after `--_nsys-num-iters`:

```python
    p.add_argument("--_capture-golden-refs", action="store_true", default=False, help=argparse.SUPPRESS)
    p.add_argument("--_verify-correctness", action="store_true", default=False, help=argparse.SUPPRESS)
    p.add_argument("--_correctness-mode", type=str, default="exact_greedy", help=argparse.SUPPRESS)
    p.add_argument("--_correctness-num-questions", type=int, default=30, help=argparse.SUPPRESS)
    p.add_argument("--_max-divergent-positions", type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument("--_max-topk-failures-pct", type=float, default=5.0, help=argparse.SUPPRESS)
```

- [ ] **Step 3: Add mutual exclusivity validation after existing nsys validation (~line 1037)**

Insert after the nsys validation block:

```python
    # Validate correctness flag constraints.
    if args.capture_golden_refs and args.verify_correctness:
        raise SystemExit("--capture-golden-refs and --verify-correctness are mutually exclusive")
    if args.verify_correctness and not args.baseline_from:
        raise SystemExit("--verify-correctness requires --baseline-from (to import golden_refs.json)")
```

- [ ] **Step 4: Add child cmd forwarding after nsys forwarding (~line 1538)**

Insert after the nsys child_cmd forwarding block:

```python
        # Forward correctness flags to child.
        if args.capture_golden_refs:
            child_cmd.append("--_capture-golden-refs")
        if args.verify_correctness:
            child_cmd.extend(["--_verify-correctness",
                              "--_correctness-mode", args.correctness_mode,
                              "--_max-divergent-positions", str(args.max_divergent_positions),
                              "--_max-topk-failures-pct", str(args.max_topk_failures_pct)])
        if args.correctness_num_questions != 30:
            child_cmd.extend(["--_correctness-num-questions", str(args.correctness_num_questions)])
```

- [ ] **Step 5: Verify flag parsing works**

```bash
cd /home/jinhun/vllm
source .venv/bin/activate
python3 .claude/skills/ammo/scripts/run_vllm_bench_latency_sweep.py --help 2>&1 | grep -A1 "capture-golden\|verify-correct\|correctness-mode"
```

Expected: All 6 new flags appear in help output.

- [ ] **Step 6: Commit**

```bash
git add .claude/skills/ammo/scripts/run_vllm_bench_latency_sweep.py
git commit -m "feat(ammo): add correctness CLI flags to sweep script"
```

---

### Task 6: Implement Phase 1 Correctness in Child Runner

**Files:**
- Modify: `.claude/skills/ammo/scripts/run_vllm_bench_latency_sweep.py:573-585` (child function signature) and `:689-691` (insertion point)

- [ ] **Step 1: Extend `_run_inproc_latency_sweep_child` signature (line 573)**

Change the function signature to add correctness parameters:

```python
def _run_inproc_latency_sweep_child(
    *,
    label: str,
    model_id: str,
    tp: int,
    max_model_len: int,
    buckets: List[Dict[str, int]],
    num_iters: int,
    extra_args: List[str],
    out_root: Path,
    timeout_s_per_bucket: int,
    nsys_profile: bool = False,
    capture_golden_refs: bool = False,
    verify_correctness: bool = False,
    correctness_mode: str = "exact_greedy",
    correctness_num_questions: int = 30,
    max_divergent_positions: int = 0,
    max_topk_failures_pct: float = 5.0,
) -> int:
```

- [ ] **Step 2: Add Phase 1 correctness logic after model load (after line 689)**

Insert between `_update_status("model_loaded")` (line 689) and `for bucket in buckets:` (line 691):

```python
    # ---- Phase 1: Correctness (GSM8K greedy decode) ----
    if capture_golden_refs or verify_correctness:
        import torch
        from vllm import SamplingParams as _CorrectnessSP
        _update_status("correctness_phase_start")
        json_dir = out_root / "json"
        json_dir.mkdir(parents=True, exist_ok=True)
        try:
            prompts, gsm8k_labels = _build_gsm8k_prompts(num_questions=correctness_num_questions)
            correctness_sp = _CorrectnessSP(
                temperature=0.0, max_tokens=256,
                stop=["Question", "Assistant:", "<|separator|>"],
                seed=42, logprobs=5,
            )
            print(f"[correctness] Running GSM8K greedy decode: {len(prompts)} questions")
            t0 = time.time()
            outputs = llm.generate(prompts, sampling_params=correctness_sp, use_tqdm=False)
            duration = time.time() - t0
            print(f"[correctness] Generation done in {duration:.1f}s")

            serialized = _serialize_correctness_outputs(outputs)
            preds, accuracy = _score_gsm8k_predictions(outputs, gsm8k_labels)
            gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unknown"

            if capture_golden_refs:
                # Self-consistency check: run prompts a second time
                print("[correctness] Running self-consistency check...")
                outputs2 = llm.generate(prompts, sampling_params=correctness_sp, use_tqdm=False)
                serialized2 = _serialize_correctness_outputs(outputs2)
                deterministic = all(
                    s1["token_ids"] == s2["token_ids"]
                    for s1, s2 in zip(serialized, serialized2)
                )
                if not deterministic:
                    print("[correctness] WARNING: Self-consistency check FAILED — environment is non-deterministic.")
                    print("[correctness] Recommended mode for Stage 5: topk_relaxed")
                else:
                    print("[correctness] Self-consistency check PASSED — greedy decode is deterministic.")

                golden_data = {
                    "metadata": {
                        "num_questions": len(prompts), "num_shots": 5, "max_tokens": 256,
                        "seed": 42, "logprobs_k": 5, "gsm8k_accuracy": round(accuracy, 4),
                        "capture_duration_s": round(duration, 2), "gpu_name": gpu_name,
                        "deterministic": deterministic,
                        "baseline_preds": preds, "labels": gsm8k_labels,
                    },
                    "outputs": serialized,
                }
                golden_path = json_dir / "golden_refs.json"
                _write_json(golden_path, golden_data)
                print(f"[correctness] Golden refs saved to {golden_path}")
                _update_status("correctness_done", extra={"accuracy": accuracy, "deterministic": deterministic})

            if verify_correctness:
                golden_path = json_dir / "golden_refs.json"
                if not golden_path.exists():
                    print(f"[correctness] ERROR: golden_refs.json not found at {golden_path}")
                    _update_status("correctness_error", extra={"error": "golden_refs.json not found"})
                    return 4
                golden_data = json.loads(golden_path.read_text(encoding="utf-8"))
                golden_refs = golden_data["outputs"]
                golden_meta = golden_data.get("metadata", {})
                baseline_preds = golden_meta.get("baseline_preds")
                golden_labels = golden_meta.get("labels")

                # GPU name mismatch warning
                golden_gpu = golden_meta.get("gpu_name", "")
                if golden_gpu and golden_gpu != gpu_name:
                    print(f"[correctness] WARNING: GPU mismatch — golden refs captured on '{golden_gpu}', current GPU is '{gpu_name}'")

                # Save opt outputs
                opt_path = json_dir / "opt_outputs.json"
                _write_json(opt_path, {"outputs": serialized})

                # Run comparator
                verdict = _compare_correctness(
                    golden_refs=golden_refs, opt_outputs=serialized,
                    mode=correctness_mode,
                    max_divergent_positions=max_divergent_positions,
                    max_topk_failures_pct=max_topk_failures_pct,
                    labels=golden_labels, baseline_preds=baseline_preds, opt_preds=preds,
                )
                verdict["gsm8k_accuracy_baseline"] = golden_meta.get("gsm8k_accuracy")
                verdict["gsm8k_accuracy_optimized"] = round(accuracy, 4)
                verdict["duration_s"] = round(duration, 2)

                verdict_path = json_dir / "correctness_verdict.json"
                _write_json(verdict_path, verdict)
                print(f"[correctness] Verdict: {verdict['verdict']}")
                print(f"[correctness] Written to {verdict_path}")

                if verdict["verdict"] != "PASS":
                    _update_status("correctness_failed", extra=verdict)
                    return 3  # Correctness FAIL — don't proceed to latency
                _update_status("correctness_done", extra=verdict)

        except Exception as e:
            print(f"[correctness] ERROR: {type(e).__name__}: {e}")
            traceback.print_exc()
            _update_status("correctness_error", extra={"error": str(e)})
            return 4  # Infrastructure error
    # ---- End Phase 1 ----
```

- [ ] **Step 3: Pass correctness kwargs at the call site (~line 1457)**

Find the `_run_inproc_latency_sweep_child(` call and add the new kwargs:

```python
            nsys_profile=getattr(args, "_nsys_profile", False),
            capture_golden_refs=getattr(args, "_capture_golden_refs", False),
            verify_correctness=getattr(args, "_verify_correctness", False),
            correctness_mode=getattr(args, "_correctness_mode", "exact_greedy"),
            correctness_num_questions=getattr(args, "_correctness_num_questions", 30),
            max_divergent_positions=getattr(args, "_max_divergent_positions", 0),
            max_topk_failures_pct=getattr(args, "_max_topk_failures_pct", 5.0),
```

- [ ] **Step 4: Commit**

```bash
git add .claude/skills/ammo/scripts/run_vllm_bench_latency_sweep.py
git commit -m "feat(ammo): implement Phase 1 correctness in sweep script child runner"
```

---

### Task 7: Copy Golden Refs BEFORE Child Subprocess Launch

**Files:**
- Modify: `.claude/skills/ammo/scripts/run_vllm_bench_latency_sweep.py:~1500` (BEFORE child subprocess loop)

**CRITICAL**: The `--baseline-from` import loop at lines 1621-1652 runs AFTER children complete (children are launched synchronously at line 1507). Golden refs must be available BEFORE children start, because the child's Phase 1 reads `golden_refs.json` during execution. Placing the copy after the children would cause every `--verify-correctness` run to fail with exit code 4 ("golden_refs.json not found").

- [ ] **Step 1: Add golden refs copy BEFORE the child subprocess loop (~line 1500)**

Insert BEFORE `for run in runs_to_execute:` (line 1507), after `runs_to_execute` is built (line 1505):

```python
    # Copy golden refs BEFORE spawning children — children need it during Phase 1.
    baseline_from = Path(args.baseline_from) if args.baseline_from else None
    if baseline_from and args.verify_correctness:
        golden_src = baseline_from / "json" / "golden_refs.json"
        golden_dst = json_dir / "golden_refs.json"
        if golden_src.exists():
            shutil.copy2(str(golden_src), str(golden_dst))
            print(f"Imported golden references from {golden_src}")
        else:
            raise SystemExit(
                f"--verify-correctness requires golden_refs.json but "
                f"--baseline-from has none at {golden_src}"
            )
```

Note: `baseline_from` is defined locally here (not reusing `baseline_from_json` from the later block at line 1625, which has narrower scope). The `json_dir` variable is already defined earlier (~line 1480).

- [ ] **Step 2: Verify `json_dir` is defined before this insertion point**

Read line ~1480 to confirm `json_dir = out_root / "json"` exists before line 1500.

- [ ] **Step 3: Commit**

- [ ] **Step 2: Commit**

```bash
git add .claude/skills/ammo/scripts/run_vllm_bench_latency_sweep.py
git commit -m "feat(ammo): extend --baseline-from to copy golden_refs.json"
```

---

### Task 8: Rewrite Gate 5.1b in validation-defaults.md

**Files:**
- Modify: `.claude/skills/ammo/references/validation-defaults.md:198-229`

- [ ] **Step 1: Replace the Gate 5.1b section (lines 198-229)**

Replace the entire "Gate 5.1b: Baseline tensor comparison" section with:

```markdown
### Gate 5.1b: E2E Greedy Decode Correctness (HARD GATE)

**Gate 5.1b is a hard gate.** Correctness failure blocks the track from proceeding to latency benchmarks.

**Owner**: Champion via sweep script (deterministic — no validator involvement).

**Mechanism**: The sweep script's Phase 1 runs GSM8K greedy decode with `logprobs=5`, comparing optimized outputs against golden refs captured in Stage 1.

**Invocation**:
- Stage 1 (capture): `--capture-golden-refs` → saves `json/golden_refs.json`
- Stage 5 (verify): `--verify-correctness --baseline-from $STAGE1_DIR` → writes `json/correctness_verdict.json`

**Two modes** (selected via `--correctness-mode`):

| Mode | Default for | Gate logic |
|------|-------------|------------|
| `exact_greedy` | BF16/non-quantization tracks | Every token must match. `--max-divergent-positions 0` by default. |
| `topk_relaxed` | FP8/quantization tracks | Bidirectional top-5 containment. `--max-topk-failures-pct 5.0` by default. Length mismatches count as failures. |

**GSM8K accuracy gate** (quantization tracks only): The optimized model cannot get ANY question wrong that the baseline got correct ("zero questions lost"). This is a superset check, not a percentage threshold. Enabled automatically when `--correctness-mode topk_relaxed`.

**Self-consistency check** (Stage 1 only): Golden ref capture runs prompts twice to verify greedy decode is deterministic. If non-deterministic, metadata records `deterministic: false` and recommends `topk_relaxed` for Stage 5.

**Exit codes**: 3 = correctness FAIL (real divergence), 4 = infrastructure error (retry).

**Replaces**: Component-level tensor capture/compare (retired). The `NOT_APPLICABLE.md` escape clause is eliminated — E2E correctness works for all module types.
```

- [ ] **Step 2: Commit**

```bash
git add .claude/skills/ammo/references/validation-defaults.md
git commit -m "docs(ammo): rewrite Gate 5.1b definition for E2E correctness via GSM8K"
```

---

### Task 9: Update parallel-tracks.md Pass Criteria

**Files:**
- Modify: `.claude/skills/ammo/orchestration/parallel-tracks.md:120-151` (validator spawn), `:159-168` (verification layers), `:189-207` (pass criteria)

- [ ] **Step 1: Update Validation Spawn Protocol (lines 120-151)**

In the validator spawn prompt template, change:

```
    Write YOUR OWN independent tests and benchmarks (Gates 5.1, 5.2, 5.3).
```

to:

```
    Write YOUR OWN independent synthetic correctness tests (Gate 5.1a only).
    Gates 5.1b, 5.2, 5.3 are handled by the champion via the sweep script.
```

Remove `kernel-benchmark-template.py` from Key references (line 144 in the validator spawn prompt).

- [ ] **Step 2: Update Two Layers of Verification (lines 159-168)**

Replace the content of the verification layers section:

```markdown
### Two Layers of Verification

```
Layer 1: Independent Validator (Sonnet)
  Writes OWN synthetic correctness tests (Gate 5.1a)
  Reports raw structured results — no interpretation

Layer 2: Champion (Opus)
  Runs sweep with --verify-correctness (Gates 5.1b/5.2/5.3)
  Evaluates E2E results against min_e2e_improvement_pct threshold
  Cross-checks Gate 5.1a against correctness_verdict.json
  Writes final validation_results.md with evidence chain
```
```

- [ ] **Step 3: Update Pass Criteria (lines 200-207)**

Replace the Gate 5.1 bullet:

```markdown
- Gate 5.1: Correctness — both sub-gates must pass:
  - 5.1a: Validator's independent synthetic correctness tests pass
  - 5.1b: Sweep script `--verify-correctness` verdict is PASS in `correctness_verdict.json` (deterministic — no N/A escape)
```

- [ ] **Step 4: Commit**

```bash
git add .claude/skills/ammo/orchestration/parallel-tracks.md
git commit -m "docs(ammo): update parallel-tracks for Gate 5.1b redesign (validator scoped to 5.1a)"
```

---

### Task 10: Update integration-logic.md for Stage 6 Correctness Check

**Files:**
- Modify: `.claude/skills/ammo/orchestration/integration-logic.md:49-76`

- [ ] **Step 1: Add correctness check to Combined Validation Workflow (after line 66)**

After the existing `# Run combined E2E benchmark` line, add:

```markdown
# Run combined correctness check (mandatory for multi-candidate integration)
python .claude/skills/ammo/scripts/run_vllm_bench_latency_sweep.py \
    --artifact-dir {artifact_dir} --labels opt \
    --baseline-from {stage1_dir} --verify-correctness \
    --correctness-mode topk_relaxed --correctness-num-questions 50

# If correctness fails: bisect — drop track with worst individual failure rate, re-run
```

- [ ] **Step 2: Add correctness to Combined Result Evaluation table (line 73-76)**

Add a new row:

```markdown
| Combined correctness fails (topk or accuracy gate) | Bisect: drop worst track, re-validate smaller combination |
```

- [ ] **Step 3: Commit**

```bash
git add .claude/skills/ammo/orchestration/integration-logic.md
git commit -m "docs(ammo): add Stage 6 integration correctness check for cumulative quantization guard"
```

---

### Task 11: Narrow Validator Agent Definition

**Files:**
- Modify: `.claude/agents/ammo-impl-validator.md`

- [ ] **Step 1: Read the current validator agent definition**

Read the file to identify exact sections and line numbers for removal.

- [ ] **Step 2: Remove Gates 5.1b, 5.2, 5.3 sections**

Remove the following sections entirely:
- Gate 5.1b: Baseline Tensor Comparison section
- Gate 5.2: Independent Kernel Benchmarks section
- Gate 5.3a: Kernel Execution Proof section
- Gate 5.3b: E2E Sweep section
- Per-BS Tiered Verdict section

- [ ] **Step 3: Update frontmatter and intro**

Change description to reflect narrowed scope: Gate 5.1a synthetic kernel tests only.

- [ ] **Step 4: Add note about other gates**

Add after Gate 5.1a section:

```markdown
> **Note**: Gates 5.1b (E2E correctness), 5.2 (kernel benchmarks), and 5.3 (E2E latency)
> are now deterministic outputs of the sweep script. The champion runs these via
> `--verify-correctness` / `--capture-golden-refs`. See the Gate 5.1b redesign spec.
```

- [ ] **Step 5: Strip validation report template and DA checks**

Remove Gates 5.2, 5.3a, 5.3b from the report template. Remove DA checks that reference those gates.

- [ ] **Step 6: Commit**

```bash
git add .claude/agents/ammo-impl-validator.md
git commit -m "docs(ammo): narrow validator agent to Gate 5.1a only"
```

---

### Task 12: Update Champion Agent Definition

**Files:**
- Modify: `.claude/agents/ammo-impl-champion.md`

- [ ] **Step 1: Read the current champion agent definition**

Read the file to identify the tensor capture section.

- [ ] **Step 2: Replace tensor capture section with sweep-based correctness**

Remove the "Baseline Tensor Capture (Gate 5.1b — BLOCKING)" section and its checkpoint. Replace with:

```markdown
### E2E Correctness (Gate 5.1b — via Sweep Script)

Gate 5.1b is handled by the sweep script's Phase 1 correctness check. When running
the E2E sweep in Stage 5:

```bash
python .claude/skills/ammo/scripts/run_vllm_bench_latency_sweep.py \
    --artifact-dir $ARTIFACT_DIR --labels opt \
    --baseline-from $STAGE1_DIR --verify-correctness \
    --correctness-mode {exact_greedy|topk_relaxed}
```

- Use `exact_greedy` (default) for BF16/non-quantization tracks
- Use `topk_relaxed` for FP8/quantization tracks (includes "zero questions lost" accuracy gate)
- If correctness fails (exit code 3), the sweep stops before latency — fix the kernel before re-running
- No separate delegate or checkpoint needed — correctness is verified inline with the sweep
```

- [ ] **Step 3: Update validation request section**

Remove references to baseline_tensors from the VALIDATION_REQUEST message. Validator now only runs Gate 5.1a.

- [ ] **Step 4: Commit**

```bash
git add .claude/agents/ammo-impl-champion.md
git commit -m "docs(ammo): update champion agent for sweep-based Gate 5.1b"
```

---

### Task 13: Delete Old Template Files

**Files:**
- Delete: `.claude/skills/ammo/references/tensor-capture-template.py`
- Delete: `.claude/skills/ammo/references/tensor-compare-template.py`

- [ ] **Step 1: Delete template files**

```bash
git rm .claude/skills/ammo/references/tensor-capture-template.py
git rm .claude/skills/ammo/references/tensor-compare-template.py
```

- [ ] **Step 2: Commit**

```bash
git commit -m "chore(ammo): delete retired tensor capture/compare templates (replaced by Gate 5.1b redesign)"
```

---

### Task 14: Final Integration Test (Manual, GPU Required)

**Files:** None (verification only)

- [ ] **Step 1: Run all unit tests**

```bash
cd /home/jinhun/vllm
source .venv/bin/activate
PYTHONPATH=.claude/skills/ammo/scripts pytest .claude/skills/ammo/scripts/test_correctness_comparator.py -v
```

Expected: All tests PASS.

- [ ] **Step 2: Test golden ref capture (requires GPU + model)**

```bash
source .venv/bin/activate
CVD=$(python .claude/skills/ammo/scripts/gpu_reservation.py reserve --num-gpus 1) && \
CUDA_VISIBLE_DEVICES=$CVD python .claude/skills/ammo/scripts/run_vllm_bench_latency_sweep.py \
    --artifact-dir /tmp/test_gate_5_1b --labels baseline \
    --capture-golden-refs --correctness-num-questions 5
```

Expected: `[correctness] Golden refs saved to .../json/golden_refs.json`, `[correctness] Self-consistency check PASSED`

- [ ] **Step 3: Test verify-correctness with same model (should PASS)**

```bash
CVD=$(python .claude/skills/ammo/scripts/gpu_reservation.py reserve --num-gpus 1) && \
CUDA_VISIBLE_DEVICES=$CVD python .claude/skills/ammo/scripts/run_vllm_bench_latency_sweep.py \
    --artifact-dir /tmp/test_gate_5_1b --labels opt --out-name e2e_opt \
    --baseline-from /tmp/test_gate_5_1b/e2e_latency \
    --verify-correctness --correctness-num-questions 5
```

Expected: `[correctness] Verdict: PASS`, exit code 0.

- [ ] **Step 4: Test mutual exclusivity validation**

```bash
python .claude/skills/ammo/scripts/run_vllm_bench_latency_sweep.py \
    --artifact-dir /tmp/test --capture-golden-refs --verify-correctness 2>&1
```

Expected: `--capture-golden-refs and --verify-correctness are mutually exclusive`

- [ ] **Step 5: Clean up test artifacts**

```bash
rm -rf /tmp/test_gate_5_1b
```

---

### Task 15: Update impl-track-rules.md Validator Scope

**Files:**
- Modify: `.claude/skills/ammo/references/impl-track-rules.md:31-33,54`

This file directly instructs champion and validator on their scope. Without updating it, validators will be told to write benchmarks and run E2E sweeps they no longer own.

- [ ] **Step 1: Read the file to verify line numbers**

Read `.claude/skills/ammo/references/impl-track-rules.md` and locate:
- Lines 31-33: "Layer 1: Independent Validator" description listing Gates 5.1/5.2/5.3
- Line 54: "Validator re-runs ALL gates from scratch"

- [ ] **Step 2: Update Layer 1 validator scope**

Change lines 31-33 from describing Gates 5.1/5.2/5.3 to Gate 5.1a only:

```
Layer 1: Independent Validator (Sonnet)
  Writes OWN synthetic correctness tests (Gate 5.1a only)
  Reports raw correctness results — no interpretation
```

- [ ] **Step 3: Update re-run scope**

Change line 54 from "Validator re-runs ALL gates" to:

```
Validator re-runs Gate 5.1a from scratch with fresh independent tests
```

- [ ] **Step 4: Commit**

```bash
git add .claude/skills/ammo/references/impl-track-rules.md
git commit -m "docs(ammo): update impl-track-rules validator scope to Gate 5.1a only"
```

---

### Task 16: Update ammo-delegate.md Reference Table

**Files:**
- Modify: `.claude/agents/ammo-delegate.md:85`

- [ ] **Step 1: Read the file and locate the kernel-benchmark-template reference**

- [ ] **Step 2: Update the reference**

Add a note that `kernel-benchmark-template.py` is now champion-owned (Gate 5.2), not validator-directed via delegate.

- [ ] **Step 3: Commit**

```bash
git add .claude/agents/ammo-delegate.md
git commit -m "docs(ammo): update delegate reference table for Gate 5.2 ownership change"
```

---

### Task 17: Deprecate Old Tensor Capture Design Spec

**Files:**
- Modify: `.claude/skills/ammo/docs/specs/2026-03-27-tensor-capture-gate-design.md`

- [ ] **Step 1: Add deprecation header at the top of the file**

Insert at line 1:

```markdown
> **DEPRECATED (2026-04-01)**: This design was superseded by the Gate 5.1b Redesign.
> See `docs/superpowers/specs/2026-04-01-gate-5-1b-redesign-design.md` for the current design.

```

- [ ] **Step 2: Commit**

```bash
git add .claude/skills/ammo/docs/specs/2026-03-27-tensor-capture-gate-design.md
git commit -m "docs(ammo): deprecate old tensor capture gate design spec"
```

---

### Task 18: Update SKILL.md Task Graph and Gate Ownership (CRITICAL)

**Files:**
- Modify: `.claude/skills/ammo/SKILL.md:~160,210,263,340,449`

This is the top-level orchestration skill. Without updating it, the orchestrator will spawn validators with wrong scope and give researchers wrong instructions.

- [ ] **Step 1: Read SKILL.md and locate the 5 affected sections**

Read the full file. Find:
- ~Line 160: Stage 1 researcher instruction (missing `--capture-golden-refs`)
- ~Line 210: Stage 4-5 description (validator writes benchmarks, runs E2E)
- ~Line 263: T8c task graph entry (validator runs Gates 5.1/5.2/5.3)
- ~Line 340: DA verification (validator after Gates 5.1/5.2/5.3)
- ~Line 449: Helper scripts (sweep script description)

- [ ] **Step 2: Update Stage 1 researcher instruction**

Add `--capture-golden-refs` to the Stage 1 baseline sweep command.

- [ ] **Step 3: Update Stage 4-5 description**

Change validator description from "independently writes its own correctness tests, benchmark scripts, and runs E2E sweeps" to "independently writes its own synthetic correctness tests (Gate 5.1a)".

- [ ] **Step 4: Update T8c task graph entry**

Change from `"Independent validation Gates 5.1/5.2/5.3 (validator)"` to `"Gate 5.1a validation (validator) + E2E sweep with --verify-correctness (champion)"`.

- [ ] **Step 5: Update DA verification**

Change from `"After Gates 5.1/5.2/5.3"` to `"After Gate 5.1a"`. Remove `"Gate 5.2 cross-check"`.

- [ ] **Step 6: Update helper scripts section**

Add `--capture-golden-refs` and `--verify-correctness` to the sweep script's flag list.

- [ ] **Step 7: Commit**

```bash
git add .claude/skills/ammo/SKILL.md
git commit -m "docs(ammo): update SKILL.md task graph and gate ownership for 5.1b redesign"
```

---

### Task 19: Update Transcript Monitor Gate Completeness Check (CRITICAL)

**Files:**
- Modify: `.claude/agents/ammo-transcript-monitor.md:208`

Without this fix, the monitor will produce **false positive CRITICAL flags** on valid champion/validator workflows, claiming gates are missing when they're actually handled by the sweep script.

- [ ] **Step 1: Read the file and locate line 208**

Find the gate completeness check that says "all 5.1/5.2/5.3a/5.3b gates run before declaring success".

- [ ] **Step 2: Update to reflect new ownership model**

Change to reflect: validator owns Gate 5.1a only. Champion owns Gates 5.1b/5.2/5.3 via sweep script (`correctness_verdict.json` for 5.1b, sweep output for 5.2/5.3).

- [ ] **Step 3: Commit**

```bash
git add .claude/agents/ammo-transcript-monitor.md
git commit -m "docs(ammo): update transcript monitor gate completeness check for 5.1b redesign"
```

---

### Task 20: Update Resolver Post-Merge Testing (CRITICAL)

**Files:**
- Modify: `.claude/agents/ammo-resolver.md:94`

- [ ] **Step 1: Read and locate line 94**

Find: `"Run BOTH tracks' existing validator correctness tests on the merged code"`

- [ ] **Step 2: Update to include sweep correctness**

Add a step for running `--verify-correctness` on the merged code for E2E coverage, in addition to the 5.1a pytest tests:

```markdown
Run BOTH tracks' Gate 5.1a validator tests on the merged code, then run the sweep
with `--verify-correctness` to validate E2E correctness of the merged combination.
```

- [ ] **Step 3: Commit**

```bash
git add .claude/agents/ammo-resolver.md
git commit -m "docs(ammo): update resolver post-merge testing for 5.1b redesign"
```

---

### Task 21: Update Test Agent Scenarios (MEDIUM)

**Files:**
- Modify: `.claude/skills/ammo/tests/agents/test-impl-champion.md`
- Modify: `.claude/skills/ammo/tests/agents/test-implementer.md`
- Modify: `.claude/skills/ammo/tests/agents/test-transcript-monitor.md`

These test scenarios describe validator reporting Gates 5.2/5.3 results — now those come from the sweep script. Without updating, test evals will score agents against an obsolete workflow.

- [ ] **Step 1: Update test-impl-champion.md**

Read the file and update scenarios IC2, IC3, IC7, IC10 to frame Gates 5.2/5.3 results as coming from the sweep script output, not the validator. Keep IC1, IC4, IC5, IC8 unchanged (Gate 5.1a scenarios remain valid).

- [ ] **Step 2: Update test-implementer.md**

Read the file and update scenario I11 (lines 308-332): change "Validator reports Gate 5.3 results" to "Sweep script reports Gate 5.3 results". Remove "Champion requests validator to run crossover probing" — champion now handles this via sweep.

- [ ] **Step 3: Update test-transcript-monitor.md**

Read the file and update lines 106-107 (Gate 5.2 source) and line 227 (Gate 5.3b source) to reflect sweep ownership.

- [ ] **Step 4: Commit**

```bash
git add .claude/skills/ammo/tests/agents/test-impl-champion.md \
       .claude/skills/ammo/tests/agents/test-implementer.md \
       .claude/skills/ammo/tests/agents/test-transcript-monitor.md
git commit -m "test(ammo): update agent test scenarios for Gate 5.1b redesign ownership changes"
```
