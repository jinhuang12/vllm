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
    assert result["verdict"] == "FAIL"  # overall verdict overridden by accuracy gate
    assert result["gsm8k_accuracy_gate"]["verdict"] == "FAIL"
    assert result["gsm8k_accuracy_gate"]["lost_questions"] == [0]


def test_accuracy_gate_override_token_pass_but_accuracy_fail():
    """Token-level passes but accuracy gate loses a question → overall FAIL."""
    from run_vllm_bench_latency_sweep import _compare_correctness
    # Tokens match perfectly (containment will pass trivially)
    golden = [{"token_ids": [10], "logprobs": [{"top_logprobs": {"10": -0.1, "20": -2.0, "30": -3.0, "40": -4.0, "50": -5.0}}],
               "text": "", "num_tokens": 1, "prompt_index": 0},
              {"token_ids": [10], "logprobs": [{"top_logprobs": {"10": -0.1, "20": -2.0, "30": -3.0, "40": -4.0, "50": -5.0}}],
               "text": "", "num_tokens": 1, "prompt_index": 1}]
    opt = [{"token_ids": [10], "logprobs": [{"top_logprobs": {"10": -0.1, "20": -2.0, "30": -3.0, "40": -4.0, "50": -5.0}}],
            "text": "", "num_tokens": 1, "prompt_index": 0},
           {"token_ids": [10], "logprobs": [{"top_logprobs": {"10": -0.1, "20": -2.0, "30": -3.0, "40": -4.0, "50": -5.0}}],
            "text": "", "num_tokens": 1, "prompt_index": 1}]
    labels = [42, 99]
    baseline_preds = [42, 99]  # both correct
    opt_preds = [42, 0]  # Q1 wrong
    result = _compare_correctness(golden_refs=golden, opt_outputs=opt, mode="topk_relaxed",
                                   max_divergent_positions=0, max_topk_failures_pct=100.0,
                                   labels=labels, baseline_preds=baseline_preds, opt_preds=opt_preds)
    assert result["containment_failures"] == 0  # token-level passes
    assert result["gsm8k_accuracy_gate"]["verdict"] == "FAIL"
    assert result["gsm8k_accuracy_gate"]["lost_questions"] == [1]
    assert result["verdict"] == "FAIL"  # accuracy gate overrides to FAIL


def test_accuracy_gate_mismatched_question_counts():
    """Baseline had more questions than opt → truncate, don't false-fail."""
    from run_vllm_bench_latency_sweep import _compare_correctness
    golden = [{"token_ids": [10], "logprobs": [{"top_logprobs": {"10": -0.1, "20": -2.0, "30": -3.0, "40": -4.0, "50": -5.0}}],
               "text": "", "num_tokens": 1, "prompt_index": 0}]
    opt = [{"token_ids": [10], "logprobs": [{"top_logprobs": {"10": -0.1, "20": -2.0, "30": -3.0, "40": -4.0, "50": -5.0}}],
            "text": "", "num_tokens": 1, "prompt_index": 0}]
    # Baseline had 3 questions, opt only answered 1
    labels = [42, 99, 77]
    baseline_preds = [42, 99, 77]  # all correct
    opt_preds = [42]  # correct for the one question it answered
    result = _compare_correctness(golden_refs=golden, opt_outputs=opt, mode="topk_relaxed",
                                   max_divergent_positions=0, max_topk_failures_pct=100.0,
                                   labels=labels, baseline_preds=baseline_preds, opt_preds=opt_preds)
    # Should only compare the 1 overlapping question, not false-fail on indices 1-2
    assert result["gsm8k_accuracy_gate"]["verdict"] == "PASS"
    assert result["gsm8k_accuracy_gate"]["lost_questions"] == []


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
