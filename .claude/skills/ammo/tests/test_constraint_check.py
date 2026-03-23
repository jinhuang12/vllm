#!/usr/bin/env python3
"""TDD tests for constraint_check.py — debate pre-screening gate (R1).

Tests cover:
  TestSmemBudget: SMEM budget checks (5 tests)
  TestTechniqueBlacklist: technique blacklist matching (3 tests)
  TestParseProposalConstraints: proposal markdown parsing (2 tests)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from constraint_check import (
    check_smem_budget,
    check_technique_blacklist,
    parse_proposal_constraints,
    ConstraintResult,
)


# ---------------------------------------------------------------------------
# TestSmemBudget
# ---------------------------------------------------------------------------

class TestSmemBudget:
    """Tests for check_smem_budget."""

    def test_within_budget_passes(self):
        """SMEM usage below limit should pass."""
        result = check_smem_budget(smem_bytes=49152, sm_smem_limit=102400)
        assert result.passed is True
        assert result.skipped is False

    def test_exceeds_budget_fails(self):
        """SMEM usage above limit should fail with KB-formatted reason."""
        result = check_smem_budget(smem_bytes=110592, sm_smem_limit=102400)
        assert result.passed is False
        assert result.skipped is False
        # Reason should mention KB values
        assert "KB" in result.reason

    def test_at_boundary_passes(self):
        """SMEM usage exactly equal to limit should pass."""
        result = check_smem_budget(smem_bytes=102400, sm_smem_limit=102400)
        assert result.passed is True

    def test_double_buffer_doubles_effective_smem(self):
        """With double_buffered=True, effective SMEM = smem_bytes * 2."""
        # 60 KB * 2 = 120 KB > 102.4 KB (L40S limit)
        result = check_smem_budget(smem_bytes=61440, sm_smem_limit=102400, double_buffered=True)
        assert result.passed is False
        assert "KB" in result.reason

    def test_none_smem_bytes_skips(self):
        """If smem_bytes is None, the check should be skipped (passed=True, skipped=True)."""
        result = check_smem_budget(smem_bytes=None, sm_smem_limit=102400)
        assert result.passed is True
        assert result.skipped is True


# ---------------------------------------------------------------------------
# TestTechniqueBlacklist
# ---------------------------------------------------------------------------

class TestTechniqueBlacklist:
    """Tests for check_technique_blacklist."""

    def test_blocked_technique_fails(self):
        """A technique matching a blacklist entry should fail."""
        blacklist = ["split-k", "persistent warp"]
        result = check_technique_blacklist("Split-K GEMM tiling", blacklist)
        assert result.passed is False
        assert "split-k" in result.reason.lower()

    def test_non_blacklisted_passes(self):
        """A technique not in the blacklist should pass."""
        blacklist = ["split-k", "persistent warp"]
        result = check_technique_blacklist("Triton GEMM with fused epilogue", blacklist)
        assert result.passed is True

    def test_empty_blacklist_passes(self):
        """An empty blacklist should always pass."""
        result = check_technique_blacklist("Split-K GEMM tiling", [])
        assert result.passed is True


# ---------------------------------------------------------------------------
# TestParseProposalConstraints
# ---------------------------------------------------------------------------

class TestParseProposalConstraints:
    """Tests for parse_proposal_constraints."""

    def test_extracts_smem_claim_kb(self):
        """Parser should extract SMEM claim in KB and convert to bytes."""
        proposal = """
## Candidate Specification

Technique: Triton tiled GEMM

Shared memory usage: 48 KB per warp

Implementation notes: standard tiling approach.
"""
        result = parse_proposal_constraints(proposal)
        assert result["smem_bytes"] == 48 * 1024

    def test_extracts_technique_from_section(self):
        """Parser should extract the technique name from Candidate Specification section."""
        proposal = """
## Candidate Specification

Technique: Split-K GEMM with persistent warps

Shared memory: 32 KB
"""
        result = parse_proposal_constraints(proposal)
        assert result["technique"] is not None
        assert "Split-K" in result["technique"] or "split-k" in result["technique"].lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
