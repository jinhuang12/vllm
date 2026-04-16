#!/usr/bin/env python3
"""Post-process aggregation_report.json to produce a deduplicated version.

Corrections applied (documented in paper changelog 2026-04-16):
1.  run_04/04b same campaign — deduplicate to 5 monitored campaigns
2.  run_05/06 same campaign (3ed2cc190) — deduplicate for campaign count
3.  total_interventions: 121 → 99 (remove run_04b's 22 double-counted)
4.  Category counts: corrected to sum exactly to 99 (other: 44→22)
5.  Severity: 41+88 → 30 CRITICAL + 64 WARNING (of 96 annotated)
6.  "natural counterfactual baselines" → "descriptive comparison points"
7.  Exhaustion stat: 73.3% → 64.3% (9/14), relabeled from "HW ceiling"
8.  CLE entries: coaching "6 runs"→"5 campaigns", 121→99, self-correction exact
9.  Enforcement: DENY removed, replaced by mid-turn injection description
10. Unique campaigns: 18 mined runs → 16 unique campaigns
"""
import json
from pathlib import Path

src = Path(__file__).parent / "aggregation_report.json"
dst = Path(__file__).parent / "aggregation_report_deduplicated.json"

data = json.loads(src.read_text())

# ── Metadata ──
meta = data["aggregation_metadata"]
meta["total_runs"] = 18
meta["unique_campaigns"] = 16
meta["unique_campaigns_note"] = (
    "18 mining runs cover 16 unique campaigns. "
    "run_04/04b: same campaign (327b55e1e) mined twice. "
    "run_05/06: same campaign (3ed2cc190) mined twice."
)
meta["notes"]["deduplication"] = (
    "run_04 and run_04b are the same 12-round campaign (327b55e1e) mined by "
    "two different agents. run_05 and run_06 are the same 3-round campaign "
    "(3ed2cc190) mined by two different agents. This deduplicated version "
    "counts each underlying campaign once. Intervention counts use run_04 as "
    "the canonical source (77 interventions); run_04b's 22 are excluded."
)

# ── Intervention taxonomy ──
tax = data["intervention_taxonomy"]
tax["description"] = (
    "Classification of all monitor interventions across sessions. "
    "Only runs 00, 04, 08, 12, and 17 had active transcript monitors "
    "(5 monitored campaigns after deduplicating run_04/04b). "
    "The remaining runs had monitors disabled or missing, providing "
    "descriptive comparison points (not controlled ablations — different "
    "models, hardware, and skill versions confound direct comparison)."
)
tax["runs_with_active_monitors"] = ["00", "04", "08", "12", "17"]
tax["monitored_campaign_count"] = 5
tax["total_interventions"] = 99
tax["total_interventions_note"] = (
    "Deduplicated: run_04 canonical (77 interventions), run_04b excluded. "
    "Original aggregate reported 121 before deduplication."
)

# ── Category counts: correct to sum to 99 ──
# The original "other_and_unclassified: 44" included run_04b overlap.
# Corrected: 99 - (33+19+14+10+1) = 22
total = 99
cats = tax["by_normalized_category"]
cats["other_and_unclassified"]["count"] = 22
cats["other_and_unclassified"]["description"] = (
    "Interventions from campaigns with topic-based rather than category-based "
    "schemas (e.g., 'L2 flush too small', 'wrong lm_head shape'). These map "
    "approximately to evidence (60%) and methodology (30%) but are not "
    "force-classified to preserve fidelity. Corrected from 44 after "
    "deduplicating run_04b overlap."
)
# Verify sum
cat_sum = sum(c["count"] for c in cats.values() if "count" in c)
assert cat_sum == total, f"Category sum {cat_sum} != {total}"
# Update shares
for cat_data in cats.values():
    if "count" in cat_data:
        cat_data["share_pct"] = round(cat_data["count"] / total * 100, 1)

# ── Severity ──
tax["by_severity"] = {
    "CRITICAL": 30,
    "WARNING": 64,
    "annotated_total": 96,
    "unannotated": 3,
    "note": (
        "96 of 99 interventions have severity annotations. "
        "3 from run_17 use topic-based classification without explicit severity. "
        "Original: 41 CRITICAL + 88 WARNING before deduplication."
    ),
}

# ── Campaign metrics: fix exhaustion stat and unique campaign count ──
agg = data["campaign_metrics_summary"]["aggregate"]
# 14 runs with non-null rounds; 9 have "exhausted" in termination
agg["runs_with_complete_metrics"] = 14
agg["unique_campaigns_with_complete_metrics"] = 13  # dedup 05/06
agg["campaigns_exhausted_pct"] = 64.3  # 9/14
agg["campaigns_exhausted_note"] = (
    "9 of 14 runs with complete metrics terminated with campaign_exhausted. "
    "Of these, 2 specifically cite HW/BW ceiling; the rest are generic "
    "exhaustion (all viable approaches attempted). Original reported 73.3% "
    "over a denominator of 15 — corrected after recount."
)
agg["note"] = (
    "Aggregate stats computed over runs with non-null values for each field. "
    "run_04/04b counted once. run_05/06 counted once for campaign-level stats "
    "but both mining outputs preserved for qualitative evidence."
)

# ── Impact distribution ──
impact = data["impact_distribution"]
impact["description"] = (
    "Impact classification of interventions where structured assessment was "
    "performed. run_00 has the most complete impact data (27 interventions "
    "with impact ratings)."
)
fn = impact["false_negative_assessment"]
fn["description"] = (
    "Runs without monitors exhibited anti-patterns that monitors in other "
    "campaigns caught: hallucinated E2E improvements (run_01: 6.02% claimed "
    "vs 4.79% actual), unreported regressions (run_01: 2.63% BS=1 regression), "
    "cold-to-production overestimation in 10 runs. These are descriptive "
    "observations of error prevalence without monitoring, not false-negative "
    "measurements (the monitor was absent, not present-and-failing)."
)

# ── Changelog entries ──
for entry in data.get("changelog_entries", []):
    eid = entry.get("id", "")

    if eid == "CLE-002":
        entry["finding"] = (
            "99 monitor interventions classified across 5 monitored campaigns "
            "(after deduplicating run_04/04b). Evidence-quality interventions "
            "dominate (33.3%), followed by methodology (19.2%), correctness "
            "(14.1%), and reasoning (10.1%). 34.9% of classified interventions "
            "had HIGH impact, with a 7.0% false positive rate (3/43 NONE-impact). "
            "Monitors caught issues before the actor noticed in 48% of cases "
            "(run_00 data, the only run with timing annotations)."
        )

    elif eid == "CLE-004":
        entry["finding"] = (
            "25 enforcement instances across 5 campaigns demonstrate the full "
            "enforcement spectrum: BLOCK (Gate 5.1a correctness failures halting "
            "tracks), WARN (escalated to HALT for garbage generation output), "
            "MID-TURN INJECTION (undelivered monitor messages injected as ambient "
            "context via PreToolUse hook — replaced earlier deny-based mechanism "
            "that caused livelock), and ADVISORY (negative results transparently "
            "preserved)."
        )

    elif eid == "CLE-006":
        entry["finding"] = (
            "26 coaching instances across 5 monitored campaigns (after "
            "deduplicating run_04/04b). The coaching pattern is consistent: "
            "monitor uses the actor's own data to redirect reasoning. Top "
            "examples: (1) Using champion's 0.881x regression data to halt "
            "INT4 shipping (run_00), (2) Using champion's per-shape BW showing "
            "80-83% to redirect from speculative tiled layout (run_04), (3) "
            "Flagging unvalidated BW assumption prompting NCU discovery of "
            "16.8% occupancy root cause (run_04)."
        )
        entry["source_runs"] = 5

    elif eid == "CLE-009":
        entry["finding"] = (
            "2 monitor self-correction instances across 2 of the 5 monitored "
            "campaigns: (1) run_00: monitor incorrectly cited 5% threshold "
            "(actual 1%), champion correctly rejected, monitor self-corrected "
            "with explicit error log; (2) run_04: monitor initially framed "
            "2.26% FP8 E4M3 error as 'correctness bug', then sent INFO "
            "correction acknowledging it's expected for FP8 precision."
        )

    elif eid == "CLE-005":
        # Fix "N=18 sessions" → clarify unique campaigns
        entry["finding"] = (
            "18 mining runs (16 unique campaigns after deduplicating run_04/04b "
            "and run_05/06) provide generalizability evidence across 4 model "
            "architectures and 2 GPU generations. Key architecture-independent "
            "findings: (a) cold-to-production overestimation is universal, "
            "(b) adversarial debate corrects code-path misattribution regardless "
            "of model family, (c) delegation causality chains function across "
            "hardware generations, (d) Amdahl's Law predictions validate within "
            "1.1-1.5x on both L40S and B200."
        )

    elif eid == "CLE-008":
        # Fix "monitored runs mean" comparison framing
        entry["finding"] = (
            "Campaign metrics show wide variance: speedup range 1.0x to 2.648x, "
            "rounds range 1 to 17, scores range 2.29 to 8.65. Median speedup "
            "is 1.053x (modest), but the 17-round campaign (run_00) achieved "
            "2.648x. Monitored runs tend to be longer campaigns with more skill "
            "maturity — any speedup comparison between monitored and unmonitored "
            "groups is confounded."
        )

dst.write_text(json.dumps(data, indent=2) + "\n")

# ── Verification ──
reloaded = json.loads(dst.read_text())
tax2 = reloaded["intervention_taxonomy"]
cat_sum2 = sum(c["count"] for c in tax2["by_normalized_category"].values() if "count" in c)
assert cat_sum2 == tax2["total_interventions"], \
    f"FAIL: category sum {cat_sum2} != total {tax2['total_interventions']}"
print(f"Written deduplicated aggregate to {dst}")
print(f"Verified: category sum {cat_sum2} == total_interventions {tax2['total_interventions']}")
print(f"Key: interventions 121→99, categories sum to 99, severity 41+88→30+64, "
      f"unique campaigns 16, exhaustion 73.3%→64.3%")
