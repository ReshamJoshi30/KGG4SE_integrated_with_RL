# rl/reward_functions.py
"""
Reward shaping for KG repair RL agent.

Design principles
-----------------
1. Sparse primary signal (+10 consistency) is kept — it's the true objective.
2. Dense secondary signals guide the agent toward the primary reward.
3. All penalties are calibrated so a random agent gets negative expected reward,
   pushing it to learn something useful rather than no-op indefinitely.
4. Proportional penalties for regressions (not flat -3) so the agent can
   distinguish "introduced 1 new violation" from "destroyed the KG."
5. Step efficiency bonus: rewards finishing quickly, discouraging the agent
   from taking the maximum number of steps on an already-consistent KG.

Reward breakdown
----------------
+10.0   KG became fully consistent  (primary goal achieved)
-10.0   Consistent KG became inconsistent  (regression to worse state)
+ 2.0   Each disjoint violation resolved
+ 2.0   Each transitive disjoint violation resolved
+ 2.0   Each unsat class resolved
+ 1.0   Each parse/process error resolved
- 2.0   Per NEW violation introduced  (proportional, replaces flat -3)
- 0.5   No net change (no-op penalty)
- 0.1   Per step cost (encourages efficiency)

Total if KG fixed in 1 step: +10 - 0.1 = +9.9
Total for no-op loop: -0.6/step-> agent learns to avoid wasted steps
"""
from __future__ import annotations
from typing import Any, Dict, Tuple


def compute_reward(
    new_report: Dict[str, Any],
    old_report: Dict[str, Any],
    step_num: int = 0,
    max_steps: int = 15,
) -> Tuple[float, Dict[str, Any]]:
    """
    Compute shaped reward from before/after reasoner reports.

    Args:
        new_report: Reasoner report AFTER applying the repair action.
        old_report: Reasoner report BEFORE the action.
        step_num:   Current step number (used for step cost and efficiency bonus).
        max_steps:  Max steps per episode (used for efficiency bonus).

    Returns:
        (reward, metrics_dict)
    """
    if old_report is None:
        old_report = {}

    was_consistent = old_report.get("is_consistent", False)
    now_consistent = new_report.get("is_consistent", False)

    old_disj  = len(old_report.get("disjoint_violations",            []))
    new_disj  = len(new_report.get("disjoint_violations",            []))
    old_trans = len(old_report.get("transitive_disjoint_violations", []))
    new_trans = len(new_report.get("transitive_disjoint_violations", []))
    old_unsat = len(old_report.get("unsat_classes",                  []))
    new_unsat = len(new_report.get("unsat_classes",                  []))
    old_parse = old_report.get("parse_errors",   0)
    new_parse = new_report.get("parse_errors",   0)
    old_proc  = old_report.get("process_errors", 0)
    new_proc  = new_report.get("process_errors", 0)

    reward = 0.0

    # ── Primary: consistency flip ────────────────────────────────────────────
    if not was_consistent and now_consistent:
        # Efficiency bonus: reward finishing faster
        # +10 if done in 1 step, scaling down to +10 at step max_steps
        efficiency = max(0.0, 1.0 - (step_num / max(max_steps, 1)) * 0.5)
        reward += 10.0 + (2.0 * efficiency)   # up to +12 for very fast finish
    elif was_consistent and not now_consistent:
        reward -= 10.0

    # ── Secondary: violation deltas ──────────────────────────────────────────
    disj_delta  = old_disj  - new_disj    # positive = improvement
    trans_delta = old_trans - new_trans
    unsat_delta = old_unsat - new_unsat
    parse_delta = old_parse - new_parse
    proc_delta  = old_proc  - new_proc

    # Reward for fixing violations
    reward += 2.0 * max(0, disj_delta)
    reward += 2.0 * max(0, trans_delta)
    reward += 2.0 * max(0, unsat_delta)
    reward += 1.0 * max(0, parse_delta)
    reward += 1.0 * max(0, proc_delta)

    # Proportional penalty for introducing NEW violations
    # (flat -3 was too blunt — couldn't distinguish severity)
    new_violations = (
        max(0, -disj_delta) +
        max(0, -trans_delta) +
        max(0, -unsat_delta)
    )
    if new_violations > 0:
        reward -= 2.0 * new_violations   # -2 per new violation introduced

    # ── Per-step cost: discourages unnecessary steps ─────────────────────────
    # Small constant penalty so the agent learns to repair efficiently.
    # Without this, an agent that solved the KG at step 1 but keeps stepping
    # until max_steps gets the same total as one that stopped at step 1.
    reward -= 0.1

    # ── No-op penalty: extra penalty for no useful change ───────────────────
    net_change = abs(disj_delta) + abs(trans_delta) + abs(unsat_delta)
    consistency_changed = (was_consistent != now_consistent)
    if net_change == 0 and not consistency_changed:
        reward -= 0.5   # on top of the per-step cost = -0.6 total for wasted step

    metrics = {
        "unsat":           unsat_delta,
        "disj":            disj_delta,
        "trans":           trans_delta,
        "parse":           parse_delta,
        "proc":            proc_delta,
        "new_violations":  new_violations,
        "was_consistent":  was_consistent,
        "now_consistent":  now_consistent,
        "step_num":        step_num,
    }

    return reward, metrics