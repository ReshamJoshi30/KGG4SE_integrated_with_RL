"""
report_quality.py

Supervisor-ready quality & progress report for the KG repair framework.

Run after a training session to see:
  - KG statistics before and after repair
  - Inconsistency profile (violation breakdown by type)
  - RL training progress (rewards, success rate, epsilon decay)
  - Human vs RL action breakdown
  - What got fixed: concrete violations resolved
  - Saved plots (PNG) for slides / thesis

Usage
-----
    python report_quality.py
    python report_quality.py --before outputs/intermediate/merged_kg.owl
                             --after  outputs/rl_repair_steps/repaired_step_9.owl
                             --trace  outputs/rl_repair_traces/repair_trace.jsonl
                             --feedback outputs/models/human_feedback.jsonl
                             --plots  outputs/reports
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _short(iri: str) -> str:
    if "#" in iri:
        return iri.rsplit("#", 1)[-1]
    return iri.rsplit("/", 1)[-1]


def _run_reasoner(owl_path: str, reasoner: str = "hermit") -> dict:
    from reasoning.reasoner_wrapper import run_reasoner
    print(f"  Running {reasoner} on {Path(owl_path).name} ...")
    return run_reasoner(owl_path, reasoner=reasoner)


def _triple_count(owl_path: str) -> int:
    try:
        import rdflib
        g = rdflib.Graph()
        g.parse(str(owl_path))
        return len(g)
    except Exception:
        return -1


def _entity_counts(owl_path: str) -> dict:
    """Return {subjects, predicates, objects, individuals, classes}."""
    try:
        import rdflib
        from rdflib.namespace import RDF, OWL
        g = rdflib.Graph()
        g.parse(str(owl_path))
        subjects    = {str(s) for s, p, o in g if isinstance(s, rdflib.URIRef)}
        predicates  = {str(p) for s, p, o in g}
        objects_uri = {str(o) for s, p, o in g if isinstance(o, rdflib.URIRef)}
        individuals = {str(s) for s, p, o in g
                       if p == RDF.type and o == OWL.NamedIndividual}
        classes     = {str(o) for s, p, o in g
                       if p == RDF.type and o == OWL.Class}
        return {
            "triples":     len(g),
            "subjects":    len(subjects),
            "predicates":  len(predicates),
            "objects":     len(objects_uri),
            "individuals": len(individuals),
            "classes":     len(classes),
        }
    except Exception as e:
        return {"triples": -1, "error": str(e)}


def _load_trace(trace_path: str) -> list[dict]:
    p = Path(trace_path)
    if not p.exists():
        return []
    steps = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                steps.append(json.loads(line))
    return steps


def _load_feedback(fb_path: str) -> list[dict]:
    p = Path(fb_path)
    if not p.exists():
        return []
    records = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _load_episode_data(trace_path: str) -> dict:
    """
    Infer per-episode summary from the flat trace JSONL.
    Groups steps by detecting reward sign changes or reading episode markers.
    Returns raw step list for plotting.
    """
    return _load_trace(trace_path)


# ---------------------------------------------------------------------------
# Text report sections
# ---------------------------------------------------------------------------

W = 70  # console width

def _banner(title: str):
    print("\n" + "=" * W)
    print(f"  {title}")
    print("=" * W)


def _section(title: str):
    print(f"\n  -- {title} --")


def _row(label: str, val_before, val_after=None, unit: str = ""):
    if val_after is None:
        print(f"    {label:<40} {val_before}{unit}")
    else:
        delta = ""
        if isinstance(val_before, (int, float)) and isinstance(val_after, (int, float)):
            d = val_after - val_before
            sign = "+" if d >= 0 else ""
            delta = f"  ({sign}{d}{unit})"
        print(f"    {label:<40} {val_before}{unit}  ->  {val_after}{unit}{delta}")


def print_kg_statistics(before_counts: dict, after_counts: dict | None = None):
    _section("KG Statistics")
    for key in ("triples", "subjects", "predicates", "objects",
                "individuals", "classes"):
        b = before_counts.get(key, "?")
        a = after_counts.get(key, "?") if after_counts else None
        _row(key.capitalize(), b, a)


def print_consistency_profile(before_report: dict, after_report: dict | None = None):
    _section("Consistency Profile")

    def _safe_len(x):
        return len(x) if isinstance(x, list) else int(x)

    metrics = [
        ("is_consistent",         "Is Consistent"),
        ("unsat_classes",         "Unsatisfiable Classes"),
        ("disjoint_violations",   "Disjoint Violations"),
    ]

    for key, label in metrics:
        bv = before_report.get(key, "?")
        bv_display = bv if isinstance(bv, bool) else _safe_len(bv)
        if after_report:
            av = after_report.get(key, "?")
            av_display = av if isinstance(av, bool) else _safe_len(av)
            _row(label, bv_display, av_display)
        else:
            _row(label, bv_display)

    # Violation breakdown from all_issues
    b_issues = before_report.get("all_issues", {})
    a_issues = after_report.get("all_issues", {}) if after_report else {}

    if b_issues or a_issues:
        print()
        print("    Violation breakdown by type:")
        all_types = set(b_issues.get("by_type", {}).keys()) | \
                    set(a_issues.get("by_type", {}).keys())
        for vtype in sorted(all_types):
            bv = b_issues.get("by_type", {}).get(vtype, 0)
            av = a_issues.get("by_type", {}).get(vtype, 0) if a_issues else None
            _row(f"    {vtype}", bv, av)
        _row("    TOTAL violations",
             b_issues.get("total_violations", "?"),
             a_issues.get("total_violations", "?") if a_issues else None)


def print_repair_trace_summary(steps: list[dict]):
    if not steps:
        print("    No trace data found.")
        return
    _section("Repair Trace Summary")
    _row("Total repair steps taken", len(steps))

    total_reward = sum(s.get("reward", 0) for s in steps)
    _row("Total accumulated reward", f"{total_reward:.2f}")

    action_counts: dict[str, int] = {}
    for s in steps:
        atype = s.get("action", {}).get("action_type", "unknown")
        action_counts[atype] = action_counts.get(atype, 0) + 1

    print("\n    Actions used:")
    for atype, cnt in sorted(action_counts.items(), key=lambda x: -x[1]):
        print(f"      {atype:<40} x{cnt}")

    # Show which steps improved things
    improving = [s for s in steps if s.get("reward", 0) > 0]
    degrading  = [s for s in steps if s.get("reward", 0) < 0]
    neutral    = [s for s in steps if s.get("reward", 0) == 0]
    print(f"\n    Steps that improved KG:  {len(improving)}")
    print(f"    Steps that degraded KG:  {len(degrading)}")
    print(f"    Neutral steps:           {len(neutral)}")


def print_human_feedback_summary(records: list[dict]):
    _section("Human Feedback Summary")
    if not records:
        print("    No human feedback recorded yet.")
        return
    _row("Human-supervised decisions", len(records))

    etype_counts: dict[str, int] = {}
    for r in records:
        et = r.get("error_type", "unknown")
        etype_counts[et] = etype_counts.get(et, 0) + 1

    print("\n    Human interventions by error type:")
    for et, cnt in sorted(etype_counts.items(), key=lambda x: -x[1]):
        print(f"      {et:<40} x{cnt}")

    avg_reward = (sum(r.get("reward", 0) for r in records) / len(records))
    _row("Avg reward from human actions", f"{avg_reward:.2f}")


def print_quantification(before_report: dict, after_report: dict | None,
                         before_counts: dict, after_counts: dict | None):
    _section("Quantified Improvement")

    if after_report and after_counts:
        # Consistency
        b_consistent = before_report.get("is_consistent", False)
        a_consistent = after_report.get("is_consistent", False)
        if not b_consistent and a_consistent:
            print("    [OK] KG achieved FULL CONSISTENCY after repair")
        elif not b_consistent and not a_consistent:
            print("    [!] KG still inconsistent after repair")
        else:
            print("    [OK] KG was already consistent")

        # Violation reduction
        b_total = before_report.get("all_issues", {}).get("total_violations", 0)
        a_total = after_report.get("all_issues", {}).get("total_violations", 0)
        if b_total > 0:
            pct = (b_total - a_total) / b_total * 100
            print(f"    Violations reduced: {b_total} -> {a_total}  "
                  f"({pct:.1f}% reduction)")

        # Triple count change
        b_triples = before_counts.get("triples", 0)
        a_triples = after_counts.get("triples", 0)
        delta_t = a_triples - b_triples
        print(f"    Triple count change: {b_triples} -> {a_triples}  "
              f"({'removed' if delta_t < 0 else 'added'} {abs(delta_t)})")

        # Unsat classes
        b_unsat = len(before_report.get("unsat_classes", []))
        a_unsat = len(after_report.get("unsat_classes", []))
        if b_unsat != a_unsat:
            print(f"    Unsatisfiable classes: {b_unsat} -> {a_unsat}")
    else:
        print("    (provide --after OWL for quantified comparison)")


# ---------------------------------------------------------------------------
# Matplotlib plots
# ---------------------------------------------------------------------------

def _try_import_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")   # non-interactive backend (no display needed)
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        print("  [NOTE] matplotlib not installed — skipping plots.")
        print("         pip install matplotlib")
        return None


def plot_violation_comparison(before_report: dict, after_report: dict,
                               plots_dir: Path):
    plt = _try_import_matplotlib()
    if plt is None:
        return

    b_by_type = before_report.get("all_issues", {}).get("by_type", {})
    a_by_type = after_report.get("all_issues", {}).get("by_type", {})
    all_types = sorted(set(b_by_type) | set(a_by_type))

    if not all_types:
        return

    b_vals = [b_by_type.get(t, 0) for t in all_types]
    a_vals = [a_by_type.get(t, 0) for t in all_types]

    x = range(len(all_types))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar([i - width/2 for i in x], b_vals, width,
                   label="Before Repair", color="#e74c3c", alpha=0.8)
    bars2 = ax.bar([i + width/2 for i in x], a_vals, width,
                   label="After Repair",  color="#2ecc71", alpha=0.8)

    ax.set_xlabel("Violation Type")
    ax.set_ylabel("Count")
    ax.set_title("KG Inconsistency: Violation Counts Before vs After Repair")
    ax.set_xticks(list(x))
    ax.set_xticklabels([t.replace("_", "\n") for t in all_types],
                       rotation=0, ha="center", fontsize=8)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Annotate bars with numbers
    for bar in bars1:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.05,
                    str(int(h)), ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.05,
                    str(int(h)), ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    out = plots_dir / "violation_comparison.png"
    fig.savefig(str(out), dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_repair_steps(steps: list[dict], plots_dir: Path):
    plt = _try_import_matplotlib()
    if plt is None or not steps:
        return

    rewards   = [s.get("reward", 0) for s in steps]
    unsat_d   = [s.get("metrics", {}).get("unsat", 0) for s in steps]
    disj_d    = [s.get("metrics", {}).get("disj", 0) for s in steps]
    triple_b  = [s.get("diff", {}).get("triple_count_before", 0) for s in steps]
    triple_a  = [s.get("diff", {}).get("triple_count_after",  0) for s in steps]
    step_nums = list(range(len(steps)))

    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    fig.suptitle("RL Repair Progress — Step-by-Step", fontsize=13)

    # Panel 1: reward per step
    axes[0].bar(step_nums, rewards,
                color=["#2ecc71" if r >= 0 else "#e74c3c" for r in rewards],
                alpha=0.8)
    axes[0].axhline(0, color="black", linewidth=0.5)
    axes[0].set_ylabel("Reward")
    axes[0].set_title("Reward per Repair Step")
    axes[0].grid(axis="y", alpha=0.3)

    # Panel 2: violation deltas
    axes[1].plot(step_nums, unsat_d, marker="o", label="Unsat delta",
                 color="#3498db")
    axes[1].plot(step_nums, disj_d,  marker="s", label="Disjoint delta",
                 color="#9b59b6")
    axes[1].axhline(0, color="black", linewidth=0.5, linestyle="--")
    axes[1].set_ylabel("Violation Change")
    axes[1].set_title("Violation Count Change per Step (positive = improved)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # Panel 3: triple count
    if any(t > 0 for t in triple_b):
        axes[2].plot(step_nums, triple_b, label="Before step",
                     color="#e67e22", linestyle="--")
        axes[2].plot(step_nums, triple_a, label="After step",
                     color="#1abc9c")
        axes[2].set_ylabel("Triple Count")
        axes[2].set_title("KG Triple Count Over Repair Steps")
        axes[2].legend()
        axes[2].grid(alpha=0.3)
    else:
        axes[2].text(0.5, 0.5, "Triple count data not available",
                     ha="center", va="center", transform=axes[2].transAxes)

    axes[2].set_xlabel("Repair Step")
    fig.tight_layout()
    out = plots_dir / "repair_steps.png"
    fig.savefig(str(out), dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_action_distribution(steps: list[dict], plots_dir: Path):
    plt = _try_import_matplotlib()
    if plt is None or not steps:
        return

    counts: dict[str, int] = {}
    for s in steps:
        atype = s.get("action", {}).get("action_type", "unknown")
        counts[atype] = counts.get(atype, 0) + 1

    if not counts:
        return

    labels = list(counts.keys())
    sizes  = list(counts.values())
    colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12",
              "#9b59b6", "#1abc9c", "#e67e22", "#34495e"]

    fig, ax = plt.subplots(figsize=(8, 6))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=None, autopct="%1.0f%%",
        colors=colors[:len(sizes)], startangle=140
    )
    ax.legend(wedges, [f"{l} ({s})" for l, s in zip(labels, sizes)],
              loc="center left", bbox_to_anchor=(1, 0.5), fontsize=9)
    ax.set_title("Repair Action Types Used by RL Agent")
    fig.tight_layout()
    out = plots_dir / "action_distribution.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_kg_stats_comparison(before_counts: dict, after_counts: dict,
                              plots_dir: Path):
    plt = _try_import_matplotlib()
    if plt is None:
        return

    keys   = ["triples", "subjects", "predicates", "objects",
              "individuals", "classes"]
    labels = ["Triples", "Subjects", "Predicates", "Objects",
              "Individuals", "Classes"]
    b_vals = [before_counts.get(k, 0) for k in keys]
    a_vals = [after_counts.get(k, 0)  for k in keys]

    x = range(len(keys))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([i - width/2 for i in x], b_vals, width,
           label="Before Repair", color="#3498db", alpha=0.8)
    ax.bar([i + width/2 for i in x], a_vals, width,
           label="After Repair",  color="#2ecc71", alpha=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Count")
    ax.set_title("KG Statistics: Before vs After Repair")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = plots_dir / "kg_stats_comparison.png"
    fig.savefig(str(out), dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Helpers for extraction / alignment quality
# ---------------------------------------------------------------------------

def _load_json_safe(path: str) -> list:
    import json as _j
    p = Path(path)
    if not p.exists():
        return []
    try:
        with open(p) as fh:
            data = _j.load(fh)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _load_alignment_report(csv_path: str) -> list[dict]:
    """Load alignment_report.csv into a list of row dicts."""
    p = Path(csv_path)
    if not p.exists():
        return []
    import csv as _csv
    rows = []
    try:
        with open(p, newline="", encoding="utf-8") as fh:
            reader = _csv.DictReader(fh)
            for row in reader:
                rows.append(row)
    except Exception:
        pass
    return rows


def _compute_extraction_metrics(
    raw_path:     str,
    cleaned_path: str,
    align_csv:    str,
) -> dict:
    """
    Compute Precision / Recall / F1 at each pipeline stage.

    Definitions (pipeline-yield framing, suitable for thesis):
      Precision  at stage X = triples passing quality check at X / triples entering X
      Recall     at stage X = cumulative triples reaching X / N_raw
      F1         = harmonic mean of Precision and Recall

    Additionally for alignment: use average match scores and status breakdown.
    """
    raw     = _load_json_safe(raw_path)
    cleaned = _load_json_safe(cleaned_path)
    align   = _load_alignment_report(align_csv)

    n_raw     = max(len(raw),     1)
    n_cleaned = max(len(cleaned), 1)
    n_align   = max(len(align),   1)

    # ── Stage 1: Raw LLM output ──────────────────────────────────────────────
    # Baseline — defines 100 % (all extracted triples are the starting point)
    p_raw, r_raw = 1.0, 1.0

    # ── Stage 2: Cleaned ────────────────────────────────────────────────────
    # Precision  = kept / total entering (what fraction survived cleaning)
    # Recall     = kept / raw            (fraction of original still present)
    p_clean = len(cleaned) / n_raw
    r_clean = len(cleaned) / n_raw

    # ── Stage 3: Aligned ────────────────────────────────────────────────────
    # ok / ok:*minted count = successfully aligned
    n_ok = sum(1 for r in align
               if r.get("status", "").startswith("ok"))
    n_warn = sum(1 for r in align if r.get("status", "") == "warning")
    n_err  = sum(1 for r in align if r.get("status", "") == "error")

    # Avg match scores (s / p / o)
    def _score(key: str) -> float:
        vals = []
        for row in align:
            try:
                vals.append(float(row.get(key, 0) or 0))
            except ValueError:
                pass
        return sum(vals) / len(vals) if vals else 0.0

    avg_s = _score("s_score")
    avg_p = _score("p_score")
    avg_o = _score("o_score")

    # Precision  = ok-aligned / all aligned triples
    # Recall     = ok-aligned / raw triples (end-to-end survival)
    p_align = n_ok / n_align
    r_align = n_ok / n_raw

    def _f1(p: float, r: float) -> float:
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    return {
        "raw":     {"n": len(raw),     "precision": p_raw,   "recall": r_raw,   "f1": _f1(p_raw, r_raw)},
        "cleaned": {"n": len(cleaned), "precision": p_clean, "recall": r_clean, "f1": _f1(p_clean, r_clean)},
        "aligned": {"n": n_ok,         "precision": p_align, "recall": r_align, "f1": _f1(p_align, r_align),
                    "ok": n_ok, "warning": n_warn, "error": n_err,
                    "avg_s": avg_s, "avg_p": avg_p, "avg_o": avg_o},
    }


# ---------------------------------------------------------------------------
# New plots: training history, accuracy/loss, epsilon, precision/recall/F1
# ---------------------------------------------------------------------------

def _load_training_history(history_file: str) -> dict:
    """Load training_history.json written by train_repair.py."""
    p = Path(history_file)
    if not p.exists():
        return {}
    import json as _json
    with open(p) as fh:
        return _json.load(fh)


def _rolling_avg(values: list, window: int = 5) -> list:
    out = []
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        out.append(sum(values[lo:i+1]) / (i - lo + 1))
    return out


def plot_accuracy_over_episodes(history: dict, plots_dir: Path):
    """Accuracy (success rate) and reward over training episodes."""
    plt = _try_import_matplotlib()
    if plt is None or not history:
        return
    eps_data = history.get("per_episode", [])
    if not eps_data:
        return

    episodes  = [d["episode"]  for d in eps_data]
    rewards   = [d["reward"]   for d in eps_data]
    acc       = [d["success_rate_so_far"] * 100 for d in eps_data]
    smooth_r  = _rolling_avg(rewards, window=5)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    fig.suptitle("RL Training Progress — Accuracy & Reward over Episodes", fontsize=13)

    # Panel 1: Accuracy
    ax1.plot(episodes, acc, color="#2ecc71", linewidth=2, label="Cumulative accuracy %")
    ax1.axhline(50, color="gray", linestyle="--", linewidth=0.8, label="50% baseline")
    ax1.fill_between(episodes, acc, alpha=0.15, color="#2ecc71")
    ax1.set_ylabel("Success Rate (%)")
    ax1.set_ylim(0, 105)
    ax1.legend(loc="lower right")
    ax1.grid(alpha=0.3)

    # Panel 2: Reward
    ax2.bar(episodes, rewards,
            color=["#2ecc71" if r >= 0 else "#e74c3c" for r in rewards],
            alpha=0.5, label="Episode reward")
    ax2.plot(episodes, smooth_r, color="#3498db", linewidth=2,
             label=f"Rolling avg (w=5)")
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Total Reward")
    ax2.legend(loc="upper left")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    out = plots_dir / "training_accuracy_reward.png"
    fig.savefig(str(out), dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_loss_and_epsilon(history: dict, plots_dir: Path):
    """DQN training loss and epsilon decay over episodes."""
    plt = _try_import_matplotlib()
    if plt is None or not history:
        return
    eps_data = history.get("per_episode", [])
    if not eps_data:
        return

    episodes = [d["episode"] for d in eps_data]
    losses   = [d["avg_loss"] if d["avg_loss"] is not None else float("nan")
                for d in eps_data]
    epsilons = [d["epsilon"]  for d in eps_data]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    fig.suptitle("DQN Training — Loss & Epsilon Decay", fontsize=13)

    # Panel 1: Loss
    valid_ep  = [e for e, l in zip(episodes, losses) if l == l]  # skip NaN
    valid_los = [l for l in losses if l == l]
    if valid_ep:
        ax1.plot(valid_ep, valid_los, color="#e74c3c", linewidth=1.5)
        ax1.fill_between(valid_ep, valid_los, alpha=0.2, color="#e74c3c")
    ax1.set_ylabel("Avg DQN Loss (MSE)")
    ax1.set_title("Training Loss — lower is better (agent is learning)")
    ax1.grid(alpha=0.3)

    # Panel 2: Epsilon
    ax2.plot(episodes, epsilons, color="#9b59b6", linewidth=2)
    ax2.fill_between(episodes, epsilons, alpha=0.15, color="#9b59b6")
    ax2.axhline(0.05, color="gray", linestyle="--", linewidth=0.8,
                label="epsilon_min = 0.05 (full exploit)")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Epsilon (exploration rate)")
    ax2.set_title("Epsilon Decay — 1.0 = random, 0.05 = RL exploits learned policy")
    ax2.set_ylim(0, 1.05)
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    out = plots_dir / "training_loss_epsilon.png"
    fig.savefig(str(out), dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_precision_recall_f1(before_report: dict, after_report: dict,
                              steps: list, plots_dir: Path):
    """
    Compute and plot Precision, Recall, F1 for the repair process.

    Definitions (violation-level):
      TP  = violation reduction steps (disjoint/unsat delta > 0)
      FP  = violation increase steps  (delta < 0)
      FN  = violations still present after repair
            = violations_before - violations_fixed
    """
    plt = _try_import_matplotlib()
    if plt is None:
        return

    # Count TP and FP from trace
    tp = sum(1 for s in steps
             if s.get("metrics", {}).get("disj", 0) > 0
             or s.get("metrics", {}).get("unsat", 0) > 0
             or s.get("metrics", {}).get("trans", 0) > 0)
    fp = sum(1 for s in steps
             if s.get("metrics", {}).get("disj", 0) < 0
             or s.get("metrics", {}).get("unsat", 0) < 0
             or s.get("metrics", {}).get("trans", 0) < 0)

    b_disj  = len(before_report.get("disjoint_violations", []))
    b_unsat = len(before_report.get("unsat_classes", []))
    b_total = b_disj + b_unsat

    a_disj  = len(after_report.get("disjoint_violations", []))
    a_unsat = len(after_report.get("unsat_classes", []))
    fixed   = max(0, b_total - (a_disj + a_unsat))
    fn      = b_total - fixed  # violations not fixed

    precision = tp / (tp + fp)   if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn)   if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    # Accuracy = episodes that ended consistent (from history if steps give us that)
    before_consistent = before_report.get("is_consistent", False)
    after_consistent  = after_report.get("is_consistent",  False)
    repair_accuracy   = 1.0 if (not before_consistent and after_consistent) else \
                        (0.0 if not after_consistent else 1.0)

    print(f"\n  Precision : {precision:.3f}  ({tp} beneficial steps / {tp+fp} total impactful steps)")
    print(f"  Recall    : {recall:.3f}  ({tp} fixed / {tp+fn} fixable violations)")
    print(f"  F1 Score  : {f1:.3f}")
    print(f"  Repair Accuracy : {'100%' if repair_accuracy == 1.0 else '0% (KG still inconsistent)'}")

    metrics_labels = ["Precision", "Recall", "F1 Score", "Repair Acc."]
    metrics_values = [precision, recall, f1, repair_accuracy]
    colors = ["#3498db", "#2ecc71", "#e67e22", "#9b59b6"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("KG Repair — Precision, Recall, F1 Score", fontsize=13)

    # Bar chart
    bars = axes[0].bar(metrics_labels, metrics_values, color=colors, alpha=0.85, width=0.5)
    axes[0].set_ylim(0, 1.15)
    axes[0].set_ylabel("Score")
    axes[0].set_title("Repair Quality Metrics")
    axes[0].grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, metrics_values):
        axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                     f"{val:.3f}", ha="center", va="bottom", fontweight="bold")

    # TP/FP/FN breakdown
    breakdown_labels = [f"TP={tp}\n(fixed steps)", f"FP={fp}\n(broke steps)",
                        f"FN={fn}\n(unfixed)"]
    breakdown_values = [tp, fp, fn]
    b_colors = ["#2ecc71", "#e74c3c", "#f39c12"]
    axes[1].bar(breakdown_labels, breakdown_values, color=b_colors, alpha=0.85, width=0.5)
    axes[1].set_ylabel("Count")
    axes[1].set_title("Step Classification (TP / FP / FN)")
    axes[1].grid(axis="y", alpha=0.3)
    for i, (lab, val) in enumerate(zip(breakdown_labels, breakdown_values)):
        axes[1].text(i, val + 0.1, str(val), ha="center", va="bottom", fontweight="bold")

    fig.tight_layout()
    out = plots_dir / "precision_recall_f1.png"
    fig.savefig(str(out), dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_before_after_accuracy(before_report: dict, after_report: dict,
                                history: dict, plots_dir: Path):
    """Summary dashboard: before vs after + overall accuracy gauge."""
    plt = _try_import_matplotlib()
    if plt is None:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("KG Repair — Before vs After Summary", fontsize=13)

    # Panel 1: Violation counts before/after
    cats   = ["Disjoint\nViolations", "Unsat\nClasses"]
    before = [len(before_report.get("disjoint_violations", [])),
              len(before_report.get("unsat_classes", []))]
    after  = [len(after_report.get("disjoint_violations", [])),
              len(after_report.get("unsat_classes", []))]
    x = range(len(cats))
    axes[0].bar([i - 0.2 for i in x], before, 0.38, label="Before", color="#e74c3c", alpha=0.8)
    axes[0].bar([i + 0.2 for i in x], after,  0.38, label="After",  color="#2ecc71", alpha=0.8)
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(cats)
    axes[0].set_ylabel("Count")
    axes[0].set_title("Violations Before vs After Repair")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)
    for i, (b, a) in enumerate(zip(before, after)):
        pct = f"↓{(b-a)/b*100:.0f}%" if b > 0 and a < b else ("✓" if b == 0 else "↑")
        axes[0].text(i, max(b, a) + 0.2, pct, ha="center", fontsize=10, fontweight="bold")

    # Panel 2: Training accuracy over episodes (from history)
    eps_data = history.get("per_episode", [])
    if eps_data:
        ep_nums = [d["episode"] for d in eps_data]
        acc     = [d["success_rate_so_far"] * 100 for d in eps_data]
        axes[1].plot(ep_nums, acc, color="#3498db", linewidth=2.5)
        axes[1].fill_between(ep_nums, acc, alpha=0.2, color="#3498db")
        axes[1].axhline(50, color="gray", linestyle="--", linewidth=0.8)
        final_acc = acc[-1] if acc else 0
        axes[1].axhline(final_acc, color="#e74c3c", linestyle=":", linewidth=1.2,
                        label=f"Final: {final_acc:.1f}%")
        axes[1].set_xlabel("Episode")
        axes[1].set_ylabel("Accuracy (%)")
        axes[1].set_ylim(0, 105)
        axes[1].set_title("Accuracy over Training Episodes")
        axes[1].legend()
        axes[1].grid(alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, "No training history\navailable",
                     ha="center", va="center", transform=axes[1].transAxes)
        axes[1].set_title("Accuracy over Training Episodes")

    # Panel 3: Consistency gauge (pie/donut)
    b_ok = before_report.get("is_consistent", False)
    a_ok = after_report.get("is_consistent", False)
    status  = ["Before\n(inconsistent)" if not b_ok else "Before\n(consistent)",
               "After\n(consistent)"    if a_ok      else "After\n(inconsistent)"]
    pie_col = ["#e74c3c" if not b_ok else "#2ecc71",
               "#2ecc71" if a_ok      else "#e74c3c"]
    axes[2].pie([1, 1], labels=status, colors=pie_col, startangle=90,
                autopct=None, wedgeprops={"linewidth": 2, "edgecolor": "white"})
    axes[2].set_title("Consistency Status")

    fig.tight_layout()
    out = plots_dir / "before_after_summary.png"
    fig.savefig(str(out), dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


# ── 1. Total Reward vs Episode ───────────────────────────────────────────────

def plot_reward_curve(history: dict, plots_dir: Path):
    """Plot 1: Total reward vs episode — shows RL is actually learning."""
    plt = _try_import_matplotlib()
    if plt is None or not history:
        return
    eps_data = history.get("per_episode", [])
    if not eps_data:
        return

    episodes = [d["episode"] for d in eps_data]
    rewards  = [d["reward"]  for d in eps_data]
    smooth   = _rolling_avg(rewards, window=5)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(episodes, rewards,
           color=["#2ecc71" if r >= 0 else "#e74c3c" for r in rewards],
           alpha=0.45, label="Episode reward")
    ax.plot(episodes, smooth, color="#2c3e50", linewidth=2.5,
            label="Rolling avg (w=5)")
    ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("Plot 1 — Total Reward vs Episode  (↑ = RL is learning)")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out = plots_dir / "p1_reward_vs_episode.png"
    fig.savefig(str(out), dpi=150); plt.close(fig)
    print(f"  Saved: {out}")


# ── 2. Unsatisfiable Classes vs Episode ──────────────────────────────────────

def plot_unsat_over_episodes(history: dict, plots_dir: Path):
    """Plot 2: Unsatisfiable classes per episode — structural improvement."""
    plt = _try_import_matplotlib()
    if plt is None or not history:
        return
    eps_data = history.get("per_episode", [])
    if not eps_data or "end_unsat" not in eps_data[0]:
        return

    episodes   = [d["episode"]   for d in eps_data]
    end_unsat  = [d["end_unsat"] for d in eps_data]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(episodes, end_unsat, color="#e74c3c", linewidth=2,
            marker="o", markersize=4, label="Unsat classes at episode end")
    ax.fill_between(episodes, end_unsat, alpha=0.15, color="#e74c3c")
    ax.axhline(0, color="#2ecc71", linewidth=1.2, linestyle="--",
               label="0 = fully satisfiable")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Unsatisfiable Classes")
    ax.set_title("Plot 2 — Unsatisfiable Classes vs Episode  (↓ = structural improvement)")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out = plots_dir / "p2_unsat_vs_episode.png"
    fig.savefig(str(out), dpi=150); plt.close(fig)
    print(f"  Saved: {out}")


# ── 3. Disjoint Violations vs Episode ────────────────────────────────────────

def plot_disj_over_episodes(history: dict, plots_dir: Path):
    """Plot 3: Disjoint violations per episode — logical error reduction."""
    plt = _try_import_matplotlib()
    if plt is None or not history:
        return
    eps_data = history.get("per_episode", [])
    if not eps_data or "end_disj" not in eps_data[0]:
        return

    episodes  = [d["episode"]  for d in eps_data]
    end_disj  = [d["end_disj"] for d in eps_data]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(episodes, end_disj, color="#9b59b6", linewidth=2,
            marker="s", markersize=4, label="Disjoint violations at episode end")
    ax.fill_between(episodes, end_disj, alpha=0.15, color="#9b59b6")
    ax.axhline(0, color="#2ecc71", linewidth=1.2, linestyle="--",
               label="0 = no violations")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Disjoint Violations")
    ax.set_title("Plot 3 — Disjoint Violations vs Episode  (↓ = fewer logical errors)")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out = plots_dir / "p3_disj_vs_episode.png"
    fig.savefig(str(out), dpi=150); plt.close(fig)
    print(f"  Saved: {out}")


# ── 4. Total Errors Before vs After RL ───────────────────────────────────────

def plot_errors_before_after(before_report: dict, after_report: dict,
                              plots_dir: Path):
    """Plot 4: Total errors before vs after RL — direct comparison."""
    plt = _try_import_matplotlib()
    if plt is None:
        return

    cats = [
        ("Disjoint\nViolations",  "disjoint_violations"),
        ("Unsat\nClasses",        "unsat_classes"),
        ("Transitive\nDisjoint",  "transitive_disjoint_violations"),
    ]
    labels  = [c[0] for c in cats]
    befores = [len(before_report.get(c[1], [])) for c in cats]
    afters  = [len(after_report.get(c[1],  [])) for c in cats]

    x = range(len(cats))
    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar([i - 0.22 for i in x], befores, 0.42,
                label="Before RL Repair", color="#e74c3c", alpha=0.85)
    b2 = ax.bar([i + 0.22 for i in x], afters,  0.42,
                label="After RL Repair",  color="#2ecc71", alpha=0.85)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Count")
    ax.set_title("Plot 4 — Total Errors Before vs After RL Repair")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    for bar, val in zip(b1, befores):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.05,
                    str(val), ha="center", va="bottom", fontsize=10, fontweight="bold")
    for bar, bval, aval in zip(b2, befores, afters):
        label = str(aval)
        if bval > 0 and aval < bval:
            pct = (bval - aval) / bval * 100
            label = f"{aval}\n(↓{pct:.0f}%)"
        elif aval == 0:
            label = "0 ✓"
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                label, ha="center", va="bottom", fontsize=9, fontweight="bold",
                color="#27ae60")

    fig.tight_layout()
    out = plots_dir / "p4_errors_before_after.png"
    fig.savefig(str(out), dpi=150); plt.close(fig)
    print(f"  Saved: {out}")


# ── 5. Extraction & Alignment P/R/F1 ─────────────────────────────────────────

def plot_extraction_quality(extr_metrics: dict, align_rows: list,
                             plots_dir: Path):
    """
    Plot 5: Precision / Recall / F1 across Raw LLM-> Cleaned-> Aligned.
    Also shows alignment status breakdown and average match scores.
    """
    plt = _try_import_matplotlib()
    if plt is None or not extr_metrics:
        return

    stages  = ["Raw LLM", "Cleaned", "Aligned"]
    prec    = [extr_metrics[k]["precision"] for k in ("raw", "cleaned", "aligned")]
    rec     = [extr_metrics[k]["recall"]    for k in ("raw", "cleaned", "aligned")]
    f1s     = [extr_metrics[k]["f1"]        for k in ("raw", "cleaned", "aligned")]
    counts  = [extr_metrics[k]["n"]         for k in ("raw", "cleaned", "aligned")]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Plot 5 — Extraction & Alignment Quality: Precision / Recall / F1",
                 fontsize=13)

    # Panel A: P/R/F1 grouped bar
    x = range(len(stages))
    w = 0.25
    axes[0].bar([i - w for i in x], prec, w, label="Precision", color="#3498db", alpha=0.85)
    axes[0].bar([i      for i in x], rec,  w, label="Recall",    color="#2ecc71", alpha=0.85)
    axes[0].bar([i + w  for i in x], f1s,  w, label="F1 Score",  color="#e67e22", alpha=0.85)
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(stages)
    axes[0].set_ylim(0, 1.2)
    axes[0].set_ylabel("Score")
    axes[0].set_title("P / R / F1 per Stage")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)
    for i, (p, r, f) in enumerate(zip(prec, rec, f1s)):
        axes[0].text(i - w, p + 0.02, f"{p:.2f}", ha="center", fontsize=8)
        axes[0].text(i,     r + 0.02, f"{r:.2f}", ha="center", fontsize=8)
        axes[0].text(i + w, f + 0.02, f"{f:.2f}", ha="center", fontsize=8)

    # Panel B: Triple count funnel
    axes[1].bar(stages, counts, color=["#3498db", "#2ecc71", "#e67e22"], alpha=0.85)
    axes[1].set_ylabel("Triple Count")
    axes[1].set_title("Triple Counts at Each Stage")
    axes[1].grid(axis="y", alpha=0.25)
    for i, c in enumerate(counts):
        axes[1].text(i, c + 0.1, str(c), ha="center", fontweight="bold")

    # Panel C: Alignment status breakdown + match scores
    al = extr_metrics.get("aligned", {})
    ok, warn, err = al.get("ok", 0), al.get("warning", 0), al.get("error", 0)
    total_al = ok + warn + err or 1
    status_labels = [f"OK\n({ok})", f"Warning\n({warn})", f"Error\n({err})"]
    status_vals   = [ok, warn, err]
    status_cols   = ["#2ecc71", "#f39c12", "#e74c3c"]
    non_zero = [(l, v, c) for l, v, c in zip(status_labels, status_vals, status_cols) if v > 0]
    if non_zero:
        axes[2].pie([v for _, v, _ in non_zero],
                    labels=[l for l, _, _ in non_zero],
                    colors=[c for _, _, c in non_zero],
                    autopct="%1.0f%%", startangle=90)
    avg_s = al.get("avg_s", 0)
    avg_p = al.get("avg_p", 0)
    avg_o = al.get("avg_o", 0)
    axes[2].set_title(
        f"Alignment Status\n"
        f"Avg match — S:{avg_s:.0f}% P:{avg_p:.0f}% O:{avg_o:.0f}%"
    )

    fig.tight_layout()
    out = plots_dir / "p5_extraction_alignment_quality.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {out}")


# ── Bonus: KG Structural Quality across pipeline stages ──────────────────────

def plot_pipeline_quality_line(before_report: dict, after_report: dict,
                                history: dict, plots_dir: Path):
    """
    Line plot: consistency, disjoint violations, unsat classes
    across 3 checkpoints: Before RL | After Ep 1 | After Ep N (final).
    """
    plt = _try_import_matplotlib()
    if plt is None:
        return

    eps_data = history.get("per_episode", []) if history else []

    # Checkpoint 0: before RL (before_report)
    c0_disj  = len(before_report.get("disjoint_violations", []))
    c0_unsat = len(before_report.get("unsat_classes", []))

    # Checkpoint 1: after episode 1
    c1_disj  = eps_data[0].get("end_disj",  c0_disj)  if eps_data else c0_disj
    c1_unsat = eps_data[0].get("end_unsat", c0_unsat) if eps_data else c0_unsat

    # Checkpoint N: final (after_report)
    cN_disj  = len(after_report.get("disjoint_violations", [])) if after_report else c0_disj
    cN_unsat = len(after_report.get("unsat_classes",       [])) if after_report else c0_unsat

    checkpoints = ["Before RL", "After Ep 1", f"After Ep {len(eps_data)} (Final)"]
    disj_vals   = [c0_disj,  c1_disj,  cN_disj]
    unsat_vals  = [c0_unsat, c1_unsat, cN_unsat]
    consist     = [
        before_report.get("is_consistent", False),
        eps_data[0].get("consistent", False) if eps_data else False,
        after_report.get("is_consistent",  False) if after_report else False,
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("KG Structural Quality Across Pipeline Stages", fontsize=13)

    x = range(len(checkpoints))
    mk = "o"
    axes[0].plot(x, disj_vals,  color="#9b59b6", linewidth=2.5, marker=mk, markersize=8,
                 label="Disjoint violations")
    axes[0].plot(x, unsat_vals, color="#e74c3c", linewidth=2.5, marker="s", markersize=8,
                 label="Unsat classes")
    for xi, (dv, uv) in enumerate(zip(disj_vals, unsat_vals)):
        axes[0].annotate(str(dv), (xi, dv), textcoords="offset points",
                         xytext=(0, 8), ha="center", color="#9b59b6", fontweight="bold")
        axes[0].annotate(str(uv), (xi, uv), textcoords="offset points",
                         xytext=(0, -14), ha="center", color="#e74c3c", fontweight="bold")
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(checkpoints, fontsize=9)
    axes[0].set_ylabel("Count")
    axes[0].set_title("Violations over RL Training")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    # Consistency status bars
    consist_vals = [1 if c else 0 for c in consist]
    cols = ["#2ecc71" if c else "#e74c3c" for c in consist]
    axes[1].bar(checkpoints, consist_vals, color=cols, alpha=0.85, width=0.5)
    axes[1].set_yticks([0, 1])
    axes[1].set_yticklabels(["Inconsistent", "Consistent"])
    axes[1].set_title("Consistency Status")
    for i, c in enumerate(consist):
        axes[1].text(i, 0.5, "✓ Consistent" if c else "✗ Inconsistent",
                     ha="center", va="center", fontsize=10, fontweight="bold",
                     color="white")
    axes[1].grid(axis="y", alpha=0.25)

    fig.tight_layout()
    out = plots_dir / "p6_pipeline_quality_line.png"
    fig.savefig(str(out), dpi=150); plt.close(fig)
    print(f"  Saved: {out}")


# ── Bonus: Repair Efficiency ──────────────────────────────────────────────────

def plot_repair_efficiency(history: dict, steps: list, plots_dir: Path):
    """Steps per episode, error reduction rate, triple delta."""
    plt = _try_import_matplotlib()
    if plt is None or not history:
        return
    eps_data = history.get("per_episode", [])
    if not eps_data:
        return

    episodes  = [d["episode"] for d in eps_data]
    n_steps   = [d["steps"]   for d in eps_data]

    # Error reduction rate: (start_disj + start_unsat) - (end_disj + end_unsat)
    start_err = [d.get("start_disj", 0) + d.get("start_unsat", 0) for d in eps_data]
    end_err   = [d.get("end_disj",   0) + d.get("end_unsat",   0) for d in eps_data]
    err_red   = [s - e for s, e in zip(start_err, end_err)]

    # Triple delta from repair trace (net triples added/removed per step)
    triple_deltas = [
        s.get("diff", {}).get("triple_count_after", 0) -
        s.get("diff", {}).get("triple_count_before", 0)
        for s in steps if s.get("diff")
    ]
    step_nums = list(range(1, len(triple_deltas) + 1))

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle("RL Repair Efficiency Metrics", fontsize=13)

    # Panel 1: Steps per episode
    axes[0].bar(episodes, n_steps, color="#3498db", alpha=0.8)
    axes[0].axhline(sum(n_steps)/len(n_steps), color="#e74c3c",
                    linestyle="--", linewidth=1.5, label=f"Mean={sum(n_steps)/len(n_steps):.1f}")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Steps Taken")
    axes[0].set_title("Steps per Episode\n(fewer = more efficient)")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)

    # Panel 2: Error reduction per episode
    axes[1].bar(episodes, err_red,
                color=["#2ecc71" if e >= 0 else "#e74c3c" for e in err_red], alpha=0.8)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Errors Reduced")
    axes[1].set_title("Error Reduction per Episode\n(>0 = RL fixed something)")
    axes[1].grid(axis="y", alpha=0.25)

    # Panel 3: Triple delta per repair step (noise vs. meaning check)
    if triple_deltas:
        pos = sum(1 for d in triple_deltas if d > 0)
        neg = sum(1 for d in triple_deltas if d < 0)
        zer = sum(1 for d in triple_deltas if d == 0)
        axes[2].bar(step_nums, triple_deltas,
                    color=["#2ecc71" if d > 0 else ("#e74c3c" if d < 0 else "#bdc3c7")
                           for d in triple_deltas], alpha=0.8)
        axes[2].axhline(0, color="black", linewidth=0.8)
        axes[2].set_xlabel("Repair Step")
        axes[2].set_ylabel("Triple Count Change")
        axes[2].set_title(f"Triple Growth per Step\n"
                          f"Added:{pos}  Removed:{neg}  Neutral:{zer}")
        axes[2].grid(axis="y", alpha=0.25)
    else:
        axes[2].text(0.5, 0.5, "No trace data\navailable",
                     ha="center", va="center", transform=axes[2].transAxes)
        axes[2].set_title("Triple Growth per Step")

    fig.tight_layout()
    out = plots_dir / "p7_repair_efficiency.png"
    fig.savefig(str(out), dpi=150); plt.close(fig)
    print(f"  Saved: {out}")


def plot_phase_accuracy(history: dict, plots_dir: Path):
    """Accuracy broken down by training phase (Phase 1/2/3)."""
    plt = _try_import_matplotlib()
    if plt is None or not history:
        return
    eps_data = history.get("per_episode", [])
    if not eps_data:
        return

    phases = {"human_supervised": [], "human_first": [], "rl_automated": []}
    for d in eps_data:
        ph = d.get("phase", "rl_automated")
        phases.setdefault(ph, []).append(d["consistent"])

    labels = {
        "human_supervised": "Phase 1\nHuman Supervised",
        "human_first":      "Phase 2\nHuman First-Step",
        "rl_automated":     "Phase 3\nRL Automated",
    }
    phase_acc   = []
    phase_label = []
    phase_color = ["#f39c12", "#3498db", "#2ecc71"]
    for key in ("human_supervised", "human_first", "rl_automated"):
        vals = phases.get(key, [])
        if vals:
            phase_acc.append(sum(vals) / len(vals) * 100)
            phase_label.append(labels[key])

    if not phase_acc:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(phase_label, phase_acc,
                  color=phase_color[:len(phase_acc)], alpha=0.85, width=0.45)
    ax.set_ylim(0, 115)
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("Repair Accuracy by Training Phase")
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, phase_acc):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1.5,
                f"{val:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=12)
    fig.tight_layout()
    out = plots_dir / "accuracy_by_phase.png"
    fig.savefig(str(out), dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate supervisor-ready KG quality report")
    parser.add_argument(
        "--before",
        default=str(config.DEFAULT_PATHS["repair_kg"]["input"]),
        help="OWL file BEFORE repair (default: merged_kg.owl)",
    )
    parser.add_argument(
        "--after",
        default=None,
        help="OWL file AFTER repair (latest rl_repair_steps/ file if omitted)",
    )
    parser.add_argument(
        "--trace",
        default=str(config.RL_REPAIR_TRACES_DIR / "repair_trace.jsonl"),
        help="Repair trace JSONL",
    )
    parser.add_argument(
        "--feedback",
        default=str(config.RL_MODELS_DIR / "human_feedback.jsonl"),
        help="Human feedback JSONL",
    )
    parser.add_argument(
        "--reasoner",
        default=config.DEFAULT_REASONER,
        choices=["hermit", "konclude"],
    )
    parser.add_argument(
        "--plots",
        default=str(config.OUTPUTS_REPORTS),
        help="Directory to save PNG plots",
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Skip matplotlib plots",
    )
    parser.add_argument(
        "--history",
        default=str(config.RL_MODELS_DIR / "training_history.json"),
        help="Training history JSON from train_repair.py",
    )
    parser.add_argument(
        "--raw-triples",
        default=str(config.DEFAULT_PATHS["generate_triples"]["output"]),
        help="Raw triples JSON output by generate_triples step",
    )
    parser.add_argument(
        "--cleaned-triples",
        default=str(config.DEFAULT_PATHS["clean_triples"]["output"]),
        help="Cleaned triples JSON output by clean_triples step",
    )
    parser.add_argument(
        "--alignment-report",
        default=str(config.DEFAULT_PATHS["align_triples"]["report"]),
        help="Alignment report CSV output by align_triples step",
    )
    args = parser.parse_args()

    plots_dir = Path(args.plots)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # ── Auto-detect "after" OWL ──────────────────────────────────────────
    after_owl = args.after
    if after_owl is None:
        steps_dir = Path(config.RL_REPAIR_STEPS_DIR)
        candidates = sorted(steps_dir.glob("repaired_step_*.owl"),
                            key=lambda p: p.stat().st_mtime, reverse=True)
        if candidates:
            after_owl = str(candidates[0])
            print(f"[report] Auto-detected latest repaired OWL: {after_owl}")
        else:
            print("[report] No repaired OWL found in rl_repair_steps/; "
                  "will report before-only.")

    # ── Print header ─────────────────────────────────────────────────────
    _banner("KG REPAIR FRAMEWORK — QUALITY REPORT")
    print(f"  Before OWL : {args.before}")
    print(f"  After OWL  : {after_owl or '(not available)'}")
    print(f"  Reasoner   : {args.reasoner}")
    print(f"  Trace      : {args.trace}")

    # ── Run reasoner on before ────────────────────────────────────────────
    _banner("STEP 1: Analysing BEFORE state")
    if not Path(args.before).exists():
        print(f"  [ERROR] Before OWL not found: {args.before}")
        sys.exit(1)

    before_counts = _entity_counts(args.before)
    before_report = _run_reasoner(args.before, args.reasoner)

    # ── Run reasoner on after ─────────────────────────────────────────────
    after_counts  = None
    after_report  = None
    if after_owl and Path(after_owl).exists():
        _banner("STEP 2: Analysing AFTER state")
        after_counts = _entity_counts(after_owl)
        after_report = _run_reasoner(after_owl, args.reasoner)
    else:
        _banner("STEP 2: Skipped (no after OWL)")

    # ── Load trace, feedback & training history ───────────────────────────
    steps   = _load_trace(args.trace)
    records = _load_feedback(args.feedback)
    train_history = _load_training_history(args.history)
    if train_history:
        print(f"[report] Training history loaded: {len(train_history.get('per_episode',[]))} episodes")
    else:
        print("[report] No training history found — skipping episode plots."
              f"\n         Expected: {args.history}")

    # ── Extraction / alignment quality metrics ────────────────────────────
    extr_metrics = _compute_extraction_metrics(
        args.raw_triples,
        args.cleaned_triples,
        args.alignment_report,
    )
    align_rows = _load_alignment_report(args.alignment_report)
    if extr_metrics.get("raw", {}).get("n", 0) > 0:
        print(f"[report] Extraction metrics: "
              f"raw={extr_metrics['raw']['n']}  "
              f"cleaned={extr_metrics['cleaned']['n']}  "
              f"aligned={extr_metrics['aligned']['n']}")

    # ── Text report ───────────────────────────────────────────────────────
    _banner("KG STATISTICS")
    print_kg_statistics(before_counts, after_counts)

    _banner("INCONSISTENCY PROFILE")
    print_consistency_profile(before_report, after_report)

    _banner("REPAIR TRACE")
    print_repair_trace_summary(steps)

    _banner("HUMAN SUPERVISION")
    print_human_feedback_summary(records)

    _banner("QUANTIFIED IMPROVEMENT")
    print_quantification(before_report, after_report,
                         before_counts, after_counts or {})

    # ── Plots ─────────────────────────────────────────────────────────────
    if not args.no_plots:
        _banner("GENERATING PLOTS")
        print(f"  Output directory: {plots_dir}")

        if after_report:
            plot_violation_comparison(before_report, after_report, plots_dir)
            if before_counts and after_counts:
                plot_kg_stats_comparison(before_counts, after_counts, plots_dir)

        if steps:
            plot_repair_steps(steps, plots_dir)
            plot_action_distribution(steps, plots_dir)

        # ── Episode-level training plots ──────────────────────────────────
        if train_history:
            plot_accuracy_over_episodes(train_history, plots_dir)
            plot_loss_and_epsilon(train_history, plots_dir)
            plot_phase_accuracy(train_history, plots_dir)

        # ── Precision / Recall / F1 (repair-level) ────────────────────────
        if after_report and steps:
            _banner("PRECISION / RECALL / F1")
            plot_precision_recall_f1(before_report, after_report, steps, plots_dir)

        # ── Before/After summary dashboard ────────────────────────────────
        if after_report:
            plot_before_after_accuracy(before_report, after_report,
                                       train_history, plots_dir)

        # ── P1: Total Reward vs Episode ────────────────────────────────────
        if train_history:
            plot_reward_curve(train_history, plots_dir)

        # ── P2: Unsatisfiable Classes vs Episode ──────────────────────────
        if train_history:
            plot_unsat_over_episodes(train_history, plots_dir)

        # ── P3: Disjoint Violations vs Episode ────────────────────────────
        if train_history:
            plot_disj_over_episodes(train_history, plots_dir)

        # ── P4: Total Errors Before vs After RL ───────────────────────────
        if after_report:
            plot_errors_before_after(before_report, after_report, plots_dir)

        # ── P5: Extraction & Alignment Quality ────────────────────────────
        if extr_metrics.get("raw", {}).get("n", 0) > 0:
            plot_extraction_quality(extr_metrics, align_rows, plots_dir)

        # ── P6: Pipeline Quality Line ──────────────────────────────────────
        if after_report:
            plot_pipeline_quality_line(
                before_report, after_report, train_history, plots_dir)

        # ── P7: Repair Efficiency ──────────────────────────────────────────
        if train_history:
            plot_repair_efficiency(train_history, steps, plots_dir)

        print()
        print("  Plots saved to:", plots_dir)
        print("  Files generated:")
        for p in sorted(plots_dir.glob("*.png")):
            print(f"    {p.name}")

    _banner("REPORT COMPLETE")
    print()


if __name__ == "__main__":
    main()
