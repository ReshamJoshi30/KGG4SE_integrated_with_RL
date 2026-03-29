"""
generate_thesis_diagrams.py
---------------------------
Generates publication-quality thesis architecture diagrams as high-resolution PNGs.

Diagrams produced (12 total):
  fig01_pipeline_architecture.png   – End-to-end 9-step pipeline
  fig02_rl_mdp_loop.png             – RL MDP / training loop
  fig03_dueling_dqn.png             – Dueling Double-DQN network architecture
  fig04_curriculum_training.png     – Three-phase curriculum training strategy
  fig05_reward_function.png         – Reward function design & behaviour
  fig06_core_contributions.png      – Core research contributions summary
  fig07_data_flow.png               – Detailed data flow with file artifacts
  fig08_ontology_alignment.png      – Three-tier ontology alignment process
  fig09_repair_candidates.png       – Violation types-> repair actions mapping
  fig10_state_action_space.png      – RL state & action space visualization
  fig11_layered_architecture.png    – Layered system architecture view
  fig12_triple_extraction.png       – LLM-based triple extraction pipeline

Usage:
    python generate_thesis_diagrams.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np
import os

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "diagrams")
os.makedirs(OUT_DIR, exist_ok=True)
DPI = 200

# ─── Professional colour palette ──────────────────────────────────────────
C_BLUE     = "#2563EB"
C_LBLUE    = "#DBEAFE"
C_DBLUE    = "#1E40AF"
C_GREEN    = "#16A34A"
C_LGREEN   = "#DCFCE7"
C_DGREEN   = "#15803D"
C_ORANGE   = "#EA580C"
C_LORANGE  = "#FED7AA"
C_PURPLE   = "#7C3AED"
C_LPURPLE  = "#EDE9FE"
C_RED      = "#DC2626"
C_LRED     = "#FEE2E2"
C_GRAY     = "#1F2937"
C_MGRAY    = "#6B7280"
C_LGRAY    = "#F3F4F6"
C_TEAL     = "#0D9488"
C_LTEAL    = "#CCFBF1"
C_YELLOW   = "#CA8A04"
C_LYELLOW  = "#FEF9C3"
C_PINK     = "#DB2777"
C_LPINK    = "#FCE7F3"
C_INDIGO   = "#4F46E5"
C_LINDIGO  = "#E0E7FF"
C_WHITE    = "#FFFFFF"
C_BG       = "#FAFBFC"
C_BORDER   = "#D1D5DB"


def savefig(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight",
                facecolor=fig.get_facecolor(), pad_inches=0.3)
    plt.close(fig)
    print(f"  ✓ {name}")


def _rounded_box(ax, x, y, w, h, label, bg, fg, sub="", fontsize=9,
                 sub_fontsize=7, lw=1.8, zorder=3, pad=0.1, alpha=1.0):
    """Draw a rounded rectangle with label and optional subtitle."""
    bx = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                         boxstyle=f"round,pad={pad}",
                         linewidth=lw, edgecolor=fg, facecolor=bg,
                         zorder=zorder, alpha=alpha)
    ax.add_patch(bx)
    dy = 0.12 if sub else 0
    ax.text(x, y + dy, label, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color=C_GRAY,
            multialignment="center", zorder=zorder + 1)
    if sub:
        ax.text(x, y - 0.25, sub, ha="center", va="center",
                fontsize=sub_fontsize, color=fg, zorder=zorder + 1,
                fontstyle="italic", multialignment="center")
    return bx


def _arrow(ax, x1, y1, x2, y2, color=None, lw=1.5, style="-|>",
           ls="-", rad=0.0, zorder=2):
    """Draw an arrow from (x1,y1) to (x2,y2)."""
    if color is None:
        color = C_MGRAY
    conn = f"arc3,rad={rad}" if rad else None
    props = dict(arrowstyle=style, color=color, lw=lw, linestyle=ls)
    if conn:
        props["connectionstyle"] = conn
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=props, zorder=zorder)


def _label(ax, x, y, text, fontsize=8, color=None, ha="center",
           va="center", weight="normal", style="normal", family=None,
           bbox_cfg=None, zorder=5, rotation=0):
    """Place text at (x,y) with optional box."""
    if color is None:
        color = C_GRAY
    kw = dict(ha=ha, va=va, fontsize=fontsize, fontweight=weight,
              color=color, zorder=zorder, fontstyle=style, rotation=rotation)
    if family:
        kw["fontfamily"] = family
    if bbox_cfg:
        kw["bbox"] = bbox_cfg
    ax.text(x, y, text, **kw)


# ═══════════════════════════════════════════════════════════════════════════
# FIG 1 – End-to-End Pipeline Architecture
# ═══════════════════════════════════════════════════════════════════════════

def fig01_pipeline():
    fig, ax = plt.subplots(figsize=(16, 11))
    fig.patch.set_facecolor(C_WHITE)
    ax.set_xlim(-0.5, 16); ax.set_ylim(-0.5, 11)
    ax.axis("off")

    # Title
    _label(ax, 8, 10.5, "KGG4SE Framework — End-to-End Pipeline Architecture",
           fontsize=16, weight="bold", color=C_DBLUE)

    # Column headers
    cols = [
        (3.0, "Phase 1: LLM Extraction", C_BLUE),
        (8.0, "Phase 2: Ontology Integration", C_GREEN),
        (13.0, "Phase 3: QA & RL Repair", C_PURPLE),
    ]
    for cx, lbl, col in cols:
        _rounded_box(ax, cx, 9.6, 4.2, 0.55, lbl, col, col,
                     fontsize=10, lw=0)
        ax.text(cx, 9.6, lbl, ha="center", va="center",
                fontsize=10, fontweight="bold", color=C_WHITE, zorder=10)

    # Pipeline steps
    steps = [
        # Column 1
        ("Step 1\nCorpus Ingestion", 3.0, 8.5, C_LBLUE, C_BLUE,
         "CSV-> plain text"),
        ("Step 2\nLLM Triple Extraction", 3.0, 7.0, C_LBLUE, C_BLUE,
         "GPT-4o-mini / LLaMA-3"),
        ("Step 3\nTriple Cleaning", 3.0, 5.5, C_LBLUE, C_BLUE,
         "Normalize + deduplicate"),
        # Column 2
        ("Step 4\nOntology Alignment", 8.0, 8.5, C_LGREEN, C_GREEN,
         "CSV + fuzzy match (≥85%)"),
        ("Step 5\nKG Construction", 8.0, 7.0, C_LGREEN, C_GREEN,
         "Merge + OWL serialization"),
        ("Step 6\nOWL Reasoning", 8.0, 5.5, C_LORANGE, C_ORANGE,
         "Konclude / HermiT"),
        # Column 3
        ("Step 7\nViolation Extraction", 13.0, 8.5, C_LPURPLE, C_PURPLE,
         "Parse reasoner output"),
        ("Step 8\nRL Repair Agent", 13.0, 7.0, C_LPURPLE, C_PURPLE,
         "Dueling DQN + PER"),
        ("Step 9\nValidated KG", 13.0, 5.5, C_LGREEN, C_DGREEN,
         "Consistent OWL ontology"),
    ]

    W, H = 3.4, 1.0
    for (label, x, y, bg, fg, sub) in steps:
        _rounded_box(ax, x, y, W, H, label, bg, fg, sub=sub,
                     fontsize=9, sub_fontsize=7.5)

    # Vertical arrows within columns
    for col_x in [3.0, 8.0, 13.0]:
        col_steps = [(s[2], s[1]) for s in steps if s[1] == col_x]
        col_steps.sort(key=lambda s: -s[0])
        for i in range(len(col_steps) - 1):
            y1, x1 = col_steps[i]
            y2, x2 = col_steps[i + 1]
            _arrow(ax, x1, y1 - H / 2 - 0.05, x2, y2 + H / 2 + 0.05,
                   color=C_MGRAY)

    # Horizontal arrows between columns
    for y in [8.5, 7.0, 5.5]:
        _arrow(ax, 3.0 + W/2 + 0.1, y, 8.0 - W/2 - 0.1, y,
               color=C_MGRAY, lw=1.3)
        _arrow(ax, 8.0 + W/2 + 0.1, y, 13.0 - W/2 - 0.1, y,
               color=C_MGRAY, lw=1.3)

    # Feedback loop (Step 9-> Step 8)
    _arrow(ax, 13.0 + 0.3, 5.5 + H/2 + 0.05, 13.0 + 0.3, 7.0 - H/2 - 0.05,
           color=C_RED, lw=2.0, ls="--", style="<|-")
    _label(ax, 13.8, 6.25, "Repair\nFeedback\nLoop", fontsize=7.5,
           color=C_RED, style="italic")

    # Data artifacts between columns
    artifacts = [
        (5.5, 8.5, "corpus_triples.json"),
        (5.5, 7.0, "aligned_triples.ttl"),
        (5.5, 5.5, "merged_kg.owl"),
        (10.5, 8.5, "violations\n+ evidence"),
        (10.5, 7.0, "repair\ncandidates"),
        (10.5, 5.5, "reasoned_kg.owl"),
    ]
    for (x, y, lbl) in artifacts:
        _label(ax, x, y, lbl, fontsize=6.8, color=C_MGRAY, style="italic",
               bbox_cfg=dict(boxstyle="round,pad=0.12", facecolor="#F9FAFB",
                             edgecolor=C_BORDER, lw=0.8))

    # Input/Output boxes
    _label(ax, 3.0, 9.15, "INPUT: Domain Text Corpus\n(Automotive Electronics)",
           fontsize=8.5, color=C_BLUE, weight="bold",
           bbox_cfg=dict(boxstyle="round,pad=0.25", facecolor=C_LBLUE,
                         edgecolor=C_BLUE, lw=1.3))
    _label(ax, 13.0, 4.7, "OUTPUT: Consistent OWL Knowledge Graph\n(Aligned to GENIALOnt / BFO 2020)",
           fontsize=8.5, color=C_DGREEN, weight="bold",
           bbox_cfg=dict(boxstyle="round,pad=0.25", facecolor=C_LGREEN,
                         edgecolor=C_GREEN, lw=1.3))

    # Technologies sidebar
    tech_text = (
        "Technologies & Libraries\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "LLM:      GPT-4o-mini / LLaMA-3\n"
        "KG:       RDFLib · owlready2\n"
        "Align:    RapidFuzz (fuzzy)\n"
        "Reason: Konclude · HermiT\n"
        "RL:        PyTorch · DQN · PER\n"
        "Lang:     Python 3.10+"
    )
    _label(ax, 2.5, 2.8, tech_text, fontsize=7.5, color=C_GRAY,
           ha="left", va="top", family="monospace",
           bbox_cfg=dict(boxstyle="round,pad=0.4", facecolor=C_LGRAY,
                         edgecolor=C_BORDER, lw=1.0))

    # Pipeline note
    _label(ax, 8.0, 1.0,
           "All steps inherit from PipelineStep abstract base class  ·  "
           "CLI orchestration via pipeline.py  ·  Configurable via config.py",
           fontsize=8, color=C_MGRAY, style="italic")

    savefig(fig, "fig01_pipeline_architecture.png")


# ═══════════════════════════════════════════════════════════════════════════
# FIG 2 – RL MDP Training Loop
# ═══════════════════════════════════════════════════════════════════════════

def fig02_rl_loop():
    fig, ax = plt.subplots(figsize=(15, 10))
    fig.patch.set_facecolor(C_WHITE)
    ax.set_xlim(0, 15); ax.set_ylim(0, 10)
    ax.axis("off")

    _label(ax, 7.5, 9.5, "Reinforcement Learning MDP — KG Repair Training Loop",
           fontsize=15, weight="bold", color=C_DBLUE)

    # ── ENVIRONMENT BOX ──
    env_w, env_h = 5.0, 1.8
    _rounded_box(ax, 7.5, 8.0, env_w, env_h,
                 "ENVIRONMENT", C_LORANGE, C_ORANGE,
                 sub="RepairEnv: OWL KG + Konclude/HermiT Reasoner",
                 fontsize=13, sub_fontsize=8.5, lw=2.2)

    # Internal env details
    env_details = [
        "• Load merged_kg.owl (fresh copy each episode)",
        "• Run OWL reasoner-> detect violations",
        "• Extract axiom-level evidence",
        "• Build candidate actions per violation",
    ]
    _rounded_box(ax, 2.5, 8.0, 4.0, 1.8, "", C_LGRAY, C_BORDER, lw=1)
    _label(ax, 2.5, 8.65, "Environment Internals", fontsize=8.5,
           weight="bold", color=C_ORANGE)
    for i, item in enumerate(env_details):
        _label(ax, 1.0, 8.3 - i * 0.3, item, fontsize=7, color=C_GRAY,
               ha="left", family="monospace")

    # ── AGENT BOX ──
    agent_w, agent_h = 5.0, 1.8
    _rounded_box(ax, 7.5, 4.2, agent_w, agent_h,
                 "DQN AGENT", C_LPURPLE, C_PURPLE,
                 sub="Dueling Double-DQN with ε-greedy exploration",
                 fontsize=13, sub_fontsize=8.5, lw=2.2)

    # Agent details
    agent_details = [
        "• Policy network: selects actions",
        "• Target network: stable Q-value estimates",
        "• Action masking (invalid-> Q = −∞)",
        "• Gradient clipping (max_norm = 1.0)",
    ]
    _rounded_box(ax, 12.5, 4.2, 4.0, 1.8, "", C_LGRAY, C_BORDER, lw=1)
    _label(ax, 12.5, 4.85, "Agent Internals", fontsize=8.5,
           weight="bold", color=C_PURPLE)
    for i, item in enumerate(agent_details):
        _label(ax, 11.0, 4.5 - i * 0.3, item, fontsize=7, color=C_GRAY,
               ha="left", family="monospace")

    # ── STATE VECTOR ──
    _rounded_box(ax, 2.5, 5.8, 4.0, 2.6, "", C_LINDIGO, C_INDIGO, lw=1.5)
    _label(ax, 2.5, 6.9, "State Vector  s ∈ ℝ¹⁸", fontsize=9.5,
           weight="bold", color=C_INDIGO)
    state_items = [
        "[0–9]   violation type one-hot",
        "[10]      # available repair actions",
        "[11]      # low-risk actions",
        "[12]      has evidence flag",
        "[13]      step progress ratio",
        "[14]      has entity IRI flag",
        "[15]      queue progress ratio",
        "[16]      last step improved",
        "[17]      min action risk score",
    ]
    for i, item in enumerate(state_items):
        _label(ax, 0.8, 6.55 - i * 0.3, item, fontsize=6.8,
               color=C_GRAY, ha="left", family="monospace")

    # ── REPLAY BUFFER ──
    _rounded_box(ax, 7.5, 2.3, 5.5, 1.2,
                 "Prioritized Experience Replay Buffer", C_LTEAL, C_TEAL,
                 sub="α=0.6  ·  β: 0.4→1.0  ·  expert boost ×2",
                 fontsize=9.5, sub_fontsize=7.5, lw=1.8)

    # ── REWARD SIGNAL ──
    _rounded_box(ax, 7.5, 0.7, 6.0, 0.8,
                 "Reward:  +10 (consistent)  |  +2 (fix)  |  "
                 "−0.5 (no-op)  |  −2 (new violation)  |  −10 (regression)",
                 C_LYELLOW, C_YELLOW, fontsize=7.5, lw=1.5)

    # ── ARROWS (MDP Cycle) ──
    # Env-> State (emit s_t)
    _arrow(ax, 5.0, 7.5, 4.5, 6.9, color=C_BLUE, lw=2)
    _label(ax, 4.2, 7.4, "sₜ", fontsize=11, color=C_BLUE, weight="bold",
           style="italic")

    # State-> Agent (observe)
    _arrow(ax, 3.5, 4.8, 5.0, 4.5, color=C_BLUE, lw=2)

    # Agent-> Env (action)
    _arrow(ax, 10.0, 4.9, 10.0, 7.1, color=C_PURPLE, lw=2)
    _label(ax, 10.6, 6.0, "aₜ = apply_fix()", fontsize=9,
           color=C_PURPLE, style="italic")

    # Env-> Reward
    _arrow(ax, 7.5, 7.1, 7.5, 1.1, color=C_YELLOW, lw=1.5, ls="--",
           rad=0.6)
    _label(ax, 8.8, 1.6, "rₜ , sₜ₊₁", fontsize=10,
           color=C_YELLOW, weight="bold", style="italic")

    # Agent ↔ Buffer
    _arrow(ax, 7.0, 3.3, 7.0, 2.9, color=C_TEAL, lw=1.5)
    _arrow(ax, 8.0, 2.9, 8.0, 3.3, color=C_TEAL, lw=1.5, ls=":")
    _label(ax, 6.0, 3.1, "store (s,a,r,s')", fontsize=7.5,
           color=C_TEAL, style="italic")
    _label(ax, 9.2, 3.1, "sample batch", fontsize=7.5,
           color=C_TEAL, style="italic")

    # Reward-> Agent
    _arrow(ax, 5.5, 0.7, 5.5, 3.8, color=C_YELLOW, lw=1.3, ls="--", rad=-0.2)

    # ── NOTES ──
    _label(ax, 1.5, 2.3,
           "Soft Target Update\n"
           r"$\theta^{-} \leftarrow \tau\theta + (1-\tau)\theta^{-}$"
           "\nτ = 0.005 (every step)",
           fontsize=8.5, color=C_PURPLE,
           bbox_cfg=dict(boxstyle="round,pad=0.3", facecolor=C_LPURPLE,
                         edgecolor=C_PURPLE, lw=1))

    _label(ax, 1.5, 0.8,
           "ε-Greedy Exploration\n"
           r"$\varepsilon_t = \varepsilon_{t-1} \times 0.994$"
           "\nstart=1.0-> min=0.05",
           fontsize=8.5, color=C_ORANGE,
           bbox_cfg=dict(boxstyle="round,pad=0.3", facecolor=C_LORANGE,
                         edgecolor=C_ORANGE, lw=1))

    savefig(fig, "fig02_rl_mdp_loop.png")


# ═══════════════════════════════════════════════════════════════════════════
# FIG 3 – Dueling Double-DQN Architecture
# ═══════════════════════════════════════════════════════════════════════════

def fig03_dqn():
    fig, ax = plt.subplots(figsize=(16, 8))
    fig.patch.set_facecolor(C_WHITE)
    ax.set_xlim(-0.5, 16); ax.set_ylim(-0.5, 8)
    ax.axis("off")

    _label(ax, 8, 7.5, "Dueling Double-DQN Network Architecture",
           fontsize=15, weight="bold", color=C_DBLUE)

    # ── Input Layer ──
    _rounded_box(ax, 1.0, 4.0, 1.8, 1.2, "Input\nState", C_LBLUE, C_BLUE,
                 sub="s ∈ ℝ¹⁸", fontsize=9, lw=2)

    # ── Shared Feature Extractor ──
    # Background box
    bx = FancyBboxPatch((2.2, 2.6), 5.6, 2.8,
                         boxstyle="round,pad=0.15",
                         linewidth=1.5, edgecolor=C_BORDER,
                         facecolor=C_LGRAY, zorder=1, alpha=0.5)
    ax.add_patch(bx)
    _label(ax, 5.0, 5.2, "Shared Feature Extractor", fontsize=9,
           weight="bold", color=C_GRAY)

    layers_shared = [
        (3.2, 4.0, "Linear\n18→144", C_LGRAY, C_GRAY),
        (4.6, 4.0, "LayerNorm\n+ ReLU", C_LGRAY, C_GRAY),
        (5.8, 4.0, "Dropout\np=0.1", C_LGRAY, C_GRAY),
        (7.0, 4.0, "Linear\n144→72", C_LGRAY, C_GRAY),
    ]
    for (x, y, lbl, bg, fg) in layers_shared:
        _rounded_box(ax, x, y, 1.2, 0.9, lbl, bg, fg, fontsize=7.5,
                     lw=1.3, pad=0.06)

    # Arrows in shared
    _arrow(ax, 1.9, 4.0, 2.6, 4.0, color=C_GRAY)
    for i in range(len(layers_shared) - 1):
        x1 = layers_shared[i][0] + 0.6
        x2 = layers_shared[i + 1][0] - 0.6
        _arrow(ax, x1 + 0.05, 4.0, x2 - 0.05, 4.0, color=C_GRAY)

    # ── Split point ──
    _label(ax, 7.9, 4.0, "→", fontsize=16, color=C_GRAY, weight="bold")

    # ── Value Stream (top) ──
    bx_v = FancyBboxPatch((8.3, 5.0), 5.2, 1.8,
                           boxstyle="round,pad=0.12",
                           linewidth=1.8, edgecolor=C_GREEN,
                           facecolor=C_LGREEN, zorder=1, alpha=0.4)
    ax.add_patch(bx_v)
    _label(ax, 10.9, 6.55, "Value Stream  V(s)", fontsize=10,
           weight="bold", color=C_DGREEN)

    _rounded_box(ax, 9.2, 5.9, 1.3, 0.7, "Linear\n72→36", C_LGREEN, C_GREEN,
                 fontsize=7.5, lw=1.3)
    _rounded_box(ax, 10.7, 5.9, 1.0, 0.7, "ReLU", C_LGREEN, C_GREEN,
                 fontsize=7.5, lw=1.3)
    _rounded_box(ax, 12.2, 5.9, 1.3, 0.7, "Linear\n36→1", C_LGREEN, C_DGREEN,
                 fontsize=7.5, lw=1.5)

    _arrow(ax, 9.85, 5.9, 10.2, 5.9, color=C_GREEN)
    _arrow(ax, 11.2, 5.9, 11.55, 5.9, color=C_GREEN)

    # ── Advantage Stream (bottom) ──
    bx_a = FancyBboxPatch((8.3, 1.2), 5.2, 1.8,
                           boxstyle="round,pad=0.12",
                           linewidth=1.8, edgecolor=C_PURPLE,
                           facecolor=C_LPURPLE, zorder=1, alpha=0.4)
    ax.add_patch(bx_a)
    _label(ax, 10.9, 2.75, "Advantage Stream  A(s,a)", fontsize=10,
           weight="bold", color=C_PURPLE)

    _rounded_box(ax, 9.2, 2.1, 1.3, 0.7, "Linear\n72→36", C_LPURPLE, C_PURPLE,
                 fontsize=7.5, lw=1.3)
    _rounded_box(ax, 10.7, 2.1, 1.0, 0.7, "ReLU", C_LPURPLE, C_PURPLE,
                 fontsize=7.5, lw=1.3)
    _rounded_box(ax, 12.2, 2.1, 1.3, 0.7, "Linear\n36→|A|", C_LPURPLE, C_PURPLE,
                 fontsize=7.5, lw=1.5)

    _arrow(ax, 9.85, 2.1, 10.2, 2.1, color=C_PURPLE)
    _arrow(ax, 11.2, 2.1, 11.55, 2.1, color=C_PURPLE)

    # Split arrows from shared to streams
    _arrow(ax, 7.7, 4.3, 8.5, 5.5, color=C_GREEN, lw=1.8)
    _arrow(ax, 7.7, 3.7, 8.5, 2.5, color=C_PURPLE, lw=1.8)

    # ── Aggregation ──
    _rounded_box(ax, 14.5, 4.0, 2.5, 1.6,
                 "Aggregation", C_LYELLOW, C_YELLOW,
                 sub="Q(s,a) = V(s) +\n[A(s,a) − mean A]",
                 fontsize=10, sub_fontsize=7.5, lw=2)

    # Streams-> Aggregation
    _arrow(ax, 12.85, 5.9, 13.25, 4.6, color=C_GREEN, lw=1.5, ls="--")
    _arrow(ax, 12.85, 2.1, 13.25, 3.4, color=C_PURPLE, lw=1.5, ls="--")

    _label(ax, 13.5, 5.5, "V(s)\nscalar", fontsize=7, color=C_GREEN,
           weight="bold")
    _label(ax, 13.5, 2.7, "A(s,a)\n|A|=10", fontsize=7, color=C_PURPLE,
           weight="bold")

    # ── Double DQN formula ──
    _label(ax, 8.0, 0.5,
           "Double DQN Update:   "
           r"$a^* = \arg\max_{a} Q_{\theta}(s', a)$"
           "    (policy selects)      "
           r"$y = r + \gamma \cdot Q_{\theta^{-}}(s', a^*)$"
           "    (target evaluates)"
           "      Huber Loss  ·  Gradient Clip max_norm=1.0  ·  Adam lr=1e-4",
           fontsize=8, color=C_GRAY,
           bbox_cfg=dict(boxstyle="round,pad=0.3", facecolor=C_LPURPLE,
                         edgecolor=C_PURPLE, lw=1.2))

    savefig(fig, "fig03_dueling_dqn.png")


# ═══════════════════════════════════════════════════════════════════════════
# FIG 4 – Three-Phase Curriculum Training
# ═══════════════════════════════════════════════════════════════════════════

def fig04_curriculum():
    fig, ax = plt.subplots(figsize=(15, 9))
    fig.patch.set_facecolor(C_WHITE)
    ax.set_xlim(0, 15); ax.set_ylim(-0.5, 9)
    ax.axis("off")

    _label(ax, 7.5, 8.5, "Three-Phase Curriculum Training Strategy",
           fontsize=15, weight="bold", color=C_DBLUE)

    phases = [
        {
            "title": "Phase 1: Human Supervised",
            "ep": "Episodes 1–3", "x": 2.5,
            "bg": C_LGREEN, "fg": C_GREEN,
            "items": [
                "Human selects every action",
                "DQN observes all transitions",
                "Expert transitions: PER ×2 boost",
                "Goal: inject expert policy early",
                "ε ≈ 1.0 (full exploration)",
            ],
            "icon": "H -> A"
        },
        {
            "title": "Phase 2: Human First-Step",
            "ep": "Episodes 4–8", "x": 7.5,
            "bg": C_LYELLOW, "fg": C_YELLOW,
            "items": [
                "Human guides step 0 only",
                "Agent handles remaining steps",
                "Learns multi-step strategies",
                "Goal: chained repair sequences",
                "ε gradually decays",
            ],
            "icon": "H -> A A A"
        },
        {
            "title": "Phase 3: RL Automated",
            "ep": "Episodes 9–50", "x": 12.5,
            "bg": C_LPURPLE, "fg": C_PURPLE,
            "items": [
                "Fully autonomous DQN",
                "ε-greedy exploration only",
                "Confidence gate for human help",
                "Goal: end-to-end optimization",
                "ε ≈ 0.05 (converged)",
            ],
            "icon": "A A A A"
        },
    ]

    PW, PH = 4.2, 3.5
    for ph in phases:
        x = ph["x"]
        _rounded_box(ax, x, 5.8, PW, PH, "", ph["bg"], ph["fg"], lw=2.0)

        # Title bar
        _label(ax, x, 7.3, ph["title"], fontsize=10, weight="bold",
               color=C_GRAY)
        _label(ax, x, 6.9, ph["ep"], fontsize=8.5, color=ph["fg"],
               weight="bold")

        # Items
        for i, item in enumerate(ph["items"]):
            _label(ax, x - PW / 2 + 0.3, 6.45 - i * 0.35,
                   f"• {item}", fontsize=7.5, color=C_GRAY, ha="left")

        # Autonomy indicator (H=Human, A=Agent)
        _label(ax, x, 4.2, ph["icon"], fontsize=12, family="monospace",
               weight="bold", color=ph["fg"])

    # Autonomy gradient arrow
    _arrow(ax, 1.0, 3.7, 14.0, 3.7, color=C_GRAY, lw=2.0, style="-|>")
    _label(ax, 7.5, 3.35, "Increasing Agent Autonomy ->", fontsize=10,
           weight="bold", color=C_GRAY, style="italic")

    # Timeline bar
    bar_y = 2.6
    bar_h = 0.45
    # Phase segments
    segments = [
        (0.8, 4.0, C_LGREEN, C_GREEN, "Phase 1\n(6%)"),
        (4.8, 4.4, C_LYELLOW, C_YELLOW, "Phase 2\n(10%)"),
        (9.2, 5.0, C_LPURPLE, C_PURPLE, "Phase 3\n(84%)"),
    ]
    for (x, w, bg, fg, lbl) in segments:
        bx = FancyBboxPatch((x, bar_y - bar_h / 2), w, bar_h,
                             boxstyle="round,pad=0.05", linewidth=1.5,
                             edgecolor=fg, facecolor=bg, zorder=3)
        ax.add_patch(bx)
        _label(ax, x + w / 2, bar_y, lbl, fontsize=7.5, color=fg,
               weight="bold")

    _label(ax, 7.5, 2.0, "Training Episodes  (total: 50, configurable)",
           fontsize=9, color=C_MGRAY)

    # ε-decay curve
    eps_x = np.linspace(0.8, 14.2, 300)
    steps = np.linspace(0, 750, 300)  # ~50 episodes × 15 steps
    eps_val = np.maximum(0.05, 1.0 * (0.994 ** steps))
    eps_y = 0.3 + eps_val * 1.3
    ax.plot(eps_x, eps_y, color=C_ORANGE, lw=2.5, zorder=5)
    ax.fill_between(eps_x, 0.3, eps_y, alpha=0.08, color=C_ORANGE)
    _label(ax, 14.5, eps_y[-1] + 0.1, "ε", fontsize=14, color=C_ORANGE,
           weight="bold")
    _label(ax, 0.5, 1.55, "1.0", fontsize=7, color=C_ORANGE)
    _label(ax, 14.3, 0.42, "0.05", fontsize=7, color=C_ORANGE)

    _label(ax, 7.5, -0.1,
           "ε-decay: per-step × 0.994   |   start = 1.0   |   min = 0.05   |   "
           "~750 total agent steps over 50 episodes",
           fontsize=8, color=C_ORANGE, style="italic")

    savefig(fig, "fig04_curriculum_training.png")


# ═══════════════════════════════════════════════════════════════════════════
# FIG 5 – Reward Function Design
# ═══════════════════════════════════════════════════════════════════════════

def fig05_reward():
    fig, axes = plt.subplots(1, 2, figsize=(16, 7),
                              gridspec_kw={"width_ratios": [1, 1.2]})
    fig.patch.set_facecolor(C_WHITE)
    fig.suptitle("Reward Function Design — Components & Training Behaviour",
                 fontsize=14, fontweight="bold", color=C_DBLUE, y=0.97)

    # ── Left: Component bar chart ──
    ax = axes[0]
    ax.set_facecolor(C_WHITE)
    components = [
        ("KG Consistent\n(+ efficiency bonus)", +12, C_DGREEN),
        ("KG Consistent\n(base)", +10, C_GREEN),
        ("Violation Resolved", +2, "#86EFAC"),
        ("Per-Step Cost", -0.1, C_LORANGE),
        ("No-Op / No Change", -0.5, C_ORANGE),
        ("New Violation\nIntroduced", -2, C_RED),
        ("Regression\n(Lost Consistency)", -10, "#991B1B"),
    ]
    labels = [c[0] for c in components]
    values = [c[1] for c in components]
    colors = [c[2] for c in components]

    bars = ax.barh(labels, values, color=colors, edgecolor=C_GRAY,
                   linewidth=0.6, height=0.55)
    ax.axvline(0, color=C_GRAY, lw=1.5)
    for bar, val in zip(bars, values):
        xpos = val + (0.4 if val >= 0 else -0.4)
        ax.text(xpos, bar.get_y() + bar.get_height() / 2,
                f"{val:+.1f}", va="center",
                ha="left" if val >= 0 else "right",
                fontsize=9, fontweight="bold", color=C_GRAY)
    ax.set_xlabel("Reward Value", fontsize=10, color=C_GRAY)
    ax.set_title("Reward Signal Components", fontsize=12, fontweight="bold",
                 color=C_GRAY, pad=10)
    ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(-13, 15)

    # ── Right: Simulated training curves ──
    ax2 = axes[1]
    ax2.set_facecolor(C_WHITE)
    np.random.seed(42)
    episodes = np.arange(1, 51)

    # Simulate episode rewards across phases
    r_phase1 = np.random.normal(4.0, 3.0, 3)
    r_phase2 = np.random.normal(5.5, 2.5, 5)
    r_phase3 = np.random.normal(7.0, 2.0, 42)
    ep_rewards = np.concatenate([r_phase1, r_phase2, r_phase3])
    ep_rewards = np.clip(ep_rewards, -5, 14)

    # Cumulative
    cum_rewards = np.cumsum(ep_rewards)

    ax2.bar(episodes, ep_rewards, width=0.7, alpha=0.5, color=C_BLUE,
            edgecolor=C_BLUE, linewidth=0.3, label="Episode reward")

    ax2_twin = ax2.twinx()
    ax2_twin.plot(episodes, cum_rewards, color=C_RED, lw=2.5,
                  label="Cumulative reward", zorder=5)
    ax2_twin.fill_between(episodes, cum_rewards, alpha=0.05, color=C_RED)

    # Phase shading
    ax2.axvspan(1, 3.5, alpha=0.12, color=C_GREEN, label="Phase 1")
    ax2.axvspan(3.5, 8.5, alpha=0.12, color=C_YELLOW, label="Phase 2")
    ax2.axvspan(8.5, 50, alpha=0.08, color=C_PURPLE, label="Phase 3")
    ax2.axvline(3.5, color="#9CA3AF", lw=1, ls="--")
    ax2.axvline(8.5, color="#9CA3AF", lw=1, ls="--")

    ax2.set_xlabel("Episode", fontsize=10, color=C_GRAY)
    ax2.set_ylabel("Episode Reward", fontsize=10, color=C_BLUE)
    ax2_twin.set_ylabel("Cumulative Reward", fontsize=10, color=C_RED)

    ax2.set_title("Simulated Training Progress", fontsize=12,
                  fontweight="bold", color=C_GRAY, pad=10)

    # Combined legend
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2,
               fontsize=8, loc="upper left")

    ax2.tick_params(labelsize=8)
    ax2_twin.tick_params(labelsize=8)
    ax2.spines[["top"]].set_visible(False)
    ax2_twin.spines[["top"]].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    savefig(fig, "fig05_reward_function.png")


# ═══════════════════════════════════════════════════════════════════════════
# FIG 6 – Core Research Contributions
# ═══════════════════════════════════════════════════════════════════════════

def fig06_contributions():
    fig, ax = plt.subplots(figsize=(16, 10))
    fig.patch.set_facecolor(C_WHITE)
    ax.set_xlim(-0.5, 16); ax.set_ylim(-0.5, 10)
    ax.axis("off")

    _label(ax, 8, 9.5, "KGG4SE — Core Research Contributions",
           fontsize=16, weight="bold", color=C_DBLUE)

    contributions = [
        {
            "num": "C1", "title": "LLM-Driven\nTriple Extraction",
            "x": 2.5, "y": 7.5, "bg": C_LBLUE, "fg": C_BLUE,
            "items": [
                "GPT-4o-mini (T=0.0)",
                "Sentence chunking",
                "Garbage-term filtering",
                "LLaMA-3 fallback (local)",
                "Output: S|P|O triples",
            ]
        },
        {
            "num": "C2", "title": "Three-Level\nOntology Alignment",
            "x": 8.0, "y": 7.5, "bg": C_LGREEN, "fg": C_GREEN,
            "items": [
                "L1: CSV lookup maps",
                "L2: Exact IRI match",
                "L3: RapidFuzz (≥85%)",
                "GENIALOnt (BFO-based)",
                "OWL 2 via RDFLib",
            ]
        },
        {
            "num": "C3", "title": "Dual-Reasoner\nQA Pipeline",
            "x": 13.5, "y": 7.5, "bg": C_LORANGE, "fg": C_ORANGE,
            "items": [
                "Konclude (OWL 2, primary)",
                "HermiT (DL, evaluation)",
                "7 violation categories",
                "Heuristic axiom extraction",
                "Unified report schema",
            ]
        },
        {
            "num": "C4", "title": "Evidence-Based\nRepair Candidates",
            "x": 2.5, "y": 3.0, "bg": C_LTEAL, "fg": C_TEAL,
            "items": [
                "Priority-ordered actions",
                "Risk levels: low/med/high",
                "8 atomic repair operations",
                "Ontology-aware indexing",
                "Domain/range checking",
            ]
        },
        {
            "num": "C5", "title": "Dueling Double-DQN\n+ PER Agent",
            "x": 8.0, "y": 3.0, "bg": C_LPURPLE, "fg": C_PURPLE,
            "items": [
                "18-dim state encoding",
                "10-action repair space",
                "Dueling V(s) + A(s,a)",
                "PER (α=0.6, β→1.0)",
                "Action masking + clipping",
            ]
        },
        {
            "num": "C6", "title": "Three-Phase\nCurriculum Training",
            "x": 13.5, "y": 3.0, "bg": C_LYELLOW, "fg": C_YELLOW,
            "items": [
                "Ph.1: human supervised",
                "Ph.2: human first-step",
                "Ph.3: fully autonomous",
                "Per-step ε-decay (0.994)",
                "Expert transition ×2 boost",
            ]
        },
    ]

    CW, CH = 4.5, 3.0
    for c in contributions:
        x, y = c["x"], c["y"]
        _rounded_box(ax, x, y, CW, CH, "", c["bg"], c["fg"], lw=2.0)

        # Badge
        badge = Circle((x - CW / 2 + 0.35, y + CH / 2 - 0.35), 0.28,
                        color=c["fg"], zorder=6)
        ax.add_patch(badge)
        _label(ax, x - CW / 2 + 0.35, y + CH / 2 - 0.35,
               c["num"], fontsize=9, weight="bold", color=C_WHITE)

        # Title
        _label(ax, x, y + CH / 2 - 0.35, c["title"], fontsize=10,
               weight="bold", color=C_GRAY)

        # Items
        for i, item in enumerate(c["items"]):
            _label(ax, x - CW / 2 + 0.35, y + CH / 2 - 0.95 - i * 0.38,
                   f"• {item}", fontsize=7.8, color=C_GRAY, ha="left")

    # Center branding
    _label(ax, 8.0, 5.35, "KGG4SE", fontsize=26, weight="bold",
           color=C_DBLUE, style="italic",
           bbox_cfg=dict(boxstyle="round,pad=0.4", facecolor=C_WHITE,
                         edgecolor=C_DBLUE, lw=2.8))
    _label(ax, 8.0, 4.7,
           "Knowledge Graph Generation for Software Engineering",
           fontsize=10, color=C_GRAY)
    _label(ax, 8.0, 4.3, "Automotive Electronics Domain",
           fontsize=9, color=C_ORANGE, weight="bold")

    # Connecting lines from center to each contribution
    for c in contributions:
        cy = 5.35
        if c["y"] > 5:
            ty = c["y"] - CH / 2
            cy = 5.6
        else:
            ty = c["y"] + CH / 2
            cy = 5.0
        _arrow(ax, 8.0, cy, c["x"], ty, color=C_BORDER, lw=1.0, ls=":")

    _label(ax, 8.0, 0.3,
           "Framework integrates:  Large Language Models  ·  OWL 2 Ontologies  ·  "
           "Automated Reasoners  ·  Deep Reinforcement Learning (PyTorch)",
           fontsize=9, color=C_MGRAY, style="italic")

    savefig(fig, "fig06_core_contributions.png")


# ═══════════════════════════════════════════════════════════════════════════
# FIG 7 – Detailed Data Flow with File Artifacts
# ═══════════════════════════════════════════════════════════════════════════

def fig07_data_flow():
    fig, ax = plt.subplots(figsize=(16, 12))
    fig.patch.set_facecolor(C_WHITE)
    ax.set_xlim(-0.5, 16); ax.set_ylim(-1, 12)
    ax.axis("off")

    _label(ax, 8, 11.5, "KGG4SE — Data Flow & File Artifact Pipeline",
           fontsize=15, weight="bold", color=C_DBLUE)

    # Flow: top to bottom
    flow_items = [
        # (label, sub, y, bg, fg, artifact_name)
        ("corpus.csv", "Raw domain text data\n(automotive electronics)",
         10.5, C_LBLUE, C_BLUE, None),
        ("Step 1: prepare_corpus", "CSV-> plain text (one entry per line)",
         9.5, C_LBLUE, C_BLUE, "corpus.txt"),
        ("Step 2: generate_triples", "LLM API call (GPT-4o-mini / LLaMA-3)\n"
         "Parse S|P|O lines, filter garbage terms",
         8.2, C_LBLUE, C_BLUE, "corpus_triples.json"),
        ("Step 3: clean_triples", "Normalize entities, alias substitution,\n"
         "split compounds, deduplicate",
         6.9, C_LBLUE, C_BLUE, "corpus_triples_cleaned.json"),
        ("Step 4: align_triples", "CSV map-> exact match-> fuzzy match (≥85%)\n"
         "Map to GENIALOnt URIs, mint individuals",
         5.6, C_LGREEN, C_GREEN, "aligned_triples.ttl"),
        ("Step 5: build_kg", "Union base ontology + aligned triples\n"
         "Serialize to RDF/XML and Turtle",
         4.3, C_LGREEN, C_GREEN, "merged_kg.owl"),
        ("Step 6: run_reasoner", "Konclude classification / HermiT DL\n"
         "Detect inconsistencies & unsat classes",
         3.0, C_LORANGE, C_ORANGE, "merged_kg_reasoned.owl"),
        ("Step 7: parse_reasoner", "Extract violation evidence\n"
         "Axiom-level heuristic analysis",
         1.7, C_LORANGE, C_ORANGE, "reasoned_triples.json"),
        ("Step 8: check_quality", "Quality statistics, duplicate detection\n"
         "Top-N analysis",
         0.5, C_LGRAY, C_MGRAY, "quality_report.json"),
    ]

    PW, PH = 4.5, 0.9
    AW = 3.0  # artifact box width

    for i, (label, sub, y, bg, fg, artifact) in enumerate(flow_items):
        # Process box
        _rounded_box(ax, 5.5, y, PW, PH, label, bg, fg,
                     fontsize=9 if i > 0 else 10, lw=1.8)

        # Description
        _label(ax, 5.5, y - PH / 2 - 0.25, sub, fontsize=7, color=C_MGRAY,
               style="italic")

        # Arrow to next
        if i < len(flow_items) - 1:
            next_y = flow_items[i + 1][2]
            _arrow(ax, 5.5, y - PH / 2 - 0.45, 5.5, next_y + PH / 2 + 0.05,
                   color=C_MGRAY, lw=1.3)

        # Artifact output box (right side)
        if artifact:
            ax_x = 11.0
            _label(ax, ax_x, y, artifact, fontsize=8, color=fg, weight="bold",
                   family="monospace",
                   bbox_cfg=dict(boxstyle="round,pad=0.2", facecolor=bg,
                                 edgecolor=fg, lw=1.2, alpha=0.7))
            # Dashed arrow from process to artifact
            _arrow(ax, 5.5 + PW / 2 + 0.1, y, ax_x - 1.3, y,
                   color=fg, lw=1.0, ls="--")

    # ── RL Repair Branch ──
    _rounded_box(ax, 13.0, 2.0, 3.5, 4.5, "", C_LPURPLE, C_PURPLE, lw=2.0)
    _label(ax, 13.0, 4.0, "Step 9: RL Repair", fontsize=10, weight="bold",
           color=C_PURPLE)

    rl_items = [
        "RepairEnv loads KG",
        "Detect violations",
        "Build repair candidates",
        "DQN selects action",
        "apply_fix() modifies KG",
        "Re-run reasoner",
        "Compute reward",
        "Update DQN weights",
    ]
    for i, item in enumerate(rl_items):
        _label(ax, 11.8, 3.5 - i * 0.35, f"{i+1}. {item}",
               fontsize=7, color=C_GRAY, ha="left")

    # Arrow from reasoner to RL
    _arrow(ax, 7.75, 3.0, 11.2, 3.0, color=C_PURPLE, lw=1.8, ls="--")
    _label(ax, 9.5, 3.2, "violations", fontsize=7.5, color=C_PURPLE,
           style="italic")

    # RL output
    _label(ax, 13.0, 0.2, "dqn_repair_final.pt\n+ training_history.json\n"
           "+ repair_trace.jsonl",
           fontsize=7.5, color=C_PURPLE, weight="bold", family="monospace",
           bbox_cfg=dict(boxstyle="round,pad=0.2", facecolor=C_LPURPLE,
                         edgecolor=C_PURPLE, lw=1.2))

    # Feedback loop arrow
    _arrow(ax, 14.5, 0.8, 14.5, 3.8, color=C_RED, lw=2.0, ls="--",
           style="<|-")
    _label(ax, 15.0, 2.3, "Repair\nLoop", fontsize=8, color=C_RED,
           style="italic")

    # Config inputs
    _label(ax, 1.0, 5.6, "entity_map.csv\nrelation_map.csv",
           fontsize=7.5, color=C_TEAL, weight="bold", family="monospace",
           bbox_cfg=dict(boxstyle="round,pad=0.2", facecolor=C_LTEAL,
                         edgecolor=C_TEAL, lw=1))
    _arrow(ax, 2.3, 5.6, 3.25, 5.6, color=C_TEAL, lw=1.2, ls="--")

    _label(ax, 1.0, 4.3, "GENIALOnt\n(BFO ontology)",
           fontsize=7.5, color=C_GREEN, weight="bold",
           bbox_cfg=dict(boxstyle="round,pad=0.2", facecolor=C_LGREEN,
                         edgecolor=C_GREEN, lw=1))
    _arrow(ax, 2.2, 4.3, 3.25, 4.3, color=C_GREEN, lw=1.2, ls="--")

    savefig(fig, "fig07_data_flow.png")


# ═══════════════════════════════════════════════════════════════════════════
# FIG 8 – Three-Tier Ontology Alignment
# ═══════════════════════════════════════════════════════════════════════════

def fig08_ontology_alignment():
    fig, ax = plt.subplots(figsize=(15, 9))
    fig.patch.set_facecolor(C_WHITE)
    ax.set_xlim(-0.5, 15); ax.set_ylim(-0.5, 9)
    ax.axis("off")

    _label(ax, 7.5, 8.5, "Three-Tier Ontology Alignment Strategy",
           fontsize=15, weight="bold", color=C_DBLUE)

    # Input
    _rounded_box(ax, 2.5, 7.2, 4.0, 1.0,
                 "Cleaned Triple", C_LBLUE, C_BLUE,
                 sub='e.g., (cpu, is_part_of, computer)',
                 fontsize=10, sub_fontsize=8)

    # Input components
    for i, (lbl, col) in enumerate([("Subject", C_BLUE), ("Predicate", C_GREEN),
                                      ("Object", C_PURPLE)]):
        _rounded_box(ax, 1.5 + i * 2.0, 5.8, 1.6, 0.7, lbl, C_LGRAY, col,
                     fontsize=9, lw=1.5)
        _arrow(ax, 1.5 + i * 2.0, 6.15, 2.5, 6.7, color=col, lw=1.2)

    # Three tiers
    tiers = [
        {
            "num": "Tier 1", "title": "CSV Lookup",
            "x": 3.0, "y": 4.0,
            "bg": C_LGREEN, "fg": C_GREEN,
            "desc": "Hand-crafted mapping tables\n"
                    "entity_map.csv-> classes/individuals\n"
                    "relation_map.csv-> properties\n"
                    "Priority: HIGHEST (exact domain knowledge)",
            "example": "cpu-> CentralProcessingUnit"
        },
        {
            "num": "Tier 2", "title": "Exact IRI Match",
            "x": 7.5, "y": 4.0,
            "bg": C_LYELLOW, "fg": C_YELLOW,
            "desc": "Ontology index scan\n"
                    "Match by rdfs:label or localname\n"
                    "Case-insensitive comparison\n"
                    "Priority: MEDIUM",
            "example": "sensor-> Sensor (rdfs:label)"
        },
        {
            "num": "Tier 3", "title": "Fuzzy Match",
            "x": 12.0, "y": 4.0,
            "bg": C_LPURPLE, "fg": C_PURPLE,
            "desc": "RapidFuzz string similarity\n"
                    "Threshold: score ≥ 85%\n"
                    "Fallback: mint new /ind/ individual\n"
                    "Priority: LOWEST",
            "example": "temp_sensor-> TemperatureSensor (91%)"
        },
    ]

    TW, TH = 3.8, 2.8
    for t in tiers:
        x, y = t["x"], t["y"]
        _rounded_box(ax, x, y, TW, TH, "", t["bg"], t["fg"], lw=2.0)

        _label(ax, x, y + TH / 2 - 0.3, f'{t["num"]}: {t["title"]}',
               fontsize=10, weight="bold", color=C_GRAY)

        lines = t["desc"].split("\n")
        for i, line in enumerate(lines):
            _label(ax, x - TW / 2 + 0.2, y + TH / 2 - 0.7 - i * 0.32,
                   line, fontsize=7.5, color=C_GRAY, ha="left")

        # Example
        _label(ax, x, y - TH / 2 + 0.3, t["example"],
               fontsize=7.5, color=t["fg"], weight="bold", style="italic",
               bbox_cfg=dict(boxstyle="round,pad=0.12", facecolor=C_WHITE,
                             edgecolor=t["fg"], lw=0.8))

    # Flow arrows: each tier tries, if fail-> next
    _arrow(ax, 3.0, 5.15, 3.0, 5.4, color=C_GRAY, lw=1.5)
    _arrow(ax, 7.5, 5.15, 7.5, 5.4, color=C_GRAY, lw=1.5)
    _arrow(ax, 12.0, 5.15, 12.0, 5.4, color=C_GRAY, lw=1.5)

    # Tier fallback arrows
    _arrow(ax, 4.9, 4.8, 5.6, 4.8, color=C_RED, lw=1.5, ls="--")
    _label(ax, 5.25, 5.05, "miss", fontsize=7.5, color=C_RED, style="italic")
    _arrow(ax, 9.4, 4.8, 10.1, 4.8, color=C_RED, lw=1.5, ls="--")
    _label(ax, 9.75, 5.05, "miss", fontsize=7.5, color=C_RED, style="italic")

    # Output
    _rounded_box(ax, 7.5, 1.2, 6.0, 1.2,
                 "Aligned Triple (OWL URIs)", C_LGREEN, C_DGREEN,
                 sub="e.g., (genialont:ind/cpu, genialont:isPartOf, genialont:Computer)",
                 fontsize=10, sub_fontsize=7.5, lw=2)

    # Arrows from tiers to output
    for x in [3.0, 7.5, 12.0]:
        _arrow(ax, x, 4.0 - TH / 2, 7.5, 1.8, color=C_GREEN, lw=1.2, ls=":")
    _label(ax, 3.5, 2.2, "hit", fontsize=7.5, color=C_GREEN, weight="bold")

    # Special handling note
    _label(ax, 7.5, -0.1,
           "Special handling:  domain/range type assertions auto-added  ·  "
           "external namespace types skipped  ·  class→individual conversion",
           fontsize=8, color=C_MGRAY, style="italic")

    savefig(fig, "fig08_ontology_alignment.png")


# ═══════════════════════════════════════════════════════════════════════════
# FIG 9 – Violation Types-> Repair Actions Mapping
# ═══════════════════════════════════════════════════════════════════════════

def fig09_repair_candidates():
    fig, ax = plt.subplots(figsize=(16, 10))
    fig.patch.set_facecolor(C_WHITE)
    ax.set_xlim(-0.5, 16); ax.set_ylim(-0.5, 10)
    ax.axis("off")

    _label(ax, 8, 9.5, "Violation Types-> Repair Action Mapping",
           fontsize=15, weight="bold", color=C_DBLUE)

    # Violation types (left)
    violations = [
        ("Disjoint Violation", C_RED, "Individual typed into\ntwo disjoint classes"),
        ("Domain Violation", C_ORANGE, "Subject not typed as\nproperty domain class"),
        ("Range Violation", C_YELLOW, "Object not typed as\nproperty range class"),
        ("Class-as-Individual", C_PURPLE, "owl:Class used as\nstatement subject"),
        ("Functional Property\nViolation", C_PINK, "Functional property\nhas 2+ fillers"),
        ("Property Type\nMismatch", C_TEAL, "ObjectProperty used\nwith Literal value"),
        ("Unsatisfiable Class", C_GRAY, "Class cannot have\nany instances"),
    ]

    # Repair actions (right)
    actions = [
        ("drop_class_assertion", "low", C_GREEN),
        ("add_type_assertion", "low", C_GREEN),
        ("remove_property_assertion", "medium", C_YELLOW),
        ("drop_entity", "high", C_RED),
        ("remap_entity", "medium", C_YELLOW),
        ("mint_individual", "low", C_GREEN),
        ("remove_disjoint_axiom", "medium", C_YELLOW),
        ("no-op", "none", C_LGRAY),
    ]

    VW, VH = 3.2, 0.9
    AW_box, AH = 3.2, 0.7

    # Draw violation boxes
    vy_start = 8.5
    vy_step = 1.2
    for i, (name, color, desc) in enumerate(violations):
        vy = vy_start - i * vy_step
        _rounded_box(ax, 2.5, vy, VW, VH, name, f"{color}22", color,
                     fontsize=8.5, lw=1.8)
        _label(ax, 2.5, vy - VH / 2 - 0.15, desc, fontsize=6.5,
               color=C_MGRAY, style="italic")

    # Draw action boxes
    ay_start = 8.3
    ay_step = 1.0
    for i, (name, risk, color) in enumerate(actions):
        ay = ay_start - i * ay_step
        _rounded_box(ax, 13.0, ay, AW_box, AH, name, f"{color}22", color,
                     fontsize=8, lw=1.5)
        risk_colors = {"low": C_GREEN, "medium": C_YELLOW, "high": C_RED, "none": C_MGRAY}
        _label(ax, 14.8, ay, f"[{risk}]", fontsize=7, color=risk_colors[risk],
               weight="bold")

    # Connection lines (violation-> actions)
    connections = [
        # (violation_idx, action_indices)
        (0, [0, 3]),           # Disjoint-> drop_class, drop_entity
        (1, [1, 2]),           # Domain-> add_type, remove_prop
        (2, [1, 2]),           # Range-> add_type, remove_prop
        (3, [5, 3]),           # Class-as-ind-> mint_individual, drop_entity
        (4, [2]),              # Functional-> remove_prop
        (5, [2]),              # Type mismatch-> remove_prop
        (6, [3, 4]),           # Unsat-> drop_entity, remap_entity
    ]

    for vi, ais in connections:
        vy = vy_start - vi * vy_step
        for ai in ais:
            ay = ay_start - ai * ay_step
            _arrow(ax, 2.5 + VW / 2 + 0.1, vy,
                   13.0 - AW_box / 2 - 0.1, ay,
                   color="#9CA3AF", lw=0.8, rad=0.05)

    # Labels
    _label(ax, 2.5, 9.1, "Violation Types (Detected)", fontsize=11,
           weight="bold", color=C_RED)
    _label(ax, 13.0, 9.1, "Repair Actions (Candidates)", fontsize=11,
           weight="bold", color=C_GREEN)

    # Risk legend
    _label(ax, 8.0, 0.5,
           "Risk Levels:   ■ low (safe, reversible)   "
           "■ medium (structural change)   "
           "■ high (data loss possible)",
           fontsize=8.5, color=C_GRAY,
           bbox_cfg=dict(boxstyle="round,pad=0.3", facecolor=C_LGRAY,
                         edgecolor=C_BORDER, lw=1))

    savefig(fig, "fig09_repair_candidates.png")


# ═══════════════════════════════════════════════════════════════════════════
# FIG 10 – RL State & Action Space Visualization
# ═══════════════════════════════════════════════════════════════════════════

def fig10_state_action():
    fig, axes = plt.subplots(1, 2, figsize=(16, 8),
                              gridspec_kw={"width_ratios": [1.2, 1]})
    fig.patch.set_facecolor(C_WHITE)
    fig.suptitle("RL State & Action Space — Detailed Encoding",
                 fontsize=14, fontweight="bold", color=C_DBLUE, y=0.97)

    # ── Left: State vector heatmap ──
    ax = axes[0]
    ax.set_facecolor(C_WHITE)

    state_features = [
        ("Dim 0–9", "Violation Type\n(one-hot)", "Binary",
         "10 error categories:\ndisjoint, domain, range,\nclass-as-individual, etc."),
        ("Dim 10", "Num Actions\nAvailable", "[0, 1]",
         "Normalized count of\ncandidate repair actions"),
        ("Dim 11", "Low-Risk\nActions", "[0, 1]",
         "Proportion of safe\n(low-risk) options"),
        ("Dim 12", "Has Evidence\nFlag", "{0, 1}",
         "Whether axiom-level\nevidence was found"),
        ("Dim 13", "Step Progress\nRatio", "[0, 1]",
         "current_step / max_steps\n(efficiency signal)"),
        ("Dim 14", "Has Entity\nIRI", "{0, 1}",
         "Whether a concrete\nentity was identified"),
        ("Dim 15", "Queue Progress\nRatio", "[0, 1]",
         "remaining / initial\nviolation count"),
        ("Dim 16", "Last Step\nImproved", "{0, 1}",
         "Momentum signal:\ndid last action help?"),
        ("Dim 17", "Min Action\nRisk", "[0, 1]",
         "Safest available\naction risk / 5.0"),
    ]

    colors_state = [C_BLUE, C_BLUE, C_BLUE, C_BLUE, C_BLUE,
                    C_BLUE, C_BLUE, C_BLUE, C_BLUE, C_BLUE,
                    C_GREEN, C_GREEN, C_TEAL, C_ORANGE,
                    C_TEAL, C_PURPLE, C_YELLOW, C_PINK]

    y_pos = 0.92
    header_h = 0.03
    row_h = 0.085

    ax.text(0.05, 0.97, "State Vector  s ∈ ℝ¹⁸", fontsize=13,
            fontweight="bold", color=C_INDIGO, transform=ax.transAxes)

    # Table header
    for col, (header, x_pos) in enumerate(
        [("Index", 0.05), ("Feature", 0.20), ("Range", 0.55), ("Description", 0.70)]):
        ax.text(x_pos, y_pos, header, fontsize=9, fontweight="bold",
                color=C_GRAY, transform=ax.transAxes, va="top")

    ax.plot([0.02, 0.98], [y_pos - 0.01, y_pos - 0.01],
            color=C_GRAY, lw=1.2, transform=ax.transAxes, clip_on=False)

    for i, (idx, feature, rng, desc) in enumerate(state_features):
        y = y_pos - 0.035 - i * row_h
        bg_color = C_LGRAY if i % 2 == 0 else C_WHITE
        rect = mpatches.FancyBboxPatch((0.02, y - row_h / 2 + 0.02), 0.96,
                                         row_h - 0.02,
                                         boxstyle="round,pad=0.005",
                                         facecolor=bg_color, alpha=0.5,
                                         edgecolor="none",
                                         transform=ax.transAxes, zorder=0)
        ax.add_patch(rect)
        ax.text(0.05, y, idx, fontsize=7.5, color=C_BLUE, fontweight="bold",
                transform=ax.transAxes, va="center", family="monospace")
        ax.text(0.20, y, feature, fontsize=7.5, color=C_GRAY,
                transform=ax.transAxes, va="center")
        ax.text(0.55, y, rng, fontsize=7.5, color=C_MGRAY, family="monospace",
                transform=ax.transAxes, va="center")
        ax.text(0.70, y, desc, fontsize=6.5, color=C_MGRAY,
                transform=ax.transAxes, va="center")

    ax.axis("off")

    # ── Right: Action space ──
    ax2 = axes[1]
    ax2.set_facecolor(C_WHITE)

    repair_actions = [
        ("drop_class_assertion", "Remove a class type\nfrom individual", "Low"),
        ("add_type_assertion", "Assert missing type\nfor individual", "Low"),
        ("remove_property", "Remove object/data\nproperty assertion", "Medium"),
        ("drop_entity", "Remove entire entity\nfrom KG", "High"),
        ("remap_entity", "Remap IRI to\nalternative entity", "Medium"),
        ("mint_individual", "Create new individual\nfor class URI", "Low"),
        ("remove_disjoint", "Remove disjoint\naxiom constraint", "Medium"),
        ("remove_specific_prop", "Remove specific\nproperty by IRI", "Medium"),
        ("drop_class_assertion_2", "Remove second class\ntype assertion", "Low"),
        ("no-op (padding)", "Placeholder for\nunused action slots", "None"),
    ]

    ax2.text(0.5, 0.97, "Action Space  |A| = 10", fontsize=13,
             fontweight="bold", color=C_PURPLE,
             transform=ax2.transAxes, ha="center")

    risk_colors = {"Low": C_GREEN, "Medium": C_YELLOW, "High": C_RED, "None": C_MGRAY}

    for i, (name, desc, risk) in enumerate(repair_actions):
        y = 0.88 - i * 0.088
        rc = risk_colors[risk]

        # Risk indicator circle
        circle = plt.Circle((0.05, y), 0.015, color=rc,
                            transform=ax2.transAxes, zorder=5)
        ax2.add_patch(circle)

        ax2.text(0.10, y, f"a{i}", fontsize=8, fontweight="bold",
                 color=C_PURPLE, transform=ax2.transAxes, va="center",
                 family="monospace")
        ax2.text(0.20, y, name, fontsize=8, color=C_GRAY,
                 transform=ax2.transAxes, va="center", family="monospace")
        ax2.text(0.70, y, desc, fontsize=6.5, color=C_MGRAY,
                 transform=ax2.transAxes, va="center")
        ax2.text(0.95, y, f"[{risk}]", fontsize=7, color=rc,
                 fontweight="bold", transform=ax2.transAxes, va="center")

    ax2.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    savefig(fig, "fig10_state_action_space.png")


# ═══════════════════════════════════════════════════════════════════════════
# FIG 11 – Layered System Architecture
# ═══════════════════════════════════════════════════════════════════════════

def fig11_layered_arch():
    fig, ax = plt.subplots(figsize=(14, 10))
    fig.patch.set_facecolor(C_WHITE)
    ax.set_xlim(-0.5, 14); ax.set_ylim(-0.5, 10)
    ax.axis("off")

    _label(ax, 7, 9.5, "KGG4SE — Layered System Architecture",
           fontsize=15, weight="bold", color=C_DBLUE)

    # Layers (bottom to top)
    layers = [
        {
            "name": "Layer 1: Data & Configuration",
            "y": 1.0, "bg": C_LGRAY, "fg": C_GRAY,
            "modules": [
                ("corpus.csv\ncorpus.txt", C_LGRAY),
                ("entity_map.csv\nrelation_map.csv", C_LGRAY),
                ("GENIALOnt.owl\n(BFO ontology)", C_LGRAY),
                ("config.py\n(hyperparams)", C_LGRAY),
            ]
        },
        {
            "name": "Layer 2: LLM & Extraction",
            "y": 3.0, "bg": C_LBLUE, "fg": C_BLUE,
            "modules": [
                ("generate_triples.py\n(GPT-4o-mini/LLaMA)", C_LBLUE),
                ("clean_triples.py\n(normalize + dedup)", C_LBLUE),
                ("prepare_corpus.py\n(CSV→text)", C_LBLUE),
            ]
        },
        {
            "name": "Layer 3: Ontology & KG Build",
            "y": 5.0, "bg": C_LGREEN, "fg": C_GREEN,
            "modules": [
                ("ontology_index.py\n(IRI indexing)", C_LGREEN),
                ("align_triples.py\n(3-tier alignment)", C_LGREEN),
                ("build_kg.py\n(OWL merge)", C_LGREEN),
            ]
        },
        {
            "name": "Layer 4: Reasoning & QA",
            "y": 7.0, "bg": C_LORANGE, "fg": C_ORANGE,
            "modules": [
                ("konclude_runner.py\nhermit_runner.py", C_LORANGE),
                ("axiom_extractor.py\n(evidence)", C_LORANGE),
                ("quality_extractor.py\n(metrics)", C_LORANGE),
                ("repair_candidates.py\n(action gen)", C_LORANGE),
            ]
        },
        {
            "name": "Layer 5: RL Repair Agent",
            "y": 8.8, "bg": C_LPURPLE, "fg": C_PURPLE,
            "modules": [
                ("env_repair.py\n(MDP environment)", C_LPURPLE),
                ("dqn_agent.py\n(policy+target)", C_LPURPLE),
                ("dqn_model.py\n(dueling net)", C_LPURPLE),
                ("train_repair.py\n(curriculum)", C_LPURPLE),
                ("replay_buffer.py\n(PER)", C_LPURPLE),
            ]
        },
    ]

    LW = 13.0  # layer width
    LH = 1.3   # layer height

    for layer in layers:
        y = layer["y"]
        # Layer background
        bx = FancyBboxPatch((0.5, y - LH / 2), LW, LH,
                             boxstyle="round,pad=0.1",
                             linewidth=2.0, edgecolor=layer["fg"],
                             facecolor=layer["bg"], zorder=1, alpha=0.3)
        ax.add_patch(bx)

        # Layer name (left)
        _label(ax, 0.8, y + LH / 2 - 0.15, layer["name"], fontsize=9,
               weight="bold", color=layer["fg"], ha="left")

        # Module boxes
        n_mods = len(layer["modules"])
        mod_w = min(2.2, (LW - 1.5) / n_mods - 0.3)
        total_w = n_mods * (mod_w + 0.3) - 0.3
        start_x = 0.5 + (LW - total_w) / 2

        for j, (mod_name, mod_bg) in enumerate(layer["modules"]):
            mx = start_x + j * (mod_w + 0.3) + mod_w / 2
            _rounded_box(ax, mx, y - 0.15, mod_w, 0.7, mod_name,
                         C_WHITE, layer["fg"], fontsize=6.5, lw=1.2,
                         pad=0.05, zorder=3)

    # Inter-layer arrows
    for i in range(len(layers) - 1):
        y1 = layers[i]["y"] + LH / 2
        y2 = layers[i + 1]["y"] - LH / 2
        _arrow(ax, 7, y1, 7, y2, color=C_MGRAY, lw=1.5)

    # Side label: pipeline.py orchestrates
    _label(ax, 14.2, 5.0,
           "pipeline.py\n(CLI orchestration)\n\nPipelineStep\nabstract base",
           fontsize=8, color=C_GRAY, weight="bold",
           bbox_cfg=dict(boxstyle="round,pad=0.3", facecolor=C_LGRAY,
                         edgecolor=C_BORDER, lw=1.2))

    savefig(fig, "fig11_layered_architecture.png")


# ═══════════════════════════════════════════════════════════════════════════
# FIG 12 – LLM Triple Extraction Pipeline
# ═══════════════════════════════════════════════════════════════════════════

def fig12_triple_extraction():
    fig, ax = plt.subplots(figsize=(15, 9))
    fig.patch.set_facecolor(C_WHITE)
    ax.set_xlim(-0.5, 15); ax.set_ylim(-0.5, 9)
    ax.axis("off")

    _label(ax, 7.5, 8.5, "LLM-Based Triple Extraction Pipeline",
           fontsize=15, weight="bold", color=C_DBLUE)

    # ── Step 1: Input text ──
    _rounded_box(ax, 2.5, 7.2, 4.0, 1.0,
                 "Domain Text Corpus", C_LBLUE, C_BLUE,
                 sub="Automotive electronics literature",
                 fontsize=10, sub_fontsize=8, lw=2)

    # ── Step 2: Chunking ──
    _rounded_box(ax, 2.5, 5.7, 4.0, 1.0,
                 "Text Chunking", C_LGRAY, C_MGRAY,
                 sub="Sentence-boundary splitting\n"
                     "(max LLM_MAX_CHARS per chunk)",
                 fontsize=9, sub_fontsize=7.5)
    _arrow(ax, 2.5, 6.7, 2.5, 6.2, color=C_MGRAY)

    # ── Step 3: LLM API ──
    _rounded_box(ax, 7.5, 5.7, 4.5, 1.2, "", C_LYELLOW, C_YELLOW, lw=2)
    _label(ax, 7.5, 6.05, "LLM API Call", fontsize=11, weight="bold",
           color=C_GRAY)

    # Two providers
    _rounded_box(ax, 6.3, 5.3, 1.8, 0.5, "GPT-4o-mini", C_LGREEN, C_GREEN,
                 fontsize=8, lw=1.2)
    _rounded_box(ax, 8.7, 5.3, 1.8, 0.5, "LLaMA-3\n(Ollama)", C_LPURPLE, C_PURPLE,
                 fontsize=7.5, lw=1.2)

    _label(ax, 7.5, 5.3, "OR", fontsize=8, weight="bold", color=C_MGRAY)

    _arrow(ax, 4.5, 5.7, 5.25, 5.7, color=C_MGRAY, lw=1.5)

    # Prompt template
    _label(ax, 12.5, 6.0,
           "Prompt Template:\n"
           "\"Extract factual triples from\n"
           "the following text. Format:\n"
           "Subject|Predicate|Object\n"
           "(temperature=0.0)\"",
           fontsize=7.5, color=C_GRAY, family="monospace",
           bbox_cfg=dict(boxstyle="round,pad=0.3", facecolor=C_LGRAY,
                         edgecolor=C_BORDER, lw=1))

    _arrow(ax, 11.5, 5.7, 9.75, 5.7, color=C_MGRAY, lw=1.0, ls="--")

    # ── Step 4: Parse response ──
    _rounded_box(ax, 7.5, 3.8, 4.5, 1.0,
                 "Response Parsing", C_LGRAY, C_MGRAY,
                 sub="Extract S|P|O lines from LLM output",
                 fontsize=9, sub_fontsize=7.5)
    _arrow(ax, 7.5, 5.1, 7.5, 4.3, color=C_MGRAY)

    # Example triple
    _label(ax, 12.5, 3.8,
           "Example:\n"
           "ECU | is_component_of | Vehicle\n"
           "Sensor | measures | Temperature\n"
           "CPU | is_part_of | Microcontroller",
           fontsize=7.5, color=C_BLUE, family="monospace",
           bbox_cfg=dict(boxstyle="round,pad=0.25", facecolor=C_LBLUE,
                         edgecolor=C_BLUE, lw=1))
    _arrow(ax, 11.0, 3.8, 9.75, 3.8, color=C_BLUE, lw=1.0, ls="--")

    # ── Step 5: Garbage filtering ──
    _rounded_box(ax, 7.5, 2.3, 4.5, 1.0,
                 "Garbage-Term Filtering", C_LRED, C_RED,
                 sub="Remove: market, forecast, revenue,\n"
                     "USD, billion, vendor, statistics...",
                 fontsize=9, sub_fontsize=7.5)
    _arrow(ax, 7.5, 3.3, 7.5, 2.8, color=C_MGRAY)

    # ── Step 6: Output ──
    _rounded_box(ax, 7.5, 0.7, 4.5, 0.9,
                 "corpus_triples.json", C_LGREEN, C_DGREEN,
                 sub="Array of {subject, predicate, object} dicts",
                 fontsize=10, sub_fontsize=8, lw=2)
    _arrow(ax, 7.5, 1.8, 7.5, 1.15, color=C_GREEN, lw=1.5)

    # ── Statistics note ──
    _label(ax, 2.0, 2.5,
           "Key Parameters:\n"
           "━━━━━━━━━━━━━━\n"
           "Temperature: 0.0\n"
           "Max chars/chunk: configurable\n"
           "Dedup: by (S, P, O) key\n"
           "Cost: ~$0.15/1M tokens",
           fontsize=7.5, color=C_GRAY, family="monospace",
           bbox_cfg=dict(boxstyle="round,pad=0.3", facecolor=C_LGRAY,
                         edgecolor=C_BORDER, lw=1))

    savefig(fig, "fig12_triple_extraction.png")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"Generating thesis diagrams to: {OUT_DIR}")
    print("=" * 55)

    fig01_pipeline()
    fig02_rl_loop()
    fig03_dqn()
    fig04_curriculum()
    fig05_reward()
    fig06_contributions()
    fig07_data_flow()
    fig08_ontology_alignment()
    fig09_repair_candidates()
    fig10_state_action()
    fig11_layered_arch()
    fig12_triple_extraction()

    print("=" * 55)
    print(f"All 12 diagrams saved to: {OUT_DIR}")
