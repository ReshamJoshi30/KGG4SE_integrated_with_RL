# KGG4SE Framework — Full Implementation Notes

## Overview

**KGG4SE** (Knowledge Graph Generation for Software Engineering) is a thesis
framework that automatically extracts, structures, and repairs a domain-specific
OWL knowledge graph from automotive-electronics text corpora using:

1. **LLM-based triple extraction** (GPT-4o-mini or local llama3/Ollama)
2. **Ontology alignment** against a BFO-based automotive ontology
3. **Reasoner-based consistency checking** (Konclude / HermiT)
4. **Deep Q-Network (DQN) reinforcement learning** to repair OWL violations
5. **Human-in-the-loop phased training** for cold-start bootstrapping

---

## Architecture

```
corpus.txt
    │
    ▼  [Step 1] generate_triples  (LLM: GPT-4o-mini or llama3)
corpus_triples.json   (raw Subject|Predicate|Object triples)
    │
    ▼  [Step 2] clean_triples     (garbage filter, alias normalise)
corpus_triples_cleaned.json
    │
    ▼  [Step 3] align_triples     (fuzzy ontology alignment)
sample_corpus_aligned_triples.ttl  +  alignment_report.csv
    │
    ▼  [Step 4] build_kg          (merge with base ontology)
merged_kg.owl  ← this is the KG to repair (contains injected violations)
    │
    ▼  [Step 5] run_reasoner      (Konclude/HermiT consistency check)
inconsistency report: {unsat_classes, disjoint_violations, ...}
    │
    ▼  [Step 6] RL Repair         (DQN agent — RepairEnv)
repaired_step_N.owl  (OWL after each repair action)
    │
    ▼  [Step 7] report_quality    (plots + text report)
outputs/reports/*.png
```

---

## Quick-Start Commands

```cmd
:: Step 1: Set OpenAI key (Windows CMD)
set OPENAI_API_KEY=sk-...

:: Step 2: Generate triples
python -m pipeline generate_triples

:: Step 3: Run RL training (fully automated — no prompts)
python -m rl.train_repair --no-human

:: Step 4: Run RL training with custom parameters
python -m rl.train_repair --no-human --episodes 50 --max-steps 15

:: Step 5: Generate all reports and plots
python report_quality.py

:: Step 6: Quick integration test (1 episode)
python -m rl.train_repair --test
```

---

## Module Descriptions

### `config.py`
Central configuration singleton. All paths, LLM settings, and RL hyperparameters
live here. Import anywhere — all modules rely on this.

Key settings:
| Setting | Value | Notes |
|---|---|---|
| `USE_OPENAI` | `True` | Route triple extraction through OpenAI API |
| `OPENAI_MODEL` | `"gpt-4o-mini"` | Best value for thesis (cheap + accurate) |
| `DEFAULT_REASONER` | `"konclude"` | Faster than HermiT for OWL 2 EL |
| `RL_EPISODES` | `50` | Increased from 20 — DQN needs 50–100+ |
| `RL_MAX_STEPS_PER_EPISODE` | `15` | Reduced from 20 for faster episodes |
| `RL_EPSILON_DECAY` | `0.994` | Per-step (was per-episode — see Bug Fix #1) |
| `RL_HUMAN_SUPERVISED_EPISODES` | `3` | Phase 1 count |
| `RL_HUMAN_FIRST_EPISODES` | `5` | Phase 2 count |

### `llm_kg/generate_triples.py`
Extracts `Subject | Predicate | Object` triples from corpus text.

- **`_query_openai()`** — calls OpenAI Chat Completions API at temperature=0.0
- **`_query_ollama()`** — calls local Ollama instance (llama3)
- Routing: `if config.USE_OPENAI: _query_openai() else _query_ollama()`
- Temperature=0.0 ensures deterministic, reproducible extraction

### `rl/env_repair.py` — `RepairEnv`
OpenAI Gym-compatible environment wrapping the OWL repair process.

**State vector (18 dimensions):**
| Index | Feature | Description |
|---|---|---|
| 0–9 | Error type one-hot | 10 violation categories |
| 10 | Num actions | Number of available repair actions (normalised 0–1) |
| 11 | Low-risk actions | Count of low-risk actions (normalised) |
| 12 | Has evidence | Binary: does the error have supporting evidence? |
| 13 | Step progress | `step / max_steps` |
| 14 | Has entity IRI | Binary: is a concrete IRI involved? |
| **15** | **Queue progress** | `remaining_violations / initial_violations` |
| **16** | **Last step improved** | Binary momentum signal (1 if previous step helped) |
| **17** | **Min action risk** | Safest available action risk score (normalised /5) |

Features 15–17 were added to give the agent progress awareness, momentum, and
risk sensitivity — preventing random-walk behaviour late in episodes.

**Key methods:**
- `reset()` — loads fresh copy of `base_owl`, runs reasoner, builds error queue
- `step(action_idx)` — applies fix, re-runs reasoner, returns (state, reward, done, info)
- `get_current_action_count()` — number of actions for current error
- `state_dim()` — returns 18

### `rl/dqn_agent.py` — `DQN_Agent`
Standard Double-DQN with experience replay.

- Input: 18-dim state vector
- Output: Q-values for up to `output_dim` actions
- Epsilon-greedy exploration with configurable decay
- Target network updated every `RL_TARGET_UPDATE_INTERVAL` episodes
- `decay_epsilon()` called per-step in the training loop

### `rl/train_repair.py` — Training Loop
Main entry point for RL training. Three-phase curriculum:

**Phase 1 — Human Supervised** (`RL_HUMAN_SUPERVISED_EPISODES` = 3 episodes):
Human selects every action. RL observes and stores all transitions (+ replay boost).

**Phase 2 — Human First-Step** (`RL_HUMAN_FIRST_EPISODES` = 5 episodes):
Human selects only the first action per episode. RL handles all subsequent steps.

**Phase 3 — RL Automated** (remaining 42 episodes):
Fully autonomous. Human only asked when Q-value confidence margin < threshold.

**CLI flags:**
```
--no-human          Disable ALL human input (scripted or automated)
--scripted FILE     Load action indices from JSON for unattended runs
--episodes N        Override RL_EPISODES
--max-steps N       Override RL_MAX_STEPS_PER_EPISODE
--reasoner hermit   Switch reasoner (default: konclude)
--owl PATH          Override base OWL path
--test              Run one test episode instead of full training
```

**Training history saved** to `outputs/models/training_history.json`:
```json
{
  "config": { "episodes": 50, "max_steps": 15, ... },
  "per_episode": [
    {
      "episode": 1, "phase": "human_supervised",
      "steps": 8, "reward": 12.5, "consistent": true,
      "epsilon": 0.94, "avg_loss": 0.023,
      "success_rate_so_far": 1.0,
      "start_unsat": 2, "start_disj": 1,
      "end_unsat": 0, "end_disj": 0
    }, ...
  ]
}
```

### `rl/human_loop.py` — `HumanLoop`
Manages human-in-the-loop feedback during training.

- `scripted_actions_file` kwarg: pre-load action indices from JSON file
- `interactive_mode = False` (set by `--no-human`) skips all `input()` calls
- Scripted queue pops entries in order; `null` entries defer to RL

### `rl/reward_functions.py` — Reward Function
```
+10.0   KG becomes fully consistent (major milestone)
+2.0    per disjoint violation fixed
+2.0    per unsatisfiable class resolved
+1.0    per transitive disjoint resolved
-1.0    per violation introduced (regression penalty)
-0.5    neutral step (no-op penalty — added in this session)
```
The **neutral-step penalty** is critical: without it, the agent learns to
take random no-op actions without penalty, wasting repair budget.

### `rl/diff_tracker.py` — `DiffTracker`
Logs step-by-step OWL diffs to `outputs/rl_repair_traces/repair_trace.jsonl`.
Each JSONL line records: step_id, action, reward, metrics, triple count delta.

### `reasoning/reasoner_wrapper.py`
Wraps both HermiT (Java JAR) and Konclude (native binary) reasoners.
Returns a unified dict:
```python
{
    "is_consistent": bool,
    "unsat_classes": [...],
    "disjoint_violations": [...],
    "transitive_disjoint_violations": [...],
    "all_issues": { "by_type": {...}, "total_violations": N }
}
```

### `report_quality.py` — Quality Report Generator
Generates all plots and the text report after training. Run with:
```
python report_quality.py
python report_quality.py --before outputs/intermediate/merged_kg.owl \
                         --after  outputs/rl_repair_steps/repaired_step_9.owl \
                         --history outputs/models/training_history.json \
                         --raw-triples outputs/intermediate/corpus_triples.json \
                         --cleaned-triples outputs/intermediate/corpus_triples_cleaned.json \
                         --alignment-report outputs/reports/alignment_report.csv
```

---

## All Implemented Improvements

### Bug Fix #1 — Epsilon Decay (Critical)
**Problem:** `agent.decay_epsilon()` was called once per episode. After 50 episodes
at decay=0.995: epsilon ≈ 0.78 (agent 78% random). After 20 episodes: 0.90.
The agent was nearly random for the entire training run — this was the primary
cause of ~55% accuracy.

**Fix:** Moved `agent.decay_epsilon()` inside the step loop (per-step decay).
Changed decay rate to `RL_EPSILON_DECAY = 0.994` per step.

With 50 episodes × 10 avg steps = ~500 steps:
epsilon at end = 0.994^500 ≈ 0.05 (hits minimum — agent fully exploits learned policy).

### Bug Fix #2 — `--no-human` Still Prompting
**Problem:** Even with `--no-human` flag and phase counts set to 0, Phase 3
called `should_ask_human()` which returned True when Q-value margin < 0.65
(always early in training — agent never confident).

**Fix:** Added `no_human` kwarg to `train_repair()`. When True, skips
`human_loop.enable_interactive()` entirely, so `interactive_mode` stays False
and `should_ask_human()` always returns False.

### Improvement #1 — OpenAI API Integration
- Added `_query_openai()` in `llm_kg/generate_triples.py`
- `USE_OPENAI = True` in `config.py` routes to OpenAI instead of Ollama
- Temperature = 0.0 for deterministic extraction
- GPT-4o-mini follows the structured prompt far more strictly than local llama3:
  fewer garbage triples-> fewer ontology violations-> smaller repair task for RL

### Improvement #2 — Training Configuration
| Parameter | Old | New | Reason |
|---|---|---|---|
| `RL_EPISODES` | 20 | 50 | DQN needs 50–100+ to converge |
| `RL_MAX_STEPS_PER_EPISODE` | 20 | 15 | Faster episodes, more variety |
| `RL_HUMAN_SUPERVISED_EPISODES` | 5 | 3 | More episodes for RL |
| `RL_HUMAN_FIRST_EPISODES` | 10 | 5 | Only 8 human eps of 50 total |

### Improvement #3 — Expanded RL State (15-> 18 dims)
Three new features improve agent decision-making:
- **Queue progress [15]**: Tells agent how much work remains
- **Last step improved [16]**: Momentum — encourages continuing good actions
- **Min action risk [17]**: Agent can prefer low-risk actions under uncertainty

### Improvement #4 — Neutral-Step Penalty (-0.5)
Without this, the agent accumulates 0.0 reward for no-op actions and does not
learn to avoid them. The -0.5 penalty makes wasted steps costly.

### Improvement #5 — Training History Logging
Saves `outputs/models/training_history.json` after each run with per-episode:
reward, consistency flag, epsilon, avg DQN loss, cumulative accuracy,
start/end unsat counts, start/end disjoint violation counts.

### Improvement #6 — Scripted Actions for Unattended Runs
`outputs/models/scripted_actions.json` contains 110 zero-indices (default
"choose first action" for all human-phase prompts). Created with:
```
python -m rl.train_repair --scripted outputs/models/scripted_actions.json
```

---

## Generated Plots (14 total)

### KG Extraction Quality
| File | Description |
|---|---|
| `p5_extraction_alignment_quality.png` | Precision/Recall/F1 at Raw→Cleaned→Aligned stages, triple count funnel, alignment status pie |

### RL Repair Effectiveness
| File | Description |
|---|---|
| `p1_reward_vs_episode.png` | Total reward per episode with rolling average |
| `p2_unsat_vs_episode.png` | Unsatisfiable class count at episode end |
| `p3_disj_vs_episode.png` | Disjoint violation count at episode end |
| `p4_errors_before_after.png` | Error counts before vs after RL with % reduction |
| `p6_pipeline_quality_line.png` | Violations across Before RL-> Ep1-> Final checkpoints |
| `p7_repair_efficiency.png` | Steps per episode, error reduction rate, triple delta |

### Training Diagnostics
| File | Description |
|---|---|
| `training_accuracy_reward.png` | Cumulative accuracy % + per-episode reward |
| `training_loss_epsilon.png` | DQN loss curve + epsilon decay |
| `accuracy_by_phase.png` | Success rate broken down by Phase 1/2/3 |

### Repair Summary
| File | Description |
|---|---|
| `violation_comparison.png` | Violation type counts before vs after |
| `repair_steps.png` | Reward + violation delta + triple count per repair step |
| `action_distribution.png` | Pie chart of action types used |
| `kg_stats_comparison.png` | KG triples/entities before vs after |
| `before_after_summary.png` | Dashboard: violations + accuracy trend + consistency status |
| `precision_recall_f1.png` | Repair P/R/F1 + TP/FP/FN breakdown |

---

## Precision / Recall / F1 Definitions

### Pipeline-Level (Extraction Quality)
| Stage | Precision | Recall |
|---|---|---|
| Raw LLM | 1.0 (baseline) | 1.0 (baseline) |
| Cleaned | kept/raw | kept/raw |
| Aligned | ok-aligned/all-aligned | ok-aligned/raw |

### Repair-Level (RL Performance)
| Metric | Formula | Meaning |
|---|---|---|
| **TP** | Steps where unsat/disjoint count decreased | Repairs that helped |
| **FP** | Steps where unsat/disjoint count increased | Repairs that hurt |
| **FN** | Violations remaining after all repair | What the agent missed |
| **Precision** | TP / (TP + FP) | Of impactful steps, how many helped? |
| **Recall** | TP / (TP + FN) | Of all violations, how many were fixed? |
| **F1** | 2·P·R / (P + R) | Harmonic mean |

---

## Reasoner Details

### Konclude (default)
- Location: `reasoning/Konclude/`
- OWL 2 EL reasoner — much faster than HermiT for large ontologies
- Native binary (no JVM overhead)
- Used via `reasoning/reasoner_wrapper.py`

### HermiT
- Full OWL 2 DL reasoner — more thorough, supports OWL Full features
- Slower, requires Java
- Use with `--reasoner hermit` for thorough final evaluation

---

## Results Summary (after all improvements)

With GPT-4o-mini extraction + fixed epsilon decay + 50 episodes:

| Metric | Before Improvements | After |
|---|---|---|
| Epsilon at end of training | ~0.90 (90% random) | ~0.05 (95% exploit) |
| Training episodes | 20 | 50 |
| Human episodes | 15/20 (75%) | 8/50 (16%) |
| RL episodes | 5/20 (25%) | 42/50 (84%) |
| Extraction triple quality | llama3 noisy triples | GPT-4o-mini clean triples |
| Neutral step penalty | None | −0.5 per no-op |
| State dimensions | 15 | 18 |

---

## File Index

| File | Role |
|---|---|
| `config.py` | Central config (all paths, settings, hyperparameters) |
| `pipeline.py` | Orchestrates all pipeline steps |
| `report_quality.py` | Post-training quality report + 14 plots |
| `test_repair.py` | Quick 1-episode integration test |
| `llm_kg/generate_triples.py` | LLM triple extraction (OpenAI / Ollama) |
| `llm_kg/clean_triples.py` | Garbage filter + alias normalisation |
| `llm_kg/align_triples.py` | Fuzzy ontology alignment-> aligned TTL |
| `llm_kg/build_kg.py` | Merge aligned triples with base ontology |
| `reasoning/reasoner_wrapper.py` | Unified Konclude / HermiT interface |
| `rl/env_repair.py` | RepairEnv (Gym environment for OWL repair) |
| `rl/dqn_agent.py` | DQN agent (Double-DQN + experience replay) |
| `rl/train_repair.py` | 3-phase training loop + argparse CLI |
| `rl/human_loop.py` | Human-in-the-loop feedback (scripted/interactive) |
| `rl/reward_functions.py` | Reward function (violation-based + no-op penalty) |
| `rl/apply_fix.py` | Applies repair actions to OWL graph |
| `rl/diff_tracker.py` | JSONL trace logger for repair steps |
| `outputs/models/training_history.json` | Per-episode training log |
| `outputs/models/scripted_actions.json` | Pre-scripted action queue for automation |
| `outputs/reports/*.png` | Generated plots (14 files) |
| `IMPLEMENTATION_NOTES.md` | This file |
