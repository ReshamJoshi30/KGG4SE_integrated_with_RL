# KGG4SE-RL — Implementation Notes

## Overview

**KGG4SE-RL** (Knowledge Graph Generation for Software Engineering with Reinforcement Learning) is a thesis framework that automatically builds and repairs a domain-specific OWL Knowledge Graph from automotive-electronics text corpora using:

1. **LLM-based triple extraction** — GPT-4o-mini or local Llama 3 (via Ollama)
2. **Ontology alignment** — fuzzy matching against a BFO-based automotive ontology
3. **Reasoner-based consistency checking** — Konclude (default) or HermiT
4. **Deep Q-Network (DQN) reinforcement learning** — repairs OWL violations automatically
5. **Human-in-the-loop phased training** — bootstraps the agent before handing control over

---

## Architecture Overview

```
corpus.csv
    │
    ▼  Step 1 — prepare-corpus
corpus.txt
    │
    ▼  Step 2 — generate-triples   (LLM: GPT-4o-mini or Llama 3)
corpus_triples.json
    │
    ▼  Step 3 — clean-triples
corpus_triples_cleaned.json
    │
    ▼  Step 4 — align-triples      (fuzzy ontology alignment)
aligned_triples.ttl  +  alignment_report.csv
    │
    ▼  Step 5 — build-kg           (merge with base ontology)
merged_kg.owl
    │
    ▼  Step 6 — run-reasoner       (Konclude / HermiT)
inconsistency report: {unsat_classes, disjoint_violations, ...}
    │
    ▼  Step 7 — parse-reasoner
reasoned_triples.json
    │
    ▼  Step 8 — check-quality
quality_report.json
    │
    ▼  Step 9 — repair-kg          (DQN RL agent)
dqn_repair_final.pt  +  repair_trace.jsonl
```

---

## Module Descriptions

### `config.py`

Central configuration singleton. All paths, LLM settings, and RL hyperparameters live here. Every module imports this — do not hardcode paths elsewhere.

Key settings:

| Setting | Default | Description |
|---|---|---|
| `USE_OPENAI` | `True` | Route LLM calls to OpenAI API |
| `OPENAI_MODEL` | `"gpt-4o-mini"` | OpenAI model for triple extraction |
| `MODEL_NAME` | `"llama3"` | Ollama model (used when `USE_OPENAI = False`) |
| `DEFAULT_REASONER` | `"konclude"` | OWL reasoner (faster than HermiT for large ontologies) |
| `FUZZY_CUTOFF` | `85` | Minimum fuzzy match score for alignment (0–100) |
| `RL_EPISODES` | `50` | Total DQN training episodes |
| `RL_MAX_STEPS_PER_EPISODE` | `15` | Max repair actions allowed per episode |
| `RL_EPSILON_DECAY` | `0.994` | Exploration decay rate per step |
| `RL_HUMAN_SUPERVISED_EPISODES` | `3` | Phase 1 episode count |
| `RL_HUMAN_FIRST_EPISODES` | `5` | Phase 2 episode count |

---

### `llm_kg/generate_triples.py`

Extracts `Subject | Predicate | Object` triples from plain text using an LLM.

- **`_query_openai()`** — calls OpenAI Chat Completions API at `temperature=0.0`
- **`_query_ollama()`** — calls local Ollama instance
- Routing: `if config.USE_OPENAI: _query_openai() else _query_ollama()`
- Temperature is fixed at 0.0 for deterministic, reproducible extraction
- Garbage terms are filtered; a fallback triple is emitted if the LLM returns nothing

---

### `Scripts/align_triples.py` — `AlignTriplesStep`

Maps each triple's subject, predicate, and object to ontology URIs.

**Alignment logic per triple field:**
1. Check manual CSV override (`config/relation_map.csv` or `config/entity_map.csv`)
2. Try exact string match against the ontology index
3. Try fuzzy match via RapidFuzz (threshold: `FUZZY_CUTOFF`)
4. If `ALLOW_CREATE_INDIVIDUALS = True`, mint a new individual URI for unmatched entities

Class URIs are converted to individual URIs automatically. Domain and range constraints from the ontology are checked; triples that cannot satisfy them are skipped.

---

### `reasoning/reasoner_wrapper.py`

Unified interface to both OWL reasoners. Both return the same dict structure:

```python
{
    "is_consistent":              bool,
    "unsat_classes":              list[str],
    "disjoint_violations":        list[tuple],
    "transitive_disjoint_violations": list[tuple],
    "all_issues": {
        "by_type":         dict,
        "total_violations": int
    }
}
```

**Konclude** (default) — native binary, OWL 2 EL, no JVM required, faster.
**HermiT** — full OWL 2 DL, more thorough, requires Java.

---

### `reasoning/axiom_extractor.py`

Heuristic evidence extractor. Given an inconsistent OWL file, it searches the RDF graph to identify the most likely root cause:

- Disjoint class violations
- Domain violations (individual typed to wrong class for a property)
- Range violations (literal or entity of wrong type used as object)
- Class-used-as-individual errors
- Unsatisfiable class membership

Returns a structured evidence dict consumed by `RepairEnv` to build the state vector.

---

### `qa/repair_candidates.py`

Takes the reasoner report and axiom evidence and generates a list of candidate repair actions. Each candidate includes:

- `action_type` — e.g. `remap_entity`, `drop_entity`, `add_type_assertion`
- `target` — the IRI of the entity to act on
- `detail` — the specific change to make
- `risk` — estimated risk score (0–5)

Supports 10+ action types covering all major OWL 2 violation categories.

---

### `qa/apply_fix.py`

Applies a single repair action to the RDF graph. Each action type modifies the in-memory graph and serializes the result to a new OWL file in `outputs/rl_repair_steps/`.

---

### `rl/env_repair.py` — `RepairEnv`

OpenAI Gym-compatible environment wrapping the full repair loop.

**State vector — 18 dimensions:**

| Index | Feature | Description |
|---|---|---|
| 0–9 | Error type encoding | One per violation category (10 types) |
| 10 | Num actions | Available repair actions, normalised 0–1 |
| 11 | Low-risk actions | Count of low-risk actions, normalised |
| 12 | Has evidence | Binary: axiom extractor found concrete evidence |
| 13 | Step progress | `current_step / max_steps` |
| 14 | Has entity IRI | Binary: a concrete IRI is involved in the violation |
| 15 | Queue progress | `remaining_violations / initial_violations` |
| 16 | Last step improved | Binary momentum: 1 if previous action reduced violations |
| 17 | Min action risk | Safest available action risk score, normalised (÷5) |

**Key methods:**
- `reset()` — loads a fresh copy of `base_owl`, runs the reasoner, builds the error queue, returns initial state
- `step(action_idx)` — applies the selected repair action, re-runs the reasoner, computes reward, returns `(state, reward, done, info)`
- `state_dim()` — returns 18
- `action_space_n()` — returns the max number of repair actions

---

### `rl/dqn_agent.py` — `DQN_Agent`

Double DQN with prioritized experience replay and action masking.

- **Policy network** and **target network** — both are instances of `dqn_model.py` (architecture: `input_dim → 256 → 128 → 64 → output_dim`)
- **Action masking** — Q-values for actions beyond `num_valid_actions` are masked to −∞ so the agent never selects invalid actions
- **Soft target updates** — Polyak averaging (τ = 0.005) every step instead of hard copy every N episodes
- **ε-greedy exploration** — epsilon decays per step from 1.0 down to `epsilon_min = 0.05`

---

### `rl/replay_buffer.py` — `PrioritizedReplayBuffer`

Prioritized Experience Replay (PER). Transitions with higher TD-error are sampled more often, so the agent learns more from surprising or impactful steps.

- `alpha = 0.6` — controls how much priority is used (0 = uniform)
- `beta` anneals from 0.4 → 1.0 over the full training run (importance sampling correction)
- Expert transitions (human actions) are pushed with elevated priority via `push_expert()`

---

### `rl/train_repair.py` — Training Loop

Main entry point for RL training. Implements the three-phase curriculum.

**Phase 1 — Human Supervised** (episodes 1 to `RL_HUMAN_SUPERVISED_EPISODES`):
Human selects every repair action. The agent observes all transitions and stores them in the replay buffer with elevated priority.

**Phase 2 — Human First-Step** (next `RL_HUMAN_FIRST_EPISODES` episodes):
Human selects only the first action per episode. The agent handles all subsequent steps autonomously.

**Phase 3 — RL Automated** (remaining episodes):
Fully autonomous. Human is only prompted when the agent's Q-value confidence margin falls below the threshold.

**CLI flags for `rl.train_repair`:**

| Flag | Description |
|---|---|
| `--episodes N` | Override `RL_EPISODES` |
| `--max-steps N` | Override `RL_MAX_STEPS_PER_EPISODE` |
| `--no-human` | Disable all human input (Phase 3 only) |
| `--scripted FILE` | Load pre-recorded human action indices from JSON |
| `--reasoner hermit` | Switch to HermiT (default: konclude) |
| `--owl PATH` | Override the input OWL file path |
| `--test` | Run a single test episode |

**Training history** is saved after each run to `outputs/models/training_history.json`:

```json
{
  "config": { "episodes": 50, "max_steps": 15, "reasoner": "konclude" },
  "per_episode": [
    {
      "episode": 1,
      "phase": "human_supervised",
      "steps": 8,
      "reward": 12.5,
      "consistent": true,
      "epsilon": 0.94,
      "avg_loss": 0.023,
      "success_rate_so_far": 1.0,
      "start_unsat": 2,
      "start_disj": 1,
      "end_unsat": 0,
      "end_disj": 0
    }
  ]
}
```

---

### `rl/reward_functions.py` — Reward Function

| Event | Reward |
|---|---|
| KG becomes fully consistent | +10.0 |
| Each disjoint violation fixed | +2.0 |
| Each unsatisfiable class resolved | +2.0 |
| Each transitive disjoint resolved | +1.0 |
| Each violation introduced (regression) | −1.0 |
| Neutral step (no change to violations) | −0.5 |
| Each step taken (efficiency penalty) | −0.1 |

---

### `rl/human_loop.py` — `HumanLoop`

Manages human interaction during training.

- **Interactive mode** — prompts the user in the terminal to select an action index
- **Scripted mode** — loads action indices from a JSON file and pops them in order; `null` entries defer to the RL agent
- **Disabled** — when `--no-human` is passed, `interactive_mode = False` and the agent always acts autonomously

Human decisions are logged to `outputs/models/human_feedback.jsonl` and can be replayed as scripted actions in future runs.

---

### `rl/diff_tracker.py` — `DiffTracker`

Logs step-by-step OWL graph changes to `outputs/rl_repair_traces/repair_trace.jsonl`. Each line records:

```json
{
  "step_id": "0019",
  "episode": 3,
  "action_type": "remap_entity",
  "reward": 2.0,
  "metrics": { "unsat": 1, "disjoint": 0 },
  "triple_count_delta": -3
}
```

---

### `report_quality.py` — Quality Report Generator

Generates plots and a quality report after training. Run standalone:

```bash
python report_quality.py

# With custom paths
python report_quality.py \
  --before          outputs/intermediate/merged_kg.owl \
  --after           outputs/rl_repair_steps/repaired_step_final.owl \
  --history         outputs/models/training_history.json \
  --raw-triples     outputs/intermediate/corpus_triples.json \
  --cleaned-triples outputs/intermediate/corpus_triples_cleaned.json \
  --alignment-report outputs/reports/alignment_report.csv
```

---

## Generated Plots

### Extraction Quality
| Plot | Description |
|---|---|
| `p5_extraction_alignment_quality.png` | Precision/Recall/F1 at raw → cleaned → aligned stages, triple count funnel, alignment status breakdown |

### RL Repair Effectiveness
| Plot | Description |
|---|---|
| `p1_reward_vs_episode.png` | Total reward per episode with rolling average |
| `p2_unsat_vs_episode.png` | Unsatisfiable class count at end of each episode |
| `p3_disj_vs_episode.png` | Disjoint violation count at end of each episode |
| `p4_errors_before_after.png` | Error counts before vs. after RL with % reduction |
| `p6_pipeline_quality_line.png` | Violations across Before RL → Episode 1 → Final |
| `p7_repair_efficiency.png` | Steps per episode, error reduction rate, triple delta |

### Training Diagnostics
| Plot | Description |
|---|---|
| `training_accuracy_reward.png` | Cumulative accuracy % + per-episode reward |
| `training_loss_epsilon.png` | DQN loss curve + epsilon decay over steps |
| `accuracy_by_phase.png` | Success rate broken down by Phase 1 / 2 / 3 |

### Repair Summary
| Plot | Description |
|---|---|
| `violation_comparison.png` | Violation type counts before vs. after repair |
| `repair_steps.png` | Reward + violation delta + triple count per repair step |
| `action_distribution.png` | Pie chart of action types used across training |
| `kg_stats_comparison.png` | KG triple and entity counts before vs. after |
| `before_after_summary.png` | Dashboard: violations + accuracy trend + consistency |
| `precision_recall_f1.png` | Repair precision, recall, F1 + TP/FP/FN breakdown |

---

## Precision / Recall / F1 Definitions

### Extraction Quality (Pipeline-level)

| Stage | Precision | Recall |
|---|---|---|
| Raw LLM output | 1.0 (baseline) | 1.0 (baseline) |
| After cleaning | kept / raw | kept / raw |
| After alignment | ok-aligned / all-aligned | ok-aligned / raw |

### Repair Quality (RL-level)

| Metric | Formula | Meaning |
|---|---|---|
| **TP** | Steps where violation count decreased | Repairs that helped |
| **FP** | Steps where violation count increased | Repairs that hurt |
| **FN** | Violations remaining after all repair steps | What the agent missed |
| **Precision** | TP / (TP + FP) | Of all impactful steps, how many helped? |
| **Recall** | TP / (TP + FN) | Of all violations present, how many were fixed? |
| **F1** | 2 · P · R / (P + R) | Harmonic mean |

---

## File Index

| File | Role |
|---|---|
| `pipeline.py` | CLI entry point for all 9 pipeline commands |
| `config.py` | Central configuration (all paths, settings, hyperparameters) |
| `report_quality.py` | Post-training quality report and plots |
| `inject_violations.py` | Injects test violations to evaluate RL repair |
| `generate_thesis_diagrams.py` | Generates architecture diagrams to `outputs/diagrams/` |
| `llm_kg/generate_triples.py` | LLM triple extraction (OpenAI / Ollama) |
| `Scripts/clean_triples.py` | Normalisation, deduplication, token filter |
| `Scripts/align_triples.py` | Fuzzy ontology alignment → aligned TTL |
| `Scripts/build_kg.py` | Merge aligned triples with base ontology |
| `Scripts/run_reasoner.py` | Reasoner step wrapper |
| `reasoning/reasoner_wrapper.py` | Unified Konclude / HermiT interface |
| `reasoning/axiom_extractor.py` | Heuristic inconsistency root-cause finder |
| `qa/repair_candidates.py` | Generates repair action candidates from reasoner report |
| `qa/apply_fix.py` | Applies atomic repair actions to the OWL graph |
| `rl/env_repair.py` | RepairEnv — Gym-compatible RL environment |
| `rl/dqn_agent.py` | Double DQN agent with action masking |
| `rl/dqn_model.py` | PyTorch neural network (256 → 128 → 64 → N) |
| `rl/replay_buffer.py` | Prioritized Experience Replay buffer |
| `rl/train_repair.py` | 3-phase curriculum training loop + CLI |
| `rl/human_loop.py` | Human-in-the-loop interface (interactive / scripted) |
| `rl/reward_functions.py` | Reward shaping (violation-based + efficiency penalty) |
| `rl/diff_tracker.py` | JSONL trace logger for repair steps |
| `rl/triple_display.py` | Terminal UI for violations and action choices |
| `outputs/models/training_history.json` | Per-episode training log |
| `outputs/rl_repair_traces/repair_trace.jsonl` | Step-level repair trace |
