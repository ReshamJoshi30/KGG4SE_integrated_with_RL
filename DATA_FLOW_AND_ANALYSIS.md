# KGG4SE-RL — Data Flow and Analysis

This document describes the end-to-end data flow through the pipeline, the file artifacts produced at each step, and the full module dependency graph.

---

## 1. End-to-End Pipeline Data Flow

```
┌──────────────────────────────────────────────────────────────────────────┐
│                       KGG4SE-RL PIPELINE DATA FLOW                       │
│                        (pipeline.py orchestrator)                        │
└──────────────────────────────────────────────────────────────────────────┘

 ┌──────────────────┐
 │  data/01_corpus/ │
 │  corpus.csv      │   (raw CSV with "title" column)
 └────────┬─────────┘
          │
          ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │  Step 1: prepare-corpus   (Scripts/prepare_corpus.py)               │
 │  ─────────────────────────────────────────────────────              │
 │  Input:  data/01_corpus/corpus.csv                                  │
 │  Output: data/01_corpus/corpus.txt                                  │
 │  Logic:  Read CSV → extract text column → write one line per entry  │
 │  Deps:   pandas                                                     │
 └────────┬────────────────────────────────────────────────────────────┘
          │
          ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │  Step 2: generate-triples  (llm_kg/generate_triples.py)             │
 │  ─────────────────────────────────────────────────────              │
 │  Input:  data/01_corpus/corpus.txt                                  │
 │  Output: outputs/intermediate/corpus_triples.json                   │
 │  Logic:  Read text lines → format LLM prompt                        │
 │          → send to OpenAI or Ollama at temperature=0.0              │
 │          → parse "S | P | O" lines from response                    │
 │          → filter garbage terms → write JSON list of triples        │
 │  Deps:   requests, OpenAI API / Ollama                              │
 └────────┬────────────────────────────────────────────────────────────┘
          │
          ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │  Step 3: clean-triples  (Scripts/clean_triples.py)                  │
 │  ─────────────────────────────────────────────────                  │
 │  Input:  outputs/intermediate/corpus_triples.json                   │
 │  Output: outputs/intermediate/corpus_triples_cleaned.json           │
 │  Logic:  normalize_entity() → lowercase, strip, underscore          │
 │          → apply alias substitutions (config.CLEAN_ALIASES)         │
 │          → split compound objects on "and"                          │
 │          → drop entities with > max_tokens tokens                   │
 │          → deduplicate by (subject, predicate, object) key          │
 └────────┬────────────────────────────────────────────────────────────┘
          │
          ▼
 ┌──────────────────────────────────────────────────────────────────────────┐
 │  Step 4: align-triples  (Scripts/align_triples.py)                       │
 │  ──────────────────────────────────────────────────                      │
 │  Input:  outputs/intermediate/corpus_triples_cleaned.json                │
 │          ontology/LLM_GENIALOntBFO_cleaned.owl                           │
 │          config/relation_map.csv   (predicate text → URI overrides)      │
 │          config/entity_map.csv     (entity text → URI overrides)         │
 │  Output: outputs/intermediate/aligned_triples.ttl                        │
 │          outputs/reports/alignment_report.csv                            │
 │                                                                          │
 │  Logic:                                                                  │
 │  1. Parse ontology → build entity index + property index                 │
 │  2. Load CSV override maps                                               │
 │  3. For each triple (s, p, o):                                           │
 │     a. Resolve predicate: CSV map → exact match → fuzzy (RapidFuzz)      │
 │     b. Resolve subject:   CSV map → fuzzy match against ontology index   │
 │     c. Resolve object:    CSV map → fuzzy match against ontology index   │
 │     d. Mint new individual URI if no match and allow_create = True       │
 │     e. Convert class URIs to individual URIs                             │
 │     f. Check domain/range constraints — skip triple if unsatisfiable     │
 │     g. Add aligned triple to RDF graph                                   │
 │  4. Serialize graph as Turtle (.ttl)                                     │
 │  5. Write alignment report CSV (one row per triple, with status)         │
 │  Deps:   rdflib, rapidfuzz                                               │
 └────────┬─────────────────────────────────────────────────────────────────┘
          │
          ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │  Step 5: build-kg  (Scripts/build_kg.py)                            │
 │  ────────────────────────────────────────                           │
 │  Input:  outputs/intermediate/aligned_triples.ttl                   │
 │          ontology/LLM_GENIALOntBFO_cleaned.owl                      │
 │  Output: outputs/intermediate/merged_kg.owl   (RDF/XML)             │
 │          outputs/intermediate/merged_kg.ttl   (Turtle)              │
 │  Logic:  Parse base ontology + parse aligned triples                │
 │          → merge (graph union) → serialize both formats             │
 │  Deps:   rdflib                                                     │
 └────────┬────────────────────────────────────────────────────────────┘
          │
          ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │  Step 6: run-reasoner  (Scripts/run_reasoner.py)                    │
 │  ─────────────────────────────────────────────                      │
 │  Input:  outputs/intermediate/merged_kg.owl                         │
 │  Output: outputs/knowledge_graph/merged_kg_reasoned.owl             │
 │          outputs/reasoning/konclude_stats_<uuid>.txt                │
 │          outputs/reasoning/konclude_raw_<uuid>.log                  │
 │  Logic:  Route to Konclude (default) or HermiT                      │
 │          → run consistency check                                    │
 │          → return unified report {is_consistent, unsat_classes,     │
 │            disjoint_violations, ...}                                │
 │          → save reasoned OWL                                        │
 │  Note:   Inconsistency is reported but does not stop the pipeline   │
 │  Deps:   reasoning/reasoner_wrapper.py, Konclude binary / Java      │
 └────────┬────────────────────────────────────────────────────────────┘
          │
          ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │  Step 7: parse-reasoner  (Scripts/parse_reasoner_output.py)         │
 │  ───────────────────────────────────────────────────────            │
 │  Input:  outputs/knowledge_graph/merged_kg_reasoned.owl             │
 │  Output: outputs/intermediate/reasoned_triples.json                 │
 │  Logic:  Parse RDF/XML with rdflib                                  │
 │          → flatten all triples to [{subject, predicate, object}]    │
 │  Deps:   rdflib                                                     │
 └────────┬────────────────────────────────────────────────────────────┘
          │
          ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │  Step 8: check-quality  (Scripts/check_quality.py)                  │
 │  ─────────────────────────────────────────────────                  │
 │  Input:  outputs/intermediate/reasoned_triples.json                 │
 │  Output: outputs/quality/quality_report.json                        │
 │  Logic:  Compute stats (unique subjects / predicates / objects)     │
 │          → find duplicate triples                                   │
 │          → top-N frequency per field                                │
 │          → produce structured JSON quality report                   │
 └────────┬────────────────────────────────────────────────────────────┘
          │
          │  (optional — triggered via `pipeline.py repair-kg`)
          ▼
 ┌──────────────────────────────────────────────────────────────────────────┐
 │  Step 9: repair-kg  (rl/ module — RL-based KG Repair)                    │
 │  ─────────────────────────────────────────────────────                   │
 │  Input:  outputs/intermediate/merged_kg.owl                              │
 │          ontology/LLM_GENIALOntBFO_cleaned.owl                           │
 │  Output: outputs/models/dqn_repair_final.pt                              │
 │          outputs/models/training_history.json                            │
 │          outputs/rl_repair_traces/repair_trace.jsonl                     │
 │                                                                          │
 │  ┌───────────────────────────────────────────────────────────────┐       │
 │  │  RepairEnv (rl/env_repair.py)                                 │       │
 │  │  reset()  → run reasoner → extract diagnostics → encode state │       │
 │  │  step()   → apply_fix   → re-run reasoner  → compute reward   │       │
 │  │  State: 18-dim vector   Actions: repair candidates list       │       │
 │  └───────────────────────────────────────────────────────────────┘       │
 │  ┌───────────────────────────────────────────────────────────────┐       │
 │  │  DQN_Agent (rl/dqn_agent.py)                                  │       │
 │  │  Policy net + target net  (rl/dqn_model.py: 256→128→64→N)     │       │
 │  │  ε-greedy exploration → Double Q-learning update              │       │
 │  │  PrioritizedReplayBuffer (rl/replay_buffer.py)                │       │
 │  └───────────────────────────────────────────────────────────────┘       │
 │  ┌───────────────────────────────────────────────────────────────┐       │
 │  │  Reasoner Integration                                         │       │
 │  │  reasoning/reasoner_wrapper.py → routes to:                   │       │
 │  │    ├── reasoning/hermit_runner.py    (owlready2 + HermiT)     │       │
 │  │    └── reasoning/konclude_runner.py  (subprocess + Konclude)  │       │
 │  │  reasoning/axiom_extractor.py → heuristic evidence extraction │       │
 │  └───────────────────────────────────────────────────────────────┘       │
 │  ┌───────────────────────────────────────────────────────────────┐       │
 │  │  Repair Execution                                             │       │
 │  │  qa/repair_candidates.py → generates repair action list       │       │
 │  │  qa/apply_fix.py         → applies atomic graph operations    │       │
 │  │  rl/diff_tracker.py      → logs before/after diffs (JSONL)    │       │
 │  │  rl/human_loop.py        → optional human-in-the-loop         │       │
 │  └───────────────────────────────────────────────────────────────┘       │
 └──────────────────────────────────────────────────────────────────────────┘
```

---

## 2. File-Level Data Flow

```
corpus.csv
    │
    ▼  [prepare_corpus.py]
corpus.txt
    │
    ▼  [generate_triples.py] ← OpenAI API / Ollama
corpus_triples.json
    │
    ▼  [clean_triples.py] ← config.CLEAN_ALIASES
corpus_triples_cleaned.json
    │
    ▼  [align_triples.py] ← ontology.owl + relation_map.csv + entity_map.csv
aligned_triples.ttl  +  alignment_report.csv
    │
    ▼  [build_kg.py] ← ontology.owl
merged_kg.owl  +  merged_kg.ttl
    │
    ▼  [run_reasoner.py] ← Konclude / HermiT
merged_kg_reasoned.owl  +  reasoning logs
    │
    ▼  [parse_reasoner_output.py]
reasoned_triples.json
    │
    ▼  [check_quality.py]
quality_report.json

    ══════════════ OPTIONAL RL REPAIR LOOP ══════════════

merged_kg.owl
    │
    ▼  [env_repair.py]
    │    ← run_reasoner()       → inconsistency report
    │    ← axiom_extractor()    → root cause evidence
    │    ← repair_candidates()  → list of actions
    │    ← apply_fix()          → updated OWL
    │    ← compute_reward()     → reward signal
    │
    ▼  [dqn_agent.py + replay_buffer.py]
    │    policy_net learns state → action mapping
    │
    ├── outputs/models/dqn_repair_ep{N}.pt    (per-episode checkpoints)
    ├── outputs/models/dqn_repair_final.pt    (final model)
    ├── outputs/models/training_history.json  (per-episode metrics)
    └── outputs/rl_repair_traces/repair_trace.jsonl
```

---

## 3. Module Dependency Graph

```
pipeline.py
├── config.py
├── core/base_step.py                        (PipelineStep ABC)
├── Scripts/prepare_corpus.py               → pandas, config
├── llm_kg/generate_triples.py              → requests, config
├── Scripts/clean_triples.py                → config
├── Scripts/align_triples.py                → rdflib, rapidfuzz, config
├── Scripts/build_kg.py                     → rdflib, config
├── Scripts/run_reasoner.py                 → reasoning/reasoner_wrapper.py
├── Scripts/parse_reasoner_output.py        → rdflib, config
├── Scripts/check_quality.py                → config
└── rl/train_repair.py
    ├── rl/env_repair.py
    │   ├── alignment/ontology_index.py     → owlready2
    │   ├── reasoning/reasoner_wrapper.py
    │   │   ├── reasoning/hermit_runner.py  → owlready2, rdflib
    │   │   ├── reasoning/konclude_runner.py → subprocess
    │   │   │   └── reasoning/parse_reasoner_stats.py
    │   │   └── reasoning/axiom_extractor.py → owlready2, rdflib
    │   ├── qa/repair_candidates.py
    │   └── qa/apply_fix.py                → rdflib
    ├── rl/dqn_agent.py                    → torch
    │   └── rl/dqn_model.py               → torch
    ├── rl/replay_buffer.py                → numpy
    ├── rl/diff_tracker.py                 → rdflib
    └── rl/human_loop.py
```

---

## 4. Intermediate File Reference

| File | Produced by | Consumed by |
|---|---|---|
| `data/01_corpus/corpus.txt` | `prepare-corpus` | `generate-triples` |
| `outputs/intermediate/corpus_triples.json` | `generate-triples` | `clean-triples` |
| `outputs/intermediate/corpus_triples_cleaned.json` | `clean-triples` | `align-triples` |
| `outputs/intermediate/aligned_triples.ttl` | `align-triples` | `build-kg` |
| `outputs/reports/alignment_report.csv` | `align-triples` | `report_quality.py` |
| `outputs/intermediate/merged_kg.owl` | `build-kg` | `run-reasoner`, `repair-kg` |
| `outputs/intermediate/merged_kg.ttl` | `build-kg` | (reference copy) |
| `outputs/knowledge_graph/merged_kg_reasoned.owl` | `run-reasoner` | `parse-reasoner` |
| `outputs/reasoning/*.log` | `run-reasoner` | (diagnostic review) |
| `outputs/intermediate/reasoned_triples.json` | `parse-reasoner` | `check-quality` |
| `outputs/quality/quality_report.json` | `check-quality` | `report_quality.py` |
| `outputs/rl_repair_steps/repaired_step_NNNN.owl` | `qa/apply_fix.py` | `reasoning/reasoner_wrapper.py` (next step) |
| `outputs/rl_repair_traces/repair_trace.jsonl` | `rl/diff_tracker.py` | `report_quality.py` |
| `outputs/models/dqn_repair_final.pt` | `rl/train_repair.py` | Deployment / evaluation |
| `outputs/models/training_history.json` | `rl/train_repair.py` | `report_quality.py` |
| `outputs/models/human_feedback.jsonl` | `rl/human_loop.py` | Re-use as scripted input |
