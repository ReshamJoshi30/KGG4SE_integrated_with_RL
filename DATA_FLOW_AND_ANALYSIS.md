# Project Analysis: Data Flow

## 1. Data Flow Graph

```
┌───────────────────────────────────────────────────────────────────────────┐
│                        KG PIPELINE DATA FLOW                              │
│                        (pipeline.py orchestrator)                         │
└───────────────────────────────────────────────────────────────────────────┘

 ┌──────────────────┐
 │  data/01_corpus/ │
 │  corpus.csv      │  (raw CSV with "title" column)
 └────────┬─────────┘
          │
          ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │  Step 1: prepare-corpus  (Scripts/prepare_corpus.py)                │
 │  PrepareCorpusStep                                                  │
 │  ─────────────────────                                              │
 │  Input:  data/01_corpus/corpus.csv                                  │
 │  Output: data/01_corpus/corpus.txt                                  │
 │  Logic:  Read CSV-> extract "title" column-> write one line/entry   │
 │  Deps:   pandas                                                     │
 └────────┬────────────────────────────────────────────────────────────┘
          │
          ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │  Step 2: generate-triples  (llm_kg/generate_triples.py)             │
 │  GenerateTriplesStep                                                │
 │  ────────────────────                                               │
 │  Input:  data/01_corpus/corpus.txt                                  │
 │  Output: outputs/intermediate/corpus_triples.json                   │
 │  Logic:  Read text-> crop to max_chars-> format prompt              │
 │         -> send to Ollama LLM-> parse "S | P | O" lines             │
 │         -> filter garbage terms-> fallback triple if empty          │
 │  Deps:   requests, Ollama service at localhost:11434                │
 └────────┬────────────────────────────────────────────────────────────┘
          │
          ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │  Step 3: clean-triples  (Scripts/clean_triples.py)                  │
 │  CleanTriplesStep                                                   │
 │  ─────────────────                                                  │
 │  Input:  outputs/intermediate/corpus_triples.json                   │
 │  Output: outputs/intermediate/corpus_triples_cleaned.json           │
 │  Logic:  normalize_entity() (lowercase, strip, underscore)          │
 │         -> apply alias substitutions (config.CLEAN_ALIASES)         │
 │         -> split compound objects on "and"                          │
 │         -> drop clause-like entities (>5 tokens)                    │
 │         -> deduplicate by (subj, pred, obj) key                     │
 └────────┬────────────────────────────────────────────────────────────┘
          │
          ▼
 ┌─────────────────────────────────────────────────────────────────────────┐
 │  Step 4: align-triples  (Scripts/align_triples.py)                      │
 │  AlignTriplesStep                                                       │
 │  ─────────────────                                                      │
 │  Input:  outputs/intermediate/corpus_triples_cleaned.json               │
 │          ontology/LLM_GENIALOntBFO_cleaned.owl (OWL ontology)           │
 │          config/relation_map.csv  (predicate text-> URI mapping)        │
 │          config/entity_map.csv    (entity text-> URI mapping)           │
 │  Output: outputs/intermediate/sample_corpus_aligned_triples.ttl         │
 │          outputs/reports/alignment_report.csv                           │
 │                                                                         │
 │  Logic:                                                                 │
 │  1. Parse ontology-> build entity index + property index                │
 │  2. Load CSV maps (relation_map, entity_map)                            │
 │  3. For each triple (s, p, o):                                          │
 │     a. Resolve predicate: CSV map-> exact match-> fuzzy (RapidFuzz)     │
 │     b. Resolve subject:   CSV map (with safety check)-> fuzzy match     │
 │     c. Resolve object:    CSV map (with safety check)-> fuzzy match     │
 │     d. Mint individuals for unmatched entities (if allow_create=True)   │
 │     e. Convert class URIs to individual URIs (ensure_individual)        │
 │     f. Check external namespace ancestors (FIX9: skip conflicts)        │
 │     g. Satisfy domain/range constraints (auto-add types or skip)        │
 │     h. Add triple to aligned RDF graph                                  │
 │  4. Serialize graph as Turtle (.ttl)                                    │
 │  5. Write alignment report CSV                                          │
 │  Deps:   rdflib, rapidfuzz (optional)                                   │
 └────────┬────────────────────────────────────────────────────────────────┘
          │
          ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │  Step 5: build-kg  (Scripts/build_kg.py)                            │
 │  BuildKGStep                                                        │
 │  ────────────                                                       │
 │  Input:  outputs/intermediate/sample_corpus_aligned_triples.ttl     │
 │          ontology/LLM_GENIALOntBFO_cleaned.owl                      │
 │  Output: outputs/intermediate/merged_kg.owl  (RDF/XML)              │
 │          outputs/intermediate/merged_kg.ttl  (Turtle)               │
 │  Logic:  Parse base ontology + parse aligned triples                │
 │         -> merge (graph union)-> serialize both formats             │
 │  Deps:   rdflib                                                     │
 └────────┬────────────────────────────────────────────────────────────┘
          │
          ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │  Step 6: run-reasoner  (Scripts/run_reasoner.py)                    │
 │  RunReasonerStep                                                    │
 │  ────────────────                                                   │
 │  Input:  outputs/intermediate/merged_kg.owl                         │
 │  Output: outputs/knowledge_graph/merged_kg_reasoned.owl             │
 │  Logic:  Load OWL via owlready2-> run HermiT (sync_reasoner)        │
 │         -> check for ontology collapse (EquivalentClasses > 50)     │
 │         -> save reasoned ontology                                   │
 │  Note:   Inconsistency is a WARNING, not a hard failure             │
 │  Deps:   owlready2, Java (HermiT JVM)                               │
 └────────┬────────────────────────────────────────────────────────────┘
          │
          ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │  Step 7: parse-reasoner  (Scripts/parse_reasoner_output.py)         │
 │  ParseReasonerOutputStep                                            │
 │  ────────────────────────                                           │
 │  Input:  outputs/knowledge_graph/merged_kg_reasoned.owl             │
 │  Output: outputs/intermediate/reasoned_triples.json                 │
 │  Logic:  Parse RDF/XML with rdflib-> flatten all triples to         │
 │          [{subject, predicate, object}, …] JSON                     │
 │  Deps:   rdflib                                                     │
 └────────┬────────────────────────────────────────────────────────────┘
          │
          ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │  Step 8: check-quality  (Scripts/check_quality.py)                  │
 │  CheckQualityStep                                                   │
 │  ─────────────────                                                  │
 │  Input:  outputs/intermediate/reasoned_triples.json                 │
 │  Output: outputs/quality/quality_report.json                        │
 │  Logic:  Compute stats (unique subjects/predicates/objects)         │
 │         -> find duplicate triples                                   │
 │         -> top-N frequency per field                                │
 │         -> produce structured quality report                        │
 └────────┬────────────────────────────────────────────────────────────┘
          │
          ▼
     PIPELINE COMPLETE
          │
          │ (optional — triggered via `pipeline.py repair-kg`)
          ▼
 ┌─────────────────────────────────────────────────────────────────────────┐
 │  Step 9: repair-kg  (rl/ module — RL-based KG Repair)                   │
 │  RepairKGStep                                                           │
 │  ────────────                                                           │
 │  Input:  outputs/intermediate/merged_kg.owl (or evaluation mock)        │
 │          ontology/LLM_GENIALOntBFO_cleaned.owl                          │
 │  Output: outputs/models/dqn_repair_final.pt                             │
 │                                                                         │
 │  Sub-components:                                                        │
 │  ┌────────────────────────────────────────────────────────────────┐     │
 │  │  RepairEnv (rl/env_repair.py) — RL Environment                 │     │
 │  │  ──────────────────────────────────────────────                │     │
 │  │  reset()-> run reasoner-> extract diagnostics-> encode state   │     │
 │  │  step(action)-> apply_fix-> re-run reasoner-> compute reward   │     │
 │  │                                                                │     │
 │  │  State: 8-dim vector encoding error type, action count, etc.   │     │
 │  │  Actions: Repair candidates from qa/repair_candidates.py       │     │
 │  │  Reward: compute_reward() from rl/reward_functions.py          │     │
 │  └────────────────────────────────────────────────────────────────┘     │
 │  ┌────────────────────────────────────────────────────────────────┐     │
 │  │  DQN_Agent (rl/dqn_agent.py) — Neural Network Agent            │     │
 │  │  ───────────────────────────────────────────────               │     │
 │  │  Policy net + target net (rl/dqn_model.py: 256→128→64→N)       │     │
 │  │  ε-greedy exploration-> Q-learning update                      │     │
 │  │  ReplayBuffer (rl/replay_buffer.py) for experience storage     │     │
 │  └────────────────────────────────────────────────────────────────┘     │
 │  ┌────────────────────────────────────────────────────────────────┐     │
 │  │  Reasoner Integration                                          │     │
 │  │  ────────────────────                                          │     │
 │  │  reasoning/reasoner_wrapper.py-> routes to:                    │     │
 │  │    ├── reasoning/hermit_runner.py   (owlready2 + HermiT)       │     │
 │  │    └── reasoning/konclude_runner.py (subprocess + Konclude)    │     │
 │  │  reasoning/axiom_extractor.py-> heuristic evidence extraction  │     │
 │  │  reasoning/parse_reasoner_stats.py-> Konclude output parsing   │     │
 │  └────────────────────────────────────────────────────────────────┘     │
 │  ┌────────────────────────────────────────────────────────────────┐     │
 │  │  Repair Pipeline                                               │     │
 │  │  ───────────────                                               │     │
 │  │  qa/repair_candidates.py-> generates repair action lists       │     │
 │  │  qa/apply_fix.py        -> applies atomic graph operations     │     │
 │  │  rl/diff_tracker.py     -> logs before/after diffs (JSONL)     │     │
 │  │  rl/human_loop.py       -> optional human-in-the-loop          │     │
 │  └────────────────────────────────────────────────────────────────┘     │
 └─────────────────────────────────────────────────────────────────────────┘
```

### File-Level Data Flow Diagram

```
corpus.csv
    │
    ▼ [prepare_corpus.py]
corpus.txt
    │
    ▼ [generate_triples.py] ← Ollama LLM API
corpus_triples.json
    │
    ▼ [clean_triples.py] ← config.CLEAN_ALIASES
corpus_triples_cleaned.json
    │
    ▼ [align_triples.py] ← ontology.owl + relation_map.csv + entity_map.csv
sample_corpus_aligned_triples.ttl  +  alignment_report.csv
    │
    ▼ [build_kg.py] ← ontology.owl
merged_kg.owl  +  merged_kg.ttl
    │
    ▼ [run_reasoner.py] ← HermiT (owlready2)
merged_kg_reasoned.owl
    │
    ▼ [parse_reasoner_output.py]
reasoned_triples.json
    │
    ▼ [check_quality.py]
quality_report.json

    ═══════════════ OPTIONAL RL REPAIR LOOP ═══════════════

merged_kg.owl (or evaluation mock)
    │
    ▼ [env_repair.py] ← run_reasoner()-> axiom_extractor
    │                 ← repair_candidates()-> actions list
    │                 ← apply_fix()-> new OWL file
    │                 ← compute_reward()-> reward signal
    │
    ▼ [dqn_agent.py + replay_buffer.py]
    │   policy_net learns state→action mapping
    │
    ▼ outputs/models/dqn_repair_final.pt
```

### Dependency Graph (Module Imports)

```
pipeline.py
├── config.py
├── core/base_step.py (PipelineStep ABC)
├── Scripts/prepare_corpus.py   -> pandas, config
├── llm_kg/generate_triples.py -> requests, config
├── Scripts/clean_triples.py   -> config
├── Scripts/align_triples.py   -> rdflib, rapidfuzz, config
├── Scripts/build_kg.py        -> rdflib, config
├── Scripts/run_reasoner.py    -> owlready2, config
├── Scripts/parse_reasoner_output.py-> rdflib, config
├── Scripts/check_quality.py   -> config
└── rl/train_repair.py
    ├── rl/env_repair.py
    │   ├── alignment/ontology_index.py-> owlready2
    │   ├── reasoning/reasoner_wrapper.py
    │   │   ├── reasoning/hermit_runner.py-> owlready2, rdflib
    │   │   │   └── reasoning/axiom_extractor.py-> owlready2, rdflib
    │   │   └── reasoning/konclude_runner.py-> subprocess
    │   │       └── reasoning/parse_reasoner_stats.py
    │   ├── qa/repair_candidates.py
    │   └── qa/apply_fix.py-> rdflib
    ├── rl/dqn_agent.py-> torch
    │   └── rl/dqn_model.py-> torch
    ├── rl/replay_buffer.py-> numpy
    ├── rl/diff_tracker.py-> rdflib
    └── rl/human_loop.py
```

---

## 2. Notes

- This document is focused on end-to-end architecture and execution flow.
- Module-level implementation details and internal validation notes are maintained separately.
