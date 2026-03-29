# KGG4SE-RL — Module Integration Guide

This document describes how the modules in this project connect to each other and how data flows between them. It is intended for anyone extending or integrating with this codebase.

---

## How the Modules Connect

All modules are already wired together through `pipeline.py` and `config.py`. No manual connection is needed.

### Full Call Chain

```
pipeline.py  (CLI entry point)
    │
    ├── config.py                          (imported by every module)
    ├── core/base_step.py                  (PipelineStep abstract base class)
    │
    ├── Scripts/prepare_corpus.py          → pandas
    ├── llm_kg/generate_triples.py         → requests, OpenAI API / Ollama
    ├── Scripts/clean_triples.py           → config
    ├── Scripts/align_triples.py           → rdflib, rapidfuzz, config
    ├── Scripts/build_kg.py                → rdflib, config
    ├── Scripts/run_reasoner.py            → reasoning/reasoner_wrapper.py
    ├── Scripts/parse_reasoner_output.py   → rdflib
    ├── Scripts/check_quality.py           → config
    │
    └── rl/train_repair.py
        │
        ├── rl/env_repair.py               (RepairEnv)
        │   ├── alignment/ontology_index.py    → owlready2
        │   ├── reasoning/reasoner_wrapper.py
        │   │   ├── reasoning/hermit_runner.py   → owlready2, rdflib
        │   │   ├── reasoning/konclude_runner.py → subprocess
        │   │   └── reasoning/axiom_extractor.py → owlready2, rdflib
        │   ├── qa/repair_candidates.py
        │   └── qa/apply_fix.py            → rdflib
        │
        ├── rl/dqn_agent.py                → torch
        │   └── rl/dqn_model.py            → torch
        ├── rl/replay_buffer.py            → numpy
        ├── rl/diff_tracker.py             → rdflib
        └── rl/human_loop.py
```

---

## RL Repair Loop — Step by Step

This is what happens inside each call to `RepairEnv.step(action_idx)`:

```
1. RepairEnv.step(action_idx)
        │
        ▼
2. qa/apply_fix.apply_fix(owl_graph, action)
        │  Modifies the in-memory RDF graph
        │  Saves snapshot to outputs/rl_repair_steps/repaired_step_NNNN.owl
        │
        ▼
3. reasoning/reasoner_wrapper.run_reasoner(owl_path)
        │  Routes to konclude_runner or hermit_runner
        │  Returns: {is_consistent, unsat_classes, disjoint_violations, ...}
        │
        ▼
4. reasoning/axiom_extractor.extract_inconsistency_explanation(owl_path)
        │  Heuristic search for root cause
        │  Returns: {error_type, evidence_triples, involved_iris, ...}
        │
        ▼
5. qa/repair_candidates.make_repair_candidates(report, evidence)
        │  Generates list of candidate actions for the next step
        │
        ▼
6. rl/reward_functions.compute_reward(before_report, after_report, step_num)
        │  Returns shaped scalar reward
        │
        ▼
7. rl/diff_tracker.log_step(...)
        │  Appends step record to outputs/rl_repair_traces/repair_trace.jsonl
        │
        ▼
8. Returns (new_state, reward, done, info) to train_repair.py
```

---

## Verifying Imports

To confirm all modules are importable after installation:

```bash
python -c "from reasoning.axiom_extractor import extract_inconsistency_explanation; print('axiom_extractor OK')"
python -c "from qa.repair_candidates import make_repair_candidates; print('repair_candidates OK')"
python -c "from qa.apply_fix import apply_fix; print('apply_fix OK')"
python -c "from rl.env_repair import RepairEnv; print('env_repair OK')"
python -c "from rl.diff_tracker import DiffTracker; print('diff_tracker OK')"
python -c "from rl.human_loop import HumanLoop; print('human_loop OK')"
python -c "from rl.train_repair import train_repair; print('train_repair OK')"
```

All lines should print without errors. If any fail, check that `requirements.txt` dependencies are installed and that you are running from the project root.

---

## Running a Single Test Episode

To verify the full repair loop works end-to-end without running full training:

```bash
python -m rl.train_repair --test --no-human
```

This runs exactly one episode with up to 5 repair steps and prints the result.

---

## Configuration Points for Extension

If you want to extend or adapt this codebase:

| What you want to change | Where to change it |
|---|---|
| Add a new repair action type | `qa/apply_fix.py` — add a new `elif action_type == "..."` branch |
| Add a new violation category | `qa/repair_candidates.py` — add detection and candidate generation for the new type |
| Change the state vector | `rl/env_repair.py` — modify `_encode_state()` and update `state_dim()` |
| Change the reward function | `rl/reward_functions.py` — modify `compute_reward()` |
| Add a new reasoner | `reasoning/reasoner_wrapper.py` — add a new runner and register it |
| Add a new pipeline step | Create a class inheriting from `core/base_step.py:PipelineStep`, then register it in `pipeline.py` |
| Change LLM prompt | `llm_kg/generate_triples.py` — modify `config.PROMPT_TEMPLATE` in `config.py` |

---

## Output Files Produced by the RL Module

| File | Produced by | Description |
|---|---|---|
| `outputs/rl_repair_steps/repaired_step_NNNN_<uuid>.owl` | `qa/apply_fix.py` | OWL snapshot after each repair step |
| `outputs/rl_repair_traces/repair_trace.jsonl` | `rl/diff_tracker.py` | Step-by-step training trace (JSONL) |
| `outputs/models/dqn_repair_ep{N}.pt` | `rl/train_repair.py` | Model checkpoint per episode |
| `outputs/models/dqn_repair_final.pt` | `rl/train_repair.py` | Final trained model |
| `outputs/models/training_history.json` | `rl/train_repair.py` | Per-episode metrics log |
| `outputs/models/human_feedback.jsonl` | `rl/human_loop.py` | Human action decisions (reusable as scripted input) |
