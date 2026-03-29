# Project Analysis: Data Flow, Bugs & Test Review

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
 │  Input:  outputs/intermediate/merged_kg.owl (or broken mock)            │
 │          ontology/LLM_GENIALOntBFO_cleaned.owl                          │
 │  Output: outputs/models/dqn_repair_final.pt                             │
 │                                                                         │
 │  Sub-components:                                                        │
 │  ┌────────────────────────────────────────────────────────────────┐     │
 │  │  RepairEnv (rl/env_repair.py) — RL Environment                 │     │
 │  │  ──────────────────────────────────────────────                │     │
 │  │  reset()-> run reasoner-> extract errors-> encode state        │     │
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

merged_kg.owl (or broken mock)
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
└── rl/train_repair.py ← BROKEN (see Bug #1)
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

## 2. Potential Bugs

### BUG #1 — CRITICAL: `rl/train_repair.py` is a copy of `test_repair.py`, missing `train_repair()` function

**File:** [rl/train_repair.py](rl/train_repair.py)

**Problem:** The pipeline's `RepairKGStep` (in [pipeline.py](pipeline.py#L99)) does:

```python
from rl.train_repair import train_repair
agent, episode_rewards, episode_successes = train_repair(**kwargs)
```

But `rl/train_repair.py` contains the **exact same code** as `test_repair.py` — a single-episode test function called `test_repair_one_episode()`. There is **no `train_repair` function** defined anywhere in the codebase. Running `python pipeline.py repair-kg` will crash with an `ImportError`.

**Impact:** The entire RL repair pipeline step is non-functional. The `repair-kg` CLI command will fail immediately.

**Fix:** Implement a proper `train_repair()` function that:

- Creates `RepairEnv`, `DQN_Agent`, `ReplayBuffer`
- Runs multiple episodes with experience replay
- Returns `(agent, episode_rewards, episode_successes)`

---

### BUG #2 — CRITICAL: `logger` undefined in `reasoning/axiom_extractor.py`

**File:** [reasoning/axiom_extractor.py](reasoning/axiom_extractor.py#L129)

**Problem:** Line 129 uses `logger.debug(...)` but no `logger` is ever defined or imported. The file imports `owlready2`, `rdflib`, etc., but does **not** import `logging` or create a logger. This will raise a `NameError: name 'logger' is not defined` at runtime when the `_find_external_type_violations()` function detects a violation.

**Impact:** When `axiom_extractor` finds an external type violation, it crashes instead of returning the evidence dict. This breaks the RL repair loop's evidence extraction pathway.

**Fix:** Add at the top of the file:

```python
import logging
logger = logging.getLogger(__name__)
```

---

### BUG #3 — MEDIUM: `_build_subclass_index` uses mutable default argument for memoisation

**File:** [Scripts/align_triples.py](Scripts/align_triples.py#L221-L230)

**Problem:** The `ancestors()` inner function uses a mutable default argument `memo: dict = {}` for memoisation. This memo dict **persists across calls** because Python evaluates default arguments once at function definition time. In a single pipeline run this is benign (intentional memoization), but if `_build_subclass_index` is called multiple times with different ontology graphs (e.g., in tests or when re-running alignment), the stale memo from the previous graph will contaminate results.

```python
def ancestors(cls: str, memo: dict = {}) -> set[str]:  # ← mutable default!
```

**Impact:** Incorrect subclass hierarchy in subsequent calls within the same process.

**Fix:** Use `None` sentinel and create dict inside:

```python
def ancestors(cls: str, memo: dict = None) -> set[str]:
    if memo is None:
        memo = {}
```

Or better, use a separate `memo = {}` defined in the enclosing `_build_subclass_index` scope.

---

### BUG #4 — MEDIUM: Module-level `_EXT_ANCESTOR_MEMO` never cleared

**File:** [Scripts/align_triples.py](Scripts/align_triples.py#L414)

**Problem:** `_EXT_ANCESTOR_MEMO: dict[str, bool] = {}` is a module-level cache that memoises whether a class has an external ancestor. It is never cleared between pipeline runs in the same process. If the ontology changes between runs (e.g., after editing), stale cached results will be used.

**Impact:** Incorrect external-ancestor detection in long-running processes or test suites that call `align_triples()` multiple times.

---

### BUG #5 — MEDIUM: `run_all()` does not pass default paths to steps

**File:** [pipeline.py](pipeline.py#L573-L605)

**Problem:** In the `run_all()` function, most steps receive empty kwargs dicts:

```python
step_kwargs: dict[str, dict] = {
    ...
    "clean-triples":    {},
    "build-kg":         {},
    "run-reasoner":     {},
    "parse-reasoner":   {},
    "check-quality":    {},
}
```

Each step's `run()` method falls back to `config.DEFAULT_PATHS` when kwargs are missing, so this **works correctly** at runtime. However, the pipeline's output from each step does NOT flow to the next step — each step independently reads from `config.DEFAULT_PATHS`. If `prepare-corpus` writes to a non-default output path, `generate-triples` still reads from the default path. The `run-all` mode only works if all steps use default paths.

**Impact:** The `--input` flag passed to `run-all` only affects the `prepare-corpus` step. All subsequent steps ignore it and read from their default paths regardless.

---

### BUG #6 — MEDIUM: `HermiT` runner re-assigns `JAVA_MEMORY` after import (no effect)

**File:** [reasoning/hermit_runner.py](reasoning/hermit_runner.py#L9-L10)

**Problem:**

```python
from owlready2 import JAVA_MEMORY
JAVA_MEMORY = "1024M"
```

This imports `JAVA_MEMORY` from `owlready2` and then rebinds the **local** name. It does NOT modify `owlready2.JAVA_MEMORY`. The JVM will use the default memory setting.

**Impact:** JVM memory is not increased to 1024M as intended. Large ontologies may cause out-of-memory errors.

**Fix:**

```python
import owlready2
owlready2.JAVA_MEMORY = 1024
```

---

### BUG #7 — LOW: `DQN_Agent.select_action()` may return invalid action indexes

**File:** [rl/dqn_agent.py](rl/dqn_agent.py#L59-L66)

**Problem:** The agent's output dimension is fixed at `action_space_n() = 10` (defined in `env_repair.py`), but the actual number of valid actions per state is variable (typically 2–5). When exploiting (not exploring), `select_action()` returns `q_values.argmax()` which can be any index 0–9. If the env has only 3 actions, indices 3–9 are invalid.

The environment's `step()` method partially guards against this:

```python
if action_idx < 0 or action_idx >= len(actions):
    action_idx = 0  # silent fallback
```

But this means the agent's learned Q-values for indices > actual action count are wasted, and the agent always silently falls back to action 0 — degrading learning.

**Impact:** Suboptimal RL training; agent frequently takes action 0 unintentionally.

**Fix:** Mask invalid actions in `select_action()` by setting Q-values for invalid indices to `-inf` before taking `argmax`.

---

### BUG #8 — LOW: `evaluate_agent()` in `dqn_agent.py` references gym-style API that doesn't exist

**File:** [rl/dqn_agent.py](rl/dqn_agent.py#L149-L171)

**Problem:** The `evaluate_agent()` function at the bottom of `dqn_agent.py` references `env.y[env.current_index]` and `env.reset()` returning a single value. This is a classification environment API (gym-style), not the `RepairEnv` API which returns `(state, done)` from `reset()`. This function cannot work with `RepairEnv`.

**Impact:** Dead code — will crash if called. No current callers, but misleading.

---

### BUG #9 — LOW: `trainer.py` uses gym-style API incompatible with RepairEnv

**File:** [rl/trainer.py](rl/trainer.py)

**Problem:** `trainer.py` imports `gym` and defines `train_dqn()` using the standard gym API:

```python
state = env.reset()          # expects single return value
state, _, done, _ = env.step(action)  # expects 4-tuple
```

But `RepairEnv.reset()` returns `(state, done)` — a 2-tuple. This trainer is from a previous classification task and is **not compatible** with the repair environment.

**Impact:** `trainer.py` is unusable with the current RL repair system. It's dead code.

---

### BUG #10 — LOW: `individual_counter` is logged but never incremented

**File:** [Scripts/align_triples.py](Scripts/align_triples.py#L458)

**Problem:** In `align_triples()`, `individual_counter = {}` is declared for logging purposes. The log message references `sum(individual_counter.values())`, but `individual_counter` is never incremented anywhere. The `ensure_individual()` function receives it as a parameter (for "API compat") but doesn't use it. The log always shows `individuals created: 0`.

**Impact:** Misleading log output — cosmetic only, no functional impact.

---

### BUG #11 — LOW: `_ensure_domain_range` checks domain before range but only auto-fixes untyped entities

**File:** [Scripts/align_triples.py](Scripts/align_triples.py#L299-L350)

**Problem:** The domain check adds an rdf:type for untyped subjects, but if the subject is already typed with something incompatible, it skips the triple. The range check does the same. However, there's a logical gap: if a subject has no types and the domain type is added, and then the range check fails, the subject is left with a type assertion that was only added to satisfy a triple that ultimately got skipped. This orphan type could cause conflicts later.

**Impact:** Possible stale type assertions on subjects for skipped triples.

---

### BUG #12 — LOW: Duplicate `start_time` / `t0` in hermit_runner

**File:** [reasoning/hermit_runner.py](reasoning/hermit_runner.py#L30-L31)

**Problem:** Both `start_time = time.perf_counter()` and `t0 = time.time()` are captured. Only `start_time` is used for the final elapsed time. `t0` is a leftover from refactoring and is never used.

**Impact:** Dead variable — no functional impact.

---

## 3. Test Analysis

### test_reasoner.py

**File:** [test_reasoner.py](test_reasoner.py)

| Test # | Description                             | Expected          | Reasoner |
| ------ | --------------------------------------- | ----------------- | -------- |
| 1      | Consistent mock                         | Consistent: True  | HermiT   |
| 2      | Broken mock (injected domain violation) | Consistent: False | HermiT   |
| 3      | Consistent mock                         | Consistent: True  | Konclude |
| 4      | Live pipeline output                    | Consistent: True  | HermiT   |
| 5      | Broken mock (RL target)                 | Consistent: False | HermiT   |

**Issues found:**

1. **Test 5 is a duplicate of Test 2** — same file, same expectation, same reasoner. Tests the identical scenario without adding value. Should test a different error type or use Konclude.

2. **No isolation** — tests depend on external files (`outputs/test/merged_kg_mock.owl`, etc.). If these files don't exist, tests silently SKIP instead of FAIL, with a suggestion to run `regenerate_mock.py`. This means CI could pass with zero tests actually executed.

3. **No assertions / no test framework** — the test function returns `bool` but uses `print()` for output instead of `assert` or a test framework like `pytest`. The exit code is always 0 regardless of test results.

4. **Test 3 (Konclude)** will always SKIP/FAIL on most systems because Konclude requires a Windows binary (`Konclude.exe`) and the project runs on macOS (current OS). The runner tries to find `Konclude.exe` / `Konclude.bat` — neither exists on macOS.

### test_repair.py

**File:** [test_repair.py](test_repair.py)

**Issues found:**

1. **Not a proper test** — it's a manual integration smoke test that runs one random episode. There are no assertions, no expected outcomes, and no pass/fail criteria. It always prints "TEST COMPLETE" even if everything broke.

2. **Depends on external files** — requires `outputs/test/merged_kg_broken.owl` and `ontology/LLM_GENIALOntBFO_cleaned.owl`. No graceful handling if missing.

3. **Uses `random.randint` without seed** — results are non-reproducible across runs.

4. **Imports `random` inside the loop** — the `import random` statement is inside the `while` loop body. Python handles this efficiently (cached after first import), but it's unconventional and should be at module level.

5. **rl/train_repair.py is identical to test_repair.py** (Bug #1) — suggests a copy-paste error where the actual training orchestrator was never written or was accidentally overwritten.

### General Test Gaps

- **No unit tests** for pure functions like `normalize_entity()`, `split_object_on_and()`, `clean_triples()`, `compute_stats()`, `find_duplicates()`, `best_match()`, etc.
- **No test for the alignment step** — the most complex and bug-prone module has zero test coverage.
- **No mock-based tests** — LLM calls, reasoner calls, and file I/O are never mocked.
- **No CI configuration** — no `pytest.ini`, `tox.ini`, or GitHub Actions workflow.

---

## 4. Summary of Severity

| #   | Bug                                                    | Severity     | Status               |
| --- | ------------------------------------------------------ | ------------ | -------------------- |
| 1   | `train_repair()` function missing — RL pipeline broken | **CRITICAL** | Needs implementation |
| 2   | `logger` undefined in axiom_extractor.py               | **CRITICAL** | Needs 2-line fix     |
| 3   | Mutable default arg in `ancestors()` memo              | Medium       | Needs fix            |
| 4   | `_EXT_ANCESTOR_MEMO` never cleared                     | Medium       | Needs fix            |
| 5   | `run-all` doesn't chain step outputs                   | Medium       | Design limitation    |
| 6   | `JAVA_MEMORY` assignment has no effect                 | Medium       | Needs fix            |
| 7   | DQN may select invalid action indexes                  | Low          | Needs masking        |
| 8   | `evaluate_agent()` uses wrong env API                  | Low          | Dead code            |
| 9   | `trainer.py` incompatible with RepairEnv               | Low          | Dead code            |
| 10  | `individual_counter` never incremented                 | Low          | Cosmetic             |
| 11  | Orphan type assertions on skipped triples              | Low          | Edge case            |
| 12  | Dead `t0` variable in hermit_runner                    | Low          | Cosmetic             |
