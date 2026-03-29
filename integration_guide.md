# COMPLETE INTEGRATION GUIDE FOR RL REPAIR MODULE

## STEP 1: UPDATE config.py

Open your existing `config.py` and **ADD THIS TO THE END** (after the last line):

```python
# Copy everything from config_snippet_for_rl.py and paste here
```

Or simply append the config_snippet_for_rl.py file:
```bash
type config_snippet_for_rl.py >> config.py
```

## STEP 2: REPLACE pipeline.py

Replace your existing `pipeline.py` with the new enhanced version I provided.

```bash
copy pipeline.py D:\Thesis\kgg4se_framework_gaurav\kgg4se_framework\pipeline.py
```

## STEP 3: ADD test_repair.py

Copy test_repair.py to your project root:

```bash
copy test_repair.py D:\Thesis\kgg4se_framework_gaurav\kgg4se_framework\test_repair.py
```

## STEP 4: KEEP test_reasoner.py

**YES, KEEP IT!** It's useful for testing reasoners independently.

test_reasoner.py = Tests reasoners only
test_repair.py   = Tests full RL repair loop

## CONNECTIONS - ALREADY DONE! ✅

The RL module is ALREADY CONNECTED to reasoning:

```
RepairEnv (rl/env_repair.py)
    ↓ calls
run_reasoner() (reasoning/reasoner_wrapper.py)
    ↓ which uses
konclude_runner.py / hermit_runner.py
    ↓ and triggers
axiom_extractor.py (NEW)
    ↓ generates
repair_candidates (qa/repair_candidates.py)
    ↓ executed by
apply_fix (qa/apply_fix.py)
    ↓ creates new OWL
    ↓ loops back to
run_reasoner() again
```

**NO MANUAL CONNECTION NEEDED** - Everything flows automatically!

## TESTING SEQUENCE

### Test 1: Verify Imports (1 min)
```bash
python -c "from reasoning.axiom_extractor import extract_inconsistency_explanation; print('✅ axiom_extractor OK')"
python -c "from qa.repair_candidates import make_repair_candidates; print('✅ repair_candidates OK')"
python -c "from qa.apply_fix import apply_fix; print('✅ apply_fix OK')"
python -c "from rl.env_repair import RepairEnv; print('✅ env_repair OK')"
python -c "from rl.diff_tracker import DiffTracker; print('✅ diff_tracker OK')"
python -c "from rl.human_loop import HumanLoop; print('✅ human_loop OK')"
python -c "from rl.train_repair import train_repair; print('✅ train_repair OK')"
```

**Expected:** All print ✅ messages
**If fails:** Missing file - check folder structure

---

### Test 2: Test Evidence Extraction (1 min)
```bash
python -c "from reasoning.axiom_extractor import extract_inconsistency_explanation; evidence = extract_inconsistency_explanation('outputs/intermediate/merged_kg.owl'); print('Evidence type:', evidence.get('error_type') if evidence else 'None')"
```

**Expected output:**
```
[axiom_extractor] Found evidence via heuristic search: disjoint_violation
Evidence type: disjoint_violation
```
OR
```
[axiom_extractor] WARNING: No evidence found
Evidence type: unknown_inconsistency
```

**If fails:** Check if merged_kg.owl exists

---

### Test 3: Test Pipeline Command (1 min)
```bash
python pipeline.py repair-kg --help
```

**Expected output:**
```
usage: pipeline repair-kg [-h] [--input INPUT] [--ontology ONTOLOGY]
                         [--episodes EPISODES] [--max-steps MAX_STEPS]
                         [--reasoner {konclude,hermit}] [--interactive]
                         [--batch-size BATCH_SIZE] [--save-dir SAVE_DIR]

optional arguments:
  --input INPUT         OWL file to repair (default: merged_kg.owl).
  --ontology ONTOLOGY   Base ontology for alignment.
  ...
```

**If fails:** pipeline.py not updated correctly

---

### Test 4: Run Quick Test (2-3 min)
```bash
python test_repair.py
```

**Expected output:**
```
TESTING RL REPAIR MODULE - ONE EPISODE
============================================================

[1/4] Initializing RepairEnv...
✅ RepairEnv initialized

[2/4] Initializing DQN Agent...
✅ DQN Agent initialized (state_dim=8, action_dim=10)

[3/4] Initializing DiffTracker...
✅ DiffTracker initialized-> outputs\rl_repair_traces

[4/4] Running test episode (max 5 steps)...
------------------------------------------------------------
Episode started, initial done=False

--- Step 0 ---
Error type: disjoint_violation
Available actions: 4
Example action: Remove 'Alice is a Person'
Choosing action: 1/3
Reward: 10.00
Metrics: unsat=0, disj=0
Done: True

============================================================
EPISODE SUMMARY
============================================================
Total steps: 1
Total reward: 10.00
Final consistent: True

Trace file: outputs\rl_repair_traces\repair_trace.jsonl
Actions taken: ['drop_class_assertion']

✅ TEST COMPLETE
```

**What this tests:**
- RepairEnv initialization ✓
- DQN Agent creation ✓
- Evidence extraction ✓
- Repair candidates generation ✓
- Action execution ✓
- Diff tracking ✓
- Reasoner feedback loop ✓

**If it completes:** Everything works! 🎉
**If fails:** See error message and troubleshoot

---

### Test 5: Run 5 Training Episodes (5-10 min)
```bash
python pipeline.py repair-kg --episodes 5 --max-steps 10 --reasoner hermit
```

**Expected output:**
```
TRAINING DQN AGENT FOR KG REPAIR
============================================================
Base OWL: outputs\intermediate\merged_kg.owl
Episodes: 5
Max steps/episode: 10
Reasoner: hermit
Interactive: False
============================================================

[RepairEnv] Reasoner report: consistent=False, unsat=0, disjoint_violations=0
[axiom_extractor] Found evidence via heuristic search: disjoint_violation
[RepairEnv] Generated 1 repair candidates

============================================================
EPISODE 1/5
============================================================
[RepairEnv] Step 0: Applying action 0
[apply_fix] Removed ClassAssertion(...)
[RepairEnv] Base reward: 10.00, Risk penalty: 1.00, Final: 9.00
  [Step 1] Action: 0, Reward: 9.00, Total: 9.00
[RepairEnv] Episode finished: steps=1, final_reward=9.00

[Episode 1] Summary:
  Steps taken: 1
  Total reward: 9.00
  Final consistent: True
  Epsilon: 0.990

...

============================================================
TRAINING STATISTICS
============================================================
Total episodes: 5
Success rate: 4/5 (80.0%)
Average reward: 8.50
Best reward: 10.00 (episode 2)
Worst reward: -5.00 (episode 4)

============================================================
RL REPAIR TRAINING COMPLETE
============================================================
Success rate: 80.0% (4/5 episodes)
Average reward: 8.50
Model saved to: outputs\models
```

**What this tests:**
- Full training loop ✓
- DQN learning ✓
- Multi-episode training ✓
- Model saving ✓

---

### Test 6: Check Output Files
```bash
dir outputs\rl_repair_steps
dir outputs\rl_repair_traces
dir outputs\models
```

**Expected:**
```
outputs\rl_repair_steps\
  repaired_step_0000_abc123.owl
  repaired_step_0001_def456.owl
  ...

outputs\rl_repair_traces\
  repair_trace.jsonl

outputs\models\
  human_feedback.jsonl
  dqn_repair_final.pt  (if training completed)
```

---

## TROUBLESHOOTING

### Error: "ModuleNotFoundError: No module named 'rl'"
**Fix:** Make sure you created the `rl/` folder and `rl/__init__.py` file

### Error: "ModuleNotFoundError: No module named 'reasoning.axiom_extractor'"
**Fix:** Copy `axiom_extractor.py` to `reasoning/` folder

### Error: "AttributeError: module 'config' has no attribute 'RL_EPISODES'"
**Fix:** Add the config snippet to your `config.py` file

### Error: File not found: merged_kg.owl
**Fix:** Run the main pipeline first to generate merged_kg.owl:
```bash
python pipeline.py run-all --input data/01_corpus/corpus.csv
```

### Evidence extractor returns "unknown_inconsistency"
**This is OK!** It means:
- Your KG is inconsistent (reasoners detect it)
- But the heuristic couldn't find the exact cause
- RL will still try random fixes

**To improve:** Check HermiT error message manually and add parsing logic

---

## WHAT EACH FILE DOES

| File | Purpose | Already Connected? |
|------|---------|-------------------|
| reasoning/axiom_extractor.py | Finds WHY KG is broken | ✅ Used by RepairEnv |
| qa/repair_candidates.py | Generates fix suggestions | ✅ Used by RepairEnv |
| qa/apply_fix.py | Executes fixes | ✅ Used by RepairEnv |
| rl/env_repair.py | RL environment | ✅ Calls reasoning + qa |
| rl/dqn_agent.py | DQN neural network | ✅ Used by train_repair |
| rl/diff_tracker.py | Logs changes | ✅ Used by train_repair |
| rl/human_loop.py | Collects feedback | ✅ Used by train_repair |
| rl/train_repair.py | Training orchestrator | ✅ Called by pipeline |
| pipeline.py | CLI entry point | ✅ Imports RepairKGStep |
| test_repair.py | Quick test script | ✅ Tests all modules |

**ALL FILES ARE ALREADY CONNECTED!** No manual wiring needed.

---

## QUICK START COMMANDS

```bash
# 1. Quick sanity check (30 seconds)
python test_repair.py

# 2. Train for 5 episodes (5 minutes)
python pipeline.py repair-kg --episodes 5 --max-steps 10

# 3. Full training (20 minutes)
python pipeline.py repair-kg --episodes 20 --max-steps 20

# 4. Interactive mode (human helps)
python pipeline.py repair-kg --episodes 10 --interactive
```

---

## SUCCESS CRITERIA

✅ Test 1 passes-> Imports work
✅ Test 2 passes-> Evidence extraction works
✅ Test 3 passes-> Pipeline integration works
✅ Test 4 passes-> Full loop works
✅ Test 5 passes-> Training works
✅ Test 6 passes-> Files created

**IF ALL PASS-> STEP 2 IS COMPLETE! 🎉**

Next: Move to Step 3 (already implemented in train_repair.py)