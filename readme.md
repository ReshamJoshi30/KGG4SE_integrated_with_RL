# KGG4SE-RL: Knowledge Graph Generation for Software Engineering with Reinforcement Learning

This project implements an end-to-end pipeline for automatically building and repairing an OWL Knowledge Graph (KG) from an automotive text corpus.

**What it does:**
1. Extracts `Subject | Predicate | Object` triples from raw text using an LLM (GPT-4o-mini or Llama 3)
2. Aligns extracted triples to an existing OWL ontology using fuzzy string matching
3. Merges aligned triples with the base ontology to build a complete KG
4. Checks OWL consistency using a reasoner (Konclude or HermiT) to detect violations
5. Repairs detected inconsistencies using a Deep Q-Network (DQN) reinforcement learning agent — with optional human-in-the-loop guidance

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Pipeline](#running-the-pipeline)
- [RL Repair: Human-in-the-Loop Mode](#rl-repair-human-in-the-loop-mode)
- [RL Repair: Fully Automated Mode](#rl-repair-fully-automated-mode)
- [Testing Violations with inject_violations.py](#testing-violations)
- [Output Structure](#output-structure)
- [Project Structure](#project-structure)

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.12+ | |
| Java 8+ | Only required if using HermiT reasoner |
| OpenAI API key | Required when `USE_OPENAI = True` in `config.py` |
| Ollama | Required when using local Llama 3 instead of OpenAI |

**Konclude** (the default OWL reasoner) is bundled at `reasoning/Konclude/Binaries/Konclude.exe` — no separate installation needed on Windows.

---

## Installation

```bash
pip install -r requirements.txt
```

PyTorch is required for the RL repair agent. If it is not pulled automatically:

```bash
# CPU only
pip install torch

# With CUDA GPU — see https://pytorch.org/get-started/locally/
```

---

## Configuration

All settings live in `config.py`. Key settings to review before running:

```python
# LLM backend
USE_OPENAI   = True            # set False to use local Ollama instead
OPENAI_MODEL = "gpt-4o-mini"
MODEL_NAME   = "llama3"        # Ollama model (used when USE_OPENAI = False)
OLLAMA_URL   = "http://localhost:11434/api/generate"

# Reasoner
DEFAULT_REASONER = "konclude"  # or "hermit"

# RL training
RL_EPISODES                  = 50   # total training episodes
RL_MAX_STEPS_PER_EPISODE     = 15   # max repair actions per episode
RL_HUMAN_SUPERVISED_EPISODES = 3    # Phase 1: human decides every action
RL_HUMAN_FIRST_EPISODES      = 5    # Phase 2: human decides first action only
```

### Set the OpenAI API key

```bash
# Linux / macOS
export OPENAI_API_KEY="sk-..."

# Windows (Command Prompt)
set OPENAI_API_KEY=sk-...

# Windows (PowerShell)
$env:OPENAI_API_KEY="sk-..."
```

### Use local Ollama instead of OpenAI

```bash
# Pull the model once
ollama pull llama3
```

Then in `config.py` set `USE_OPENAI = False`.

---

## Running the Pipeline

### Option A — Run everything in one command

```bash
python pipeline.py run-all
```

This runs all 8 steps in sequence using the defaults from `config.py`. The pipeline stops immediately if any step fails.

```bash
# Custom corpus, model, and alignment settings
python pipeline.py run-all \
  --input        data/01_corpus/corpus.csv \
  --model        llama3 \
  --fuzzy-cutoff 85
```

### Option B — Run steps individually

Each step reads from the previous step's output. Run them in order.

---

#### Step 1 — `prepare-corpus`

Reads a CSV file and writes each row's text to a plain `.txt` file. The CSV must have a `title` column (configurable).

```bash
python pipeline.py prepare-corpus

# Custom paths
python pipeline.py prepare-corpus \
  --input       data/01_corpus/corpus.csv \
  --output      data/01_corpus/corpus.txt \
  --text-column title
```

**Input:** `data/01_corpus/corpus.csv`
**Output:** `data/01_corpus/corpus.txt`

---

#### Step 2 — `generate-triples`

Sends each line of the corpus to the LLM and extracts `Subject | Predicate | Object` triples. Works with both OpenAI and local Ollama.

```bash
python pipeline.py generate-triples

# Custom paths and model
python pipeline.py generate-triples \
  --input   data/01_corpus/corpus.txt \
  --output  outputs/intermediate/corpus_triples.json \
  --model   llama3 \
  --timeout 120
```

**Input:** `data/01_corpus/corpus.txt`
**Output:** `outputs/intermediate/corpus_triples.json`

---

#### Step 3 — `clean-triples`

Normalizes text, removes duplicates, and filters out triples with overly long tokens.

```bash
python pipeline.py clean-triples

# Custom paths
python pipeline.py clean-triples \
  --input      outputs/intermediate/corpus_triples.json \
  --output     outputs/intermediate/corpus_triples_cleaned.json \
  --max-tokens 5
```

**Input:** `outputs/intermediate/corpus_triples.json`
**Output:** `outputs/intermediate/corpus_triples_cleaned.json`

---

#### Step 4 — `align-triples`

Maps each triple's subject, predicate, and object to URIs from the base ontology using fuzzy string matching. Creates new URIs for entities with no match when `--allow-create` is set.

```bash
python pipeline.py align-triples

# Custom paths and settings
python pipeline.py align-triples \
  --input        outputs/intermediate/corpus_triples_cleaned.json \
  --output       outputs/intermediate/aligned_triples.ttl \
  --ontology     ontology/LLM_GENIALOntBFO_cleaned.owl \
  --fuzzy-cutoff 85 \
  --allow-create
```

**Input:** `outputs/intermediate/corpus_triples_cleaned.json`
**Output:** `outputs/intermediate/aligned_triples.ttl` + `outputs/reports/alignment_report.csv`

---

#### Step 5 — `build-kg`

Merges the aligned triples with the base ontology into a single OWL Knowledge Graph file.

```bash
python pipeline.py build-kg

# Custom paths
python pipeline.py build-kg \
  --input      outputs/intermediate/aligned_triples.ttl \
  --ontology   ontology/LLM_GENIALOntBFO_cleaned.owl \
  --output-owl outputs/intermediate/merged_kg.owl \
  --output-ttl outputs/intermediate/merged_kg.ttl
```

**Input:** `outputs/intermediate/aligned_triples.ttl`
**Output:** `outputs/intermediate/merged_kg.owl`

---

#### Step 6 — `run-reasoner`

Runs an OWL reasoner on the merged KG to check for logical inconsistencies (unsatisfiable classes, disjoint violations, domain/range errors, etc.).

```bash
python pipeline.py run-reasoner

# Use HermiT instead of default Konclude
python pipeline.py run-reasoner --reasoner hermit

# Custom input/output
python pipeline.py run-reasoner \
  --input  outputs/intermediate/merged_kg.owl \
  --output outputs/knowledge_graph/merged_kg_reasoned.owl
```

**Input:** `outputs/intermediate/merged_kg.owl`
**Output:** `outputs/knowledge_graph/merged_kg_reasoned.owl` + logs in `outputs/reasoning/`

---

#### Step 7 — `parse-reasoner`

Parses the reasoned OWL file into a flat JSON list of triples.

```bash
python pipeline.py parse-reasoner

# Custom paths
python pipeline.py parse-reasoner \
  --input  outputs/knowledge_graph/merged_kg_reasoned.owl \
  --output outputs/intermediate/reasoned_triples.json
```

**Input:** `outputs/knowledge_graph/merged_kg_reasoned.owl`
**Output:** `outputs/intermediate/reasoned_triples.json`

---

#### Step 8 — `check-quality`

Generates a quality report: triple counts, duplicate rates, top-N predicates, and sample triples.

```bash
python pipeline.py check-quality

# Custom paths and options
python pipeline.py check-quality \
  --input       outputs/intermediate/reasoned_triples.json \
  --output      outputs/quality/quality_report.json \
  --sample-size 20 \
  --top-n       10
```

**Input:** `outputs/intermediate/reasoned_triples.json`
**Output:** `outputs/quality/quality_report.json`

---

## RL Repair: Human-in-the-Loop Mode

After building the KG and running the reasoner, the repair agent can fix detected inconsistencies. Human-in-the-loop mode uses a **3-phase curriculum** where the human gradually hands control over to the RL agent.

### How the three phases work

| Phase | Episodes | Who decides the repair action? |
|---|---|---|
| **Phase 1: Human Supervised** | Episodes 1 – `RL_HUMAN_SUPERVISED_EPISODES` | Human selects every action. The agent observes and learns. |
| **Phase 2: Human First** | Next `RL_HUMAN_FIRST_EPISODES` episodes | Human selects only the first action per episode. The agent handles the rest. |
| **Phase 3: RL Automated** | Remaining episodes | The agent acts fully autonomously. Human is only prompted when agent confidence falls below the threshold. |

### Running with human-in-the-loop

```bash
python pipeline.py repair-kg --interactive
```

Or directly via the training module:

```bash
python -m rl.train_repair --episodes 50 --max-steps 15
```

### What happens during a human-supervised step

When it is your turn to decide, the terminal will display:
- The current list of detected violations
- The candidate repair actions available (e.g. `remap_entity`, `drop_entity`, `add_type_assertion`)
- A prompt asking you to enter the action number

Example interaction:

```
Episode 1 | Step 1 | Phase: HUMAN_SUPERVISED
----------------------------------------------
Violations detected: 3
  [0] disjoint_violation: Vehicle and Component are disjoint (individual: car_001)
  [1] domain_violation:   hasSpeed expects a Vehicle, got Component (car_001)
  [2] range_violation:    hasSpeed range is Speed, got string literal

Repair candidates:
  [0] remap_entity       — remap car_001 to Vehicle class
  [1] drop_entity        — remove car_001 from the graph
  [2] add_type_assertion — add rdf:type Vehicle to car_001

Enter action number (or press Enter to skip): 0

Action applied: remap_entity — car_001 remapped to Vehicle
Reward: +2.0 (1 violation fixed)
```

### Saving and reusing human feedback

Human decisions are saved to `outputs/models/human_feedback.jsonl`. On subsequent runs the agent can load this file to pre-seed its experience replay buffer:

```bash
python -m rl.train_repair \
  --episodes   50 \
  --max-steps  15 \
  --scripted   outputs/models/human_feedback.jsonl
```

---

## RL Repair: Fully Automated Mode

Once the agent has been trained (or to skip human guidance entirely), run in no-human mode:

```bash
python pipeline.py repair-kg --no-human
```

Or directly:

```bash
python -m rl.train_repair \
  --episodes  50 \
  --max-steps 15 \
  --no-human
```

### Additional options

```bash
python -m rl.train_repair \
  --episodes  50 \
  --max-steps 15 \
  --no-human \
  --reasoner  konclude \           # or hermit
  --owl       outputs/intermediate/merged_kg.owl \
  --batch-size 32
```

### What the agent learns

The DQN agent receives an 18-dimensional state vector encoding:
- 10 error-type counts (disjoint violations, domain errors, range errors, unsatisfiable classes, etc.)
- 8 graph-level features (triple count, violation rate, etc.)

It selects from the list of repair candidate actions and receives a shaped reward:

| Event | Reward |
|---|---|
| KG becomes fully consistent | +10 |
| Each violation fixed | +2 |
| New violation introduced | -2 |
| KG becomes more inconsistent | -10 |
| Each step taken (efficiency penalty) | -0.1 |

Trained checkpoints are saved every episode to `outputs/models/dqn_repair_ep{N}.pt`.
The final model is saved to `outputs/models/dqn_repair_final.pt`.
Full training trace (state, action, reward per step) is written to `outputs/rl_repair_traces/repair_trace.jsonl`.

---

## Testing Violations

`inject_violations.py` deliberately injects OWL inconsistencies into a KG file. This is used to verify that the reasoner detects them and that the RL agent can fix them.

```bash
# Inject violations into the merged KG
python inject_violations.py

# Then run the reasoner to confirm detection
python pipeline.py run-reasoner

# Then run the RL agent to fix them
python pipeline.py repair-kg --no-human
```

This is the standard workflow for evaluating RL repair performance.

---

## Logging

```bash
python pipeline.py --log-level DEBUG run-all     # verbose output
python pipeline.py --log-level WARNING run-all   # errors and warnings only
```

Default log level is `INFO`.

---

## Output Structure

```
outputs/
├── intermediate/            # Step-by-step artifacts
│   ├── corpus_triples.json          # raw extracted triples
│   ├── corpus_triples_cleaned.json  # cleaned triples
│   ├── aligned_triples.ttl          # ontology-aligned triples
│   ├── merged_kg.owl                # merged KG before reasoning
│   └── reasoned_triples.json        # parsed triples after reasoning
├── knowledge_graph/         # Final reasoned KG
│   └── merged_kg_reasoned.owl
├── quality/                 # Quality report
│   └── quality_report.json
├── reports/                 # Alignment report and plots
│   └── alignment_report.csv
├── reasoning/               # Raw reasoner outputs (OWL, logs, stats)
├── models/                  # DQN model checkpoints
│   ├── dqn_repair_ep{N}.pt          # per-episode checkpoints
│   ├── dqn_repair_final.pt          # final trained model
│   └── human_feedback.jsonl         # saved human decisions
├── rl_repair_steps/         # Repaired OWL snapshot at each step
├── rl_repair_traces/        # Training trace log
│   └── repair_trace.jsonl
└── diagrams/                # Architecture and flow diagrams (PNG)
```

---

## Project Structure

```
KGG4SE_RL/
├── pipeline.py                  # Main CLI entry point for all pipeline commands
├── config.py                    # Central configuration (all paths, LLM, RL settings)
├── inject_violations.py         # Inject test violations into a KG
├── report_quality.py            # Standalone quality report script
├── generate_thesis_diagrams.py  # Generates architecture diagrams (outputs/diagrams/)
├── requirements.txt
│
├── data/01_corpus/              # Input corpus
│   ├── corpus.csv               # Full automotive corpus
│   └── sample_corpus.csv        # Small sample for quick testing
│
├── ontology/                    # Base OWL ontologies
│   └── LLM_GENIALOntBFO_cleaned.owl   # Default ontology used for alignment
│
├── config/                      # Manual mapping override files
│   ├── entity_map.csv           # Text -> URI mappings for entities
│   └── relation_map.csv         # Text -> URI mappings for predicates
│
├── core/                        # PipelineStep abstract base class
├── alignment/                   # Ontology index builder for fuzzy matching
├── llm_kg/                      # LLM triple extraction (OpenAI + Ollama)
├── Scripts/                     # Individual pipeline step implementations
├── qa/                          # Repair candidate generation and fix application
├── reasoning/                   # Reasoner wrappers (Konclude, HermiT), axiom extractor
│   └── Konclude/                # Bundled Konclude binary and configs (Windows)
│
└── rl/                          # Full RL system
    ├── train_repair.py          # Training loop with 3-phase curriculum
    ├── dqn_agent.py             # Double DQN agent with action masking
    ├── dqn_model.py             # PyTorch neural network model
    ├── env_repair.py            # OpenAI Gym-compatible repair environment
    ├── replay_buffer.py         # Prioritized Experience Replay (PER)
    ├── reward_functions.py      # Reward shaping logic
    ├── human_loop.py            # Human-in-the-loop interface
    ├── diff_tracker.py          # Tracks graph changes per repair step
    └── triple_display.py        # Terminal UI for violations and actions
```

---

## Quick Start — Sample Data, No API Key

To test the full pipeline on a small sample using local Ollama (no OpenAI key needed):

```bash
# 1. Pull the model
ollama pull llama3

# 2. Set in config.py:  USE_OPENAI = False

# 3. Run the pipeline on the sample corpus
python pipeline.py run-all --input data/01_corpus/sample_corpus.csv

# 4. Inject violations to test the RL agent
python inject_violations.py

# 5. Run RL repair in fully automated mode
python pipeline.py repair-kg --no-human
```
