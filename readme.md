# KGG4SE-RL: Knowledge Graph Generation for Software Engineering with Reinforcement Learning

An end-to-end pipeline that:
1. Extracts RDF triples from an automotive text corpus using an LLM (GPT-4o-mini or Llama 3)
2. Aligns triples to an OWL ontology using fuzzy matching
3. Builds a merged Knowledge Graph (KG)
4. Checks OWL consistency using a reasoner (Konclude or HermiT)
5. Repairs inconsistencies automatically using a Deep Q-Network (DQN) reinforcement learning agent with a human-in-the-loop curriculum

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.12+ | |
| Java | 8+ | Required only if using HermiT reasoner |
| OpenAI API key | — | Required if `USE_OPENAI = True` in `config.py` |
| Ollama | latest | Required if using local Llama 3 instead of OpenAI |

**Konclude** (the default reasoner) is bundled at `reasoning/Konclude/Binaries/Konclude.exe` — no installation needed on Windows.

---

## Installation

```bash
pip install -r requirements.txt
```

PyTorch is required for the RL repair agent. Install it separately if the above does not pull it automatically:

```bash
# CPU only
pip install torch

# With CUDA GPU support — see https://pytorch.org/get-started/locally/
```

---

## Configuration

All settings live in `config.py`. The most important ones to check before running:

```python
# --- LLM backend ---
USE_OPENAI   = True            # False-> use local Ollama instead
OPENAI_MODEL = "gpt-4o-mini"
MODEL_NAME   = "llama3"        # Ollama model name (used when USE_OPENAI = False)
OLLAMA_URL   = "http://localhost:11434/api/generate"

# --- Reasoner ---
DEFAULT_REASONER = "konclude"  # or "hermit"

# --- RL training ---
RL_EPISODES                  = 50
RL_MAX_STEPS_PER_EPISODE     = 15
RL_HUMAN_SUPERVISED_EPISODES = 3   # Phase 1: human decides all actions
RL_HUMAN_FIRST_EPISODES      = 5   # Phase 2: human seeds first action only
```

### OpenAI API key

Set the key as an environment variable before running:

```bash
# Linux / macOS
export OPENAI_API_KEY="sk-..."

# Windows (Command Prompt)
set OPENAI_API_KEY=sk-...

# Windows (PowerShell)
$env:OPENAI_API_KEY="sk-..."
```

### Using local Ollama instead

```bash
# Pull the model once
ollama pull llama3

# Then set in config.py:
USE_OPENAI = False
```

---

## Pipeline Steps

### Run the full pipeline (recommended)

```bash
python pipeline.py run-all
```

This executes all 8 steps in order using the defaults in `config.py`.

```bash
# With custom starting corpus and settings
python pipeline.py run-all \
  --input        data/01_corpus/corpus.csv \
  --model        llama3 \
  --fuzzy-cutoff 85
```

---

### Step 1 — `prepare-corpus`

Converts a CSV file (requires a `title` column) to a plain-text corpus.

```bash
python pipeline.py prepare-corpus

# Custom paths
python pipeline.py prepare-corpus \
  --input       data/01_corpus/corpus.csv \
  --output      data/01_corpus/corpus.txt \
  --text-column title
```

---

### Step 2 — `generate-triples`

Sends corpus text to the LLM and extracts `Subject | Predicate | Object` triples.

```bash
python pipeline.py generate-triples

# Custom paths
python pipeline.py generate-triples \
  --input   data/01_corpus/corpus.txt \
  --output  outputs/intermediate/corpus_triples.json \
  --model   llama3 \
  --timeout 120
```

---

### Step 3 — `clean-triples`

Normalizes, deduplicates, and filters raw triples.

```bash
python pipeline.py clean-triples

# Custom paths
python pipeline.py clean-triples \
  --input      outputs/intermediate/corpus_triples.json \
  --output     outputs/intermediate/corpus_triples_cleaned.json \
  --max-tokens 5
```

---

### Step 4 — `align-triples`

Maps cleaned triples to ontology URIs using fuzzy string matching. Outputs a `.ttl` file and an alignment report CSV.

```bash
python pipeline.py align-triples

# Custom paths
python pipeline.py align-triples \
  --input        outputs/intermediate/corpus_triples_cleaned.json \
  --output       outputs/intermediate/aligned_triples.ttl \
  --ontology     ontology/LLM_GENIALOntBFO_cleaned.owl \
  --fuzzy-cutoff 85 \
  --allow-create
```

---

### Step 5 — `build-kg`

Merges the base ontology with aligned triples into a single KG (`.owl` + `.ttl`).

```bash
python pipeline.py build-kg

# Custom paths
python pipeline.py build-kg \
  --input      outputs/intermediate/aligned_triples.ttl \
  --ontology   ontology/LLM_GENIALOntBFO_cleaned.owl \
  --output-owl outputs/intermediate/merged_kg.owl \
  --output-ttl outputs/intermediate/merged_kg.ttl
```

---

### Step 6 — `run-reasoner`

Runs an OWL reasoner (Konclude or HermiT) on the merged KG to detect inconsistencies.

```bash
python pipeline.py run-reasoner

# Custom paths
python pipeline.py run-reasoner \
  --input  outputs/intermediate/merged_kg.owl \
  --output outputs/knowledge_graph/merged_kg_reasoned.owl
```

Switch reasoner at runtime:

```bash
python pipeline.py run-reasoner --reasoner hermit
```

---

### Step 7 — `parse-reasoner`

Parses the reasoned OWL file into a flat JSON triples list.

```bash
python pipeline.py parse-reasoner

# Custom paths
python pipeline.py parse-reasoner \
  --input  outputs/knowledge_graph/merged_kg_reasoned.owl \
  --output outputs/intermediate/reasoned_triples.json
```

---

### Step 8 — `check-quality`

Generates a quality report with statistics, duplicates, and top-N frequencies.

```bash
python pipeline.py check-quality

# Custom paths
python pipeline.py check-quality \
  --input       outputs/intermediate/reasoned_triples.json \
  --output      outputs/quality/quality_report.json \
  --sample-size 20 \
  --top-n       10
```

---

### Step 9 — `repair-kg` (RL agent)

Trains a Double DQN agent to automatically repair OWL inconsistencies detected by the reasoner. Uses a 3-phase curriculum:

- **Phase 1** (`RL_HUMAN_SUPERVISED_EPISODES`): Human selects every repair action
- **Phase 2** (`RL_HUMAN_FIRST_EPISODES`): Human selects the first action per episode, agent handles the rest
- **Phase 3**: Fully autonomous RL agent with human fallback

```bash
python pipeline.py repair-kg

# Skip human-in-the-loop (fully automated, for testing)
python pipeline.py repair-kg --no-human

# Or invoke the training module directly
python -m rl.train_repair --episodes 50 --max-steps 15 --no-human
```

Trained model checkpoints are saved to `outputs/models/`. Repair traces are written to `outputs/rl_repair_traces/repair_trace.jsonl`.

---

## Logging

```bash
python pipeline.py --log-level DEBUG run-all
python pipeline.py --log-level WARNING check-quality
```

Default log level is `INFO`.

---

## Output Structure

```
outputs/
├── intermediate/        # Step-by-step artifacts (triples, aligned TTL, merged KG)
├── knowledge_graph/     # Final reasoned KG (.owl)
├── quality/             # Quality report (JSON)
├── reports/             # Alignment CSV report and plots
├── reasoning/           # Raw reasoner outputs (OWL, logs, stats)
├── models/              # Trained DQN checkpoints (.pt)
└── rl_repair_traces/    # Training trace log (JSONL)
```

---

## Project Structure

```
KGG4SE_RL/
├── pipeline.py          # Main CLI entry point
├── config.py            # Central configuration
├── requirements.txt
├── data/
│   └── 01_corpus/       # Input corpus (CSV and TXT)
├── ontology/            # Base OWL ontologies
├── config/              # Entity and relation mapping CSVs
├── core/                # PipelineStep base class
├── alignment/           # Ontology index and fuzzy matching
├── llm_kg/              # LLM triple extraction (OpenAI / Ollama)
├── qa/                  # Repair candidate generation and fix application
├── reasoning/           # Reasoner wrappers (Konclude, HermiT), axiom extractor
│   └── Konclude/        # Bundled Konclude binary (Windows)
└── rl/                  # DQN agent, training loop, RL environment, replay buffer
```

---

## Quick Start (sample data, no API key needed)

To test the pipeline on the small sample corpus using local Ollama:

```bash
# 1. Set config.py: USE_OPENAI = False
# 2. Start Ollama and pull the model
ollama pull llama3

# 3. Run the full pipeline on sample data
python pipeline.py run-all --input data/01_corpus/sample_corpus.csv
```
