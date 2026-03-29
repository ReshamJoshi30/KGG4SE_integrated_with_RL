# config.py
"""
Central configuration for the KG pipeline.
All paths, LLM settings, and pipeline defaults are here.
Import this module anywhere — it behaves as a singleton (module-level state).
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Project root  (config.py sits at project root)
# ---------------------------------------------------------------------------
ROOT: Path = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------
DATA_DIR = ROOT / "data" / "01_corpus"
ONTOLOGY_DIR = ROOT / "ontology"
OUTPUTS_INTER = ROOT / "outputs" / "intermediate"
OUTPUTS_KG = ROOT / "outputs" / "knowledge_graph"
OUTPUTS_QUALITY = ROOT / "outputs" / "quality"
OUTPUTS_REPORTS = ROOT / "outputs" / "reports"
CONFIG_DIR = ROOT / "config"

# Ensure output dirs exist at import time (safe to call multiple times)
for _d in (OUTPUTS_INTER, OUTPUTS_KG, OUTPUTS_QUALITY, OUTPUTS_REPORTS):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# LLM / Ollama settings
# ---------------------------------------------------------------------------
OLLAMA_URL:    str = "http://localhost:11434/api/generate"
MODEL_NAME:    str = "llama3"
LLM_TIMEOUT:   int = 120          # seconds
LLM_MAX_CHARS: int = 3_000        # safety crop before sending to LLM
LLM_MAX_CHUNKS: int = 10          # 0 = process all chunks; >0 = cap for testing

# ---------------------------------------------------------------------------
# OpenAI API settings  (alternative to local Ollama)
# ---------------------------------------------------------------------------
# Set USE_OPENAI = True to route triple generation through the OpenAI API
# instead of a local Ollama instance.
#
# API key: set the environment variable before running any pipeline step:
#   Windows CMD :  set OPENAI_API_KEY=sk-...
#   PowerShell  :  $env:OPENAI_API_KEY="sk-..."
#   Linux/Mac   :  export OPENAI_API_KEY=sk-...
#
# Model recommendation:
#   "gpt-4o-mini" — best value: cheap, fast, much better than llama3 locally.
#                   ~$0.15 per 1M input tokens.  Recommended for thesis runs.
#   "gpt-4o"      — highest accuracy, ~20x more expensive than mini.
#                   Use for final evaluation / demo runs.
#
# Why OpenAI improves accuracy:
#   llama3 (local) often extracts generic/noisy triples and ignores domain
#   constraints.  GPT-4o-mini follows the structured prompt strictly,
#   producing cleaner Subject|Predicate|Object lines with fewer garbage
#   triples, which means fewer ontology violations in merged_kg.owl and
#   a smaller, more tractable repair problem for the RL agent.
import os as _os
USE_OPENAI:    bool = True                     # flip to True to use OpenAI
OPENAI_MODEL:  str  = "gpt-4o-mini"           # or "gpt-4o" for max quality
OPENAI_API_KEY: str = _os.getenv("OPENAI_API_KEY", "")  # read from env var

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------
PROMPT_TEMPLATE: str = """
Extract technical relationships from automotive electrical systems text.

OUTPUT FORMAT - CRITICAL RULES:
- Output ONLY triples, nothing else
- NO numbering (1., 2., etc.)
- NO explanatory text
- NO headers or titles
- ONE triple per line
- Format: Subject | Predicate | Object

EXTRACT:
- Sensor measurements (temperature_sensor | measures | temperature)
- Control relationships (ecu | controls | actuator)
- Signal transmission (can_bus | transmits | data)
- Component connections (wiring | connects | ecu)
- Power delivery (battery | powers | system)

FOCUS ON:
- Sensors: temperature, pressure, speed, position, proximity sensors
- Actuators: motors, valves, solenoids, relays
- Controllers: ecu, dcu, bcm, tcm
- Signals: can_bus, lin_bus, flexray
- Components: wiring, connectors, batteries

AVOID:
- Market data, forecasts, revenue, statistics
- Vendor names, country names, industry reports

Text:
{TEXT}

Triples (output ONLY the triples below, no other text):
"""

# ---------------------------------------------------------------------------
# Garbage / noise filter terms  (used by generate_triples step)
# ---------------------------------------------------------------------------
GARBAGE_TERMS: set = {
    "market", "forecast", "report", "growth", "analysis", "share",
    "usd", "revenue", "size", "opportunities", "cagr", "billion",
    "million", "percent", "rate", "reach", "industry", "outlook", "studies",
    "vendor", "country", "insight", "knowledge", "statistics", "products",
    "trends", "future", "advances", "indicates",
}

# ---------------------------------------------------------------------------
# Cleaning settings  (used by clean_triples step)
# ---------------------------------------------------------------------------
CLEAN_MAX_TOKENS: int = 5          # max underscore-separated tokens in an entity
CLEAN_ALIASES: dict = {
    "central_processing_unit": "cpu",
    "random_access_memory":    "ram",
    "device_driver":           "driver",
    "operating_system":        "operating_system",
}

# ---------------------------------------------------------------------------
# Alignment settings  (used by align_triples step)
# ---------------------------------------------------------------------------
FUZZY_CUTOFF:             int = 85
ALLOW_CREATE_INDIVIDUALS: bool = True  # FIXED: Enable individual creation
BASE_INDIVIDUAL_NS:       str = "http://cpsagila.cs.uni-kl.de/GENIALOnt/ind/"

# ---------------------------------------------------------------------------
# Default file paths  (each step reads/writes these unless overridden via CLI)
# ---------------------------------------------------------------------------
DEFAULT_PATHS: dict = {
    # step            input                                   output
    "prepare_corpus": {
        "input":  DATA_DIR / "corpus.csv",
        "output": DATA_DIR / "corpus.txt",
    },
    "generate_triples": {
        "input":  DATA_DIR / "corpus.txt",
        "output": OUTPUTS_INTER / "corpus_triples.json",
    },
    "clean_triples": {
        "input":  OUTPUTS_INTER / "corpus_triples.json",
        "output": OUTPUTS_INTER / "corpus_triples_cleaned.json",
    },
    "align_triples": {
        "input":       OUTPUTS_INTER / "corpus_triples_cleaned.json",
        "output":      OUTPUTS_INTER / "sample_corpus_aligned_triples.ttl",
        "ontology":    ONTOLOGY_DIR / "LLM_GENIALOntBFO_cleaned.owl",
        "rel_map":     CONFIG_DIR / "relation_map.csv",
        "ent_map":     CONFIG_DIR / "entity_map.csv",
        "report":      OUTPUTS_REPORTS / "alignment_report.csv",
    },
    "build_kg": {
        "input":    OUTPUTS_INTER / "sample_corpus_aligned_triples.ttl",
        "ontology": ONTOLOGY_DIR / "LLM_GENIALOntBFO_cleaned.owl",
        "output_owl": OUTPUTS_INTER / "merged_kg.owl",
        "output_ttl": OUTPUTS_INTER / "merged_kg.ttl",
    },
    "run_reasoner": {
        "input":  OUTPUTS_INTER / "merged_kg.owl",
        "output": OUTPUTS_KG / "merged_kg_reasoned.owl",
    },
    "parse_reasoner": {
        "input":  OUTPUTS_KG / "merged_kg_reasoned.owl",
        "output": OUTPUTS_INTER / "reasoned_triples.json",
    },
    "check_quality": {
        "input":  OUTPUTS_INTER / "reasoned_triples.json",
        "output": OUTPUTS_QUALITY / "quality_report.json",
    },
}

# ---------------------------------------------------------------------------
# Fallback triples  (returned when LLM produces no usable output)
# ---------------------------------------------------------------------------
FALLBACK_TRIPLE: dict = {
    "subject":   "TemperatureSensor",
    "predicate": "hasComponent",
    "object":    "Thermistor",
}

# ---------------------------------------------------------------------------
# Reasoner settings
# ---------------------------------------------------------------------------
DEFAULT_REASONER: str = "konclude"      # "konclude" or "hermit"
KONCLUDE_DIR:     Path = ROOT / "reasoning" / "konclude"
REASONER_OUT_DIR: Path = ROOT / "outputs" / "reasoning"

# Create reasoner output dir at import time
REASONER_OUT_DIR.mkdir(parents=True, exist_ok=True)

# # ===========================================================================
# # RL REPAIR SETTINGS
# # ===========================================================================

# # ---------------------------------------------------------------------------
# # RL Repair directories
# # ---------------------------------------------------------------------------
# RL_REPAIR_STEPS_DIR = ROOT / "outputs" / "rl_repair_steps"
# RL_REPAIR_TRACES_DIR = ROOT / "outputs" / "rl_repair_traces"
# RL_MODELS_DIR = ROOT / "outputs" / "models"

# # Create RL dirs at import time
# for _d in (RL_REPAIR_STEPS_DIR, RL_REPAIR_TRACES_DIR, RL_MODELS_DIR):
#     _d.mkdir(parents=True, exist_ok=True)

# # ---------------------------------------------------------------------------
# # RL training hyperparameters
# # ---------------------------------------------------------------------------
# RL_EPISODES = 50                    # Number of training episodes
#                                     # (20 was too few — DQN needs 50-100+ to converge)
# RL_MAX_STEPS_PER_EPISODE = 15       # Max repair steps per episode
# RL_BATCH_SIZE = 32                  # DQN training batch size
# RL_TARGET_UPDATE_INTERVAL = 5       # Update target network every N episodes
# RL_INTERACTIVE = False              # Enable human-in-the-loop by default

# # ---------------------------------------------------------------------------
# # DQN epsilon-greedy decay
# # ---------------------------------------------------------------------------
# # epsilon_decay is applied PER STEP (not per episode) — see train_repair.py.
# # Target: decay from 1.0 to epsilon_min≈0.05 over ~50 episodes × 10 avg steps
# # = ~500 steps.  Formula: decay = 0.05^(1/500) ≈ 0.994.
# #
# # Old value (0.995 per-episode over 20 eps) left epsilon≈0.90 at end of
# # training — agent was 90% random throughout and never exploited its learning.
# RL_EPSILON_DECAY: float = 0.994    # per-step decay (set in DQN_Agent constructor)

# # ---------------------------------------------------------------------------
# # Human-in-the-loop phased training
# # ---------------------------------------------------------------------------
# # Phase 1 — HUMAN_SUPERVISED: human decides every action for N episodes
# #   RL observes and stores transitions; good for cold-start bootstrapping
# # Phase 2 — HUMAN_FIRST: human decides only the first action of each episode
# #   RL handles all subsequent steps; blends human seed with RL autonomy
# # Phase 3 — RL_AUTOMATED: fully autonomous; human only asked when
# #   RL confidence (Q-value margin) falls below the threshold
# #
# # Total episodes consumed by phases 1+2 must be < RL_EPISODES.
# # With 50 total episodes: 3+5=8 human episodes, 42 automated — gives RL
# # enough room to converge.  Old split (5+10 of 20) left only 5 automated.
# RL_TRAINING_MODE: str = "human_supervised"  # starting mode (ignored if phases used)
# RL_HUMAN_SUPERVISED_EPISODES: int = 3       # Phase 1 episode count
# RL_HUMAN_FIRST_EPISODES: int = 5            # Phase 2 episode count (episodes 4–8)
# RL_AUTO_CONFIDENCE_THRESHOLD: float = 0.65  # Phase 3: ask human below this margin

# # Reward boost added when human-chosen transitions are stored in the replay
# # buffer a second time.  Raises effective sampling priority of expert moves.
# RL_HUMAN_REPLAY_BOOST: float = 2.0

# # ---------------------------------------------------------------------------
# # Add repair-kg to DEFAULT_PATHS
# # ---------------------------------------------------------------------------
# DEFAULT_PATHS["repair_kg"] = {
#     "input":     OUTPUTS_INTER / "merged_kg.owl",
#     "ontology":  ONTOLOGY_DIR / "LLM_GENIALOntBFO_cleaned.owl",
#     "output":    RL_MODELS_DIR / "dqn_repair_final.pt",
#     "traces":    RL_REPAIR_TRACES_DIR,
# }

# # ---------------------------------------------------------------------------
# # Reproducibility — global random seed
# # ---------------------------------------------------------------------------
# RANDOM_SEED: int = 42


# def set_global_seed(seed: int | None = None) -> None:
#     """
#     Seed all random number generators for reproducible results.

#     Seeds: stdlib random, numpy, and torch (CPU + CUDA).
#     Call once at the start of any entry point (pipeline.py, train, test).

#     Args:
#         seed: Seed value.  Defaults to ``RANDOM_SEED`` from this module.
#     """
#     import random as _random

#     if seed is None:
#         seed = RANDOM_SEED

#     _random.seed(seed)

#     try:
#         import numpy as _np
#         _np.random.seed(seed)
#     except ImportError:
#         pass

#     try:
#         import torch as _torch
#         _torch.manual_seed(seed)
#         if _torch.cuda.is_available():
#             _torch.cuda.manual_seed_all(seed)
#             _torch.backends.cudnn.deterministic = True
#             _torch.backends.cudnn.benchmark = False
#     except ImportError:
#         pass
# ============================================================
# PASTE THIS BLOCK INTO config.py
# Replace everything from "# RL REPAIR SETTINGS" to the end
# ============================================================

# ===========================================================================
# RL REPAIR SETTINGS
# ===========================================================================

# ---------------------------------------------------------------------------
# RL Repair directories
# ---------------------------------------------------------------------------
RL_REPAIR_STEPS_DIR = ROOT / "outputs" / "rl_repair_steps"
RL_REPAIR_TRACES_DIR = ROOT / "outputs" / "rl_repair_traces"
RL_MODELS_DIR = ROOT / "outputs" / "models"

for _d in (RL_REPAIR_STEPS_DIR, RL_REPAIR_TRACES_DIR, RL_MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# RL training hyperparameters
# ---------------------------------------------------------------------------
RL_EPISODES = 100                   # Increased: benchmark KG needs more episodes
RL_MAX_STEPS_PER_EPISODE = 20       # Increased: 8 violation types need more steps
RL_BATCH_SIZE = 32
RL_TARGET_UPDATE_INTERVAL = 5
RL_INTERACTIVE = False

# ---------------------------------------------------------------------------
# DQN epsilon-greedy decay
# ---------------------------------------------------------------------------
# Calibration: benchmark KG produces ~8-12 steps per episode.
# 100 episodes × 10 avg steps = ~1000 steps total.
# Target: decay from 1.0-> 0.05 over 1000 steps.
# Formula: 0.05^(1/1000) ≈ 0.997
#
# Old value (0.994) was calibrated for 500 steps — but simple KG only
# produced 150 steps total, leaving epsilon at 0.39 (39% random).
# New value ensures full explore→exploit curve is visible in plots.
RL_EPSILON_DECAY: float = 0.997    # per-step decay

# ---------------------------------------------------------------------------
# Human-in-the-loop phased training
# ---------------------------------------------------------------------------
RL_TRAINING_MODE: str = "human_supervised"
RL_HUMAN_SUPERVISED_EPISODES: int = 0   # Set to 0 for --no-human runs
RL_HUMAN_FIRST_EPISODES: int = 0
RL_AUTO_CONFIDENCE_THRESHOLD: float = 0.65
RL_HUMAN_REPLAY_BOOST: float = 2.0

# ---------------------------------------------------------------------------
# Benchmark KG path (injected violations)
# ---------------------------------------------------------------------------
RL_BENCHMARK_KG = ROOT / "outputs" / "intermediate" / "merged_kg_broken.owl"

# ---------------------------------------------------------------------------
# Add repair-kg to DEFAULT_PATHS
# ---------------------------------------------------------------------------
DEFAULT_PATHS["repair_kg"] = {
    "input":     RL_BENCHMARK_KG,           # CHANGED: points to broken KG
    "ontology":  ONTOLOGY_DIR / "LLM_GENIALOntBFO_cleaned.owl",
    "output":    RL_MODELS_DIR / "dqn_repair_final.pt",
    "traces":    RL_REPAIR_TRACES_DIR,
}

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED: int = 42

def set_global_seed(seed: int | None = None) -> None:
    import random as _random
    if seed is None:
        seed = RANDOM_SEED
    _random.seed(seed)
    try:
        import numpy as _np
        _np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch as _torch
        _torch.manual_seed(seed)
        if _torch.cuda.is_available():
            _torch.cuda.manual_seed_all(seed)
            _torch.backends.cudnn.deterministic = True
            _torch.backends.cudnn.benchmark = False
    except ImportError:
        pass