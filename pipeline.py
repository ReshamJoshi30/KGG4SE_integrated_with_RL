# pipeline.py
"""
Unified CLI entry point for the KG pipeline.

Usage examples
--------------
# Run individual steps
python pipeline.py prepare-corpus --input data/01_corpus/corpus.csv
python pipeline.py generate-triples --input data/01_corpus/corpus.txt
python pipeline.py clean-triples
python pipeline.py align-triples --fuzzy-cutoff 90
python pipeline.py build-kg
python pipeline.py run-reasoner
python pipeline.py parse-reasoner
python pipeline.py check-quality --sample-size 20

# RL-based KG repair
python pipeline.py repair-kg --episodes 20 --max-steps 20
python pipeline.py repair-kg --interactive  # Human-in-the-loop mode

# Run full pipeline end-to-end
python pipeline.py run-all --input data/01_corpus/corpus.csv

# Override any input/output path
python pipeline.py clean-triples \\
    --input  outputs/intermediate/my_triples.json \\
    --output outputs/intermediate/my_triples_cleaned.json

# Change LLM model
python pipeline.py generate-triples --model mistral

# Enable individual minting during alignment
python pipeline.py align-triples --allow-create
"""

from core.base_step import PipelineStep  # For RepairKGStep
from Scripts.check_quality import CheckQualityStep
from Scripts.parse_reasoner_output import ParseReasonerOutputStep
from Scripts.run_reasoner import RunReasonerStep
from Scripts.build_kg import BuildKGStep
from Scripts.align_triples import AlignTriplesStep
from Scripts.clean_triples import CleanTriplesStep
from llm_kg.generate_triples import GenerateTriplesStep
from Scripts.prepare_corpus import PrepareCorpusStep
import argparse
import logging
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Logging — configured before any pipeline imports so all modules
# inherit the same handler/formatter
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("pipeline")

# ---------------------------------------------------------------------------
# Step imports  (after logging is configured)
# ---------------------------------------------------------------------------
import config  # noqa: E402  (must come after logging setup)

# RL repair step


# ---------------------------------------------------------------------------
# RL Repair Step Definition
# ---------------------------------------------------------------------------
class RepairKGStep(PipelineStep):
    """
    Train RL agent to repair KG inconsistencies.

    Uses DQN agent to learn repair actions from reasoner feedback.
    """

    def run(self, **kwargs):
        """
        Execute RL-based KG repair training.

        Args:
            base_owl: Path to OWL file to repair
            ontology_path: Path to base ontology
            episodes: Number of training episodes
            max_steps_per_episode: Max steps per episode
            reasoner: Which reasoner to use
            interactive: Enable human-in-the-loop
            batch_size: DQN training batch size
            target_update_interval: Target network update frequency
            save_dir: Where to save trained models

        Returns:
            Path to saved model
        """
        from rl.train_repair import train_repair

        logger.info("Starting RL-based KG repair training")
        logger.info("Base OWL: %s", kwargs.get("base_owl"))
        logger.info("Episodes: %s", kwargs.get("episodes"))
        logger.info("Reasoner: %s", kwargs.get("reasoner"))

        agent, episode_rewards, episode_successes = train_repair(**kwargs)

        success_rate = sum(episode_successes) / len(episode_successes) * 100
        avg_reward = sum(episode_rewards) / len(episode_rewards)

        logger.info("=" * 60)
        logger.info("RL REPAIR TRAINING COMPLETE")
        logger.info("=" * 60)
        logger.info("Success rate: %.1f%% (%d/%d episodes)",
                    success_rate, sum(episode_successes), len(episode_successes))
        logger.info("Average reward: %.2f", avg_reward)
        logger.info("Model saved to: %s", kwargs["save_dir"])

        return Path(kwargs["save_dir"]) / "dqn_repair_final.pt"


# ---------------------------------------------------------------------------
# Step registry  (Chain of Responsibility — ordered for run-all)
# ---------------------------------------------------------------------------
STEP_REGISTRY: dict[str, Any] = {
    "prepare-corpus":   PrepareCorpusStep,
    "generate-triples": GenerateTriplesStep,
    "clean-triples":    CleanTriplesStep,
    "align-triples":    AlignTriplesStep,
    "build-kg":         BuildKGStep,
    "run-reasoner":     RunReasonerStep,
    "parse-reasoner":   ParseReasonerOutputStep,
    "check-quality":    CheckQualityStep,
    "repair-kg":        RepairKGStep,  # RL-based repair
}

# Full pipeline execution order
PIPELINE_ORDER: list[str] = [
    "prepare-corpus",
    "generate-triples",
    "clean-triples",
    "align-triples",
    "build-kg",
    "run-reasoner",
    "parse-reasoner",
    "check-quality",
]


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """
    Build the top-level argument parser with one subcommand per step
    plus a ``run-all`` orchestration command.

    Returns:
        Configured ArgumentParser.
    """
    parser = argparse.ArgumentParser(
        prog="pipeline",
        description="KG Pipeline — build a knowledge graph from a text corpus.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )

    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    subparsers.required = True

    # ------------------------------------------------------------------ #
    # prepare-corpus
    # ------------------------------------------------------------------ #
    p_corpus = subparsers.add_parser(
        "prepare-corpus",
        help="Convert a CSV corpus to plain text.",
    )
    p_corpus.add_argument(
        "--input",  type=Path,
        default=config.DEFAULT_PATHS["prepare_corpus"]["input"],
        help="Source CSV file (must contain --text-column column).",
    )
    p_corpus.add_argument(
        "--output", type=Path,
        default=config.DEFAULT_PATHS["prepare_corpus"]["output"],
        help="Destination .txt file.",
    )
    p_corpus.add_argument(
        "--text-column", default="title",
        help="CSV column to extract text from (default: title).",
    )

    # ------------------------------------------------------------------ #
    # generate-triples
    # ------------------------------------------------------------------ #
    p_gen = subparsers.add_parser(
        "generate-triples",
        help="Extract S|P|O triples from corpus using Ollama LLM.",
    )
    p_gen.add_argument(
        "--input",  type=Path,
        default=config.DEFAULT_PATHS["generate_triples"]["input"],
        help="Corpus .txt file.",
    )
    p_gen.add_argument(
        "--output", type=Path,
        default=config.DEFAULT_PATHS["generate_triples"]["output"],
        help="Output triples .json file.",
    )
    p_gen.add_argument(
        "--model", default=config.MODEL_NAME,
        help=f"Ollama model name (default: {config.MODEL_NAME}).",
    )
    p_gen.add_argument(
        "--ollama-url", default=config.OLLAMA_URL,
        help=f"Ollama API endpoint (default: {config.OLLAMA_URL}).",
    )
    p_gen.add_argument(
        "--timeout", type=int, default=config.LLM_TIMEOUT,
        help=f"Request timeout in seconds (default: {config.LLM_TIMEOUT}).",
    )
    p_gen.add_argument(
        "--max-chars", type=int, default=config.LLM_MAX_CHARS,
        help=f"Max corpus chars sent to LLM (default: {config.LLM_MAX_CHARS}).",
    )
    p_gen.add_argument(
        "--max-chunks", type=int, default=config.LLM_MAX_CHUNKS,
        help="Max chunks to process (0 = all). Use e.g. 10 for quick tests.",
    )

    # ------------------------------------------------------------------ #
    # clean-triples
    # ------------------------------------------------------------------ #
    p_clean = subparsers.add_parser(
        "clean-triples",
        help="Normalize, deduplicate, and filter raw triples.",
    )
    p_clean.add_argument(
        "--input",  type=Path,
        default=config.DEFAULT_PATHS["clean_triples"]["input"],
        help="Raw triples .json file.",
    )
    p_clean.add_argument(
        "--output", type=Path,
        default=config.DEFAULT_PATHS["clean_triples"]["output"],
        help="Cleaned triples .json file.",
    )
    p_clean.add_argument(
        "--max-tokens", type=int, default=config.CLEAN_MAX_TOKENS,
        help=f"Max entity token length (default: {config.CLEAN_MAX_TOKENS}).",
    )

    # ------------------------------------------------------------------ #
    # align-triples
    # ------------------------------------------------------------------ #
    p_align = subparsers.add_parser(
        "align-triples",
        help="Map cleaned triples to ontology URIs.",
    )
    p_align.add_argument(
        "--input",  type=Path,
        default=config.DEFAULT_PATHS["align_triples"]["input"],
        help="Cleaned triples .json file.",
    )
    p_align.add_argument(
        "--output", type=Path,
        default=config.DEFAULT_PATHS["align_triples"]["output"],
        help="Aligned triples .ttl file.",
    )
    p_align.add_argument(
        "--ontology", type=Path,
        default=config.DEFAULT_PATHS["align_triples"]["ontology"],
        help="OWL ontology file.",
    )
    p_align.add_argument(
        "--rel-map", type=Path,
        default=config.DEFAULT_PATHS["align_triples"]["rel_map"],
        help="Predicate CSV map file.",
    )
    p_align.add_argument(
        "--ent-map", type=Path,
        default=config.DEFAULT_PATHS["align_triples"]["ent_map"],
        help="Entity CSV map file.",
    )
    p_align.add_argument(
        "--report", type=Path,
        default=config.DEFAULT_PATHS["align_triples"]["report"],
        help="Alignment report CSV file.",
    )
    p_align.add_argument(
        "--fuzzy-cutoff", type=int, default=config.FUZZY_CUTOFF,
        help=f"Fuzzy match threshold 0-100 (default: {config.FUZZY_CUTOFF}).",
    )
    p_align.add_argument(
        "--allow-create", action="store_true",
        default=config.ALLOW_CREATE_INDIVIDUALS,
        help="Mint new URIs for unmatched entities.",
    )

    # ------------------------------------------------------------------ #
    # build-kg
    # ------------------------------------------------------------------ #
    p_build = subparsers.add_parser(
        "build-kg",
        help="Merge base ontology with aligned triples into KG.",
    )
    p_build.add_argument(
        "--input",  type=Path,
        default=config.DEFAULT_PATHS["build_kg"]["input"],
        help="Aligned triples .ttl file.",
    )
    p_build.add_argument(
        "--ontology", type=Path,
        default=config.DEFAULT_PATHS["build_kg"]["ontology"],
        help="Base OWL ontology file.",
    )
    p_build.add_argument(
        "--output-owl", type=Path,
        default=config.DEFAULT_PATHS["build_kg"]["output_owl"],
        help="Output merged KG .owl file.",
    )
    p_build.add_argument(
        "--output-ttl", type=Path,
        default=config.DEFAULT_PATHS["build_kg"]["output_ttl"],
        help="Output merged KG .ttl file.",
    )

    # ------------------------------------------------------------------ #
    # run-reasoner
    # ------------------------------------------------------------------ #
    p_reason = subparsers.add_parser(
        "run-reasoner",
        help="Run HermiT reasoner on merged KG.",
    )
    p_reason.add_argument(
        "--input",  type=Path,
        default=config.DEFAULT_PATHS["run_reasoner"]["input"],
        help="Merged KG .owl file.",
    )
    p_reason.add_argument(
        "--output", type=Path,
        default=config.DEFAULT_PATHS["run_reasoner"]["output"],
        help="Reasoned KG .owl file.",
    )

    # ------------------------------------------------------------------ #
    # parse-reasoner
    # ------------------------------------------------------------------ #
    p_parse = subparsers.add_parser(
        "parse-reasoner",
        help="Parse reasoned OWL into flat triples JSON.",
    )
    p_parse.add_argument(
        "--input",  type=Path,
        default=config.DEFAULT_PATHS["parse_reasoner"]["input"],
        help="Reasoned KG .owl file.",
    )
    p_parse.add_argument(
        "--output", type=Path,
        default=config.DEFAULT_PATHS["parse_reasoner"]["output"],
        help="Output triples .json file.",
    )

    # ------------------------------------------------------------------ #
    # check-quality
    # ------------------------------------------------------------------ #
    p_quality = subparsers.add_parser(
        "check-quality",
        help="Generate quality report from reasoned triples.",
    )
    p_quality.add_argument(
        "--input",  type=Path,
        default=config.DEFAULT_PATHS["check_quality"]["input"],
        help="Reasoned triples .json file.",
    )
    p_quality.add_argument(
        "--output", type=Path,
        default=config.DEFAULT_PATHS["check_quality"]["output"],
        help="Quality report .json file.",
    )
    p_quality.add_argument(
        "--sample-size", type=int, default=10,
        help="Number of sample triples in report (default: 10).",
    )
    p_quality.add_argument(
        "--top-n", type=int, default=10,
        help="Top-N frequency entries per field (default: 10).",
    )

    # ------------------------------------------------------------------ #
    # repair-kg (NEW: RL-based KG repair)
    # ------------------------------------------------------------------ #
    p_repair = subparsers.add_parser(
        "repair-kg",
        help="Train RL agent to repair KG inconsistencies.",
    )
    p_repair.add_argument(
        "--input", type=Path,
        default=config.DEFAULT_PATHS.get("repair_kg", {}).get("input",
                                                              config.OUTPUTS_INTER / "merged_kg.owl"),
        help="OWL file to repair (default: merged_kg.owl).",
    )
    p_repair.add_argument(
        "--ontology", type=Path,
        default=config.DEFAULT_PATHS.get("repair_kg", {}).get("ontology",
                                                              config.ONTOLOGY_DIR / "LLM_GENIALOntBFO_cleaned.owl"),
        help="Base ontology for alignment.",
    )
    p_repair.add_argument(
        "--episodes", type=int,
        default=getattr(config, "RL_EPISODES", 20),
        help="Number of training episodes (default: 20).",
    )
    p_repair.add_argument(
        "--max-steps", type=int,
        default=getattr(config, "RL_MAX_STEPS_PER_EPISODE", 20),
        help="Max repair steps per episode (default: 20).",
    )
    p_repair.add_argument(
        "--reasoner",
        choices=["konclude", "hermit"],
        default=config.DEFAULT_REASONER,
        help=f"Reasoner to use (default: {config.DEFAULT_REASONER}).",
    )
    p_repair.add_argument(
        "--interactive", action="store_true",
        help="Enable human-in-the-loop mode.",
    )
    p_repair.add_argument(
        "--batch-size", type=int,
        default=getattr(config, "RL_BATCH_SIZE", 32),
        help="DQN training batch size (default: 32).",
    )
    p_repair.add_argument(
        "--save-dir", type=Path,
        default=getattr(config, "RL_MODELS_DIR", Path("outputs/models")),
        help="Directory to save trained models.",
    )

    # ------------------------------------------------------------------ #
    # run-all
    # ------------------------------------------------------------------ #
    p_all = subparsers.add_parser(
        "run-all",
        help="Run the full pipeline end-to-end.",
    )
    p_all.add_argument(
        "--input", type=Path,
        default=config.DEFAULT_PATHS["prepare_corpus"]["input"],
        help="Source CSV file (pipeline entry point).",
    )
    p_all.add_argument(
        "--text-column", default="title",
        help="CSV column to extract text from (default: title).",
    )
    p_all.add_argument(
        "--model", default=config.MODEL_NAME,
        help=f"Ollama model name (default: {config.MODEL_NAME}).",
    )
    p_all.add_argument(
        "--ollama-url", default=config.OLLAMA_URL,
        help="Ollama API endpoint.",
    )
    p_all.add_argument(
        "--fuzzy-cutoff", type=int, default=config.FUZZY_CUTOFF,
        help="Alignment fuzzy match threshold.",
    )
    p_all.add_argument(
        "--allow-create", action="store_true",
        default=config.ALLOW_CREATE_INDIVIDUALS,
        help="Mint new URIs for unmatched entities.",
    )

    return parser


# ---------------------------------------------------------------------------
# Step kwargs builders  (map parsed CLI args-> step kwargs)
# ---------------------------------------------------------------------------

def _kwargs_for_step(command: str, args: argparse.Namespace) -> dict:
    """
    Build the kwargs dict for a given step from parsed CLI args.

    Args:
        command: Step name matching STEP_REGISTRY key.
        args:    Parsed argparse Namespace.

    Returns:
        Dict of kwargs to pass to step.execute().
    """
    mapping = {
        "prepare-corpus": lambda: {
            "input_path":   args.input,
            "output_path":  args.output,
            "text_column":  args.text_column,
        },
        "generate-triples": lambda: {
            "input_path":   args.input,
            "output_path":  args.output,
            "model":        args.model,
            "ollama_url":   args.ollama_url,
            "timeout":      args.timeout,
            "max_chars":    args.max_chars,
            "max_chunks":   args.max_chunks,
        },
        "clean-triples": lambda: {
            "input_path":   args.input,
            "output_path":  args.output,
            "max_tokens":   args.max_tokens,
        },
        "align-triples": lambda: {
            "input_path":    args.input,
            "output_path":   args.output,
            "ontology_path": args.ontology,
            "rel_map_path":  args.rel_map,
            "ent_map_path":  args.ent_map,
            "report_path":   args.report,
            "fuzzy_cutoff":  args.fuzzy_cutoff,
            "allow_create":  args.allow_create,
        },
        "build-kg": lambda: {
            "input_path":       args.input,
            "ontology_path":    args.ontology,
            "output_owl_path":  args.output_owl,
            "output_ttl_path":  args.output_ttl,
        },
        "run-reasoner": lambda: {
            "input_path":   args.input,
            "output_path":  args.output,
        },
        "parse-reasoner": lambda: {
            "input_path":   args.input,
            "output_path":  args.output,
        },
        "check-quality": lambda: {
            "input_path":   args.input,
            "output_path":  args.output,
            "sample_size":  args.sample_size,
            "top_n_size":   args.top_n,
        },
        "repair-kg": lambda: {
            "base_owl":              str(args.input),
            "ontology_path":         str(args.ontology),
            "episodes":              args.episodes,
            "max_steps_per_episode": args.max_steps,
            "reasoner":              args.reasoner,
            "interactive":           args.interactive,
            "batch_size":            args.batch_size,
            "target_update_interval": getattr(config, "RL_TARGET_UPDATE_INTERVAL", 5),
            "save_dir":              str(args.save_dir),
        },
    }
    return mapping[command]()


# ---------------------------------------------------------------------------
# Single step runner
# ---------------------------------------------------------------------------

def run_step(command: str, kwargs: dict) -> Path:
    """
    Instantiate and execute a single pipeline step.

    Args:
        command: Step name matching STEP_REGISTRY key.
        kwargs:  Step kwargs dict.

    Returns:
        Output path returned by the step.

    Raises:
        SystemExit: On FileNotFoundError or RuntimeError — prints
                    a clean error message and exits with code 1.
    """
    step = STEP_REGISTRY[command]()
    try:
        return step.execute(**kwargs)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        logger.error("Step [%s] failed: %s", command, exc)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Full pipeline runner  (Chain of Responsibility)
# ---------------------------------------------------------------------------

def run_all(args: argparse.Namespace) -> None:
    """
    Execute all pipeline steps in order.

    Each step's output automatically feeds the next step's input
    via config.DEFAULT_PATHS. Per-step overrides (model, fuzzy_cutoff
    etc.) are forwarded from the run-all CLI args where applicable.

    Stops immediately on the first step failure.

    Args:
        args: Parsed run-all CLI args.
    """
    logger.info("=" * 60)
    logger.info("Starting full pipeline run")
    logger.info("=" * 60)

    # Per-step kwargs for run-all — paths use config defaults,
    # only user-configurable overrides are wired from args
    step_kwargs: dict[str, dict] = {
        "prepare-corpus": {
            "input_path":  args.input,
            "text_column": args.text_column,
        },
        "generate-triples": {
            "model":      args.model,
            "ollama_url": args.ollama_url,
        },
        "clean-triples":    {},
        "align-triples": {
            "fuzzy_cutoff": args.fuzzy_cutoff,
            "allow_create": args.allow_create,
        },
        "build-kg":         {},
        "run-reasoner":     {},
        "parse-reasoner":   {},
        "check-quality":    {},
    }

    for i, command in enumerate(PIPELINE_ORDER, start=1):
        logger.info("-" * 60)
        logger.info("Step %d/%d — %s", i, len(PIPELINE_ORDER), command)
        logger.info("-" * 60)
        run_step(command, step_kwargs[command])

    logger.info("=" * 60)
    logger.info("Pipeline complete ✓")
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    # Seed all RNGs for reproducibility
    config.set_global_seed()

    parser = build_parser()
    args = parser.parse_args()

    # Apply log level override
    logging.getLogger().setLevel(args.log_level)

    if args.command == "run-all":
        run_all(args)
    else:
        kwargs = _kwargs_for_step(args.command, args)
        run_step(args.command, kwargs)


if __name__ == "__main__":
    main()
