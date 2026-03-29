"""
rl/diff_tracker.py

Track and log changes between repair steps.
Outputs detailed JSON trace of what changed.
"""

import json
from pathlib import Path
from typing import Dict, Any, List
from rdflib import Graph


class DiffTracker:
    """
    Logs differences between consecutive OWL files during repair.
    """

    def __init__(self, output_dir: str = "outputs/rl_repair_traces"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.trace_file = self.output_dir / "repair_trace.jsonl"

        # Clear previous trace
        if self.trace_file.exists():
            self.trace_file.unlink()

    def log_step(
        self,
        step_id: int,
        before_owl: str,
        after_owl: str,
        action: Dict[str, Any],
        reward: float,
        metrics: Dict[str, Any]
    ):
        """
        Log a single repair step.

        Args:
            step_id: Step number
            before_owl: Path to OWL before action
            after_owl: Path to OWL after action
            action: Action dict that was applied
            reward: Reward received
            metrics: Reasoner metrics after action
        """
        # Compute diff
        diff = self._compute_diff(before_owl, after_owl)

        # Create log entry
        entry = {
            "step": step_id,
            "action": action,
            "reward": reward,
            "metrics": metrics,
            "diff": diff,
            "snapshots": {
                "before": before_owl,
                "after": after_owl
            }
        }

        # Append to JSONL file
        with open(self.trace_file, "a") as f:
            json.dump(entry, f)
            f.write("\n")

        print(f"[DiffTracker] Logged step {step_id}-> {self.trace_file}")

    def _compute_diff(self, before_owl: str, after_owl: str) -> Dict[str, Any]:
        """
        Compute differences between two OWL files.

        Returns:
            Dict with keys: triples_added, triples_removed, triple_count_before, triple_count_after
        """
        try:
            g_before = Graph()
            g_before.parse(before_owl)

            g_after = Graph()
            g_after.parse(after_owl)

            # Compute set differences
            added = g_after - g_before
            removed = g_before - g_after

            return {
                # Limit to first 10
                "triples_added": [self._triple_to_str(t) for t in added][:10],
                "triples_removed": [self._triple_to_str(t) for t in removed][:10],
                "triple_count_before": len(g_before),
                "triple_count_after": len(g_after),
                "num_added": len(added),
                "num_removed": len(removed)
            }

        except Exception as e:
            print(f"[DiffTracker] Error computing diff: {e}")
            return {
                "error": str(e),
                "triples_added": [],
                "triples_removed": [],
                "num_added": 0,
                "num_removed": 0
            }

    def _triple_to_str(self, triple) -> str:
        """Convert rdflib triple to readable string."""
        s, p, o = triple
        return f"({self._shorten_iri(s)}, {self._shorten_iri(p)}, {self._shorten_iri(o)})"

    def _shorten_iri(self, iri) -> str:
        """Shorten IRI to last component."""
        iri_str = str(iri)
        if "#" in iri_str:
            return iri_str.split("#")[-1]
        elif "/" in iri_str:
            return iri_str.split("/")[-1]
        return iri_str

    def get_episode_summary(self) -> Dict[str, Any]:
        """
        Read trace file and generate episode summary.

        Returns:
            Dict with: total_steps, total_reward, final_metrics, actions_taken
        """
        if not self.trace_file.exists():
            return {"error": "No trace file found"}

        steps = []
        with open(self.trace_file, "r") as f:
            for line in f:
                steps.append(json.loads(line))

        if not steps:
            return {"error": "Empty trace"}

        total_reward = sum(s["reward"] for s in steps)
        actions_taken = [
            s["action"].get("action_type") or s["action"].get(
                "type", "unknown")
            for s in steps
        ]

        return {
            "total_steps": len(steps),
            "total_reward": total_reward,
            "final_metrics": steps[-1]["metrics"],
            "actions_taken": actions_taken,
            "trace_file": str(self.trace_file)
        }
