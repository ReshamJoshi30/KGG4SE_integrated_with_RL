"""
rl/env_repair.py
RL environment for KG repair from reasoner feedback.

Key fixes in this version
--------------------------
1. step() accepts step_num kwarg — passed to compute_reward() for the
   efficiency bonus (agent learns to repair faster, not just repair at all).

2. _last_step_improved is now correctly computed as:
   "did violations DECREASE?" rather than "were there any violation deltas?"
   The old logic set it to True even when violations *increased*, giving the
   agent a misleading momentum signal.

3. Reward function call updated to pass step_num and max_steps.
"""
import numpy as np
from pathlib import Path
from alignment.ontology_index import OntologyIndex
from reasoning.reasoner_wrapper import run_reasoner
from qa.repair_candidates import make_repair_candidates
from qa.apply_fix import apply_fix
from rl.reward_functions import compute_reward


class RepairEnv:
    """
    Reinforcement Learning environment for OWL/KG repair.

    State: Encoded error vector (18-dimensional)
    Action: Index into candidate repair actions
    Reward: Improvement in reasoner metrics (shaped reward)
    """

    _ERROR_TYPE_MAP = {
        "disjoint_violation":            0,
        "transitive_disjoint_violation": 1,
        "class_used_as_individual":      2,
        "domain_violation":              3,
        "range_violation":               4,
        "functional_property_violation": 5,
        "property_type_violation":       6,
        "external_type_violation":       7,
        "unsat_class":                   8,
        "unknown_inconsistency":         9,
    }
    _NUM_ERROR_TYPES = 10

    def __init__(self, base_owl: str, ontology_path: str,
                 max_steps: int = 20, reasoner: str = "hermit"):
        self.base_owl    = base_owl
        self.onto_idx    = OntologyIndex(ontology_path)
        self.max_steps   = max_steps
        self.reasoner    = reasoner

        self._last_report: dict = None
        self._step = 0
        self._error_queue = []
        self._current_error = None
        self._current_owl   = base_owl

        self._initial_queue_len: int  = 0
        self._last_step_improved: bool = False

        self.episode_history = []

    # ----- State representation -----

    def state_dim(self):
        """18-dimensional state vector. See _encode_error() for layout."""
        return self._NUM_ERROR_TYPES + 8  # 18

    def _encode_error(self, err):
        """Convert error object to fixed-size 18-dim numpy vector."""
        vec = np.zeros(self.state_dim(), dtype=np.float32)

        etype = err.get("error_type", "unknown")
        idx   = self._ERROR_TYPE_MAP.get(etype, self._NUM_ERROR_TYPES - 1)
        vec[idx] = 1.0

        actions  = err.get("actions", [])
        vec[10]  = min(len(actions), 10.0) / 10.0

        low_risk = sum(
            1 for a in actions
            if isinstance(a, dict) and a.get("risk") in ("low", "none")
        )
        vec[11]  = min(low_risk, 5.0) / 5.0
        vec[12]  = 1.0 if err.get("evidence") else 0.0
        vec[13]  = self._step / self.max_steps

        entity   = err.get("entity") or err.get("violating_entity", "")
        vec[14]  = 1.0 if (entity and entity != "Unknown") else 0.0

        remaining = len(self._error_queue)
        vec[15]   = (min(1.0, remaining / self._initial_queue_len)
                     if self._initial_queue_len > 0 else 0.0)

        # FIX: True only when violations actually DECREASED last step
        vec[16]   = 1.0 if self._last_step_improved else 0.0

        if actions:
            penalties  = [a.get("risk_penalty", 5.0) for a in actions if isinstance(a, dict)]
            min_penalty = min(penalties) if penalties else 5.0
            vec[17]    = min(1.0, min_penalty / 5.0)

        return vec

    # ----- Core RL loop -----

    def reset(self):
        """Reset environment to initial state. Returns (state, done)."""
        self._current_owl  = self.base_owl
        self._step         = 0
        self._last_report  = None
        self._last_step_improved = False
        self._error_queue  = self._get_errors(self._current_owl)
        self._initial_queue_len = len(self._error_queue)
        self.episode_history = []
        return self._next_state()

    def _get_errors(self, owl_path):
        """Run reasoner and extract error candidates."""
        report = run_reasoner(owl_path, reasoner=self.reasoner)

        print(f"[RepairEnv] Reasoner: consistent={report['is_consistent']}, "
              f"unsat={len(report.get('unsat_classes', []))}, "
              f"disjoint={len(report.get('disjoint_violations', []))}")

        if (not report["is_consistent"] and
                len(report.get("unsat_classes", [])) == 0 and
                len(report.get("disjoint_violations", [])) == 0):
            print("[RepairEnv] No specific errors, extracting evidence...")
            try:
                from reasoning.axiom_extractor import extract_inconsistency_explanation
                evidence = extract_inconsistency_explanation(owl_path)
                if evidence:
                    report["evidence"] = evidence
                    print(f"[RepairEnv] Evidence: {evidence.get('error_type')}")
            except Exception as e:
                print(f"[RepairEnv] Evidence extraction failed: {e}")

        repairs = make_repair_candidates(report, self.onto_idx)
        print(f"[RepairEnv] Generated {len(repairs)} repair candidates")
        self._last_report = report
        return repairs

    def _next_state(self):
        """Get next error from queue and encode as state. Returns (state, done)."""
        if not self._error_queue:
            print("[RepairEnv] No more errors — terminal state")
            return np.zeros(self.state_dim(), dtype=np.float32), True
        self._current_error = self._error_queue.pop(0)
        state = self._encode_error(self._current_error)
        return state, False

    def step(self, action_idx: int, step_num: int = None):
        """
        Apply repair action, re-run reasoner, compute reward.

        Args:
            action_idx: Index of action to take.
            step_num:   Current step number (used by reward function for
                        efficiency bonus). Defaults to self._step if None.

        Returns:
            (next_state, reward, done, info)
        """
        if step_num is None:
            step_num = self._step

        print(f"[RepairEnv] Step {self._step}: action {action_idx}")

        actions = (self._current_error or {}).get("actions", []) or []
        if not actions:
            print("[RepairEnv] No actions available-> terminating")
            return self._encode_terminal_state(), 0.0, True, {}

        if action_idx < 0 or action_idx >= len(actions):
            print(f"[RepairEnv] Invalid action {action_idx} for {len(actions)} actions-> clamping to 0")
            action_idx = 0

        repaired_owl = apply_fix(
            current_owl=self._current_owl,
            error_obj=self._current_error,
            action_idx=action_idx,
            onto_idx=self.onto_idx,
            step_id=self._step,
        )

        new_report = run_reasoner(repaired_owl, reasoner=self.reasoner)

        # Reward: pass step_num and max_steps for efficiency shaping
        base_reward, metrics = compute_reward(
            new_report, self._last_report,
            step_num=step_num,
            max_steps=self.max_steps,
        )

        # FIX: _last_step_improved = violations DECREASED (not just non-zero delta)
        # Old code: checked if metrics > 0, which is True even for regressions.
        self._last_step_improved = (
            metrics.get("unsat",  0) > 0 or   # violations fixed (positive delta)
            metrics.get("disj",   0) > 0 or
            metrics.get("trans",  0) > 0 or
            metrics.get("now_consistent", False)
        )

        # Risk penalty
        chosen_action = self._current_error["actions"][action_idx]
        if isinstance(chosen_action, dict):
            risk_penalty = chosen_action.get("risk_penalty", 0.0)
            reward = base_reward - risk_penalty
            print(f"[RepairEnv] Base={base_reward:.2f}  Risk={risk_penalty:.2f}  Final={reward:.2f}")
        else:
            reward = base_reward

        # Save entity from current error before _next_state() overwrites it
        if self._current_error:
            current_entity = (self._current_error.get("entity") or
                              self._current_error.get("violating_entity"))
            if current_entity and current_entity != "Unknown":
                new_report["last_entity"] = current_entity

        self._last_report = new_report
        self._current_owl = repaired_owl
        self._step       += 1

        self.episode_history.append({
            "step":     self._step,
            "action":   chosen_action,
            "reward":   reward,
            "metrics":  metrics,
            "owl_file": repaired_owl,
        })

        # Refresh candidates from new report
        self._error_queue = make_repair_candidates(new_report, self.onto_idx)

        state, done_queue = self._next_state()
        done = done_queue or (self._step >= self.max_steps)

        if done:
            print(f"[RepairEnv] Episode done: steps={self._step}, reward={reward:.2f}")

        info = {
            "metrics":       metrics,
            "current_error": self._current_error,
            "owl_file":      repaired_owl,
            "step":          self._step,
        }
        return state, reward, done, info

    def _encode_terminal_state(self):
        return np.zeros(self.state_dim(), dtype=np.float32)

    def action_space_n(self):
        """Max actions — DQN output layer size."""
        return 10

    def get_current_action_count(self):
        """Actual number of valid actions for the current error."""
        if self._current_error is None:
            return 0
        return len(self._current_error.get("actions", []))

    def get_current_triple_context(self, max_triples: int = 8):
        """Return RDF triples involving the current entity (for human display)."""
        if not self._current_error or not self._current_owl:
            return []
        entity = (self._current_error.get("entity") or
                  self._current_error.get("violating_entity", ""))
        if not entity or entity == "Unknown":
            return []
        try:
            import rdflib
            g = rdflib.Graph()
            g.parse(str(self._current_owl))
            ref = rdflib.URIRef(entity)
            triples = []
            for t in g.triples((ref, None, None)):
                triples.append(t)
                if len(triples) >= max_triples:
                    break
            remaining = max_triples - len(triples)
            for t in g.triples((None, None, ref)):
                triples.append(t)
                remaining -= 1
                if remaining <= 0:
                    break
            return triples
        except Exception:
            return []