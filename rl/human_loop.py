"""
rl/human_loop.py

Human-in-the-loop feedback collection for the phased RL repair trainer.

Key responsibilities
--------------------
1.  Display inconsistency context in plain language using triple_display.
2.  Prompt the human annotator to choose a repair action.
3.  Store the chosen (state, action, reward) tuple in the replay buffer
    immediately — with an optional priority boost — so that expert moves
    are used to train the DQN from the very first episode.
4.  Persist feedback to a JSONL file for offline analysis.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from rl.triple_display import format_error_for_human


class HumanLoop:
    """
    Collects human feedback on repair actions and injects it into the
    replay buffer so that the DQN agent can learn from expert decisions.

    Scripted-actions mode
    ---------------------
    Pass ``scripted_actions_file`` pointing to a JSON file that contains a
    list of integers (action indices) or ``null`` values::

        [0, 1, null, 0, 2, null, ...]

    Each call to ``ask_user()`` pops one entry from the front of the list:
      * integer -> returned immediately (no terminal prompt)
      * null    -> returns None (RL agent decides, same as pressing Enter)

    When the list is exhausted, ``ask_user()`` falls back to the normal
    interactive prompt.  This lets you pre-write a full training run and
    execute it unattended, editing the file to reflect your chosen actions.
    """

    def __init__(
        self,
        feedback_file: str = "outputs/models/human_feedback.jsonl",
        scripted_actions_file: Optional[str] = None,
    ):
        self.feedback_file = Path(feedback_file)
        self.feedback_file.parent.mkdir(parents=True, exist_ok=True)
        self.interactive_mode = False   # set by caller

        # Scripted-actions queue: list[Optional[int]]
        self._scripted_queue: List[Optional[int]] = []
        if scripted_actions_file:
            self._load_scripted_actions(scripted_actions_file)

    # ------------------------------------------------------------------
    # Scripted-actions helpers
    # ------------------------------------------------------------------

    def _load_scripted_actions(self, path: str) -> None:
        """Load a pre-written action sequence from a JSON file."""
        import json as _json
        p = Path(path)
        if not p.exists():
            print(f"[HumanLoop] WARNING: scripted actions file not found: {p}")
            return
        with open(p) as fh:
            data = _json.load(fh)
        if not isinstance(data, list):
            print(f"[HumanLoop] WARNING: scripted actions file must contain a JSON list.")
            return
        self._scripted_queue = list(data)
        print(f"[HumanLoop] Loaded {len(self._scripted_queue)} scripted actions from {p}")

    # ------------------------------------------------------------------
    # Mode toggles
    # ------------------------------------------------------------------

    def enable_interactive(self):
        self.interactive_mode = True
        print("[HumanLoop] Interactive mode ENABLED")

    def disable_interactive(self):
        self.interactive_mode = False
        print("[HumanLoop] Interactive mode DISABLED")

    # ------------------------------------------------------------------
    # Decision gate
    # ------------------------------------------------------------------

    def should_ask_human(
        self,
        state,
        actions: List[Dict],
        agent_confidence: float = 1.0,
        confidence_threshold: float = 0.65,
    ) -> bool:
        """
        Return True if the human should be consulted for the current step.

        Called by the RL_AUTOMATED phase to selectively fall back to a human
        when the agent is uncertain.
        """
        if not self.interactive_mode:
            return False
        # Ask when agent's Q-value margin is low
        if agent_confidence < confidence_threshold:
            return True
        # Always ask for high/critical risk actions (safety net)
        for action in actions:
            if isinstance(action, dict) and action.get("risk") in ("high", "critical"):
                return True
        return False

    # ------------------------------------------------------------------
    # Human prompt
    # ------------------------------------------------------------------

    def ask_user(
        self,
        error_obj: Dict[str, Any],
        actions: List[Dict],
        owl_path: Optional[str] = None,
        agent_action: Optional[int] = None,
    ) -> Optional[int]:
        """
        Show the inconsistency context and repair options to the human,
        then collect their choice.

        Parameters
        ----------
        error_obj    : Current error dict from RepairEnv._current_error
        actions      : Candidate repair action dicts
        owl_path     : Path to the live OWL file — used to display the
                       graph neighbourhood triples.  Pass None to skip.
        agent_action : RL agent's preferred action index (shown as hint).

        Returns
        -------
        int   — chosen action index
        None  — human skipped; caller should fall back to the RL agent
        """
        # ── Scripted-actions shortcut ────────────────────────────────────────
        if self._scripted_queue:
            entry = self._scripted_queue.pop(0)
            n = len(actions)
            if entry is None:
                print(f"[HumanLoop] Scripted: defer to RL agent "
                      f"({len(self._scripted_queue)} actions remaining)")
                return None
            if isinstance(entry, int) and 0 <= entry < n:
                print(f"[HumanLoop] Scripted: action [{entry}] chosen "
                      f"({len(self._scripted_queue)} actions remaining)")
                return entry
            print(f"[HumanLoop] Scripted: invalid entry {entry!r} for "
                  f"{n} actions — deferring to RL")
            return None

        # ── Normal interactive prompt ─────────────────────────────────────
        # Render full context display
        display = format_error_for_human(
            error_obj=error_obj,
            owl_path=owl_path,
            agent_action=agent_action,
        )
        print(display)

        # Prompt loop
        n = len(actions)
        hint = (f"  (RL agent suggests [{agent_action}])"
                if agent_action is not None else "")
        print(f"\n{hint}")

        while True:
            try:
                raw = input(
                    f"Choose repair action [0-{n-1}], "
                    "or press Enter to let the RL agent decide: "
                ).strip()

                if raw == "":
                    print(" -> Handing decision to RL agent.")
                    return None

                idx = int(raw)
                if 0 <= idx < n:
                    print(f" -> Human chose action [{idx}].")
                    return idx
                print(f"  Invalid choice. Enter a number between 0 and {n-1}.")

            except ValueError:
                print("  Invalid input. Enter a number or press Enter to skip.")
            except KeyboardInterrupt:
                print("\n -> Interrupted. Handing decision to RL agent.")
                return None

    # ------------------------------------------------------------------
    # Replay buffer injection
    # ------------------------------------------------------------------

    def inject_into_replay_buffer(
        self,
        replay_buffer,
        state,
        action_idx: int,
        reward: float,
        next_state,
        done: float,
        boost: float = 2.0,
    ):
        """
        Push the human-chosen transition into the replay buffer twice:

        First push  — with the actual observed reward.
        Second push — with reward + *boost*, so the expert transition is
                      sampled more often during DQN mini-batch training.

        This is a lightweight form of prioritised experience replay that
        does not require changing the ReplayBuffer internals.

        Parameters
        ----------
        replay_buffer : ReplayBuffer instance (supports .push())
        state         : State at the time of the human decision
        action_idx    : Action index the human chose
        reward        : Reward received after the action
        next_state    : Next state returned by the environment
        done          : Terminal flag (float 0.0 or 1.0)
        boost         : Extra reward added on the second push
        """
        replay_buffer.push(state, action_idx, reward, next_state, done)
        replay_buffer.push(state, action_idx, reward + boost, next_state, done)

    # ------------------------------------------------------------------
    # Feedback persistence
    # ------------------------------------------------------------------

    def store_feedback(
        self,
        state,
        actions: List[Dict],
        chosen_action_idx: int,
        reward: float,
        error_obj: Dict[str, Any],
    ):
        """
        Append one human decision to the JSONL feedback log.

        The stored records can be used later for:
          - Offline imitation learning / behavioural cloning
          - Auditing which errors required human intervention
          - Analysing whether human choices generalise across episodes
        """
        record = {
            "state":             (state.tolist()
                                  if hasattr(state, "tolist") else list(state)),
            "actions":           actions,
            "chosen_action_idx": chosen_action_idx,
            "reward":            reward,
            "error_type":        error_obj.get("error_type"),
            "entity":            error_obj.get("entity_label",
                                               error_obj.get("entity", "unknown")),
        }
        with open(self.feedback_file, "a") as fh:
            json.dump(record, fh)
            fh.write("\n")
        print(f"[HumanLoop] Feedback stored-> {self.feedback_file}")

    def load_feedback(self) -> List[Dict[str, Any]]:
        """Load all stored human feedback records from the JSONL file."""
        if not self.feedback_file.exists():
            return []
        records = []
        with open(self.feedback_file, "r") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records
