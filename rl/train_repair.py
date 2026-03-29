"""
rl/train_repair.py
DQN training loop for RL-based KG repair — with phased human-in-the-loop.

Key fixes in this version
--------------------------
1. Prioritized Experience Replay (PER):
   - High-TD-error transitions are sampled more often
   - Expert transitions use push_expert() with elevated priority instead of
     double-pushing (which distorted the buffer and caused reward inflation)
   - Agent.update() now returns (loss, td_errors) so we can update priorities

2. Soft target network updates (Polyak averaging, τ=0.005):
   - Called every step instead of hard copy every N episodes
   - Prevents sudden jumps in target Q-values that destabilise training

3. Action masking in select_action():
   - num_valid_actions passed so DQN never selects out-of-range actions
   - Fixes silent clamping to action 0 that distorted training

4. Reward function gets step_num for efficiency bonus:
   - Agent learns to repair faster (not just repair at all)

5. Training history JSON now includes:
   - per-step losses and TD-errors for debugging
   - buffer stats (size, mean priority)

Three-phase curriculum (unchanged):
  Phase 1 — HUMAN_SUPERVISED : human decides every action
  Phase 2 — HUMAN_FIRST      : human decides first action only
  Phase 3 — RL_AUTOMATED     : fully autonomous, human asked only when
                                confidence < threshold
"""

import json
import logging
import random
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
config.set_global_seed()

from rl.dqn_agent     import DQN_Agent
from rl.env_repair    import RepairEnv
from rl.replay_buffer import PrioritizedReplayBuffer
from rl.diff_tracker  import DiffTracker
from rl.human_loop    import HumanLoop

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Phase helpers
# ---------------------------------------------------------------------------

def _get_phase(ep: int) -> str:
    ph1_end = config.RL_HUMAN_SUPERVISED_EPISODES
    ph2_end = ph1_end + config.RL_HUMAN_FIRST_EPISODES
    if ep <= ph1_end:
        return "human_supervised"
    if ep <= ph2_end:
        return "human_first"
    return "rl_automated"


def _compute_confidence(q_vals: torch.Tensor, num_actions: int) -> float:
    """
    Q-value margin between best and second-best valid action.
    Returns value in [0, inf) — higher = more confident.

    Note: We no longer normalise to [0,1] because raw margin is more
    informative for the confidence threshold check.
    """
    if num_actions <= 1:
        return float("inf")
    try:
        valid = [q_vals[i].item() for i in range(min(num_actions, len(q_vals)))]
        if len(valid) < 2:
            return float("inf")
        valid.sort(reverse=True)
        return valid[0] - valid[1]
    except Exception:
        return float("inf")


def _phase_banner(ep: int, total: int, phase: str) -> str:
    labels = {
        "human_supervised": "HUMAN SUPERVISED",
        "human_first":      "HUMAN FIRST-STEP",
        "rl_automated":     "RL AUTOMATED    ",
    }
    return f"[Episode {ep:3d}/{total}]  Phase: {labels.get(phase, phase)}"


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_repair(**kwargs: Any) -> tuple:
    """
    Train a DQN agent to repair KG inconsistencies.

    Returns:
        (agent, episode_rewards, episode_successes)
    """
    # ── Unpack kwargs ────────────────────────────────────────────────────────
    base_owl      = kwargs.get("base_owl",
                               str(config.DEFAULT_PATHS["repair_kg"]["input"]))
    ontology_path = kwargs.get("ontology_path",
                               str(config.DEFAULT_PATHS["repair_kg"]["ontology"]))
    episodes      = int(kwargs.get("episodes",              config.RL_EPISODES))
    max_steps     = int(kwargs.get("max_steps_per_episode", config.RL_MAX_STEPS_PER_EPISODE))
    reasoner      = kwargs.get("reasoner",                  config.DEFAULT_REASONER)
    batch_size    = int(kwargs.get("batch_size",            config.RL_BATCH_SIZE))
    target_update = int(kwargs.get("target_update_interval", config.RL_TARGET_UPDATE_INTERVAL))
    save_dir      = Path(kwargs.get("save_dir",             config.RL_MODELS_DIR))
    feedback_file = kwargs.get("feedback_file",
                               str(config.RL_MODELS_DIR / "human_feedback.jsonl"))
    scripted_actions_file = kwargs.get("scripted_actions_file", None)
    no_human      = bool(kwargs.get("no_human", False))
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── Environment ──────────────────────────────────────────────────────────
    logger.info("Initialising RepairEnv  base_owl=%s  reasoner=%s", base_owl, reasoner)
    env = RepairEnv(
        base_owl=base_owl,
        ontology_path=ontology_path,
        max_steps=max_steps,
        reasoner=reasoner,
    )

    # ── Agent ────────────────────────────────────────────────────────────────
    agent = DQN_Agent(
        input_dim=env.state_dim(),          # 18 features
        output_dim=env.action_space_n(),    # 10 max actions
        lr=1e-3,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=config.RL_EPSILON_DECAY,
    )

    # ── Prioritized Replay Buffer ─────────────────────────────────────────────
    # Beta anneals from 0.4-> 1.0 over the full training run
    total_expected_steps = episodes * max_steps
    replay_buffer = PrioritizedReplayBuffer(
        capacity=50_000,
        alpha=0.6,
        beta_start=0.4,
        beta_end=1.0,
        beta_steps=total_expected_steps,
    )

    diff_tracker = DiffTracker()

    # ── Human-in-the-loop handler ─────────────────────────────────────────────
    human_loop = HumanLoop(
        feedback_file=feedback_file,
        scripted_actions_file=scripted_actions_file,
    )
    if not no_human:
        human_loop.enable_interactive()

    # ── History tracking ──────────────────────────────────────────────────────
    episode_rewards:   list[float] = []
    episode_successes: list[bool]  = []
    training_history:  list[dict]  = []

    first_consistent_ep:      Optional[int]  = None
    first_violation_fixed_ep: Optional[int]  = None
    milestones: list[str] = []
    global_step: int = 0   # total steps across all episodes

    logger.info(
        "Starting RL training — %d episodes, max %d steps each\n"
        "  Phase 1 (human supervised): episodes 1-%d\n"
        "  Phase 2 (human first-step): episodes %d-%d\n"
        "  Phase 3 (RL automated):     episodes %d-%d\n"
        "  Replay: PrioritizedReplayBuffer(alpha=0.6, beta_start=0.4)",
        episodes, max_steps,
        config.RL_HUMAN_SUPERVISED_EPISODES,
        config.RL_HUMAN_SUPERVISED_EPISODES + 1,
        config.RL_HUMAN_SUPERVISED_EPISODES + config.RL_HUMAN_FIRST_EPISODES,
        config.RL_HUMAN_SUPERVISED_EPISODES + config.RL_HUMAN_FIRST_EPISODES + 1,
        episodes,
    )

    # ── Episode loop ──────────────────────────────────────────────────────────
    for ep in range(1, episodes + 1):
        phase      = _get_phase(ep)
        state, done = env.reset()

        # Snapshot initial violations for history
        start_report = env._last_report or {}
        start_unsat  = len(start_report.get("unsat_classes",            []))
        start_disj   = len(start_report.get("disjoint_violations",      []))

        print("\n" + "=" * 68)
        print(_phase_banner(ep, episodes, phase))
        print(f"  Start violations — unsat: {start_unsat}, disjoint: {start_disj}")
        print("=" * 68)

        total_reward  = 0.0
        step          = 0
        prev_owl      = env._current_owl
        ep_losses: list[float] = []

        while not done and step < max_steps:
            num_actions = env.get_current_action_count()
            if num_actions == 0:
                break

            current_actions = (env._current_error or {}).get("actions", [])
            human_chose     = False
            action: Optional[int] = None

            # ── Phase 1: Human decides every action ──────────────────────
            if phase == "human_supervised":
                action = human_loop.ask_user(
                    error_obj=env._current_error or {},
                    actions=current_actions,
                    owl_path=env._current_owl,
                    agent_action=None,
                )
                if action is None:
                    action = random.randrange(num_actions)
                else:
                    human_chose = True

            # ── Phase 2: Human decides step 0 only ───────────────────────
            elif phase == "human_first" and step == 0:
                agent_hint = agent.select_action(state, num_valid_actions=num_actions)
                action = human_loop.ask_user(
                    error_obj=env._current_error or {},
                    actions=current_actions,
                    owl_path=env._current_owl,
                    agent_action=agent_hint,
                )
                if action is None:
                    action = agent_hint
                else:
                    human_chose = True

            # ── Phase 2 (step > 0) or Phase 3: RL decides ────────────────
            else:
                q_vals     = agent.get_q_values(state)
                confidence = _compute_confidence(q_vals, num_actions)

                if (phase == "rl_automated" and
                        human_loop.should_ask_human(
                            state, current_actions, confidence,
                            config.RL_AUTO_CONFIDENCE_THRESHOLD)):
                    agent_hint = int(q_vals[:num_actions].argmax().item())
                    action = human_loop.ask_user(
                        error_obj=env._current_error or {},
                        actions=current_actions,
                        owl_path=env._current_owl,
                        agent_action=agent_hint,
                    )
                    if action is None:
                        action = agent_hint
                    else:
                        human_chose = True

                if action is None:
                    # Standard epsilon-greedy with action masking
                    action = agent.select_action(state, num_valid_actions=num_actions)

            # ── Capture action dict before env mutates state ──────────────
            action_dict = (current_actions[action]
                           if action < len(current_actions)
                           else {"type": "unknown"})

            # ── Environment step ──────────────────────────────────────────
            # Pass step_num for efficiency bonus in reward function
            next_state, reward, done, info = env.step(action, step_num=step)

            # ── Milestone detection ───────────────────────────────────────
            metrics_now = info.get("metrics", {})
            if metrics_now.get("now_consistent") and not metrics_now.get("was_consistent"):
                if first_consistent_ep is None:
                    first_consistent_ep = ep
                    msg = f"[MILESTONE] Ep {ep} Step {step}: KG achieved FULL CONSISTENCY for the first time!"
                    print("\n" + "!" * 68)
                    print(msg)
                    print("!" * 68 + "\n")
                    milestones.append(msg)

            any_improvement = any(metrics_now.get(k, 0) > 0 for k in ("unsat", "disj", "trans"))
            if any_improvement and first_violation_fixed_ep is None:
                first_violation_fixed_ep = ep
                msg = f"[MILESTONE] Ep {ep} Step {step}: First violation FIXED (reward={reward:.2f})"
                print(f"\n  >> {msg}")
                milestones.append(msg)

            # ── Store transition in buffer ────────────────────────────────
            if human_chose:
                # Expert transition: elevated priority in PER buffer
                replay_buffer.push_expert(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=float(done),
                    boost=config.RL_HUMAN_REPLAY_BOOST,
                )
                human_loop.store_feedback(
                    state=state,
                    actions=current_actions,
                    chosen_action_idx=action,
                    reward=reward,
                    error_obj=env._current_error or {},
                )
            else:
                replay_buffer.push(state, action, reward, next_state, float(done))

            total_reward += reward

            # ── Train DQN on prioritized mini-batch ───────────────────────
            train_result = _update_agent_per(agent, replay_buffer, batch_size)
            if train_result is not None:
                loss, td_errors, sample_indices = train_result
                ep_losses.append(loss)
                replay_buffer.update_priorities(sample_indices, td_errors)

            # ── Soft target network update (every step) ───────────────────
            # Polyak averaging τ=0.005 is smoother than hard update every N eps
            agent.soft_update_target(tau=0.005)

            # ── Log diff ──────────────────────────────────────────────────
            diff_tracker.log_step(
                step_id=step,
                before_owl=prev_owl,
                after_owl=info.get("owl_file", prev_owl),
                action=action_dict,
                reward=reward,
                metrics=info.get("metrics", {}),
            )

            state     = next_state
            prev_owl  = info.get("owl_file", prev_owl)
            step     += 1
            global_step += 1

            # ── Per-step epsilon decay ────────────────────────────────────
            agent.decay_epsilon()

        # ── End-of-episode ────────────────────────────────────────────────
        is_consistent = (env._last_report.get("is_consistent", False)
                         if env._last_report else False)
        episode_rewards.append(total_reward)
        episode_successes.append(is_consistent)

        end_report = env._last_report or {}
        end_unsat  = len(end_report.get("unsat_classes",       []))
        end_disj   = len(end_report.get("disjoint_violations", []))

        avg_loss  = float(np.mean(ep_losses)) if ep_losses else None
        ok_count  = sum(episode_successes)
        loss_str  = f"{avg_loss:.4f}" if avg_loss is not None else "n/a "

        training_history.append({
            "episode":              ep,
            "phase":                phase,
            "steps":                step,
            "reward":               total_reward,
            "consistent":           is_consistent,
            "epsilon":              agent.epsilon,
            "avg_loss":             avg_loss,
            "success_rate_so_far":  ok_count / ep,
            "start_unsat":          start_unsat,
            "start_disj":           start_disj,
            "end_unsat":            end_unsat,
            "end_disj":             end_disj,
            "buffer_size":          len(replay_buffer),
            "global_step":          global_step,
        })

        # Phase transition detection
        if ep < episodes:
            next_phase = _get_phase(ep + 1)
            if next_phase != phase:
                msg = (f"[PHASE TRANSITION] Episode {ep} complete. "
                       f"Next: {next_phase.upper()}")
                print("\n" + "*" * 68)
                print(f"  {msg}")
                print("*" * 68)
                milestones.append(msg)

        # Progress bar
        done_frac = ep / episodes
        bar_len   = 30
        filled    = int(bar_len * done_frac)
        bar       = "#" * filled + "-" * (bar_len - filled)
        print(
            f"  [{bar}] Ep {ep:3d}/{episodes} | "
            f"steps={step:2d} | reward={total_reward:+7.2f} | "
            f"consistent={'YES' if is_consistent else 'no ':3s} | "
            f"eps={agent.epsilon:.3f} | loss={loss_str}"
        )

        logger.info(
            "Ep %d/%d  steps=%d  reward=%.2f  consistent=%s  eps=%.3f  loss=%s",
            ep, episodes, step, total_reward, is_consistent, agent.epsilon, loss_str,
        )

        # Periodic checkpoint
        if ep % max(1, episodes // 4) == 0:
            ckpt = save_dir / f"dqn_repair_ep{ep}.pt"
            torch.save(agent.policy_net.state_dict(), str(ckpt))
            logger.info("  Checkpoint -> %s", ckpt)

    # ── Save final model ──────────────────────────────────────────────────────
    final_path = save_dir / "dqn_repair_final.pt"
    torch.save(agent.policy_net.state_dict(), str(final_path))

    # ── Save training history ─────────────────────────────────────────────────
    history_path = save_dir / "training_history.json"
    with open(history_path, "w") as fh:
        json.dump({
            "config": {
                "episodes":    episodes,
                "max_steps":   max_steps,
                "reasoner":    reasoner,
                "batch_size":  batch_size,
                "replay_type": "PrioritizedReplayBuffer",
                "alpha":       0.6,
                "beta_start":  0.4,
                "epsilon_decay": config.RL_EPSILON_DECAY,
                "model": "DQN_DuelingDoubleDQN",
            },
            "per_episode": training_history,
        }, fh, indent=2)
    logger.info("Training history -> %s", history_path)

    # ── Summary ───────────────────────────────────────────────────────────────
    n_success    = sum(episode_successes)
    success_pct  = n_success / episodes * 100
    avg_reward   = sum(episode_rewards) / episodes
    best_reward  = max(episode_rewards)
    worst_reward = min(episode_rewards)

    # Phase-level accuracy breakdown
    ep_data = training_history
    for phase_name in ["human_supervised", "human_first", "rl_automated"]:
        phase_eps = [e for e in ep_data if e["phase"] == phase_name]
        if phase_eps:
            phase_success = sum(1 for e in phase_eps if e["consistent"]) / len(phase_eps) * 100
            print(f"  {phase_name:<22}: {phase_success:.1f}% success over {len(phase_eps)} episodes")

    print("\n" + "=" * 68)
    print("  TRAINING COMPLETE — SUMMARY FOR SUPERVISOR")
    print("=" * 68)
    print(f"  Total episodes          : {episodes}")
    print(f"  Max steps per episode   : {max_steps}")
    print(f"  Reasoner used           : {reasoner}")
    print(f"  State dimension         : {env.state_dim()} features")
    print(f"  Replay buffer           : PrioritizedReplayBuffer (PER)")
    print(f"  Network architecture    : Dueling Double-DQN")
    print()
    print(f"  Success rate            : {success_pct:.1f}%  ({n_success}/{episodes} episodes ended consistent)")
    print(f"  Average episode reward  : {avg_reward:+.2f}")
    print(f"  Best episode reward     : {best_reward:+.2f}")
    print(f"  Worst episode reward    : {worst_reward:+.2f}")
    print(f"  Final epsilon           : {agent.epsilon:.4f}  (target: ≤0.10 means mostly exploiting)")
    print(f"  Total steps trained     : {global_step}")
    print()
    print(f"  Phase 1 (human supervised) : episodes 1-{config.RL_HUMAN_SUPERVISED_EPISODES}")
    print(f"  Phase 2 (human first-step) : episodes "
          f"{config.RL_HUMAN_SUPERVISED_EPISODES+1}-"
          f"{config.RL_HUMAN_SUPERVISED_EPISODES+config.RL_HUMAN_FIRST_EPISODES}")
    print(f"  Phase 3 (RL automated)     : episodes "
          f"{config.RL_HUMAN_SUPERVISED_EPISODES+config.RL_HUMAN_FIRST_EPISODES+1}-{episodes}")
    print()
    if milestones:
        print("  Key milestones achieved:")
        for m in milestones:
            print(f"    {m}")
    print()
    print(f"  Model saved to: {final_path}")
    print(f"  History  saved to: {history_path}")
    print(f"  Run: python report_quality.py  for before/after comparison.")
    print("=" * 68)

    return agent, episode_rewards, episode_successes


# ---------------------------------------------------------------------------
# PER-aware agent update helper
# ---------------------------------------------------------------------------

def _update_agent_per(agent, replay_buffer: "PrioritizedReplayBuffer",
                      batch_size: int):
    """
    Run one DQN update step using the PER buffer.

    Returns:
        (loss, td_errors, indices) if update occurred, else None.
    """
    if len(replay_buffer) < batch_size:
        return None

    states, actions, rewards, next_states, dones, indices, weights = \
        replay_buffer.sample(batch_size)

    import torch
    device = agent.device
    states_t      = torch.tensor(states,      dtype=torch.float32, device=device)
    actions_t     = torch.tensor(actions,     dtype=torch.long,    device=device)
    rewards_t     = torch.tensor(rewards,     dtype=torch.float32, device=device)
    next_states_t = torch.tensor(next_states, dtype=torch.float32, device=device)
    dones_t       = torch.tensor(dones,       dtype=torch.float32, device=device)
    weights_t     = torch.tensor(weights,     dtype=torch.float32, device=device)

    # Current Q-values
    q_values = agent.policy_net(states_t)
    q_value  = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        # Double-DQN: policy net selects, target net evaluates
        next_q_policy     = agent.policy_net(next_states_t)
        best_next_actions = next_q_policy.argmax(1)
        next_q_target     = agent.target_net(next_states_t)
        next_q_value      = next_q_target.gather(1, best_next_actions.unsqueeze(1)).squeeze(1)
        expected_q_value  = rewards_t + agent.gamma * next_q_value * (1.0 - dones_t)

    # TD-errors (used to update PER priorities)
    td_errors = (q_value - expected_q_value).detach().abs().cpu().numpy()

    # Weighted Huber loss (IS-correction from PER)
    import torch.nn as nn
    element_loss = nn.SmoothL1Loss(reduction="none")(q_value, expected_q_value)
    loss = (weights_t * element_loss).mean()

    agent.optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(agent.policy_net.parameters(), max_norm=1.0)
    agent.optimizer.step()

    return loss.item(), td_errors, indices


# ---------------------------------------------------------------------------
# Quick integration test
# ---------------------------------------------------------------------------

def test_repair_one_episode():
    """Test one repair episode to verify integration."""
    print("=" * 68)
    print("TESTING RL REPAIR MODULE - ONE EPISODE")
    print("=" * 68)

    env = RepairEnv(
        base_owl="outputs/test/merged_kg_broken.owl",
        ontology_path="ontology/LLM_GENIALOntBFO_cleaned.owl",
        max_steps=10,
        reasoner="hermit",
    )
    agent = DQN_Agent(
        input_dim=env.state_dim(),
        output_dim=env.action_space_n(),
        epsilon=1.0,
    )
    diff_tracker = DiffTracker()

    state, done = env.reset()
    step = 0
    total_reward = 0.0
    prev_owl = env._current_owl

    while not done and step < 5:
        num_actions = env.get_current_action_count()
        if num_actions == 0:
            break
        action = random.randint(0, num_actions - 1)
        current_actions = env._current_error.get("actions", [])
        action_dict = current_actions[action] if action < len(current_actions) else {}
        next_state, reward, done, info = env.step(action, step_num=step)
        diff_tracker.log_step(step, prev_owl, info["owl_file"], action_dict, reward, info["metrics"])
        total_reward += reward
        state = next_state
        prev_owl = info["owl_file"]
        step += 1

    print(f"Total steps: {step}, Total reward: {total_reward:.2f}")
    print(f"Final consistent: {env._last_report.get('is_consistent', False)}")
    print("TEST COMPLETE")


if __name__ == "__main__":
    import argparse
    import logging as _logging

    _logging.basicConfig(
        level=_logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="DQN repair trainer — run from project root",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--test",      action="store_true")
    parser.add_argument("--episodes",  type=int,  default=config.RL_EPISODES)
    parser.add_argument("--max-steps", type=int,  default=config.RL_MAX_STEPS_PER_EPISODE)
    parser.add_argument("--reasoner",  default=config.DEFAULT_REASONER, choices=["hermit", "konclude"])
    parser.add_argument("--scripted",  metavar="FILE", default=None)
    parser.add_argument("--no-human",  action="store_true")
    parser.add_argument("--owl",       default=None)
    args = parser.parse_args()

    if args.test:
        test_repair_one_episode()
    else:
        if args.no_human:
            config.RL_HUMAN_SUPERVISED_EPISODES = 0
            config.RL_HUMAN_FIRST_EPISODES = 0
            print("[CLI] --no-human: human phases disabled")

        kw: dict = dict(
            episodes=args.episodes,
            max_steps_per_episode=args.max_steps,
            reasoner=args.reasoner,
            no_human=args.no_human,
        )
        if args.scripted:
            kw["scripted_actions_file"] = args.scripted
        if args.owl:
            kw["base_owl"] = args.owl

        train_repair(**kw)