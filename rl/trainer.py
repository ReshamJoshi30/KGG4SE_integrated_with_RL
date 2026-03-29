"""rl/trainer.py

Generic DQN training loop that works with RepairEnv.

RepairEnv.reset() returns ``(state, done)`` which differs from the
standard Gym convention (single state).  This module handles both
conventions so it can also be used with Gym-compatible envs in future.
"""

import random
import logging
from typing import Any

logger = logging.getLogger(__name__)


def train_dqn(
    env,
    agent,
    replay_buffer,
    episodes: int = 300,
    batch_size: int = 64,
    target_update_interval: int = 10,
) -> tuple[list[float], list[float]]:
    """Train a DQN agent inside *env*.

    Compatible with both RepairEnv (returns ``(state, done)`` from
    ``reset()``) and standard Gym envs (returns ``state``).

    Args:
        env:   RL environment (RepairEnv or Gym-like).
        agent: DQN_Agent instance.
        replay_buffer: ReplayBuffer instance.
        episodes: Number of training episodes.
        batch_size: Mini-batch size for DQN updates.
        target_update_interval: Sync target net every N episodes.

    Returns:
        (episode_rewards, all_losses)
    """
    episode_rewards: list[float] = []
    all_losses: list[float] = []

    for episode in range(episodes):
        # Handle both (state, done) and plain state returns
        reset_out = env.reset()
        if isinstance(reset_out, tuple):
            state, done = reset_out
        else:
            state, done = reset_out, False

        total_reward = 0.0

        while not done:
            # Action masking for RepairEnv
            num_actions = getattr(
                env, "get_current_action_count", lambda: None)()
            if num_actions is not None:
                if num_actions == 0:
                    break
                if random.random() < agent.epsilon:
                    action = random.randrange(num_actions)
                else:
                    action = agent.select_action(state)
                    action = min(action, num_actions - 1)
            else:
                action = agent.select_action(state)

            next_state, reward, done, _ = env.step(action)

            replay_buffer.push(state, action, reward, next_state, float(done))
            total_reward += reward

            loss = agent.update(replay_buffer, batch_size)
            if loss is not None:
                all_losses.append(loss)

            state = next_state

        agent.decay_epsilon()

        if (episode + 1) % target_update_interval == 0:
            agent.update_target()

        if (episode + 1) % 10 == 0:
            logger.info(
                "Episode %d/%d — Reward: %.2f  Eps: %.3f",
                episode + 1, episodes, total_reward, agent.epsilon,
            )

        episode_rewards.append(total_reward)

    return episode_rewards, all_losses
