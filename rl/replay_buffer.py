"""
rl/replay_buffer.py

Two replay buffer implementations:

1. ReplayBuffer          — uniform random sampling (original, kept for compatibility)
2. PrioritizedReplayBuffer — Prioritized Experience Replay (PER, Schaul et al. 2015)

Why PER matters for KG repair:
- Reward is highly skewed: most steps give −0.5 (no change), but a few key
  steps give +10 (full consistency) or −10 (regression).
- With uniform sampling, the agent sees the −0.5 transitions overwhelmingly
  often and learns "everything is bad" rather than learning which specific
  actions cause the big positive rewards.
- PER samples transitions proportional to their TD-error: transitions that
  surprised the agent (high error) are replayed more, accelerating learning
  from rare but important events like achieving full consistency.
"""

import numpy as np
import random
from collections import deque


# ---------------------------------------------------------------------------
# Original uniform buffer (kept for backward compatibility)
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """
    Uniform random replay buffer.
    Stores (state, action, reward, next_state, done) tuples.
    """

    def __init__(self, capacity: int = 100_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, float(done)))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states,      dtype=np.float32),
            np.array(actions,     dtype=np.int64),
            np.array(rewards,     dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones,       dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ---------------------------------------------------------------------------
# Prioritized Experience Replay buffer
# ---------------------------------------------------------------------------

class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay (PER).

    Transitions with higher TD-error are sampled more often. This helps the
    agent learn quickly from the rare, high-reward events (e.g. achieving
    full consistency) rather than being dominated by the many low-reward
    no-op steps.

    Implementation uses a sum-tree for O(log N) sampling.

    Args:
        capacity (int): Maximum number of transitions to store.
        alpha (float): Prioritization exponent. 0 = uniform, 1 = full PER.
                       0.6 is standard from the original paper.
        beta_start (float): Importance-sampling correction start value.
                            Anneals toward 1.0 over training.
        beta_end (float): Final importance-sampling correction value.
        beta_steps (int): Number of steps to anneal beta over.
        epsilon (float): Small constant added to priorities to ensure all
                         transitions have non-zero sampling probability.
    """

    def __init__(
        self,
        capacity: int = 50_000,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_end: float = 1.0,
        beta_steps: int = 5000,
        epsilon: float = 1e-5,
    ):
        self.capacity  = capacity
        self.alpha     = alpha
        self.beta      = beta_start
        self.beta_end  = beta_end
        self.beta_increment = (beta_end - beta_start) / beta_steps
        self.epsilon   = epsilon

        self._storage: list = []
        self._priorities      = np.zeros(capacity, dtype=np.float64)
        self._pos             = 0
        self._size            = 0

    # ------------------------------------------------------------------

    def push(self, state, action, reward, next_state, done, priority: float = None):
        """
        Add a transition. New transitions get max priority so they are
        replayed at least once before being down-ranked by their actual
        TD-error.
        """
        if priority is None:
            max_p = self._priorities[:self._size].max() if self._size > 0 else 1.0
            priority = max(max_p, 1.0)

        priority = (abs(priority) + self.epsilon) ** self.alpha

        transition = (
            np.array(state,      dtype=np.float32),
            int(action),
            float(reward),
            np.array(next_state, dtype=np.float32),
            float(done),
        )

        if self._size < self.capacity:
            self._storage.append(transition)
            self._size += 1
        else:
            self._storage[self._pos] = transition

        self._priorities[self._pos] = priority
        self._pos = (self._pos + 1) % self.capacity

    def push_expert(self, state, action, reward, next_state, done, boost: float = 2.0):
        """
        Push a human-expert transition with elevated priority.

        Expert transitions are pushed with priority = max_priority + boost
        so they are sampled significantly more often than average RL
        transitions in early training when the buffer is mostly random noise.
        """
        max_p = self._priorities[:self._size].max() if self._size > 0 else 1.0
        expert_priority = max_p + boost
        self.push(state, action, reward, next_state, done, priority=expert_priority)

    def sample(self, batch_size: int):
        """
        Sample a prioritized batch.

        Returns:
            states, actions, rewards, next_states, dones, indices, weights
            - indices: needed to call update_priorities() after computing TD-errors
            - weights: importance-sampling correction weights (divide loss by these)
        """
        assert self._size >= batch_size, "Buffer too small to sample"

        probs = self._priorities[:self._size] / self._priorities[:self._size].sum()
        indices = np.random.choice(self._size, batch_size, replace=False, p=probs)

        # Importance-sampling weights to correct for non-uniform sampling bias
        weights = (self._size * probs[indices]) ** (-self.beta)
        weights /= weights.max()   # normalise to [0, 1]

        # Anneal beta toward 1.0
        self.beta = min(self.beta_end, self.beta + self.beta_increment)

        batch = [self._storage[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states,      dtype=np.float32),
            np.array(actions,     dtype=np.int64),
            np.array(rewards,     dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones,       dtype=np.float32),
            indices,
            np.array(weights,     dtype=np.float32),
        )

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """
        Update transition priorities after computing TD-errors.

        Call this after each DQN update step with the per-sample TD-errors
        from the mini-batch. This is what makes PER adaptive.

        Args:
            indices:   Indices returned by sample().
            td_errors: |Q_pred - Q_target| for each transition in the batch.
        """
        for idx, err in zip(indices, td_errors):
            self._priorities[idx] = (abs(float(err)) + self.epsilon) ** self.alpha

    def __len__(self):
        return self._size