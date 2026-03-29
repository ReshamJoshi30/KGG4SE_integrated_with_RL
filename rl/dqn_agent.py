import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim

from rl.dqn_model import DQN


class DQN_Agent:
    """
    Double DQN agent for OWL KG repair.

    Key improvements over vanilla DQN:
    - True Double-DQN: policy net selects best action, target net evaluates it.
      This decouples action selection from evaluation, reducing Q-value
      overestimation that causes the agent to be overconfident and stop exploring.
    - Dynamic output_dim clamping: Q-values for invalid actions are masked to
      -inf so the agent never accidentally picks an out-of-range action.
    - Gradient clipping (max_norm=1.0) prevents exploding gradients from large
      reward signals like the +10 consistency bonus.

    Args:
        input_dim (int): Number of features in the state vector (18).
        output_dim (int): Maximum action space size (env.action_space_n()).
        lr (float): Adam learning rate.
        gamma (float): Discount factor for future rewards.
        epsilon (float): Initial exploration rate.
        epsilon_min (float): Minimum exploration rate.
        epsilon_decay (float): Per-step multiplicative decay factor.
    """

    def __init__(self, input_dim, output_dim, lr=1e-3, gamma=0.99,
                 epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.994):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN(input_dim, output_dim).to(self.device)
        self.target_net = DQN(input_dim, output_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.num_actions = output_dim

    def select_action(self, state, num_valid_actions: int = None):
        """
        Epsilon-greedy action selection with action masking.

        When num_valid_actions is provided, only Q-values for valid actions
        [0 .. num_valid_actions-1] are considered. This prevents the agent from
        selecting actions that don't exist in the current error's action list,
        which would cause env.step() to clamp to index 0 and corrupt training.

        Args:
            state (np.ndarray): Current state feature vector.
            num_valid_actions (int): How many actions are actually valid right now.
                                     If None, uses full output_dim.
        Returns:
            int: Chosen action index.
        """
        n = num_valid_actions if num_valid_actions is not None else self.num_actions
        n = max(1, min(n, self.num_actions))

        if random.random() < self.epsilon:
            return random.randrange(n)

        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_t).squeeze(0)

        # Mask invalid actions to -inf so argmax never picks them
        if n < self.num_actions:
            q_values[n:] = float('-inf')

        return q_values.argmax().item()

    def update(self, replay_buffer, batch_size: int = 64):
        """
        Double-DQN update step.

        Standard DQN computes: target = r + γ * max_a Q_target(s', a)
        This overestimates because the same network both picks AND evaluates actions.

        Double-DQN fixes this:
          1. Policy net selects: a* = argmax_a Q_policy(s', a)
          2. Target net evaluates: Q_target(s', a*)
          3. Target = r + γ * Q_target(s', a*)  [for non-terminal transitions]

        This gives unbiased value estimates and prevents the agent from becoming
        overconfident in early training (which leads to premature exploitation of
        poor strategies).

        Args:
            replay_buffer: ReplayBuffer with .sample() method.
            batch_size: Mini-batch size.
        Returns:
            float | None: Training loss, or None if buffer too small.
        """
        if len(replay_buffer) < batch_size:
            return None

        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

        states_t      = torch.tensor(states,      dtype=torch.float32, device=self.device)
        actions_t     = torch.tensor(actions,     dtype=torch.long,    device=self.device)
        rewards_t     = torch.tensor(rewards,     dtype=torch.float32, device=self.device)
        next_states_t = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones_t       = torch.tensor(dones,       dtype=torch.float32, device=self.device)

        # Current Q-values from policy net (for actions actually taken)
        q_values  = self.policy_net(states_t)
        q_value   = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # --- Double-DQN: policy net picks, target net evaluates ---
            # Step 1: Use POLICY net to find best next action
            next_q_policy  = self.policy_net(next_states_t)
            best_next_actions = next_q_policy.argmax(1)   # shape: (batch,)

            # Step 2: Use TARGET net to get the value of that action
            next_q_target = self.target_net(next_states_t)
            next_q_value  = next_q_target.gather(
                1, best_next_actions.unsqueeze(1)
            ).squeeze(1)

            # Bellman target
            expected_q_value = rewards_t + self.gamma * next_q_value * (1.0 - dones_t)

        loss = nn.SmoothL1Loss()(q_value, expected_q_value)  # Huber loss (more stable than MSE)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping: prevents large +10 / -10 reward spikes from
        # causing exploding gradients that destabilise the policy net.
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.item()

    def update_target(self):
        """Hard copy policy-> target network. Call every N episodes."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def soft_update_target(self, tau: float = 0.005):
        """
        Soft (Polyak) update: θ_target ← τ·θ_policy + (1-τ)·θ_target

        Smoother than hard updates; prevents sudden target shifts that cause
        oscillating Q-values. Recommended when episodes are short (<10 steps).
        Call every step rather than every N episodes.

        Args:
            tau: Interpolation factor. 0.005 is standard in SAC/TD3 literature.
        """
        for tp, pp in zip(self.target_net.parameters(), self.policy_net.parameters()):
            tp.data.copy_(tau * pp.data + (1.0 - tau) * tp.data)

    def decay_epsilon(self):
        """Multiply epsilon by decay factor; floor at epsilon_min."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_q_values(self, state):
        """
        Return raw Q-values for all actions from the policy net.

        Used by train_repair.py to compute confidence (Q-value margin between
        best and second-best action) for the human-intervention gate.

        Returns:
            torch.Tensor: 1-D tensor of shape (output_dim,).
        """
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_t)
        return q_values.flatten()   # Raw values — no scaling


def evaluate_agent(env, agent, max_steps: int = 50) -> dict:
    """
    Run one greedy episode to evaluate the trained agent.

    Sets epsilon=0.0 so the agent always exploits its learned policy.
    Restores epsilon afterwards.

    Args:
        env:       RepairEnv (or Gym-like) environment.
        agent:     Trained DQN_Agent.
        max_steps: Safety cap.
    Returns:
        dict: total_reward, steps, is_consistent
    """
    reset_out = env.reset()
    state, done = (reset_out if isinstance(reset_out, tuple) else (reset_out, False))

    saved_eps   = agent.epsilon
    agent.epsilon = 0.0  # greedy

    total_reward = 0.0
    step = 0

    while not done and step < max_steps:
        num_actions = getattr(env, "get_current_action_count", lambda: None)()
        if num_actions is not None and num_actions == 0:
            break

        action = agent.select_action(state, num_valid_actions=num_actions)
        state, reward, done, info = env.step(action)
        total_reward += reward
        step += 1

    agent.epsilon = saved_eps

    is_consistent = False
    if hasattr(env, "_last_report") and env._last_report:
        is_consistent = env._last_report.get("is_consistent", False)

    return {"total_reward": total_reward, "steps": step, "is_consistent": is_consistent}