import torch
import torch.nn as nn


class DQN(nn.Module):
    """
    Deep Q-Network for OWL KG repair action selection.

    Architecture improvements over the original:
    - Dueling network heads: separates VALUE estimation (how good is this state?)
      from ADVANTAGE estimation (how much better is action A vs the average?).
      This helps the agent learn the state value independently of which action
      to take — critical when many actions have similar outcomes (e.g. all
      low-risk actions are roughly equivalent).
    - Batch normalisation after the first layer: stabilises training when
      reward magnitudes vary widely (+10 for consistency vs -0.5 for no-op).
    - Dropout (p=0.1): light regularisation to prevent overfitting to the
      small set of error patterns seen in a single KG.

    Dueling formula (Wang et al. 2016):
        Q(s, a) = V(s) + A(s, a) - mean_a'[A(s, a')]

    Args:
        input_dim (int): State vector size (18 in this system).
        output_dim (int): Number of possible actions (env.action_space_n()).
    """

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()

        # Shared feature extractor
        h1 = max(128, input_dim * 8)
        h2 = max(64,  h1 // 2)

        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.LayerNorm(h1),   # LayerNorm works with any batch size incl. 1
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(h1, h2),
            nn.ReLU(),
        )

        # Value stream: V(s) — scalar
        self.value_stream = nn.Sequential(
            nn.Linear(h2, h2 // 2),
            nn.ReLU(),
            nn.Linear(h2 // 2, 1),
        )

        # Advantage stream: A(s, a) — one value per action
        self.advantage_stream = nn.Sequential(
            nn.Linear(h2, h2 // 2),
            nn.ReLU(),
            nn.Linear(h2 // 2, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning Q-values for all actions.

        Args:
            x: Input tensor (batch_size, input_dim)
        Returns:
            Q-values tensor (batch_size, output_dim)
        """
        features   = self.feature_net(x)
        value      = self.value_stream(features)           # (B, 1)
        advantages = self.advantage_stream(features)        # (B, A)

        # Dueling combination: subtract mean advantage for identifiability
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values