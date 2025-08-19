#!/usr/bin/env python3
"""
PQN (Parallelized Q-Network) implementation for Tessellate.
Based on "Simplifying Deep Temporal Difference Learning" paper.

Key features:
- LayerNorm after final hidden layer (critical for stability)
- L2 regularization on weights
- No target network needed
- No replay buffer needed
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class PQN(nn.Module):
    """
    Q-Network with LayerNorm for stability in off-policy learning.
    
    Architecture based on paper:
    - Input: state vector (104 dims for Tessellate)
    - Hidden layers with ReLU
    - LayerNorm after final hidden layer (before output)
    - Output: Q-values for all actions (100 for Tessellate)
    """
    
    def __init__(
        self,
        state_dim: int = 104,
        action_dim: int = 100,
        hidden_dims: Tuple[int, ...] = (256, 256),
        use_layernorm: bool = True,
        layernorm_eps: float = 1e-5
    ):
        """
        Args:
            state_dim: Dimension of state vector
            action_dim: Number of actions
            hidden_dims: Sizes of hidden layers
            use_layernorm: Whether to use LayerNorm (should be True)
            layernorm_eps: Epsilon for LayerNorm stability
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.use_layernorm = use_layernorm
        
        # Build network layers
        layers = []
        input_dim = state_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        self.feature_layers = nn.Sequential(*layers)
        
        # Critical: LayerNorm after final hidden layer
        if use_layernorm:
            self.layer_norm = nn.LayerNorm(input_dim, eps=layernorm_eps, elementwise_affine=True)
        else:
            self.layer_norm = nn.Identity()
        
        # Output layer (no activation)
        self.output_layer = nn.Linear(input_dim, action_dim)
        
        # Initialize weights carefully
        self._initialize_weights()
        
    def _initialize_weights(self):
        """
        Initialize weights according to paper recommendations.
        Careful initialization is important for stability.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier/Glorot initialization
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        # Smaller initialization for output layer
        nn.init.uniform_(self.output_layer.weight, -0.001, 0.001)
        
    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass computing Q-values.
        
        Args:
            states: Batch of states [batch_size, state_dim]
            
        Returns:
            Q-values for all actions [batch_size, action_dim]
        """
        # Extract features
        features = self.feature_layers(states)
        
        # Apply LayerNorm (critical for stability)
        normalized = self.layer_norm(features)
        
        # Compute Q-values
        q_values = self.output_layer(normalized)
        
        return q_values
    
    def get_l2_loss(self, weight_decay: float = 1e-3) -> torch.Tensor:
        """
        Compute L2 regularization loss on network weights.
        
        As per paper, we focus on final layer weights (φ in paper notation).
        
        Args:
            weight_decay: L2 regularization coefficient
            
        Returns:
            L2 loss term
        """
        l2_loss = 0.0
        
        # Regularize all linear layer weights
        for module in self.modules():
            if isinstance(module, nn.Linear):
                l2_loss += torch.sum(module.weight ** 2)
        
        return weight_decay * l2_loss
    
    def compute_td_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        gamma: float = 0.99,
        weight_decay: float = 1e-3
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute TD loss without target network.
        
        This is the key innovation from the paper:
        - No target network needed due to LayerNorm
        - L2 regularization provides additional stability
        
        Args:
            states: Current states [batch_size, state_dim]
            actions: Actions taken [batch_size]
            rewards: Rewards received [batch_size]
            next_states: Next states [batch_size, state_dim]
            dones: Episode termination flags [batch_size]
            gamma: Discount factor
            weight_decay: L2 regularization coefficient
            
        Returns:
            loss: Total loss (TD + L2)
            metrics: Dictionary of metrics for logging
        """
        batch_size = states.shape[0]
        
        # Current Q-values for taken actions
        current_q_values = self(states)
        current_q = current_q_values.gather(1, actions.long().unsqueeze(1)).squeeze(1)
        
        # Next Q-values (no target network!)
        with torch.no_grad():
            next_q_values = self(next_states)
            next_q_max = next_q_values.max(dim=1)[0]
            
            # Compute TD targets
            td_targets = rewards + gamma * next_q_max * (1 - dones)
        
        # TD error
        td_error = td_targets - current_q
        td_loss = torch.mean(td_error ** 2)
        
        # L2 regularization (critical for stability)
        l2_loss = self.get_l2_loss(weight_decay)
        
        # Total loss
        total_loss = td_loss + l2_loss
        
        # Metrics for monitoring
        metrics = {
            'td_loss': td_loss.item(),
            'l2_loss': l2_loss.item(),
            'total_loss': total_loss.item(),
            'q_mean': current_q_values.mean().item(),
            'q_std': current_q_values.std().item(),
            'q_max': current_q_values.max().item(),
            'td_error_mean': td_error.mean().item(),
            'td_error_std': td_error.std().item()
        }
        
        return total_loss, metrics


class PQNTrainer:
    """
    Training utilities for PQN.
    Handles optimization and provides stable training.
    """
    
    def __init__(
        self,
        model: PQN,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-3,
        gamma: float = 0.99,
        device: str = 'cpu'
    ):
        """
        Args:
            model: PQN model to train
            learning_rate: Learning rate (1e-4 recommended)
            weight_decay: L2 regularization coefficient
            gamma: Discount factor
            device: Device to run on ('cpu' or 'cuda')
        """
        self.model = model.to(device)
        self.device = device
        self.gamma = gamma
        self.weight_decay = weight_decay
        
        # Adam optimizer (as per paper)
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            eps=1e-8  # Slightly higher eps for stability
        )
        
        self.training_steps = 0
        
    def train_batch(self, batch: dict) -> dict:
        """
        Train on a single batch of transitions.
        
        Args:
            batch: Dictionary with keys: states, actions, rewards, next_states, dones
            
        Returns:
            Metrics dictionary
        """
        # Convert to tensors and move to device
        states = torch.FloatTensor(batch['states']).to(self.device)
        actions = torch.LongTensor(batch['actions']).to(self.device)
        rewards = torch.FloatTensor(batch['rewards']).to(self.device)
        next_states = torch.FloatTensor(batch['next_states']).to(self.device)
        dones = torch.FloatTensor(batch['dones']).to(self.device)
        
        # Compute loss
        loss, metrics = self.model.compute_td_loss(
            states, actions, rewards, next_states, dones,
            gamma=self.gamma, weight_decay=self.weight_decay
        )
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability (optional but recommended)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        
        # Update weights
        self.optimizer.step()
        
        self.training_steps += 1
        metrics['training_steps'] = self.training_steps
        
        return metrics
    
    def evaluate(self, batch: dict) -> dict:
        """
        Evaluate model on a batch without training.
        
        Args:
            batch: Dictionary with transition data
            
        Returns:
            Metrics dictionary
        """
        self.model.eval()
        with torch.no_grad():
            states = torch.FloatTensor(batch['states']).to(self.device)
            actions = torch.LongTensor(batch['actions']).to(self.device)
            rewards = torch.FloatTensor(batch['rewards']).to(self.device)
            next_states = torch.FloatTensor(batch['next_states']).to(self.device)
            dones = torch.FloatTensor(batch['dones']).to(self.device)
            
            _, metrics = self.model.compute_td_loss(
                states, actions, rewards, next_states, dones,
                gamma=self.gamma, weight_decay=self.weight_decay
            )
        
        self.model.train()
        return metrics


def test_pqn():
    """Test PQN model creation and forward pass."""
    print("Testing PQN model...\n")
    
    # Create model
    model = PQN(state_dim=104, action_dim=100, hidden_dims=(256, 256))
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test forward pass
    batch_size = 32
    states = torch.randn(batch_size, 104)
    q_values = model(states)
    print(f"Input shape: {states.shape}")
    print(f"Output shape: {q_values.shape}")
    print(f"Q-values range: [{q_values.min():.3f}, {q_values.max():.3f}]")
    
    # Test TD loss computation
    actions = torch.randint(0, 100, (batch_size,))
    rewards = torch.randn(batch_size) * 10
    next_states = torch.randn(batch_size, 104)
    dones = torch.zeros(batch_size)
    dones[::10] = 1  # Some episodes end
    
    loss, metrics = model.compute_td_loss(
        states, actions, rewards, next_states, dones
    )
    
    print(f"\nTD Loss: {metrics['td_loss']:.4f}")
    print(f"L2 Loss: {metrics['l2_loss']:.4f}")
    print(f"Total Loss: {metrics['total_loss']:.4f}")
    print(f"Q-value stats: mean={metrics['q_mean']:.3f}, std={metrics['q_std']:.3f}, max={metrics['q_max']:.3f}")
    
    # Test trainer
    print("\nTesting PQNTrainer...")
    trainer = PQNTrainer(model, learning_rate=1e-4)
    
    # Create fake batch
    batch = {
        'states': states.numpy(),
        'actions': actions.numpy(),
        'rewards': rewards.numpy(),
        'next_states': next_states.numpy(),
        'dones': dones.numpy(),
        'masks': torch.ones(batch_size, 100).numpy()
    }
    
    # Train one step
    metrics = trainer.train_batch(batch)
    print(f"Training step {metrics['training_steps']}: Loss = {metrics['total_loss']:.4f}")


if __name__ == "__main__":
    test_pqn()