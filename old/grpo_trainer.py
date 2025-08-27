#!/usr/bin/env python3
"""
GRPO (Group Relative Policy Optimization) trainer for Tessellate.
Minimal implementation following DeepSeek R1-Zero approach.
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
import numpy as np
from dataclasses import dataclass
from llm_game_interface import LLMGameInterface


@dataclass
class GRPOConfig:
    """Configuration for GRPO training."""
    # Group generation
    k_completions: int = 2  # Number of completions per state
    
    # Optimization
    learning_rate: float = 1e-6
    clip_range: float = 0.2  # PPO-style clipping
    kl_coef: float = 0.01  # KL divergence coefficient
    
    # Rewards
    invalid_move_penalty: float = -100.0  # Penalty for invalid moves
    
    # Training
    batch_size: int = 2  # Number of games in parallel
    max_thinking_tokens: int = 512  # Max tokens for thinking
    max_total_tokens: int = 1024  # Max tokens total
    
    # Normalization
    advantage_normalization: bool = True
    reward_scaling: float = 0.001  # Scale raw score differentials


class GRPOTrainer:
    """
    GRPO trainer for Tessellate reasoning model.
    
    Following DeepSeek R1-Zero approach:
    1. Generate K completions for each state
    2. Evaluate completions via game rollouts
    3. Compute group-relative advantages
    4. Update policy with PPO-style clipping and KL constraint
    """
    
    def __init__(
        self,
        model,  # HuggingFace model
        tokenizer,  # HuggingFace tokenizer
        config: GRPOConfig = GRPOConfig(),
        device: str = "mps"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        
        # Game interface for rollouts
        self.game_interface = LLMGameInterface(batch_size=config.batch_size * config.k_completions)
        
        # Optimizer (only create if model exists)
        if model is not None:
            self.optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config.learning_rate,
                betas=(0.9, 0.999)
            )
        else:
            self.optimizer = None
        
        # Reference model for KL divergence (frozen copy)
        self.ref_model = None  # Will be set after initial checkpoint
        
    def set_reference_model(self):
        """Create frozen reference model for KL divergence."""
        # Clone current model as reference
        self.ref_model = type(self.model).from_pretrained(
            self.model.config._name_or_path,
            torch_dtype=self.model.dtype,
            device_map=self.device
        )
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False
    
    def generate_completions(
        self, 
        state_texts: List[str]
    ) -> Tuple[List[str], torch.Tensor]:
        """
        Generate K completions for each state.
        
        Returns:
            completions: List of completion texts
            log_probs: Log probabilities of completions
        """
        completions = []
        all_log_probs = []
        
        for state_text in state_texts:
            # Create prompt
            messages = [{
                "role": "user",
                "content": f"Current game state:\n{state_text}\n\nAnalyze and select your move."
            }]
            
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Generate K completions
            for _ in range(self.config.k_completions):
                inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=self.config.max_total_tokens,
                        temperature=0.8,  # Some diversity for group generation
                        do_sample=True,
                        return_dict_in_generate=True,
                        output_scores=True
                    )
                
                # Extract generated text
                generated_ids = outputs.sequences[0][len(inputs.input_ids[0]):]
                completion = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                completions.append(completion)
                
                # Calculate log probabilities
                # Note: This is simplified - real implementation needs proper log prob calculation
                if outputs.scores:
                    log_probs = torch.stack([
                        score.log_softmax(dim=-1).max() 
                        for score in outputs.scores
                    ]).mean()
                    all_log_probs.append(log_probs)
        
        return completions, torch.stack(all_log_probs) if all_log_probs else torch.zeros(len(completions))
    
    def evaluate_completions(
        self,
        state_texts: List[str],
        completions: List[str]
    ) -> np.ndarray:
        """
        Evaluate completions by playing out games.
        
        Returns:
            rewards: Array of rewards for each completion
        """
        rewards = []
        
        # Process each completion
        for i, completion in enumerate(completions):
            # Parse action from completion
            action = self.game_interface.parse_action(completion)
            
            if action is None:
                # Invalid format penalty
                rewards.append(self.config.invalid_move_penalty)
                continue
            
            # For now, simplified: just use the score differential from existing game
            # In full implementation, would play out rest of game
            # This is a placeholder - need actual game rollout
            reward = np.random.randn() * 100  # Placeholder
            rewards.append(reward)
        
        return np.array(rewards)
    
    def compute_advantages(self, rewards: np.ndarray, k: int) -> np.ndarray:
        """
        Compute group-relative advantages.
        
        For each group of K completions, normalize rewards.
        """
        advantages = np.zeros_like(rewards, dtype=np.float32)
        
        # Process in groups of K
        for i in range(0, len(rewards), k):
            group_rewards = rewards[i:i+k].astype(np.float32)
            
            if self.config.advantage_normalization and len(group_rewards) > 1:
                # Normalize within group
                mean = group_rewards.mean()
                std = group_rewards.std()
                if std > 1e-8:
                    group_advantages = (group_rewards - mean) / std
                else:
                    # If all rewards are the same, no preference
                    group_advantages = group_rewards - mean
            else:
                # Just center
                group_advantages = group_rewards - group_rewards.mean()
            
            advantages[i:i+k] = group_advantages
        
        return advantages
    
    def compute_kl_divergence(self, logits: torch.Tensor, ref_logits: torch.Tensor) -> torch.Tensor:
        """
        Compute KL divergence between current policy and reference.
        """
        log_probs = F.log_softmax(logits, dim=-1)
        ref_log_probs = F.log_softmax(ref_logits, dim=-1)
        
        kl = (torch.exp(ref_log_probs) * (ref_log_probs - log_probs)).sum(dim=-1)
        return kl.mean()
    
    def grpo_loss(
        self,
        completions: List[str],
        advantages: torch.Tensor,
        old_log_probs: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute GRPO loss with PPO-style clipping and KL constraint.
        
        L = -E[min(r*A, clip(r)*A)] + β*KL(π||π_ref)
        where r = π(a|s) / π_old(a|s)
        """
        # Get current log probabilities
        # This is simplified - need proper implementation
        current_log_probs = old_log_probs  # Placeholder
        
        # Compute probability ratios
        ratios = torch.exp(current_log_probs - old_log_probs)
        
        # PPO-style clipping
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - self.config.clip_range, 1 + self.config.clip_range) * advantages
        
        # Policy loss (maximize advantage)
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # KL divergence loss (if reference model exists)
        kl_loss = torch.tensor(0.0)
        if self.ref_model is not None:
            # Simplified KL calculation
            kl_loss = self.config.kl_coef * current_log_probs.mean()  # Placeholder
        
        return policy_loss + kl_loss
    
    def train_step(self, trajectories: List[Dict]) -> Dict:
        """
        Single GRPO training step.
        
        Args:
            trajectories: List of game trajectories with states
            
        Returns:
            Metrics dict
        """
        # Sample states from trajectories
        sampled_states = []
        for traj in trajectories:
            # Sample a few states from each trajectory
            indices = np.random.choice(len(traj['trajectory']), size=min(5, len(traj['trajectory'])), replace=False)
            for idx in indices:
                sampled_states.append(traj['trajectory'][idx]['state_text'])
        
        # Limit batch size
        sampled_states = sampled_states[:self.config.batch_size]
        
        # Generate K completions per state
        completions, old_log_probs = self.generate_completions(sampled_states)
        
        # Evaluate completions
        rewards = self.evaluate_completions(sampled_states, completions)
        
        # Scale rewards
        rewards = rewards * self.config.reward_scaling
        
        # Compute advantages
        advantages = self.compute_advantages(rewards, self.config.k_completions)
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        
        # Compute loss
        loss = self.grpo_loss(completions, advantages, old_log_probs)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return {
            'loss': loss.item(),
            'mean_reward': rewards.mean(),
            'std_reward': rewards.std(),
            'mean_advantage': advantages.mean().item(),
        }


def test_grpo():
    """Test GRPO implementation with dummy data."""
    print("Testing GRPO trainer...")
    
    # Would need actual model here
    # from transformers import AutoModelForCausalLM, AutoTokenizer
    # model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B-Thinking-2507")
    # tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Thinking-2507")
    
    # For now, just test structure
    config = GRPOConfig(k_completions=2, batch_size=2)
    print(f"Config: K={config.k_completions}, batch={config.batch_size}")
    print(f"Reward scaling: {config.reward_scaling}")
    print(f"KL coefficient: {config.kl_coef}")
    
    # Test advantage computation
    rewards = np.array([100, -50, 200, 150])  # 2 groups of 2
    trainer = GRPOTrainer(None, None, config)  # Dummy trainer
    advantages = trainer.compute_advantages(rewards, k=2)
    print(f"\nRewards: {rewards}")
    print(f"Advantages: {advantages}")
    print(f"Sum per group: {advantages[0:2].sum():.2f}, {advantages[2:4].sum():.2f}")


if __name__ == "__main__":
    test_grpo()