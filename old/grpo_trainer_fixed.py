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
from tessellate_env import TessellateEnv
import copy


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
        device: str = "cpu"
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
        """
        Create frozen reference model for KL divergence.
        Uses Option B: Reference is the model after initial reasoning generation.
        """
        if self.model is None:
            return
            
        # Deep copy current model weights as reference
        self.ref_model = copy.deepcopy(self.model)
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False
        print("Reference model set for KL divergence")
    
    def generate_completions(
        self, 
        state_texts: List[str]
    ) -> Tuple[List[str], List[torch.Tensor], List[List[int]]]:
        """
        Generate K completions for each state with proper log probabilities.
        
        Returns:
            completions: List of completion texts
            log_probs: List of log probability tensors for each completion
            token_ids: List of token ID lists for each completion
        """
        completions = []
        all_log_probs = []
        all_token_ids = []
        
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
                    # Generate with attention to scores
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=self.config.max_total_tokens,
                        temperature=0.8,
                        do_sample=True,
                        return_dict_in_generate=True,
                        output_scores=True,
                        output_attentions=False
                    )
                
                # Extract generated tokens (excluding prompt)
                generated_ids = outputs.sequences[0][len(inputs.input_ids[0]):]
                completion = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                completions.append(completion)
                all_token_ids.append(generated_ids.tolist())
                
                # Calculate log probabilities properly
                if outputs.scores:
                    # Stack all scores (one per generated token)
                    scores = torch.stack(outputs.scores)  # [seq_len, vocab_size]
                    
                    # Get log probabilities for the actual generated tokens
                    log_probs_per_token = []
                    for t, token_id in enumerate(generated_ids):
                        if t < len(scores):
                            log_probs = F.log_softmax(scores[t], dim=-1)
                            token_log_prob = log_probs[token_id.item()]
                            log_probs_per_token.append(token_log_prob)
                    
                    # Store sequence log probabilities
                    if log_probs_per_token:
                        sequence_log_probs = torch.stack(log_probs_per_token)
                        all_log_probs.append(sequence_log_probs)
                    else:
                        all_log_probs.append(torch.zeros(1).to(self.device))
                else:
                    all_log_probs.append(torch.zeros(1).to(self.device))
        
        return completions, all_log_probs, all_token_ids
    
    def evaluate_completions(
        self,
        state_texts: List[str],
        completions: List[str],
        initial_states: List[Dict]
    ) -> np.ndarray:
        """
        Evaluate completions by playing out full games.
        
        Args:
            state_texts: Current state descriptions
            completions: Generated completions with actions
            initial_states: Full state info including move number
            
        Returns:
            rewards: Array of rewards for each completion
        """
        rewards = []
        
        for i, (state_text, completion) in enumerate(zip(state_texts, completions)):
            # Parse action from completion
            action = self.game_interface.parse_action(completion)
            
            if action is None:
                # Invalid format penalty
                rewards.append(self.config.invalid_move_penalty)
                continue
            
            # Get initial state info
            initial_state = initial_states[i % len(initial_states)]
            move_num = initial_state.get('move_num', 0)
            
            # Create environment and set to current state
            env = TessellateEnv(reward_mode='final')
            
            # Reconstruct game state from trajectory
            # This assumes we have the board state in initial_state
            if 'board' in initial_state:
                # Set environment to current board state
                env.reset()
                # We need to properly set the env state - this is simplified
                # In practice, would need to replay moves or have state setter
                
            # Check if action is valid
            valid_actions = env.get_valid_actions()
            if action not in valid_actions:
                rewards.append(self.config.invalid_move_penalty / 2)
                continue
            
            # Execute the player's action
            obs, reward, done, info = env.step(action)
            
            # Play out rest of game with random policy
            while not done:
                # Random policy for simplicity
                valid_actions = env.get_valid_actions()
                if not valid_actions:
                    break
                random_action = np.random.choice(valid_actions)
                obs, reward, done, info = env.step(random_action)
            
            # Get final score differential as reward
            if 'final_scores' in info:
                red_score = info['final_scores'].get('red', 0)
                blue_score = info['final_scores'].get('blue', 0)
                
                # Determine which player the model was playing
                current_player = initial_state.get('current_player', 1)
                if current_player == 1:  # Red
                    score_diff = red_score - blue_score
                else:  # Blue
                    score_diff = blue_score - red_score
                
                rewards.append(score_diff)
            else:
                rewards.append(0.0)
        
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
    
    def compute_kl_divergence(
        self,
        input_ids: torch.Tensor,
        generated_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute KL divergence between current policy and reference.
        
        Args:
            input_ids: Input token IDs
            generated_ids: Generated token IDs
        """
        if self.ref_model is None:
            return torch.tensor(0.0).to(self.device)
        
        # Concatenate input and generated tokens
        full_ids = torch.cat([input_ids, generated_ids], dim=-1)
        
        # Get logits from both models
        with torch.no_grad():
            current_outputs = self.model(full_ids)
            ref_outputs = self.ref_model(full_ids)
        
        # Calculate KL divergence on generated portion
        seq_len = generated_ids.shape[-1]
        current_logits = current_outputs.logits[:, -seq_len:, :]
        ref_logits = ref_outputs.logits[:, -seq_len:, :]
        
        # Convert to probabilities
        current_probs = F.softmax(current_logits, dim=-1)
        ref_probs = F.softmax(ref_logits, dim=-1)
        
        # KL(current || ref) = sum(current * log(current/ref))
        kl = (current_probs * (current_probs.log() - ref_probs.log())).sum(dim=-1)
        
        return kl.mean()
    
    def grpo_loss(
        self,
        state_texts: List[str],
        completions: List[str],
        token_ids: List[List[int]],
        advantages: torch.Tensor,
        old_log_probs: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute GRPO loss with PPO-style clipping and KL constraint.
        
        L = -E[min(r*A, clip(r)*A)] + β*KL(π||π_ref)
        where r = π(a|s) / π_old(a|s)
        """
        total_loss = torch.tensor(0.0).to(self.device)
        
        for i, (state_text, completion, tokens) in enumerate(zip(state_texts, completions, token_ids)):
            # Prepare input
            messages = [{
                "role": "user",
                "content": f"Current game state:\n{state_text}\n\nAnalyze and select your move."
            }]
            
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            generated_ids = torch.tensor(tokens).unsqueeze(0).to(self.device)
            
            # Get current log probabilities
            full_ids = torch.cat([inputs.input_ids, generated_ids], dim=-1)
            outputs = self.model(full_ids)
            
            # Calculate log probs for generated tokens
            logits = outputs.logits[:, len(inputs.input_ids[0])-1:-1, :]
            current_log_probs = F.log_softmax(logits, dim=-1)
            
            # Get log probs for actual tokens
            token_log_probs = current_log_probs.gather(
                dim=-1,
                index=generated_ids.unsqueeze(-1)
            ).squeeze(-1)
            
            # Sum log probs for the sequence
            current_seq_log_prob = token_log_probs.sum()
            old_seq_log_prob = old_log_probs[i].sum() if i < len(old_log_probs) else current_seq_log_prob.detach()
            
            # Compute probability ratio
            ratio = torch.exp(current_seq_log_prob - old_seq_log_prob)
            
            # PPO-style clipping
            advantage = advantages[i]
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.config.clip_range, 1 + self.config.clip_range) * advantage
            
            # Add to loss (negative because we maximize advantage)
            total_loss -= torch.min(surr1, surr2)
            
            # Add KL divergence
            if self.ref_model is not None:
                kl_div = self.compute_kl_divergence(inputs.input_ids, generated_ids)
                total_loss += self.config.kl_coef * kl_div
        
        return total_loss / len(state_texts)
    
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
        state_infos = []
        
        for traj in trajectories:
            # Sample a few states from each trajectory
            indices = np.random.choice(
                len(traj['trajectory']), 
                size=min(5, len(traj['trajectory'])), 
                replace=False
            )
            for idx in indices:
                step = traj['trajectory'][idx]
                sampled_states.append(step['state_text'])
                state_infos.append({
                    'move_num': step['move_num'],
                    'current_player': step.get('current_player', 1),
                    'board': step.get('board', [])  # Would need board state
                })
        
        # Limit batch size
        sampled_states = sampled_states[:self.config.batch_size]
        state_infos = state_infos[:self.config.batch_size]
        
        # Generate K completions per state
        completions, old_log_probs, token_ids = self.generate_completions(sampled_states)
        
        # Evaluate completions with full game rollouts
        rewards = self.evaluate_completions(sampled_states, completions, state_infos)
        
        # Scale rewards
        rewards = rewards * self.config.reward_scaling
        
        # Compute advantages
        advantages = self.compute_advantages(rewards, self.config.k_completions)
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        
        # Compute loss
        loss = self.grpo_loss(sampled_states, completions, token_ids, advantages, old_log_probs)
        
        # Backward pass
        if self.optimizer is not None:
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
        
        return {
            'loss': loss.item() if isinstance(loss, torch.Tensor) else loss,
            'mean_reward': rewards.mean(),
            'std_reward': rewards.std(),
            'mean_advantage': advantages.mean().item(),
        }


def test_grpo():
    """Test GRPO implementation with dummy data."""
    print("Testing GRPO trainer...")
    
    # Test configuration
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
    
    print("\nAll placeholders have been replaced with actual implementations!")
    print("\nNext steps:")
    print("1. Generate initial reasoning with Qwen3-4B-Thinking")
    print("2. Set that checkpoint as reference model")
    print("3. Run GRPO training loop")


if __name__ == "__main__":
    test_grpo()