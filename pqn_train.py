#!/usr/bin/env python3
"""
Train PQN on Tessellate random play data.
This script demonstrates extracting signal from random trajectories.
"""

import torch
import numpy as np
from pathlib import Path
import time
from datetime import datetime
import json
from typing import Optional

from data_loader import TessellateDataLoader
from pqn_model import PQN, PQNTrainer
from tessellate_env import TessellateEnv


class PQNExperiment:
    """
    Full training pipeline for PQN on Tessellate data.
    """
    
    def __init__(
        self,
        data_dir: str = 'rl_data_large',
        max_files: Optional[int] = None,
        batch_size: int = 10000,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-3,
        gamma: float = 0.99,
        hidden_dims: tuple = (256, 256),
        device: str = 'auto'
    ):
        """
        Args:
            data_dir: Directory containing preprocessed data
            max_files: Number of NPZ files to use (None = all)
            batch_size: Batch size for training
            learning_rate: Learning rate for Adam
            weight_decay: L2 regularization coefficient
            gamma: Discount factor
            hidden_dims: Hidden layer dimensions
            device: Device to use ('auto', 'cpu', or 'cuda')
        """
        # Setup device
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        print(f"Using device: {self.device}")
        
        # Create data loader
        self.data_loader = TessellateDataLoader(
            data_dir=data_dir,
            max_files=max_files,
            batch_size=batch_size,
            shuffle_files=True,
            shuffle_within_files=True,
            verbose=True
        )
        
        # Create model
        self.model = PQN(
            state_dim=104,
            action_dim=100,
            hidden_dims=hidden_dims,
            use_layernorm=True
        )
        
        # Create trainer
        self.trainer = PQNTrainer(
            model=self.model,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            gamma=gamma,
            device=self.device
        )
        
        # Metrics tracking
        self.metrics_history = []
        self.eval_history = []
        
    def evaluate_vs_random(self, n_games: int = 100) -> dict:
        """
        Evaluate trained agent against random opponent.
        
        Args:
            n_games: Number of games to play
            
        Returns:
            Evaluation metrics
        """
        self.model.eval()
        wins = 0
        total_reward = 0
        game_lengths = []
        
        for _ in range(n_games):
            env = TessellateEnv()
            state = env.reset()
            game_reward = 0
            steps = 0
            
            while not env.is_terminal():
                valid_actions = env.get_valid_actions()
                if not valid_actions:
                    break
                
                # Agent plays Red (player 1)
                if env.game.current_turn == 1:
                    # Use trained Q-network
                    with torch.no_grad():
                        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                        q_values = self.model(state_tensor).squeeze(0).cpu().numpy()
                        
                        # Mask invalid actions
                        q_values_masked = np.full(100, -np.inf)
                        q_values_masked[valid_actions] = q_values[valid_actions]
                        
                        # Greedy action selection
                        action = np.argmax(q_values_masked)
                else:
                    # Random opponent plays Blue
                    action = np.random.choice(valid_actions)
                
                state, reward, done, info = env.step(action)
                game_reward += reward
                steps += 1
                
                if done:
                    if info['winner'] == 1:
                        wins += 1
                    break
            
            total_reward += game_reward
            game_lengths.append(steps)
        
        self.model.train()
        
        win_rate = wins / n_games * 100
        avg_reward = total_reward / n_games
        avg_length = np.mean(game_lengths)
        
        return {
            'win_rate': win_rate,
            'avg_reward': avg_reward,
            'avg_game_length': avg_length,
            'n_games': n_games
        }
    
    def train(
        self,
        n_batches: int = 10000,
        eval_every: int = 1000,
        save_every: int = 5000,
        eval_games: int = 20
    ):
        """
        Main training loop.
        
        Args:
            n_batches: Number of batches to train on
            eval_every: Evaluate every N batches
            save_every: Save model every N batches
            eval_games: Number of games for evaluation
        """
        print(f"\nStarting PQN training")
        print(f"Training for {n_batches} batches")
        print(f"Batch size: {self.data_loader.batch_size}")
        print(f"Total transitions: {n_batches * self.data_loader.batch_size:,}")
        print("-" * 50)
        
        start_time = time.time()
        
        for batch_idx, batch in enumerate(self.data_loader.iterate_batches(max_batches=n_batches)):
            # Train on batch
            metrics = self.trainer.train_batch(batch)
            self.metrics_history.append(metrics)
            
            # Print progress
            if batch_idx % 100 == 0:
                elapsed = time.time() - start_time
                batches_per_sec = (batch_idx + 1) / elapsed
                transitions_per_sec = batches_per_sec * self.data_loader.batch_size
                
                print(f"Batch {batch_idx:5d}/{n_batches} | "
                      f"Loss: {metrics['total_loss']:8.4f} | "
                      f"TD: {metrics['td_loss']:8.4f} | "
                      f"Q-mean: {metrics['q_mean']:7.3f} | "
                      f"Q-std: {metrics['q_std']:6.3f} | "
                      f"Speed: {transitions_per_sec:,.0f} trans/sec")
            
            # Evaluate
            if (batch_idx + 1) % eval_every == 0:
                print(f"\n--- Evaluating at batch {batch_idx + 1} ---")
                eval_metrics = self.evaluate_vs_random(n_games=eval_games)
                eval_metrics['batch'] = batch_idx + 1
                eval_metrics['time'] = time.time() - start_time
                self.eval_history.append(eval_metrics)
                
                print(f"Win rate vs random: {eval_metrics['win_rate']:.1f}%")
                print(f"Avg reward: {eval_metrics['avg_reward']:.2f}")
                print(f"Avg game length: {eval_metrics['avg_game_length']:.1f}")
                print("-" * 50)
            
            # Save checkpoint
            if (batch_idx + 1) % save_every == 0:
                self.save_checkpoint(batch_idx + 1)
        
        # Final evaluation
        print("\n" + "=" * 50)
        print("FINAL EVALUATION")
        print("=" * 50)
        final_eval = self.evaluate_vs_random(n_games=100)
        print(f"Final win rate vs random: {final_eval['win_rate']:.1f}%")
        print(f"Final avg reward: {final_eval['avg_reward']:.2f}")
        
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time/60:.1f} minutes")
        print(f"Total transitions seen: {n_batches * self.data_loader.batch_size:,}")
        
        return final_eval
    
    def save_checkpoint(self, batch_idx: int):
        """Save model checkpoint and training history."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_dir = Path('checkpoints')
        checkpoint_dir.mkdir(exist_ok=True)
        
        # Save model
        model_path = checkpoint_dir / f'pqn_model_batch{batch_idx}_{timestamp}.pt'
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.trainer.optimizer.state_dict(),
            'batch_idx': batch_idx,
            'metrics_history': self.metrics_history[-1000:],  # Last 1000 batches
            'eval_history': self.eval_history
        }, model_path)
        
        print(f"Saved checkpoint to {model_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Loaded checkpoint from {checkpoint_path}")
        return checkpoint


def run_baseline_experiment():
    """
    Run baseline PQN experiment with sensible defaults.
    This demonstrates learning from random play data.
    """
    print("=" * 60)
    print("PQN BASELINE EXPERIMENT")
    print("Learning from random play data without replay buffer")
    print("=" * 60)
    
    # Create experiment with moderate settings
    experiment = PQNExperiment(
        data_dir='rl_data_large',
        max_files=10,  # Use 10 files = 1M transitions
        batch_size=10000,  # Large batch for stability
        learning_rate=1e-4,  # Conservative learning rate
        weight_decay=1e-3,  # L2 regularization
        gamma=0.99,
        hidden_dims=(256, 256)
    )
    
    # Train for a reasonable number of batches
    # 100 batches = 1M transitions seen
    final_metrics = experiment.train(
        n_batches=100,
        eval_every=20,
        save_every=50,
        eval_games=20
    )
    
    # Save final results
    results = {
        'final_win_rate': final_metrics['win_rate'],
        'final_avg_reward': final_metrics['avg_reward'],
        'eval_history': experiment.eval_history,
        'experiment_config': {
            'max_files': 10,
            'batch_size': 10000,
            'learning_rate': 1e-4,
            'weight_decay': 1e-3,
            'gamma': 0.99,
            'hidden_dims': (256, 256)
        }
    }
    
    results_path = Path('pqn_baseline_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_path}")
    
    return results


def run_quick_test():
    """
    Quick test to verify everything works.
    Uses minimal data for fast iteration.
    """
    print("Running quick test...")
    
    experiment = PQNExperiment(
        data_dir='rl_data_large',
        max_files=1,  # Only 1 file
        batch_size=1000,  # Smaller batches
        learning_rate=1e-4,
        weight_decay=1e-3
    )
    
    # Train for just a few batches
    experiment.train(
        n_batches=10,
        eval_every=5,
        save_every=100,
        eval_games=5
    )
    
    print("Quick test completed!")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--quick':
        run_quick_test()
    else:
        run_baseline_experiment()