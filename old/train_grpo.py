#!/usr/bin/env python3
"""
Main GRPO training script for Tessellate.
Ties together all components for end-to-end training.
"""

import json
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import argparse
from typing import List, Dict

from grpo_trainer_fixed import GRPOTrainer, GRPOConfig
from game_state_utils import reconstruct_game_state


def load_trajectories(data_dir: Path, max_files: int = None) -> List[Dict]:
    """Load game trajectories from JSONL files."""
    trajectories = []
    
    files = sorted(data_dir.glob('*_trajectories_*.jsonl'))
    if max_files:
        files = files[:max_files]
    
    for file_path in files:
        with open(file_path, 'r') as f:
            for line in f:
                trajectories.append(json.loads(line))
                if max_files and len(trajectories) >= max_files * 100:
                    return trajectories
    
    return trajectories


def main():
    parser = argparse.ArgumentParser(description='GRPO Training for Tessellate')
    
    # Model settings
    parser.add_argument('--model', default='Qwen/Qwen3-4B-Thinking-2507', 
                       help='HuggingFace model ID')
    parser.add_argument('--device', default='cpu', 
                       help='Device (mps, cuda, cpu)')
    
    # Data settings
    parser.add_argument('--data-dir', default='llm_data_with_reasoning',
                       help='Directory with trajectory data')
    parser.add_argument('--max-trajectories', type=int, default=100,
                       help='Maximum trajectories to load')
    
    # Training settings
    parser.add_argument('--batch-size', type=int, default=2,
                       help='Batch size (number of games)')
    parser.add_argument('--k-completions', type=int, default=2,
                       help='Number of completions per state')
    parser.add_argument('--learning-rate', type=float, default=1e-6,
                       help='Learning rate')
    parser.add_argument('--num-epochs', type=int, default=1,
                       help='Number of training epochs')
    parser.add_argument('--steps-per-epoch', type=int, default=100,
                       help='Training steps per epoch')
    
    # GRPO settings
    parser.add_argument('--kl-coef', type=float, default=0.01,
                       help='KL divergence coefficient')
    parser.add_argument('--clip-range', type=float, default=0.2,
                       help='PPO clip range')
    parser.add_argument('--reward-scaling', type=float, default=0.001,
                       help='Reward scaling factor')
    
    # Output settings
    parser.add_argument('--output-dir', default='grpo_checkpoints',
                       help='Directory for checkpoints')
    parser.add_argument('--save-every', type=int, default=50,
                       help='Save checkpoint every N steps')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 50)
    print("GRPO Training for Tessellate")
    print("=" * 50)
    
    # Load model and tokenizer
    print(f"\n1. Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if args.device != "cpu" else torch.float32,
        device_map="auto"
    )
    
    # Create GRPO config
    config = GRPOConfig(
        k_completions=args.k_completions,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        clip_range=args.clip_range,
        kl_coef=args.kl_coef,
        reward_scaling=args.reward_scaling
    )
    
    print(f"\n2. GRPO Configuration:")
    print(f"   K completions: {config.k_completions}")
    print(f"   Batch size: {config.batch_size}")
    print(f"   Learning rate: {config.learning_rate}")
    print(f"   KL coefficient: {config.kl_coef}")
    print(f"   Reward scaling: {config.reward_scaling}")
    
    # Initialize trainer
    trainer = GRPOTrainer(model, tokenizer, config, device=args.device)
    
    # Set reference model (Option B: after initial reasoning generation)
    print("\n3. Setting reference model for KL divergence")
    trainer.set_reference_model()
    
    # Load trajectories
    print(f"\n4. Loading trajectories from {args.data_dir}")
    trajectories = load_trajectories(Path(args.data_dir), args.max_trajectories)
    print(f"   Loaded {len(trajectories)} trajectories")
    
    if len(trajectories) == 0:
        print("ERROR: No trajectories found!")
        print("Please run these steps first:")
        print("1. python prepare_llm_data.py")
        print("2. python generate_reasoning.py")
        return
    
    # Training loop
    print(f"\n5. Starting GRPO training")
    print(f"   Epochs: {args.num_epochs}")
    print(f"   Steps per epoch: {args.steps_per_epoch}")
    
    global_step = 0
    
    for epoch in range(args.num_epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{args.num_epochs}")
        print(f"{'='*50}")
        
        epoch_metrics = {
            'loss': [],
            'reward': [],
            'advantage': []
        }
        
        pbar = tqdm(range(args.steps_per_epoch), desc=f"Epoch {epoch + 1}")
        
        for step in pbar:
            # Sample batch of trajectories
            import random
            batch_trajectories = random.sample(
                trajectories, 
                min(args.batch_size, len(trajectories))
            )
            
            # Train step
            metrics = trainer.train_step(batch_trajectories)
            
            # Update metrics
            epoch_metrics['loss'].append(metrics['loss'])
            epoch_metrics['reward'].append(metrics['mean_reward'])
            epoch_metrics['advantage'].append(metrics['mean_advantage'])
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{metrics['loss']:.4f}",
                'reward': f"{metrics['mean_reward']:.2f}",
                'adv': f"{metrics['mean_advantage']:.3f}"
            })
            
            global_step += 1
            
            # Save checkpoint
            if global_step % args.save_every == 0:
                checkpoint_path = output_dir / f"checkpoint_{global_step}"
                print(f"\nSaving checkpoint to {checkpoint_path}")
                model.save_pretrained(checkpoint_path)
                tokenizer.save_pretrained(checkpoint_path)
        
        # Print epoch summary
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Avg Loss: {sum(epoch_metrics['loss'])/len(epoch_metrics['loss']):.4f}")
        print(f"  Avg Reward: {sum(epoch_metrics['reward'])/len(epoch_metrics['reward']):.2f}")
        print(f"  Avg Advantage: {sum(epoch_metrics['advantage'])/len(epoch_metrics['advantage']):.3f}")
    
    # Save final model
    final_path = output_dir / "final_model"
    print(f"\n6. Saving final model to {final_path}")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    
    print("\n" + "="*50)
    print("Training Complete!")
    print("="*50)
    print(f"\nModel saved to: {final_path}")
    print("\nTo test the trained model:")
    print("1. Load from checkpoint")
    print("2. Use llm_game_interface.py to play games")
    print("3. Compare performance vs baseline")


if __name__ == "__main__":
    main()