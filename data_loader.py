#!/usr/bin/env python3
"""
Efficient data loader for Tessellate preprocessed RL data.
Loads NPZ files from rl_data_large/ and provides batched iteration.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import random


class TessellateDataLoader:
    """
    Efficiently load and iterate through preprocessed Tessellate data.
    
    Features:
    - Lazy loading of NPZ files (one at a time)
    - Configurable batch sizes
    - Optional shuffling within files
    - Memory-efficient streaming
    """
    
    def __init__(
        self, 
        data_dir: str = 'rl_data_large',
        max_files: Optional[int] = None,
        batch_size: int = 10000,
        shuffle_files: bool = True,
        shuffle_within_files: bool = True,
        verbose: bool = True
    ):
        """
        Args:
            data_dir: Directory containing rl_data_*.npz files
            max_files: Maximum number of files to use (None = all)
            batch_size: Number of transitions per batch
            shuffle_files: Whether to shuffle file order
            shuffle_within_files: Whether to shuffle transitions within each file
            verbose: Print loading information
        """
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.shuffle_within_files = shuffle_within_files
        self.verbose = verbose
        
        # Find all NPZ files
        self.files = sorted(self.data_dir.glob('rl_data_*.npz'))
        if max_files:
            self.files = self.files[:max_files]
            
        if shuffle_files:
            random.shuffle(self.files)
            
        if verbose:
            print(f"Found {len(self.files)} NPZ files in {data_dir}/")
            if len(self.files) > 0:
                # Check first file to get data statistics
                with np.load(self.files[0]) as data:
                    n_transitions = data['states'].shape[0]
                    state_dim = data['states'].shape[1]
                    print(f"Each file contains {n_transitions:,} transitions")
                    print(f"State dimension: {state_dim}")
                    print(f"Total transitions available: {len(self.files) * n_transitions:,}")
        
        self.current_file_idx = 0
        self.current_data = None
        self.current_position = 0
        
    def load_next_file(self) -> bool:
        """
        Load the next NPZ file into memory.
        Returns True if successful, False if no more files.
        """
        if self.current_file_idx >= len(self.files):
            return False
            
        file_path = self.files[self.current_file_idx]
        if self.verbose:
            print(f"Loading file {self.current_file_idx + 1}/{len(self.files)}: {file_path.name}")
        
        # Load NPZ file
        data = np.load(file_path)
        
        # Extract all arrays
        self.current_data = {
            'states': data['states'],
            'actions': data['actions'],
            'rewards': data['rewards'],
            'next_states': data['next_states'],
            'masks': data['masks'],
            'dones': data['dones']
        }
        data.close()
        
        # Shuffle within file if requested
        if self.shuffle_within_files:
            n = len(self.current_data['states'])
            indices = np.random.permutation(n)
            for key in self.current_data:
                self.current_data[key] = self.current_data[key][indices]
        
        self.current_file_idx += 1
        self.current_position = 0
        return True
    
    def get_batch(self) -> Optional[Dict[str, np.ndarray]]:
        """
        Get next batch of transitions.
        Returns None when no more data available.
        """
        # Load first/next file if needed
        if self.current_data is None:
            if not self.load_next_file():
                return None
        
        # Check if we need to load next file
        while self.current_position >= len(self.current_data['states']):
            if not self.load_next_file():
                return None
        
        # Get batch from current file
        start = self.current_position
        end = min(start + self.batch_size, len(self.current_data['states']))
        
        batch = {
            'states': self.current_data['states'][start:end],
            'actions': self.current_data['actions'][start:end],
            'rewards': self.current_data['rewards'][start:end],
            'next_states': self.current_data['next_states'][start:end],
            'masks': self.current_data['masks'][start:end],
            'dones': self.current_data['dones'][start:end]
        }
        
        self.current_position = end
        
        # If batch is smaller than requested and we have more files, fill it
        if len(batch['states']) < self.batch_size and self.current_file_idx < len(self.files):
            if self.load_next_file():
                remaining = self.batch_size - len(batch['states'])
                next_batch = self.get_batch()
                if next_batch:
                    for key in batch:
                        batch[key] = np.concatenate([batch[key], next_batch[key][:remaining]])
        
        return batch
    
    def iterate_batches(self, max_batches: Optional[int] = None):
        """
        Generator to iterate through all batches.
        
        Args:
            max_batches: Maximum number of batches to yield (None = all)
        """
        batch_count = 0
        while True:
            if max_batches and batch_count >= max_batches:
                break
                
            batch = self.get_batch()
            if batch is None:
                break
                
            yield batch
            batch_count += 1
    
    def reset(self):
        """Reset to beginning of dataset."""
        self.current_file_idx = 0
        self.current_data = None
        self.current_position = 0
    
    def get_statistics(self) -> Dict:
        """
        Compute statistics over a sample of the data.
        Useful for understanding reward distribution, state ranges, etc.
        """
        print("Computing dataset statistics (sampling first file)...")
        
        # Load first file for statistics
        with np.load(self.files[0]) as data:
            states = data['states']
            actions = data['actions']
            rewards = data['rewards']
            dones = data['dones']
            
            stats = {
                'n_transitions': len(states),
                'state_mean': np.mean(states, axis=0),
                'state_std': np.std(states, axis=0),
                'state_min': np.min(states, axis=0),
                'state_max': np.max(states, axis=0),
                'action_counts': np.bincount(actions.astype(int), minlength=100),
                'reward_mean': np.mean(rewards),
                'reward_std': np.std(rewards),
                'reward_min': np.min(rewards),
                'reward_max': np.max(rewards),
                'reward_nonzero_pct': np.mean(rewards != 0) * 100,
                'terminal_state_pct': np.mean(dones) * 100,
                'unique_rewards': np.unique(rewards)
            }
            
        return stats


def test_loader():
    """Test the data loader functionality."""
    print("Testing TessellateDataLoader...\n")
    
    # Create loader with small batch size for testing
    loader = TessellateDataLoader(
        data_dir='rl_data_large',
        max_files=2,  # Only use 2 files for testing
        batch_size=1000,
        shuffle_files=False,
        verbose=True
    )
    
    # Test iteration
    print("\nTesting batch iteration:")
    for i, batch in enumerate(loader.iterate_batches(max_batches=3)):
        print(f"  Batch {i}: {batch['states'].shape[0]} transitions")
        print(f"    State shape: {batch['states'].shape}")
        print(f"    Actions shape: {batch['actions'].shape}")
        print(f"    Rewards: {batch['rewards'][:5]}...")
        
    # Get statistics
    print("\nDataset statistics:")
    stats = loader.get_statistics()
    print(f"  Reward range: [{stats['reward_min']:.2f}, {stats['reward_max']:.2f}]")
    print(f"  Reward mean: {stats['reward_mean']:.4f}")
    print(f"  Non-zero rewards: {stats['reward_nonzero_pct']:.2f}%")
    print(f"  Terminal states: {stats['terminal_state_pct']:.2f}%")
    print(f"  Unique rewards: {stats['unique_rewards']}")
    
    # Check action distribution (should be roughly uniform for random play)
    action_probs = stats['action_counts'] / stats['action_counts'].sum()
    print(f"  Action entropy: {-np.sum(action_probs * np.log(action_probs + 1e-10)):.2f}")
    print(f"    (Max entropy for 100 actions: {np.log(100):.2f})")


if __name__ == "__main__":
    test_loader()