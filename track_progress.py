#!/usr/bin/env python3
"""
Track model training progress over iterations
Helps understand if the model is learning meaningful strategies
"""

import json
import torch
import numpy as np
from pathlib import Path
from evaluate_agents import evaluate_model_vs_random
from train_rl import TessellateValueNet, load_games, create_training_data, train_model
import matplotlib.pyplot as plt

def train_and_evaluate_iterations(n_iterations=5, games_per_iteration=[1000, 2000, 5000, 10000, 20000]):
    """
    Train models with increasing amounts of data and track performance
    """
    results = []
    
    # Check which data files exist
    large_data_path = Path("game_data/all_games_20250815_141840_1000000.json")
    small_data_path = Path("game_data/random_games_1000.json")
    
    if large_data_path.exists():
        data_path = large_data_path
        print(f"Using large dataset: {data_path}")
    else:
        data_path = small_data_path
        print(f"Using small dataset: {data_path}")
        games_per_iteration = [100, 200, 500, 1000]  # Adjust for smaller dataset
    
    for i, n_games in enumerate(games_per_iteration):
        print(f"\n{'='*60}")
        print(f"Iteration {i+1}/{len(games_per_iteration)}: Training on {n_games} games")
        print(f"{'='*60}")
        
        # Load games
        games = load_games(data_path, max_games=n_games)
        
        # Create training data
        print("\nCreating training data...")
        X, y = create_training_data(games, samples_per_game=10, gamma=0.9)
        
        # Train model
        model = TessellateValueNet()
        epochs = min(30, 10 + n_games // 500)  # Scale epochs with data
        print(f"Training for {epochs} epochs...")
        model, train_losses, val_losses = train_model(model, X, y, epochs=epochs, batch_size=64)
        
        # Save this iteration's model
        model_path = f"tessellate_model_iter_{i+1}_{n_games}_games.pt"
        torch.save(model.state_dict(), model_path)
        
        # Evaluate
        print(f"\nEvaluating model trained on {n_games} games...")
        win_rate = evaluate_model_vs_random(model_path, n_games=50, verbose=False)
        
        # Store results
        result = {
            'iteration': i+1,
            'n_training_games': n_games,
            'n_training_samples': len(X),
            'epochs': epochs,
            'final_train_loss': train_losses[-1],
            'final_val_loss': val_losses[-1],
            'win_rate_vs_random': win_rate
        }
        results.append(result)
        
        print(f"\nResults:")
        print(f"  Training games: {n_games}")
        print(f"  Training samples: {len(X)}")
        print(f"  Final train loss: {train_losses[-1]:.4f}")
        print(f"  Final val loss: {val_losses[-1]:.4f}")
        print(f"  Win rate vs Random: {win_rate:.1f}%")
    
    # Save all results
    with open("training_progress.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print("TRAINING PROGRESS SUMMARY")
    print(f"{'='*60}")
    print(f"{'Games':<10} {'Samples':<10} {'Train Loss':<12} {'Val Loss':<12} {'Win Rate':<10}")
    print("-" * 60)
    
    for r in results:
        print(f"{r['n_training_games']:<10} {r['n_training_samples']:<10} "
              f"{r['final_train_loss']:<12.4f} {r['final_val_loss']:<12.4f} "
              f"{r['win_rate_vs_random']:<10.1f}%")
    
    # Plot if we have multiple iterations
    if len(results) > 1:
        games_list = [r['n_training_games'] for r in results]
        win_rates = [r['win_rate_vs_random'] for r in results]
        
        print(f"\nWin Rate Progression:")
        for g, w in zip(games_list, win_rates):
            bar_length = int(w / 2)  # Scale to 50 chars max
            bar = "█" * bar_length
            print(f"{g:>6} games: {bar} {w:.1f}%")
    
    return results

def analyze_learning_features(model_path="tessellate_value_model.pt"):
    """
    Analyze what features the model has learned
    """
    print("\n=== Analyzing Learned Features ===")
    
    # Load model
    model = TessellateValueNet()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Analyze first layer weights
    first_layer_weights = model.fc1.weight.data.numpy()
    
    # Board position weights (first 100 weights correspond to board positions)
    board_weights = first_layer_weights[:, :100]
    
    # Find neurons that respond strongly to specific patterns
    print("\nTop 5 neurons by weight variance (likely feature detectors):")
    neuron_variances = np.var(board_weights, axis=1)
    top_neurons = np.argsort(neuron_variances)[-5:][::-1]
    
    for i, neuron_idx in enumerate(top_neurons, 1):
        weights = board_weights[neuron_idx]
        max_weight_pos = np.argmax(np.abs(weights))
        r, c = max_weight_pos // 10, max_weight_pos % 10
        print(f"  {i}. Neuron {neuron_idx}: Max weight at position ({r},{c}), "
              f"variance: {neuron_variances[neuron_idx]:.4f}")
    
    # Check if model learned position preferences
    avg_position_importance = np.mean(np.abs(board_weights), axis=0)
    position_grid = avg_position_importance.reshape(10, 10)
    
    print("\nAverage position importance (darker = more important):")
    max_importance = position_grid.max()
    for r in range(10):
        row_str = ""
        for c in range(10):
            val = position_grid[r, c] / max_importance
            if val > 0.75:
                row_str += "█"
            elif val > 0.5:
                row_str += "▓"
            elif val > 0.25:
                row_str += "░"
            else:
                row_str += "·"
        print(row_str)
    
    # Player preference (weight 100 is the current player indicator)
    player_weights = first_layer_weights[:, 100]
    print(f"\nPlayer indicator statistics:")
    print(f"  Mean weight: {np.mean(player_weights):.4f}")
    print(f"  Std weight: {np.std(player_weights):.4f}")
    print(f"  Min/Max: [{np.min(player_weights):.4f}, {np.max(player_weights):.4f}]")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--analyze":
        # Just analyze existing model
        analyze_learning_features()
    else:
        # Run progressive training
        results = train_and_evaluate_iterations()
        
        # Analyze final model
        analyze_learning_features()