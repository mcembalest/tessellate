#!/usr/bin/env python3
"""
Minimal RL training for Tessellate
Focus on interpretability with simplest possible model
No hand-crafted features - let the model learn patterns
"""

import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
import time
from typing import List, Dict, Tuple

# Game constants
BOARD_SIZE = 10
EMPTY = 0
RED = 1
BLUE = 2
BLOCKED = 3

class TessellateValueNet(nn.Module):
    """
    Value network that learns from both immediate score changes and final outcomes
    Input: Raw board state (10x10 = 100 values) + current player (1 value)
    Hidden: Single layer with 128 neurons
    Output: Single value estimate combining immediate and future rewards
    """
    def __init__(self, input_size=101, hidden_size=128):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.tanh(self.fc2(x))  # Output between -1 and 1
        return x

def board_to_tensor(board_state: List[List[int]], current_player: int) -> torch.Tensor:
    """
    Convert board state to tensor (no feature engineering!)
    Just flatten the board and append current player
    """
    # Flatten board to 100 values
    flat_board = np.array(board_state).flatten()
    # Append current player
    state = np.append(flat_board, current_player)
    return torch.FloatTensor(state)

def load_games(filepath: str, max_games: int = None) -> List[Dict]:
    """Load games from JSON file"""
    print(f"Loading games from {filepath}...")
    with open(filepath, 'r') as f:
        games = json.load(f)
    if max_games:
        games = games[:max_games]
    print(f"Loaded {len(games)} games")
    return games

def calculate_score_after_move(board: List[List[int]], player: int) -> int:
    """
    Calculate score for a player after move is made
    This reimplements the scoring logic from tessellate.py
    """
    visited = [[False for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    islands = []
    
    def get_neighbors(r, c):
        """Get neighbors using Tessellate adjacency rules"""
        neighbors = []
        pow_neg1_r = 1 if r % 2 == 0 else -1
        pow_neg1_c = 1 if c % 2 == 0 else -1
        pow_neg1_r_plus_1 = 1 if (r + 1) % 2 == 0 else -1
        pow_neg1_c_plus_1 = 1 if (c + 1) % 2 == 0 else -1
        pow_neg1_r_c_1 = 1 if (r + c + 1) % 2 == 0 else -1
        
        potential = [
            (r + pow_neg1_r, c + pow_neg1_c),
            (r - 1, c - pow_neg1_r_c_1),
            (r + 1, c + pow_neg1_r_c_1),
            (r + pow_neg1_r_plus_1, c),
            (r, c + pow_neg1_c_plus_1)
        ]
        
        for nr, nc in potential:
            if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                neighbors.append((nr, nc))
        return neighbors
    
    def dfs_island_size(start_r, start_c):
        """DFS to count island size"""
        if visited[start_r][start_c]:
            return 0
        
        stack = [(start_r, start_c)]
        size = 0
        
        while stack:
            r, c = stack.pop()
            if visited[r][c]:
                continue
            
            visited[r][c] = True
            size += 1
            
            for nr, nc in get_neighbors(r, c):
                if board[nr][nc] == player and not visited[nr][nc]:
                    stack.append((nr, nc))
        
        return size
    
    # Find all islands
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r][c] == player and not visited[r][c]:
                island_size = dfs_island_size(r, c)
                if island_size > 0:
                    islands.append(island_size)
    
    # Calculate score as product of island sizes
    if not islands:
        return 0
    
    score = 1
    for size in islands:
        score *= size
    return score

def reconstruct_board_states_with_values(game: Dict, gamma: float = 0.9) -> List[Tuple[np.ndarray, int, float]]:
    """
    Reconstruct board states with combined value targets
    Returns: List of (board_state, current_player, target_value)
    """
    states = []
    board = [[EMPTY for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    
    # Determine game outcome
    winner = game.get('winner')
    
    # Process each move
    for i, move in enumerate(game['moves']):
        player = move['player']
        r, c = move['position']
        
        # Store state BEFORE move
        board_copy = [row[:] for row in board]
        
        # Get score before move
        score_before = move['score_before'][str(player)]
        
        # Apply move temporarily to calculate score after
        board[r][c] = player
        
        # Block adjacent corners
        c_adj = c + (1 if c % 2 == 0 else -1)
        if 0 <= c_adj < BOARD_SIZE and board[r][c_adj] == EMPTY:
            board[r][c_adj] = BLOCKED
            
        r_adj = r + (1 if r % 2 == 0 else -1)
        if 0 <= r_adj < BOARD_SIZE and board[r_adj][c] == EMPTY:
            board[r_adj][c] = BLOCKED
        
        # Calculate score after move
        score_after = calculate_score_after_move(board, player)
        
        # Immediate reward: normalized score change
        # Normalize by dividing by 100 to keep values reasonable
        immediate_reward = (score_after - score_before) / 100.0
        
        # Future reward: discounted final outcome
        moves_remaining = len(game['moves']) - i - 1
        discount = gamma ** moves_remaining
        
        if winner == player:
            future_reward = discount * 1.0
        elif winner is None or winner == 0:
            future_reward = 0.0
        else:
            future_reward = discount * -1.0
        
        # Combine signals with weighting
        # Weight immediate rewards more early, future rewards more late
        progress = i / len(game['moves'])
        immediate_weight = 1.0 - progress * 0.5  # 1.0 -> 0.5
        future_weight = 0.5 + progress * 0.5     # 0.5 -> 1.0
        
        combined_value = immediate_weight * immediate_reward + future_weight * future_reward
        
        states.append((board_copy, player, combined_value))
    
    return states

def create_training_data(games: List[Dict], samples_per_game: int = 10, gamma: float = 0.9):
    """
    Create training data from games with dual-signal value targets
    """
    X = []
    y = []
    
    # Track statistics
    immediate_rewards = []
    future_rewards = []
    
    for game_idx, game in enumerate(games):
        if game_idx % 100 == 0:
            print(f"Processing game {game_idx}/{len(games)}")
        
        states = reconstruct_board_states_with_values(game, gamma)
        
        # Sample random positions from the game
        if len(states) > samples_per_game:
            indices = np.random.choice(len(states), samples_per_game, replace=False)
            sampled_states = [states[i] for i in indices]
        else:
            sampled_states = states
        
        for board, player, target_value in sampled_states:
            X.append(board_to_tensor(board, player))
            y.append(target_value)
    
    print(f"\nValue statistics:")
    print(f"  Mean: {np.mean(y):.3f}")
    print(f"  Std:  {np.std(y):.3f}")
    print(f"  Min:  {np.min(y):.3f}")
    print(f"  Max:  {np.max(y):.3f}")
    
    return torch.stack(X), torch.FloatTensor(y)

def train_model(model, X, y, epochs=10, batch_size=64, learning_rate=0.001):
    """Training loop for value prediction"""
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()  # For regression
    
    # Split into train/val
    n_samples = len(X)
    n_train = int(0.8 * n_samples)
    indices = torch.randperm(n_samples)
    
    X_train = X[indices[:n_train]]
    y_train = y[indices[:n_train]]
    X_val = X[indices[n_train:]]
    y_val = y[indices[n_train:]]
    
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
    
    print(f"\nTraining for {epochs} epochs...")
    print(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}")
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            
            predictions = model(batch_X).squeeze()
            loss = criterion(predictions, batch_y)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                predictions = model(batch_X).squeeze()
                loss = criterion(predictions, batch_y)
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss:   {avg_val_loss:.4f}")
    
    return model, train_losses, val_losses

def analyze_model(model, sample_games: List[Dict], n_examples: int = 3):
    """
    Analyze value predictions at different game stages
    """
    print("\n=== Model Analysis ===")
    
    for i in range(min(n_examples, len(sample_games))):
        game = sample_games[i]
        states = reconstruct_board_states_with_values(game)
        
        winner = game.get('winner', 0)
        winner_str = "Red" if winner == RED else "Blue" if winner == BLUE else "Tie"
        
        print(f"\nGame {i+1} - Winner: {winner_str}")
        print(f"Final scores - Red: {game['final_scores']['red']}, Blue: {game['final_scores']['blue']}")
        
        # Check predictions at different points
        checkpoints = [0, len(states)//4, len(states)//2, 3*len(states)//4, len(states)-1]
        
        for pos in checkpoints:
            if pos < len(states):
                board, player, true_value = states[pos]
                x = board_to_tensor(board, player).unsqueeze(0)
                
                with torch.no_grad():
                    pred_value = model(x).item()
                
                move_pct = (pos / len(states)) * 100
                player_str = "Red" if player == RED else "Blue"
                
                # Show if prediction aligns with outcome
                sign_correct = (pred_value > 0 and true_value > 0) or \
                              (pred_value < 0 and true_value < 0) or \
                              (abs(pred_value) < 0.1 and abs(true_value) < 0.1)
                
                print(f"  Move {pos+1}/{len(states)} ({move_pct:.0f}%) [{player_str}]: "
                      f"Pred: {pred_value:+.3f}, True: {true_value:+.3f} "
                      f"{'✓' if sign_correct else '✗'}")

def visualize_position_values(model):
    """
    Visualize model's value estimates for placing pieces at each position
    """
    print("\n=== Position Value Heatmap ===")
    print("(Empty board, RED to move)")
    print("Higher values = better for current player\n")
    
    # Empty board
    board = [[EMPTY for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    
    values = np.zeros((BOARD_SIZE, BOARD_SIZE))
    
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            # Temporarily place piece
            board[r][c] = RED
            x = board_to_tensor(board, BLUE).unsqueeze(0)  # Next player would be BLUE
            
            with torch.no_grad():
                value = model(x).item()
            values[r, c] = value
            
            # Remove piece
            board[r][c] = EMPTY
    
    # Normalize for visualization
    max_val = np.abs(values).max()
    
    # ASCII heatmap
    for r in range(BOARD_SIZE):
        row_str = ""
        for c in range(BOARD_SIZE):
            val = values[r, c]
            if val > max_val * 0.5:
                row_str += "█"  # Very good
            elif val > max_val * 0.25:
                row_str += "▓"  # Good
            elif val > -max_val * 0.25:
                row_str += "░"  # Neutral
            else:
                row_str += "·"  # Bad
        print(row_str)
    
    print(f"\nValue range: [{values.min():.3f}, {values.max():.3f}]")
    print(f"Best position: {np.unravel_index(values.argmax(), values.shape)}")
    print(f"Worst position: {np.unravel_index(values.argmin(), values.shape)}")

def main():
    # Load data - try to use larger dataset if available
    large_data_path = Path("game_data/all_games_20250815_141840_1000000.json")
    small_data_path = Path("game_data/random_games_1000.json")
    
    if large_data_path.exists():
        data_path = large_data_path
        max_games = 10000  # Use 10k games for better training
    else:
        data_path = small_data_path
        max_games = 1000
    
    if not data_path.exists():
        print(f"Data file not found: {data_path}")
        print("Please generate games first using: python generate_games.py")
        return
    
    # Load games
    games = load_games(data_path, max_games=max_games)
    
    # Create training data with dual signals
    print("\nCreating training data with dual signals...")
    print("  - Immediate reward: score change from move")
    print("  - Future reward: discounted final outcome")
    X, y = create_training_data(games, samples_per_game=10, gamma=0.9)
    print(f"Training samples: {len(X)}")
    
    # Create model
    model = TessellateValueNet()
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Train - more epochs for larger dataset
    epochs = 30 if max_games > 1000 else 20
    model, train_losses, val_losses = train_model(model, X, y, epochs=epochs, batch_size=64)
    
    # Analyze
    analyze_model(model, games[:5])
    
    # Visualize
    visualize_position_values(model)
    
    # Save model
    torch.save(model.state_dict(), "tessellate_value_model.pt")
    print("\nModel saved to tessellate_value_model.pt")

if __name__ == "__main__":
    main()