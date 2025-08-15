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
    Simplest possible value network
    Input: Raw board state (10x10 = 100 values) + current player (1 value)
    Hidden: Single layer with 128 neurons
    Output: Value estimate (who's winning)
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

def reconstruct_board_states(game: Dict) -> List[Tuple[np.ndarray, int, float]]:
    """
    Reconstruct board states from a game's moves
    Returns: List of (board_state, current_player, value)
    """
    states = []
    board = [[EMPTY for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    
    # Determine game outcome for value labels
    winner = game.get('winner')
    if winner == RED:
        final_value = 1.0  # Red wins
    elif winner == BLUE:
        final_value = -1.0  # Blue wins
    else:
        final_value = 0.0  # Tie
    
    # Process each move
    for i, move in enumerate(game['moves']):
        player = move['player']
        r, c = move['position']
        
        # Store state BEFORE move (for learning)
        board_copy = [row[:] for row in board]
        
        # Value estimate: linear interpolation from 0 to final value
        # This is crude but simple - better would be TD learning
        progress = i / len(game['moves'])
        value = final_value * progress
        
        states.append((board_copy, player, value))
        
        # Apply move
        board[r][c] = player
        
        # Block adjacent corners
        c_adj = c + (1 if c % 2 == 0 else -1)
        if 0 <= c_adj < BOARD_SIZE and board[r][c_adj] == EMPTY:
            board[r][c_adj] = BLOCKED
            
        r_adj = r + (1 if r % 2 == 0 else -1)
        if 0 <= r_adj < BOARD_SIZE and board[r_adj][c] == EMPTY:
            board[r_adj][c] = BLOCKED
    
    return states

def create_training_data(games: List[Dict], samples_per_game: int = 5):
    """
    Create training data from games
    Sample a few positions from each game to avoid correlation
    """
    X = []
    y = []
    
    for game_idx, game in enumerate(games):
        if game_idx % 100 == 0:
            print(f"Processing game {game_idx}/{len(games)}")
        
        states = reconstruct_board_states(game)
        
        # Sample random positions from the game
        if len(states) > samples_per_game:
            indices = np.random.choice(len(states), samples_per_game, replace=False)
            sampled_states = [states[i] for i in indices]
        else:
            sampled_states = states
        
        for board, player, value in sampled_states:
            X.append(board_to_tensor(board, player))
            y.append(value)
    
    return torch.stack(X), torch.FloatTensor(y)

def train_model(model, X, y, epochs=10, batch_size=64, learning_rate=0.001):
    """Simple training loop"""
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    dataset = torch.utils.data.TensorDataset(X, y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print(f"\nTraining for {epochs} epochs...")
    for epoch in range(epochs):
        total_loss = 0
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            
            predictions = model(batch_X).squeeze()
            loss = criterion(predictions, batch_y)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    return model

def analyze_model(model, sample_games: List[Dict], n_examples: int = 3):
    """
    Analyze what the model learned
    Show value predictions for different board positions
    """
    print("\n=== Model Analysis ===")
    
    for i in range(min(n_examples, len(sample_games))):
        game = sample_games[i]
        states = reconstruct_board_states(game)
        
        print(f"\nGame {i+1} - Winner: {game.get('winner', 'Tie')}")
        print(f"Final scores - Red: {game['final_scores']['red']}, Blue: {game['final_scores']['blue']}")
        
        # Check predictions at different points
        positions = [0, len(states)//4, len(states)//2, 3*len(states)//4, len(states)-1]
        
        for pos in positions:
            if pos < len(states):
                board, player, true_value = states[pos]
                x = board_to_tensor(board, player).unsqueeze(0)
                
                with torch.no_grad():
                    pred_value = model(x).item()
                
                move_pct = (pos / len(states)) * 100
                print(f"  Move {pos}/{len(states)} ({move_pct:.0f}%): "
                      f"Predicted: {pred_value:+.3f}, True: {true_value:+.3f}")

def visualize_value_heatmap(model):
    """
    Create a heatmap showing model's value estimates for placing RED at each position
    This shows which positions the model thinks are good
    """
    print("\n=== Position Value Heatmap (RED's perspective) ===")
    print("Higher values = better for RED, Lower = better for BLUE")
    
    # Empty board
    board = [[EMPTY for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    
    values = np.zeros((BOARD_SIZE, BOARD_SIZE))
    
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r][c] == EMPTY:
                # Temporarily place RED piece
                board[r][c] = RED
                x = board_to_tensor(board, BLUE).unsqueeze(0)  # Next player would be BLUE
                
                with torch.no_grad():
                    value = model(x).item()
                values[r, c] = value
                
                # Remove piece
                board[r][c] = EMPTY
    
    # Simple ASCII heatmap
    for r in range(BOARD_SIZE):
        row_str = ""
        for c in range(BOARD_SIZE):
            val = values[r, c]
            if val > 0.1:
                row_str += "█"  # Very good for RED
            elif val > 0:
                row_str += "▓"  # Good for RED
            elif val > -0.1:
                row_str += "░"  # Neutral
            else:
                row_str += "·"  # Good for BLUE
        print(row_str)

def main():
    # Load data
    data_path = Path("game_data/random_games_1000.json")
    
    # For full training, use the 1M game file:
    # data_path = Path("game_data/all_games_20250815_141840_1000000.json")
    
    if not data_path.exists():
        print(f"Data file not found: {data_path}")
        print("Please generate games first using: python generate_games.py")
        return
    
    # Load games (use subset for quick testing)
    games = load_games(data_path, max_games=100)  # Use 1000 for real training
    
    # Create training data
    print("\nCreating training data...")
    X, y = create_training_data(games, samples_per_game=5)
    print(f"Training samples: {len(X)}")
    
    # Create model
    model = TessellateValueNet()
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Train
    model = train_model(model, X, y, epochs=5, batch_size=32)
    
    # Analyze
    analyze_model(model, games[:5])
    
    # Visualize
    visualize_value_heatmap(model)
    
    # Save model
    torch.save(model.state_dict(), "tessellate_value_model.pt")
    print("\nModel saved to tessellate_value_model.pt")

if __name__ == "__main__":
    main()