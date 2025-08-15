#!/usr/bin/env python3
"""
Agent evaluation framework for Tessellate
Compares trained value network against random baseline
"""

import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from tessellate import TessellateGame, TileState
from agents import RandomAgent, play_game
import time

# Constants
BOARD_SIZE = 10
EMPTY = 0
RED = 1
BLUE = 2
BLOCKED = 3

class TessellateValueNet(nn.Module):
    """
    Value network for Tessellate
    Predicts position value combining immediate and future rewards
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

class ModelAgent:
    """
    Agent that uses a trained value network to select moves
    Evaluates each legal move and picks the one with highest value
    """
    
    def __init__(self, model_path: str = None, model: nn.Module = None):
        if model:
            self.model = model
        elif model_path:
            # Load model from file
            self.model = TessellateValueNet()
            self.model.load_state_dict(torch.load(model_path))
        else:
            raise ValueError("Must provide either model or model_path")
        
        self.model.eval()
    
    def select_move(self, game: TessellateGame) -> Optional[Tuple[int, int]]:
        """Select move with highest value estimate"""
        valid_moves = game.get_valid_moves()
        if not valid_moves:
            return None
        
        best_move = None
        best_value = -float('inf')
        
        current_player = game.current_turn
        
        # Evaluate each valid move
        for r, c in valid_moves:
            # Simulate move
            old_board = [row[:] for row in game.board]
            
            # Make move temporarily
            game.board[r][c] = current_player
            
            # Block adjacent corners
            c_adj = c + (1 if c % 2 == 0 else -1)
            if 0 <= c_adj < BOARD_SIZE and game.board[r][c_adj] == EMPTY:
                game.board[r][c_adj] = BLOCKED
            r_adj = r + (1 if r % 2 == 0 else -1)
            if 0 <= r_adj < BOARD_SIZE and game.board[r_adj][c] == EMPTY:
                game.board[r_adj][c] = BLOCKED
            
            # Get value estimate for this state
            next_player = BLUE if current_player == RED else RED
            board_tensor = self.board_to_tensor(game.board, next_player)
            
            with torch.no_grad():
                value = self.model(board_tensor.unsqueeze(0)).item()
            
            # If current player is BLUE, negate value (since model trained from RED perspective)
            if current_player == BLUE:
                value = -value
            
            if value > best_value:
                best_value = value
                best_move = (r, c)
            
            # Restore board
            game.board = [row[:] for row in old_board]
        
        return best_move
    
    def board_to_tensor(self, board: List[List[int]], current_player: int) -> torch.Tensor:
        """Convert board state to tensor"""
        flat_board = np.array(board).flatten()
        state = np.append(flat_board, current_player)
        return torch.FloatTensor(state)

def evaluate_model_vs_random(model_path: str = None, n_games: int = 100, verbose: bool = True):
    """
    Evaluate trained model against random agent
    Returns win rate of model
    """
    if not model_path:
        model_path = "tessellate_value_model.pt"
    
    if not Path(model_path).exists():
        print(f"Model not found at {model_path}")
        print("Please train a model first using: python train_rl.py")
        return None
    
    # Load model
    model = TessellateValueNet()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    model_agent = ModelAgent(model=model)
    random_agent = RandomAgent()
    
    results = {
        'model_wins': 0,
        'random_wins': 0,
        'ties': 0,
        'model_total_score': 0,
        'random_total_score': 0
    }
    
    if verbose:
        print(f"\nEvaluating Model vs Random over {n_games} games...")
    
    # Play half games with model as RED
    for game_num in range(n_games // 2):
        game_record = play_game(model_agent, random_agent, verbose=False)
        
        if game_record['winner'] == RED:
            results['model_wins'] += 1
        elif game_record['winner'] == BLUE:
            results['random_wins'] += 1
        else:
            results['ties'] += 1
        
        results['model_total_score'] += game_record['final_scores']['red']
        results['random_total_score'] += game_record['final_scores']['blue']
        
        if verbose and (game_num + 1) % 10 == 0:
            print(f"  Games {game_num + 1}/{n_games // 2} (Model as RED)...")
    
    # Play half games with model as BLUE
    for game_num in range(n_games // 2):
        game_record = play_game(random_agent, model_agent, verbose=False)
        
        if game_record['winner'] == BLUE:
            results['model_wins'] += 1
        elif game_record['winner'] == RED:
            results['random_wins'] += 1
        else:
            results['ties'] += 1
        
        results['model_total_score'] += game_record['final_scores']['blue']
        results['random_total_score'] += game_record['final_scores']['red']
        
        if verbose and (game_num + 1) % 10 == 0:
            print(f"  Games {game_num + 1}/{n_games // 2} (Model as BLUE)...")
    
    # Calculate statistics
    model_win_rate = 100 * results['model_wins'] / n_games
    random_win_rate = 100 * results['random_wins'] / n_games
    tie_rate = 100 * results['ties'] / n_games
    
    avg_model_score = results['model_total_score'] / n_games
    avg_random_score = results['random_total_score'] / n_games
    
    if verbose:
        print(f"\n=== Results ===")
        print(f"Model win rate:  {model_win_rate:.1f}%")
        print(f"Random win rate: {random_win_rate:.1f}%")
        print(f"Tie rate:        {tie_rate:.1f}%")
        print(f"\nAverage scores:")
        print(f"Model:  {avg_model_score:.1f}")
        print(f"Random: {avg_random_score:.1f}")
    
    return model_win_rate

def main():
    """
    Evaluate trained model against random baseline
    """
    print("=== Tessellate Model Evaluation ===")
    
    # Check if trained model exists
    model_path = Path("tessellate_value_model.pt")
    if not model_path.exists():
        print(f"No trained model found at {model_path}")
        print("Please train a model first using: python train_rl.py")
        return
    
    # Run evaluation
    start_time = time.time()
    win_rate = evaluate_model_vs_random(str(model_path), n_games=100, verbose=True)
    elapsed = time.time() - start_time
    
    print(f"\n=== Evaluation Complete ===")
    print(f"Time: {elapsed:.1f} seconds")
    print(f"Final Model Win Rate: {win_rate:.1f}%")
    
    # Save results
    results = {
        'model_win_rate': win_rate,
        'n_games': 100,
        'timestamp': time.strftime('%Y%m%d_%H%M%S')
    }
    
    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to evaluation_results.json")

if __name__ == "__main__":
    main()