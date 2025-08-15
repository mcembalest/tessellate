#!/usr/bin/env python3
"""
Agent implementations for Tessellate
"""

import numpy as np
import random
import json
import time
from typing import List, Dict, Any
from tessellate import TessellateGame, TileState

class RandomAgent:
    """Agent that plays random valid moves"""
    
    def __init__(self, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def select_move(self, game: TessellateGame) -> tuple:
        """Select a random valid move"""
        valid_moves = game.get_valid_moves()
        if not valid_moves:
            return None
        return random.choice(valid_moves)

def play_game(agent1, agent2, verbose=False) -> Dict[str, Any]:
    """Play a full game between two agents"""
    game = TessellateGame()
    agents = {TileState.RED: agent1, TileState.BLUE: agent2}
    
    move_count = 0
    game_record = {
        'moves': [],
        'final_scores': None,
        'winner': None,
        'total_moves': 0
    }
    
    while not game.game_over:
        current_agent = agents[game.current_turn]
        move = current_agent.select_move(game)
        
        if move is None:
            break
            
        if verbose and move_count % 10 == 0:
            print(f"Move {move_count}: {game.current_turn} plays {move}")
        
        # Record the move
        game_record['moves'].append({
            'player': int(game.current_turn),
            'position': move,
            'score_before': game.scores.copy()
        })
        
        # Make the move
        success = game.make_move(move[0], move[1])
        if not success:
            print(f"Failed move: {move}")
            break
            
        move_count += 1
    
    # Record final state
    game_record['final_scores'] = {
        'red': game.scores[TileState.RED],
        'blue': game.scores[TileState.BLUE]
    }
    game_record['winner'] = int(game.get_winner()) if game.get_winner() else None
    game_record['total_moves'] = move_count
    
    if verbose:
        print(f"Game finished in {move_count} moves")
        print(f"Final scores: RED={game.scores[TileState.RED]}, BLUE={game.scores[TileState.BLUE]}")
        print(f"Winner: {game.get_winner()}")
    
    return game_record

def generate_games(n_games: int, verbose=False) -> List[Dict[str, Any]]:
    """Generate n_games using random agents"""
    games = []
    start_time = time.time()
    
    for i in range(n_games):
        if verbose and i % 10 == 0:
            print(f"Generating game {i+1}/{n_games}")
        
        # Create new random agents for each game (different seeds)
        agent1 = RandomAgent()
        agent2 = RandomAgent()
        
        game_record = play_game(agent1, agent2, verbose=False)
        games.append(game_record)
    
    elapsed = time.time() - start_time
    
    if verbose:
        print(f"\nGenerated {n_games} games in {elapsed:.2f} seconds")
        print(f"Rate: {n_games/elapsed:.1f} games/second")
        
        # Statistics
        red_wins = sum(1 for g in games if g['winner'] == TileState.RED)
        blue_wins = sum(1 for g in games if g['winner'] == TileState.BLUE)
        ties = sum(1 for g in games if g['winner'] is None)
        
        avg_red_score = np.mean([g['final_scores']['red'] for g in games])
        avg_blue_score = np.mean([g['final_scores']['blue'] for g in games])
        max_score = max(max(g['final_scores']['red'], g['final_scores']['blue']) for g in games)
        
        print(f"\nStatistics:")
        print(f"  Red wins: {red_wins} ({100*red_wins/n_games:.1f}%)")
        print(f"  Blue wins: {blue_wins} ({100*blue_wins/n_games:.1f}%)")
        print(f"  Ties: {ties} ({100*ties/n_games:.1f}%)")
        print(f"  Avg RED score: {avg_red_score:.1f}")
        print(f"  Avg BLUE score: {avg_blue_score:.1f}")
        print(f"  Max score seen: {max_score}")
    
    return games

def save_games(games: List[Dict[str, Any]], filename: str):
    """Save games to JSON file"""
    with open(filename, 'w') as f:
        json.dump(games, f, indent=2)
    print(f"Saved {len(games)} games to {filename}")

def load_games(filename: str) -> List[Dict[str, Any]]:
    """Load games from JSON file"""
    with open(filename, 'r') as f:
        games = json.load(f)
    print(f"Loaded {len(games)} games from {filename}")
    return games

if __name__ == "__main__":
    # Test with a small number of games
    print("=== Testing Random Agent ===\n")
    
    # Play one verbose game
    print("Playing one game with output:")
    agent1 = RandomAgent(seed=42)
    agent2 = RandomAgent(seed=43)
    game_record = play_game(agent1, agent2, verbose=True)
    
    print(f"\nGame had {len(game_record['moves'])} moves")
    print(f"First 5 moves: {game_record['moves'][:5]}")
    
    # Generate 100 games
    print("\n=== Generating 100 games ===")
    games = generate_games(100, verbose=True)
    
    # Save games
    save_games(games, "random_games_100.json")
    
    # Verify saved file
    loaded = load_games("random_games_100.json")
    print(f"Verification: Loaded {len(loaded)} games back")