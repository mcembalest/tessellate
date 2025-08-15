#!/usr/bin/env python3
"""
Agent implementations for Tessellate
"""

import random
from typing import Dict, Any
from tessellate import TessellateGame, TileState

class RandomAgent:
    """Agent that plays random valid moves"""
    
    def __init__(self, seed=None):
        if seed is not None:
            random.seed(seed)
    
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
        game_record['moves'].append({
            'player': int(game.current_turn),
            'position': move,
            'score_before': game.scores.copy()
        })        
        success = game.make_move(move[0], move[1])
        if not success:
            print(f"Failed move: {move}")
            break
            
        move_count += 1
    
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

