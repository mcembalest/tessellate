#!/usr/bin/env python3
"""
Utilities for reconstructing Tessellate game states.
Needed for GRPO training to evaluate actions from arbitrary positions.
"""

import numpy as np
from tessellate_env import TessellateEnv
from typing import Dict, List, Tuple


def reconstruct_game_state(
    trajectory: List[Dict],
    target_move: int
) -> TessellateEnv:
    """
    Reconstruct a game state by replaying moves up to target_move.
    
    Args:
        trajectory: Full game trajectory with all moves
        target_move: Move number to reconstruct to (0-49)
        
    Returns:
        TessellateEnv set to the target game state
    """
    env = TessellateEnv(reward_mode='final')
    env.reset()
    
    # Replay moves up to target
    for i in range(min(target_move, len(trajectory))):
        move_data = trajectory[i]
        action = move_data.get('action_id')
        
        if action is not None:
            # Execute the action
            obs, reward, done, info = env.step(action)
            
            if done and i < target_move - 1:
                print(f"Warning: Game ended early at move {i}")
                break
    
    return env


def extract_board_from_state_text(state_text: str) -> np.ndarray:
    """
    Parse the visual board representation back into array form.
    
    Args:
        state_text: Text containing visual board
        
    Returns:
        100-element board array
    """
    board = np.zeros(100, dtype=np.int8)
    
    # Find the Board: section
    lines = state_text.split('\n')
    board_start = -1
    for i, line in enumerate(lines):
        if line.startswith('Board:'):
            board_start = i + 1
            break
    
    if board_start == -1:
        return board
    
    # Parse the visual board
    # Format is 5x5 squares, each shown as 2 lines
    symbol_map = {'·': 0, 'R': 1, 'B': 2}
    
    board_lines = []
    for i in range(board_start, min(board_start + 11, len(lines))):
        line = lines[i].strip()
        if line and not line.isspace():
            board_lines.append(line)
    
    # Process pairs of lines (top/bottom of each square row)
    square_row = 0
    for i in range(0, len(board_lines), 3):  # Every 3 lines (2 for square + 1 blank)
        if i + 1 >= len(board_lines):
            break
            
        top_line = board_lines[i]
        bot_line = board_lines[i + 1]
        
        # Parse each square in the row
        top_squares = top_line.split()
        bot_squares = bot_line.split()
        
        for square_col, (top, bot) in enumerate(zip(top_squares, bot_squares)):
            if len(top) >= 2 and len(bot) >= 2:
                # Map visual position to logical position
                # Top-left corner
                r, c = square_row * 2, square_col * 2
                if r < 10 and c < 10:
                    board[r * 10 + c] = symbol_map.get(top[0], 0)
                # Top-right corner  
                if r < 10 and c + 1 < 10:
                    board[r * 10 + c + 1] = symbol_map.get(top[1], 0)
                # Bottom-left corner
                if r + 1 < 10 and c < 10:
                    board[(r + 1) * 10 + c] = symbol_map.get(bot[0], 0)
                # Bottom-right corner
                if r + 1 < 10 and c + 1 < 10:
                    board[(r + 1) * 10 + c + 1] = symbol_map.get(bot[1], 0)
        
        square_row += 1
    
    return board


def create_env_from_state(state_dict: Dict) -> TessellateEnv:
    """
    Create an environment from a state dictionary.
    
    Args:
        state_dict: Dictionary with board, scores, move_num, etc.
        
    Returns:
        TessellateEnv configured to match the state
    """
    env = TessellateEnv(reward_mode='final')
    
    # We need to modify TessellateEnv to support state injection
    # For now, this is a simplified version
    env.reset()
    
    # This would need proper implementation in TessellateEnv
    # to set board, scores, current_player, etc.
    
    return env


def test_board_parsing():
    """Test board parsing from visual representation."""
    sample_state = """Move 25/49, Blue to play
Scores: Red=8 Blue=12

Board:
·R ·· R· ·B ··
·· B· ·· B· ··

·· ·B ·· R· ·B
·· R· ·· ·· ··

R· R· B· ·R R·
·B ·· ·· ·· ··

·· ·R ·· ·R ··
B· ·· ·· R· ··

B· ·· ·· ·· ·B
·B ·R B· R· ··"""
    
    board = extract_board_from_state_text(sample_state)
    print("Parsed board array:")
    print(board.reshape(10, 10))
    
    # Count pieces
    red_count = (board == 1).sum()
    blue_count = (board == 2).sum()
    print(f"\nPieces: Red={red_count}, Blue={blue_count}")


if __name__ == "__main__":
    test_board_parsing()