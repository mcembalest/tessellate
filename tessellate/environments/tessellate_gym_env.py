#!/usr/bin/env python3
"""
Example Usage:
    env = TessellateEnv()
    obs = env.reset()
    
    while not env.is_terminal():
        action = agent.select_action(obs)
        obs, reward, done, info = env.step(action)
"""

import numpy as np
from typing import Tuple, Dict, List, Optional
from tessellate import TessellateGame

class TessellateEnv:
    """
    Gym-like environment for Tessellate
    
    Observation Space:
        - Shape: (104,) 
        - [0:100]: Flattened 10x10 board (0=empty, 1=red, 2=blue, 3=blocked)
        - [100]: Current player (1=red, 2=blue)
        - [101]: Red score (current)
        - [102]: Blue score (current)
        - [103]: Move number (0-49)
    
    Action Space:
        - Discrete(100) - index into flattened 10x10 grid
        - Invalid actions are masked via get_valid_actions()
    
    Rewards:
        - Immediate: score_change / 100.0 (normalized)
        - Terminal: +1 for win, -1 for loss, 0 for tie
        - Can be configured via reward_mode parameter
    """
    
    def __init__(self, reward_mode='mixed', gamma=0.9):
        """
        Args:
            reward_mode: 'immediate' (score change), 'terminal' (win/loss), 
                        'mixed' (both), 'sparse' (only terminal)
            gamma: Discount factor for mixed rewards
        """
        self.reward_mode = reward_mode
        self.gamma = gamma
        self.game = None
        self.move_history = []
        self.previous_scores = {}
        
    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """Reset environment to initial state"""
        if seed is not None:
            np.random.seed(seed)
        
        self.game = TessellateGame()
        self.move_history = []
        self.previous_scores = {1: 0, 2: 0}
        
        return self._get_observation()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take an action in the environment
        
        Args:
            action: Index in flattened board (0-99)
        
        Returns:
            observation: Current state
            reward: Immediate reward
            done: Whether episode is terminal
            info: Additional information
        """
        if self.game is None:
            raise ValueError("Must call reset() before step()")
        
        # Convert action to board position
        row, col = action // 10, action % 10
        
        # Check if action is valid
        valid_moves = self.game.get_valid_moves()
        if (row, col) not in valid_moves:
            # Invalid move - could return negative reward or raise error
            return self._get_observation(), -1.0, True, {'invalid_move': True}
        
        # Store current state
        current_player = self.game.current_turn
        
        # Make move
        self.game.make_move(row, col)
        self.move_history.append((row, col))
        
        # Calculate reward
        reward = self._calculate_reward(current_player)
        
        # Check if game is over
        done = self.game.game_over
        
        # Prepare info
        info = {
            'move_count': len(self.move_history),
            'current_scores': self.game.scores.copy(),
            'valid_moves': len(self.game.get_valid_moves()) if not done else 0
        }
        
        if done:
            info['winner'] = self.game.get_winner()
            info['final_scores'] = self.game.scores.copy()
        
        return self._get_observation(), reward, done, info
    
    def _calculate_reward(self, player: int) -> float:
        """Calculate reward based on reward mode"""
        current_scores = self.game.scores.copy()
        
        if self.reward_mode == 'sparse':
            # Only terminal rewards
            if self.game.game_over:
                winner = self.game.get_winner()
                if winner == player:
                    return 1.0
                elif winner == 0:  # Tie
                    return 0.0
                else:
                    return -1.0
            return 0.0
        
        elif self.reward_mode == 'immediate':
            # Score change only
            score_change = current_scores[player] - self.previous_scores[player]
            self.previous_scores = current_scores
            return score_change / 100.0  # Normalize
        
        elif self.reward_mode == 'terminal':
            # Only win/loss at end
            if self.game.game_over:
                winner = self.game.get_winner()
                if winner == player:
                    return 1.0
                elif winner == 0:
                    return 0.0
                else:
                    return -1.0
            return 0.0
        
        else:  # mixed
            # Combination of immediate and future
            score_change = current_scores[player] - self.previous_scores[player]
            immediate_reward = score_change / 100.0
            
            if self.game.game_over:
                winner = self.game.get_winner()
                if winner == player:
                    terminal_reward = 1.0
                elif winner == 0:
                    terminal_reward = 0.0
                else:
                    terminal_reward = -1.0
                
                # Weight terminal reward more heavily
                return immediate_reward * 0.3 + terminal_reward * 0.7
            
            self.previous_scores = current_scores
            return immediate_reward
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation"""
        if self.game is None:
            return np.zeros(104)
        
        # Flatten board
        board_flat = np.array(self.game.board).flatten()
        
        # Create full observation
        obs = np.zeros(104, dtype=np.float32)
        obs[:100] = board_flat
        obs[100] = self.game.current_turn
        obs[101] = self.game.scores.get(1, 0)  # Red score
        obs[102] = self.game.scores.get(2, 0)  # Blue score
        obs[103] = len(self.move_history)  # Move number (0-49)
        
        return obs
    
    def get_valid_actions(self) -> List[int]:
        """Get list of valid actions (indices in flattened board)"""
        if self.game is None:
            return []
        
        valid_moves = self.game.get_valid_moves()
        return [r * 10 + c for r, c in valid_moves]
    
    def is_terminal(self) -> bool:
        """Check if game is over"""
        if self.game is None:
            return True
        return self.game.game_over
    
    def render(self, mode='ascii'):
        """Render current state"""
        if self.game is None:
            print("No game in progress")
            return
        
        if mode == 'ascii':
            print(self.game.render_ascii())
        else:
            # Could add support for returning RGB array for video recording
            pass
    
    def get_state_dict(self) -> Dict:
        """Get serializable state for saving/loading"""
        if self.game is None:
            return {}
        
        return {
            'board': self.game.board,
            'scores': self.game.scores,
            'current_turn': self.game.current_turn,
            'move_history': self.move_history
        }
    
    def set_state_dict(self, state: Dict):
        """Load state from dictionary"""
        self.game = TessellateGame()
        self.game.board = state['board']
        self.game.scores = state['scores']
        self.game.current_turn = state['current_turn']
        self.move_history = state['move_history']
        self.previous_scores = self.game.scores.copy()

def create_env_suite():
    """
    Create a suite of environment configurations for research
    
    Returns different reward configurations to test algorithm robustness
    """
    return {
        'sparse': TessellateEnv(reward_mode='sparse'),
        'immediate': TessellateEnv(reward_mode='immediate'),
        'terminal': TessellateEnv(reward_mode='terminal'),
        'mixed': TessellateEnv(reward_mode='mixed'),
        'mixed_discounted': TessellateEnv(reward_mode='mixed', gamma=0.99),
    }

if __name__ == "__main__":
    # Demo the environment
    print("=== Tessellate Environment Demo ===\n")
    
    env = TessellateEnv(reward_mode='mixed')
    obs = env.reset()
    
    print(f"Initial observation shape: {obs.shape}")
    print(f"Valid actions: {len(env.get_valid_actions())} moves available")
    
    # Play a few random moves
    total_reward = 0
    for i in range(10):
        valid_actions = env.get_valid_actions()
        if not valid_actions:
            break
        
        action = np.random.choice(valid_actions)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        
        r, c = action // 10, action % 10
        print(f"Move {i+1}: ({r},{c}) -> Reward: {reward:.3f}, Done: {done}")
        
        if done:
            print(f"\nGame Over! Winner: {info.get('winner', 'Tie')}")
            print(f"Final scores: {info['final_scores']}")
            break
    
    print(f"\nTotal reward collected: {total_reward:.3f}")