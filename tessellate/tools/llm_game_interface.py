#!/usr/bin/env python3
"""
LLM-Tessellate game interface for batched gameplay and GRPO training.
Designed for efficient batch processing while supporting small batch sizes.
"""

import re
from typing import List, Dict, Tuple, Optional
import numpy as np
from tessellate_env import TessellateEnv
from prepare_llm_data import LLMDataPreparator


class LLMGameInterface:
    """
    Interface between LLM outputs and Tessellate game environment.
    Supports batched gameplay for GRPO training efficiency.
    """
    
    def __init__(self, batch_size: int = 2, max_retries: int = 3):
        """
        Args:
            batch_size: Number of games to run in parallel
            max_retries: Number of retries for invalid actions
        """
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.data_prep = LLMDataPreparator()
        
        # Initialize batch of environments
        self.envs = [TessellateEnv(reward_mode='final') for _ in range(batch_size)]
        self.active_games = [True] * batch_size
        self.game_histories = [[] for _ in range(batch_size)]
        
    def reset_batch(self) -> List[str]:
        """Reset all environments and return initial states as text."""
        states = []
        for i in range(self.batch_size):
            obs = self.envs[i].reset()
            self.active_games[i] = True
            self.game_histories[i] = []
            
            # Convert observation to text
            state_text = self._obs_to_text(obs, self.envs[i])
            states.append(state_text)
        
        return states
    
    def _obs_to_text(self, obs: np.ndarray, env: TessellateEnv) -> str:
        """Convert environment observation to LLM-readable text."""
        # Extract components from observation
        board = obs[:100]
        current_player = int(obs[100])
        red_score = int(obs[101])
        blue_score = int(obs[102])
        move_num = int(obs[103])
        
        # Create text representation
        player = "Red" if current_player == 1 else "Blue"
        board_visual = self.data_prep.board_to_visual(board.tolist())
        
        return f"Move {move_num}/49, {player} to play\nScores: Red={red_score} Blue={blue_score}\n\nBoard:\n{board_visual}"
    
    def parse_action(self, llm_output: str) -> Optional[int]:
        """
        Parse LLM output to extract action.
        Expected format: <think>...</think>\nAction: (x, y, corner)
        
        Returns:
            Action index (0-99) or None if parsing fails
        """
        # Extract action after thinking
        action_match = re.search(r'Action:\s*\((\d+),\s*(\d+),\s*(UL|UR|LL|LR)\)', llm_output)
        
        if not action_match:
            return None
        
        try:
            square_r = int(action_match.group(1))
            square_c = int(action_match.group(2))
            corner = action_match.group(3)
            
            # Validate ranges
            if not (0 <= square_r < 5 and 0 <= square_c < 5):
                return None
            
            # Convert to action index
            action = self.data_prep.coordinates_to_action(square_r, square_c, corner)
            return action
            
        except (ValueError, KeyError):
            return None
    
    def step_batch(self, llm_outputs: List[str]) -> Tuple[List[str], List[float], List[bool], List[Dict]]:
        """
        Process batch of LLM outputs and step environments.
        
        Args:
            llm_outputs: List of LLM responses (one per active game)
            
        Returns:
            next_states: List of next state texts
            rewards: List of rewards
            dones: List of done flags
            infos: List of info dicts (includes parsing failures)
        """
        next_states = []
        rewards = []
        dones = []
        infos = []
        
        for i in range(self.batch_size):
            if not self.active_games[i]:
                # Game already finished
                next_states.append("")
                rewards.append(0.0)
                dones.append(True)
                infos.append({'game_finished': True})
                continue
            
            # Parse action from LLM output
            action = self.parse_action(llm_outputs[i])
            
            if action is None:
                # Invalid action format
                next_states.append(self._obs_to_text(self.envs[i].get_observation(), self.envs[i]))
                rewards.append(-10.0)  # Penalty for invalid format
                dones.append(False)
                infos.append({'error': 'invalid_action_format', 'output': llm_outputs[i]})
                continue
            
            # Check if action is valid
            valid_actions = self.envs[i].get_valid_actions()
            if action not in valid_actions:
                # Invalid move
                next_states.append(self._obs_to_text(self.envs[i].get_observation(), self.envs[i]))
                rewards.append(-5.0)  # Penalty for invalid move
                dones.append(False)
                infos.append({'error': 'invalid_move', 'action': action, 'valid': valid_actions})
                continue
            
            # Execute action
            obs, reward, done, info = self.envs[i].step(action)
            
            # Store in history
            self.game_histories[i].append({
                'llm_output': llm_outputs[i],
                'action': action,
                'reward': reward,
                'done': done
            })
            
            if done:
                self.active_games[i] = False
                # Use score differential as final reward
                final_scores = info.get('final_scores', {})
                red_score = final_scores.get('red', 0)
                blue_score = final_scores.get('blue', 0)
                
                # Determine which player the model was playing as most recently
                current_player = int(self.envs[i].get_observation()[100])
                if current_player == 1:  # Red
                    reward = red_score - blue_score
                else:  # Blue
                    reward = blue_score - red_score
            
            next_states.append(self._obs_to_text(obs, self.envs[i]) if not done else "")
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
        
        return next_states, rewards, dones, infos
    
    def get_batch_status(self) -> Dict:
        """Get status of all games in batch."""
        return {
            'active_games': sum(self.active_games),
            'completed_games': self.batch_size - sum(self.active_games),
            'total_moves': [len(h) for h in self.game_histories],
            'current_scores': [self._get_current_scores(i) for i in range(self.batch_size)]
        }
    
    def _get_current_scores(self, game_idx: int) -> Dict:
        """Get current scores for a game."""
        if not self.active_games[game_idx]:
            return {'red': 0, 'blue': 0, 'status': 'completed'}
        
        obs = self.envs[game_idx].get_observation()
        return {
            'red': int(obs[101]),
            'blue': int(obs[102]),
            'status': 'active'
        }
    
    def extract_reasoning(self, llm_output: str) -> str:
        """Extract reasoning from LLM output."""
        think_match = re.search(r'<think>(.*?)</think>', llm_output, re.DOTALL)
        if think_match:
            return think_match.group(1).strip()
        return ""


class LLMPlayer:
    """
    Wrapper for LLM model to play Tessellate.
    This will be replaced with actual model in GRPO training.
    """
    
    def __init__(self, model=None, temperature: float = 0.8):
        """
        Args:
            model: HuggingFace model (to be specified)
            temperature: Sampling temperature
        """
        self.model = model
        self.temperature = temperature
    
    def generate(self, state_texts: List[str]) -> List[str]:
        """
        Generate actions for batch of states.
        
        For now, returns placeholder responses.
        Will be replaced with actual model generation.
        """
        responses = []
        for state in state_texts:
            # Placeholder - will be replaced with model.generate()
            response = "<think>I should place a tile in the center area.</think>\nAction: (2, 2, UL)"
            responses.append(response)
        return responses


def test_interface():
    """Test the game interface with dummy player."""
    print("Testing LLM Game Interface...")
    
    # Create interface
    interface = LLMGameInterface(batch_size=2)
    player = LLMPlayer()
    
    # Reset games
    states = interface.reset_batch()
    print(f"\nInitial states for {len(states)} games")
    
    # Play a few moves
    for move in range(3):
        print(f"\n--- Move {move} ---")
        
        # Generate actions
        llm_outputs = player.generate(states)
        
        # Step environments
        next_states, rewards, dones, infos = interface.step_batch(llm_outputs)
        
        # Print status
        status = interface.get_batch_status()
        print(f"Active games: {status['active_games']}")
        print(f"Scores: {status['current_scores']}")
        
        # Check for errors
        for i, info in enumerate(infos):
            if 'error' in info:
                print(f"Game {i} error: {info['error']}")
        
        states = next_states
        
        if all(dones):
            print("\nAll games completed!")
            break
    
    print("\nInterface test complete!")


if __name__ == "__main__":
    test_interface()