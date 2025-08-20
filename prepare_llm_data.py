#!/usr/bin/env python3
"""
Prepare Tessellate data for LLM + RL training.
Creates full 50-move trajectories with visual board representations.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
from tqdm import tqdm


class LLMDataPreparator:
    """Convert raw game data to LLM-ready format with chain-of-thought reasoning."""
    
    def __init__(self):
        self.corner_map = {(0,0): 'UL', (0,1): 'UR', (1,0): 'LL', (1,1): 'LR'}
        self.reverse_corner_map = {v: k for k, v in self.corner_map.items()}
    
    def action_to_coordinates(self, action: int) -> Tuple[int, int, str]:
        """Convert action index (0-99) to (square_x, square_y, corner) format."""
        r, c = divmod(action, 10)
        square_r, square_c = r // 2, c // 2
        corner = self.corner_map[(r % 2, c % 2)]
        return (square_r, square_c, corner)
    
    def coordinates_to_action(self, square_r: int, square_c: int, corner: str) -> int:
        """Convert (square_x, square_y, corner) to action index (0-99)."""
        corner_r, corner_c = self.reverse_corner_map[corner]
        r = square_r * 2 + corner_r
        c = square_c * 2 + corner_c
        return r * 10 + c
    
    def board_to_visual(self, board_array: List[float]) -> str:
        """Convert 100-element board array to visual ASCII representation."""
        board = np.array(board_array).reshape(10, 10)
        
        # Map values: 0=empty(·), 1=red(R), 2=blue(B), 3=blocked(·)
        symbols = {0: '·', 1: 'R', 2: 'B', 3: '·'}
        
        lines = []
        for square_r in range(5):
            top_line = []
            bot_line = []
            for square_c in range(5):
                ul = board[square_r*2, square_c*2]
                ur = board[square_r*2, square_c*2 + 1]
                ll = board[square_r*2 + 1, square_c*2]
                lr = board[square_r*2 + 1, square_c*2 + 1]
                
                top_line.append(f'{symbols[ul]}{symbols[ur]}')
                bot_line.append(f'{symbols[ll]}{symbols[lr]}')
            
            lines.append(' '.join(top_line))
            lines.append(' '.join(bot_line))
            if square_r < 4:  # Add spacing between square rows
                lines.append('')
        
        return '\n'.join(lines)
    
    def create_state_text(self, state_data: Dict) -> str:
        """Create full state representation for LLM."""
        move_num = state_data['move_number']
        player = "Red" if state_data['current_player'] == 1 else "Blue"
        red_score = state_data['scores']['red']
        blue_score = state_data['scores']['blue']
        
        board_visual = self.board_to_visual(state_data['board'])
        
        return f"Move {move_num}/49, {player} to play\nScores: Red={red_score} Blue={blue_score}\n\nBoard:\n{board_visual}"
    
    def process_game_to_trajectory(self, prompts_data: List[Dict]) -> Dict:
        """Convert sequence of prompts to full game trajectory."""
        # Verify we have a complete game
        if len(prompts_data) != 50:
            raise ValueError(f"Expected 50 moves, got {len(prompts_data)}")
        
        trajectory = []
        
        for i, state_data in enumerate(prompts_data):
            state_text = self.create_state_text(state_data)
            
            # Convert action to coordinate format
            action_id = state_data['actual_move']
            square_r, square_c, corner = self.action_to_coordinates(action_id)
            action_text = f"({square_r}, {square_c}, {corner})"
            
            step = {
                'move_num': i,
                'state_text': state_text,
                'reasoning': "<think></think>",  # Placeholder for now
                'action_text': f"Action: {action_text}",
                'action_id': action_id,
                'valid': action_id in state_data['valid_moves'],
                'current_player': state_data['current_player']
            }
            trajectory.append(step)
        
        # Calculate final reward based on score differential
        final_state = prompts_data[-1]
        final_red = final_state['scores']['red']
        final_blue = final_state['scores']['blue']
        score_diff = final_red - final_blue
        
        # Determine winner
        if score_diff > 0:
            winner = 'red'
        elif score_diff < 0:
            winner = 'blue'
        else:
            winner = 'tie'
        
        return {
            'trajectory': trajectory,
            'final_scores': {'red': final_red, 'blue': final_blue},
            'winner': winner,
            'score_differential': abs(score_diff),
            'reward_red': score_diff,  # Positive if red wins
            'reward_blue': -score_diff  # Positive if blue wins
        }
    
    def process_prompts_file(self, filepath: Path) -> List[Dict]:
        """Process a single prompts file into game trajectories."""
        trajectories = []
        current_game = []
        
        with open(filepath, 'r') as f:
            for line in f:
                data = json.loads(line)
                current_game.append(data)
                
                # Check if game is complete
                if len(current_game) == 50:
                    trajectory = self.process_game_to_trajectory(current_game)
                    trajectories.append(trajectory)
                    current_game = []
        
        return trajectories
    
    def process_all_data(
        self, 
        input_dir: Path, 
        output_dir: Path,
        max_files: Optional[int] = None
    ):
        """Process all prompt files into LLM trajectories."""
        output_dir.mkdir(exist_ok=True)
        
        # Find all prompt files
        prompt_files = sorted(input_dir.glob('rl_prompts_*.jsonl'))
        if max_files:
            prompt_files = prompt_files[:max_files]
        
        print(f"Processing {len(prompt_files)} prompt files...")
        
        all_trajectories = []
        file_num = 0
        
        for prompt_file in tqdm(prompt_files):
            try:
                trajectories = self.process_prompts_file(prompt_file)
                all_trajectories.extend(trajectories)
                
                # Save in batches
                if len(all_trajectories) >= 100:
                    output_file = output_dir / f'llm_trajectories_{file_num:03d}.jsonl'
                    with open(output_file, 'w') as f:
                        for traj in all_trajectories[:100]:
                            f.write(json.dumps(traj) + '\n')
                    
                    print(f"Saved {output_file.name} with {min(100, len(all_trajectories))} games")
                    all_trajectories = all_trajectories[100:]
                    file_num += 1
                    
            except Exception as e:
                print(f"Error processing {prompt_file.name}: {e}")
                continue
        
        # Save remaining trajectories
        if all_trajectories:
            output_file = output_dir / f'llm_trajectories_{file_num:03d}.jsonl'
            with open(output_file, 'w') as f:
                for traj in all_trajectories:
                    f.write(json.dumps(traj) + '\n')
            print(f"Saved {output_file.name} with {len(all_trajectories)} games")
    
    def test_action_parser(self):
        """Test the action parsing and formatting."""
        test_cases = [
            (0, (0, 0, 'UL')),
            (1, (0, 0, 'UR')),
            (10, (0, 0, 'LL')),
            (11, (0, 0, 'LR')),
            (44, (2, 2, 'UL')),
            (99, (4, 4, 'LR')),
        ]
        
        print("Testing action parser:")
        for action_id, expected in test_cases:
            result = self.action_to_coordinates(action_id)
            reverse = self.coordinates_to_action(*result)
            status = "✓" if result == expected and reverse == action_id else "✗"
            print(f"  {status} Action {action_id:2d} -> {result} -> {reverse}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', default='rl_data_large', help='Input directory')
    parser.add_argument('--output-dir', default='llm_data', help='Output directory')
    parser.add_argument('--max-files', type=int, help='Maximum files to process')
    parser.add_argument('--test', action='store_true', help='Run tests only')
    args = parser.parse_args()
    
    preparator = LLMDataPreparator()
    
    if args.test:
        preparator.test_action_parser()
    else:
        input_dir = Path(args.input_dir)
        output_dir = Path(args.output_dir)
        preparator.process_all_data(input_dir, output_dir, args.max_files)


if __name__ == "__main__":
    main()