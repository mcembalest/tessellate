#!/usr/bin/env python3
"""
Tessellate Data Preprocessor
=============================
Pure data transformation from game JSONs to RL-ready format.
No assumptions, no filtering, just clean data engineering.

Input: 1M+ games from game_data/*.json (50 moves each)
Output: rl_data_XXX.npz files with transitions (s, a, r, s', mask, done)
        rl_data_XXX.jsonl files for LLM/GRPO training
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
from tqdm import tqdm

class TessellatePreprocessor:
    """
    Convert raw game data to RL training format.
    
    State Representation (104 dims):
    - [0:100]: Board (0=empty, 1=red, 2=blue, 3=blocked)
    - [100]: Current player (1 or 2)
    - [101]: Red score (unbounded integer)
    - [102]: Blue score (unbounded integer)
    - [103]: Move number (0-49)
    """
    
    def __init__(self):
        self.board_size = 100  # 10x10 grid
        
    def process_game(self, game: Dict) -> List[Dict]:
        """
        Process single game into transitions for both players.
        
        Returns list of transitions: (s, a, r, s', mask, done)
        Stores transitions for BOTH players (100M total from 1M games)
        """
        moves = game['moves']
        final_scores = game['final_scores']
        winner = game['winner']
        
        # Reconstruct all board states
        board_states = self._build_board_states(moves)
        
        # Generate transitions for both players
        transitions = []
        
        for player_id in [1, 2]:
            player_transitions = self._extract_player_transitions(
                board_states, moves, final_scores, winner, player_id
            )
            transitions.extend(player_transitions)
            
        return transitions
    
    def _build_board_states(self, moves: List[Dict]) -> List[np.ndarray]:
        """
        Build sequence of board states from moves.
        
        Board Encoding:
        - Position (r,c) → index = r*10 + c
        - 0 = empty, 1 = red, 2 = blue, 3 = blocked
        - Reconstruct by replaying moves sequentially
        """
        board_states = []
        board = np.zeros(100, dtype=np.int8)
        
        for move in moves:
            # Save current board state
            board_states.append(board.copy())
            
            # Apply move
            r, c = move['position']
            idx = r * 10 + c
            board[idx] = move['player']
            
            # Apply blocking
            self._apply_blocking(board, r, c)
            
        # Add final board state
        board_states.append(board.copy())
        
        return board_states
    
    def _apply_blocking(self, board: np.ndarray, r: int, c: int):
        """
        Apply blocking rules when triangle placed at (r,c).
        
        When placing at position (r,c), block the two adjacent corners in same square.
        Example: placing at (0,0) blocks (0,1) and (1,0), leaving only (1,1) available.
        """
        square_r, square_c = r // 2, c // 2
        corner_r, corner_c = r % 2, c % 2
        
        # Determine which corners to block based on which corner was placed
        # Block the two adjacent corners (diagonal opposite remains available)
        if corner_r == 0 and corner_c == 0:  # Top-left placed
            blocks = [(2*square_r, 2*square_c+1), (2*square_r+1, 2*square_c)]  # Block top-right, bottom-left
        elif corner_r == 0 and corner_c == 1:  # Top-right placed
            blocks = [(2*square_r, 2*square_c), (2*square_r+1, 2*square_c+1)]  # Block top-left, bottom-right  
        elif corner_r == 1 and corner_c == 0:  # Bottom-left placed
            blocks = [(2*square_r, 2*square_c), (2*square_r+1, 2*square_c+1)]  # Block top-left, bottom-right
        else:  # Bottom-right placed (1,1)
            blocks = [(2*square_r, 2*square_c+1), (2*square_r+1, 2*square_c)]  # Block top-right, bottom-left
            
        # Apply blocks
        for br, bc in blocks:
            if 0 <= br < 10 and 0 <= bc < 10:
                idx = br * 10 + bc
                if board[idx] == 0:  # Only block if empty
                    board[idx] = 3
    
    def _extract_player_transitions(
        self, 
        board_states: List[np.ndarray],
        moves: List[Dict],
        final_scores: Dict,
        winner: int,
        player_id: int
    ) -> List[Dict]:
        """Extract transitions from one player's perspective"""
        transitions = []
        
        for i, move in enumerate(moves):
            if move['player'] != player_id:
                continue
                
            # Current state
            board = board_states[i]
            scores = move['score_before']
            
            state = self._create_state_vector(
                board, player_id, scores['1'], scores['2'], i
            )
            
            # Action
            r, c = move['position']
            action = r * 10 + c
            
            # Next state (after both players move, or game end)
            next_move_idx = min(i + 2, len(moves))
            if next_move_idx < len(moves):
                next_board = board_states[next_move_idx]
                next_scores = moves[next_move_idx]['score_before']
                next_player = moves[next_move_idx]['player']
            else:
                next_board = board_states[-1]
                next_scores = {'1': final_scores['red'], '2': final_scores['blue']}
                next_player = 3 - player_id  # Other player
                
            next_state = self._create_state_vector(
                next_board, next_player, next_scores['1'], next_scores['2'], next_move_idx
            )
            
            # Reward calculation (sparse - only at game end)
            # Terminal reward: win/loss ± score difference
            # All intermediate rewards = 0
            done = (next_move_idx >= len(moves))
            if done:
                my_score = final_scores['red'] if player_id == 1 else final_scores['blue']
                opp_score = final_scores['blue'] if player_id == 1 else final_scores['red']
                score_diff = my_score - opp_score  # Unbounded
                
                if winner == player_id:
                    reward = 1.0 + score_diff  # Win bonus + margin
                elif winner == 3 - player_id:
                    reward = -1.0 + score_diff  # Loss penalty softened by closeness
                else:
                    reward = 0.0  # Tie (rare)
            else:
                reward = 0.0
                
            # Valid actions mask for next state
            valid_mask = self._compute_valid_actions(next_board)
            
            transitions.append({
                'state': state,
                'action': action,
                'reward': reward,
                'next_state': next_state,
                'valid_mask': valid_mask,
                'done': done
            })
            
        return transitions
    
    def _create_state_vector(
        self,
        board: np.ndarray,
        current_player: int,
        red_score: int,
        blue_score: int,
        move_num: int
    ) -> np.ndarray:
        """Create 104-dimensional state vector"""
        state = np.zeros(104, dtype=np.float32)
        state[:100] = board
        state[100] = current_player
        state[101] = red_score
        state[102] = blue_score  
        state[103] = move_num
        return state
    
    def _compute_valid_actions(self, board: np.ndarray) -> np.ndarray:
        """
        Compute mask of valid actions.
        Empty positions = True, occupied/blocked = False
        Models should apply mask before softmax.
        """
        return (board == 0).astype(np.float32)
    
    def _create_llm_prompt(self, transition: Dict) -> Dict:
        """
        Create LLM-friendly prompt from transition.
        
        LLM/GRPO Compatibility:
        1. State Serialization: Board state → text format for LLM input
        2. Action Format: Integer position (0-99) as text
        3. Valid moves included for action masking
        4. Format ready for custom TessellateEvaluator scoring
        
        This format can be used with GRPO where:
        - LLM generates: "42" or "Place at position 42"
        - Evaluator parses move and checks validity
        - Rewards computed based on game outcome
        """
        state = transition['state']
        board = state[:100]
        current_player = int(state[100])
        red_score = int(state[101])
        blue_score = int(state[102])
        move_num = int(state[103])
        
        # Find valid actions (indices where board == 0)
        valid_moves = [i for i, val in enumerate(board) if val == 0]
        
        # Create structured prompt data
        prompt_data = {
            'state_text': f"Move {move_num}/49, {'Red' if current_player == 1 else 'Blue'} to play, Scores: Red={red_score} Blue={blue_score}",
            'board': board.tolist(),
            'valid_moves': valid_moves,
            'actual_move': int(transition['action']),
            'reward': float(transition['reward']),
            'terminal': bool(transition['done']),
            
            # Additional context for LLM training
            'current_player': current_player,
            'scores': {'red': red_score, 'blue': blue_score},
            'move_number': move_num,
            
            # For GRPO evaluator
            'ground_truth_action': int(transition['action']),
            'was_winning_move': float(transition['reward']) > 0 if transition['done'] else None
        }
        
        return prompt_data
    
    def process_dataset(
        self,
        input_files: List[Path],
        output_dir: Path,
        chunk_size: int = 100000
    ):
        """Process full dataset and save in chunks (legacy method)"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        all_transitions = []
        all_trajectories = []  # For GRPO-style trajectory learning
        chunk_num = 0
        traj_chunk_num = 0
        
        for file_path in tqdm(input_files, desc="Processing files"):
            with open(file_path) as f:
                games = json.load(f)
                
            for game in tqdm(games, desc=f"Processing {file_path.name}", leave=False):
                transitions = self.process_game(game)
                all_transitions.extend(transitions)
                
                # Also save full trajectories for GRPO
                trajectory = self._create_trajectory(game, transitions)
                all_trajectories.append(trajectory)
                
                # Save transitions chunk if large enough
                if len(all_transitions) >= chunk_size:
                    self._save_chunk(all_transitions[:chunk_size], output_dir, chunk_num)
                    all_transitions = all_transitions[chunk_size:]
                    chunk_num += 1
                
                # Save trajectory chunk periodically
                if len(all_trajectories) >= 1000:  # Save every 1000 games
                    self._save_trajectories(all_trajectories, output_dir, traj_chunk_num)
                    all_trajectories = []
                    traj_chunk_num += 1
        
        # Save remaining transitions and trajectories
        if all_transitions:
            self._save_chunk(all_transitions, output_dir, chunk_num)
        if all_trajectories:
            self._save_trajectories(all_trajectories, output_dir, traj_chunk_num)
    
    def process_dataset_smart(
        self,
        file_info: List[Dict],
        output_dir: Path,
        chunk_size: int = 100000,
        max_games: int = None
    ):
        """Smart processing that handles huge files and provides better progress tracking"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        all_transitions = []
        all_trajectories = []
        chunk_num = 0
        traj_chunk_num = 0
        games_processed = 0
        
        # Calculate total for progress bar
        if max_games:
            total_games = max_games
        else:
            total_games = sum(f['estimated_games'] for f in file_info)
        
        with tqdm(total=total_games, desc="Processing games", unit="games") as pbar:
            for file_data in file_info:
                if max_games and games_processed >= max_games:
                    break
                
                file_path = file_data['path']
                
                # Skip huge files for small batches
                if file_data['is_huge'] and max_games and max_games < 100000:
                    print(f"\nSkipping huge file {file_path.name} for small batch")
                    continue
                
                try:
                    with open(file_path) as f:
                        data = json.load(f)
                    
                    # Handle different file formats
                    if isinstance(data, list):
                        games = data
                    elif isinstance(data, dict) and 'games' in data:
                        games = data['games']
                    else:
                        # Skip non-game files
                        continue
                    
                    # Limit games if needed
                    if max_games:
                        remaining = max_games - games_processed
                        games = games[:remaining]
                    
                    # Process games
                    for game in games:
                        transitions = self.process_game(game)
                        all_transitions.extend(transitions)
                        
                        trajectory = self._create_trajectory(game, transitions)
                        all_trajectories.append(trajectory)
                        
                        games_processed += 1
                        pbar.update(1)
                        
                        # Save chunks as needed
                        if len(all_transitions) >= chunk_size:
                            self._save_chunk(all_transitions[:chunk_size], output_dir, chunk_num)
                            all_transitions = all_transitions[chunk_size:]
                            chunk_num += 1
                        
                        if len(all_trajectories) >= 1000:
                            self._save_trajectories(all_trajectories, output_dir, traj_chunk_num)
                            all_trajectories = []
                            traj_chunk_num += 1
                            
                except Exception as e:
                    print(f"\nError processing {file_path.name}: {e}")
                    continue
        
        # Save remaining data
        if all_transitions:
            self._save_chunk(all_transitions, output_dir, chunk_num)
        if all_trajectories:
            self._save_trajectories(all_trajectories, output_dir, traj_chunk_num)
        
        print(f"\nProcessed {games_processed:,} games total")
            
    def _save_chunk(self, transitions: List[Dict], output_dir: Path, chunk_num: int):
        """
        Save transitions chunk to NPZ and JSONL formats.
        
        NPZ format for traditional RL (DQN, PPO, etc.)
        JSONL format for LLM/GRPO training
        """
        
        # Convert to arrays for NPZ
        states = np.array([t['state'] for t in transitions])
        actions = np.array([t['action'] for t in transitions])
        rewards = np.array([t['reward'] for t in transitions])
        next_states = np.array([t['next_state'] for t in transitions])
        valid_masks = np.array([t['valid_mask'] for t in transitions])
        dones = np.array([t['done'] for t in transitions])
        
        # Save NPZ for traditional RL
        npz_path = output_dir / f'rl_data_{chunk_num:03d}.npz'
        np.savez_compressed(
            npz_path,
            states=states,
            actions=actions,
            rewards=rewards,
            next_states=next_states,
            masks=valid_masks,
            dones=dones
        )
        
        # Save JSONL for LLM/GRPO training
        jsonl_path = output_dir / f'rl_data_{chunk_num:03d}.jsonl'
        with open(jsonl_path, 'w') as f:
            for t in transitions:
                # Convert arrays to lists for JSON serialization
                json_transition = {
                    'state': t['state'].tolist(),
                    'action': int(t['action']),
                    'reward': float(t['reward']),
                    'next_state': t['next_state'].tolist(),
                    'valid_mask': t['valid_mask'].tolist(),
                    'done': bool(t['done'])
                }
                f.write(json.dumps(json_transition) + '\n')
        
        # Save LLM-friendly prompts format
        prompts_path = output_dir / f'rl_prompts_{chunk_num:03d}.jsonl'
        with open(prompts_path, 'w') as f:
            for t in transitions:
                # Format state as text prompt for LLM
                prompt_data = self._create_llm_prompt(t)
                f.write(json.dumps(prompt_data) + '\n')
        
        print(f"Saved chunk {chunk_num}: {len(transitions)} transitions")
    
    def _create_trajectory(self, game: Dict, transitions: List[Dict]) -> Dict:
        """
        Create full trajectory for GRPO-style learning.
        
        GRPO needs full episodes to:
        - Generate multiple completions per state
        - Score them based on final outcome
        - Learn from trajectory-level rewards
        """
        return {
            'game_id': hash(str(game['moves'])),  # Unique ID
            'winner': game['winner'],
            'final_scores': game['final_scores'],
            'transitions': transitions,
            'total_moves': len(game['moves']),
            'score_differential': abs(game['final_scores']['red'] - game['final_scores']['blue'])
        }
    
    def _save_trajectories(self, trajectories: List[Dict], output_dir: Path, chunk_num: int):
        """
        Save full trajectories for GRPO training.
        
        These files contain complete games that can be:
        - Sampled for state selection
        - Used to generate alternative action sequences
        - Scored based on game outcomes
        """
        traj_path = output_dir / f'rl_trajectories_{chunk_num:03d}.jsonl'
        with open(traj_path, 'w') as f:
            for traj in trajectories:
                # Convert numpy arrays to lists
                clean_traj = {
                    'game_id': traj['game_id'],
                    'winner': traj['winner'],
                    'final_scores': traj['final_scores'],
                    'total_moves': traj['total_moves'],
                    'score_differential': traj['score_differential'],
                    'transitions': [
                        {
                            'state': t['state'].tolist() if hasattr(t['state'], 'tolist') else t['state'],
                            'action': int(t['action']),
                            'reward': float(t['reward']),
                            'done': bool(t['done'])
                        } for t in traj['transitions']
                    ]
                }
                f.write(json.dumps(clean_traj) + '\n')
        
        print(f"Saved trajectory chunk {chunk_num}: {len(trajectories)} games")


def get_file_info(input_dir):
    """Analyze input files and warn about large ones"""
    files = list(Path(input_dir).glob('*.json'))
    
    file_info = []
    total_games_estimate = 0
    
    for f in files:
        size_mb = f.stat().st_size / (1024 * 1024)
        # Estimate games: ~3.4KB per game in JSON
        estimated_games = int(f.stat().st_size / 3400)
        
        file_info.append({
            'path': f,
            'size_mb': size_mb,
            'estimated_games': estimated_games,
            'is_huge': size_mb > 1000  # Files over 1GB
        })
        total_games_estimate += estimated_games
    
    return file_info, total_games_estimate

def main():
    parser = argparse.ArgumentParser(description='Preprocess Tessellate game data')
    
    # Mode-based operation (new)
    parser.add_argument('--mode', choices=['quick', 'small', 'medium', 'large', 'full', 'custom'],
                       help='Preset processing modes for convenience')
    
    # Traditional arguments
    parser.add_argument('--input-dir', type=Path, default=Path('game_data'),
                       help='Directory containing game JSON files')
    parser.add_argument('--output-dir', type=Path, default=None,
                       help='Output directory for processed data')
    parser.add_argument('--chunk-size', type=int, default=100000,
                       help='Number of transitions per output file')
    parser.add_argument('--max-games', type=int, default=None,
                       help='Maximum number of games to process')
    parser.add_argument('--include-huge', action='store_true',
                       help='Include huge files (>1GB) - by default these are skipped')
    
    args = parser.parse_args()
    
    print("""
╔════════════════════════════════════════════╗
║        TESSELLATE PREPROCESSOR             ║
╚════════════════════════════════════════════╝
""")
    
    # Handle preset modes
    if args.mode:
        if args.mode == 'quick':
            args.max_games = args.max_games or 100
            args.output_dir = args.output_dir or Path('rl_data_quick')
        elif args.mode == 'small':
            args.max_games = args.max_games or 10000
            args.output_dir = args.output_dir or Path('rl_data_small')
        elif args.mode == 'medium':
            args.max_games = args.max_games or 100000
            args.output_dir = args.output_dir or Path('rl_data_medium')
        elif args.mode == 'large':
            args.max_games = args.max_games or 500000
            args.output_dir = args.output_dir or Path('rl_data_large')
        elif args.mode == 'full':
            args.max_games = None
            args.output_dir = args.output_dir or Path('rl_data_full')
    else:
        # Default output dir if not specified
        args.output_dir = args.output_dir or Path('rl_data')
    
    # Analyze input files
    print("Analyzing input files...")
    file_info, total_est = get_file_info(args.input_dir)
    
    # Separate and warn about huge files
    huge_files = [f for f in file_info if f['is_huge']]
    normal_files = [f for f in file_info if not f['is_huge']]
    
    if huge_files and not args.include_huge:
        print(f"\n⚠️  Found {len(huge_files)} HUGE file(s) that will be SKIPPED:")
        for f in huge_files:
            print(f"   - {f['path'].name}: {f['size_mb']:.1f}MB (~{f['estimated_games']:,} games)")
        print(f"   Use --include-huge to process these files\n")
    
    # Select files to process
    if args.include_huge:
        files_to_process = file_info
    else:
        files_to_process = normal_files
    
    # Sort by size (process smaller files first for better progress feedback)
    files_to_process.sort(key=lambda x: x['size_mb'])
    
    print(f"Files to process: {len(files_to_process)}")
    print(f"Estimated games: {sum(f['estimated_games'] for f in files_to_process):,}")
    if args.max_games:
        print(f"Max games limit: {args.max_games:,}")
    print(f"Output directory: {args.output_dir}/")
    print(f"Chunk size: {args.chunk_size:,} transitions")
    
    # Process with the smart file handling
    preprocessor = TessellatePreprocessor()
    preprocessor.process_dataset_smart(
        files_to_process, 
        args.output_dir, 
        args.chunk_size,
        args.max_games
    )
    
    print(f"\n✅ Preprocessing complete! Data saved to {args.output_dir}/")


if __name__ == "__main__":
    main()