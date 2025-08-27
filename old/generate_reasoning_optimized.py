#!/usr/bin/env python3
"""
Optimized reasoning generation for Tessellate using Qwen3-4B-Thinking-FP8 on MPS.
Includes game rules in initial prompt, uses FP8 for efficiency.
"""

import json
import torch
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse


TESSELLATE_RULES = """Tessellate is a two-player board game with these rules:
- Board: 5x5 grid of squares, each square has 4 corners (UL=upper-left, UR=upper-right, LL=lower-left, LR=lower-right)
- Players alternate placing triangular tiles in corners (Red goes first)
- When you place a tile, it blocks the 2 adjacent corners in that square
- Connected tiles of the same color form islands
- Score = product of all your island sizes (e.g., 3 islands of sizes 2,3,4 = 2×3×4 = 24 points)
- Game ends after 50 moves (25 per player)
- Highest score wins

Strategy tips:
- Many small islands score less than few large islands (due to multiplication)
- Blocking opponent's connections is crucial
- Center positions offer more connectivity"""


class OptimizedReasoningGenerator:
    """Generate reasoning chains efficiently using FP8 model on MPS."""
    
    def __init__(self, model_name: str = "Qwen/Qwen3-4B-Thinking-2507-FP8", device: str = "mps"):
        """
        Initialize with FP8 model for faster inference.
        
        Args:
            model_name: HuggingFace model ID (FP8 version)
            device: Device to use (mps for Mac GPU)
        """
        print(f"Loading {model_name} on {device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # FP8 model should be faster and use less memory
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,  # FP8 models often use float16 interface
            device_map={"": device} if device == "mps" else "auto"
        )
        
        self.device = device
        print(f"Model loaded on {device}")
        
        # Cache for avoiding repeated rule explanations
        self.rules_included = set()
    
    def create_prompt(self, state_text: str, game_id: str, move_num: int) -> str:
        """
        Create prompt with rules only on first move of each game.
        
        Args:
            state_text: Current game state in text format
            game_id: Unique game identifier
            move_num: Current move number (0-49)
        """
        # Include rules only for first move of each game
        if game_id not in self.rules_included and move_num < 5:
            self.rules_included.add(game_id)
            prompt = f"""{TESSELLATE_RULES}

Current game state:
{state_text}

Analyze this position step by step, considering island formations and blocking opportunities.
Then select your move in the format: Action: (row, col, corner)
Where row and col are 0-4 (for the 5x5 grid), corner is UL/UR/LL/LR."""
        else:
            # Subsequent moves: shorter prompt without rules
            prompt = f"""Current Tessellate position:
{state_text}

Analyze the board and select your move.
Format: Action: (row, col, corner)"""
        
        return prompt
    
    def generate_reasoning(
        self, 
        state_text: str, 
        game_id: str,
        move_num: int,
        max_tokens: int = 1024,
        temperature: float = 0.6
    ) -> str:
        """
        Generate reasoning with controlled token budget.
        
        FP8 should be much faster, allowing us to process more games.
        """
        prompt = self.create_prompt(state_text, game_id, move_num)
        
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        
        # Generate with thinking
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=0.95,
                top_k=20,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        
        # Parse thinking content
        think_close_token = 151668  # </think> token
        
        if think_close_token in output_ids:
            index = output_ids.index(think_close_token)
            thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip()
            final_content = self.tokenizer.decode(output_ids[index+1:], skip_special_tokens=True).strip()
            
            # Return formatted with explicit tags
            return f"<think>{thinking_content}</think>\n{final_content}"
        else:
            # If no </think> found within budget, truncate and add it
            thinking_content = self.tokenizer.decode(output_ids[:900], skip_special_tokens=True).strip()
            return f"<think>{thinking_content}... [truncated]</think>\nAction: (2, 2, UL)"
    
    def process_trajectory(self, trajectory: Dict, moves_per_game: int = 5) -> Dict:
        """
        Add reasoning to selected moves in a game trajectory.
        
        Args:
            trajectory: Game trajectory with states and actions
            moves_per_game: Number of moves to generate reasoning for
        """
        moves = trajectory['trajectory']
        game_id = str(trajectory.get('game_id', 'unknown'))
        
        # Sample moves evenly throughout the game
        if moves_per_game < len(moves):
            # Focus on early and mid-game (more strategic)
            indices = [0, 10, 20, 30, 40][:moves_per_game]
        else:
            indices = range(len(moves))
        
        # Generate reasoning for sampled moves
        for idx in indices:
            move = moves[idx]
            if move['reasoning'] == "<think></think>":  # Only generate if empty
                try:
                    reasoning_output = self.generate_reasoning(
                        move['state_text'],
                        game_id,
                        move['move_num'],
                        max_tokens=1024,  # Controlled budget
                        temperature=0.6
                    )
                    
                    # Extract just the thinking part
                    if "<think>" in reasoning_output and "</think>" in reasoning_output:
                        start = reasoning_output.index("<think>")
                        end = reasoning_output.index("</think>") + len("</think>")
                        move['reasoning'] = reasoning_output[start:end]
                    else:
                        move['reasoning'] = f"<think>{reasoning_output}</think>"
                        
                except Exception as e:
                    print(f"Error generating reasoning for game {game_id}, move {idx}: {e}")
                    continue
        
        # Clear game from cache after processing
        self.rules_included.discard(game_id)
        
        return trajectory
    
    def process_file(
        self, 
        input_file: Path, 
        output_file: Path,
        max_games: Optional[int] = None,
        moves_per_game: int = 5
    ):
        """Process a file of trajectories with progress tracking."""
        games_processed = 0
        
        with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
            lines = infile.readlines()
            total = min(len(lines), max_games) if max_games else len(lines)
            
            for line in tqdm(lines[:total], desc=f"Processing {input_file.name}"):
                trajectory = json.loads(line)
                
                # Add reasoning to trajectory
                trajectory_with_reasoning = self.process_trajectory(
                    trajectory, 
                    moves_per_game
                )
                
                # Save updated trajectory
                outfile.write(json.dumps(trajectory_with_reasoning) + '\n')
                games_processed += 1
        
        print(f"Processed {games_processed} games")


def benchmark_performance():
    """Quick benchmark to test FP8 model speed on MPS."""
    import time
    
    print("Benchmarking FP8 model performance on MPS...")
    generator = OptimizedReasoningGenerator(device="mps")
    
    # Test state
    test_state = """Move 0/49, Red to play
Scores: Red=1 Blue=1

Board:
·· ·· ·· ·· ··
·· ·· ·· ·· ··

·· ·· ·· ·· ··
·· ·· ·· ·· ··

·· ·· ·· ·· ··
·· ·· ·· ·· ··

·· ·· ·· ·· ··
·· ·· ·· ·· ··

·· ·· ·· ·· ··
·· ·· ·· ·· ··"""
    
    # Time generation
    start = time.time()
    result = generator.generate_reasoning(test_state, "test_game", 0, max_tokens=512)
    elapsed = time.time() - start
    
    print(f"\nGeneration time: {elapsed:.2f} seconds")
    print(f"Output length: {len(result)} characters")
    
    if elapsed < 10:
        print("✓ Performance is acceptable for training!")
    else:
        print("✗ Still too slow, may need further optimization")
    
    return elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', default='llm_data', help='Input directory with trajectories')
    parser.add_argument('--output-dir', default='llm_data_with_reasoning', help='Output directory')
    parser.add_argument('--max-games', type=int, default=10, help='Max games to process')
    parser.add_argument('--moves-per-game', type=int, default=5, help='Moves to generate reasoning for per game')
    parser.add_argument('--device', default='mps', help='Device (mps, cuda, cpu)')
    parser.add_argument('--benchmark', action='store_true', help='Run performance benchmark')
    args = parser.parse_args()
    
    if args.benchmark:
        benchmark_performance()
        return
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Initialize generator
    generator = OptimizedReasoningGenerator(device=args.device)
    
    # Process files
    input_dir = Path(args.input_dir)
    for input_file in sorted(input_dir.glob('llm_trajectories_*.jsonl')):
        output_file = output_dir / input_file.name.replace('llm_trajectories', 'reasoning_trajectories')
        
        print(f"\nProcessing {input_file.name}...")
        generator.process_file(
            input_file,
            output_file,
            max_games=args.max_games,
            moves_per_game=args.moves_per_game
        )
        
        # For testing, only process first file
        if args.max_games:
            break
    
    print("\nReasoning generation complete!")


if __name__ == "__main__":
    main()