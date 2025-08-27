#!/usr/bin/env python3
"""
Generate initial reasoning chains for Tessellate games using Qwen3-4B-Thinking model.
This creates the bootstrap data needed for GRPO training.
"""

import json
import torch
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse


class ReasoningGenerator:
    """Generate reasoning chains for Tessellate game states."""
    
    def __init__(self, model_name: str = "Qwen/Qwen3-4B-Thinking-2507", device: str = "cpu"):
        """
        Initialize with Qwen thinking model.
        
        Args:
            model_name: HuggingFace model ID
            device: Device to use (mps for Mac, cuda, or cpu)
        """
        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Use float32 for CPU/MPS to avoid numerical issues
        # Only use float16 for CUDA
        if device == "cuda" and torch.cuda.is_available():
            dtype = torch.float16
        else:
            dtype = torch.float32
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto" if device == "cuda" else None
        )
        
        # Move to device if not using device_map
        if device != "cuda":
            self.model = self.model.to(device)
            
        self.device = device
        print(f"Model loaded on {device} with dtype {dtype}")
        
    def create_prompt(self, state_text: str, action_text: str) -> str:
        """
        Create prompt for the model to reason about a game state.
        
        Args:
            state_text: Current game state in text format
            action_text: The action that was actually taken (for guidance)
        """
        prompt = f"""You are playing Tessellate, a strategic board game. In this game:
- Players take turns placing triangular tiles in a 5x5 grid of squares
- Each square can hold 4 triangular tiles (one in each corner: UL, UR, LL, LR)  
- When you place a tile, it blocks 2 adjacent corners in that square
- Tiles of the same color form islands when connected by edges
- Your score is the product (multiplication) of all your island sizes
- The game ends after 50 moves total (25 per player)

Current game state:
{state_text}

Analyze this position strategically. Consider:
1. Current island configurations for both players
2. Opportunities to expand your islands or split opponent's islands
3. Key squares that could connect or block territories
4. The multiplicative scoring (many small islands vs few large ones)

Then select your move in the format: Action: (square_row, square_col, corner)
Where square_row and square_col are 0-4, and corner is UL, UR, LL, or LR.

Think step by step about the best move."""
        
        return prompt
    
    def generate_reasoning(self, state_text: str, action_text: str, max_tokens: int = 1024) -> str:
        """
        Generate reasoning for a single game state.
        
        Returns:
            Full model output including thinking and action
        """
        prompt = self.create_prompt(state_text, action_text)
        
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        # Generate with thinking
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_tokens,
                temperature=0.7,  # Slightly higher for stability
                top_p=0.9,  # Slightly lower for stability
                top_k=50,  # Higher k for more options
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        
        # Parse thinking content (model automatically includes </think> with token 151668)
        try:
            # Find </think> token
            index = len(output_ids) - output_ids[::-1].index(151668)
            thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip()
            final_content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip()
        except ValueError:
            # No </think> found, treat all as final content
            thinking_content = ""
            final_content = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        
        # Format output with explicit <think> tags for consistency
        if thinking_content:
            return f"<think>{thinking_content}</think>\n{final_content}"
        else:
            return final_content
    
    def process_trajectory(self, trajectory: Dict, sample_moves: int = 5) -> Dict:
        """
        Add reasoning to a game trajectory.
        
        Args:
            trajectory: Game trajectory with states and actions
            sample_moves: Number of moves to generate reasoning for (for efficiency)
        """
        moves = trajectory['trajectory']
        
        # Sample moves evenly throughout the game
        if sample_moves < len(moves):
            indices = [int(i * len(moves) / sample_moves) for i in range(sample_moves)]
        else:
            indices = range(len(moves))
        
        # Generate reasoning for sampled moves
        for idx in indices:
            move = moves[idx]
            if move['reasoning'] == "<think></think>":  # Only generate if empty
                try:
                    reasoning_output = self.generate_reasoning(
                        move['state_text'],
                        move['action_text']
                    )
                    
                    # Extract just the thinking part
                    if "<think>" in reasoning_output and "</think>" in reasoning_output:
                        start = reasoning_output.index("<think>")
                        end = reasoning_output.index("</think>") + len("</think>")
                        move['reasoning'] = reasoning_output[start:end]
                    else:
                        move['reasoning'] = f"<think>{reasoning_output}</think>"
                        
                except Exception as e:
                    print(f"Error generating reasoning for move {idx}: {e}")
                    continue
        
        return trajectory
    
    def process_file(
        self, 
        input_file: Path, 
        output_file: Path,
        max_games: Optional[int] = None,
        moves_per_game: int = 5
    ):
        """
        Process a file of trajectories, adding reasoning.
        
        Args:
            input_file: Input JSONL file with trajectories
            output_file: Output JSONL file with reasoning added
            max_games: Maximum number of games to process
            moves_per_game: Number of moves to generate reasoning for per game
        """
        games_processed = 0
        
        with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
            for line in tqdm(infile, desc=f"Processing {input_file.name}"):
                if max_games and games_processed >= max_games:
                    break
                
                trajectory = json.loads(line)
                
                # Add reasoning to trajectory
                trajectory_with_reasoning = self.process_trajectory(
                    trajectory, 
                    sample_moves=moves_per_game
                )
                
                # Save updated trajectory
                outfile.write(json.dumps(trajectory_with_reasoning) + '\n')
                games_processed += 1
        
        print(f"Processed {games_processed} games")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', default='llm_data', help='Input directory with trajectories')
    parser.add_argument('--output-dir', default='llm_data_with_reasoning', help='Output directory')
    parser.add_argument('--max-games', type=int, default=10, help='Max games to process (for testing)')
    parser.add_argument('--moves-per-game', type=int, default=5, help='Moves to generate reasoning for per game')
    parser.add_argument('--model', default='Qwen/Qwen3-4B-Thinking-2507', help='Model to use')
    parser.add_argument('--device', default='cpu', help='Device (mps, cuda, cpu)')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Initialize generator
    generator = ReasoningGenerator(model_name=args.model, device=args.device)
    
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