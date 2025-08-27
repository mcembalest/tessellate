#!/usr/bin/env python3
"""
Test reasoning generation for a single Tessellate move.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def test_tessellate_reasoning():
    """Generate reasoning for a Tessellate move."""
    
    print("Loading Qwen3-4B-Thinking...")
    model_name = "Qwen/Qwen3-4B-Thinking-2507"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map=None
    )
    model = model.to("cpu")
    model.eval()
    
    print("Model loaded successfully\n")
    
    # Create a Tessellate prompt
    prompt = """You are playing Tessellate, a strategic board game. In this game:
- Players take turns placing triangular tiles in a 5x5 grid of squares
- Each square can hold 4 triangular tiles (one in each corner: UL, UR, LL, LR)  
- When you place a tile, it blocks 2 adjacent corners in that square
- Tiles of the same color form islands when connected by edges
- Your score is the product (multiplication) of all your island sizes
- The game ends after 50 moves total (25 per player)

Current game state:
Move 0/49, Red to play
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
·· ·· ·· ·· ··

Analyze this position strategically and select your move in the format: Action: (square_row, square_col, corner)
Where square_row and square_col are 0-4, and corner is UL, UR, LL, or LR."""

    messages = [{"role": "user", "content": prompt}]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    inputs = tokenizer(text, return_tensors="pt").to("cpu")
    
    print("Generating reasoning...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    
    # Decode output
    output_ids = outputs[0][len(inputs.input_ids[0]):].tolist()
    
    # Find thinking content
    try:
        # Find </think> token (151668)
        index = len(output_ids) - output_ids[::-1].index(151668)
        thinking = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip()
        action = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip()
    except ValueError:
        # No </think> found
        full_output = tokenizer.decode(output_ids, skip_special_tokens=True)
        thinking = ""
        action = full_output
    
    print("\n" + "="*50)
    print("THINKING:")
    print("="*50)
    if thinking:
        print(thinking[:500] + "..." if len(thinking) > 500 else thinking)
    else:
        print("(No explicit thinking)")
    
    print("\n" + "="*50)
    print("ACTION:")
    print("="*50)
    print(action[:200] if len(action) > 200 else action)
    
    # Check if action format is correct
    import re
    action_match = re.search(r'Action:\s*\((\d+),\s*(\d+),\s*(UL|UR|LL|LR)\)', action)
    if action_match:
        print(f"\n✓ Valid action format detected: {action_match.group()}")
    else:
        print("\n✗ No valid action format found")


if __name__ == "__main__":
    test_tessellate_reasoning()