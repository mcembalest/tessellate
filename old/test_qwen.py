#!/usr/bin/env python3
"""
Test Qwen3-4B-Thinking model to debug generation issues.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def test_basic_generation():
    """Test basic model generation without game-specific content."""
    
    print("Loading Qwen3-4B-Thinking...")
    model_name = "Qwen/Qwen3-4B-Thinking-2507"
    
    # Load with float32 for stability
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map=None
    )
    
    # Move to CPU explicitly
    device = "cpu"
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded on {device}")
    
    # Simple test prompt
    prompt = "What is 2 + 2?"
    messages = [{"role": "user", "content": prompt}]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    print(f"\nInput text: {text[:100]}...")
    
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt").to(device)
    print(f"Input shape: {inputs.input_ids.shape}")
    
    # Try generation with minimal settings
    print("\nGenerating...")
    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,  # Greedy for debugging
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        
        # Decode
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\nGenerated text:\n{generated}")
        
    except Exception as e:
        print(f"Generation failed: {e}")
        
        # Try with even simpler settings
        print("\nTrying with minimal generation...")
        with torch.no_grad():
            # Just get logits
            model_outputs = model(**inputs)
            logits = model_outputs.logits
            print(f"Logits shape: {logits.shape}")
            print(f"Logits range: [{logits.min():.2f}, {logits.max():.2f}]")
            
            # Check for NaN/Inf
            if torch.isnan(logits).any():
                print("WARNING: NaN values in logits!")
            if torch.isinf(logits).any():
                print("WARNING: Inf values in logits!")


def test_tessellate_prompt():
    """Test with actual Tessellate game prompt."""
    
    print("\n" + "="*50)
    print("Testing Tessellate-specific prompt")
    print("="*50)
    
    model_name = "Qwen/Qwen3-4B-Thinking-2507"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Simple game state
    game_prompt = """You are playing Tessellate. The board is empty.

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

Select your move in format: Action: (row, col, corner)"""

    messages = [{"role": "user", "content": game_prompt}]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    print(f"Prompt length: {len(text)} characters")
    
    # Check tokenization
    inputs = tokenizer(text, return_tensors="pt")
    print(f"Token count: {inputs.input_ids.shape[1]}")
    
    # Check special tokens
    print(f"\nSpecial tokens:")
    print(f"  BOS token: {tokenizer.bos_token} (ID: {tokenizer.bos_token_id})")
    print(f"  EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
    print(f"  PAD token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    
    # Check for think tokens
    think_open = tokenizer.encode("<think>", add_special_tokens=False)
    think_close = tokenizer.encode("</think>", add_special_tokens=False)
    print(f"  <think> tokens: {think_open}")
    print(f"  </think> tokens: {think_close}")


if __name__ == "__main__":
    print("Testing Qwen3-4B-Thinking model")
    print("="*50)
    
    # Test basic generation
    test_basic_generation()
    
    # Test with Tessellate prompt
    test_tessellate_prompt()