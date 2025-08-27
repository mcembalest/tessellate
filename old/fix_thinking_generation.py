#!/usr/bin/env python3
"""
Fix thinking tag generation for Qwen3-4B-Thinking-2507.
The model needs specific generation parameters to properly close thinking tags.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def test_thinking_with_proper_params():
    """Test generation with parameters that encourage thinking tag closure."""
    
    model_name = 'Qwen/Qwen3-4B-Thinking-2507'
    print(f"Testing {model_name} with proper generation parameters...")
    print("="*60)
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map=None
    )
    model = model.to("cpu")
    model.eval()
    
    print("Model loaded.\n")
    
    # Test prompts of varying complexity
    test_prompts = [
        "What is 2+2?",
        "Explain why the sky is blue in simple terms.",
        "What is the best opening move in chess and why?",
    ]
    
    think_close_token = 151668  # </think> token ID
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"Test {i}: {prompt}")
        print("-" * 40)
        
        messages = [{'role': 'user', 'content': prompt}]
        text = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        inputs = tokenizer(text, return_tensors="pt")
        
        # Try different generation strategies
        strategies = [
            {
                "name": "Default (from docs)",
                "params": {
                    "max_new_tokens": 32768,  # Long output as recommended
                    "temperature": 0.6,
                    "top_p": 0.95,
                    "top_k": 20,
                    "min_p": 0.0,
                    "do_sample": True,
                    "pad_token_id": tokenizer.pad_token_id,
                }
            },
            {
                "name": "Force </think> with eos",
                "params": {
                    "max_new_tokens": 1000,
                    "temperature": 0.7,
                    "do_sample": True,
                    "pad_token_id": tokenizer.pad_token_id,
                    "eos_token_id": [tokenizer.eos_token_id, think_close_token],  # Add </think> as EOS
                }
            },
            {
                "name": "Lower temperature",
                "params": {
                    "max_new_tokens": 1000,
                    "temperature": 0.3,
                    "do_sample": True,
                    "pad_token_id": tokenizer.pad_token_id,
                }
            }
        ]
        
        for strategy in strategies:
            print(f"\n  Strategy: {strategy['name']}")
            
            try:
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        **strategy['params']
                    )
                
                generated_ids = outputs[0][len(inputs.input_ids[0]):].tolist()
                
                # Check for </think> token
                if think_close_token in generated_ids:
                    idx = generated_ids.index(think_close_token)
                    print(f"    ✓ Found </think> at position {idx}")
                    
                    # Extract thinking and final response
                    thinking_ids = generated_ids[:idx]
                    response_ids = generated_ids[idx+1:]
                    
                    thinking_text = tokenizer.decode(thinking_ids, skip_special_tokens=True)
                    response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
                    
                    print(f"    Thinking length: {len(thinking_text)} chars")
                    print(f"    Thinking preview: {thinking_text[:100]}...")
                    print(f"    Response preview: {response_text[:100]}...")
                    
                    # Found working parameters!
                    return strategy['params']
                    
                else:
                    print(f"    ✗ No </think> token in first {len(generated_ids)} tokens")
                    
            except Exception as e:
                print(f"    Error: {e}")
        
        print()
    
    print("\nConclusion: Model may need longer generation to reach </think> token")
    print("Recommendation: Use max_new_tokens=32768 as per documentation")
    return None


def test_with_tessellate_prompt():
    """Test with actual Tessellate prompt using working parameters."""
    
    print("\n" + "="*60)
    print("Testing with Tessellate prompt")
    print("="*60)
    
    model_name = 'Qwen/Qwen3-4B-Thinking-2507'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map=None
    )
    model = model.to("cpu")
    model.eval()
    
    # Simplified Tessellate prompt
    prompt = """You are playing Tessellate. The board is currently empty.

Move 0/49, Red to play
Scores: Red=1 Blue=1

Board (5x5 grid, · means empty):
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

Choose your move in format: Action: (row, col, corner)
Where row/col are 0-4, corner is UL/UR/LL/LR."""

    messages = [{'role': 'user', 'content': prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt")
    
    print("Generating with recommended parameters (this may take a moment)...")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=2000,  # Shorter for testing
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    generated_ids = outputs[0][len(inputs.input_ids[0]):].tolist()
    think_close_token = 151668
    
    if think_close_token in generated_ids:
        idx = generated_ids.index(think_close_token)
        print(f"\n✓ SUCCESS! Found </think> at position {idx}")
        
        thinking_ids = generated_ids[:idx]
        response_ids = generated_ids[idx+1:]
        
        thinking = tokenizer.decode(thinking_ids, skip_special_tokens=True)
        response = tokenizer.decode(response_ids, skip_special_tokens=True)
        
        print(f"\nTHINKING ({len(thinking)} chars):")
        print(thinking[:500] + "..." if len(thinking) > 500 else thinking)
        
        print(f"\nRESPONSE:")
        print(response[:200])
        
        # Check for action format
        import re
        if re.search(r'Action:\s*\((\d+),\s*(\d+),\s*(UL|UR|LL|LR)\)', response):
            print("\n✓ Valid action format found!")
        else:
            print("\n✗ No valid action format")
            
    else:
        print(f"\n✗ No </think> token found in {len(generated_ids)} tokens")
        print("The model may need even more tokens to complete thinking")
        
        # Show what was generated
        text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        print(f"\nGenerated text preview:\n{text[:500]}...")


if __name__ == "__main__":
    # First test to find working parameters
    working_params = test_thinking_with_proper_params()
    
    # Then test with Tessellate
    test_with_tessellate_prompt()