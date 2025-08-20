#!/usr/bin/env python3
"""
Check if Qwen3-4B-Thinking-2507 properly handles thinking tags.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def check_thinking_setup():
    """Verify thinking tag setup in model and tokenizer."""
    
    model_name = 'Qwen/Qwen3-4B-Thinking-2507'
    print(f"Checking {model_name} thinking tag configuration...")
    print("="*60)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Check if thinking tokens exist
    print("\n1. Checking special tokens in vocabulary:")
    think_open = tokenizer.encode('<think>', add_special_tokens=False)
    think_close = tokenizer.encode('</think>', add_special_tokens=False)
    
    print(f"   <think> token IDs: {think_open}")
    print(f"   </think> token IDs: {think_close}")
    
    if len(think_open) == 1 and len(think_close) == 1:
        print("   ✓ Think tokens are single tokens (good!)")
        print(f"   <think> = token #{think_open[0]}")
        print(f"   </think> = token #{think_close[0]}")
    else:
        print("   ✗ Think tokens are not single tokens (problem!)")
    
    # Check chat template
    print("\n2. Checking chat template:")
    messages = [{'role': 'user', 'content': 'What is 2+2?'}]
    
    # Try with default parameters
    template_default = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    print(f"   Default template output:\n   {repr(template_default)}")
    
    if '<think>' in template_default:
        print("   ✓ Default template includes <think>")
    else:
        print("   ✗ Default template does NOT include <think>")
        print("   Note: According to docs, <think> should be added automatically")
    
    # Check tokenizer config
    print("\n3. Checking tokenizer configuration:")
    if hasattr(tokenizer, 'chat_template'):
        print(f"   Chat template exists: {tokenizer.chat_template is not None}")
    
    # Test actual generation
    print("\n4. Testing actual generation:")
    print("   Loading model (this may take a moment)...")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map=None
    )
    model = model.to("cpu")
    model.eval()
    
    print("   Model loaded. Generating...")
    
    # Prepare input
    inputs = tokenizer(template_default, return_tensors="pt")
    
    # Generate with minimal settings
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,  # Greedy for consistency
            pad_token_id=tokenizer.pad_token_id,
        )
    
    # Check generated tokens
    generated_ids = outputs[0][len(inputs.input_ids[0]):].tolist()
    
    print(f"\n   First 10 generated token IDs: {generated_ids[:10]}")
    
    # Check if thinking tokens appear
    if 151667 in generated_ids:  # <think>
        print("   ✓ Found <think> token (151667) in generation")
    else:
        print("   ✗ No <think> token (151667) in generation")
        
    if 151668 in generated_ids:  # </think>
        idx = generated_ids.index(151668)
        print(f"   ✓ Found </think> token (151668) at position {idx}")
    else:
        print("   ✗ No </think> token (151668) in generation")
    
    # Decode to see actual output
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
    print(f"\n   Generated text (with special tokens):")
    print(f"   {repr(generated_text[:200])}")
    
    # Try decoding without special tokens
    generated_clean = tokenizer.decode(generated_ids, skip_special_tokens=True)
    print(f"\n   Generated text (without special tokens):")
    print(f"   {generated_clean[:200]}")
    
    # Manual check for think pattern
    print("\n5. Summary:")
    if '<think>' in generated_text or 'think>' in generated_text:
        print("   ✓ Model IS generating thinking tags")
    else:
        print("   ✗ Model is NOT generating thinking tags")
        print("\n   Possible issues:")
        print("   - Chat template might need special configuration")
        print("   - Model might need specific generation parameters")
        print("   - Check if model needs enable_thinking parameter")


if __name__ == "__main__":
    check_thinking_setup()