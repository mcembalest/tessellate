# Tessellate GRPO Training Pipeline - Progress Report

## Overview
Built complete infrastructure for training an LLM to play Tessellate using GRPO (Group Relative Policy Optimization), following DeepSeek R1-Zero methodology.

## Core Pipeline Files

### 1. Data Preparation
- **`prepare_llm_data.py`** - Converts raw game data to LLM format with visual board representation
  - Input: `rl_prompts_*.jsonl` files  
  - Output: Full 50-move trajectories with both players
  - Board format: 5x5 visual grid (R=red, B=blue, ·=empty/blocked)
  - Action format: `(square_row, square_col, corner)`

### 2. Reasoning Generation  
- **`generate_reasoning.py`** - Adds chain-of-thought reasoning using Qwen3-4B-Thinking-2507
  - Generates `<think>...</think>` tags for game analysis
  - Issue: Model takes 7+ minutes per game on CPU, may not close thinking tags

### 3. GRPO Training
- **`grpo_trainer_fixed.py`** - Complete GRPO implementation
  - K=2 completions per state for group-relative advantages
  - Full game rollouts for reward evaluation  
  - KL divergence against post-reasoning checkpoint (Option B)
  - PPO-style clipping for stable updates

- **`train_grpo.py`** - Main training orchestrator
  - Loads trajectories, manages training loop
  - Saves checkpoints periodically
  - Configurable hyperparameters

### 4. Game Integration
- **`llm_game_interface.py`** - Batched game playing interface
  - Parses LLM outputs to game actions
  - Manages multiple games in parallel (batch_size=2)
  - Handles invalid moves with retry mechanism

- **`game_state_utils.py`** - Game state reconstruction utilities
  - Rebuilds game states from trajectories
  - Parses visual board back to array format
  - Supports mid-game evaluation

### 5. Testing & Debugging
- **`test_qwen.py`** - Basic model generation test
- **`test_tessellate_reasoning.py`** - Tests reasoning on Tessellate prompts  
- **`debug_thinking.py`** - Debugs thinking tag generation
- **`check_thinking_tags.py`** - Verifies tokenizer/template configuration
- **`fix_thinking_generation.py`** - Tests different generation strategies
- **`example_llm_trajectory.json`** - Sample trajectory format

## Data Status
- **Prepared**: 6,000 games in `llm_data/` (from 3 prompt files)
- **Format**: Each game has 50 moves with visual boards and coordinate actions
- **Reasoning**: Not yet generated due to computational constraints

## Key Design Decisions
1. **GRPO Configuration**: K=2, batch_size=2 (minimal for 96GB Mac)
2. **Rewards**: Raw score differentials, no artificial normalization
3. **KL Reference**: Model checkpoint after initial reasoning generation
4. **Rollouts**: Full game completion with random policy after model's move
5. **Action Format**: `(row, col, corner)` where corner ∈ {UL, UR, LL, LR}

## Current Blockers
1. **Thinking Tag Generation**: Qwen3-4B-Thinking-2507 requires 32K+ tokens for thinking, taking 7+ minutes per game on CPU
2. **Model Confusion**: Model doesn't know Tessellate rules, overthinks unfamiliar game
3. **Computational Cost**: CPU inference too slow for practical training iteration

## Next Steps
1. Resolve thinking generation efficiency issue
2. Generate initial reasoning for 5k games
3. Run GRPO training loop
4. Evaluate performance improvement

## Command Sequence (When Ready)
```bash
# 1. Prepare data (✓ completed)
python prepare_llm_data.py --max-files 3

# 2. Generate reasoning (⚠️ blocked by performance)
python generate_reasoning.py --max-games 100 --moves-per-game 5

# 3. Train with GRPO
python train_grpo.py --batch-size 2 --k-completions 2 --num-epochs 1
```