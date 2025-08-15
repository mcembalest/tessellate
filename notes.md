# Tessellate Test-Time Scaling Implementation Notes

## Core Concept
Demonstrate how AI performance scales with inference-time computation in the Tessellate game environment.

## Technical Architecture

### Model Design
- **Base**: Recurrent neural network with weight sharing across iterations
- **Training**: Self-play with variable iteration counts during training
- **Key Innovation**: Same model weights, different iteration budgets at inference

### Three Compute Levels
1. **Low** (10 iterations): ~50-60% win rate vs random
2. **Medium** (100 iterations): ~70-80% win rate vs random  
3. **High** (1000 iterations): ~85-95% win rate vs random

### Implementation Requirements

#### Python Components
- `tessellate_scaled_ai.py`: Model architecture with recall mechanism
- `train_with_scaling.py`: Training loop with progressive loss
- Progressive loss weight α = 0.01 for game environments
- Recall architecture: concatenate board state at each iteration

#### JavaScript Integration
- Export trained model to ONNX format
- Use ONNX.js for browser inference
- Implement iteration control in `game-pvp.js`
- Add compute budget selector in UI

#### Training Protocol
1. Initialize with random self-play
2. Use Algorithm 1 from Chapter 5:
   - Sample n ~ U{0, max_iters-1}
   - Sample k ~ U{1, max_iters-n}
   - Train on partial solutions
3. Combine progressive loss with standard loss
4. Train for 10k games minimum

## Key Algorithms

### Recall Architecture (from Chapter 5)
```
f_recall(x; m) := h(r^m_recall(p(x), x))
where r_recall(φ, x) := r([φ, x])
```
- Prevents forgetting the board state
- Enables convergence to fixed point
- Critical for avoiding overthinking

### Progressive Training
- Start from random intermediate evaluation
- Train to improve from that point
- Makes network iteration-agnostic
- Prevents iteration-specific behaviors

## Evaluation Metrics

### Performance Indicators
- Win rate vs baselines (random, greedy)
- Convergence speed (iterations to stable evaluation)
- Overthinking detection (performance degradation with excess iterations)

### Baselines
1. **Random**: Uniform selection from valid moves
2. **Greedy**: Maximize immediate score gain
3. **Fixed-depth**: Standard NN with fixed forward pass

## Educational Framework

### Learning Objectives
1. Test-time compute scaling improves performance in closed domains
2. More iterations enable deeper search of game tree
3. Limitations: doesn't generalize beyond trained domain

### Key Demonstrations
- Show move evaluation heatmap evolving over iterations
- Display convergence metric (Δφ < threshold)
- Compare performance at different compute budgets
- Highlight critical game moments where AI thinks longer

## Implementation Timeline

### Phase 1: Model Training (2 days)
- Implement recall architecture
- Add progressive loss training
- Run self-play training
- Validate scaling behavior

### Phase 2: Browser Integration (1 day)
- Export to ONNX
- Integrate with existing game UI
- Add compute budget controls
- Implement visualization

### Phase 3: Educational Polish (1 day)
- Create comparison visualizations
- Add explanatory text
- Test with target audience
- Package for submission

## Critical Success Factors

1. **Model must show clear scaling**: Performance should monotonically improve with iterations
2. **No overthinking**: Model should converge, not oscillate
3. **Interpretable behavior**: Visualizations should clearly show thinking process
4. **Educational clarity**: Non-experts should understand compute scaling concept

## Code Integration Points

### Existing Files to Modify
- `index.html`: Add AI opponent option
- `game-pvp.js`: Integrate AI move selection
- `browser.html`: Add thinking visualization

### New Files Required
- `tessellate_scaled_ai.py`: Model implementation
- `train_scaled.py`: Training script
- `ai_player.js`: Browser-side inference
- `models/tessellate_scaled.onnx`: Trained model

## References
- Chapter 4: Initial recurrent architectures for games
- Chapter 5: Recall architecture and progressive training
- SPCT paper: Principle-based evaluation at scale