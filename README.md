# Tessellate

A simple and surprising board game.

<img width="430" height="533" alt="Screenshot 2025-08-05 at 6 54 24 AM" src="https://github.com/user-attachments/assets/e1099218-2efc-4d6b-9dc3-114493b3c8f8" />

## Tessellate Game Rules

1. **Board**: 5x5 grid of squares, each square can hold 2 triangular tiles
2. **Tiles**: Right triangles that fill half a square
3. **Placement**: Placing in one corner blocks the two adjacent corners in that square
4. **Islands**: Tiles of same color connected by full edges (BOTH hypotenuse AND leg connections count)
5. **Scoring**: Product of all island sizes (multiplicative scoring!)
6. **Win Condition**: Highest score when all 25 squares are filled with all 50 tiles.

### Game Specifications

- **Grid**: 10x10 logical grid (MUST match JavaScript)
- **States per position**: 4 values (EMPTY=0, RED=1, BLUE=2, BLOCKED=3)
- **Total positions**: 100 (but only 50 can have tiles)

## Reinforcement Learning on Tessellate

### Model Architecture and Hyperparameters
- **Type**: 2-layer MLP (simple for interpretability)
- **Input**: Flattened board state (100 positions) + current player (1) = 101 dims
- **Output**: Next move prediction (100 logits for each board position)
- **Training**: Behavioral cloning on 1M+ random games
- **Architecture**: 2-layer MLP (no attention needed)
- **Hidden dim**: 128 neurons
- **Learning rate**: 0.001
- **Batch size**: 64
- **Training data**: 1M+ random games
- **Loss**: Cross-entropy for next-move prediction

### Expected Features to Detect
1. **Tile-level**: Individual position importance
2. **Island-level**: Detection and size tracking
3. **Action-level**: 
   - Grow (extend existing island)
   - Merge (connect two islands)
   - Block (defensive moves)

### Data Format

#### Game JSON Structure
```json
{
  "moves": [
    {
      "player": 1,  // 1=RED, 2=BLUE
      "position": [r, c],  // Array format [row, col]
      "score_before": {"1": score, "2": score}
    }
  ],
  "final_scores": {"red": N, "blue": N},
  "winner": 1/2/null,  // null for tie
  "total_moves": 50
}
```

### Interpretability Tools
1. **Neuron activation analysis**: Which neurons detect islands/merges
2. **Probing classifiers**: Train linear probes on hidden states
3. **Feature visualization**: What patterns activate neurons
4. **Behavioral tests**: Compare model vs heuristic agent decisions

## Implementation Checklist

### ✅ Phase 1: Foundation (COMPLETE)
- [x] **1.1** Create this plan.md file
- [x] **1.2** Set up Python project structure (tessellate.py)
- [x] **1.3** Port basic game state class from JavaScript
- [x] **1.4** Implement board initialization (10x10 grid)
- [x] **1.5** Implement valid moves function
- [x] **1.6** Implement tile placement logic (with blocking)

### ✅ Phase 2: Game Logic (COMPLETE)
- [x] **2.1** Implement island detection (DFS for connected components)
- [x] **2.2** Implement scoring (product of island sizes)
- [x] **2.3** Add game-over detection
- [x] **2.4** Create simple text/ASCII visualization
- [x] **2.5** Validate with manual test game
- [x] **2.6** Compare outputs with JavaScript version

### ✅ Phase 3: Basic Agents & Visualization (COMPLETE)
- [x] **3.1** Implement random agent
- [x] **3.2** Generate 100+ test games with random play
- [x] **3.3** Verify score distributions make sense
- [x] **3.4** Log game statistics (avg score, game length)
- [x] **3.5** Save games in JSON format
- [x] **3.6** Build lichess-style game viewer
- [x] **3.7** Fix triangle rendering issues

### Phase 4: Neural Network & Training (In Progress)
- [x] **4.1** Install PyTorch
- [x] **4.2** Define 2-layer MLP architecture (101→128→100)
- [x] **4.3** Implement next-token prediction
- [x] **4.4** Create training pipeline
- [ ] **4.5** Train on 1M+ games
- [ ] **4.6** Implement evaluation agents

### Phase 5: Evaluation Agents
- [ ] **5.1** Random baseline agent
- [ ] **5.2** Merge-avoiding heuristic agent
- [ ] **5.3** Model-based agent (picks highest probability move)
- [ ] **5.4** Tournament system (100+ games per matchup)
- [ ] **5.5** Win rate and score analysis
- [ ] **5.6** Strategy comparison

### Phase 6: Mechanistic Interpretability
- [ ] **6.1** Probe for island detection neurons
- [ ] **6.2** Analyze merge vs non-merge decisions
- [ ] **6.3** Position preference heatmaps
- [ ] **6.4** Hidden layer feature analysis
- [ ] **6.5** Ablation studies
- [ ] **6.6** Document emergent strategies

### Phase 7: Final Analysis & Documentation
- [ ] **7.1** Compare learned strategies across agents
- [ ] **7.2** Statistical analysis of model predictions
- [ ] **7.3** Create visualization notebook
- [ ] **7.4** Write up key findings
- [ ] **7.5** Package code for reproduction

## RL Model Notes


---

## Implementation Details (DO NOT DELETE)

### Coordinate System
- **Logical Grid**: 10x10 array where each position represents a triangle corner
- **Visual Grid**: 5x5 squares displayed to user
- **Mapping**: Position (r,c) → Square(r÷2, c÷2), Corner(r%2, c%2)
  - (even, even) = top-left triangle
  - (even, odd) = top-right triangle  
  - (odd, even) = bottom-left triangle
  - (odd, odd) = bottom-right triangle

### Triangle Drawing (browser.html critical code)

```javascript
// MUST use visualCellSize calculated from canvas width
visualCellSize = canvas.width / VISUAL_GRID_SIZE;

// Triangle vertices for position (r,c)
const visualY = Math.floor(r / 2);
const visualX = Math.floor(c / 2);
const x0 = visualX * visualCellSize;
const y0 = visualY * visualCellSize;
const x1 = (visualX + 1) * visualCellSize;
const y1 = (visualY + 1) * visualCellSize;

// Corner points
const tl = { x: x0, y: y0 };
const tr = { x: x1, y: y0 };
const bl = { x: x0, y: y1 };
const br = { x: x1, y: y1 };

// Triangle orientation based on position parity
if (r%2 === 0 && c%2 === 0) points = [tl, tr, bl];
else if (r%2 === 0 && c%2 !== 0) points = [tr, br, tl];
else if (r%2 !== 0 && c%2 === 0) points = [bl, tl, br];
else points = [br, bl, tr];
```

### Adjacency Logic
- Tiles connect via BOTH hypotenuse and leg edges
- Neighbor calculation uses parity-based offsets
- All neighbor relationships are symmetric
- Corner positions have 1-3 neighbors, middle positions have 5

### Blocking Mechanism
When placing tile at (r,c):
1. Block horizontally adjacent: `c + (c%2 === 0 ? 1 : -1)`
2. Block vertically adjacent: `r + (r%2 === 0 ? 1 : -1)`
3. This prevents overlapping triangles in same square

<img width="2048" height="2048" alt="image" src="https://github.com/user-attachments/assets/47aca105-4e9a-4416-a06a-98b9c511a86d" />