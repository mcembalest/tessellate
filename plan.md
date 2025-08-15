# Tessellate RL Implementation Plan
## Imemdiate Project Goals

Train an interpretable RL model on Tessellate to understand emergent strategic features without hand-crafted heuristics.

---

## Game Rules Summary
1. **Board**: 5x5 grid of squares, each square can hold 2 triangular tiles
2. **Tiles**: Right triangles that fill half a square
3. **Placement**: Placing in one corner blocks the two adjacent corners in that square
4. **Islands**: Tiles of same color connected by full edges (BOTH hypotenuse AND leg connections count)
5. **Scoring**: Product of all island sizes (multiplicative scoring!)
6. **Win Condition**: Highest score when all 25 squares are filled with all 50 tiles.

## Current Implementation Status

### ✅ Completed Components

#### Game Engine (`tessellate.py`)
- Fully functional Python port from JavaScript
- 10x10 logical grid representation (preserving JS structure)
- Verified adjacency logic with comprehensive testing
- Island detection via DFS
- Multiplicative scoring system
- ~75 games/second generation speed

#### Data Generation (`agents.py`)
- Random agent implementation
- Game recording with full move history
- JSON serialization format
- Batch game generation (1000+ games)
- Statistics tracking (win rates, score distributions)

#### Game Viewer (`viewer.html`)
- Lichess-style interface for browsing games
- Move-by-move navigation with keyboard shortcuts
- Real-time score updates
- Game list with statistics
- Auto-play animation mode
- **Critical Fix**: Triangle rendering must use exact coordinate mapping from original JS

## Critical Implementation Details (MUST PRESERVE)

### Coordinate System
- **Logical Grid**: 10x10 array where each position represents a triangle corner
- **Visual Grid**: 5x5 squares displayed to user
- **Mapping**: Position (r,c) → Square(r÷2, c÷2), Corner(r%2, c%2)
  - (even, even) = top-left triangle
  - (even, odd) = top-right triangle  
  - (odd, even) = bottom-left triangle
  - (odd, odd) = bottom-right triangle

### Triangle Drawing (viewer.html critical code)
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

### Adjacency Logic (Complex but Verified)
- Tiles connect via BOTH hypotenuse and leg edges
- Neighbor calculation uses parity-based offsets
- All neighbor relationships are symmetric
- Corner positions have 1-3 neighbors, middle positions have 5

### Blocking Mechanism
When placing tile at (r,c):
1. Block horizontally adjacent: `c + (c%2 === 0 ? 1 : -1)`
2. Block vertically adjacent: `r + (r%2 === 0 ? 1 : -1)`
3. This prevents overlapping triangles in same square

## Technical Specifications

### State Representation
- **Grid**: 10x10 logical grid (MUST match JavaScript)
- **States per position**: 4 values (EMPTY=0, RED=1, BLUE=2, BLOCKED=3)
- **Total positions**: 100 (but only 50 can have tiles)

### Model Architecture
- **Type**: 2-layer transformer
- **Input**: Flattened board state (100 positions × 4 one-hot = 400 dims)
- **Output**: Policy head (100 logits) + Value head (1 scalar)
- **Training**: Self-play with MCTS

### Features to Detect (Interpretability Focus)
1. **Tile-level**: Individual position importance
2. **Island-level**: Detection and size tracking
3. **Action-level**: 
   - Grow (extend existing island)
   - Merge (connect two islands)
   - Block (defensive moves)
4. **Strategic**: Emergent multiplication dynamics

---

## Data Format

### Game JSON Structure
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

### Current Statistics (1000 random games)
- Win distribution: ~49% RED, ~49% BLUE, ~2% ties
- Average scores: RED ~660, BLUE ~640
- Max score observed: 3888
- All games complete in exactly 50 moves

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

### Phase 4: Neural Network (Hours 12-20)
- [ ] **4.1** Install PyTorch
- [ ] **4.2** Define 2-layer transformer architecture
- [ ] **4.3** Implement forward pass
- [ ] **4.4** Test with dummy input (shape validation)
- [ ] **4.5** Add policy and value heads
- [ ] **4.6** Implement loss functions

### Phase 5: MCTS Integration (Hours 20-28)
- [ ] **5.1** Basic MCTS node structure
- [ ] **5.2** Selection phase (UCB formula)
- [ ] **5.3** Expansion phase
- [ ] **5.4** Simulation phase (using neural network)
- [ ] **5.5** Backpropagation phase
- [ ] **5.6** Integration test (1 full game with MCTS)

### Phase 6: Self-Play Training (Hours 28-36)
- [ ] **6.1** Self-play game generation
- [ ] **6.2** Experience buffer
- [ ] **6.3** Training loop (network updates)
- [ ] **6.4** Save/load model checkpoints
- [ ] **6.5** Basic training metrics (loss curves)
- [ ] **6.6** Run overnight training

### Phase 7: Interpretability (Hours 36-44)
- [ ] **7.1** Attention weight extraction
- [ ] **7.2** Feature probe: Island detection
- [ ] **7.3** Feature probe: Action classification
- [ ] **7.4** Visualize learned positional preferences
- [ ] **7.5** Analyze strategic patterns
- [ ] **7.6** Document findings

### Phase 8: Polish & Analysis (Hours 44-48)
- [ ] **8.1** Generate substantial game dataset
- [ ] **8.2** Statistical analysis of learned strategies
- [ ] **8.3** Create visualization notebook
- [ ] **8.4** Write up key findings
- [ ] **8.5** Package code for easy reproduction

---

## Key Implementation Notes

### Adjacency Logic (Critical!)
```python
def get_neighbors(r, c):
    """Get all adjacent tiles (sharing full edges - hypotenuse or leg)"""
    neighbors = []
    # Complex neighbor logic from JavaScript
    # Both straight edges (legs) and diagonal edges (hypotenuse) count
    return neighbors
```

### MCTS Parameters (Tunable)
- **Simulations per move**: Start with 100 (fast), increase if needed
- **Exploration constant (c_puct)**: 1.0 initially
- **Dirichlet noise**: Alpha=0.3 for exploration during self-play
- **Temperature**: 1.0 for first 15 moves, then 0.1

### Neural Network Hyperparameters
- **Embedding dim**: 64 (small for interpretability)
- **Attention heads**: 4
- **Hidden dim**: 128
- **Learning rate**: 0.001
- **Batch size**: 32
- **Training games before update**: 100

### Interpretability Tools
1. **Attention maps**: Which positions attend to which
2. **Probing classifiers**: Train linear probes on hidden states
3. **Feature visualization**: What patterns activate neurons
4. **Behavioral tests**: Specific board positions to test understanding

---

## Success Metrics
- **Phase 1-3**: Game runs without crashes, scores computed correctly
- **Phase 4-5**: Network makes legal moves, MCTS improves play
- **Phase 6**: Self-play games show strategic improvement over time
- **Phase 7**: Can identify at least 3 emergent features (islands, merging, blocking)

## Fallback Plans
- If MCTS too slow: Reduce to greedy search with network evaluation
- If transformer too complex: Fall back to small CNN
- If training unstable: Use behavior cloning on good random games first
- If 48 hours not enough: Prioritize working pipeline over perfect model

---

## Quick Start Commands
```bash
# Setup
python tessellate.py --test  # Verify game logic

# Generate data
python tessellate.py --random-games 1000  # Baseline data

# Train
python tessellate.py --self-play --games 10000  # Start training

# Analyze
python tessellate.py --analyze model_checkpoint.pt  # Interpret
```

## Key Lessons Learned

### Visual Verification is Critical
- Always verify game logic with visual confirmation before proceeding
- Triangle rendering bugs can make correct logic appear wrong
- Test with known positions and expected outputs

### Coordinate System Complexity
- The 10x10 logical grid to 5x5 visual grid mapping is non-trivial
- Position parity determines triangle orientation within square
- Must preserve exact JavaScript coordinate system for consistency

### Adjacency is Non-Obvious  
- Both hypotenuse AND leg connections form islands (not intuitive)
- Neighbor relationships must be symmetric
- Complex parity-based calculation but verified to work correctly

### Data Pipeline Works Well
- ~75 games/second generation is sufficient for initial experiments
- JSON format preserves full game history for replay
- Random play produces reasonable score distributions

## Next Steps (When Resuming)

### Immediate Priority: More Data & Analysis
1. Generate 10,000+ games for better statistics
2. Analyze common patterns in high-scoring games
3. Identify strategic positions (corners, edges, center)

### Model Development Path
1. Start with simple features (island counts, positions)
2. Use small transformer for interpretability
3. Focus on attention patterns to understand decision-making
4. MCTS only after baseline model works

### Tools to Build
1. Position heatmaps (where do winners play?)
2. Island size distribution analysis
3. Move sequence pattern mining
4. Score trajectory visualization

## Resources & References
- Original game: `index.html` (JavaScript implementation)
- Python engine: `tessellate.py` (verified port)
- Game viewer: `viewer.html` (lichess-style browser)
- Board representation: 10x10 logical grid (MUST match JS)
- Scoring: Multiplicative (unique strategic element)
- Goal: Interpretability > Performance