# Tessellate: An RL Research Environment

A two-player board game with multiplicative scoring that provides an interesting testbed for reinforcement learning research.

<img width="430" height="533" alt="Tessellate Game" src="https://github.com/user-attachments/assets/e1099218-2efc-4d6b-9dc3-114493b3c8f8" />



## Quick Start

```bash
# Option 1: Jupyter Notebook (recommended)
jupyter notebook tessellate_rl.ipynb

# Option 2: Direct Python
pip install torch numpy
python -c "from tessellate_env import TessellateEnv; env = TessellateEnv(); print(env.reset().shape)"

# Explore games visually
python -m http.server 8000
# Open http://localhost:8000/browser.html
```

## Game Rules

Tessellate is a two-player game played on a 5×5 grid where:
- Players take turns placing triangular tiles (25 moves each)
- Tiles of the same color form islands when connected by edges
- Score = product of all your island sizes (multiplicative scoring!)
- Highest score wins after all 50 tiles are placed

The multiplicative scoring creates interesting strategic trade-offs between growing existing islands and starting new ones.

## For RL Researchers

### Why Tessellate?

- **Non-trivial**: Random play baseline achieves ~50% win rate
- **Fast simulation**: ~3000 games/second on CPU
- **Perfect information**: Fully observable, deterministic
- **Short episodes**: Exactly 50 moves per game
- **Rich strategy space**: Multiplicative scoring creates complex value functions

### Environment API

```python
from tessellate_env import TessellateEnv

# Create environment
env = TessellateEnv(reward_mode='mixed')
obs = env.reset()

# Game loop
while not env.is_terminal():
    valid_actions = env.get_valid_actions()
    action = agent.select_action(obs, valid_actions)
    obs, reward, done, info = env.step(action)
```

**Observation Space**: `(101,)` array
- `[0:100]`: Flattened 10×10 board (0=empty, 1=red, 2=blue, 3=blocked)
- `[100]`: Current player (1 or 2)

**Action Space**: Integer 0-99 (index into flattened board)

**Reward Modes**:
- `'sparse'`: +1/-1 only at game end
- `'immediate'`: Score change at each move
- `'mixed'`: Combination of immediate and terminal rewards

## Repository Structure

```
tessellate/
├── tessellate_rl.ipynb    # Start here! Complete RL tutorial
├── tessellate_env.py      # RL environment
├── tessellate.py          # Core game logic  
├── browser.html           # Game viewer
├── index.html             # Play the game
└── game_data/             # Pre-generated games
```

## Implementation Details

### Coordinate System

The game uses a 10×10 logical grid mapped to a 5×5 visual grid:
- Logical position `(r,c)` → Visual square `(r÷2, c÷2)`, Corner `(r%2, c%2)`
- Even/even = top-left triangle
- Even/odd = top-right triangle
- Odd/even = bottom-left triangle
- Odd/odd = bottom-right triangle


## Planned Features

### Score Ratio Sparklines (TODO)

Add visual momentum indicators to the game browser:

**Sparkline Specification:**
- **Location**: 
  - Small sparkline next to each game in the sidebar list
  - Larger sparkline above/below main game canvas
- **Calculation**: 
  - Score ratio = Red Score / Blue Score
  - Handle edge cases: min(score, 1) to avoid divide by zero
  - Ratio > 1: Red is winning
  - Ratio < 1: Blue is winning  
  - Ratio = 1: Even game
- **Visual Style**:
  - Thin line graph (2px stroke)
  - Height: 20px for sidebar, 40px for main view
  - Color: Gradient from blue (ratio < 1) through white (ratio = 1) to red (ratio > 1)
- **Scale**:
  - Logarithmic Y-axis centered at y=1
  - Range: 0.1x to 10x (one order of magnitude each direction)
  - Horizontal line at y=1 for reference
- **Interactivity**:
  - Click anywhere on sparkline to jump to that move
  - Hover shows tooltip with exact scores
  - Current move position marked with vertical line

**Implementation Notes**:
- Use Canvas API for performance with 1000+ games
- Cache sparkline renders for game list
- Update main sparkline on each move during playback

## Citation

If you use Tessellate in your research, please cite:
```bibtex
@software{tessellate2024,
  title = {Tessellate},
  author = {Cembalest, Max},
  year = {2025},
  url = {https://github.com/maxcembalest/tessellate}
}
```
