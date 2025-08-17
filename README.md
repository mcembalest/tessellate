# Tessellate: An RL Research Environment

A two-player board game with that provides an interesting testbed for reinforcement learning research. The game has nontrivial scoring dynamics and moments of criticality (in the sense of chaos theory).

<img width="430" height="533" alt="Tessellate Game" src="https://github.com/user-attachments/assets/e1099218-2efc-4d6b-9dc3-114493b3c8f8" />

## Game Rules

Tessellate is a two-player game played on a square grid where:
- Players take turns placing triangular tiles
- Tiles of the same color form islands when connected by edges
- Score = product (multiplication) of all your island sizes
- Highest score wins after all tiles are placed

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


## Implementation Details

### Coordinate System

The game uses a 10×10 logical grid mapped to a 5×5 visual grid:
- Logical position `(r,c)` → Visual square `(r÷2, c÷2)`, Corner `(r%2, c%2)`
- Even/even = top-left triangle
- Even/odd = top-right triangle
- Odd/even = bottom-left triangle
- Odd/odd = bottom-right triangle

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
