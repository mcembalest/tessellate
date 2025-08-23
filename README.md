# Tessellate

A simple and surprising two-player board game as a testing ground for studying reinforcement learning.

<img width="430" height="533" alt="Tessellate Game" src="https://github.com/user-attachments/assets/e1099218-2efc-4d6b-9dc3-114493b3c8f8" />

## Game Rules

Tessellate is played on a 10×10 grid where:
- Players take turns placing right triangular tiles in square cells
- Tiles of the same color form islands when connected by edges
- Score = product (multiplication) of all your island sizes
- Placing a tile blocks two adjacent corners (strategic blocking)
- Highest score wins after 50 moves

## Using This Repository

```bash
# Generate random game simulations
uv run generate_games.py --num-games 10000

# Process games into data formatted for RL 
uv run preprocess_tessellate.py --mode large

# Train PQN agent
uv run pqn_train.py
```

### Project Structure

```
tessellate/
├── Core Game
│   ├── tessellate.py            # Game logic
│   ├── tessellate_env.py        # RL environment
│   └── tessellate_agent.py      # Agent interface
├── PQN Implementation
│   ├── pqn_model.py             # Neural network with LayerNorm
│   ├── pqn_train.py             # Training without replay buffer
│   └── data_loader.py           # Efficient data streaming
└── Data Pipeline
    ├── generate_games.py        # Generate self-play games
    └── preprocess_tessellate.py # Convert to RL format
```

### Example RL Environment Usage

```python
from tessellate_env import TessellateEnv

env = TessellateEnv(reward_mode='mixed')
obs = env.reset()

while not env.is_terminal():
    valid_actions = env.get_valid_actions()
    action = agent.select_action(obs, valid_actions)
    obs, reward, done, info = env.step(action)
```

### RL Environment Design

**Observation Space**: `(104,)` array
- `[0:100]`: Flattened 10×10 board (0=empty, 1=red, 2=blue, 3=blocked)
- `[100]`: Current player (1 or 2)
- `[101]`: Red score
- `[102]`: Blue score  
- `[103]`: Move number (0-49)

**Action Space**: Integer 0-99 (index into flattened board)

**Reward**: given at end of game, proportional to size of win/loss disparity

### Tessellate Coordinate System Logic

The game uses a 10×10 logical grid mapped to a 5×5 visual grid:
- Logical `(r,c)` → Square `(r÷2, c÷2)`, Corner `(r%2, c%2)`
- (0,0) = top-left, (0,1) = top-right, (1,0) = bottom-left, (1,1) = bottom-right

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


### Bot-vs-Bot (local) via JSONL protocol

Two PQN checkpoints can now play each other locally using a lightweight stdin/stdout JSONL protocol.

Files:
- agent_protocol.py – message schema (JSONL)
- pqn_agent_cli.py – loads a PQN checkpoint and answers move requests
- referee.py – runs the match using TessellateEnv and records games compatible with the browser viewer

Example:

```bash
uv run referee.py \n  --agent1 "uv run pqn_agent_cli.py --checkpoint checkpoints/pqn_model_batch50_20250819_170514.pt --device cpu" \
  --agent2 "uv run pqn_agent_cli.py --checkpoint checkpoints/pqn_model_batch100_20250819_170515.pt --device cpu" \
  --games 5 \
  --out game_data/pqn_vs_pqn.json
```

Open browser.html and load the generated JSON to replay the games.
