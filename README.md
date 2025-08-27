# Tessellate

A simple and surprising two-player board game as a testing ground for studying reinforcement learning.

<img width="430" height="533" alt="Tessellate Game" src="https://github.com/user-attachments/assets/e1099218-2efc-4d6b-9dc3-114493b3c8f8" />

## Game Rules

Tessellate is played on a 10×10 grid where:
- Players take turns placing right triangular tiles in square cells
- Tiles of the same color form islands when connected by edges
- Score = product (multiplication) of all your island sizes
- Highest score wins after all the squares are filled

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

## Play in the browser: Human (Red) vs LLM (Blue)

The in-browser PvP UI (index.html) supports playing against an AI. You can use either:
- the built-in in-browser PQN model (leave Agent URL blank), or
- a TextArena LLM agent via a tiny local HTTP server (recommended for LLMs).

Quick start with a TextArena LLM agent:

1) Start the agent server (requires OPENROUTER_API_KEY in your environment):

```bash
export OPENROUTER_API_KEY=...  # your key
uv run python -m tessellate.cli.tessellate_textarena_agent_server   --model "GPT-4o-mini"   --host 127.0.0.1 --port 8001
```

This launches an HTTP server with a /move endpoint that the browser calls each time the AI needs to move.

2) Open index.html in your browser. In the controls:
- Check "Play vs Agent"
- Ensure "AI Side" is Blue (default)
- Agent URL defaults to http://127.0.0.1:8001 (keep or edit as needed)

3) Play as Red. The LLM will play as Blue and its short rationale will appear under the scoreboard.

Under the hood, the browser sends a compact state vector and valid moves to the server. The server translates the state into a TextArena-style board view, prompts the LLM agent for a move, parses common coordinate formats (e.g., [A0], A0-UR), and returns the chosen move back to the browser.
