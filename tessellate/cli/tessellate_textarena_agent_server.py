#!/usr/bin/env python3
"""
Run a lightweight HTTP server that lets the browser play Tessellate against
an LLM via TextArena's agents (e.g., OpenRouterAgent).

Endpoint:
  POST /move
    body: { "state": [104 numbers], "valid_actions": [ints like r*10+c] }
    returns: { "action": int, "explanation": str }

How it works (browser <-> server):
  - The browser keeps the authoritative board and turn.
  - When it's the AI's turn, it posts the compact state vector and the list of
    valid actions to this server.
  - This server reconstructs a textual board view, crafts a short instruction
    prompt, asks the LLM agent for a move, parses the move (same formats the
    TextArena env accepts), and returns the chosen action as a flat integer
    (r*10 + c). If parsing fails or the move is invalid, we fall back to a
    random valid action.

"""

from __future__ import annotations

import argparse
import os
import random
import re
from typing import List, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import textarena as ta

from tessellate.game import TessellateGame, TileState
from tessellate.environments.tessellate_textarena_env import TessellateTaEnv, VIS_ROWS


# ---------- Utilities ----------

def flat_to_rc(action_flat: int) -> Tuple[int, int]:
    return action_flat // 10, action_flat % 10


def rc_to_visual(r: int, c: int) -> str:
    sr, sc = r // 2, c // 2
    cr, cc = r % 2, c % 2
    corner = { (0,0): 'UL', (0,1): 'UR', (1,0): 'LL', (1,1): 'LR' }[(cr, cc)]
    return f"{VIS_ROWS[sr]}{sc}-{corner}"


def valid_actions_visual(valid_actions: List[int], max_items: int = 50) -> str:
    items = []
    for a in valid_actions[:max_items]:
        r, c = flat_to_rc(a)
        items.append(rc_to_visual(r, c))
    if len(valid_actions) > max_items:
        items.append("...")
    return ", ".join(items)


def parse_llm_move(text: str) -> Tuple[int | None, int | None]:
    """Parse a move from LLM text. Returns (r, c) or (None, None).

    Accepts multiple formats, matching the TextArena env:
      - Visual square: [A0-UL] or A0-UL
      - Logical cell:  [A0] or A0   (A-J rows, 0-9 cols)
      - Raw tuple:     [r, c]
      - Tuple visual:  (square_r, square_c, CORNER)
    """
    # 1) Visual square format: [A0-UL]
    m_vis = re.search(r"\b\[?([A-Ea-e])\s*([0-4])\s*[-, ]\s*(UL|UR|LL|LR)\]?\b", text)
    if m_vis:
        sr = ord(m_vis.group(1).upper()) - ord('A')
        sc = int(m_vis.group(2))
        cr, cc = {'UL': (0,0), 'UR': (0,1), 'LL': (1,0), 'LR': (1,1)}[m_vis.group(3).upper()]
        return 2*sr + cr, 2*sc + cc

    # 2) Logical cell format: [A0]
    m_cell = re.search(r"\b\[?([A-Ja-j])\s*([0-9])\]?\b", text)
    if m_cell:
        r = ord(m_cell.group(1).upper()) - ord('A')
        c = int(m_cell.group(2))
        return r, c

    # 3) Othello-style [r, c]
    m_rc = re.search(r"\[\s*(\d)\s*,\s*(\d)\s*\]", text)
    if m_rc:
        return int(m_rc.group(1)), int(m_rc.group(2))

    # 4) Tuple visual: (square_r, square_c, CORNER)
    m_tuple = re.search(r"\(\s*(\d)\s*,\s*(\d)\s*,\s*(UL|UR|LL|LR)\s*\)", text, re.IGNORECASE)
    if m_tuple:
        sr, sc = int(m_tuple.group(1)), int(m_tuple.group(2))
        if 0 <= sr < 5 and 0 <= sc < 5:
            cr, cc = {'UL': (0,0), 'UR': (0,1), 'LL': (1,0), 'LR': (1,1)}[m_tuple.group(3).upper()]
            return 2*sr + cr, 2*sc + cc

    return None, None


def state_to_game(state: List[int]) -> TessellateGame:
    if len(state) != 104:
        raise ValueError("state must be length 104")
    game = TessellateGame()
    # Rebuild board
    for idx in range(100):
        r, c = divmod(idx, 10)
        v = state[idx]
        if v == 0:
            game.board[r][c] = TileState.EMPTY
        elif v == 1:
            game.board[r][c] = TileState.RED
        elif v == 2:
            game.board[r][c] = TileState.BLUE
        elif v == 3:
            game.board[r][c] = TileState.BLOCKED
        else:
            game.board[r][c] = TileState.EMPTY
    # Turn
    turn_v = state[100]
    game.current_turn = TileState.RED if turn_v == 1 else TileState.BLUE
    # Scores (best-effort; browser recomputes anyway)
    red_score = state[101] if isinstance(state[101], (int, float)) and state[101] else 1
    blue_score = state[102] if isinstance(state[102], (int, float)) and state[102] else 1
    game.scores = {TileState.RED: int(red_score), TileState.BLUE: int(blue_score)}
    # Placements
    game.placed_tiles_count = int(state[103])
    game.game_over = game.placed_tiles_count >= game.TOTAL_TILES
    return game


# ---------- FastAPI types ----------

class MoveRequest(BaseModel):
    state: List[int] = Field(..., description="Length-104 state vector")
    valid_actions: List[int] = Field(..., description="List of valid flat actions: r*10+c")


class MoveResponse(BaseModel):
    action: int
    explanation: str | None = None


def build_prompt(env: TessellateTaEnv, game: TessellateGame, valid_actions: List[int]) -> str:
    # Borrow Env rendering to keep visuals consistent
    env.game = game
    board_str = env._render_board(full=False)
    turn_str = "RED" if game.current_turn == TileState.RED else "BLUE"
    moves_visual = valid_actions_visual(valid_actions, max_items=50)
    instructions = (
        "You are playing Tessellate. Place one triangular tile per turn.\n"
        "- Reply with a single move. Prefer the short form [A0] or the visual form A0-UR.\n"
        "- Valid row letters: A-J (logical grid), or A-E with UL/UR/LL/LR for the visual squares.\n"
        "- Only choose empty, not blocked, positions.\n"
    )
    msg = (
        f"{board_str}\n"
        f"Turn: {turn_str}.\n"
        f"Valid moves (visual, up to 50): {moves_visual}\n\n"
        f"{instructions}"
    )
    return msg


def create_app(model_name: str) -> FastAPI:
    app = FastAPI(title="Tessellate TextArena Agent Server")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "https://tessellate-app-ytnku.ondigitalocean.app",
            "http://localhost:8424",
            "http://127.0.0.1:8424",
            "http://localhost:8080",
            "http://127.0.0.1:8080",
            "null"  # if you open index.html directly via file://
        ],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Create the agent once
    agent = ta.agents.OpenRouterAgent(model_name=model_name)
    env = TessellateTaEnv()  # for rendering helpers only

    @app.post("/move", response_model=MoveResponse)
    def move(req: MoveRequest):
        if not req.valid_actions:
            raise HTTPException(status_code=400, detail="No valid actions provided")

        try:
            game = state_to_game(req.state)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Bad state: {e}")

        # Build prompt and ask the agent
        prompt = build_prompt(env, game, req.valid_actions)
        try:
            raw = agent(prompt)
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Agent error: {type(e).__name__}: {e}")

        # Parse the agent's choice
        r, c = parse_llm_move(str(raw))
        action = None
        if r is not None and c is not None:
            action = r * 10 + c
            if action not in req.valid_actions:
                action = None

        if action is None:
            # Do not fallback randomly; let client handle the error explicitly.
            raise HTTPException(status_code=422, detail="Could not parse a valid move from agent output.")

        # Short, friendly explanation for the UI
        explanation = str(raw).strip()
        if len(explanation) > 280:
            explanation = explanation[:277] + "..."

        return MoveResponse(action=action, explanation=explanation)

    @app.get("/health")
    def health():
        return {"ok": True, "model": model_name}

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.environ.get("TESSELLATE_AGENT_MODEL", "o4-mini"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()

    app = create_app(model_name=args.model)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
