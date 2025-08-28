#!/usr/bin/env python3
"""
Run a lightweight HTTP server that lets the browser play Tessellate against
an LLM using OpenAI's Responses API (GPT-5 family).

Endpoints:
  POST /move
    body: { "state": [104 numbers], "valid_actions": [ints like r*10+c] }
    returns: { "action": int, "explanation": str }

  POST /move_stream (Server-Sent Events)
    streams content deltas and reasoning; ends with a final action event.

Environment variables:
  OPENAI_API_KEY                - Required
  TESSELLATE_AGENT_MODEL        - Optional, defaults to "gpt-5-mini"
  OPENAI_REASONING_EFFORT       - Optional, e.g. minimal|low|medium|high (default: medium)
  OPENAI_TEXT_VERBOSITY         - Optional, e.g. low|medium|high (if unset, omitted)
"""

from __future__ import annotations

import argparse
import os
import re
from typing import List, Tuple, Generator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, PlainTextResponse
from pydantic import BaseModel, Field
import uvicorn

from openai import OpenAI

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

def island_summary(game: TessellateGame) -> str:
    sizes = {TileState.RED: [], TileState.BLUE: []}
    visited = [[False for _ in range(game.LOGICAL_GRID_SIZE)] for _ in range(game.LOGICAL_GRID_SIZE)]

    def neighbors(r: int, c: int):
        pow_neg1_r = 1 if r % 2 == 0 else -1
        pow_neg1_c = 1 if c % 2 == 0 else -1
        pow_neg1_r_plus_1 = 1 if (r + 1) % 2 == 0 else -1
        pow_neg1_c_plus_1 = 1 if (c + 1) % 2 == 0 else -1
        pow_neg1_r_c_1 = 1 if (r + c + 1) % 2 == 0 else -1
        cand = [
            (r + pow_neg1_r, c + pow_neg1_c),
            (r - 1, c - pow_neg1_r_c_1),
            (r + 1, c + pow_neg1_r_c_1),
            (r + pow_neg1_r_plus_1, c),
            (r, c + pow_neg1_c_plus_1),
        ]
        return [(rr, cc) for rr, cc in cand if 0 <= rr < game.LOGICAL_GRID_SIZE and 0 <= cc < game.LOGICAL_GRID_SIZE]

    for r in range(game.LOGICAL_GRID_SIZE):
        for c in range(game.LOGICAL_GRID_SIZE):
            color = game.board[r][c]
            if color in (TileState.RED, TileState.BLUE) and not visited[r][c]:
                stack = [(r, c)]
                size = 0
                while stack:
                    rr, cc = stack.pop()
                    if not (0 <= rr < game.LOGICAL_GRID_SIZE and 0 <= cc < game.LOGICAL_GRID_SIZE):
                        continue
                    if visited[rr][cc] or game.board[rr][cc] != color:
                        continue
                    visited[rr][cc] = True
                    size += 1
                    for nr, nc in neighbors(rr, cc):
                        if not visited[nr][nc] and game.board[nr][nc] == color:
                            stack.append((nr, nc))
                if size:
                    sizes[color].append(size)

    def fmt(arr):
        if not arr:
            return '–', 1
        arr2 = sorted(arr, reverse=True)
        prod = 1
        for x in arr2:
            prod *= x
        return '×'.join(str(x) for x in arr2), prod

    red_str, red_prod = fmt(sizes[TileState.RED])
    blue_str, blue_prod = fmt(sizes[TileState.BLUE])
    return f"Islands — Red: [{red_str}] => {red_prod}; Blue: [{blue_str}] => {blue_prod}"


def build_prompt(env: TessellateTaEnv, game: TessellateGame, valid_actions: List[int]) -> str:
    # Borrow Env rendering to keep visuals consistent
    env.game = game
    board_str = env._render_board(full=False)
    turn_str = "RED" if game.current_turn == TileState.RED else "BLUE"
    moves_visual = valid_actions_visual(valid_actions, max_items=50)
    instructions = (
        "You are playing Tessellate. Place one triangular tile per turn.\n"
        "Respond in two labeled parts so we can show your thinking without private chain-of-thought.\n"
        "- Move: a single coordinate (prefer [A0] or visual A0-UR).\n"
        "- Reasoning: a concise high-level rationale (1-2 lines).\n"
        "Valid row letters: A-J (logical grid), or A-E with UL/UR/LL/LR for the visual squares. Choose only empty (not blocked) positions.\n"
    )
    summary = island_summary(game)
    msg = (
        f"{board_str}\n"
        f"{summary}\n"
        f"Turn: {turn_str}.\n"
        f"Valid moves (visual, up to 50): {moves_visual}\n\n"
        f"{instructions}"
    )
    return msg


def create_app(model_name: str) -> FastAPI:
    app = FastAPI(title="Tessellate OpenAI GPT-5 Agent Server")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "https://tessellate-app-ytnku.ondigitalocean.app",
            "https://mcembalest.github.io",
            "http://localhost:8424",
            "http://127.0.0.1:8424",
            "http://localhost:8080",
            "http://127.0.0.1:8080",
            "null"
        ],
        allow_origin_regex=r"https://.*\\.github\\.io$",
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # OpenAI client
    client = OpenAI()
    env = TessellateTaEnv()  # for rendering helpers only

    def _reasoning_effort() -> str:
        return os.environ.get("OPENAI_REASONING_EFFORT", "medium")

    def _text_verbosity() -> str | None:
        return os.environ.get("OPENAI_TEXT_VERBOSITY")

    @app.post("/move", response_model=MoveResponse)
    def move(req: MoveRequest):
        if not req.valid_actions:
            raise HTTPException(status_code=400, detail="No valid actions provided")

        try:
            game = state_to_game(req.state)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Bad state: {e}")

        prompt = build_prompt(env, game, req.valid_actions)

        kwargs: dict = {
            "model": model_name,
            "input": prompt,
            "reasoning": {"effort": _reasoning_effort()},
        }
        verbosity = _text_verbosity()
        if verbosity:
            kwargs["text"] = {"verbosity": verbosity}

        try:
            resp = client.responses.create(**kwargs)  # type: ignore[arg-type]
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"OpenAI error: {type(e).__name__}: {e}")

        # Best-effort to extract full text
        explanation = None
        try:
            explanation = getattr(resp, "output_text", None) or str(resp)
        except Exception:
            explanation = None

        # Parse the agent's choice
        r, c = parse_llm_move(explanation or "")
        action = None
        if r is not None and c is not None:
            action = r * 10 + c
            if action not in req.valid_actions:
                action = None

        if action is None:
            raise HTTPException(status_code=422, detail="Could not parse a valid move from model output.")

        return MoveResponse(action=action, explanation=(explanation or "").strip())

    @app.post("/move_stream")
    def move_stream(req: MoveRequest):
        """Stream the model's response via SSE while also returning a final action at the end.

        Event types:
          data: {"type":"content","delta":"..."}
          data: {"type":"reasoning","delta":"..."}
          data: {"type":"final","action":<int>,"full":"..."}
          data: {"type":"error","message":"..."}
        """
        if not req.valid_actions:
            return PlainTextResponse("No valid actions provided", status_code=400)

        try:
            game = state_to_game(req.state)
        except Exception as e:
            return PlainTextResponse(f"Bad state: {e}", status_code=400)

        base_prompt = build_prompt(env, game, req.valid_actions)

        # Stream with up to 2 retries if the model returns an invalid move format
        max_attempts = 2

        def event_stream() -> Generator[str, None, None]:
            import json as _json

            attempt = 0
            current_input = base_prompt

            while attempt <= max_attempts:
                full_reasoning_parts: list[str] = []
                full_content_parts: list[str] = []

                stream_kwargs: dict = {
                    "model": model_name,
                    "input": current_input,
                    "reasoning": {"effort": os.environ.get("OPENAI_REASONING_EFFORT", "medium")},
                }
                verbosity = os.environ.get("OPENAI_TEXT_VERBOSITY")
                if verbosity:
                    stream_kwargs["text"] = {"verbosity": verbosity}

                try:
                    with client.responses.stream(**stream_kwargs) as stream:  # type: ignore[arg-type]
                        for event in stream:
                            t = getattr(event, "type", None)
                            if t == "response.reasoning.delta":
                                delta = getattr(event, "delta", "")
                                s = str(delta)
                                if s:
                                    full_reasoning_parts.append(s)
                                    yield f"data: {_json.dumps({"type": "reasoning", "delta": s})}\n\n"
                            elif t == "response.output_text.delta":
                                delta = getattr(event, "delta", "")
                                s = str(delta)
                                if s:
                                    full_content_parts.append(s)
                                    yield f"data: {_json.dumps({"type": "content", "delta": s})}\n\n"
                            elif t == "response.error":
                                message = getattr(event, "error", None)
                                yield f"data: {_json.dumps({"type": "error", "message": str(message)})}\n\n"
                            elif t == "response.completed":
                                # Finalize
                                try:
                                    final_resp = stream.get_final_response()
                                    final_text = getattr(final_resp, "output_text", None) or ("".join(full_reasoning_parts) + "\n" + "".join(full_content_parts))
                                except Exception:
                                    final_text = ("".join(full_reasoning_parts) + "\n" + "".join(full_content_parts))

                                r_c = parse_llm_move(final_text)
                                action = None
                                if r_c[0] is not None and r_c[1] is not None:
                                    action = r_c[0]*10 + r_c[1]
                                    if action not in req.valid_actions:
                                        action = None
                                if action is None:
                                    attempt += 1
                                    if attempt > max_attempts:
                                        yield f"data: {_json.dumps({"type": "error", "message": "Could not parse a valid move from model output after retries."})}\n\n"
                                        return
                                    available = ", ".join(str(a) for a in req.valid_actions[:100])
                                    corrective = (
                                        "Your previous reply did not specify a valid coordinate.\n"
                                        "Only answer with ONE coordinate using either [A0] or A0-UR syntax, and it must map to one of these valid flat IDs: "
                                        f"{available}.\n"
                                        "Do not include explanations before the coordinate; you may include a short Reasoning after the Move line.\n"
                                        "Format strictly as:\nMove: [A0]\nReasoning: <one line>\n"
                                    )
                                    # Retry with corrective appended
                                    current_input = base_prompt + "\n\n" + corrective
                                    break  # break inner for; the while loop will start next attempt
                                else:
                                    final_event = {"type": "final", "action": action, "full": final_text}
                                    yield f"data: {_json.dumps(final_event)}\n\n"
                                    return
                except Exception as e:
                    yield f"data: {_json.dumps({"type": "error", "message": f"{type(e).__name__}: {e}"})}\n\n"
                    return

            # exhausted attempts without a final action
            yield f"data: {{\"type\": \"error\", \"message\": \"Stream ended without a final action.\"}}\n\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get("/health")
    def health():
        return {"ok": True, "model": model_name}

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.environ.get("TESSELLATE_AGENT_MODEL", "gpt-5-nano"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()

    app = create_app(model_name=args.model)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

