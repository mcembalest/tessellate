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

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, PlainTextResponse
from pydantic import BaseModel, Field
import uvicorn
import requests
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
        "Respond in two labeled lines so we can display your thinking without private chain-of-thought.\n"
        "- Move: a single coordinate (prefer [A0] or visual A0-UR).\n"
        "- Reasoning: a concise high-level rationale (1-2 lines).\n"
        "Valid row letters: A-J (logical grid), or A-E with UL/UR/LL/LR for the visual squares. Choose only empty (not blocked) positions.\n"
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
            "https://mcembalest.github.io",
            "https://maxcembalest.github.io",
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

        # Return full text so the browser can show richer reasoning
        explanation = str(raw).strip()

        return MoveResponse(action=action, explanation=explanation)

    @app.post("/move_stream")
    def move_stream(req: MoveRequest):
        """Stream the model's response via SSE while also returning a final action at the end.

        Event types:
          data: {"type":"content","delta":"..."}
          data: {"type":"final","action":<int>,"full":"..."}
          data: {"type":"error","message":"..."}
        """
        if not req.valid_actions:
            return PlainTextResponse("No valid actions provided", status_code=400)

        # Rebuild game for prompt and parsing
        try:
            game = state_to_game(req.state)
        except Exception as e:
            return PlainTextResponse(f"Bad state: {e}", status_code=400)

        prompt = build_prompt(env, game, req.valid_actions)

        # Prepare OpenRouter streaming request
        api_key = os.environ.get("OPENROUTER_API_KEY")
        model = model_name
        if not api_key:
            return PlainTextResponse("OPENROUTER_API_KEY not set", status_code=500)

        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        # Optional OpenRouter headers for project keys / referral tracking
        http_referer = os.environ.get("OPENROUTER_HTTP_REFERER")
        if http_referer:
            headers["HTTP-Referer"] = http_referer
        x_title = os.environ.get("OPENROUTER_X_TITLE")
        if x_title:
            headers["X-Title"] = x_title
        # Reasoning tokens configuration (env-controlled)
        reasoning_cfg: dict | None = {}
        effort = os.environ.get("OPENROUTER_REASONING_EFFORT")  # e.g., high|medium|low
        max_tok = os.environ.get("OPENROUTER_REASONING_MAX_TOKENS")  # integer string
        exclude = os.environ.get("OPENROUTER_REASONING_EXCLUDE")  # "true" to exclude
        enabled = os.environ.get("OPENROUTER_REASONING_ENABLED")  # "true" to force enable
        try:
            if max_tok is not None:
                reasoning_cfg["max_tokens"] = int(max_tok)
            if effort:
                reasoning_cfg["effort"] = effort
            if exclude is not None:
                reasoning_cfg["exclude"] = (str(exclude).lower() == "true")
            if enabled is not None:
                reasoning_cfg["enabled"] = (str(enabled).lower() == "true")
        except Exception:
            reasoning_cfg = {}
        if not reasoning_cfg:
            # Default to enabling reasoning at medium effort if not otherwise specified
            reasoning_cfg = {"effort": "medium"}

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            "reasoning": reasoning_cfg,
        }

        # Stream with up to 2 retries if the model returns an invalid move format
        final_sent: bool = False
        max_attempts = 2

        def event_stream():
            nonlocal final_sent
            import json as _json
            attempt = 0
            current_messages = list(payload["messages"])  # type: ignore
            while attempt <= max_attempts:
                full_content: list[str] = []
                full_reasoning: list[str] = []
                try:
                    current_payload = dict(payload)
                    current_payload["messages"] = list(current_messages)
                    with requests.post(url, headers=headers, json=current_payload, stream=True, timeout=None) as r:
                        r.raise_for_status()
                        buffer = ""
                        for chunk in r.iter_content(chunk_size=1024, decode_unicode=True):
                            if not chunk:
                                continue
                            buffer += chunk
                            while True:
                                line_end = buffer.find("\n")
                                if line_end == -1:
                                    break
                                line = buffer[:line_end].strip()
                                buffer = buffer[line_end + 1:]
                                if not line:
                                    continue
                                if line.startswith(":"):
                                    continue
                                if not line.startswith("data: "):
                                    continue
                                data = line[6:]
                                if data == "[DONE]":
                                    final_text = ("".join(full_reasoning) + "\n" + "".join(full_content)).strip()
                                    r_c = parse_llm_move(final_text)
                                    action = None
                                    if r_c[0] is not None and r_c[1] is not None:
                                        action = r_c[0]*10 + r_c[1]
                                        if action not in req.valid_actions:
                                            action = None
                                    if action is None:
                                        attempt += 1
                                        if attempt > max_attempts:
                                            yield f"data: {_json.dumps({"type": "error", "message": "Could not parse a valid move from agent output after retries."})}\n\n"
                                            return
                                        # append a corrective instruction and retry
                                        available = ", ".join(str(a) for a in req.valid_actions[:100])
                                        corrective = (
                                            "Your previous reply did not specify a valid coordinate.\n"
                                            "Only answer with ONE coordinate using either [A0] or A0-UR syntax, and it must map to one of these valid flat IDs: "
                                            f"{available}.\n"
                                            "Do not include explanations before the coordinate; you may include a short Reasoning after the Move line.\n"
                                            "Format strictly as:\nMove: [A0]\nReasoning: <one line>\n"
                                        )
                                        yield f"data: {_json.dumps({"type": "info", "message": "Invalid move format. Retrying..."})}\n\n"
                                        current_messages = list(current_messages) + [{"role": "user", "content": corrective}]
                                        break  # break inner while to start next attempt
                                    else:
                                        final_event = {"type": "final", "action": action, "full": final_text}
                                        yield f"data: {_json.dumps(final_event)}\n\n"
                                        final_sent = True
                                        return
                                try:
                                    obj = _json.loads(data)
                                    delta = obj.get("choices", [{}])[0].get("delta", {})
                                    reasoning_delta = delta.get("reasoning")
                                    if reasoning_delta:
                                        if isinstance(reasoning_delta, dict):
                                            reasoning_str = reasoning_delta.get("text") or reasoning_delta.get("content") or _json.dumps(reasoning_delta)
                                        else:
                                            reasoning_str = str(reasoning_delta)
                                        full_reasoning.append(reasoning_str)
                                        yield f"data: {_json.dumps({"type": "reasoning", "delta": reasoning_str})}\n\n"
                                    content = delta.get("content")
                                    if content:
                                        full_content.append(content)
                                        yield f"data: {_json.dumps({"type": "content", "delta": content})}\n\n"
                                except Exception:
                                    pass
                except Exception as e:
                    yield f"data: {_json.dumps({"type": "error", "message": f"{type(e).__name__}: {e}"})}\n\n"
                    return
            # exhausted attempts
            yield f"data: {_json.dumps({"type": "error", "message": "Stream ended without a final action."})}\n\n"

        def with_final_action():
            for ev in event_stream():
                yield ev
            # nothing else to do; event_stream handles retries and final
            return

        return StreamingResponse(
            with_final_action(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                # Disable buffering on proxies like Nginx used by some PaaS
                "X-Accel-Buffering": "no",
            },
        )

    @app.get("/health")
    def health():
        return {"ok": True, "model": model_name}

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.environ.get("TESSELLATE_AGENT_MODEL", "anthropic/claude-3.7-sonnet"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()

    app = create_app(model_name=args.model)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
