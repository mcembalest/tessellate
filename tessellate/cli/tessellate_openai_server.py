#!/usr/bin/env python3
"""
Run an HTTP server that lets the browser play Tessellate against
an LLM using OpenAI's Responses API (GPT-5 family).

"""

from __future__ import annotations

import argparse
import os
import re
from typing import Dict, List, Tuple, Generator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, PlainTextResponse
from pydantic import BaseModel, Field
import uvicorn

from openai import OpenAI
from openai import APIStatusError, APIConnectionError, AuthenticationError, BadRequestError, NotFoundError, RateLimitError
import logging

from tessellate.game import TessellateGame, TileState
from tessellate.environments.tessellate_textarena_env import TessellateTaEnv, VIS_ROWS


# ---------- Constants ----------

# API Configuration
DEFAULT_MODEL = "gpt-5-mini"
DEFAULT_REASONING_EFFORT = "medium"
DEFAULT_MAX_OUTPUT_TOKENS = 20000
DEFAULT_TEXT_VERBOSITY = None

# Game Constants
STATE_VECTOR_LENGTH = 104
LOGICAL_GRID_SIZE = 10
VISUAL_GRID_SIZE = 5
MAX_RETRY_ATTEMPTS = 2

# CORS Origins
CORS_ORIGINS = [
    "https://tessellate-app-ytnku.ondigitalocean.app",
    "https://mcembalest.github.io",
    "http://localhost:8424",
    "http://127.0.0.1:8424",
    "http://localhost:8080",
    "http://127.0.0.1:8080",
    "null"
]

# Coordinate Mappings
CORNER_MAP = {
    (0, 0): 'UL', (0, 1): 'UR',
    (1, 0): 'LL', (1, 1): 'LR'
}

TILE_STATE_MAP = {
    0: TileState.EMPTY,
    1: TileState.RED,
    2: TileState.BLUE,
    3: TileState.BLOCKED
}

# Regex Patterns
VISUAL_SQUARE_PATTERN = r"\b\[?([A-Ea-e])\s*([0-4])\s*[-, ]\s*(UL|UR|LL|LR)\]?\b"
LOGICAL_CELL_PATTERN = r"\b\[?([A-Ja-j])\s*([0-9])\]?\b"
RAW_TUPLE_PATTERN = r"\[\s*(\d)\s*,\s*(\d)\s*\]"
TUPLE_VISUAL_PATTERN = r"\(\s*(\d)\s*,\s*(\d)\s*,\s*(UL|UR|LL|LR)\s*\)"

# Error Messages
ERROR_NO_VALID_ACTIONS = "No valid actions provided"
ERROR_BAD_STATE = "Bad state: {}"
ERROR_OPENAI_ERROR = "OpenAI error: {}: {}"
ERROR_PARSE_MOVE = "Could not parse a valid move from model output."
ERROR_PARSE_MOVE_RETRIES = "Could not parse a valid move from model output after retries."
ERROR_STREAM_ENDED = "Stream ended without a final action."

# Environment Variables
ENV_OPENAI_API_KEY = "OPENAI_API_KEY"
ENV_MODEL = "TESSELLATE_AGENT_MODEL"
ENV_REASONING_EFFORT = "OPENAI_REASONING_EFFORT"
ENV_TEXT_VERBOSITY = "OPENAI_TEXT_VERBOSITY"
ENV_MAX_OUTPUT_TOKENS = "OPENAI_MAX_OUTPUT_TOKENS"

# ---------- Core Game Logic ----------

def get_openai_config() -> dict:
    """Get OpenAI API configuration from environment variables."""
    return {
        "reasoning_effort": os.environ.get(ENV_REASONING_EFFORT, DEFAULT_REASONING_EFFORT),
        "max_output_tokens": int(os.environ.get(ENV_MAX_OUTPUT_TOKENS, DEFAULT_MAX_OUTPUT_TOKENS)),
        "text_verbosity": os.environ.get(ENV_TEXT_VERBOSITY),
    }


def build_openai_request_kwargs(model_name: str, prompt: str, config: dict) -> dict:
    """Build kwargs for OpenAI API request."""
    kwargs = {
        "model": model_name,
        "input": prompt,
        "reasoning": {"effort": config["reasoning_effort"]},
        "max_output_tokens": config["max_output_tokens"],
    }

    # Only add text verbosity if it exists and is not None
    if config.get("text_verbosity") is not None:
        kwargs["text"] = {"verbosity": config["text_verbosity"]}

    return kwargs


def handle_openai_error(e: Exception, operation: str, log: logging.Logger) -> None:
    """Handle OpenAI API errors with consistent logging and HTTP exceptions."""
    status = getattr(e, "status_code", None) or getattr(getattr(e, "response", None), "status_code", None) or 502
    log.exception(f"OpenAI {operation} failed: %s", e)

    # Handle specific error types
    if isinstance(e, (AuthenticationError, BadRequestError, NotFoundError, RateLimitError, APIConnectionError, APIStatusError)):
        # Extract error message for better user experience
        error_msg = str(e)
        if hasattr(e, 'body') and e.body:
            try:
                import json
                body = json.loads(e.body)
                if 'error' in body and 'message' in body['error']:
                    error_msg = body['error']['message']
            except:
                pass
        elif hasattr(e, 'response') and hasattr(e.response, 'text'):
            try:
                import json
                body = json.loads(e.response.text)
                if 'error' in body and 'message' in body['error']:
                    error_msg = body['error']['message']
            except:
                pass

        raise HTTPException(status_code=int(status), detail=f"OpenAI error: {error_msg}")
    else:
        raise HTTPException(status_code=502, detail=f"OpenAI error: {type(e).__name__}: {e}")


def extract_response_text(response) -> str | None:
    """Extract text content from OpenAI response object."""
    try:
        return getattr(response, "output_text", None) or str(response)
    except Exception:
        return None


def extract_final_text(stream, reasoning_parts: list[str], content_parts: list[str]) -> str:
    """Extract final text from streaming response."""
    try:
        final_resp = stream.get_final_response()
        return getattr(final_resp, "output_text", None) or ("".join(reasoning_parts) + "\n" + "".join(content_parts))
    except Exception:
        return "".join(reasoning_parts) + "\n" + "".join(content_parts)


def parse_and_validate_move(text: str, valid_actions: List[int]) -> int | None:
    """Parse move from text and validate it's in valid actions."""
    r, c = parse_llm_move(text)
    if r is not None and c is not None:
        action = r * LOGICAL_GRID_SIZE + c
        if action in valid_actions:
            return action
    return None


def build_corrective_prompt(base_prompt: str, valid_actions: List[int]) -> str:
    """Build a corrective prompt for retrying invalid moves."""
    available = ", ".join(str(a) for a in valid_actions[:100])
    corrective = (
        "Your previous reply did not specify a valid coordinate.\n"
        "Only answer with ONE coordinate using either [A0] or A0-UR syntax, and it must map to one of these valid flat IDs: "
        f"{available}.\n"
        "Do not include explanations before the coordinate; you may include a short Reasoning after the Move line.\n"
        "Format strictly as:\nMove: [A0]\nReasoning: <one line>\n"
    )
    return base_prompt + "\n\n" + corrective


def _fallback_to_non_streaming(req: MoveRequest, env, client, log):
    """Fallback to non-streaming mode when streaming fails."""
    import json as _json

    def non_streaming_generator():
        try:
            game = state_to_game(req.state)
        except Exception as e:
            yield f"data: {_json.dumps({'type': 'error', 'message': ERROR_BAD_STATE.format(e)})}\n\n"
            return

        prompt = build_prompt(env, game, req.valid_actions)
        config = get_openai_config()
        # Use the model name from the outer scope (passed to create_app)
        kwargs = build_openai_request_kwargs("gpt-5-mini", prompt, config)  # Default fallback

        try:
            resp = client.responses.create(**kwargs)
            explanation = extract_response_text(resp)

            r, c = parse_llm_move(explanation or "")
            action = None
            if r is not None and c is not None:
                action = r * LOGICAL_GRID_SIZE + c
                if action not in req.valid_actions:
                    action = None

            if action is None:
                yield f"data: {_json.dumps({'type': 'error', 'message': ERROR_PARSE_MOVE})}\n\n"
            else:
                final_event = {"type": "final", "action": action, "full": explanation or ""}
                yield f"data: {_json.dumps(final_event)}\n\n"

        except Exception as e:
            handle_openai_error(e, "/move_stream_fallback", log)
            yield f"data: {_json.dumps({'type': 'error', 'message': f"Fallback failed: {type(e).__name__}: {e}"})}\n\n"

    return StreamingResponse(
        non_streaming_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ---------- Coordinate Conversion Utilities ----------

def corner_string_to_tuple(corner: str) -> Tuple[int, int]:
    """Convert corner string (UL, UR, LL, LR) to tuple coordinates."""
    corner_map_reverse = {v: k for k, v in CORNER_MAP.items()}
    return corner_map_reverse[corner.upper()]


def flat_to_rc(action_flat: int) -> Tuple[int, int]:
    """Convert flat action index to row/column coordinates."""
    return action_flat // LOGICAL_GRID_SIZE, action_flat % LOGICAL_GRID_SIZE


def rc_to_visual(r: int, c: int) -> str:
    """Convert row/column coordinates to visual square notation (e.g., A0-UL)."""
    sr, sc = r // 2, c // 2
    cr, cc = r % 2, c % 2
    corner = CORNER_MAP[(cr, cc)]
    return f"{VIS_ROWS[sr]}{sc}-{corner}"


def valid_actions_visual(valid_actions: List[int], max_items: int = 50) -> str:
    """Format valid actions as human-readable visual coordinates."""
    items = []
    for a in valid_actions[:max_items]:
        r, c = flat_to_rc(a)
        items.append(rc_to_visual(r, c))
    if len(valid_actions) > max_items:
        items.append("...")
    return ", ".join(items)


# ---------- LLM Response Parsing ----------

def _parse_visual_square(match) -> Tuple[int, int]:
    """Parse visual square format like [A0-UL] or A0-UL."""
    sr = ord(match.group(1).upper()) - ord('A')
    sc = int(match.group(2))
    cr, cc = corner_string_to_tuple(match.group(3))
    return 2*sr + cr, 2*sc + cc


def _parse_logical_cell(match) -> Tuple[int, int]:
    """Parse logical cell format like [A0] or A0."""
    r = ord(match.group(1).upper()) - ord('A')
    c = int(match.group(2))
    return r, c


def _parse_raw_tuple(match) -> Tuple[int, int]:
    """Parse raw tuple format like [r, c]."""
    return int(match.group(1)), int(match.group(2))


def _parse_tuple_visual(match) -> Tuple[int, int] | None:
    """Parse tuple visual format like (square_r, square_c, CORNER)."""
    sr, sc = int(match.group(1)), int(match.group(2))
    if 0 <= sr < VISUAL_GRID_SIZE and 0 <= sc < VISUAL_GRID_SIZE:
        cr, cc = corner_string_to_tuple(match.group(3))
        return 2*sr + cr, 2*sc + cc
    return None


def parse_llm_move(text: str) -> Tuple[int | None, int | None]:
    """Parse a move from LLM text. Returns (r, c) or (None, None).

    Accepts multiple formats, matching the TextArena env:
      - Visual square: [A0-UL] or A0-UL
      - Logical cell:  [A0] or A0   (A-J rows, 0-9 cols)
      - Raw tuple:     [r, c]
      - Tuple visual:  (square_r, square_c, CORNER)
    """
    # Define parsers in order of preference
    parsers = [
        (VISUAL_SQUARE_PATTERN, _parse_visual_square),
        (LOGICAL_CELL_PATTERN, _parse_logical_cell),
        (RAW_TUPLE_PATTERN, _parse_raw_tuple),
        (TUPLE_VISUAL_PATTERN, _parse_tuple_visual),
    ]

    for pattern, parser in parsers:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                result = parser(match)
                if result is not None:
                    return result
            except (ValueError, IndexError):
                continue

    return None, None


# ---------- Game State Management ----------

def state_to_game(state: List[int]) -> TessellateGame:
    """Convert state vector back to TessellateGame object."""
    if len(state) != STATE_VECTOR_LENGTH:
        raise ValueError(f"state must be length {STATE_VECTOR_LENGTH}")

    game = TessellateGame()

    # Rebuild board from first 100 elements
    for idx in range(100):
        r, c = divmod(idx, LOGICAL_GRID_SIZE)
        tile_value = state[idx]
        game.board[r][c] = TILE_STATE_MAP.get(tile_value, TileState.EMPTY)

    # Set current turn
    turn_v = state[100]
    game.current_turn = TileState.RED if turn_v == 1 else TileState.BLUE

    # Set scores (best-effort; browser recomputes anyway)
    red_score = state[101] if isinstance(state[101], (int, float)) and state[101] else 1
    blue_score = state[102] if isinstance(state[102], (int, float)) and state[102] else 1
    game.scores = {TileState.RED: int(red_score), TileState.BLUE: int(blue_score)}

    # Set tile placement count and game over status
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

# ---------- Island Analysis ----------

def get_neighbors(r: int, c: int) -> List[Tuple[int, int]]:
    """Get valid neighboring positions for tessellate game logic."""
    pow_neg1_r = 1 if r % 2 == 0 else -1
    pow_neg1_c = 1 if c % 2 == 0 else -1
    pow_neg1_r_plus_1 = 1 if (r + 1) % 2 == 0 else -1
    pow_neg1_c_plus_1 = 1 if (c + 1) % 2 == 0 else -1
    pow_neg1_r_c_1 = 1 if (r + c + 1) % 2 == 0 else -1

    candidates = [
        (r + pow_neg1_r, c + pow_neg1_c),
        (r - 1, c - pow_neg1_r_c_1),
        (r + 1, c + pow_neg1_r_c_1),
        (r + pow_neg1_r_plus_1, c),
        (r, c + pow_neg1_c_plus_1),
    ]

    return [(rr, cc) for rr, cc in candidates
            if 0 <= rr < LOGICAL_GRID_SIZE and 0 <= cc < LOGICAL_GRID_SIZE]


def find_connected_component(game: TessellateGame, start_r: int, start_c: int, visited: List[List[bool]]) -> int:
    """Find size of connected component starting from given position."""
    color = game.board[start_r][start_c]
    stack = [(start_r, start_c)]
    size = 0

    while stack:
        r, c = stack.pop()
        if visited[r][c] or game.board[r][c] != color:
            continue

        visited[r][c] = True
        size += 1

        for nr, nc in get_neighbors(r, c):
            if not visited[nr][nc] and game.board[nr][nc] == color:
                stack.append((nr, nc))

    return size


def get_island_sizes(game: TessellateGame) -> Dict[TileState, List[int]]:
    """Calculate island sizes for both colors."""
    sizes = {TileState.RED: [], TileState.BLUE: []}
    visited = [[False for _ in range(LOGICAL_GRID_SIZE)] for _ in range(LOGICAL_GRID_SIZE)]

    for r in range(LOGICAL_GRID_SIZE):
        for c in range(LOGICAL_GRID_SIZE):
            color = game.board[r][c]
            if color in (TileState.RED, TileState.BLUE) and not visited[r][c]:
                island_size = find_connected_component(game, r, c, visited)
                if island_size > 0:
                    sizes[color].append(island_size)

    return sizes


def format_island_sizes(island_sizes: List[int]) -> Tuple[str, int]:
    """Format island sizes as string and calculate product score."""
    if not island_sizes:
        return '–', 1

    sorted_sizes = sorted(island_sizes, reverse=True)
    product = 1
    for size in sorted_sizes:
        product *= size

    return '×'.join(str(size) for size in sorted_sizes), product


def island_summary(game: TessellateGame) -> str:
    """Generate summary of island sizes and scores for both players."""
    sizes = get_island_sizes(game)

    red_str, red_score = format_island_sizes(sizes[TileState.RED])
    blue_str, blue_score = format_island_sizes(sizes[TileState.BLUE])

    return f"Islands — Red: [{red_str}] => {red_score}; Blue: [{blue_str}] => {blue_score}"


# ---------- Prompt Building ----------

def build_prompt(env: TessellateTaEnv, game: TessellateGame, valid_actions: List[int]) -> str:
    """Build the prompt for the LLM to make a move."""
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
    prompt = (
        f"{board_str}\n"
        f"{summary}\n"
        f"Turn: {turn_str}.\n"
        f"Valid moves (visual, up to 50): {moves_visual}\n\n"
        f"{instructions}"
    )
    return prompt


def create_app(model_name: str) -> FastAPI:
    # Be tolerant of provider-prefixed model names like "openai/gpt-5-mini"
    try:
        if "/" in model_name:
            provider, name = model_name.split("/", 1)
            if provider.lower() == "openai" and name:
                model_name = name
    except Exception:
        pass

    app = FastAPI(title="Tessellate OpenAI GPT-5 Agent Server")

    # Log model information for debugging
    print(f"Initializing server with model: {model_name}")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_origin_regex=r"https://.*\\.github\\.io$",
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # OpenAI client
    client = OpenAI()
    log = logging.getLogger("tessellate.openai")
    if not log.handlers:
        logging.basicConfig(level=logging.INFO)
    env = TessellateTaEnv()  # for rendering helpers only



    @app.post("/move", response_model=MoveResponse)
    def move(req: MoveRequest):
        if not req.valid_actions:
            raise HTTPException(status_code=400, detail=ERROR_NO_VALID_ACTIONS)

        try:
            game = state_to_game(req.state)
        except Exception as e:
            raise HTTPException(status_code=400, detail=ERROR_BAD_STATE.format(e))

        prompt = build_prompt(env, game, req.valid_actions)
        config = get_openai_config()
        kwargs = build_openai_request_kwargs(model_name, prompt, config)

        try:
            resp = client.responses.create(**kwargs)  # type: ignore[arg-type]
        except Exception as e:
            handle_openai_error(e, "/move", log)

        explanation = extract_response_text(resp)

        # Parse the agent's choice
        r, c = parse_llm_move(explanation or "")
        action = None
        if r is not None and c is not None:
            action = r * LOGICAL_GRID_SIZE + c
            if action not in req.valid_actions:
                action = None

        if action is None:
            raise HTTPException(status_code=422, detail=ERROR_PARSE_MOVE)

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
            return PlainTextResponse(ERROR_NO_VALID_ACTIONS, status_code=400)

        try:
            game = state_to_game(req.state)
        except Exception as e:
            return PlainTextResponse(ERROR_BAD_STATE.format(e), status_code=400)

        base_prompt = build_prompt(env, game, req.valid_actions)
        config = get_openai_config()

        def event_stream() -> Generator[str, None, None]:
            import json as _json

            attempt = 0
            current_input = base_prompt

            while attempt <= MAX_RETRY_ATTEMPTS:
                full_reasoning_parts: list[str] = []
                full_content_parts: list[str] = []

                stream_kwargs = build_openai_request_kwargs(model_name, current_input, config)

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
                                final_text = extract_final_text(stream, full_reasoning_parts, full_content_parts)
                                action = parse_and_validate_move(final_text, req.valid_actions)

                                if action is None:
                                    attempt += 1
                                    if attempt > MAX_RETRY_ATTEMPTS:
                                        yield f"data: {_json.dumps({"type": "error", "message": ERROR_PARSE_MOVE_RETRIES})}\n\n"
                                        return
                                    current_input = build_corrective_prompt(base_prompt, req.valid_actions)
                                    break  # break inner for; the while loop will start next attempt
                                else:
                                    final_event = {"type": "final", "action": action, "full": final_text}
                                    yield f"data: {_json.dumps(final_event)}\n\n"
                                    return
                except Exception as e:
                    handle_openai_error(e, "/move_stream", log)
                    yield f"data: {_json.dumps({"type": "error", "message": f"{type(e).__name__}: {e}"})}\n\n"
                    return

            # exhausted attempts without a final action
            yield f"data: {{\"type\": \"error\", \"message\": \"{ERROR_STREAM_ENDED}\"}}\n\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
                # In some PaaS/CDN setups (e.g., HTTP/3 over QUIC), SSE can break.
                # Hint browsers to clear any HTTP/3 Alt-Svc for this origin to prefer H2.
                "Alt-Svc": "clear",
            },
        )

    @app.get("/health")
    def health():
        has_key = bool(os.environ.get("OPENAI_API_KEY"))
        return {"ok": True, "model": model_name, "openai_key": has_key}

    return app


def main():
    """Main entry point for the Tessellate OpenAI server."""
    parser = argparse.ArgumentParser(description="Run Tessellate OpenAI GPT-5 Agent Server")
    parser.add_argument(
        "--model",
        default=os.environ.get(ENV_MODEL, DEFAULT_MODEL),
        help=f"OpenAI model name (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind server to"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port to bind server to"
    )

    args = parser.parse_args()

    print(f"Starting Tessellate OpenAI server with model: {args.model}")
    print(f"Server will be available at http://{args.host}:{args.port}")

    app = create_app(model_name=args.model)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
