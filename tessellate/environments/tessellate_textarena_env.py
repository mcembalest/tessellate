"""
TextArena-compatible environment for the Tessellate game.

This wraps the core TessellateGame (see tessellate.py) into a textarena.Env
so LLM agents can play the game via natural language actions.

Action format: a single coordinate identifying where to place a tile.
Accepted formats (case-insensitive):
  - [A0]  (row letter A-J, column 0-9 inside square brackets)
  - A0    (same without brackets)

On each valid move, the environment updates the board, recomputes scores,
switches the turn, and provides both players with an updated board render
and status message.

Rewards are set at terminal: +1 winner, -1 loser, 0 for draw.
Invalid actions are handled by the TwoPlayerState error allowance mechanism.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import textarena as ta

from tessellate.game import TessellateGame, TileState


ROW_LETTERS = [chr(ord("A") + i) for i in range(10)]
VIS_ROWS = [chr(ord("A") + i) for i in range(5)]


def _pos_to_str(r: int, c: int) -> str:
    return f"{ROW_LETTERS[r]}{c}"


class TessellateTaEnv(ta.Env):
    """Two-player TextArena environment for Tessellate."""

    def __init__(self, error_allowance: int = 4):
        # Core game implementation
        self.game = TessellateGame()

        # Cached things to render quickly
        self._render_cache_full: str | None = None
        # Allow a couple of invalid attempts before auto-termination (useful for LLMs)
        self._error_allowance = error_allowance

    @property
    def terminal_render_keys(self) -> List[str]:
        return ["rendered_board"]

    def reset(self, num_players: int, seed: Optional[int] = None):
        # Initialize TextArena two-player state
        self.state = ta.TwoPlayerState(num_players=num_players, seed=seed)
        # update allowance after standard reset
        self.state.error_allowance = self._error_allowance

        # Reset core game
        self.game.reset()
        # Ensure RED starts and is mapped to player 0
        self._ensure_game_player_alignment(player_id=0)

        # Precompute render
        rendered_board = self._render_board(full=True)
        game_state = {
            "rendered_board": rendered_board,
            "scores": dict(self.game.scores),
            "turn": int(self.game.current_turn),
        }

        # Initialize prompts for both players
        self.state.reset(
            game_state=game_state,
            player_prompt_function=self._generate_player_prompt,
            role_mapping={0: "Player RED", 1: "Player BLUE"},
        )

        # Send initial observation for current player
        self._observe_current_state()

    def _generate_player_prompt(self, player_id: int, game_state: Dict[str, Any]):
        other = 1 - player_id
        return (
            f"[GAME] You are Player {player_id} in Tessellate.\n"
            "You will alternate turns placing a triangular tile on a 10x10 logical grid.\n"
            "When you place a tile, the two corners adjacent within the same square become blocked (marked x).\n"
            "Your score equals the product of the sizes of your connected groups (islands).\n"
            "Play ends after 50 total placements. The higher score wins.\n\n"
            "How to move:\n"
            "- Choose any empty cell shown as '.' (not blocked 'x').\n"
            "- Provide a single coordinate using row letter A-J and column 0-9.\n"
            "- Format: [A0] (brackets optional). Example: [D3]\n\n"
            f"You are Player {player_id} ({'RED' if player_id == 0 else 'BLUE'}). "
            f"Your opponent is Player {other} ({'BLUE' if player_id == 0 else 'RED'}).\n\n"
            "Current board:\n"
        )

    def _ensure_game_player_alignment(self, player_id: int):
        # player 0 -> RED, player 1 -> BLUE
        desired_turn = TileState.RED if player_id == 0 else TileState.BLUE
        if self.game.current_turn != desired_turn:
            # This only toggles if mismatched – safe since make_move also toggles
            self.game.current_turn = desired_turn

    def _observe_current_state(self):
        player_id = self.state.current_player_id
        self._ensure_game_player_alignment(player_id)

        available_moves = self.game.get_valid_moves()
        def to_visual(rc):
            r, c = rc
            sr, sc = r // 2, c // 2
            cr, cc = r % 2, c % 2
            corner = { (0,0): 'UL', (0,1): 'UR', (1,0): 'LL', (1,1): 'LR' }[(cr,cc)]
            return f"{VIS_ROWS[sr]}{sc}-{corner}"

        moves_str = ", ".join(to_visual(rc) for rc in available_moves[:50])
        if len(available_moves) > 50:
            moves_str += ", ..."

        msg = (
            f"{self._render_board(full=False)}\n"
            f"Scores: RED={self.game.scores[TileState.RED]}, BLUE={self.game.scores[TileState.BLUE]}\n"
            f"Your turn: Player {player_id} ({'RED' if player_id==0 else 'BLUE'}).\n"
            "Choose a move like [A0] (or the visual form A0-UL).\n"
            f"Valid moves (visual, {len(available_moves)} shown up to 50): {moves_str}"
        )
        self.state.add_observation(
            message=msg, observation_type=ta.ObservationType.GAME_BOARD
        )

    def _render_board(self, full: bool = True) -> str:
        """
        Render a 5x5 visual grid using Unicode right triangles.
        We use black triangles for RED and white triangles for BLUE.
        Corner mapping (UL/UR/LL/LR):
          - RED:  ◤ ◥ ◣ ◢
          - BLUE: ◸ ◹ ◺ ◿
        Empty/blocked corners render as '·'.
        Each visual square displays two characters per line (UL,UR then LL,LR).
        """
        board = self.game.board

        RED_TRI = {  # filled/black
            'UL': '◤', 'UR': '◥', 'LL': '◣', 'LR': '◢',
        }
        BLUE_TRI = {  # white
            'UL': '◸', 'UR': '◹', 'LL': '◺', 'LR': '◿',
        }

        def corner_char(r: int, c: int) -> str:
            val = board[r][c]
            corner = (
                'UL' if r % 2 == 0 and c % 2 == 0 else
                'UR' if r % 2 == 0 and c % 2 == 1 else
                'LL' if r % 2 == 1 and c % 2 == 0 else
                'LR'
            )
            if val == TileState.RED:
                return RED_TRI[corner]
            if val == TileState.BLUE:
                return BLUE_TRI[corner]
            return '·'

        lines: List[str] = []
        header = "    " + "  ".join(str(i) for i in range(5))
        lines.append(header)
        for sr in range(5):
            top_cells = []
            bot_cells = []
            for sc in range(5):
                ul = corner_char(2*sr, 2*sc)
                ur = corner_char(2*sr, 2*sc+1)
                ll = corner_char(2*sr+1, 2*sc)
                lr = corner_char(2*sr+1, 2*sc+1)
                top_cells.append(f"{ul}{ur}")
                bot_cells.append(f"{ll}{lr}")
            lines.append(f"{VIS_ROWS[sr]:>2}  " + " ".join(top_cells))
            lines.append("    " + " ".join(bot_cells))
            if sr < 4:
                lines.append("")
        return "\n".join(lines)

    # SimpleRenderWrapper hook
    def get_board_str(self) -> str:
        return self._render_board(full=False)

    def step(self, action: str) -> Tuple[bool, ta.Info]:
        player_id = self.state.current_player_id
        self._ensure_game_player_alignment(player_id)

        # Record player's raw action
        self.state.add_observation(
            from_id=player_id,
            to_id=player_id,
            message=action,
            observation_type=ta.ObservationType.PLAYER_ACTION,
        )

        # Parse move: accept multiple formats
        r = c = None

        # 1) Visual square format: [A0-UL] or A0-UL (A-E rows, 0-4 cols, corner UL/UR/LL/LR)
        m_vis = re.search(r"\b\[?([A-Ea-e])\s*([0-4])\s*[-, ]\s*(UL|UR|LL|LR)\]?\b", action)
        if m_vis:
            sr = ord(m_vis.group(1).upper()) - ord('A')
            sc = int(m_vis.group(2))
            corner = m_vis.group(3).upper()
            cr, cc = {'UL': (0,0), 'UR': (0,1), 'LL': (1,0), 'LR': (1,1)}[corner]
            r, c = 2*sr + cr, 2*sc + cc

        # 2) Logical cell format: [A0] or A0 (A-J rows, 0-9 cols)
        if r is None:
            m_cell = re.search(r"\b\[?([A-Ja-j])\s*([0-9])\]?\b", action)
            if m_cell:
                r = ord(m_cell.group(1).upper()) - ord('A')
                c = int(m_cell.group(2))

        # 3) Othello-style: [row, col] with integers 0..9
        if r is None:
            m_rc = re.search(r"\[\s*(\d)\s*,\s*(\d)\s*\]", action)
            if m_rc:
                r, c = int(m_rc.group(1)), int(m_rc.group(2))

        # 4) Tuple visual: (square_r, square_c, CORNER)
        if r is None:
            m_tuple = re.search(r"\(\s*(\d)\s*,\s*(\d)\s*,\s*(UL|UR|LL|LR)\s*\)", action, re.IGNORECASE)
            if m_tuple:
                sr, sc = int(m_tuple.group(1)), int(m_tuple.group(2))
                if 0 <= sr < 5 and 0 <= sc < 5:
                    cr, cc = {'UL': (0,0), 'UR': (0,1), 'LL': (1,0), 'LR': (1,1)}[m_tuple.group(3).upper()]
                    r, c = 2*sr + cr, 2*sc + cc

        if r is None or c is None:
            reason = (
                f"Invalid action format. Player {player_id} must provide a move like [A0] or A0-UL."
            )
            self.state.set_invalid_move(reason=reason)
            # Surface a clear admin message to both players for debugging
            if self.state.done:
                self.state.add_observation(
                    message="Game ended due to repeated invalid actions.",
                    observation_type=ta.ObservationType.GAME_ADMIN,
                )
        else:

            # Validate against game rules
            if not (0 <= r < 10 and 0 <= c < 10):
                self.state.set_invalid_move(
                    reason=f"Coordinates out of bounds: {_pos_to_str(r, c)}"
                )
            elif not self.game.is_playable(r, c):
                self.state.set_invalid_move(
                    reason=f"Cell {_pos_to_str(r, c)} is not playable (occupied or blocked)."
                )
            else:
                # Apply move
                move_ok = self.game.make_move(r, c)
                if not move_ok:
                    self.state.set_invalid_move(
                        reason=f"Internal: move {_pos_to_str(r, c)} could not be applied."
                    )
                    if self.state.done:
                        self.state.add_observation(
                            message="Game ended due to internal invalid move handling.",
                            observation_type=ta.ObservationType.GAME_ADMIN,
                        )
                else:
                    # Announce to both players
                    src = _pos_to_str(r, c)
                    msg_cur = (
                        f"You placed a tile at {src}. Updated board:\n{self._render_board(full=False)}\n"
                        f"Scores: RED={self.game.scores[TileState.RED]}, BLUE={self.game.scores[TileState.BLUE]}"
                    )
                    self.state.add_observation(
                        from_id=-1,
                        to_id=player_id,
                        message=msg_cur,
                        observation_type=ta.ObservationType.GAME_ACTION_DESCRIPTION,
                    )
                    msg_opp = (
                        f"Player {player_id} placed a tile at {src}. Updated board:\n{self._render_board(full=False)}\n"
                        f"Scores: RED={self.game.scores[TileState.RED]}, BLUE={self.game.scores[TileState.BLUE]}"
                    )
                    self.state.add_observation(
                        from_id=-1,
                        to_id=1 - player_id,
                        message=msg_opp,
                        observation_type=ta.ObservationType.GAME_ACTION_DESCRIPTION,
                    )

        # Check for terminal state (50 placements) and set winner if needed
        if self.game.game_over:
            red, blue = self.game.scores[TileState.RED], self.game.scores[TileState.BLUE]
            if red > blue:
                self.state.set_winner(
                    player_id=0, reason=f"RED wins {red} to {blue}."
                )
            elif blue > red:
                self.state.set_winner(
                    player_id=1, reason=f"BLUE wins {blue} to {red}."
                )
            else:
                self.state.set_draw(reason=f"Draw {red} to {blue}.")
            # Explicit admin line for clarity during debugging
            self.state.add_observation(
                message=f"[ADMIN] Terminating on game_over after {self.game.placed_tiles_count} placements.",
                observation_type=ta.ObservationType.GAME_ADMIN,
            )

        # Update terminal render
        self.state.game_state["rendered_board"] = self._render_board(full=True)

        # Advance game state within TextArena
        result = self.state.step()
        # New current player gets an observation of the board and valid moves
        self._observe_current_state()
        return result

# Register with TextArena at import time
try:
    ta.envs.registration.register(
        id="Tessellate-v0",
        entry_point=TessellateTaEnv,
    )
except Exception:
    pass
