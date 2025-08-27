import random
import re
from typing import Any, Dict, List, Tuple

import verifiers as vf
from datasets import Dataset

from tessellate import TessellateGame, TileState


ROW_LETTERS = [chr(ord("A") + i) for i in range(10)]


def _pos_to_str(r: int, c: int) -> str:
    return f"{ROW_LETTERS[r]}{c}"


def terminal_outcome_reward(*, state: vf.State, **kwargs) -> float:
    """Simple terminal reward based on winner in state."""
    winner = state.get("winner")
    if winner == "RED":
        return 1.0
    if winner == "BLUE":
        return -1.0
    return 0.0


class TessellateVerifiersEnv(vf.MultiTurnEnv):
    """
    Minimal multi-turn verifiers Environment for Tessellate.

    One model plays as RED against a random BLUE opponent controlled by the env.
    On each turn, the model should output a single move in format [A0].
    The env applies the RED move (if valid), then applies a random BLUE move,
    and returns the updated board and scores.
    """

    def __init__(
        self,
        num_train_examples: int = 8,
        num_eval_examples: int = 2,
        max_turns: int = 25,
        seed: int = 0,
        **kwargs,
    ):
        self.seed = seed
        random.seed(self.seed)

        # Build datasets with identical initial prompts (board freshly reset)
        base_prompt = self.initial_prompt()
        rows = [{"question": base_prompt, "answer": ""} for _ in range(num_train_examples)]
        eval_rows = [
            {"question": base_prompt, "answer": ""} for _ in range(num_eval_examples)
        ]
        dataset = Dataset.from_list(rows)
        eval_dataset = Dataset.from_list(eval_rows) if num_eval_examples > 0 else None

        # Create rubric: reward only at terminal
        rubric = vf.Rubric(funcs=[terminal_outcome_reward], weights=[1.0])

        super().__init__(
            dataset=dataset,
            eval_dataset=eval_dataset,
            system_prompt=None,
            parser=vf.Parser(),  # we will parse moves manually
            rubric=rubric,
            message_type="chat",
            max_turns=max_turns,
            **kwargs,
        )

    def initial_prompt(self) -> str:
        game = TessellateGame()
        board = self.render_board(game)
        return (
            "[GAME] You are playing Tessellate as Player RED.\n"
            "Rules summary:\n"
            "- Players alternately place right triangular tiles on a 5x5 grid of squares.\n"
            "- Placing a tile blocks the two adjacent corners in the same square (marked x).\n"
            "- Thus there are 4 valid moves in an empty square, but only one valid move in a square with another tile present.\n"
            "- Score = product of the sizes of your islands (connected tile groups).\n"
            "- Game ends after 50 total moves (25 per side). Higher score wins.\n\n"
            "How to act:\n"
            "- Respond with a single move like [A0] (row letter A-J, column 0-9).\n\n"
            "- This coordinate indicates the corner of the square you are placing your tile in.\n\n"
            "- For example, [A0] is the top left corner of the top left square, and [B1] is the bottom right corner of the top left square.\n\n"
            "Initial board:\n" + board + "\n\nYour move as RED:"
        )

    def setup_state(self, state: vf.State, **kwargs) -> vf.State:
        state["game"] = TessellateGame()
        state["invalid_count"] = 0
        state["done"] = False
        state["winner"] = None
        # Start position already rendered in prompt
        return state

    def is_completed(self, messages: vf.Messages, state: vf.State, **kwargs) -> bool:
        return bool(state.get("done", False))

    def env_response(
        self, messages: vf.Messages, state: vf.State, **kwargs
    ) -> tuple[vf.Messages, vf.State]:
        assert isinstance(messages, list) and messages, "messages must be non-empty list"
        assert messages[-1]["role"] == "assistant", "Expect assistant to propose a move"

        game: TessellateGame = state["game"]

        # Extract move like [A0]
        content = messages[-1]["content"] or ""
        m = re.search(r"\b\[?([A-Ja-j])\s*([0-9])\]?\b", content)
        if not m:
            state["invalid_count"] += 1
            reply = (
                "[GAME] Invalid format. Please reply with a single coordinate like [A0].\n"
                f"Board:\n{self.render_board(game)}\nYour move as RED:"
            )
            return ([{"role": "user", "content": reply}]), state

        r = ord(m.group(1).upper()) - ord("A")
        c = int(m.group(2))

        # Validate and apply RED move
        if not (0 <= r < 10 and 0 <= c < 10) or not game.is_playable(r, c):
            state["invalid_count"] += 1
            reply = (
                f"[GAME] Invalid move {ROW_LETTERS[r] if 0<=r<10 else '?'}{c}. "
                "Choose an empty '.' cell (not blocked 'x').\n"
                f"Board:\n{self.render_board(game)}\nYour move as RED:"
            )
            return ([{"role": "user", "content": reply}]), state

        # Apply RED move
        game.current_turn = TileState.RED
        game.make_move(r, c)

        # Check terminal
        if game.game_over:
            state["done"] = True
            red, blue = game.scores[TileState.RED], game.scores[TileState.BLUE]
            if red > blue:
                state["winner"] = "RED"
                outcome = f"[GAME] You placed {ROW_LETTERS[r]}{c}. RED wins {red} to {blue}!"
            elif blue > red:
                state["winner"] = "BLUE"
                outcome = f"[GAME] You placed {ROW_LETTERS[r]}{c}. BLUE wins {blue} to {red}."
            else:
                state["winner"] = "TIE"
                outcome = f"[GAME] You placed {ROW_LETTERS[r]}{c}. It's a tie {red} to {blue}."
            outcome += "\nFinal board:\n" + self.render_board(game)
            return ([{"role": "user", "content": outcome}]), state

        # Apply BLUE random move
        game.current_turn = TileState.BLUE
        blue_moves = game.get_valid_moves()
        if blue_moves:
            br, bc = random.choice(blue_moves)
            game.make_move(br, bc)
            blue_msg = f" Opponent played {ROW_LETTERS[br]}{bc}."
        else:
            blue_msg = " Opponent had no valid move."

        # Check terminal again
        if game.game_over:
            state["done"] = True
            red, blue = game.scores[TileState.RED], game.scores[TileState.BLUE]
            if red > blue:
                state["winner"] = "RED"
                outcome = f"[GAME] You placed {ROW_LETTERS[r]}{c}." + blue_msg + f" RED wins {red} to {blue}!"
            elif blue > red:
                state["winner"] = "BLUE"
                outcome = f"[GAME] You placed {ROW_LETTERS[r]}{c}." + blue_msg + f" BLUE wins {blue} to {red}."
            else:
                state["winner"] = "TIE"
                outcome = f"[GAME] You placed {ROW_LETTERS[r]}{c}." + blue_msg + f" It's a tie {red} to {blue}."
            outcome += "\nFinal board:\n" + self.render_board(game)
            return ([{"role": "user", "content": outcome}]), state

        # Otherwise, continue
        red, blue = game.scores[TileState.RED], game.scores[TileState.BLUE]
        reply = (
            f"[GAME] You placed {ROW_LETTERS[r]}{c}." + blue_msg + "\n"
            f"Scores: RED={red}, BLUE={blue}\n"
            f"Board:\n{self.render_board(game)}\n"
            "Your move as RED:"
        )
        return ([{"role": "user", "content": reply}]), state

    @staticmethod
    def render_board(game: TessellateGame) -> str:
        s = {TileState.EMPTY: ".", TileState.RED: "R", TileState.BLUE: "B", TileState.BLOCKED: "x"}
        lines = ["     " + "  ".join(str(i) for i in range(10))]
        for r in range(10):
            row = [f"{ROW_LETTERS[r]:>2}  "]
            for c in range(10):
                row.append(s[game.board[r][c]] + " ")
            lines.append("".join(row))
        return "\n".join(lines)


def load_environment(**kwargs) -> vf.Environment:
    """Factory for the verifiers environment."""
    return TessellateVerifiersEnv(**kwargs)
