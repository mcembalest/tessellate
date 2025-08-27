#!/usr/bin/env python3
"""
Referee/orchestrator to play Tessellate games between two external agent processes
that speak the JSONL protocol (see agent_protocol.py).

Usage examples:
  # Two PQN checkpoints playing each other locally
  uv run referee.py \
    --agent1 "uv run pqn_agent_cli.py --checkpoint checkpoints/pqn_model_batch50_20250819_170514.pt" \
    --agent2 "uv run pqn_agent_cli.py --checkpoint checkpoints/pqn_model_batch100_20250819_170515.pt" \
    --games 10 --out game_data/pqn_vs_pqn.json

Notes:
  - The referee is authoritative and uses TessellateEnv for rules.
  - Invalid actions immediately forfeit the game (consistent with env's -1/terminal behavior),
    but the referee will mark the game as done and set winner to the other player.
  - Outputs a JSON array of games compatible with game-browser.js.
"""

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from tessellate_env import TessellateEnv


class AgentProcess:
    def __init__(self, cmd: str, timeout: float = 5.0):
        self.cmd = cmd
        self.timeout = timeout
        self.proc: Optional[subprocess.Popen] = None

    def start(self):
        self.proc = subprocess.Popen(
            shlex.split(self.cmd),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        try:
            self.proc.stdout.flush()
            line = self._readline(timeout=0.5)
            if line:
                pass
        except Exception:
            pass

    def request_action(self, state: List[float], valid_actions: List[int], player: int) -> int:
        assert self.proc and self.proc.stdin and self.proc.stdout
        msg = {"type": "move_request", "state": state, "valid_actions": valid_actions, "player": player}
        self.proc.stdin.write(json.dumps(msg) + "\n")
        self.proc.stdin.flush()
        line = self._readline(timeout=self.timeout)
        if not line:
            raise TimeoutError("Agent did not respond in time")
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON from agent: {line}")
        if data.get("type") != "move_response" or "action" not in data:
            raise ValueError(f"Invalid response from agent: {data}")
        return int(data["action"])

    def stop(self):
        if self.proc and self.proc.stdin:
            try:
                self.proc.stdin.write(json.dumps({"type": "goodbye"}) + "\n")
                self.proc.stdin.flush()
            except Exception:
                pass
        if self.proc:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=1.0)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass

    def _readline(self, timeout: float) -> Optional[str]:
        assert self.proc and self.proc.stdout
        start = time.time()
        buf = ""
        self.proc.stdout.flush()
        while time.time() - start < timeout:
            ch = self.proc.stdout.read(1)
            if ch == "\n":
                return buf
            if ch:
                buf += ch
            else:
                time.sleep(0.01)
        return None


def play_one_game(agent_red: AgentProcess, agent_blue: AgentProcess) -> Dict:
    env = TessellateEnv(reward_mode="sparse")
    state = env.reset()
    moves: List[Dict] = []

    while not env.is_terminal():
        valid = env.get_valid_actions()
        if not valid:
            break
        current_player = 1 if state[100] == 1 else 2
        agent = agent_red if current_player == 1 else agent_blue
        try:
            action = agent.request_action(state.tolist(), valid, current_player)
        except Exception:
            winner = 2 if current_player == 1 else 1
            return {
                "moves": moves,
                "final_scores": {"red": int(env.game.scores.get(1, 1)), "blue": int(env.game.scores.get(2, 1))},
                "winner": winner,
                "total_moves": len(moves),
                "forfeit": current_player,
            }

        row, col = divmod(action, 10)
        score_before = {"1": int(env.game.scores.get(1, 1)), "2": int(env.game.scores.get(2, 1))}
        next_state, reward, done, info = env.step(action)
        if info.get("invalid_move"):
            winner = 2 if current_player == 1 else 1
            return {
                "moves": moves,
                "final_scores": {"red": int(env.game.scores.get(1, 1)), "blue": int(env.game.scores.get(2, 1))},
                "winner": winner,
                "total_moves": len(moves),
                "invalid_move": {"player": current_player, "action": int(action)},
            }
        moves.append({"player": current_player, "position": [int(row), int(col)], "score_before": score_before})
        state = next_state

    final_scores = {"red": int(env.game.scores.get(1, 1)), "blue": int(env.game.scores.get(2, 1))}
    winner = int(env.game.get_winner()) if env.game.get_winner() else None
    return {"moves": moves, "final_scores": final_scores, "winner": winner, "total_moves": len(moves)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent1", required=True, help="Command to run agent 1 (Red)")
    parser.add_argument("--agent2", required=True, help="Command to run agent 2 (Blue)")
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--out", type=Path, default=Path("game_data/pvp_games.json"))
    parser.add_argument("--timeout", type=float, default=5.0, help="Per-move timeout seconds")
    args = parser.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    agent_red = AgentProcess(args.agent1, timeout=args.timeout)
    agent_blue = AgentProcess(args.agent2, timeout=args.timeout)
    agent_red.start()
    agent_blue.start()
    games: List[Dict] = []
    try:
        for _ in range(args.games):
            games.append(play_one_game(agent_red, agent_blue))
    finally:
        agent_red.stop()
        agent_blue.stop()

    with open(args.out, "w") as f:
        json.dump(games, f)
    print(f"Saved {len(games)} games to {args.out}")


if __name__ == "__main__":
    main()
