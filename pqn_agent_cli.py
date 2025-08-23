#!/usr/bin/env python3
"""
PQN agent that serves moves over a stdin/stdout JSONL protocol.

Usage:
  uv run pqn_agent_cli.py --checkpoint checkpoints/pqn_model_batch50_...pt [--device cpu|cuda]

Protocol:
  Expects lines of JSON objects with type=move_request.
  Responds with JSON lines containing type=move_response and an integer action.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch

from pqn_model import PQN


def load_model(checkpoint_path: Path, device: str = "cpu") -> PQN:
    model = PQN(state_dim=104, action_dim=100, hidden_dims=(256, 256), use_layernorm=True)
    checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def select_action(model: PQN, device: str, state: List[float], valid_actions: List[int]) -> int:
    if not valid_actions:
        return -1
    with torch.no_grad():
        s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        q = model(s).squeeze(0).cpu().numpy()
    mask = np.full_like(q, -np.inf)
    mask[np.array(valid_actions, dtype=int)] = q[np.array(valid_actions, dtype=int)]
    action = int(np.argmax(mask))
    return action


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"], help="Compute device")
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_model(args.checkpoint, device=device)

    hello = {"type": "hello", "spec_version": 1, "agent": "PQN"}
    sys.stdout.write(json.dumps(hello) + "\n")
    sys.stdout.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        if msg.get("type") == "move_request":
            state = msg.get("state", [])
            valid = msg.get("valid_actions", [])
            action = select_action(model, device, state, valid)
            resp = {"type": "move_response", "action": action}
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()
        elif msg.get("type") == "goodbye":
            break


if __name__ == "__main__":
    main()
