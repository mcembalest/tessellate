#!/usr/bin/env python3
"""
Local HTTP server to play against a PQN agent.

Run:
  uv run agent_server.py --checkpoint checkpoints/pqn_model_batch50_20250819_170514.pt --port 8001 --device auto

Endpoint:
  POST /move { state: [104 floats], valid_actions: [ints] } -> { action: int }

Notes:
  - CORS enabled for all origins, so you can open index.html locally and fetch.
  - Uses the same action selection as pqn_agent_cli.py (mask invalid, greedy).
"""

import argparse
from pathlib import Path
from typing import List

import numpy as np
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from pqn_model import PQN


class MoveRequest(BaseModel):
    state: List[float]
    valid_actions: List[int]


class MoveResponse(BaseModel):
    action: int


def load_model(checkpoint_path: Path, device: str = "cpu") -> PQN:
    model = PQN(state_dim=104, action_dim=100, hidden_dims=(256, 256), use_layernorm=True)
    # PyTorch 2.6 defaults to weights_only=True; our checkpoints include optimizer state
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
    va = np.array(valid_actions, dtype=int)
    mask[va] = q[va]
    action = int(np.argmax(mask))
    return action


def build_app(checkpoint: Path, device: str) -> FastAPI:
    model = load_model(checkpoint, device)
    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/move", response_model=MoveResponse)
    def move(req: MoveRequest):
        action = select_action(model, device, req.state, req.valid_actions)
        return MoveResponse(action=action)

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    app = build_app(args.checkpoint, device)
    uvicorn.run(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
