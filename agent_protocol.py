#!/usr/bin/env python3
"""
Lightweight JSON Lines protocol for Tessellate agents.

Transport: stdin/stdout (one JSON per line).

Messages from referee to agent:
  {"type": "move_request", "state": [104 floats], "valid_actions": [ints], "player": 1 or 2}

Messages from agent to referee:
  {"type": "move_response", "action": int}

Optional (not required by referee):
  {"type": "hello", "spec_version": 1, "agent": "name"}
  {"type": "goodbye"}
"""

from dataclasses import dataclass
from typing import List

SPEC_VERSION = 1


@dataclass
class MoveRequest:
    state: List[float]
    valid_actions: List[int]
    player: int


@dataclass
class MoveResponse:
    action: int
