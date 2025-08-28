#!/usr/bin/env python3
"""
Programmatic test client for Tessellate agent server streaming.

Usage:
  python tools/test_stream_client.py --url http://127.0.0.1:8001 --timeout 25

It prints basic milestones and exits non-zero on failure so you can use it in CI.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from typing import List

import requests


def build_payload() -> dict:
    # Empty board: 100 zeros; set turn to BLUE (2) so AI moves.
    state = [0] * 100 + [2, 1, 1, 0]
    valid_actions = list(range(100))
    return {"state": state, "valid_actions": valid_actions}


def test_stream(url: str, timeout: int) -> int:
    payload = build_payload()
    s = requests.Session()
    started = time.time()
    print(f"POST {url}/move_stream ...", flush=True)
    r = s.post(
        f"{url}/move_stream",
        headers={"Accept": "text/event-stream", "Content-Type": "application/json"},
        json=payload,
        stream=True,
        timeout=timeout,
    )
    if not r.ok:
        print(f"HTTP {r.status_code} {r.text[:200]}", file=sys.stderr)
        return 2
    have_delta = False
    final_action = None
    buf = ""
    for chunk in r.iter_content(chunk_size=1024, decode_unicode=True):
        if chunk:
            buf += chunk
            while True:
                i = buf.find("\n")
                if i == -1:
                    break
                line = buf[:i].strip()
                buf = buf[i + 1 :]
                if not line or line.startswith(":"):
                    continue
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except Exception:
                    continue
                t = obj.get("type")
                if t in ("reasoning", "content") and isinstance(obj.get("delta"), str):
                    if not have_delta:
                        print("first delta received at +{:.2f}s".format(time.time() - started))
                    have_delta = True
                if t == "final":
                    final_action = obj.get("action")
                    print("final action:", final_action)
                    break
        if time.time() - started > timeout:
            print("timeout waiting for stream", file=sys.stderr)
            return 3
        if final_action is not None:
            break
    if not have_delta:
        print("no deltas received", file=sys.stderr)
        return 4
    if not isinstance(final_action, int) or not (0 <= final_action < 100):
        print("invalid final action", file=sys.stderr)
        return 5
    print("OK: streaming path healthy")
    return 0


def test_nonstream(url: str, timeout: int) -> int:
    payload = build_payload()
    print(f"POST {url}/move ...", flush=True)
    r = requests.post(
        f"{url}/move",
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=timeout,
    )
    if not r.ok:
        print(f"HTTP {r.status_code} {r.text[:200]}", file=sys.stderr)
        return 6
    try:
        obj = r.json()
    except Exception as e:
        print("bad JSON:", e, file=sys.stderr)
        return 7
    action = obj.get("action")
    if not isinstance(action, int) or not (0 <= action < 100):
        print("invalid action in nonstream", file=sys.stderr)
        return 8
    print("OK: non-stream path healthy")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8001")
    ap.add_argument("--timeout", type=int, default=25)
    ap.add_argument("--skip-nonstream", action="store_true")
    args = ap.parse_args()

    rc = test_stream(args.url, args.timeout)
    if rc != 0:
        sys.exit(rc)
    if not args.skip_nonstream:
        rc = test_nonstream(args.url, args.timeout)
        if rc != 0:
            sys.exit(rc)
    print("All tests passed.")


if __name__ == "__main__":
    main()
