#!/usr/bin/env python3
"""
Play Tessellate (TextArena) LLM vs LLM in the terminal.

Usage examples:
  uv run python llm_tessellate_pvp.py \
    --model0 "GPT-4o-mini" \
    --model1 "anthropic/claude-3.5-haiku" \
    --render standard
"""

import argparse
import textarena as ta

from tessellate.environments import tessellate_textarena_env  # registers Tessellate-v0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model0", default="GPT-4o-mini")
    parser.add_argument("--model1", default="anthropic/claude-3.5-haiku")
    parser.add_argument("--render", default="standard", choices=["standard", "board", "chat", "multi"], help="Render mode")
    parser.add_argument("--error-allowance", type=int, default=4, help="Invalid move allowance before termination")
    args = parser.parse_args()

    agents = {
        0: ta.agents.OpenRouterAgent(model_name=args.model0),
        1: ta.agents.OpenRouterAgent(model_name=args.model1),
    }

    env = ta.make(env_id="Tessellate-v0", error_allowance=args.error_allowance)
    env = ta.wrappers.SimpleRenderWrapper(env=env, render_mode=args.render)

    env.reset(num_players=len(agents))

    done = False
    while not done:
        player_id, observation = env.get_observation()
        action = agents[player_id](str(observation))
        done, _ = env.step(action=action)

    rewards, game_info = env.close()
    print("\nGame finished. Rewards:", rewards)
    print("Game info:", game_info)


if __name__ == "__main__":
    main()
