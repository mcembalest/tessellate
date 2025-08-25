import textarena as ta

from tessellate_textarena_env import TessellateTaEnv

ta.envs.registration.register(
        id="Tessellate-v0",
        entry_point=TessellateTaEnv,
    )

# Initialize agents
agents = {
    0: ta.agents.OpenRouterAgent(model_name="GPT-4o-mini"),
    1: ta.agents.OpenRouterAgent(model_name="anthropic/claude-3.5-haiku"),
}

# Initialize the environment
env = ta.make(env_id="Tessellate-v0")

# wrap it for additional visualizations
env = ta.wrappers.SimpleRenderWrapper(env=env) 

env.reset(num_players=len(agents))

done = False
while not done:
    player_id, observation = env.get_observation()
    action = agents[player_id](str(observation))
    done, step_info = env.step(action=action)

rewards, game_info = env.close()