from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from dqn_agent import Agent 

pong_game = "PongNoFrameskip-v4"
pong_reward_threshold = 19.0
pong_model_filepath = "pong_model.json"
pong_model_weights_filepath = "pong_model.h5"

env = make_atari(pong_game)
env = wrap_deepmind(env, frame_stack=True, scale=True)
env.seed(seed)

pong_agent = Agent(env, reward_threshold=pong_reward_threshold,
                    model_filepath=pong_model_filepath,
                    model_weights_filepath=pong_model_weights_filepath)
pong_agent.train()
pong_agent.run()