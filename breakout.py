from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from dqn_agent import Agent 

seed = 42

env = make_atari("BreakoutNoFrameskip-v4")
# Warp the frames, grey scale, stake four frame and scale to smaller ratio
env = wrap_deepmind(env, frame_stack=True, scale=True)
env.seed(seed)

breakout_agent = Agent(env)
breakout_agent.train()
breakout_agent.run()

