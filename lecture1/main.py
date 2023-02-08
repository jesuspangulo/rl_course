import sys
import gym_environments
import gym
from agent import TwoArmedBandit

num_iterations = 100 if len(sys.argv) < 2 else int(sys.argv[1])
version = "v0" if len(sys.argv) < 3 else sys.argv[2]

env = gym.make(f"TwoArmedBandit-{version}")
agent = TwoArmedBandit(0.1)

env.reset(options={"delay": 1})

for iteration in range(num_iterations):
    action = agent.get_action("random")
    _, reward, _, _, _ = env.step(action)
    agent.update(action, reward)
    agent.render()

env.close()
