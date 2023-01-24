import gym
import gym_environments
import time
from agent import ValueIteration

# RobotBattery-v0, FrozenLake-v1, FrozenLake-v2
env = gym.make('FrozenLake-v2', render_mode="human")
agent = ValueIteration(env.observation_space.n, env.action_space.n, env.P, 0.9)

agent.solve(1)
agent.render()

observation, info = env.reset()
terminated, truncated = False, False

env.render()

while not (terminated or truncated):
    action = agent.get_action(observation)
    observation, _, terminated, truncated, _ = env.step(action)

env.close()
