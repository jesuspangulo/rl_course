import gym
import gym_environments
import time
from agent import ValueIteration

env = gym.make('FrozenLake-v1', render_mode="human") #RobotBattery-v0, FrozenLake-v1
agent = ValueIteration(env.observation_space.n, env.action_space.n, env.P, 0.9)

agent.solve(10000)
agent.render()

observation, info = env.reset()
while True:
    action = agent.get_action(observation)
    observation, _, terminated, truncated, _ = env.step(action)
    env.render()
    time.sleep(0.25)
    if terminated or truncated:
        break

env.close()
