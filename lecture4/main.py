import sys
import time
import gym
import gym_environments
from agent import QLearning

# RobotBattery-v0, Taxi-v3, FrozenLake-v1, RobotMaze-v0
ENVIRONMENT = "Princess-v0"


def train(env, agent, episodes):
    for _ in range(episodes):
        observation, _ = env.reset()
        terminated, truncated = False, False
        while not (terminated or truncated):
            action = agent.get_action(observation, "random")
            new_observation, reward, terminated, truncated, _ = env.step(action)
            agent.update(observation, action, new_observation, reward, terminated)
            observation = new_observation


def play(env, agent):
    observation, _ = env.reset()
    env.render()
    time.sleep(2)
    terminated, truncated = False, False
    while not (terminated or truncated):
        action = agent.get_action(observation, "greedy")
        new_observation, reward, terminated, truncated, _ = env.step(action)
        agent.update(observation, action, new_observation, reward, terminated)
        observation = new_observation
        env.render()


if __name__ == "__main__":

    env = gym.make(ENVIRONMENT)
    agent = QLearning(
        env.observation_space.n, env.action_space.n, alpha=0.1, gamma=0.9, epsilon=0.1
    )

    episodes = 10000 if len(sys.argv) == 1 else int(sys.argv[1])

    train(env, agent, episodes)
    agent.render()
    env.close()

    env = gym.make(ENVIRONMENT, render_mode="human")
    play(env, agent)

    env.close()
