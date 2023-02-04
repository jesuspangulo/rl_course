import gym
import gym_environments
from agent import QLearning

# RobotBattery-v0, Taxi-v3, FrozenLake-v1, RobotMaze-v0
ENVIRONMENT = 'RobotBattery-v0'


def train(env, agent, episodes):
    for _ in range(episodes):
        observation, _ = env.reset()
        terminated, truncated = False, False
        while not (terminated or truncated):
            action = agent.get_action(observation, 'epsilon-greedy')
            new_observation, reward, terminated, truncated, _ = env.step(
                action)
            agent.update(
                observation,
                action,
                new_observation,
                reward,
                terminated)
            observation = new_observation


def play(env, agent):
    observation, _ = env.reset()
    terminated, truncated = False, False
    while not (terminated or truncated):
        action = agent.get_action(observation, 'greedy')
        observation, _, terminated, truncated, _ = env.step(action)
        env.render()


if __name__ == "__main__":

    env = gym.make(ENVIRONMENT)
    agent = QLearning(
        env.observation_space.n,
        env.action_space.n,
        alpha=0.1,
        gamma=0.9,
        epsilon=0.1)

    train(env, agent, episodes=1000)
    agent.render()

    env = gym.make(ENVIRONMENT, render_mode='human')
    play(env, agent)

    env.close()
