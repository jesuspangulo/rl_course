import sys
import time
import gym
import gym_environments
from agent import DoubleQLearning


def train(env, agent, episodes):
    for episode in range(episodes):
        print(episode)
        observation, _ = env.reset()
        terminated, truncated = False, False
        while not (terminated or truncated):
            action = agent.get_action(observation, "epsilon-greedy")
            new_observation, reward, terminated, truncated, _ = env.step(action)
            agent.update(observation, action, new_observation, reward, terminated)
            observation = new_observation


def play(env, agent):
    observation, _ = env.reset()
    terminated, truncated = False, False

    env.render()
    time.sleep(2)

    while not (terminated or truncated):
        action = agent.get_action(observation, "greedy")
        new_observation, reward, terminated, truncated, _ = env.step(action)
        agent.update(observation, action, new_observation, reward, terminated)
        observation = new_observation


if __name__ == "__main__":
    environments = ["CliffWalking-v0", "Taxi-v3", "Princess-v0", "Blocks-v0"]
    id = 0 if len(sys.argv) < 2 else int(sys.argv[1])
    episodes = 10000 if len(sys.argv) < 3 else int(sys.argv[2])

    env = gym.make(environments[id])
    agent = DoubleQLearning(
        env.observation_space.n, env.action_space.n, alpha=0.1, gamma=0.9, epsilon=0.5
    )

    train(env, agent, episodes)
    agent.render()
    env.close()

    env = gym.make(environments[id], render_mode="human")
    play(env, agent)
    env.close()
