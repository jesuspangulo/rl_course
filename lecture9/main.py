import sys
import gym
import gym_environments
from agent import SemiGradientSARSA
import numpy as np

GROUPS = 100


def features_size(env):
    return GROUPS * env.observation_space.shape[0]


def features_to_vector(env, features):
    max = env.observation_space.high
    min = env.observation_space.low
    sizes = (max - min) / GROUPS
    if (features == max).all():
        values = np.full(len(features), GROUPS - 1)
    else:
        values = (features - min) / sizes
    vector = np.zeros(GROUPS * len(features))
    for i in range(len(features)):
        vector[int(values[i]) + i * (GROUPS)] = 1
    return vector


def run(env, agent: SemiGradientSARSA, selection_method, episodes):
    wins = 0
    for episode in range(1, episodes + 1):
        agent.start_episode()
        if episode % 100 == 0:
            print(f"Episode {episode}, wins {wins}")
            if wins >= 80:
                break
            wins = 0
        observation, _ = env.reset()
        action = agent.get_action(
            features_to_vector(env, observation), selection_method
        )
        terminated, truncated = False, False
        while not (terminated or truncated):
            next_observation, reward, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                agent.update_terminal(
                    features_to_vector(env, observation), action, reward
                )
                wins += int(terminated)
                break
            next_action = agent.get_action(
                features_to_vector(env, next_observation), selection_method
            )
            agent.update(
                features_to_vector(env, observation),
                action,
                features_to_vector(env, next_observation),
                next_action,
                reward,
            )
            observation = next_observation
            action = next_action


if __name__ == "__main__":
    environments = ["MountainCar-v0"]
    id = 0 if len(sys.argv) < 2 else int(sys.argv[1])
    episodes = 10000 if len(sys.argv) < 3 else int(sys.argv[2])

    env = gym.make(environments[id])
    agent = SemiGradientSARSA(
        features_size(env),
        env.action_space.n,
        alpha=0.05,
        gamma=0.95,
        epsilon=0.1,
    )

    # Train
    run(env, agent, "epsilon-greedy", episodes)
    env.close()

    # Play
    env = gym.make(environments[id], render_mode="human")
    run(env, agent, "greedy", 1)
    agent.render()
    env.close()
