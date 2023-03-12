import sys
import gym
import gym_environments
from agent import ApproximateQ


def run(env, agent: ApproximateQ, selection_method, episodes):
    for episode in range(1, episodes + 1):
        if episode % 100 == 0:
            print(f"Episode: {episode}")
        observation, _ = env.reset()
        agent.start_episode()
        terminated, truncated = False, False
        while not (terminated or truncated):
            action = agent.get_action(observation, selection_method)
            next_observation, reward, terminated, truncated, _ = env.step(action)
            agent.update(observation, action, next_observation, reward)
            observation = next_observation


if __name__ == "__main__":
    environments = ["MountainCar-v0", "Pacman-v0"]
    id = 0 if len(sys.argv) < 2 else int(sys.argv[1])
    episodes = 10000 if len(sys.argv) < 3 else int(sys.argv[2])

    env = gym.make(environments[id])
    agent = ApproximateQ(
        env.observation_space.shape[0],
        env.action_space.n,
        alpha=0.1,
        gamma=0.95,
        epsilon=0.6,
    )

    # Train
    run(env, agent, "epsilon-greedy", episodes)
    env.close()

    # Play
    env = gym.make(environments[id], render_mode="human")
    run(env, agent, "greedy", 1)
    agent.render()
    env.close()
