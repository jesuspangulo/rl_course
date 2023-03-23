import sys
import gym
import gym_environments
import numpy as np
from agent import DYNAQ


def run(env, agent: DYNAQ, selection_method, episodes):
    for episode in range(episodes):
        if episode > 0:
            print(f"Episode: {episode+1}")
        observation, _ = env.reset()
        agent.start_episode()
        terminated, truncated = False, False
        while not (terminated or truncated):
            action = agent.get_action(observation, selection_method)
            next_observation, reward, terminated, truncated, _ = env.step(action)
            agent.update_q(observation, action, next_observation, reward)
            agent.update_model(observation, action, reward, next_observation)
            observation = next_observation
        if selection_method == "epsilon-greedy":
            for _ in range(100):
                state = np.random.choice(list(agent.visited_states.keys()))
                action = np.random.choice(agent.visited_states[state])
                reward, next_state = agent.model[(state, action)]
                agent.update_q(state, action, next_state, reward)


if __name__ == "__main__":
    environments = ["Princess-v0", "Blocks-v0"]
    id = 0 if len(sys.argv) < 2 else int(sys.argv[1])
    episodes = 350 if len(sys.argv) < 3 else int(sys.argv[2])

    env = gym.make(environments[id])
    agent = DYNAQ(
        env.observation_space.n, env.action_space.n, alpha=1, gamma=0.95, epsilon=0.1
    )

    # Train
    run(env, agent, "epsilon-greedy", episodes)
    env.close()

    # Play
    env = gym.make(environments[id], render_mode="human")
    run(env, agent, "greedy", 1)
    agent.render()
    env.close()
