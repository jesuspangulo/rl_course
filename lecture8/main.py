import sys
import gym
import gym_environments
import numpy as np

def run(env, episodes):
    for episode in range(episodes):
        if episode > 0:
            print(f"Episode: {episode+1}")
        observation, _ = env.reset()
        terminated, truncated = False, False
        
        while not (terminated or truncated):
            action = np.random.choice(env.action_space.n)
            next_observation, reward, terminated, truncated, _ = env.step(action)
            observation = next_observation


if __name__ == "__main__":
    environments = ["Pacman-v0"]
    id = 0 if len(sys.argv) < 2 else int(sys.argv[1])
    episodes = 350 if len(sys.argv) < 3 else int(sys.argv[2])

    env = gym.make(environments[id], render_mode="human")
    run(env, 1)
    env.close()
