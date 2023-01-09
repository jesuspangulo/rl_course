import gym
import gym_environments
from agent import TwoArmedBandit

env = gym.make('TwoArmedBandit-v0')
agent = TwoArmedBandit(0.1) 

env.reset()

for iteration in range(10):
    action = agent.get_action("random")    
    _, reward, _, _, _ = env.step(action)
    agent.update(action, reward) 
    agent.render()

env.close()