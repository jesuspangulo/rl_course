import numpy as np


class DoubleQLearning:
    def __init__(self, states_n, actions_n, alpha, gamma, epsilon):
        self.states_n = states_n
        self.actions_n = actions_n
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.reset()

    def reset(self):
        self.episode = 0
        self.iteration = 0
        self.state = 0
        self.action = 0
        self.next_state = 0
        self.reward = 0
        self.q1 = np.zeros((self.states_n, self.actions_n))
        self.q2 = np.zeros((self.states_n, self.actions_n))
        self.q12 = np.zeros((self.states_n, self.actions_n))

    def update(self, state, action, next_state, reward, terminated):
        self._update(state, action, next_state, reward, terminated)

        if np.random.uniform(0, 1) <= 0.5:
            self.q1[state, action] = self.q1[state, action] + self.alpha * (
                reward
                + self.gamma * self.q2[next_state, np.argmax(self.q1[next_state])]
                - self.q1[state, action]
            )
        else:
            self.q2[state, action] = self.q2[state, action] + self.alpha * (
                reward
                + self.gamma * self.q1[next_state, np.argmax(self.q2[next_state])]
                - self.q2[state, action]
            )

        self.q12[state][action] = self.q1[state][action] + self.q2[state][action]

    def _update(self, state, action, next_state, reward, terminated):
        self.iteration += 1
        self.state = state
        self.action = action
        self.next_state = next_state
        self.reward = reward
        if terminated:
            self.episode += 1
            self.iteration = 0

    def get_action(self, state, mode):
        if mode == "random":
            return np.random.choice(self.actions_n)
        elif mode == "greedy":
            return np.argmax(self.q12[state])
        elif mode == "epsilon-greedy":
            if np.random.uniform(0, 1) < self.epsilon:
                return np.random.choice(self.actions_n)
            else:
                return np.argmax(self.q12[state])

    def render(self, mode="values"):
        if mode == "step":
            print(
                "Episode: {}, Iteration: {}, State: {}, Action: {}, Next state: {}, Reward: {}".format(
                    self.episode,
                    self.iteration,
                    self.state,
                    self.action,
                    self.next_state,
                    self.reward,
                )
            )
        elif mode == "values":
            print("Q1: {}\nQ2: {}".format(self.q1, self.q2))
