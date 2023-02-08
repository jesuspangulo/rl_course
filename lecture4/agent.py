import numpy as np


class QLearning:
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
        self.q_table = np.zeros((self.states_n, self.actions_n))

    def update(self, current_state, action, next_state, reward, terminated):
        self._update(current_state, action, next_state, reward, terminated)
        self.q_table[current_state, action] = self.q_table[
            current_state, action
        ] + self.alpha * (
            reward
            + self.gamma * np.max(self.q_table[next_state])
            - self.q_table[current_state, action]
        )

    def _update(self, current_state, action, next_state, reward, terminated):
        self.iteration += 1
        self.state = current_state
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
            return np.argmax(self.q_table[state])
        elif mode == "epsilon-greedy":
            rdm = np.random.uniform(0, 1)
            if rdm < self.epsilon:
                return np.random.choice(self.actions_n)
            else:
                return np.argmax(self.q_table[state])

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
            print("Q-Table: {}".format(self.q_table))
