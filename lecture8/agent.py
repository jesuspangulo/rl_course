import numpy as np


class ApproximateQ:
    def __init__(self, features_n, actions_n, alpha, gamma, epsilon):
        self.features_n = features_n
        self.actions_n = actions_n
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.reset()

    def reset(self):
        self.episode = 0
        self.step = 0
        self.state = 0
        self.action = 0
        self.next_state = 0
        self.reward = 0
        self.w = np.zeros((self.actions_n, self.features_n))

    def start_episode(self):
        self.episode += 1
        self.step = 0

    def update(self, features, action, next_features, reward):
        self._update(features, action, next_features, reward)
        difference = (
            reward
            + self.gamma * np.max(self._values(next_features))
            - self._value(features, action)
        )
        for i in range(self.features_n):
            self.w[action][i] = (
                self.w[action][i] + self.alpha * difference * features[i]
            )

    def _update(self, state, action, next_state, reward):
        self.step += 1
        self.state = state
        self.action = action
        self.next_state = next_state
        self.reward = reward

    def _values(self, features):
        values = np.zeros(self.actions_n)
        for i in range(self.actions_n):
            values[i] = self._value(features, i)
        return values

    def _value(self, features, action):
        value = 0
        for i in range(self.features_n):
            value = value + (self.w[action][i] * features[i])
        return value

    def get_action(self, features, mode):
        if mode == "random":
            return np.random.choice(self.actions_n)
        elif mode == "greedy":
            return np.argmax(self._values(features))
        elif mode == "epsilon-greedy":
            if np.random.uniform(0, 1) < self.epsilon:
                return np.random.choice(self.actions_n)
            else:
                return np.argmax(self._values(features))

    def render(self, mode="step"):
        if mode == "step":
            print(
                f"Episode: {self.episode}, Step: {self.step}, State: {self.state}, ",
                end="",
            )
            print(
                f"Action: {self.action}, Next state: {self.next_state}, Reward: {self.reward}"
            )

        elif mode == "values":
            print(f"Weights: {self.w}")
