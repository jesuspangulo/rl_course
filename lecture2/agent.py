import numpy as np


class ValueIteration():
    def __init__(self, states_n, actions_n, P, gamma):
        self.states_n = states_n
        self.actions_n = actions_n
        self.P = P
        self.gamma = gamma
        self.reset()

    def reset(self):
        self.values = np.zeros(self.states_n)
        self.policy = np.zeros(self.states_n)

    def get_action(self, state):
        return int(self.policy[state])

    def render(self):
        print("Values: {}, Policy: {}".format(self.values, self.policy))

    def solve(self, iterations):
        for _ in range(iterations):
            for s in range(self.states_n):
                values = [sum([prob * (r + self.gamma * self.values[s_])
                               for prob, s_, r, _ in self.P[s][a]])
                          for a in range(self.actions_n)]
                self.values[s] = max(values)
                self.policy[s] = np.argmax(np.array(values))
