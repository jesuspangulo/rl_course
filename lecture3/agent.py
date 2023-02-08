import numpy as np


class MonteCarlo:
    def __init__(self, states_n, actions_n, gamma, epsilon):
        self.states_n = states_n
        self.actions_n = actions_n
        self.gamma = gamma
        self.epsilon = epsilon
        self.reset()

    def reset(self):
        self.episode = []
        self.q = np.zeros((self.states_n, self.actions_n))
        self.pi = np.full((self.states_n, self.actions_n), 1 / self.actions_n)
        self.returns = np.zeros((self.states_n, self.actions_n))
        self.returns_n = np.zeros((self.states_n, self.actions_n))

    def update(self, state, action, reward, terminated):
        self.episode.append((state, action, reward))
        if terminated == True:
            self._update_q()
            self._update_pi()
            self.episode = []

    def _update_q(self):
        states_actions = []
        [
            states_actions.append((state, action))
            for state, action, _ in self.episode
            if (state, action) not in states_actions
        ]
        for state, action in states_actions:
            first_occurence = next(
                i
                for i, step in enumerate(self.episode)
                if step[0] == state and step[1] == action
            )
            G = sum(
                [
                    step[2] * (self.gamma**i)
                    for i, step in enumerate(self.episode[first_occurence:])
                ]
            )
            self.returns[state][action] += G
            self.returns_n[state][action] += 1
            self.q[state][action] = (
                self.returns[state][action] / self.returns_n[state][action]
            )

    def _update_pi(self):
        states = []
        [states.append(state) for state, _, _ in self.episode if state not in states]
        for state in states:
            best_action = np.argmax(self.q[state])
            for action in range(self.actions_n):
                if action == best_action:
                    self.pi[state][action] = (
                        1 - self.epsilon + (self.epsilon / self.actions_n)
                    )
                else:
                    self.pi[state][action] = self.epsilon / self.actions_n

    def get_action(self, state):
        return np.random.choice(self.actions_n, p=self.pi[state])

    def get_best_action(self, state):
        return np.argmax(self.q[state])

    def render(self):
        print(f"Values: {self.q}\nPolicy: {self.pi}")
