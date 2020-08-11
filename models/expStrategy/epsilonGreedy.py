import numpy as np

class EpsilonGreedy:
    def __init__(self, epsilon, num_action, min_epsilon = 0.01):
        self.epsilon = epsilon
        self.num_action = num_action
        self.min_epsilon = 0.01
        self.action_time = 0

    def select_action(self):
        self.action_time += 1
        if np.random.rand() < self.epsilon:
            self.update_epsilon()
            return np.random.choice(self.num_action)
        else: 
            self.update_epsilon()
            return -1
    
    def update_epsilon(self):
        self.epsilon = max(self.min_epsilon, min(0.5, 0.99**(self.action_time / 30)))
    
    def shutdown_explore(self):
        self.epsilon = 0