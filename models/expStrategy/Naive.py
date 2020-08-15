import numpy as np

class Naive:
    def __init__(self, num_action):
        self.num_action = num_action

    def select_action(self, action_prob_dist = None, action_value_dist = None):
        if action_prob_dist != None:
            return np.random.choice(self.num_action, p = action_prob_dist)
        if action_value_dist != None:
            action_value_dist = action_value_dist / sum(action_value_dist)
            return np.random.choice(self.num_action, p = action_prob_dist)
        else:
            return np.random.choice(self.num_action)