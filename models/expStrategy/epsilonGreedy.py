import numpy as np

class EpsilonGreedy:
    def __init__(self, epsilon, num_action, min_epsilon = 0.01, decay = 0.99):
        self.epsilon = epsilon
        self.num_action = num_action
        self.min_epsilon = min_epsilon
        self.decay = decay
        self.action_time = 0

    def select_action(self, act_dist):
        act_dist_in = np.array(act_dist)
        self.action_time += 1
        if np.random.rand() < self.epsilon:
            self.update_epsilon()
            # print('Rand')
            return np.random.choice(self.num_action)
        else: 
            self.update_epsilon()
            # print('Opt')
            return np.argmax(act_dist_in)
    
    def update_epsilon(self):
        self.epsilon = max(self.min_epsilon, min(0.5, 0.99**(self.action_time / 30)))
    
    def shutdown_explore(self):
        self.epsilon = 0

# if __name__ == '__main__':
#     epg = EpsilonGreedy(1, 2)
#     print(epg.select_action([1, 1, 1, 1]))
#     epg = EpsilonGreedy(0, 2)
#     print(epg.select_action([1, 2, 1, 4]))