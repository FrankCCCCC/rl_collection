class Agent:
    def __init__(self, state_size, num_action, delay_update_every_iter, reward_discount, learning_rate, exploration_strategy):
        pass
    
    def build_model(self, name):
        pass

    def predict(self, state):
        pass

    def loss(self, states, actions, rewards, state_primes):
        pass

    def get_metrics_loss(self):
        pass
    
    def reset_metrics_loss(self):
        pass

    def select_action(self, state):
        pass

    def shutdown_explore(self):
        pass

    def update(self, batch_size):
        pass
    
    def preprocess_state(self, env_state):
        pass

    def preprocess_states(self, env_states):
        pass

    def init_buffer(self):
        pass

    def add_buffer(self, new_state, new_action, new_reward, new_state_prime):
        pass
    
    def sample(self, num_sample):
        pass