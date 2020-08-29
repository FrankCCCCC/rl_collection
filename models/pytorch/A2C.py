import torch
import numpy as np

class Agent:
    def __init__(self, state_size, num_action, reward_discount, learning_rate, exploration_strategy):
        self.state_size = state_size
        self.num_action = num_action
        self.reward_discount = reward_discount
        self.exploration_strategy = exploration_strategy
        self.session = session
        self.iter = 0
        self.eps = 0
        self.is_shutdown_explore = False

        self.data_type = tf.float32
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)
        self.avg_loss = tf.keras.metrics.Mean(name = 'loss')
        self.model = self.build_model('model')

        # For A2C loss function coefficients
        self.coef_entropy = 0
        self.coef_value = 1

    def build_model(self, name):
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.model_inputs = nn.Linear(self.state_size, 128)
                self.model_h1 = nn.Linear(128, 128)
                self.model_actor_out = nn.Linear(128, self.num_action)
                self.model_critic_out = nn.Linear(128, 1)
            
            def forward(self, input):
                x = torch.nn.functional.relu(self.model_inputs(input))
                x = torch.nn.functional.relu(self.model_h1(x))
                actor_output = torch.nn.functional.softmax(self.model_actor_out(x))
                critic_output = torch.nn.functional.softmax(self.model_critic_out(x))

                return actor_output, critic_output

        return Model()

if __name__ == '__main__':
    import models.expStrategy.epsilonGreedy as EPSG
    import envs.cartPole as cartPole
    env = cartPole.CartPoleEnv()
    NUM_STATE_FEATURES = env.get_num_state_features()
    NUM_ACTIONS = env.get_num_actions()
    EPISODE_NUM = 200
    PRINT_EVERY_EPISODE = 20
    LEARNING_RATE = 0.003
    REWARD_DISCOUNT = 0.99

    exp_stg = EPSG.EpsilonGreedy(0.2, NUM_ACTIONS)
    agent = Agent((NUM_STATE_FEATURES, ), NUM_ACTIONS, REWARD_DISCOUNT, LEARNING_RATE, exp_stg)