# ddqn.py
# Implement DDQN with Delay Network, History Replay, Epsilon Greedy
# 2020/08/10 SYC

import tensorflow as tf
import numpy as np
# import matplotlib as plt
# import policiesPractice.models.expStrategy as stg

class Agent:
    def __init__(self, state_size, num_action, delay_update_every_iter, reward_discount, learning_rate, exploration_strategy):
        self.state_size = state_size
        self.num_action = num_action
        self.reward_discount = reward_discount
        self.exploration_strategy = exploration_strategy
        self.delay_update_every_iter = delay_update_every_iter
        self.iter = 0
        self.data_type = tf.float32
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)
        self.avg_loss = tf.keras.metrics.Mean(name = 'loss')
        self.online_model = self.build_model('online')
        self.target_model = self.build_model('target')
        self.is_shutdown_explore = False

        self.buffer = []
        self.buffer_size = 50000
    
    def build_model(self, name):
        nn_input = tf.keras.Input(shape = self.state_size, dtype = self.data_type)

        x = tf.keras.layers.Dense(units = 128)(nn_input)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Dense(units = 128)(x)
        x = tf.keras.layers.ReLU()(x)
        nn_output = tf.keras.layers.Dense(units = self.num_action)(x)

        model = tf.keras.Model(name = name, inputs = nn_input, outputs = nn_output)

        return model

    def predict(self, state):
        # Online Model
        return self.online_model(tf.convert_to_tensor(state, self.data_type))

    def max_q(self, state_primes):
        # Target Model
        return tf.reduce_max(self.target_model(tf.convert_to_tensor(state_primes, self.data_type)), axis = 1)

    def loss(self, states, actions, rewards, state_primes):
        predicts = self.predict(states)
        # 
        indice = tf.stack([tf.range(len(actions)), actions], axis = 1)
        predict_qs = tf.gather_nd(predicts, indice)

        target_qs = self.max_q(state_primes)
        # Compute loss as formular: loss = E((r + gamma * max(Q(s', a'| Theta')) - Q(s, a, | Theta))^2)
        # Update model with a batch Every time. As a result, we compute the Expectation(E) of the total loss of a batch.
        return tf.reduce_mean(tf.square(rewards + self.reward_discount * target_qs - predict_qs))

    def get_metrics_loss(self):
        return self.avg_loss.result()
    
    def reset_metrics_loss(self):
        self.avg_loss.reset_states()

    def select_action(self, state):
        # Assume using Epsilon Greedy Strategy
        action = self.exploration_strategy.select_action()
        # If the index of action (return value) is -1, choose the action with highest probability that online_model predict
        if action == -1:
            predict = self.predict([state])
            return tf.argmax(predict, axis = 1)[0]
        else:
            # If the index of action (return value) is != -1, act randomly    
            return action

    def shutdown_explore(self):
        self.is_shutdown_explore = True

    def update(self, batch_size):
        with tf.GradientTape() as tape:
            sample_states, sample_actions, sample_rewards, sample_state_primes = self.sample(batch_size)
            loss = self.loss(sample_states, sample_actions, sample_rewards, sample_state_primes)
            # print("{}".format(loss))
        gradients = tape.gradient(loss, self.online_model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.online_model.trainable_variables))
        self.avg_loss.update_state(loss)

        # Update exploration rate of Epsilon Greedy Strategy
        self.exploration_strategy.update_epsilon()
        
        is_update_target = False
        if self.iter % self.delay_update_every_iter == 0:
            self.target_model.set_weights(self.online_model.get_weights())
            is_update_target = True

        self.iter += 1

        return is_update_target
    
    def preprocess_state(self, env_state):
        # Preprocess SINGLE state
        return list(env_state.values())

    def preprocess_states(self, env_states):
        # Preprocess MULTIPLE states
        state_list = []
        for env_state in env_states:
            state_list.append(list(env_state.values()))

        return state_list

    def add_buffer(self, new_state, new_action, new_reward, new_state_prime):
        # Add ONE action-state pair every time
        self.buffer.append({'state': new_state, 'action': new_action, 'reward': new_reward, 'state_prime': new_state_prime})
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)
    
    def sample(self, num_sample):
        if num_sample >= len(self.buffer):
            idx_samples = np.random.choice(len(self.buffer), num_sample).tolist()
        else:
            idx_samples = np.random.choice(len(self.buffer), num_sample, replace = False).tolist()

        states = []
        actions = []
        rewards = []
        state_primes = []

        for idx in idx_samples:
            states.append(self.buffer[idx]['state'])
            actions.append(self.buffer[idx]['action'])
            rewards.append(self.buffer[idx]['reward'])
            state_primes.append(self.buffer[idx]['state_prime'])

        return states, actions, rewards, state_primes

    


