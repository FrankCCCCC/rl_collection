# import template
import tensorflow as tf
import numpy as np

class Agent:
    def __init__(self, state_size, num_action, reward_discount, learning_rate, exploration_strategy):
        self.state_size = state_size
        self.num_action = num_action
        self.reward_discount = reward_discount
        self.exploration_strategy = exploration_strategy
        self.iter = 0
        self.data_type = tf.float32
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)
        self.avg_loss = tf.keras.metrics.Mean(name = 'loss')
        self.model = self.build_model('model')
        self.is_shutdown_explore = False

        self.buffer = []
        self.reset_buffer()

    def build_model(self, name):
        nn_input = tf.keras.Input(shape = self.state_size, dtype = self.data_type)

        x = tf.keras.layers.Dense(units = 128)(nn_input)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Dense(units = 128)(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Dense(units = self.num_action)(x)
        nn_output = tf.keras.activations.softmax(x)

        model = tf.keras.Model(name = name, inputs = nn_input, outputs = nn_output)

        return model

    def predict(self, state):
        return self.model(tf.convert_to_tensor(state, self.data_type))

    def loss(self, states, actions, rewards, state_primes):
        # Calculate accumulated reward with discount
        np_rewards = np.array(rewards)
        num_reward = np_rewards.shape[0]
        discounts = np.logspace(1, num_reward, base = self.reward_discount, num = num_reward)
        gt = np.zeros(num_reward)
        for i in range(num_reward):
            gt[i] = np.sum(np.multiply(np_rewards[i:], discounts[:num_reward - i]))
        gt = (gt - np.mean(gt)) / (np.std(gt) + 1e-9)

        predicts = self.predict(states)
        
        # indice = tf.stack([tf.range(len(actions)), actions], axis = 1)
        # predict_probs = tf.gather_nd(predicts, indice)
        # predict_log_probs = tf.math.log(predict_probs)

        # log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=predicts, labels=actions)
        log_prob = tf.reduce_sum(tf.math.log(predicts) * tf.one_hot(actions, self.num_action), axis = 1)

        # Compute loss as formular: loss = Sum of a trajectory(-gamma * log(Pr(s, a| Theta)) * Gt)
        # Update model with a trajectory Every time.
        return tf.reduce_sum(-log_prob * gt)

    def get_metrics_loss(self):
        return self.avg_loss.result()
    
    def reset_metrics_loss(self):
        self.avg_loss.reset_states()

    def select_action(self, state):
        # Assume using Epsilon Greedy Strategy
        action = self.exploration_strategy.select_action()
        # If the index of action (return value) is -1, choose the action with highest probability that model predict
        if action == -1 or self.shutdown_explore == True:
            # Predict the probability of each action(Stochastic Policy)
            predict = self.predict([state])
            # Pick then action with HIGHTEST probability
            return tf.argmax(predict, axis = 1)[0]
        else:
            # If the index of action (return value) is != -1, act randomly    
            return action

    def shutdown_explore(self):
        self.is_shutdown_explore = True
    
    def update(self):
        with tf.GradientTape() as tape:
            sample_states, sample_actions, sample_rewards, sample_state_primes = self.sample()
            loss = self.loss(sample_states, sample_actions, sample_rewards, sample_state_primes)
            # print("Loss: {}".format(loss))
        # Update gradient
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        self.avg_loss.update_state(loss)

        # Update exploration rate of Epsilon Greedy Strategy
        self.exploration_strategy.update_epsilon()

        self.iter += 1

        return loss
    
    def preprocess_state(self, env_state):
        # Preprocess SINGLE state
        return list(env_state.values())

    def preprocess_states(self, env_states):
        # Preprocess MULTIPLE states
        state_list = []
        for env_state in env_states:
            state_list.append(list(env_state.values()))

        return state_list

    def reset_buffer(self):
        # Init & Reset buffer
        # The buffer is used for Historical Replay / Trajectory Storing etc...
        self.buffer = {'state': [], 'action': [], 'reward': [], 'state_prime': []}

    def add_buffer(self, new_state, new_action, new_reward, new_state_prime):
        self.buffer['state'].append(new_state)
        self.buffer['action'].append(new_action)
        self.buffer['reward'].append(new_reward)
        self.buffer['state_prime'].append(new_state_prime)
    
    def sample(self):
        # Return whole trajectory
        return self.buffer['state'], self.buffer['action'], self.buffer['reward'], self.buffer['state_prime']