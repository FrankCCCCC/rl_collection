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

        # For A2C loss function coefficients
        self.coef_entropy = 1
        self.coef_value = 1

        self.buffer = []
        self.reset_buffer()

    def build_model(self, name):
        # Shared layers
        nn_input = tf.keras.Input(shape = self.state_size, dtype = self.data_type)
        x = tf.keras.layers.Dense(units = 128)(nn_input)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Dense(units = 128)(x)
        common = tf.keras.layers.ReLU()(x)

        # Actor Model
        actor_layer = tf.keras.layers.Dense(units = 64)(common)
        actor_layer = tf.keras.layers.ReLU()(actor_layer)
        actor_layer = tf.keras.layers.Dense(units = self.num_action)(actor_layer)
        actor_nn_output = tf.keras.activations.softmax(actor_layer)

        # Critic Model
        critic_layer = tf.keras.layers.Dense(units = 64)(common)
        critic_layer = tf.keras.layers.ReLU()(critic_layer)
        critic_nn_output = tf.keras.layers.Dense(units = 1)(critic_layer)

        # Combine into a model
        model = tf.keras.Model(name = name, inputs = nn_input, outputs = [actor_nn_output, critic_nn_output])

        return model

    def predict(self, state):
        return self.model(tf.convert_to_tensor(state, self.data_type))

    def loss(self, states, actions, rewards, state_primes, model_outputs):
        # Slice the model_outputs
        print(model_outputs)
        # np_model_output = np.concatenate(model_outputs, axis = 0, )
        act_dists = np_model_output[:, 0]
        values = np_model_output[:, 1]

        # Calculate accumulated reward Q(s, a) with discount
        np_rewards = np.array(rewards)
        num_reward = np_rewards.shape[0]
        discounts = np.logspace(1, num_reward, base = self.reward_discount, num = num_reward)
        q_values = np.zeros(num_reward)
        for i in range(num_reward):
            q_values[i] = np.sum(np.multiply(np_rewards[i:], discounts[:num_reward - i]))
        # q_values = (q_values - np.mean(q_values)) / (np.std(q_values) + 1e-9)

        # Calculate log probability log(Pr(s, a| Theta)) of actions
        indice = tf.stack([tf.range(len(actions)), actions], axis = 1)
        predict_probs = tf.gather_nd(act_dists, indice)
        predict_log_probs = tf.math.log(predict_probs)

        # Calculate the Advantage A(s, a) = Q_value(s, a) - value(s)
        advs = q_values - values

        # Calculate the cross entropy of action distribution

        # log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=act_dists, labels=actions)
        # log_prob = tf.reduce_sum(tf.math.log(act_dists) * tf.one_hot(actions, self.num_action), axis = 1)

        # Compute loss as formular: loss = Sum of a trajectory(-log(Pr(s, a| Theta)) * Advantage + coefficient of value * Value + coefficient of entropy * cross entropy of action distribution)
        # Advantage: A(s, a) = Q_value(s, a) - value(s)
        # The modification refer to the implement of Baseline A2C from OpenAI
        # Update model with a trajectory Every time.
        return tf.reduce_sum(-predict_log_probs * advs + self.coef_value * values)

    def get_metrics_loss(self):
        return self.avg_loss.result()
    
    def reset_metrics_loss(self):
        self.avg_loss.reset_states()

    def select_action(self, state):
        # Predict the probability of each action(Stochastic Policy)
        act_dist, adv = self.predict([state])
        # Assume using Epsilon Greedy Strategy
        action = self.exploration_strategy.select_action()
        # If the index of action (return value) is -1, choose the action with highest probability that model predict
        
        if action == -1 or self.shutdown_explore == True:
            print(self.predict([state]))
            print(act_dist)
            print(adv)
            # Pick then action with HIGHTEST probability
            return tf.argmax(act_dist, axis = 1)[0], [act_dist, adv]
        else:
            # If the index of action (return value) is != -1, act randomly    
            return action, zip(act_dist, adv)

    def shutdown_explore(self):
        self.is_shutdown_explore = True
        self.exploration_strategy.shutdown_explore()
    
    def update(self):
        with tf.GradientTape() as tape:
            sample_states, sample_actions, sample_rewards, sample_state_primes, sample_act_dists = self.sample()
            loss = self.loss(sample_states, sample_actions, sample_rewards, sample_state_primes, sample_act_dists)
            # print("Loss: {}".format(loss))
        # Update gradient
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        self.avg_loss.update_state(loss)

        # Update exploration rate of Epsilon Greedy Strategy
        self.exploration_strategy.update_epsilon()

        self.iter += 1

        return loss

    def reset_buffer(self):
        # Init & Reset buffer
        # The buffer is used for Historical Replay / Trajectory Storing etc...
        self.buffer = {'state': [], 'action': [], 'reward': [], 'state_prime': [], 'model_output': []}

    def add_buffer(self, new_state, new_action, new_reward, new_state_prime, new_model_output):
        self.buffer['state'].append(new_state)
        self.buffer['action'].append(new_action)
        self.buffer['reward'].append(new_reward)
        self.buffer['state_prime'].append(new_state_prime)
        self.buffer['model_output'].append(new_model_output)
    
    def sample(self):
        # Return whole trajectory
        return self.buffer['state'], self.buffer['action'], self.buffer['reward'], self.buffer['state_prime'], self.buffer['model_output']