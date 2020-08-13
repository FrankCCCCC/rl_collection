import tensorflow as tf
import numpy as np

class Agent:
    def __init__(self, state_size, num_action, reward_discount, learning_rate, exploration_strategy):
        self.state_size = state_size
        self.num_action = num_action
        self.reward_discount = reward_discount
        self.exploration_strategy = exploration_strategy
        self.iter = 0
        self.eps = 0
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
        # # Shared layers
        # nn_input = tf.keras.Input(shape = self.state_size, dtype = self.data_type)
        # x = tf.keras.layers.Dense(units = 64)(nn_input)
        # x = tf.keras.layers.ReLU()(x)
        # x = tf.keras.layers.Dense(units = 128)(x)
        # common = tf.keras.layers.ReLU()(x)

        # # Actor Model
        # actor_layer = tf.keras.layers.Dense(units = 64)(common)
        # actor_layer = tf.keras.layers.ReLU()(actor_layer)
        # actor_layer = tf.keras.layers.Dense(units = self.num_action)(actor_layer)
        # actor_nn_output = tf.keras.activations.softmax(actor_layer)

        # # Critic Model
        # critic_layer = tf.keras.layers.Dense(units = 64)(common)
        # critic_layer = tf.keras.layers.ReLU()(critic_layer)
        # critic_nn_output = tf.keras.layers.Dense(units = 1)(critic_layer)

        # # Combine into a model
        # model = tf.keras.Model(name = name, inputs = nn_input, outputs = [actor_nn_output, critic_nn_output])

        num_inputs = 4
        num_actions = 2
        num_hidden = 128

        inputs = tf.keras.layers.Input(shape=(num_inputs,))
        common = tf.keras.layers.Dense(num_hidden, activation="relu")(inputs)
        action = tf.keras.layers.Dense(num_actions, activation="softmax")(common)
        critic = tf.keras.layers.Dense(1)(common)

        model = tf.keras.Model(inputs=inputs, outputs=[action, critic])

        return model

    def predict(self, state):
        return self.model(tf.convert_to_tensor(state, self.data_type))

    def loss(self, states, actions, rewards, state_primes, model_outputs):
        # Slice the model_outputs
        np_model_output = np.array(model_outputs)
        act_dists = np_model_output[:, :self.num_action]
        values = np_model_output[:, -1]

        # Calculate accumulated reward Q(s, a) with discount
        np_rewards = np.array(rewards)
        num_reward = np_rewards.shape[0]
        discounts = np.logspace(0, num_reward, base = self.reward_discount, num = num_reward)
        # print('Discount')
        # print(discounts)
        # print("Rewards")
        # print(np_rewards)
        q_values = np.zeros(num_reward)
        for i in range(num_reward):
            q_values[i] = np.sum(np.multiply(np_rewards[i:], discounts[:num_reward - i]))
        # print('Q Values')
        # print(q_values)
        # q_values = (q_values - np.mean(q_values)) / (np.std(q_values) + 1e-9)

        # Calculate log probability log(Pr(s, a| Theta)) of actions
        indice = tf.stack([tf.range(len(actions)), actions], axis = 1)
        predict_probs = tf.gather_nd(act_dists, indice)
        predict_log_probs = tf.math.log(predict_probs)
        # print(actions)
        # print(predict_probs)
        # print(predict_log_probs)

        # Calculate the Advantage A(s, a) = Q_value(s, a) - value(s)
        advs = q_values - values
        # print('Q Values')
        # print(q_values)
        # print('Values')
        # print(values)
        # print('Advantages')
        # print(advs)

        # Calculate the cross entropy of action distribution

        # log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=act_dists, labels=actions)
        # log_prob = tf.reduce_sum(tf.math.log(act_dists) * tf.one_hot(actions, self.num_action), axis = 1)

        # Compute loss as formular: loss = Sum of a trajectory(-log(Pr(s, a| Theta)) * Advantage + coefficient of value * Value + coefficient of entropy * cross entropy of action distribution)
        # Advantage: A(s, a) = Q_value(s, a) - value(s)
        # The modification refer to the implement of Baseline A2C from OpenAI
        # Update model with a trajectory Every time.
        # return tf.reduce_mean(-predict_log_probs * advs + self.coef_value * tf.square(values))

        return self.loss2(predict_log_probs, values, rewards)

    def get_metrics_loss(self):
        return self.avg_loss.result()
    
    def reset_metrics_loss(self):
        self.avg_loss.reset_states()

    def select_action(self, state):
        # Predict the probability of each action(Stochastic Policy)
        act_dist, value = self.predict([state])
        act_dist = tf.squeeze(act_dist)
        value = tf.squeeze(value)
        print(act_dist)
        print(value)
        # Assume using Epsilon Greedy Strategy
        action = self.exploration_strategy.select_action()
        # If the index of action (return value) is -1, choose the action with highest probability that model predict
        
        if action == -1 or self.shutdown_explore == True:
            # Pick then action with HIGHTEST probability
            act_idx = tf.argmax(act_dist, axis = 0).numpy()
            # return tf.argmax(act_dist, axis = 1)[0].numpy(), np.concatenate((act_dist.numpy(), adv.numpy()), axis = None).tolist()
            # return act_idx, tf.math.log(act_dist[act_idx]), value
            pass
        else:
            # If the index of action (return value) is != -1, act randomly    
            # return action, np.concatenate((act_dist.numpy(), adv.numpy()), axis = None).tolist()
            # return action, tf.math.log(act_dist[action]), value
            pass
        return np.random.choice(self.num_action, p=np.squeeze(act_dist)), tf.math.log(act_dist[action]), value

    def shutdown_explore(self):
        self.is_shutdown_explore = True
        self.exploration_strategy.shutdown_explore()
    
    def update(self):
        with tf.GradientTape() as tape:
            sample_states, sample_actions, sample_rewards, sample_state_primes, sample_act_log_probs, values = self.sample()
            tape.watch(self.model.trainable_variables)
            # ============================================================
            predicts_act_probs, predict_values = self.predict(sample_states)
            indice = tf.stack([tf.range(len(sample_actions)), sample_actions], axis = 1)
            predict_probs = tf.gather_nd(predicts_act_probs, indice)
            predict_log_probs = tf.math.log(predict_probs)

            action_probs_history, critic_value_history, rewards_history = predict_log_probs, predict_values, sample_rewards
            loss = self.loss2(action_probs_history, critic_value_history, rewards_history)
            # =============================================================
            # Update gradient
            gradients = tape.gradient(loss, self.model.trainable_variables)
            # gradients = [gradients if gradients is not None else tf.zeros_like(var) for var, grad in zip(self.model.trainable_variables, gradients)]
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            self.avg_loss.update_state(loss)

        # Update exploration rate of Epsilon Greedy Strategy
        self.exploration_strategy.update_epsilon()

        self.iter += 1
        self.eps += 1

        return loss
    
    def update2(self, loss, tape):
        gradients = tape.gradient(loss, self.model.trainable_variables)
        # gradients = [gradients if gradients is not None else tf.zeros_like(var) for var, grad in zip(self.model.trainable_variables, gradients)]
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        self.avg_loss.update_state(loss)

        # Update exploration rate of Epsilon Greedy Strategy
        self.exploration_strategy.update_epsilon()

        self.iter += 1
        self.eps += 1

    def reset_buffer(self):
        # Init & Reset buffer
        # The buffer is used for Historical Replay / Trajectory Storing etc...
        self.buffer = {'state': [], 'action': [], 'reward': [], 'state_prime': [], 'act_log_prob': [], 'value': []}

    def add_buffer(self, new_state, new_action, new_reward, new_state_prime, new_act_log_prob, new_value):
        self.buffer['state'].append(new_state)
        self.buffer['action'].append(new_action)
        self.buffer['reward'].append(new_reward)
        self.buffer['state_prime'].append(new_state_prime)
        self.buffer['act_log_prob'].append(new_act_log_prob)
        self.buffer['value'].append(new_value)
    
    def sample(self):
        # Return whole trajectory
        return self.buffer['state'], self.buffer['action'], self.buffer['reward'], self.buffer['state_prime'], self.buffer['act_log_prob'], self.buffer['value']

    def loss2(self, action_probs_history, critic_value_history, rewards_history):
        # Calculate expected value from rewards
        # - At each timestep what was the total reward received after that timestep
        # - Rewards in the past are discounted by multiplying them with gamma
        # - These are the labels for our critic
        gamma = self.reward_discount
        huber_loss = tf.keras.losses.Huber()
        returns = []
        discounted_sum = 0
        for r in rewards_history[::-1]:
            discounted_sum = r + gamma * discounted_sum
            returns.insert(0, discounted_sum)

        # Normalize
        returns = np.array(returns)
        returns = (returns - np.mean(returns)) / (np.std(returns) + self.eps)
        returns = returns.tolist()

        # Calculating loss values to update our network
        history = zip(action_probs_history, critic_value_history, returns)
        # print('Loss2: ')
        # print(action_probs_history)
        # print(critic_value_history)
        # print(rewards_history)
        actor_losses = []
        critic_losses = []
        for log_prob, value, ret in history:
            # At this point in history, the critic estimated that we would get a
            # total reward = `value` in the future. We took an action with log probability
            # of `log_prob` and ended up recieving a total reward = `ret`.
            # The actor must be updated so that it predicts an action that leads to
            # high rewards (compared to critic's estimate) with high probability.
            diff = ret - value
            actor_losses.append(-log_prob * diff)  # actor loss

            # The critic must be updated so that it predicts a better estimate of
            # the future rewards.
            critic_losses.append(
                huber_loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0))
            )

        # Backpropagation
        loss_value = sum(actor_losses) + sum(critic_losses)
        # print(actor_losses)
        # print(critic_losses)
        # print(loss_value)

        return loss_value

    def train_on_env(self, env, is_show = False):
        with tf.GradientTape() as tape:
            tape.watch(self.model.trainable_variables)
            episode_reward = 0
            state = env.reset(is_show)

            action_log_probs = []
            critic_values = []
            rewards = []

            while not env.is_over():
                # env.render()
                action, act_log_prob, value = self.select_action(state)
                state_prime, reward, is_done, info = env.act(action)
                # self.add_buffer(state, action, reward, state_prime, act_log_prob, value)
                # print(f'State: {state}, Action: {action}, Reward: {reward}, State_Prime: {state_prime}')

                state = state_prime
                episode_reward += reward

                action_log_probs.append(act_log_prob)
                critic_values.append(value)
                rewards.append(reward)

            loss = self.loss2(action_log_probs, critic_values, rewards)
            self.update2(loss, tape)
            # self.reset_buffer()
            env.reset()

            return episode_reward