# CartPole-REINFORCE Experiment
# 2020/08/11 SYC 

import models.REINFORCE as REINFORCE
import models.expStrategy.epsilonGreedy as EPSG
import envs.flappyBird as flappyBird
import models.util as Util
import os
import logging
import matplotlib.pyplot as plt
from matplotlib.pylab import figure
import tensorflow as tf
import numpy as np
# To run tqdm on notebook, import tqdm.notebook
# from tqdm.notebook import tqdm
# Run on pure python
from tqdm import tqdm

# Config logging format
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
# Config logging module to enable on notebook
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# Test GPU and show the available logical & physical GPUs
Util.test_gpu()

# Block any pop-up windows
os.environ['SDL_VIDEODRIVER'] = 'dummy'

env = flappyBird.FlappyBirdEnv()
NUM_STATE_FEATURES = env.get_num_state_features()
NUM_ACTIONS = env.get_num_actions()
EPISODE_NUM = 20000
PRINT_EVERY_EPISODE = 20
LEARNING_RATE = 1e-4
REWARD_DISCOUNT = 0.9

# Over write the original NN model and loss function
class FlB_REIN(REINFORCE.Agent):
    # Build a new model for Flappy Bird Env
    def build_model(self, name):
        nn_input = tf.keras.Input(shape = self.state_size, dtype = self.data_type)
        initializer = tf.keras.initializers.TruncatedNormal(mean = 0, stddev = 0.1)

        x = tf.keras.layers.Dense(units = 128, kernel_initializer = initializer)(nn_input)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Dense(units = 128, kernel_initializer = initializer)(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Dense(units = self.num_action, kernel_initializer = initializer)(x)
        nn_output = tf.keras.activations.softmax(x)

        model = tf.keras.Model(name = name, inputs = nn_input, outputs = nn_output)

        print("Over-write")
        return model

    # Construct new loss function for Flappy Bird Env
    def loss(self, states, actions, rewards, state_primes):
        # Calculate accumulated reward with discount
        np_rewards = np.array(rewards)
        # np_rewards[-1] = 0
        # print(rewards)
        num_reward = np_rewards.shape[0]
        discounts = np.logspace(1, num_reward, base = self.reward_discount, num = num_reward)
        gt = np.zeros(num_reward)
        for i in range(num_reward):
            gt[i] = np.sum(np.multiply(np_rewards[i:], discounts[:num_reward - i]))
        gt = (gt - np.mean(gt)) / (np.std(gt) + 1e-9)

        # print(gt)
        # print(states)
        predicts = self.predict(states)
        print(predicts)
        
        # indice = tf.stack([tf.range(len(actions)), actions], axis = 1)
        # predict_probs = tf.gather_nd(predicts, indice)
        # predict_log_probs = tf.math.log(predict_probs)

        log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=predicts, labels=actions)
        # log_prob = tf.reduce_sum(tf.math.log(predicts) * tf.one_hot(actions, self.num_action), axis = 1)
        # print(log_prob)

        # Compute loss as formular: loss = Sum of a trajectory(-gamma * log(Pr(s, a| Theta)) * Gt)
        # Update model with a trajectory Every time.
        loss = tf.reduce_sum(-log_prob * gt)
        print(loss)
        return loss

exp_stg = EPSG.EpsilonGreedy(0.1, NUM_ACTIONS)
agent = FlB_REIN((NUM_STATE_FEATURES, ), NUM_ACTIONS, REWARD_DISCOUNT, LEARNING_RATE, exp_stg)

state = env.reset()

accum_reward = 0
bar = []
logging.info("Episode 1")
for episode in range(1, EPISODE_NUM + 1):
    
    if episode % PRINT_EVERY_EPISODE == 1:
        if episode > 1:
            bar.close()
            logging.info("Avgerage Accumulated Reward: {} | Loss: {}".format(round(accum_reward / PRINT_EVERY_EPISODE), agent.get_metrics_loss()))
            logging.info("Episode {}".format(episode))
            agent.reset_metrics_loss()
            accum_reward = 0
        bar = tqdm(total = PRINT_EVERY_EPISODE)

    while not env.is_over():
        # env.render()
        action = agent.select_action(state)
        state_prime, reward, is_done, info = env.act(action)

        agent.add_buffer(state, action, reward, state_prime)
        # print(f'State: {state}, Action: {action}, Reward: {reward}, State_Prime: {state_prime}')

        state = state_prime
        accum_reward += reward

    agent.update()
    agent.reset_buffer()

    bar.update(1)        
    env.reset()

# Close tqdm progress bar
bar.close()    
logging.info("Accumulated Reward: {} | Loss: {}".format(round(accum_reward / PRINT_EVERY_EPISODE), agent.get_metrics_loss()))
agent.reset_metrics_loss()

# Evaluate the model
agent.shutdown_explore()
agent.reset_metrics_loss()
# Reset Game
env_state = env.reset()
accum_reward = 0

while not env.is_over():
    # env.render()
    action = agent.select_action(state)
    state_prime, reward, is_done, info = env.act(action)

    state = state_prime
    accum_reward += reward

logging.info("Evaluate")
logging.info("Accumulated Reward: {}".format(accum_reward))

# Plot Reward History
figure(num=None, figsize=(16, 6), dpi=80)
plt.plot(r_his, color='blue')
# plt.plot(loss_his, color='red')
plt.xlabel('Episodes')
plt.ylabel('Avg-Accumulate Rewards')
plt.savefig('flappyBird-REINFORCE-res.svg')