# CartPole-REINFORCE Experiment
# 2020/08/11 SYC 

import models.REINFORCE as REINFORCE
import models.expStrategy.epsilonGreedy as EPSG
import envs.cartPole as cartPole
import models.util as Util
import logging
import matplotlib.pyplot as plt
from matplotlib.pylab import figure
import numpy as np
# To run tqdm on notebook, import tqdm.notebook
# from tqdm.notebook import tqdm
# Run on pure python
from tqdm import tqdm

# Config Logging format
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
# Config logging module to enable on notebook
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# Test GPU and show the available logical & physical GPUs
Util.test_gpu()

env = cartPole.CartPoleEnv()
NUM_STATE_FEATURES = env.get_num_state_features()
NUM_ACTIONS = env.get_num_actions()
EPISODE_NUM = 20
PRINT_EVERY_EPISODE = 10000
LEARNING_RATE = 2e-4
REWARD_DISCOUNT = 0.9

exp_stg = EPSG.EpsilonGreedy(0.1, NUM_ACTIONS)
agent = REINFORCE.Agent((NUM_STATE_FEATURES, ), NUM_ACTIONS, REWARD_DISCOUNT, LEARNING_RATE, exp_stg)

state = env.reset()
accum_reward = 0
# tqdm progress bar
bar = []
# Reward History
r_his = []
episode_reward = 0
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
        episode_reward += reward     

    loss = agent.update()
    agent.reset_buffer()
    r_his.append(episode_reward)
    episode_reward = 0

    bar.update(1)        
    env.reset()

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

fig = plt.gcf()
fig.set_size_inches(16, 5)
plt.plot(r_his, color='blue')
# plt.plot(loss_his, color='red')
plt.xlabel('Episodes')
plt.ylabel('Avg-Accumulate Rewards')
plt.savefig('cartPole-REINFORCE-res.svg')