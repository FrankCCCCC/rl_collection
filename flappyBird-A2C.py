# CartPole-A2C Experiment
# 2020/08/11 SYC 

import models.A2C as A2C
import models.expStrategy.epsilonGreedy as EPSG
import envs.flappyBird as FlappyBird
import models.util as Util
import logging
import matplotlib.pyplot as plt
from matplotlib.pylab import figure
import os
import numpy as np
# To run tqdm on notebook, import tqdm.notebook
# from tqdm.notebook import tqdm
# Run on pure python
from tqdm import tqdm

# Config Logging format
# logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
# Config logging module to enable on notebook
# logger = logging.getLogger()
# logger.setLevel(logging.DEBUG)

# Block any pop-up windows
os.environ['SDL_VIDEODRIVER'] = 'dummy'

# Test GPU and show the available logical & physical GPUs
Util.test_gpu()

env = FlappyBird.FlappyBirdEnv()
NUM_STATE_FEATURES = env.get_num_state_features()
NUM_ACTIONS = env.get_num_actions()
EPISODE_NUM = 10000
PRINT_EVERY_EPISODE = 50
LEARNING_RATE = 0.003
REWARD_DISCOUNT = 0.99

exp_stg = EPSG.EpsilonGreedy(0.2, NUM_ACTIONS)
agent = A2C.Agent((NUM_STATE_FEATURES, ), NUM_ACTIONS, REWARD_DISCOUNT, LEARNING_RATE, exp_stg)

state = env.reset()
accum_reward = 0

# tqdm progress bar
bar = []
# Reward & LossHistory
r_his = []
avg_r_his = [0]
loss_his = []
episode_reward = 0

print("Episode 1")
for episode in range(1, EPISODE_NUM + 1):
    if episode % PRINT_EVERY_EPISODE == 1:
        if episode > 1:
            bar.close()
            print("Avgerage Accumulated Reward: {} | Loss: {}".format(round(accum_reward / PRINT_EVERY_EPISODE), agent.get_metrics_loss()))
            print("Episode {}".format(episode))
            agent.reset_metrics_loss()
            avg_r_his.append(round(accum_reward / PRINT_EVERY_EPISODE))
            accum_reward = 0
        bar = tqdm(total = PRINT_EVERY_EPISODE)

    episode_reward, episode_loss = agent.train_on_env(env)
    accum_reward += episode_reward
    r_his.append(episode_reward)
    loss_his.append(episode_loss)
    
    episode_reward = 0

    bar.update(1)        
    env.reset()

bar.close()    
print("Accumulated Reward: {} | Loss: {}".format(round(accum_reward / PRINT_EVERY_EPISODE), agent.get_metrics_loss()))
avg_r_his.append(round(accum_reward / PRINT_EVERY_EPISODE))
agent.reset_metrics_loss()

# Evaluate the model
agent.shutdown_explore()
agent.reset_metrics_loss()
# Reset Game
env_state = env.reset()
accum_reward = 0

while not env.is_over():
    # env.render()
    action, act_log_prob, value = agent.select_action(state)
    state_prime, reward, is_done, info = env.act(action)

    state = state_prime
    accum_reward += reward

print("Evaluate")
print("Accumulated Reward: {}".format(round(accum_reward)))

# Plot Reward History
# figure(num=None, figsize=(24, 6), dpi=80)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(24, 6), dpi=80)
fig.suptitle(f'FlappyBird A2C Result (Evaluate Reward: {round(accum_reward)})')
x_datas = range(0, len(r_his))
avg_x_datas = range(0, EPISODE_NUM + 1, PRINT_EVERY_EPISODE)

ax1.plot(x_datas, r_his, color='blue')
ax1.plot(avg_x_datas, avg_r_his, color='red')
ax1.set_xlabel('Episodes')
ax1.set_ylabel('Reward / Episode')
ax1.grid()

ax2.plot(x_datas, loss_his, color='orange')
ax2.set_xlabel('Episodes')
ax2.set_ylabel('Loss / Episode')
ax2.grid()

plt.savefig('FlappyBird-A2C-res.svg')
plt.show()