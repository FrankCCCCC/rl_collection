# FlappyBIrd-DDQN Experiment
# 2020/08/11 SYC 

import models.ddqn as DDQN
import models.expStrategy.epsilonGreedy as EPSG
import envs.flappyBird as Game
import models.util as Util
import os
import logging
# To run tqdm on notebook, import tqdm.notebook
# from tqdm.notebook import tqdm
# Run on pure python
from tqdm import tqdm

# Config Logging format
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
# Config logging module to enable on notebook
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# Block any pop-up windows
os.environ['SDL_VIDEODRIVER'] = 'dummy'

# Test GPU and show the available logical & physical GPUs
Util.test_gpu()

game = Game.FlappyBirdEnv()
NUM_STATE_FEATURES = game.get_num_state_features()
NUM_ACTIONS = game.get_num_actions()
BATCH_SIZE = 32
EPISODE_NUM = 20000
PRINT_EVERY_EPISODE = 20

exp_stg = EPSG.EpsilonGreedy(0.1, NUM_ACTIONS)
agent = DDQN.Agent((NUM_STATE_FEATURES, ), NUM_ACTIONS, 1000, 0.9, 1e-5, exp_stg)

state = game.reset()
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

    while not game.is_over():
        action = agent.select_action(state)
        state_prime, reward, is_done, info = game.act(action)
        # print(f'B State: {state}, Action: {action}, Reward: {reward}, State_Prime: {state_prime}')

        agent.add_buffer(state, action, reward, state_prime)
        is_update_target = agent.update(BATCH_SIZE)

        state = state_prime
        accum_reward += reward

    bar.update(1)        
    game.reset()

logging.info("Accumulated Reward: {} | Loss: {}".format(round(accum_reward / PRINT_EVERY_EPISODE), agent.get_metrics_loss()))
agent.reset_metrics_loss()
bar.close()

# Evaluate the model
agent.shutdown_explore()
agent.reset_metrics_loss()
# Reset Game
state = game.reset()
accum_reward = 0

while not game.is_over():
    action = agent.select_action(state)
    state_prime, reward, is_done, info = game.act(action)

    state = state_prime
    accum_reward += reward
    
logging.info("Evaluate")
logging.info("Accumulated Reward: {}".format(accum_reward))