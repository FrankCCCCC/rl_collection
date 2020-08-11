# import sys
import models.ddqn as DDQN
import models.expStrategy.epsilonGreedy as EPSG
import envs.flappyBird as Game
import models.util as Util
import os
import logging
from tqdm import tqdm

# print(sys.path)
os.environ['SDL_VIDEODRIVER'] = 'dummy'
Util.test_gpu()

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
game = Game.FlappyBirdEnv()
game.reset()
NUM_STATE_FEATURES = game.get_num_state_features()
NUM_ACTIONS = game.get_num_actions()
BATCH_SIZE = 32
EPISODE_NUM = 20000
PRINT_EVERY_EPISODE = 20

exp_stg = EPSG.EpsilonGreedy(0.1, NUM_ACTIONS)
agent = DDQN.Agent((NUM_STATE_FEATURES, ), NUM_ACTIONS, 1000, 0.9, 1e-5, exp_stg)

env_state = game.get_state()
state = agent.preprocess_state(env_state)

accum_reward = 0
bar = []
logging.info("Episode 1")
for episode in range(1, EPISODE_NUM + 1):
    
    if episode % PRINT_EVERY_EPISODE == 1:
        if episode > 1:
            bar.close()
            logging.info("Accumulated Reward: {} | Loss: {}".format(round(accum_reward / PRINT_EVERY_EPISODE), agent.get_metrics_loss()))
            logging.info("Episode {}".format(episode))
            agent.reset_metrics_loss()
            accum_reward = 0
        bar = tqdm(total = PRINT_EVERY_EPISODE)

    while not game.is_over():
        # state = agent.preprocess_state(env_state)
        action = agent.select_action(state)
        reward = game.act(action)
        env_state_prime = game.get_state()
        state_prime = agent.preprocess_state(env_state_prime)

        agent.add_buffer(state, action, reward, state_prime)
        is_update_target = agent.update(BATCH_SIZE)

        state = state_prime
        accum_reward += reward

    bar.update(1)        
    game.reset()
