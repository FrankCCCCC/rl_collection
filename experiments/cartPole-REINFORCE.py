import models.REINFORCE as REINFORCE
import models.expStrategy.epsilonGreedy as EPSG
import envs.cartPole as cartPole
import models.util as Util
import logging
import matplotlib as plt
from tqdm import tqdm

# env = cartPole.CartPoleEnv()
# print(env.get_num_actions())
# print(env.env.action_space.sample())
Util.test_gpu()

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
env = cartPole.CartPoleEnv()
env.reset()
NUM_STATE_FEATURES = env.get_num_state_features()
NUM_ACTIONS = env.get_num_actions()
BATCH_SIZE = 32
EPISODE_NUM = 2000
PRINT_EVERY_EPISODE = 20
LEARNING_RATE = 1e-4
REWARD_DISCOUNT = 0.9

exp_stg = EPSG.EpsilonGreedy(0.1, NUM_ACTIONS)
agent = REINFORCE.Agent((NUM_STATE_FEATURES, ), NUM_ACTIONS, REWARD_DISCOUNT, LEARNING_RATE, exp_stg)

state = env.get_state()

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

bar.close()

# Evaluate the model
agent.shutdown_explore()
agent.reset_metrics_loss()
while not env.is_over():
    env.render()
    action = agent.select_action(state)
    state_prime, reward, is_done, info = env.act(action)

    state = state_prime
    accum_reward += reward

logging.info("Accumulated Reward: {}".format(round(accum_reward / PRINT_EVERY_EPISODE)))