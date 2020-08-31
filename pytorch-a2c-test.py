import models.pytorch.A2C as A2C
import models.expStrategy.epsilonGreedy as EPSG
import envs.cartPole as cartPole
import numpy as np

env = cartPole.CartPoleEnv()
NUM_STATE_FEATURES = env.get_num_state_features()
NUM_ACTIONS = env.get_num_actions()
EPISODE_NUM = 200
PRINT_EVERY_EPISODE = 20
LEARNING_RATE = 0.003
REWARD_DISCOUNT = 0.99

exp_stg = EPSG.EpsilonGreedy(0.2, NUM_ACTIONS)
agent = A2C.Agent(NUM_STATE_FEATURES, NUM_ACTIONS, REWARD_DISCOUNT, LEARNING_RATE, exp_stg)
print(agent.predict(np.array([[1, 1, 1, 1], [1, 1, 1, 1]])))
print(agent.select_action([1, 1, 1, 1]))

# print(agent.loss())