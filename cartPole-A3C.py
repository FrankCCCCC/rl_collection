# # CartPole-A3C Experiment
# # 2020/08/11 SYC 

# import models.A2C as A2C
# import models.expStrategy.epsilonGreedy as EPSG
# import envs.cartPole as cartPole
# import parallel.Parallel as Parallel
# import models.util as Util
# import logging
# import matplotlib.pyplot as plt
# from matplotlib.pylab import figure
# import numpy as np
# # To run tqdm on notebook, import tqdm.notebook
# # from tqdm.notebook import tqdm
# # Run on pure python
# from tqdm import tqdm

# # Config Logging format
# # logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
# # Config logging module to enable on notebook
# # logger = logging.getLogger()
# # logger.setLevel(logging.DEBUG)

# # Test GPU and show the available logical & physical GPUs
# Util.test_gpu()

# env = cartPole.CartPoleEnv()
# NUM_STATE_FEATURES = env.get_num_state_features()
# NUM_ACTIONS = env.get_num_actions()
# EPISODE_NUM = 2000
# PRINT_EVERY_EPISODE = 20
# LEARNING_RATE = 0.003
# REWARD_DISCOUNT = 0.99

# exp_stg = EPSG.EpsilonGreedy(0.2, NUM_ACTIONS)
# # agent = Agent((NUM_STATE_FEATURES, ), NUM_ACTIONS, REWARD_DISCOUNT, LEARNING_RATE, exp_stg)

# agent_params = ((NUM_STATE_FEATURES, ), NUM_ACTIONS, REWARD_DISCOUNT, LEARNING_RATE, exp_stg)

# init_local_agent_funct = lambda: A2C.Agent((NUM_STATE_FEATURES, ), NUM_ACTIONS, REWARD_DISCOUNT, LEARNING_RATE, exp_stg)
# init_local_env_funct = lambda: cartPole.CartPoleEnv()

# master = Parallel.Master(EPISODE_NUM, init_local_agent_funct, init_local_env_funct, -1)
# master.start_workers()

# state = env.reset()
# accum_reward = 0
# accum_loss = 0

# # tqdm progress bar
# bar = []
# # Reward & LossHistory
# r_his = []
# avg_r_his = [0]
# loss_his = []
# episode_reward = 0

# print("Episode 1")
# for episode in range(1, EPISODE_NUM + 1):
#     if episode % PRINT_EVERY_EPISODE == 1:
#         if episode > 1:
#             bar.close()
#             print("Avgerage Accumulated Reward: {} | Loss: {}".format(round(accum_reward / PRINT_EVERY_EPISODE), (accum_loss / PRINT_EVERY_EPISODE)))
#             print("Episode {}".format(episode))
# #             agent.reset_metrics_loss()

#             avg_r_his.append(round(accum_reward / PRINT_EVERY_EPISODE))
#             accum_reward = 0
#             accum_loss = 0
#         bar = tqdm(total = PRINT_EVERY_EPISODE)

# #     episode_reward, episode_loss = agent.train_on_env(env)
#     curr_epi, episode_reward, episode_loss, worker_id = master.get_training_info()
#     accum_reward += episode_reward
#     accum_loss += episode_loss
#     r_his.append(episode_reward)
#     loss_his.append(episode_loss)
    
#     episode_reward = 0
#     episode_loss = 0

#     bar.update(1)        
#     env.reset()

# bar.close()    
# print("Accumulated Reward: {} | Loss: {}".format(round(accum_reward / PRINT_EVERY_EPISODE), (accum_loss / PRINT_EVERY_EPISODE)))
# avg_r_his.append(round(accum_reward / PRINT_EVERY_EPISODE))
# agent.reset_metrics_loss()

# # Evaluate the model
# agent.shutdown_explore()
# # agent.reset_metrics_loss()
# Worker.global_agent.shutdown_explore()
# # Reset Game
# env_state = env.reset()
# accum_reward = 0

# while not env.is_over():
#     # env.render()
# #     action, act_log_prob, value = agent.select_action(state)
#     action, act_log_prob, value = Worker.global_agent.select_action(state)
#     state_prime, reward, is_done, info = env.act(action)

#     state = state_prime
#     accum_reward += reward

# print("Evaluate")
# print("Accumulated Reward: {}".format(accum_reward))

# # Plot Reward History
# # figure(num=None, figsize=(24, 6), dpi=80)
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(24, 6), dpi=80)
# fig.suptitle(f'CartPole A3C Result (Evaluate Reward: {accum_reward})')
# x_datas = range(0, len(r_his))
# avg_x_datas = range(0, EPISODE_NUM + 1, PRINT_EVERY_EPISODE)

# ax1.plot(x_datas, r_his, color='blue')
# ax1.plot(avg_x_datas, avg_r_his, color='red')
# ax1.set_xlabel('Episodes')
# ax1.set_ylabel('Reward / Episode')
# ax1.grid()

# ax2.plot(x_datas, loss_his, color='orange')
# ax2.set_xlabel('Episodes')
# ax2.set_ylabel('Loss / Episode')
# ax2.grid()

# plt.savefig('CartPole-A3C-res.svg')
# plt.show()

# import models.A2C as A2C
# import models.expStrategy.epsilonGreedy as EPSG
import envs.cartPole as cartPole
import multiprocessing
import ctypes
# import tensorflow as tf

class Worker(multiprocessing.Process):
    def __init__(self, worker_id, remain_episode_num, lock, episode_result_queue, master_params, init_local_agent_funct, init_local_env_funct):
        multiprocessing.Process.__init__(self)
        import tensorflow as tf
        import numpy as np
        import models.A2C as A2C
        import models.expStrategy.epsilonGreedy as EPSG
        tf.compat.v1.keras.backend.set_session(tf.compat.v1.keras.backend.get_session)
        
        self.local_agent = init_local_agent_funct()
        self.local_env = init_local_env_funct()
        
        self.worker_id = worker_id
        self.remain_episode_num = remain_episode_num
        self.lock = lock
        self.episode_result_queue = episode_result_queue
        self.master_params = master_params
        
    def run(self):
#         import tensorflow as tf
#         import numpy as np
        
        print(f'Worker {self.worker_id} is running')
#         print(f"Worker {self.worker_id} Env: ", self.local_env)
#         print(f"Worker {self.worker_id} Agent: ", self.local_agent)
        print(self.local_agent.state_size)
        print(self.local_agent.num_action)
        self.local_agent.shutdown_explore()
        print(self.local_agent.get_metrics_loss())
        
        while self.remain_episode_num.value > 0:
            self.local_env.reset()
#             episode_reward, episode_loss, episode_gradients, trajectory = self.local_agent.train_on_env(self.local_env)
            
            self.lock.acquire()
            if self.remain_episode_num.value > 0:
                self.remain_episode_num.value -= 1
                print(f'Worker {self.worker_id} putting data into queue')
            self.lock.release()
                    
#             self.local_agent.update(episode_loss, episode_gradients)
# #             self.local_agent.model.set_weights(Worker.global_agent.model.get_weights())
            
# #             with Worker.lock:
#             if self.remain_episode.value > 0:
#                 print('KK')
#                 self.lock.acquire()
#                 self.remain_episode_num.value -= 1
#                 print(f'Worker {self.worker_id} putting data into queue')
#                 self.episode_result_queue.put({'worker_id': self.worker_id,'reward': episode_reward, 'loss': episode_loss, 'gradients': episode_gradients, 'trajectory': trajectory})
#                 self.lock.release()
                
        
class Master:
    def __init__(self, episode_num, init_local_agent_funct, init_local_env_funct, worker_num = -1):
        self.global_agent = init_local_agent_funct()
        self.remain_episode_num = multiprocessing.Value('i', episode_num)
        
        self.global_lock = multiprocessing.Lock()
        self.episode_result_queue = multiprocessing.Queue()
        self.master_params = multiprocessing.Array(ctypes.c_double, 4)
        
        self.curent_episode = 0
        self.episode_num = episode_num
        self.workers = []
        
        if worker_num == -1:
            self.worker_num = multiprocessing.cpu_count()
        else:
            self.worker_num = worker_num
            
        for i in range(self.worker_num):
#             agent = init_local_agent_funct()
#             env = init_local_env_funct()
            self.workers.append(Worker(i, self.remain_episode_num, self.global_lock, self.episode_result_queue, self.master_params, init_local_agent_funct, init_local_env_funct))
            
    def start_workers(self):
        for worker, i in zip(self.workers, range(self.worker_num)):
            worker.start()
            print(f'Start Worker {i}')
            
    def join_workers(self):
        for worker, i in zip(self.workers, range(self.worker_num)):
            worker.join()
            print(f'Join Worker {i}')
            
    def get_training_info(self):
#         if Worker.remain_episode_num > 0 or not Worker.global_queue.empty():
        for i in range(self.episode_num):
            self.curent_episode += 1
            episode_info = self.episode_result_queue.get()
            epi_r = episode_info['reward']
            epi_l = episode_info['loss']
            epi_id = episode_info['worker_id']
            # print(f'Episode {self.curent_episode} | Episode Reward {epi_r} | Loss {epi_l} | Worker {epi_id}')
            return self.curent_episode, epi_r, epi_l, epi_id
        else:
            print('No training info')
            
    def is_having_training_info(self):
        if Worker.remain_episode_num > 0 or not Worker.global_queue.empty():
            return True
        else:
            return False

if __name__ == '__main__':   
    def la():
        import models.A2C as A2C
        import models.expStrategy.epsilonGreedy as EPSG
        import envs.cartPole as cartPole
        import multiprocessing
        import ctypes
        import tensorflow as tf
        
        NUM_STATE_FEATURES = env.get_num_state_features()
        NUM_ACTIONS = env.get_num_actions()
        EPISODE_NUM = 6
        PRINT_EVERY_EPISODE = 20
        LEARNING_RATE = 0.003
        REWARD_DISCOUNT = 0.99

        exp_stg = EPSG.EpsilonGreedy(0.2, NUM_ACTIONS)
        
        return A2C.Agent((NUM_STATE_FEATURES, ), NUM_ACTIONS, REWARD_DISCOUNT, LEARNING_RATE, exp_stg)

    def le():
        import envs.cartPole as cartPole
        return cartPole.CartPoleEnv()

    init_local_agent_funct = la
    init_local_env_funct = le
            
    env = cartPole.CartPoleEnv()
    NUM_STATE_FEATURES = env.get_num_state_features()
    NUM_ACTIONS = env.get_num_actions()
    EPISODE_NUM = 6
    PRINT_EVERY_EPISODE = 20
    LEARNING_RATE = 0.003
    REWARD_DISCOUNT = 0.99

    # exp_stg = EPSG.EpsilonGreedy(0.2, NUM_ACTIONS)
    # agent_params = ((NUM_STATE_FEATURES, ), NUM_ACTIONS, REWARD_DISCOUNT, LEARNING_RATE, exp_stg)

    # init_local_agent_funct = lambda : Agent((NUM_STATE_FEATURES, ), NUM_ACTIONS, REWARD_DISCOUNT, LEARNING_RATE, exp_stg)
    # init_local_env_funct = lambda : CartPoleEnv()

    master = Master(EPISODE_NUM, init_local_agent_funct, init_local_env_funct, 1)
    master.start_workers()
    master.join_workers()
    print("Join All Workers")
    # for i in range(EPISODE_NUM):
    #     print(master.episode_result_queue.get())