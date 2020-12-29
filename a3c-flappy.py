from multiprocessing import Process, Lock, Value, Array, Queue
import os
import ctypes
import tensorflow as tf
import numpy as np
import pandas as pd
import models.A2C as A2C
import envs.cartPole as cartPole
import envs.flappyBird as flappyBird
import models.util as Util

class A3C:
    def __init__(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        os.environ["SDL_AUDIODRIVER"] = "dummy"
        tf.config.set_soft_device_placement(True)

    def worker(self, proc_id, worker_id, global_remain_episode, global_alive_workers, global_grad_queue, global_var_queue, global_res_queue):
        print(f'Process {proc_id} Worker {worker_id} start')

        gpus = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
        # tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=10240)])
        with tf.device("/CPU:0"):
            local_agent, local_env = self.init_agent_env(proc_id, 'worker', worker_id)
            state = local_env.reset()
            while global_remain_episode.value > 0:
                episode_reward, loss, gradients, trajectory = local_agent.train_on_env(env = local_env, cal_gradient_vars = None)
                # print(f'Episode {global_remain_episode.value} Reward with worker {worker_id}: {episode_reward}')

                global_res_queue.put({'loss': loss, 'reward': episode_reward, 'worker_id': worker_id})
                global_grad_queue.put({'loss': loss, 'reward': episode_reward, 'gradients': gradients, 'worker_id': worker_id})
                if not global_var_queue.empty():
                    global_vars = global_var_queue.get()
                    local_agent.model.set_weights(global_vars)
    #                 print(f'Worker {worker_id} Update Weights')

                with global_remain_episode.get_lock():
                    global_remain_episode.value -= 1

        with global_alive_workers.get_lock():
            global_alive_workers.value -= 1

        print(f"Worker {worker_id} done")

    def param_server(self, proc_id, ps_id, global_remain_episode, global_alive_workers, global_grad_queue, global_var_queues):

        gpus = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)

        # tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=10240)])
        with tf.device("/CPU:0"):
            global_agent, env = self.init_agent_env(proc_id, 'ps', ps_id)

            # Setup recorder
            ckpt = tf.train.Checkpoint(model=global_agent.model, opt=global_agent.optimizer)
            recorder = Util.Recorder(ckpt=ckpt, ckpt_path='results/ckpt', plot_title='A3C FlappyBird', filename='results/a3c_flappy', save_period=5000)
            ep = recorder.restore()
            print(f"Restore {ep}")
            with global_remain_episode.get_lock():
                global_remain_episode.value = global_remain_episode.value - ep

            while ((not global_grad_queue.empty()) or (global_alive_workers.value > 0)):
                if not global_grad_queue.empty():
                    # print(f'Getting gradients from queue')
                    item = global_grad_queue.get()
                    global_agent.update(loss = item['loss'], gradients = item['gradients'])
                    recorder.record(float(item['loss']), float(item['reward']))

                    weights = global_agent.model.get_weights()
                    for i in range(len(global_var_queues)):
                        if not global_var_queues[i].full():
                            global_var_queues[i].put(weights)
                            # print(f'Put vars in queue for worker {i}')
            
            print("Complete PS apply")
            for queue in global_var_queues:
                if not queue.empty():
                    queue.get()
                    # print(f'Clear vars in queue for worker')
            print(f'PS {ps_id} done')

    def init_agent_env(self, proc_id, role, role_id):
        # env = cartPole.CartPoleEnv()
        env = flappyBird.FlappyBirdEnv()
        NUM_STATE_FEATURES = env.get_num_state_features()
        NUM_ACTIONS = env.get_num_actions()
        LEARNING_RATE = 0.0001
        REWARD_DISCOUNT = 0.99
        COEF_VALUE= 1
        COEF_ENTROPY = 0
        agent = A2C.Agent((NUM_STATE_FEATURES, ), NUM_ACTIONS, REWARD_DISCOUNT, LEARNING_RATE, COEF_VALUE, COEF_ENTROPY)

        return agent, env

    # def is_having_training_info(self):
    #     return ((not global_res_queue.empty()) or (global_alive_workers.value > 0))
    def get_res(self, global_res_queue, global_alive_workers):
        if ((not global_res_queue.empty()) or (global_alive_workers.value > 0)):
            return global_res_queue.get()
        else:
            return None

    def start(self):
        # print(tf.config.experimental.list_physical_devices(device_type=None))
        # print(tf.config.experimental.list_logical_devices(device_type=None))

        self.episode_num = 100000
        self.ps_num = 1
        self.worker_num = 20
        self.current_episode = 1
        global_remain_episode = Value('i', self.episode_num)
        global_alive_workers = Value('i', self.worker_num)
        global_res_queue = Queue()
        global_grad_queue = Queue()
        global_var_queues = [Queue(1) for i in range(self.worker_num)]

        pss = []
        workers = []
        episode_results = []
        
        for ps_id in range(self.ps_num):
            pss.append(Process(target = self.param_server, args=(ps_id, ps_id, global_remain_episode, global_alive_workers, global_grad_queue, global_var_queues)))

        for worker_id in range(self.worker_num):
            workers.append(Process(target = self.worker, args=(worker_id + self.ps_num, worker_id, global_remain_episode, global_alive_workers, global_grad_queue, global_var_queues[worker_id], global_res_queue)))

        for num in range(self.ps_num):
            pss[num].start()

        for num in range(self.worker_num):
            workers[num].start()

        while ((not global_res_queue.empty()) or (global_alive_workers.value > 0)):
            if not global_res_queue.empty():
                episode_results.append(global_res_queue.get())
                episode_res = episode_results.pop(0)
                print(f"Episode {self.current_episode} Reward with worker {episode_res['worker_id']}: {episode_res['reward']}\t| Loss: {episode_res['loss']}")
                self.current_episode += 1
            
        global_grad_queue.close()
        global_grad_queue.join_thread()

        global_res_queue.close()
        global_res_queue.join_thread()

        for queue in global_var_queues:
            queue.close()
            queue.join_thread()

        for num in range(self.worker_num):
            workers[num].join()
            print(f'Worker {num} join')

        for num in range(self.ps_num):
            pss[num].join()
            print(f'PS {num} join')

if __name__ == '__main__':
    # print(tf.config.experimental.list_physical_devices(device_type=None))
    # print(tf.config.experimental.list_logical_devices(device_type=None))

    a3c = A3C()
    a3c.start()