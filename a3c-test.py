from multiprocessing import Process, Lock, Value, Array, Queue
import os
import ctypes
import tensorflow as tf
import numpy as np
import tensorflow.compat.v1 as tfv1
import models.A2C as A2C
import models.expStrategy.epsilonGreedy as EPSG
import envs.cartPole as cartPole

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
tf.config.set_soft_device_placement(True)

def init_agent_env():
    env = cartPole.CartPoleEnv()
    NUM_STATE_FEATURES = env.get_num_state_features()
    NUM_ACTIONS = env.get_num_actions()
    PRINT_EVERY_EPISODE = 20
    LEARNING_RATE = 0.003
    REWARD_DISCOUNT = 0.99
    coef_value = 1
    coef_entropy = 0
    exp_stg = EPSG.EpsilonGreedy(0.2, NUM_ACTIONS)
    agent = A2C.Agent((NUM_STATE_FEATURES, ), NUM_ACTIONS, REWARD_DISCOUNT, LEARNING_RATE, exp_stg)

    return agent, env

def make_cluster_config(ps_num, worker_num, host = 'localhost', base_port = 8000):
    cluster_config = {
        'worker': [],
        'ps': []
    }
    port = base_port

    for ps_id in range(ps_num):
        cluster_config['ps'].append(f'{host}:{port}')
        port += 1

    for worker_id in range(worker_num):
        cluster_config['worker'].append(f'{host}:{port}')
        port += 1
    
    return cluster_config

def get_res(global_res_queue, global_alive_workers):
    while ((not global_res_queue.empty()) or (global_alive_workers.value > 0)):
        res = global_res_queue.get()
        

def param_server(proc_id, ps_id, global_remain_episode, global_alive_workers, global_grad_queue, global_var_queues):
    # server = tf.distribute.Server(cluster_spec, 'ps', ps_id)

    # config = tf.compat.v1.ConfigProto()
    # config.gpu_options.allow_growth=True   
    # with tfv1.Session(target = server.target, config=config).as_default() as sess:
    # with tf.device("/job:localhost/replica:0/task:0/device:CPU:0"):

    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # tf.config.experimental.set_memory_growth(gpus[0], True)
    with tf.device("/CPU:0"):
        global_agent, env = init_agent_env()

        while ((not global_grad_queue.empty()) or (global_alive_workers.value > 0)):
#             print(f'Getting gradients from queue')
            item = global_grad_queue.get()
            global_agent.update(loss = item['loss'], gradients = item['gradients'])

            weights = global_agent.model.get_weights()
            for i in range(len(global_var_queues)):
                if not global_var_queues[i].full():
                    global_var_queues[i].put(weights)
#                     print(f'Put vars in queue for worker {i}')
        
        print("Complete PS apply")
        for queue in global_var_queues:
            if not queue.empty():
                queue.get()
                print(f'Clear vars in queue for worker')

def worker(proc_id, worker_id, global_remain_episode, global_alive_workers, global_grad_queue, global_var_queue):
    # server = tf.distribute.Server(cluster_spec, 'worker', worker_id)
    print(f'Process {proc_id} Worker {worker_id} start')

    # config = tf.compat.v1.ConfigProto()
    # config.gpu_options.allow_growth=True   
    # with tfv1.Session(target = server.target, config=config).as_default() as sess:
    # with tf.device("/job:localhost/replica:0/task:0/device:CPU:0"):

    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # tf.config.experimental.set_memory_growth(gpus[0], True)
    with tf.device("/CPU:0"):
        local_agent, local_env = init_agent_env()
        state = local_env.reset()
        while global_remain_episode.value > 0:
            episode_reward, loss, gradients, trajectory = local_agent.train_on_env(env = local_env, cal_gradient_vars = None)
            print(f'Episode {global_remain_episode.value} Reward with worker {worker_id}: {episode_reward}')

            # global_res_queue.put({'loss': loss, 'reward': episode_reward, 'worker_id': worker_id})
            global_grad_queue.put({'loss': loss, 'gradients': gradients, 'worker_id': worker_id})
            if not global_var_queue.empty():
                global_vars = global_var_queue.get()
                local_agent.model.set_weights(global_vars)
#                 print(f'Worker {worker_id} Update Weights')

            with global_remain_episode.get_lock():
                global_remain_episode.value -= 1

    with global_alive_workers.get_lock():
        global_alive_workers.value -= 1

    print(f"Worker {worker_id} exit")


if __name__ == '__main__':
    # print(tf.config.experimental.list_physical_devices(device_type=None))
    # print(tf.config.experimental.list_logical_devices(device_type=None))

    EPISODE_NUM = 300
    ps_num = 1
    worker_num = 2
    global_remain_episode = Value('i', EPISODE_NUM)
    global_alive_workers = Value('i', worker_num)
    # global_res_queue = Queue()
    global_grad_queue = Queue()
    global_var_queues = [Queue(1) for i in range(worker_num)]
    cluster_config = make_cluster_config(ps_num, worker_num)

    pss = []
    workers = []

    cluster_spec = tf.train.ClusterSpec(cluster_config)
    
    for ps_id in range(ps_num):
        pss.append(Process(target=param_server, args=(ps_id, ps_id, global_remain_episode, global_alive_workers, global_grad_queue, global_var_queues)))

    for worker_id in range(worker_num):
        workers.append(Process(target=worker, args=(worker_id + ps_num, worker_id, global_remain_episode, global_alive_workers, global_grad_queue, global_var_queues[worker_id])))

    for num in range(ps_num):
        pss[num].start()

    for num in range(worker_num):
        workers[num].start()
        
    global_grad_queue.close()
    global_grad_queue.join_thread()

    for queue in global_var_queues:
        queue.close()
        queue.join_thread()

    for num in range(worker_num):
        workers[num].join()
        print(f'Worker {num} join')

    for num in range(ps_num):
        pss[num].join()
        print(f'PS {num} join')