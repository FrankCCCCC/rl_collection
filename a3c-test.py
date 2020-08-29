from multiprocessing import Process, Lock
import tensorflow as tf
import numpy as np
import tensorflow.compat.v1 as tfv1
import models.A3C as A3C
import models.A2C as A2C
import models.expStrategy.epsilonGreedy as EPSG
import envs.cartPole as cartPole

def tf_test(id, cluster_spec, role, role_index):
    print(f'{role} {role_index} start')

    server = tf.distribute.Server(cluster_spec, role, role_index)
    if role == 'ps':
        # with tf.device("/job:ps/task:0"):
        server.join()
    elif role == 'worker':
        env = cartPole.CartPoleEnv()
        NUM_STATE_FEATURES = env.get_num_state_features()
        NUM_ACTIONS = env.get_num_actions()
        EPISODE_NUM = 1
        PRINT_EVERY_EPISODE = 20
        LEARNING_RATE = 0.003
        REWARD_DISCOUNT = 0.99
        coef_value = 1
        coef_entropy = 0
        data_type = tf.float32
        exp_stg = EPSG.EpsilonGreedy(0.2, NUM_ACTIONS)

        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth=True   
        with tfv1.Session(config=config).as_default() as sess:
            # tf.compat.v1.keras.backend.set_session(sess)
            # with tf.device("/job:localhost/replica:0/task:0/device:CPU:0"):
            with tf.device("/CPU:0"):
                # print("List Devices")
                # print(sess.list_devices())
                agent = A2C.Agent((NUM_STATE_FEATURES, ), NUM_ACTIONS, REWARD_DISCOUNT, LEARNING_RATE, exp_stg)
                state = env.reset()
                episode_reward, episode_loss, episode_gradients, trajectory = agent.train_on_env(env)
                for i in range(EPISODE_NUM):
                    episode_reward, loss, gradients, trajectory = agent.train_on_env(env = env, cal_gradient_vars = agent.model.trainable_variables)
                    agent.update(loss = loss, gradients = gradients)
                    print(f'Episode {i} Reward with process {id}: {episode_reward}')

def f(i, lock, cluster_spec, role, role_index):
    tf_test(i, cluster_spec, role, role_index)


if __name__ == '__main__':
    lock = Lock()
    process_num = 2
    cluster_spec = tf.train.ClusterSpec({
        'worker':[
            'localhost:8001',
            'localhost:8002',
        ],
        'ps': ['localhost:8000']
    })
    cluster_map = [
        # (0, 'ps', 0),
        (0, 'worker', 0),
        (1, 'worker', 1)
    ]
    workers = [Process(target=f, args=(num, lock, cluster_spec, role, idx)) for num, role, idx in cluster_map]
    for num in range(len(cluster_map)):
        workers[num].start()

    for num in range(len(cluster_map)):
        workers[num].join()
        print(f'Process {num} join')