import threading
import multiprocessing
import queue

class Worker(threading.Thread):
    global_queue = queue.Queue()
    remain_episode_num = 0
    global_agent = {}
    lock = threading.Lock()
    def __init__(self, worker_id, local_agent, local_env):
        threading.Thread.__init__(self)
        self.local_agent = local_agent
        self.env = local_env
        self.worker_id = worker_id
        
    def run(self):
        while Worker.remain_episode_num > 0:
            self.env.reset()
            episode_reward, loss = self.local_agent.train_on_env(env = self.env, is_show = False, cal_gradient_vars = self.local_agent.model.trainable_variables)
            self.local_agent.update(loss = loss, gradients = gradients, apply_gradient_vars = Worker.global_agent.model.trainable_variables)
            self.local_agent.model.set_weights(Worker.global_agent.model.get_weights())
            
            with Worker.lock:
                Worker.remain_episode_num -= 1
                Worker.global_queue.put({'episode_reward': episode_reward, 'loss': loss, 'worker_id': self.worker_id})

class Master:
    def __init__(self, episode_num, init_local_agent_funct, init_local_env_funct, worker_num = -1):
        Worker.global_agent = init_local_agent_funct()
        Worker.remain_episode_num = episode_num
        
        self.curent_episode = 0
        self.workers = []
        if worker_num == -1:
            self.worker_num = multiprocessing.cpu_count()
        else:
            self.worker_num = worker_num
            
        for i in range(self.worker_num):
            agent = init_local_agent_funct()
            env = init_local_env_funct()
            self.workers.append(Worker(i, agent, env))
            
    def start_workers(self):
        for worker, i in zip(self.workers, range(1, self.worker_num + 1)):
            worker.start()
            print(f'Start Worker {i}')
            
    def get_training_info(self):
        if Worker.remain_episode_num > 0 or not Worker.global_queue.empty():
            self.curent_episode += 1
            episode_info = Worker.global_queue.get()
            epi_r = episode_info['episode_reward']
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