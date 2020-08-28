import socket
import json

class RemoteEnv:
    def __init__(self, host = '127.0.0.1', port = 50007, maxlen = 1024, encode_format = 'UTF-8'):
        self.is_show = False
        self.current_sate = []
        self.is_done = False
        self.maxlen = maxlen
        
        self.encode_format = encode_format
        self.host = host
        self.port = port
        self.socket = socket.socket()
        self.socket.connect((host, port))
        
        self.init()
        self.reset()

    def __del__(self):
        self.socket.close()

    def render(self):
        pass

    def init(self):
        init_msg = self.encode({'request': 'init'})
        self.socket.send(init_msg)
        init_res = self.decode(self.socket.recv(self.maxlen))
        print(init_res)

        self.state_size = init_res['state_size']
        self.action_size = init_res['action_size']
        self.action_set = [i for i in range(self.action_size)]
        
    def reset(self, is_show = False):
        msg = self.encode({'request': 'reset'})
        self.socket.send(msg)
        res =  self.decode(self.socket.recv(self.maxlen))
        self.current_sate = res['state']

        self.is_done = False
        self.is_show = is_show
        if self.is_show:
            self.render()

        return self.current_sate
    
    def act(self, action):
        if self.is_show:
            self.render()
        # return observation, reward, done, info
        act_msg = self.encode({'request': 'act', 'action': self.get_action_set()[action]})
        self.socket.send(act_msg)
        res = self.decode(self.socket.recv(self.maxlen))
        state, reward, is_done, info = res['state'], res['reward'], res['is_done'], res['info']

        self.current_sate = state
        self.is_done = is_done
        return state, reward, is_done, info

    def get_num_actions(self):
        return self.action_size

    def get_action_set(self):
        return self.action_set

    def get_screen_rgb(self):
        pass

    def get_screen_gray(self):
        pass

    def get_num_state_features(self):
        return self.state_size

    def get_state(self):
        return self.current_sate

    def is_over(self):
        return self.is_done

    def encode(self, data):
        # Convert dict object to bytes in JSON format
        return json.dumps(data).encode(encoding = self.encode_format)

    def decode(self, data):
        # Convert bytes in JSON format to dict object
        return json.loads(data.decode(encoding = self.encode_format))

if __name__ == '__main__':
    renv = RemoteEnv(host = 'localhost', port = 50007)
    state = renv.reset()
    print(f'State: {state}')
    state_prime, reward, is_done, info = renv.act(1)
    print(f'State Prime: {state_prime} Reward: {reward} Is_done: {is_done} Info: {info}')
    print(type(is_done))
