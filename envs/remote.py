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
        res['is_done'] = self.bool_convert(res['is_done'])
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
        # return json.dumps(data).encode(encoding = self.encode_format)
        return json.dumps(data).encode()

    def decode(self, data):
        # Convert bytes in JSON format to dict object
        # return json.loads(data.decode(encoding = self.encode_format))
        return json.loads(data.decode())

    def bool_convert(self, str_bool):
        if str_bool == 'True':
            return True
        elif str_bool == 'False':
            return False

if __name__ == '__main__':
    renv = RemoteEnv(host = 'localhost', port = 50007, encode_format = 'UTF-8')
    print(f"Init State_Size: {renv.state_size} | Type: {type(renv.state_size)}")
    print(f"Init Action_Size: {renv.action_size} | Type: {type(renv.action_size)}")
    
    state = renv.reset()
    print(f"State {state} | Type: {type(state)}")
    state_prime, reward, is_done, info = renv.act(1)
    # print(f'State Prime: {state_prime} Reward: {reward} Is_done: {is_done} Info: {info}')

    print(f"State Prime: {state_prime} | Type: {type(state_prime)}")
    print(f"Reward: {reward} | Type: {type(reward)}")
    print(f"Is_done: {is_done} | Type: {type(is_done)}")
    print(f"Info: {info} | Type: {type(info)}")
