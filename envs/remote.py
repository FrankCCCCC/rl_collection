import socket

class Remote:
    def __init__(self, host = '127.0.0.1', port = 10011):
        self.is_show = False
        self.current_sate = []
        self.is_done = False
        
        self.host = host
        self.port = port
        self.socket = socket.socket()
        self.socket.connect((host, port))
        self.reset()

        init_msg = self.socket.recv()
        init_msg = init_msg.split(',')
        self.state_size = init_msg[0]
        self.action_size = init_msg[1]
        self.action_set = [i for i in range(self.action_size)]

    def __del__(self):
        self.socket.close()

    def render(self):
        pass
        
    def reset(self, is_show = False):
        self.socket.send('reset')
        res = self.socket.recv()
        state = state.split(',')
        self.current_sate = state

        self.is_done = False
        self.is_show = is_show
        if self.is_show:
            self.render()

        return state
    
    def act(self, action):
        if self.is_show:
            self.render()
        # return observation, reward, done, info
        self.socket.send(self.get_action_set()[action])
        res = self.socket.recv()
        state, reward, is_done, info = res.split(';')
        state = state.split(',')

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
