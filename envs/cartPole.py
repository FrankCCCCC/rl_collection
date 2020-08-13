import gym
import matplotlib as plt

class CartPoleEnv:
    def __init__(self):
        self.fps = 30
        self.env = gym.make('CartPole-v0')  # environment interface to game
        self.env.seed(42)
        self.env.reset()
        self.is_show = False
        self.current_sate = []
        self.is_done = False

    def render(self):
        self.env.render()
        
    def reset(self, is_show = False):
        state = self.env.reset()
        state = state.tolist()
        self.current_sate = state
        self.is_done = False
        self.is_show = is_show
        if self.is_show:
            self.env.render()

        return state
    
    def act(self, action):
        if self.is_show:
            self.env.render()
        # return observation, reward, done, info
        state, reward, is_done, info = self.env.step(self.get_action_set()[action])
        state = state.tolist()
        self.current_sate = state
        self.is_done = is_done
        return state, reward, is_done, info

    def get_num_actions(self):
        return 2

    def get_action_set(self):
        return [0, 1]

    def get_screen_rgb(self):
        pass

    def get_screen_gray(self):
        pass

    def get_num_state_features(self):
        return 4

    def get_state(self):
        return self.current_sate

    def is_over(self):
        return self.is_done
