from ple.games.flappybird import FlappyBird as flappyBird
from ple import PLE

class FlappyBirdEnv:
    def __init__(self):
        self.fps = 30
        self.game = flappyBird()
        self.env = PLE(self.game, fps=self.fps, display_screen=False)  # environment interface to game
        self.env.reset_game()

    def reset(self, is_show = False):
        self.env = PLE(self.game, fps=self.fps, display_screen=is_show)  # environment interface to game
        self.env.reset_game()
        state = self.get_state()

        return state
    
    def act(self, action):
        # return state_prime, reward, done, info
        reward = self.env.act(self.env.getActionSet()[action])
        state_prime = self.get_state()
        is_done = self.is_over()
        info = ""
        return state_prime, reward, is_done, info

    def get_num_actions(self):
        return len(self.env.getActionSet())

    def get_action_set(self):
        return self.env.getActionSet()

    def get_screen_rgb(self):
        return self.env.getScreenRGB()

    def get_screen_gray(self):
        return self.env.getScreenGrayscale()

    def get_num_state_features(self):
        return len(self.game.getGameState())

    def get_state(self):
        return list(self.game.getGameState().values())

    def is_over(self):
        return self.env.game_over()
