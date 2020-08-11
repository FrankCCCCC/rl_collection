from ple.games.flappybird import FlappyBird as flappyBird
from ple import PLE

class FlappyBirdEnv:
    def __init__(self):
        self.fps = 30
        self.game = flappyBird()
        self.env = PLE(self.game, fps=self.fps, display_screen=False)  # environment interface to game
        self.env.reset_game()

    def reset(self):
        self.env.reset_game()
    
    def act(self, action):
        return self.env.act(self.env.getActionSet()[action])

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
        return self.game.getGameState()

    def is_over(self):
        return self.env.game_over()
