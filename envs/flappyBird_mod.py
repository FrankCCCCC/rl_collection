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

        # Survive reward +1, dead -1000
        if is_done:
            reward = -1000
        else:
            reward += 1

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
        # dict
        #     * player y position.
        #     * players velocity.
        #     * next pipe distance to player
        #     * next pipe top y position
        #     * next pipe bottom y position
        #     * next next pipe distance to player
        #     * next next pipe top y position
        #     * next next pipe bottom y position

        # state = {
        #     "player_y": self.player.pos_y,
        #     "player_vel": self.player.vel,
            
        #     "next_pipe_dist_to_player": next_pipe.x - self.player.pos_x,
        #     "next_pipe_top_y": next_pipe.gap_start,
        #     "next_pipe_bottom_y": next_pipe.gap_start+self.pipe_gap, 
            
        #     "next_next_pipe_dist_to_player": next_next_pipe.x - self.player.pos_x,
        #     "next_next_pipe_top_y": next_next_pipe.gap_start,
        #     "next_next_pipe_bottom_y": next_next_pipe.gap_start+self.pipe_gap 
        # }
        state = self.game.getGameState()
        state['next_pipe_top_y'] -= state['player_y']
        state['next_pipe_bottom_y'] -= state['player_y']
        state['next_next_pipe_top_y'] -= state['player_y']
        state['next_next_pipe_bottom_y'] -= state['player_y']
        return list(state.values())

    def is_over(self):
        return self.env.game_over()
