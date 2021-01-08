import pandas as pd
from IPython.display import display
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.pylab import figure

def test_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Restrict TensorFlow to only use the fourth GPU
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

class Recorder():
    def __init__(self, ckpt=None, ckpt_path=None, max_to_keep=250, plot_title=None, moving_avg_coef=0.04, filename=None, save_period=200):
        self.df = pd.DataFrame(columns=['epoch', 'loss', 'avg_loss', 'reward', 'avg_reward'])
        self.moving_avg_coef = moving_avg_coef
        self.avg_loss = 0
        self.avg_reward = 0
        self.plot_title = plot_title
        self.filename = filename
        self.n = 0
        self.save_period = save_period
        self.is_checkpoint = (ckpt != None) and (ckpt_path != None)
        if self.is_checkpoint:
            self.ckpt = ckpt
            self.ckpt_mang = tf.train.CheckpointManager(ckpt, ckpt_path, max_to_keep=max_to_keep)

    def record(self, loss, reward):
        new_avg_loss = (1 - self.moving_avg_coef) * self.avg_loss + self.moving_avg_coef * loss
        new_avg_reward = (1 - self.moving_avg_coef) * self.avg_reward + self.moving_avg_coef * reward
        self.avg_loss = new_avg_loss
        self.avg_reward = new_avg_reward
        
        self.df = self.df.append({'epoch': self.n, 'loss': loss, 'avg_loss': new_avg_loss, 'reward': reward, 'avg_reward': new_avg_reward}, ignore_index=True)

        if (self.filename and ((self.n % self.save_period == 0) or (self.n == 0))):
            # print({'epoch': self.n, 'loss': loss, 'avg_loss': new_avg_loss, 'reward': reward, 'avg_reward': new_avg_reward})
            # print(self.df)
            self.to_csv()
            self.to_plot()
            if self.is_checkpoint:
                self.ckpt_mang.save()

        self.n = self.n + 1
    
    def to_csv(self):
        self.df.to_csv(f"{self.filename}.csv")

    def to_plot(self):
        # Plot Reward History
        # figure(num=None, figsize=(24, 6), dpi=80)
        df = self.df.loc[self.df['epoch'] % 10 == 0]
        # df = self.df
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(24, 6), dpi=80)
        fig.suptitle(f'{self.plot_title}')

        ax1.plot(df['epoch'], df['reward'], color='blue', label='reward')
        ax1.plot(df['epoch'], df['avg_reward'], color='red', label='avg reward')
        ax1.set_xlabel('Episodes')
        ax1.set_ylabel('Reward / Episode')
        ax1.grid()
        ax1.legend()

        ax2.plot(df['epoch'], df['loss'], color='green', label='loss')
        ax2.plot(df['epoch'], df['avg_loss'], color='orange', label='avg loss')
        ax2.set_xlabel('Episodes')
        ax2.set_ylabel('Loss / Episode')
        ax2.set_yscale('log')
        ax2.grid()
        ax2.legend()

        plt.savefig(f"{self.filename}.svg")
        plt.savefig(f"{self.filename}.png")
        # plt.show()

    def restore(self):
        try:
            if self.is_checkpoint:
                latest = self.ckpt_mang.latest_checkpoint
                print(f"latest {latest}")
                if latest:
                    self.ckpt.restore(latest)
                    print(f"Recover from Checkpoint {latest}")
            df_old = pd.read_csv(f"{self.filename}.csv", index_col=0)
            self.df = df_old
            recover_ep = int(self.df['epoch'].iloc[-1])
            
            print(f"Recover from Record {recover_ep}")
            self.n = recover_ep + 1

            return recover_ep

        except:
            print("No CSV")
            return 0

    def display(self, head=None):
        if not head:
            display(self.df)
        else:
            display(self.df.head(head))

    def get_dataframe(self):
        return self.df