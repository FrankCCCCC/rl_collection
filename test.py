import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

strategy = tf.distribute.MirroredStrategy(['/cpu:0', '/cpu:1', '/cpu:2', '/cpu:3', '/cpu:4', '/cpu:5', '/cpu:6', '/cpu:7', '/cpu:8'])
print ('Number of devices: {}'.format(strategy.num_replicas_in_sync))