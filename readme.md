# DRL Algos Collection

### A colleciotn of implements of classical DRL algorithms.

The repository contain the implementation of A3C, A2C, DDQN, and REINFORCE(naive) with Tensorflow2.0. Some of them have been demostrated in the OpenAI Cart Pole environment.

In additon, it modulizes the API of environments(Cart Pole, Flappy Bird, and remote environment), exploration stratgies(although I still working on it). The remote environment even allows the agent to connect to the external server and interact with them.

---

## DRL Models

### A3C

  Still working on modulizing, but here is the DEMO on OpenAI Cart Pole. I use  Master-Slave strategy(which is similar to the parameter server strategy in TensorFlow1) with TensorFlow2.0 and Multiprocessing.py to implement. Worker send the updated gradients to the master, and the master receive the updated gradients and apply them to the global model. The master also keeps send the lastest model variables to the workers.

  However, TensorFlow2.0 has removed the ```tf.Session()``` which can allocate the computation task to specific device. Therefore, I use ```with tf.device()``` to specify the device of the task.

  For more detail, please read the doc: [Tricks of A3C on TensorFlow2 + Multiprocessing](./docs/tricks_of_A3C_with_tf2.md)

  and here is the DEMO

  [Tensorflow DEMO on Cart Pole](./a3c-test.py)

### A2C
  
  Implementation of Actor-Critic Network

  [Tensorflow DEMO on Cart Pole](./cartPole-A2C.py)
  
  [Tensorflow Code](./models/A2C.py)

### DDQN
  
  Implementation of Doubly Deep Q-Network with Tensorflow

  [Tensorflow Code](./models/DDQN.py)

### REINFORCE
  
  [Tensorflow Code](./models/REINFORCE.py)
  
### DDPG
  
  Working on it

### PPO

  Working on it

---

## Environments

Integrate the API of different environments.

### Cart Pole

From OpenAI Cart Pole.

[Python Code](./envs/cartPole.py)

### Flappy Bird

From PLE Flappy Bird

[Python Code](./envs/flappyBird.py)

### Remote Environment

Create a TCP client and connect to the provided server. You can see the DEMO and the details in [another repo(RL Java Integral)](https://github.com/Neural-Storage/RL_Java_Integral) which we implement a Java multi-threading server and interact with the A3C model. Thanks 
[tom1236868](https://github.com/tom1236868) for implementing the Java server.

[Python Code](./envs/remote.py)

---

### With LSTM

Reference:
- [Keras LSTM layer](https://keras.io/api/layers/recurrent_layers/lstm/)
- [Keras TimeDistributed layer](https://keras.io/api/layers/recurrent_layers/time_distributed/)
- [Github flyyufelix/VizDoom-Keras-RL](https://github.com/flyyufelix/VizDoom-Keras-RL/blob/master/a2c_lstm.py)

## Reference: 

- [当我们在谈论 DRL：从AC、PG 到 A3C、DDPG](https://zhuanlan.zhihu.com/p/36506567)

- [Deep Deterministic Policy Gradient (DDPG)](https://keras.io/examples/rl/ddpg_pendulum/)
  
- [Actor Critic Method](https://keras.io/examples/rl/actor_critic_cartpole/)

- [Deriving Policy Gradients and Implementing REINFORCE](https://medium.com/@thechrisyoon/deriving-policy-gradients-and-implementing-reinforce-f887949bd63)

- [Movan Python - Policy Gradient](https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/7_Policy_gradient_softmax/RL_brain.py)

- [Deep Reinforcement Learning: Playing CartPole through Asynchronous Advantage Actor Critic (A3C) with tf.keras and eager execution](https://blog.tensorflow.org/2018/07/deep-reinforcement-learning-keras-eager-execution.html)

- [flappy bird REINFORCE](https://github.com/GordonCai/project-deep-reinforcement-learning-with-policy-gradient/blob/master/Code/PG-Pong-Gordon-ANN-1.ipynb)
  
- [kevin-fang/reinforced-flappy-bird](https://github.com/kevin-fang/reinforced-flappy-bird/blob/master/tf_graph.py)
 
- [Tensorflow 2.0 Pitfalls](http://blog.ai.ovgu.de/posts/jens/2019/001_tf20_pitfalls/index.html)
- [分布式训练（理论篇）](https://zhuanlan.zhihu.com/p/129912419)
- [Ceruleanacg/Learning-Notes](https://github.com/Ceruleanacg/Learning-Notes)
- [重拾基础 - A3C & DPPO](https://zhuanlan.zhihu.com/p/38771094)
- [Day 23: Tensorflow 2.0: 再造訪 Distribute Strategy API](https://ithelp.ithome.com.tw/articles/10226066)
- [tf.distribute.cluster_resolver.ClusterResolver](https://www.tensorflow.org/api_docs/python/tf/distribute/cluster_resolver/ClusterResolver)
- [【04】tensorflow 到底該用 name scope 還是 variable scope](https://ithelp.ithome.com.tw/articles/10214789)
- [在Flask使用TensorFlow的几个常见错误](https://blog.csdn.net/qq_39564555/article/details/95475871)
- [keras 或 tensorflow 调用GPU报错：Blas GEMM launch failed](https://blog.csdn.net/Leo_Xu06/article/details/82023330)
- [How to run Keras.model() for prediction inside a tensorflow session?](https://stackoverflow.com/questions/50269901/how-to-run-keras-model-for-prediction-inside-a-tensorflow-session)
- [Day 23: Tensorflow 2.0: 再造訪 Distribute Strategy API](https://ithelp.ithome.com.tw/articles/10226066)
- [[第 27 天] 深度學習 TensorFlow](https://ithelp.ithome.com.tw/articles/10187702)
- [一句一句读Pytorch（更新中）](https://zhuanlan.zhihu.com/p/29916596)
- [并行处理最佳实践](https://pytorch.apachecn.org/docs/1.4/64.html)