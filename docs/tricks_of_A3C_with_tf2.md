# Tricks of A3C on TensorFlow2 + Multiprocessing

> 當初寫太快，中英夾雜請見諒

## Introduction

在Tensorflow 2.0中，引入了動態圖的實作，取代1.x的靜態圖概念，同時Google移除了Session API，在TF 1.x中可以直接使用```tf.Session```來管理並分配硬體計算資源到任務上，但在2.0卻完全行不通；同時因為TF是以C實作，在開新Process的時候會導致無法Pickle現有的資源到新的Process去，正因為上述兩個原因，在TF 2.0 用Multiprocessing會遇到相當大的麻煩，網路上的範例幾乎清一色都是TF 1.x或Pytorch的實作方式。

在本文中，我們會著重在於A3C 在Tensorflow的實作上，尤其是各種奇怪的坑，如果要深究Actor-Critic和A3C的原理的話，推薦這幾篇：


- [Deriving Policy Gradients and Implementing REINFORCE](https://medium.com/@thechrisyoon/deriving-policy-gradients-and-implementing-reinforce-f887949bd63)
  
- [当我们在谈论 DRL：从AC、PG 到 A3C、DDPG](https://zhuanlan.zhihu.com/p/36506567)

## Problems

經過漫長踩坑之旅後，終於用TF 2.0 + Multiprocessing 實作出A3C，大致上可以整理出三個重點

1. Functional Process instead of Inherited Process Class
   
2. Use ```with tf.device()``` to specify the wanted device
   
3. Limit the ```CUDA_VISIBLE_DEVICES```

接下來會細說各點

## Problem1: Functional Process instead of Inherited Process Class

眾所皆知，在Python要Spawn一個新的Process有兩種方式，一種是繼承```mutiprocessing.process```並修改```run()``` method，另一種是直接將Function傳入Process。如果使用繼承process的方式實作，會出現Cannot pickle的Error，網路上普遍的說法是因為Tensorflow底層是由C實作，所以很多物件無法轉換成Python的binary pickle檔，所以才會出現此錯誤。

## Problem2: Use ```with tf.device()``` to specify the wanted device

在很多TF 1.x的A3C實作，可以看到都用了```server = tf.train.server```這個API，然後在每個Worker的Session會用```tf.Session(target = server.target)```，給不同Worker指定不同的計算資源，但TF2.0移除了```tf.Session```，如果直接在新Process呼叫Tensorflow的API的話，就會出現Blas GEMM的Error，所以如果要只用TF 2.0的API指定計算資源的話，就可以用```tf.device()```完成。

另外，Blas GEMM的錯誤，可以用限制Tesorflow占用的GPU memory解決
```
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf_config.gpu_options.per_process_gpu_memory_fraction = 0.9
tf_config.allow_soft_placement = True
```

Reference:

[Github Issue: Eager Execution error: Blas GEMM launch failed #25403](https://github.com/tensorflow/tensorflow/issues/25403)

[keras 或 tensorflow 调用GPU报错：Blas GEMM launch failed](https://blog.csdn.net/Leo_Xu06/article/details/82023330)

## Problem3: Limit the ```CUDA_VISIBLE_DEVICES```

避免Run out of memory的問題，因為Tensorflow預設的執行方式會盡量Allocate所有能用的GPU記憶體來加快執行速度，如果同時間又有其他任務占用該GPU，就會導致TF沒辦法Allocate足夠資源導致錯誤，所以在有多個GPU的共用機器上，最好就直接指定一個沒有使用的GPU來使用。但如果只有一個GPU且沒有其他任務占用該GPU的話，一般來說不用設定。

## Run The Demo

Talk is less, show me the code

[Tensorflow2.0 + Multiprocessing DEMO on Cart Pole](../a3c-test.py)