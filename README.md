# MnistByTrace

采用简单的全连接结构，层间设置RELu激活函数

超参数:

```
train_batch_size = 128
lr = 2e-3
epochs = 20
optim = Adam
no warm up, no scheduled lr, no ddp.
test_batch_size = 8
```

结果:

![](https://i.hd-r.cn/ad50866e993c591c0a53c68feb5fdd4e.png)

![](https://i.hd-r.cn/31115e1395ae02555940282471d3d73b.png)

![8XETPKZAX7VFILG49X3f1f1c5d65d696a14.png](https://img.picgo.net/2023/04/12/8XETPKZAX7VFILG49X3f1f1c5d65d696a14.png)