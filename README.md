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

```
优化器：SGD
```

![](https://i.hd-r.cn/ad50866e993c591c0a53c68feb5fdd4e.png)

![](https://i.hd-r.cn/31115e1395ae02555940282471d3d73b.png)

![8XETPKZAX7VFILG49X3f1f1c5d65d696a14.png](https://img.picgo.net/2023/04/12/8XETPKZAX7VFILG49X3f1f1c5d65d696a14.png)

```
优化器：Adam
```

![H6IY_STRBUKX2QXMMQOb50b37e36da31362.png](https://img.picgo.net/2023/04/14/H6IY_STRBUKX2QXMMQOb50b37e36da31362.png)

![P8ZMB26A6R8MFVIRGNc64634d7782a17b1.png](https://img.picgo.net/2023/04/14/P8ZMB26A6R8MFVIRGNc64634d7782a17b1.png)

![OZ6TPWTKX6DI5WI07HPV702e2af46133b5df.png](https://img.picgo.net/2023/04/14/OZ6TPWTKX6DI5WI07HPV702e2af46133b5df.png)