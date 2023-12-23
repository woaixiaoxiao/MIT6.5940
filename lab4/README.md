# **Question 1 (50 pts)**

## 前置

这里量化的思想时，如果输入很大，那么与它相乘的W就很重要（为什么不是输出很大，所以与它相乘的W就很重要？可能是因为输出很大的话，已经经过量化了，这一次已经来不及改了，要改也要等到下次训练）

所以对于每个W，计算了它的输入的每个特征的重要性，然后根据重要性，保留前百分之一的原始值，其他的都用量化值代替，示意图如下

![image-20231223191557541](http://woaixiaoxiao-image.oss-cn-beijing.aliyuncs.com/img/image-20231223191557541.png)

之前的代码已经提供了一个`input_feat`，这是一个字典，键为模块的name，值为模块的输入对应的属性，`input_feat[i][j][k]`代表

1. 第i个模块
2. 第j个输入
3. 第k个特征

因此若直接对`input_feat[i]`求和，就会得到一个长度为k的一维列表，代表了每个特征的重要性

## Question 1.1 (20 pts)

现在已经得到了输入的每个特征的重要性了，根据torch提供的topk可以求出前百分之一的下标，然后根据下标修改矩阵，然后在量化之后再恢复即可

```python
outlier_indices = torch.topk(importance,k=int(0.01*len(importance)))[1]
```

这里有个问题就是为什么是修改矩阵的列？

`outlier = m.weight.data[:, outlier_indices].clone()`

查看pytorch对linear的实现后才发现，它的实现很骚气

![image-20231223191551510](http://woaixiaoxiao-image.oss-cn-beijing.aliyuncs.com/img/image-20231223191551510.png)

## Question 1.2 (15 pts)

随机生成百分之一个数的下标和之前的方法对比

```python
outlier_mask = torch.randint(0, length, (length//100,), dtype=torch.int32)
```

## Question 1.3 (15 pts)

输入越重要，说明对应的矩阵也重要，所以尽量不要损失精度

# **Question 2 (50 pts)**

## Question 2.1 (20 pts)

代码和之前的一样

1. 先求出重要的矩阵参数
2. 将重要的矩阵参数先扩大scalar倍
3. 量化
4. 恢复重要的矩阵参数

## Question 2.2 (15 pts)

1：121.90

2：18.93

4：21.26

之前之所以会下降，是因为量化后策略生效了

之后会上升，是因为提升的幅度太大，导致最大值超过了原来的最大值，从而增加了误差

## Question 2.3 (15 pts)

这一部分关键在这个函数，这个函数的作用是尝试找出最合适的scalar值，就是for循环枚举可能的情况，找出一个最好的

1. 第一个参数为模块
2. 第二个参数是要拉伸的矩阵（KQV）
3. 第三个参数是输入

```python
def _search_module_scale(block, linears2scale: list, x, kwargs={})
```

但是要写代码的地方没太大难度，按照提示来

首先是初始化

```python
best_error = torch.inf
best_ratio = -1
best_scales = -1
```

然后是按照公式计算scalar

```python
scales = torch.clamp(s_x,1e-5)**ratio
```

最后是量化之后恢复重要的参数

```python
fc.weight.div_(scales)
```

这一大段代码的意思应该是

1. 先是为每个模块找到了一个最优的scalar
2. 然后好像还使用了另一种优化的方法，即将前面除以一个数，后面的乘上一个数，这样可以避免出现分布不均匀的情况，并且激活还不会改变。而在这里，直接将之前求得的scalar作为这个参数
