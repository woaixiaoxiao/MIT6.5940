# K-Means Quantization

## Question 1 (10 pts)

![image-20231205113041534](http://woaixiaoxiao-image.oss-cn-beijing.aliyuncs.com/img/image-20231205113041534.png)

首先初始化了一个kmeans算法的对象，主要是传入了需要多少种

```python
kmeans = KMeans(n_clusters=n_clusters, mode='euclidean', verbose=0)
```

然后通过调用这个函数，传入了需要聚拢的数组，它应该就是会选择出n_clusters个值作为量化后的值，然后返回原来的数组中元素在新数组中的下标index

```python
labels = kmeans.fit_predict(fp32_tensor.view(-1, 1)).to(torch.long)
```

然后可以得到新的数组

```python
centroids = kmeans.centroids.to(torch.float).view(-1)
```

最后写一行代码就可以通过量化后的结果恢复到原来的数据

```python
quantized_tensor = codebook.centroids[codebook.labels].view_as(fp32_tensor)
```

## Question 2 (10 pts)

简答题

## K-Means Quantization on Whole Model

### KMeansQuantizer类

1. `__init__`输入一个model和bitwidth，调用了`quantize`得到这个模型的量化结果的字典，键为参数的名称，值为参数的codebook，即数据+index
2. `quantize`遍历了所有参数（矩阵，tensor），对它们进行量化，并返回结果的字典（输入的bitwidth可以是数字也可以是列表，即统一bitwidth还是每一层不同）
3. `apply`用于更新codebook或者得到量化后用于计算的结果

## Question 3 (10 pts)

完成QAT，即在k-means方法之上，对量化后的codebook进行反向传播。具体实现上，只需要将相关的参数求平均值即可。具体的实现上使用了python的语法糖

```python
codebook.centroids[k] = torch.mean(fp32_tensor[
    torch.where(codebook.labels==k)])
```

## 训练模型的整体代码

本质上和剪枝一样，每训练完一个batch，就会更新参数，在这里就是更新量化后的结果，通过给train传入一个callback函数实现

总的来说

1. 首先得到每一层的codebook
2. 在训练之前
    1. 对模型进行量化，将原来的参数都变成量化后的结果
    2. 虽然看起来量化之后还是那么多数量的float32，但其实因为Python的语法特性，这都是对某个tensor的引用，不额外占空间
3. 训练之后
    1. 根据反向传播优化之后的参数，更新codebook

# Linear Quantization

## Question 4

**linear_quantize**函数给出了要量化的矩阵，以及S和Z，因此只需要按照公式计算就行了

![image-20231205141558344](http://woaixiaoxiao-image.oss-cn-beijing.aliyuncs.com/img/image-20231205141558344.png)

## Question 5

对于S可以使用如下公式计算，还可以用KL散度

![image-20231205142203375](http://woaixiaoxiao-image.oss-cn-beijing.aliyuncs.com/img/image-20231205142203375.png)

Z可以用这个公式计算

![image-20231205142104670](http://woaixiaoxiao-image.oss-cn-beijing.aliyuncs.com/img/image-20231205142104670.png)

包装了一个函数，输入一个参数矩阵以及量化的bit，返回量化后的结果以及S和Z

```python
def linear_quantize_feature(
    fp_tensor, bitwidth)
```

## Special case: linear quantization on weight tensor

由于参数矩阵的数据特征往往是对称的，因此可以直接认为Z是0，由此可以得到

![image-20231205143149518](http://woaixiaoxiao-image.oss-cn-beijing.aliyuncs.com/img/image-20231205143149518.png)

### Per-channel Linear Quantization

1. 遍历每一个out_channel得到对应scale，z就直接为0

这中间需要用到一些tensor形状的骚操作

1. 可以通过`select(dim,index)`得到某个子tensor
2. 在遍历完之后得到的scale存在一个一维的tensor中，想要通过它直接得到整个tensor的量化结果，需要先将它的形状view成[-1,1,1,,1,1]，这样才能复用之前的函数，直接通过广播正确的将scale发送到正确的位置

## Quantized Inference

![image-20231205144802583](http://woaixiaoxiao-image.oss-cn-beijing.aliyuncs.com/img/image-20231205144802583.png)

### Question 6 (5 pts)

计算了$S_{bias}$的量化结果

`bias_scale = input_scale*weight_scale`

### Quantized Fully-Connected Layer

在`quantized_linear`中完成输出的scale操作

其中input和output的scale都是常数，而weight的scale的形状则是$[oc,1,1,1]$，而output的形状是$bs,oc$，因此需要将weight调成$1,oc$，这样在output和weight操作时，就可以自动广播成output的形状

然后就是在和浮点数操作的时候，需要将int转为float，这样应该是为了在计算的时候增加精度，因此需要将output的数据类型编程float

```python
output = output.to(float) * \
        (input_scale * (weight_scale).view(1, -1) \
         / output_scale)
```

#### Question 8 (15 pts)

和矩阵乘法基本一样，注意维度的转换即可

```python
output = output.to(float) * (
    weight_scale.view(1, -1, 1, 1) * input_scale / output_scale
)
```

## Question 9 (10 pts)

将0-1映射到-128-127，先乘256，再减128

```python
(x * 255 - 128).clamp(-128, 127).to(torch.int8)
```

# 总结

k-means方法

1. 只是用更少的float32来存储，但是真正要使用的时候，全是float32
2. 因此，兼容性更好，速度更快，但是因为只有几个值可以选，因此准确率较低

线性量化法

1. 将浮点数全部线性地变成整数，需要的时候又全部线性变回浮点数，其中S是float32，Z是int类型
2. 相比于k-means，可选的数更多，更加灵活。但是因为需要不停地进行计算（float-int-float），所以相对较慢。虽然可以通过公式化简直接使用int类型去计算，但是目前有些操作还不支持int，所以需要额外的硬件和软件支持



再凝练一点

普通的训练是，用float存，float算

k-means，用int存，用float算

线性量化，用int存，用int算（合并简化公式）





































