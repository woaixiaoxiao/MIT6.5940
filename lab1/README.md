已定义的类和函数

```python
# 模型相关
# 模型的定义
class VGG(nn.Module)
# 模型训练函数
def train(
  model: nn.Module,
  dataloader: DataLoader,
  criterion: nn.Module,
  optimizer: Optimizer,
  scheduler: LambdaLR,
  callbacks = None
) -> None
# 模型评估函数
def evaluate(
  model: nn.Module,
  dataloader: DataLoader,
  verbose=True,
) -> float
```

```python
# 模型指标相关
# 模型的mac（乘法和加法）
def get_model_macs(model, inputs)
# 单个tensor的稀疏性
def get_sparsity(tensor: torch.Tensor)
# 模型的稀疏性
def get_model_sparsity(model: nn.Module)
# 模型的参数数量
def get_num_parameters(model: nn.Module, 
            count_nonzero_only=False)
# 模型的size
def get_model_size(model: nn.Module, 
       data_width=32, count_nonzero_only=False)
```

```python
# 测试细粒度剪枝的效果
def test_fine_grained_prune
```

# Fine-grained Pruning

## Magnitude-based Pruning

1. 得到稀疏率
2. 根据稀疏率找到最小的第k个值（threshold）
3. 将小于这个值的都置为0（通过一个mask矩阵完成）

![image-20231203143614877](http://woaixiaoxiao-image.oss-cn-beijing.aliyuncs.com/img/image-20231203143614877.png)

### Question 2 (15 pts)

照着hints翻译代码即可，需要注意的是

1. `kthvalue`函数是默认对指定dim或者最后一维操作的，因此对于二维tensor，需要先展平，再求第k小
2. `kthvalue`函数返回的是一个类，我们只需要这个类中的value

```python
##################### YOUR CODE STARTS HERE #####################
# Step 1: calculate the #zeros (please use round())
num_zeros = round(num_elements*sparsity)
# Step 2: calculate the importance of weight
importance = torch.abs(tensor)
# Step 3: calculate the pruning threshold
threshold = torch.kthvalue(importance.view(-1),num_zeros).values
# Step 4: get binary mask (1 for nonzeros, 0 for zeros)
mask = torch.gt(importance,threshold)
##################### YOUR CODE ENDS HERE #######################
```

### Question 3 (5 pts)

留下十个非零的数，总共25个数，即稀疏率要达到15/25，即0.6

然后包装了一个类`FineGrainedPruner`

1. 在`__init__`中调用了`prune`，将这个模型的所有全连接层和卷积层的MASK都得到
2. 以后每次调用`apply`，都直接用之前得到的MASK处理模型

这样相当于就固定了MASK，每次直接用就行，不需要重新计算

## Sensitivity Scan

稀疏度和剪枝后的准确率

![image-20231203153632677](http://woaixiaoxiao-image.oss-cn-beijing.aliyuncs.com/img/image-20231203153632677.png)

参数分布

![image-20231203153659267](http://woaixiaoxiao-image.oss-cn-beijing.aliyuncs.com/img/image-20231203153659267.png)

### Question 5 (10 pts)

25%的大小，微调之后达到%92.5的准确率，其中微调的代码已经写好了

因此，在这里只需要调参，使得模型的大小位于%25以下。做法就是

1. 尽量调参数比较大的层
2. 尽量调靠后的层
3. 不要对着一个层拼命地薅

通过这样调，让准确率降到了87.66%，但是在微调之后，回到了%92.5以上，因此有着百分之五的提升空间

```python
    'backbone.conv3.weight': 0.6,
    'backbone.conv4.weight': 0.7,
    'backbone.conv5.weight': 0.8,
    'backbone.conv6.weight': 0.8,
    'backbone.conv7.weight': 0.9,
```

微调的代码就是重新训练了几轮

# Channel Pruning

## Remove Channel Weights

### Question 6 (10 pts)

#### 简单剪枝（直接保留前若干个）

在这里只需要将输出通道中的前new个保留即可

第一步是在这个函数`get_num_channels_to_keep`中根据比例和现在的通道数计算保留下来的通道数，很简单，但是要注意对于乘法的结果要round，而不能用代码提供的int，大坑！

第二步是添加一行代码，使得卷积前后的通道数匹配地上，即前一个卷积操作的输出通道，是后一个卷积操作的输入通道，而在pytorch中卷积核是以$c_{out},c_{int},h,w$排布的，因此只需要这一行代码即可

```python
next_conv.weight.set_(next_conv.weight.detach()[:,:n_keep])
```

#### Ranking Channels by Importance

这一步操作其实并不是直接对channel按importance排序后剪裁，而只是将现有的conv层都按管道的重要性排个序就行了

首先在`get_input_channel_importance`得到某个conv层的输入管道的importance，只需要一行代码`importance = torch.norm(channel_weight)`

然后模仿着对pre_conv的处理加个代码即可

```python
next_conv.weight.copy_(torch.index_select(
    next_conv.weight.detach(),1,sort_idx
))
```

这里实验给的代码用了一个很骚的操作，对于每个conv

1. 先得到这个conv所有输入层的importance

2. 对这个importance排序，但是不直接改变数组（这里是tensor的数组），而是得到一个index数组，index[i]记录着第i重要的（i越小越重要），得到这样的一个效果

    ![image-20231203172836708](http://woaixiaoxiao-image.oss-cn-beijing.aliyuncs.com/img/image-20231203172836708.png)

3. 然后根据这个index数组将原来的conv的参数重新排序后赋值

### Measure acceleration from pruning

相比于fine-grained的剪枝，基于channel的可以直接带来模型size的下降和计算速度的提升，因为实打实地减少了很多channel，节省了计算量









