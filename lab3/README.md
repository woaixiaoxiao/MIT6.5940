## Part 1. Predictors (3 Questions, 30 pts)

首先在之前，就获得了一个`ofa_network`

```python
ofa_network = OFAMCUNets(
    n_classes=2,
    bn_param=(0.1, 1e-3),
    ....,
)
ofa_network.load_state_dict()
```
在下面这个函数中展示了它的用法，说明ofa_network是一整个超网，cfg是config的缩写，

```python
def evaluate_sub_network(
    ofa_network, cfg, image_size=None)
```

通过下面两个函数可以根据cfg获取子网，然后就可以衡量并测试子网的性能

```python
ofa_network.set_active_subnet(**cfg)
subnet = ofa_network.get_active_subnet().to(device)
```

其中cfg可以通过下面这个函数得到，第一个函数可以传入随机的，也可以传入max或者min

```python
cfg = ofa_network.sample_active_subnet(
    sample_function=random.choice, 
    image_size=image_size)
```

### Question 2 (10 pts): Implement the efficiency predictor

```python
data_shape = (1, 3, spec["image_size"], spec["image_size"])
macs = count_net_flops(subnet, data_shape=data_shape)
peak_memory = count_peak_activation_size(
    subnet, data_shape=data_shape)
```

### Question 3 (10 pts): Implement the accuracy predictor

下面这个类可以将一个model变成vector，比如这个model的某个卷积核的大小可以为[4,5,6]，那么[0,1,0]代表选择卷积核为5

```python
arch_encoder = MCUNetArchEncoder()
```

准确率预测器是根据网络的结构预测准确率，在这里，它又三个linear组成

```rust
for i in range(self.n_layers):
    layers.append(
        nn.ReLU(
            nn.Linear(
                self.hidden_size if i!=0 else self.arch_encoder.n_dim,
                self.hidden_size,bias=False
            )
        )
    )
```

### Question 4 (10 pts): Complete the code for accuracy predictor training.

得到了准确率预测器之后，还需要根据要应用的数据来调整一下，这里就是常规的pytorch训练的流程.

```python
pred = acc_predictor(data)
loss = criterion(pred,label)
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

## Part 2. Neural Architecture Search (6 Questions, 65 pts + 10 bonus pts)

### Question 5 (5 pts): Complete the following random search agent

**RandomSearcher**类

1. `__init__`传入准确率预测器和效率预测器
2. `random_valid_sample`传入限制，随机得到一个模型，如果能够满足精度，则返回模型
3. `run_search`传入限制和数量n，得到n个满足进度和效率的模型，选取最好的那个

只需要补充

```python
best_idx = torch.argmax(accs)
```

### Question 6 (5 pts): Complete the following function

**search_and_measure_acc**函数

1. 传入随机查找的类，限制，以及其他参数
2. 根据以上参数得到一个随机得到的可以满足进度和效率限制的**配置**
3. 根据这个配置从超网中取出对应的网络，验证是否正确

具体的使用如下，通过`search_constraint = dict(millonMACs=millonMACs)`表明了设置限制，这里设置了macs

```rust
nas_agent = RandomSearcher(
    efficiency_predictor, acc_predictor)
subnets_rs_macs = {}
for millonMACs in [50, 100]:
    search_constraint = dict(millonMACs=millonMACs)
    print(f"Random search with constraint: MACs <= {millonMACs}M")
    subnets_rs_macs[millonMACs] = search_and_measure_acc(
        nas_agent, search_constraint, n_subnets=300)
```

至此可以理一下随机搜索的思路

1. 首先可以有准确率和效率预测器，根据这两个先得到一个可能的满足要求的配置
2. 根据这个配置从超网中取真正的网络，并验证

### Question7 (20 pts): Complete the following evolutionary search agent

```python
new_sample[key] = random.choice([sample1[key], sample2[key]])
new_sample[key][i] = random.choice(
    [sample1[key][i], sample2[key][i]])
population = sorted(population, 
    key=lambda x: x[0], reverse=True)
population = population[:parents_size]

```

后面两个question属于炼丹调参的范畴，但还是有一些思路可以遵循

1. `resolution_mutate_prob`很重要，应该是因为增加了管道，提取的信息更加丰富
2. `population_size`越大越好，因为这个代表了种群的基本盘，越多的话越可能出好的模型
3. `parent_ratio`越大也越好，依然是因为变异的群体变多了

## 总结

这个lab其实做的有点莫名其妙的，因为就是调用了已经写好了的`mcunet`，后面缕了一下思路

1. 发现所谓的搜索空间，就是先定好了一些超参数，比如卷积的通道数，卷积核的大小等等，然后提供了一些可选的值，通过这些参数的排列组合，有非常多种情况
2. 我们需要的是能满足精读和效率要求的组合，其中效率要求主要是macs和peak memory，也对应了衡量计算和存储两大部分（这两个不能完全代表latency，但有参考价值）
3. 为了更快地衡量进度，而不用重新推理一遍，建立了一个数据集，根据arch为键，acc为值。其中arch就是用vector的形式量化了model，具体来说，对于每个参数，可能有[a,b,c]三种取指，可以通过[0,1,0]表示取b
4. 最后有两种搜索的策略
    1. 随机搜索
    2. 仿生物进化的搜索
        1. 变异
        2. 融合















