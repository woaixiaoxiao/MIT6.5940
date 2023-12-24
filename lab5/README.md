# pre

```shell
(py310)tinychat-tutorial/transformer$ ./evaluate.sh reference
-------- Sanity check of reference implementation: Passed! -------- 
Section, Total time(ms), Average time(ms), Count, GOPs
reference, 2481.241943, 248.123993, 10, 1.056503

All tests completed!
```



# reference.cc

这个文件实现了一个矩阵乘法，并且由于为了在后面利用并行性和simd指令，看起来有点奇怪

首先，这里使用的量化策略为W4A8，因此一个uint8变量里其实存了两个权重，假设这个变量为w，那么这两个权重的值如下所示
$$
(w\&0x0F)-8\\
(w>>4)-8
$$

即先进行位运算，然后减8，之所以要减8是因为4bit量化的真实的权重值就应该是$[-8,7]$，但是因为c语言中没有对应的实现，所以在代码实现中就比较曲折

取出来矩阵权重之后，就可以和输入相乘，将累加的结果最后再乘上缩放比例即可

# Loop Unrolling (loop_unrolling.cc)

○   loop_unrolling

这里需要将最后一层循环手动展开，要对代码的理解更深一点

这里计算的是*A: m x k; B: n x k; C: m x n*，循环的顺序也是m,n,k，而在最初的reference的实现中，就将k这一层循环给拆成了以block_size为单位的循环，在x86部分的代码是，是拆分成了2*blocksize为单位的循环

而在每个2*blocksize中，w_int4是一个长度为32的数组，每一项都是两个4bit量化的权重，分别是[w0,w32]，[w1,w33].....，a_int8是一个长度为64的数组，对应了64个权重

在$k/(2*blocksize)$次循环中，计算出了acc，也就是数组C的对应元素

值得一提的是，这里矩阵计算的方法应该是$C=A*B^T$，其中B就是权重矩阵，因此对于C的每个元素，都是由A中代码某一行，B中的某一行决定的

在我们手动展开循环时，是以4为单位展开了C的某一行，因此，我们需要A的某一行，B的某4行，这也是这段代码的意思

```c++
const signed char *a_int8 = &A->int8_data_ptr[row * k + ch];
// pointer of the int4 weights
uint8_t *w0_int4 = &B->int4_data_ptr[(col * k + ch) / 2];
uint8_t *w1_int4 = &B->int4_data_ptr[((col + 1) * k + ch) / 2];
uint8_t *w2_int4 = &B->int4_data_ptr[((col + 2) * k + ch) / 2];
uint8_t *w3_int4 = &B->int4_data_ptr[((col + 3) * k + ch) / 2];
```

然后因为这里是以2*blocksize为单位的循环，所以对于每个w和a，都涉及两个scalar参数

不过自己只需要写下面这个代码

```c++
signed char Getpost(uint8_t a) { return (a >> 4) - 8; }
signed char Getpre(uint8_t a) { return (a & 0x0F) - 8; }
for (int qj = 0; qj < 32; qj++) {
    // TODO: decode a packed byte into two int8 in the range of (-8, 7)
    intermediate_sum0 += Getpre(w0_int4[qj]) * a_int8[qj];
    intermediate_sum0_2nd += Getpost(w0_int4[qj]) * a_int8[qj + 32];
    intermediate_sum1 += Getpre(w1_int4[qj]) * a_int8[qj];
    intermediate_sum1_2nd += Getpost(w1_int4[qj]) * a_int8[qj + 32];
    intermediate_sum2 += Getpre(w2_int4[qj]) * a_int8[qj];
    intermediate_sum2_2nd += Getpost(w2_int4[qj]) * a_int8[qj + 32];
    intermediate_sum3 += Getpre(w3_int4[qj]) * a_int8[qj];
    intermediate_sum3_2nd += Getpost(w3_int4[qj]) * a_int8[qj + 32];
    // TODO: int8 multiply and accumulate operation
}
```

如果不使用getpre和getpost函数，结果应该更好

```shell
(py310)tinychat-tutorial/transformer$ ./evaluate.sh loop_unrolling
-------- Sanity check of loop_unrolling implementation: Passed! -------- 
Section, Total time(ms), Average time(ms), Count, GOPs
loop_unrolling, 2114.875000, 211.487000, 10, 1.239525

All tests completed!
```



# Multithreading (multithreading.cc)

○   multithreading

这里的突破口在于要先看一下线程函数在做什么，可以发现，它收到了一个参数，包括了列的起始和末尾，以及paramas，因此可以判断，每个线程负责若干列

```c++
// TODO: Thread creation
for (int i = 0; i < num_thread; i++) {
    threads_args[i].start = n * i / num_thread;
    threads_args[i].end = n * (i + 1) / num_thread;
    threads_args[i].params = params;
    pthread_create(&thread_pool[i], nullptr, multithreading_worker_func, &threads_args[i]);
}

// TODO: Join threads
for (int i = 0; i < num_thread; i++) {
    pthread_join(thread_pool[i], nullptr);
}
```

相比于原始的代码，快了差不多四倍，因为就是用了4个线程跑

```shell
(py310)tinychat-tutorial/transformer$ ./evaluate.sh multithreading
-------- Sanity check of multithreading implementation: Passed! -------- 
Section, Total time(ms), Average time(ms), Count, GOPs
multithreading, 654.953979, 65.495003, 10, 4.002479

All tests completed!
```

# SIMD Programming      (simd_programming.cc)

○   simd_programming

这部分涉及到偏硬件的指令，但是可以把操作的东西看成一个很大而且很呆的数组即可

一些指令的解释

1. `_mm256_set1_epi8`，将每个字节都设置为传入的参数
2. `_mm256_loadu_si256`，将数据加载入内存
3. `_mm256_and_si256`，每个字节都和掩码相与
4. `_mm256_srli_epi16`，右移移位操作
5. `_mm256_sign_epi8`，第二个参数中每个位置的符号设置到第一个参数中对应的位置
6. `_mm256_maddubs_epi16`，对应每个8bit相乘，并且将相邻的两个16bit相乘结果累加
7. `_mm256_madd_epi16`，对每个对应的16bit相乘，并将两个相邻的16bit结果累加

这里的代码逻辑外面套两层循环对数组C的每一项进行遍历，对于每一项的计算，分成若干个2*blocksize大小的块进行，我们要写的代码就是在每一块里，利用simd指令加速

具体来说，加速的方法就是不从c语言层面循环累加，而是利用硬件指令帮我们一次性完成多个相乘累加的操作

第一步是需要将两个block中的权重分别取出来，代码如下所示，取出来的结果是在每一个字节的低4位都是一个权重

```c++
__m256i raw_w = _mm256_loadu_si256(w_start);
__m256i lowbit_w = _mm256_and_si256(raw_w, lowMask);
__m256i highbit_w = _mm256_and_si256(_mm256_srli_epi16(raw_w, 4), lowMask);
```

然后将这些权重减8，即恢复正确的零点

```c++
const __m256i zero_point = _mm256_set1_epi8(8);
__m256i w_0, w_128;
w_0 = _mm256_sub_epi8(lowbit_w, zero_point);
w_128 = _mm256_sub_epi8(highbit_w, zero_point);
```

现在就可以计算了，但是在正式计算之前，由于指令的限制，必须得先预处理一下，将第一个参数的所有值的符号都变成正数，第二个参数也相应的发生改变

最后补充乘法的操作即可

```c++
dot = _mm256_maddubs_epi16(ax, sy);
dot2 = _mm256_maddubs_epi16(ax2, sy2);
```

可以发现，随着不断地相乘以及相邻元素累加，256/8=32对w和a，逐渐变成了16和8，因此最后是这样写入结果的

```c++
 C->data_ptr[row * n + col] = ptr[0] + ptr[1] + ptr[2] + ptr[3] + ptr[4] + ptr[5] + ptr[6] + ptr[7];
```

性能提升了一倍多，

```shell
(py310)tinychat-tutorial/transformer$ ./evaluate.sh simd_programming
-------- Sanity check of simd_programming implementation: Passed! -------- 
Section, Total time(ms), Average time(ms), Count, GOPs
simd_programming, 1006.416992, 100.640999, 10, 2.604726

All tests completed!
```

# Multithreading with Loop Unrolling      (multithreading_loop_unrolling.cc)



○   multithreading_loop_unrolling

这题就是把前面两题缝合了一下，代码都不用改的

通过循环展开，速度比单纯的多线程确实要快一些

```c++
(py310)tinychat-tutorial/transformer$ ./evaluate.sh multithreading_loop_unrolling
-------- Sanity check of multithreading_loop_unrolling implementation: Passed! -------- 
Section, Total time(ms), Average time(ms), Count, GOPs
multithreading_loop_unrolling, 566.317017, 56.631001, 10, 4.628927

All tests completed!
```



# Combination of All Techniques      (all_techniques.cc)

○   all_techniques

最后一题也是究极缝合怪，不过确实用上了三种优化技术

1. 首先，将C的列分成8个部分给8个线程同时计算
2. 一个列依然是被还分成若干个2\*blocksize部分，而这里是将这若干个2\*blocksize给展开了，以每两个为单位进行展开。这里和之前的展开就不太一样了，之前的展开是同时展开这一列中的若干连续个元素
3. 最后就是使用SIMD进行硬件级别的加速

```shell
(py310)tinychat-tutorial/transformer$ ./evaluate.sh all_techniques
-------- Sanity check of all_techniques implementation: Passed! -------- 
Section, Total time(ms), Average time(ms), Count, GOPs
all_techniques, 146.906006, 14.690000, 10, 17.844336

All tests completed!
```

# 总结

额外的bonus没做，但是还是有优化的空间的，可以从

1. 内存体系，特别是利用cache，重新安排取内存的顺序，或者组织一下线程取数据的方式，应该都有机会
2. cuda还没用上

这次的lab封装的程度更大，背后应该是以这个完整的工具，整体的代码逻辑是

1. 先对输入进行量化，再计算矩阵的乘法，其中输入是8bit量化，矩阵参数是4bit量化，并且使用一种很风骚的方式交叉存储的
2. 在具体计算的时候，会使用公式恢复一下精度，其中零点和scalar都在量化的时候记录在了paramas变量中




