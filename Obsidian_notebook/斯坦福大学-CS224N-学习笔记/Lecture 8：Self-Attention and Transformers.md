1、传统RNN的局限性
线性交互距离问题：
	每个词与其相邻的词关系更近，但长距离则很难相关、交互
GPU的并行计算能力被RNN的串行结构浪费：
	因为RNN必须按时间步依次计算，所以无法利用GPU同时处理大量独立运算的优势，这直接阻碍了在超大规模数据集上的训练速度。

2、Why attention?
不依赖索引序列，同时可以并行计算
![[Pasted image 20260811105825.png]]

3、attention 工作机制
    在注意力机制中，Query 与所有 Key 逐一计算相似度分数，形成对“键空间”的软匹配（而非硬性索引）。该分数经 Softmax 归一化为权重分布后，用于对对应的 Value（词向量） 进行加权求和，得到最终的上下文向量输出。
![[Pasted image 20260811110425.png]]


**4、注意力机制公式**

w 是原始单词（one-hot 向量）
E 是词嵌入矩阵
x 是 E 与 w 相乘得到的词嵌入向量；

Q、K、V 就是可以学习的权重矩阵（参数矩阵），它们是模型通过训练数据自己学出来的,是 Self-Attention 里真正的“可学习参数。
分别把词向量x映射为‘查询’，‘键’，‘值’ 向量；

#### 关键第二步：计算两两相似度，并用 Softmax 归一化

> Compute pairwise similarities between keys and queries; normalize with softmax

$$

e_{ij} = q_i^T k_j

与

\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_j \exp(e_{ij})}

$$


#### 这一步在做什么？

- 对于第 \( i \) 个词，它用自己的 \( q_i \) 去和**所有词**（包括自己）的 \( k_j \) 计算点积
- 点积结果 \( e_{ij} \) 表示“词 i 对词 j 的关注程度”（未归一化）。也就是注意力分数
- 用 Softmax 将这一行分数压缩成概率分布  ，满足：

$$
\sum_j \alpha_{ij} = 1
$$
#### 结果
对于每个位置 \( i \)，得到它与其他所有位置之间的“注意力权重分布”。

最后做加权平均输出结果
![[Pasted image 20260811111843.png]]


5、自注意力机制作为基础模块的障碍与解决方法

障碍：
1）自注意力本身没有顺序的概念！
2）没有非线性（Nonlinearities），就无法实现深度学习！它本质上只是在做加权平均。
3）在预测序列时，需要确保模型不能“看到未来”

解决：
1）在输入中加入位置表示。
给每个输入向量硬生生加一个“身份证号”（位置向量）。这个号码携带了绝对或相对的位置信息。这样一来，Attention在计算相似度时，不仅看“词义”，还强制看“位置”，因此能区分谁在前谁在后

2）简单解决方法：对每个自注意力输出应用同一个前馈网络（Feedforward Network）。
在每个Attention输出后，接一个**标准的MLP（多层感知机）**。MLP里带有ReLU等**非线性激活函数**。让模型真正具备深度学习的拟合能力。

3）通过将未来位置的注意力权重人为设为 0，实现掩码（Mask）。
在计算Softmax之前，把**未来位置**的Attention分数**强行设为负无穷**（`-inf`）。负无穷经过Softmax后变成**绝对的0**。强迫模型只能根据前面的词进行预测。
![[Pasted image 20260811120002.png]]

6、Self-Attention模块

输入（Inputs）
    ↓
词嵌入（Embeddings）
    ↓
【重复多次：编码器块（Encoder Block）】
    掩码自注意力（Masked Self-Attention）
        ↓
    前馈网络（Feed-Forward）
        ↓
    线性层（Linear）
        ↓
    Softmax
        ↓
概率（Probabilities）
但这个架构在现实中使用率极低。
![[Pasted image 20260811120307.png]]

7、Transformer Decoder

7.1、Tansformer Decoder 总览
![[Pasted image 20260811152144.png]]
7.2、Multi-Head Attention
多头注意力机制，可以简单理解为先用一组QKV来跑一次，再换另外一组QKV跑
![[Pasted image 20260811144127.png]]
单头注意力机制局限性：
	1）对于每一个词，sa只能算出一个注意力分布，只能聚焦在一个模式上
	2）但一个词可能有多种关注需求
多头的动机：
	用多组不同的 Q,K,V 矩阵，让模型能够同时从多个角度关注不同位置

解释参数：
	d：词嵌入维度
	h: 头的数量
	i =1,...h:第i个头
实现：
	（1）定义多个头的投影矩阵，每个头都有自己的Q,K,V 矩阵，维度是 d×(d/h)，把 d 维空间压缩到d/h 维
	（2）每个头独立计算。即output_i,每个头都得到一个 n×(d/h) 的输出矩阵，表示该头对序列的“观察结果"
	(3)拼接所有头输出。按最后一维拼接：n×(h×d/h)=n×d。Y是一个可学习的线性变换矩阵，形状为 (d, d)。乘以 Y 是一个线性变换，让模型能够学习如何融合多头之间的信息，把各头独立捕获到的信息整合成一个统一的输出。
	
图示：
![[Pasted image 20260811150745.png]]



7.3、Sequcence-Stacked from Attention
将词从单独向量堆叠为大矩阵来处理 ，从而得到整个大矩阵的注意力分数
![[Pasted image 20260811143936.png]]


7.4、缩放点积注意力
![[Pasted image 20260811151204.png]]
![[Pasted image 20260811151221.png]]

7.5、残差连接和层归一化
![[Pasted image 20260811152059.png]]

残差连接：
![[Pasted image 20260811152423.png]]

层归一化：
归一化的是,每一个独立词向量内部的“各个维度”
![[Pasted image 20260811151804.png|599]]

8、Transfomer Enconder
与Transformer deconder的区别就是在多头注意力机制 ==没有==**掩码mask**
![[Pasted image 20260811153127.png]]

9、Transformer Encoder-Decoder

![[Pasted image 20260811153348.png]]
多个输入：
	Decoder Inputs 给模型“回忆”自己翻译到了哪里；来自 Encoder 的输入给模型“指引”该怎么翻译原文。两者缺一不可，共同配合完成准确的机器翻译

交叉注意力：
	**把解码器向量作为查询，再把编码器的输出作 键和值**，所以解码器的每一个值都会关注 编码器所有块输出中的全部词汇

![[Pasted image 20260811154201.png]]