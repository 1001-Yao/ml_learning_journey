  1、Masked Self-attention
  
  与Self-attention的区别是，生成的时候不在参考后面的相关向量。
  比如生成b1,不去计算a1与a2、a3和a4的注意力分数，只看a1的kqv;
  生成b2时，只计算a2和a1的注意力机制，a2自己的注意力分数，不看后面的，依此类推
  ![[Pasted image 20260813144156.png|543]]

2、Non-Autoregressive Transformer(NAT非自回归)
并行一次生成所有结果词，而不是像传统模型那样逐个生成，从而实现大幅加速(并行)。

相比于AT:![[Pasted image 20260813150053.png]]

3、Encoder-Deconder
交叉注意力，其中两个箭头来源于encoder,一个箭头是deconder
![[Pasted image 20260813150213.png|612]]

工作机制：
将编码器的向量a1,a2,a3的k与解码器的向量的q,计算注意力分数，再经过softmax,和编码器的v相乘，并相加得到v,再fc
a1, a2, a3 就是编码器（Encoder）**最后一层**的输出向量序列
**q 的数量 = 当前解码器输入序列的长度**
![[Pasted image 20260813150518.png|527]]

后续生成，如上：
![[Pasted image 20260813151713.png|570]]

4、Training
（1）
目的是让交叉熵最小，Cross Entropy 就是用来==计算模型预测结果和标准答案之间的差距==
对于单个样本：
$$
CrossEntropy(p,q) = - \sum_{i}{p(i)\log q(i)}
$$
p(i)：真实分布（Ground Truth），也就是 独热向量（正确位置是 1，其余是 0）
q(i)：模型预测的概率分布（Softmax 的输出）

最后希望总和的cross entropy最小


![[Pasted image 20260813152946.png|557]]

（2）在训练的时候会给Decoder看正确答案（Teacher Forcing）：

![[Pasted image 20260813153925.png|573]]

5 Tips
(1) Copy Mechanism
(2)Guided Attention
(3)Beam Search
