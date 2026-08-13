1、Seq2Seq
输入一个sequence,输出一个sequence，但输出的长度由模型自己决定
![[Pasted image 20260813112139.png]]
1.1 Encoder
给一排向量，输出一排向量
![[Pasted image 20260813112241.png|488]]

transformer的 Encoder架构：
![[Pasted image 20260813113451.png|547]]

详细来说：这是Block

![[5be3c429236d14e4b326fd5df61a40b6.png|560]]
 

1.2 Decoder

