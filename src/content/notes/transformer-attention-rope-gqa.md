---
title: 从一个token理解Transformer与多模态入口
date: '2026-09-22'
tags:
- 多模态算法
- Attention
- RoPE
- GQA
- RMSNorm
- 面试
summary: 手算Attention，推导RoPE，理解GQA、RMSNorm、SwiGLU与视觉token。
draft: false
obsidian: true
series: multimodal-interview
---
<span id="mm-38507a391c6e" style="display:block;scroll-margin-top:6rem"></span>

本章解释“模型怎样把已经看见的内容变成下一步预测”。读完应能在纸上完成一次两 token 注意力计算，画出 Q/K/V 的形状，解释 RoPE、GQA、RMSNorm 和 SwiGLU 各自改变哪一步。这对应 ZealD 原帖明确披露的 QKV、位置编码题；题源及其证据范围见[05-ZealD真实面试题与项目深挖](/notes/multimodal-interview-questions/)。

如果点积、矩阵乘法或条件概率还不熟，先读[09-数学与张量预备课](/notes/multimodal-math-prerequisites/)。正文中的小矩阵均为教学构造，不是某个商业模型的参数。

<span id="mm-b72f0e7cec69" style="display:block;scroll-margin-top:6rem"></span>

## 1. 先看完整路线：文字怎样变成一次预测

“杯子在桌上”会被 tokenizer 切成若干 token，再转换成整数 ID。一个 token 不必等于一个汉字或一个词。Embedding 是可训练表：ID 负责选行，取出的向量才进入神经网络。若词表大小为 $V_{\rm vocab}$、隐藏维度为 $d$，表的形状为 $[V_{\rm vocab},d]$；一批 $B$ 条、每条长 $T$ 的序列查表后形状是 $[B,T,d]$。

同一个词在输入查表时可以得到相同的初始向量，经过上下文处理后却会不同。比如“苹果手机”和“吃苹果”中的苹果，后续隐藏状态可以反映不同含义。隐藏状态是本次输入计算得到的激活；模型权重是跨输入复用的可训练参数，两者不能混用。

<figure style="margin:1.5rem 0">
<a href="/notes-assets/transformer-attention-rope-gqa/02-LLM%E7%BB%93%E6%9E%84%E5%85%A8%E8%A7%86%E5%9B%BE.png" target="_blank" rel="noopener" aria-label="查看原图：LLM结构全视图"><img src="/notes-assets/transformer-attention-rope-gqa/02-LLM%E7%BB%93%E6%9E%84%E5%85%A8%E8%A7%86%E5%9B%BE.png" alt="LLM结构全视图" width="9133" height="3700" loading="lazy" decoding="async" style="height:auto;cursor:zoom-in" /></a>
<figcaption style="font-size:0.9em">LLM结构全视图（点击查看原图）</figcaption>
</figure>

**这张大图怎么读**：先看最右侧从下往上的主线：输入→Embedding→多层 Transformer Decoder→解码输出。再看中间放大的单层：RMSNorm→Attention→残差相加→RMSNorm→FFN→残差相加。最后才看最左侧的注意力内部：QKV 投影、RoPE、相关分数、mask、softmax、加权 V 和输出投影。图中的小方格是在追踪张量形状，并不代表每一个方格都是一个真实单词；图示超参数也不是所有 LLM 的固定配置。

以常见 Pre-Norm 因果 block 为例：

$$
U=X+\operatorname{Attn}(\operatorname{Norm}(X)),\qquad
Y=U+\operatorname{FFN}(\operatorname{Norm}(U)).
$$

Attention 的主要作用是让不同位置交换信息；FFN 对每个位置的向量进行非线性加工。FFN 不直接在时间轴上混合 token，但其输入已经通过 Attention 包含上下文。残差保留原输入的直接路径；若写为 $Y=X+F(X)$，其局部雅可比是 $I+\partial F/\partial X$，这给梯度提供直接通路，却不保证任意深度下都不会梯度异常。

最后一层隐藏向量经 LM head 映射成词表大小的 logits，再经 softmax 得到“下一个 token”的概率。输入位置 $t$ 的 logits 通常用于预测 $t+1$；这个错一位关系会在 SFT 的 label shift 中再次出现，见[02-SFT与DPO的训练信号](/notes/multimodal-sft-lora-dpo/)。

<span id="mm-6819ce0e9f13" style="display:block;scroll-margin-top:6rem"></span>

## 2. Q、K、V 为什么需要三组投影

设本层输入 $X\in\mathbb R^{T\times d}$，暂时省略 batch。我们要解决两个不同问题：当前 token 应该关注谁，以及从被关注位置取出什么内容。于是学习三组线性映射：

$$
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V.
$$

Q 是用于发出查询的特征，K 是用于匹配的特征，V 是实际被加权汇总的特征。它们并非人工规定“Q 只看名词、K 只看动词”；语义功能由数据和损失共同学出。三组权重可以实现为一个合并的大线性层再切开，但概念上仍是三个投影。

单头先算每个 query 与每个 key 的点积：

$$
S_{ij}=\frac{q_i^\top k_j}{\sqrt{d_h}},\qquad
P_{ij}=\frac{\exp(S_{ij}+M_{ij})}{\sum_k\exp(S_{ik}+M_{ik})},\qquad
O_i=\sum_jP_{ij}v_j.
$$

$d_h$ 为一个 head 内 Q/K 的宽度；$i$ 是查询位置，$j$ 是被查看的位置。矩阵形式 $O=\operatorname{softmax}(QK^\top/\sqrt{d_h}+M)V$ 中，softmax 对每行的 key 维归一化。每一行在回答“这个 query 的注意力分给哪些 key”，不是让所有 query 争夺同一份概率。

<span id="mm-9a71dac109aa" style="display:block;scroll-margin-top:6rem"></span>

### 缩放为什么是平方根

假设初始化附近，$q_\ell,k_\ell$ 近似独立、零均值且方差为 1，则一个乘积的方差约为 1，$d_h$ 个乘积相加的方差约为 $d_h$。点积尺度随维度增加，softmax 容易极端尖锐；除以 $\sqrt{d_h}$ 后方差回到约 1。训练后的分量未必独立，这是一种设计动机和数值控制，不是模型永远服从该分布的定理。

softmax 的导数 $\partial p_i/\partial z_j=p_i(\mathbf1[i=j]-p_j)$ 说明，概率接近 0/1 时部分梯度很小。因此问题不是“大点积数学上不允许”，而是它可能让优化不好进行。

<span id="mm-4e76a2744cb1" style="display:block;scroll-margin-top:6rem"></span>

### 因果、padding、loss 三种 mask

因果 mask 规定位置 $i$ 不能看 $j>i$；padding mask 排除补齐长度用的无效 key；loss mask 决定哪些预测目标参与训练评分。前两者在 Attention 可见关系上生效，最后一种在损失里生效。回答 token 即使不监督 prompt，仍然需要看 prompt：把所有 prompt 都从 attention 中挡住会丢掉问题条件。

不可见分数通常加 $-\infty$，让其指数变为 0。若先对完整分数做 softmax，再把未来项清零，剩余权重之和通常不是 1，而且可见位置的归一化曾被未来项影响；除非再次正确归一化，否则不等价。整行全被遮挡还可能产生 NaN，需在有效查询的构造和 padding 处理上解决。

<span id="mm-b696d10b109c" style="display:block;scroll-margin-top:6rem"></span>

## 3. 手算两个 token，看到信息真的如何混合

取 $x_1=(1,0),x_2=(0,1)$，为便于手算，让三个投影都等于单位矩阵，故 $Q=K=V=X$，$d_h=2$。

$$
S=\frac1{\sqrt2}
\begin{bmatrix}1&0\\0&1\end{bmatrix},\qquad
S+M=
\begin{bmatrix}0.7071&-\infty\\0&0.7071\end{bmatrix}.
$$

第一行只有一个可见位置，概率为 $(1,0)$。第二行：

$$
P_{21}=\frac1{1+e^{0.7071}}\approx0.33024,\qquad
P_{22}\approx0.66976.
$$

因此 $O_1=(1,0)$、$O_2\approx(0.33024,0.66976)$。第二个位置的输出中已经混入第一个位置的 V。改动第二个输入不能影响第一个输出，因为因果 mask 没给它通路。

<figure style="margin:1.5rem 0">
<a href="/notes-assets/transformer-attention-rope-gqa/17-%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%89%8B%E7%AE%97%E4%B8%8E%E7%BC%93%E5%AD%98%E6%8E%A9%E7%A0%81.svg" target="_blank" rel="noopener" aria-label="查看原图：注意力手算与缓存掩码"><img src="/notes-assets/transformer-attention-rope-gqa/17-%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%89%8B%E7%AE%97%E4%B8%8E%E7%BC%93%E5%AD%98%E6%8E%A9%E7%A0%81.svg" alt="注意力手算与缓存掩码" width="1200" height="780" loading="lazy" decoding="async" style="height:auto;cursor:zoom-in" /></a>
<figcaption style="font-size:0.9em">注意力手算与缓存掩码（点击查看原图）</figcaption>
</figure>

**读图**：上半部把“分数→被遮挡→概率→加权结果”画为同一行计算。下半部是有缓存时的非方形可见表：已有三个 token，新来两个 token 时，第一行允许看前四列、第二行允许看全部五列。不要把普通方形下三角直接截一块当缓存 mask。具体推导见[04-推理效率与手撕考点](/notes/prefill-decode-video-tokens/)。

<span id="mm-6cba75ce1249" style="display:block;scroll-margin-top:6rem"></span>

## 4. 多头究竟多在哪里，GQA 又共享了什么

多头为每个 query 学出多组匹配空间，然后把各头结果拼接并做输出投影。头数增加不必让模型总宽度增加；常见设计令 $H_qd_h=d$，但具体模型也可能单独设定 head dimension。

| 阶段 | 张量形状 | 哪个轴在变化 |
|---|---|---|
| 输入 | $[B,T,d]$ | batch、位置、隐藏特征 |
| Q 投影并拆头 | $[B,H_q,T,d_h]$ | 多组查询 |
| K/V 投影并拆头 | $[B,H_{kv},T,d_h]$ | 独立存储的 K/V 组 |
| 逐 query head 相关分数 | $[B,H_q,T_q,T_k]$ | query 位置与 key 位置两轴 |
| 加权 V | $[B,H_q,T_q,d_h]$ | 汇总每个 query 的内容 |
| 合并并输出投影 | $[B,T_q,d]$ | 回到 block 宽度 |

<figure style="margin:1.5rem 0">
<a href="/notes-assets/transformer-attention-rope-gqa/03-MHA-GQA-MQA-MLA.png" target="_blank" rel="noopener" aria-label="查看原图：MHA-GQA-MQA-MLA"><img src="/notes-assets/transformer-attention-rope-gqa/03-MHA-GQA-MQA-MLA.png" alt="MHA-GQA-MQA-MLA" width="1772" height="1113" loading="lazy" decoding="async" style="height:auto;cursor:zoom-in" /></a>
<figcaption style="font-size:0.9em">MHA-GQA-MQA-MLA（点击查看原图）</figcaption>
</figure>

**读图**：蓝色是 Q，绿色是 K，黄色是 V。MHA 每个 Q 头对应一套 K/V；图中 GQA 四个 Q 头分成两组，每组共用一套；MQA 所有 Q 共用一套。最右 MLA 的虚线 K/V 表示可从更小的潜表示获得相应信息，它是另一种压缩参数化，不能只理解为“更少的头”。MLA 的压缩与 RoPE 解耦见[08-模型家族与论文精读路线](/notes/multimodal-models-paper-reading/)。

若 $H_q=32,H_{kv}=8$，每 4 个 Q 头共用一组 K/V。不同 Q 仍得到不同的注意力权重，因为 Q 不同；共用的是供它们读取的 key/value。高效内核可以按映射共享，不必先把缓存复制四份。

为什么能省显存？生成时每层历史 K/V 需保留。设层数 $L$、序列长 $T$、每元素 $s$ 字节，则普通 GQA 缓存近似为

$$
M_{\rm KV}=2LBT H_{kv}d_hs.
$$

取 $L=32,B=1,T=4096,d_h=128,s=2$：MHA 的 32 KV 头占 2 GiB，8 KV 头占 0.5 GiB，1 KV 头占 64 MiB。这个账单只包括缓存。每个 Q 仍要与可见历史位置做匹配，GQA 并未把长序列 attention 变成常数成本；其质量与速度权衡需在具体模型上评测。[GQA 原论文 §2](https://arxiv.org/html/2305.13245v3#S2)。

<span id="mm-145eb6ed40ad" style="display:block;scroll-margin-top:6rem"></span>

## 5. 位置编码：内容相同，顺序为何会改变含义

“先拿杯子，再开门”和“先开门，再拿杯子”拥有很相似的词集合，意义却不同。无位置表示、且没有改变排列关系的 mask 时，自注意力对 token 的排列具有置换等变性：重排输入也会相应重排输出。因果 mask 本身已经提供方向和不同的可见前缀，所以不能说所有无显式位置的因果模型“完全不知道顺序”；但精确的距离和时间结构仍需要合适的位置表示。

绝对位置方法可把位置向量加到 embedding；相对位置偏置可直接加在注意力得分上；RoPE 则旋转 Q/K 的子向量。它们操作的地方不同，不宜混称“都在输入加一个位置编码”。

<span id="mm-705018ea2a0f" style="display:block;scroll-margin-top:6rem"></span>

### RoPE 的二维推导

前文批量投影 $XW$ 用行向量约定；现在为了画旋转，把一个 head 的 $q,k$ 写成列向量，只有记号转置，含义没有改变。二维旋转矩阵：

$$
R(\phi)=
\begin{bmatrix}\cos\phi&-\sin\phi\\\sin\phi&\cos\phi\end{bmatrix}.
$$

位置 $m$ 用角度 $m\omega$ 旋转 Q，位置 $n$ 用 $n\omega$ 旋转 K。点积化为

$$
(R(m\omega)q)^\top(R(n\omega)k)
=q^\top R(m\omega)^\top R(n\omega)k
=q^\top R((n-m)\omega)k.
$$

核心是最后的 $n-m$：绝对位置分别进入旋转，二者交互时出现相对位移。取 $q=k=(1,0)$，角差为 0 时得分为 1，角差 $90^\circ$ 时为 0，$180^\circ$ 时为 -1。可见单频点积并不随距离单调衰减。

<figure style="margin:1.5rem 0">
<a href="/notes-assets/transformer-attention-rope-gqa/04-RoPE%E4%BD%8D%E7%BD%AE%E7%BC%96%E7%A0%81.png" target="_blank" rel="noopener" aria-label="查看原图：RoPE位置编码"><img src="/notes-assets/transformer-attention-rope-gqa/04-RoPE%E4%BD%8D%E7%BD%AE%E7%BC%96%E7%A0%81.png" alt="RoPE位置编码" width="1877" height="1549" loading="lazy" decoding="async" style="height:auto;cursor:zoom-in" /></a>
<figcaption style="font-size:0.9em">RoPE位置编码（点击查看原图）</figcaption>
</figure>

**读图**：大旋转矩阵只有对角线附近的二维块非零；每块把一对通道相互混合，不混合不同 token。图用 8 维示意四种频率，实际一般是在每个 head 的旋转子维度上应用；不能把图中 hidden size=8 当模型规格，也不能把标为 embedding 的向量误读为必定直接旋转输入词嵌入。通常的 decoder RoPE 作用于投影后的 Q/K，部分旋转等变体依实现而定。

不同二维对使用不同频率，例如经典形式 $\omega_j=\theta^{-2j/d_r}$，$d_r$ 是参与旋转的维度。实现可能配对相邻通道，也可能前后半区配对；权重布局与旋转实现必须匹配。数学形式能代入更大位置，不代表模型已经在那些位置训练过：频率缩放、长上下文训练、数值精度都影响外推。

缓存里旧 K 已用旧位置旋转，新 Q/K 用新位置。已有 100 个真实 token 时，新位置不能重新从 0 算。批量 padding 情况应依据每条样本的实际 position IDs，而非简单拿填充后的公共长度套用。

<span id="mm-769a919b0817" style="display:block;scroll-margin-top:6rem"></span>

## 6. RMSNorm 与残差：控制每个 token 的尺度

同一网络层中，隐藏向量的数值尺度会随训练变化。RMSNorm 用每个 token 自己的隐藏通道计算均方根，再乘可学习缩放：

$$
\operatorname{RMSNorm}(x)_j=
g_j\frac{x_j}{\sqrt{\frac1d\sum_{\ell=1}^dx_\ell^2+\epsilon}}.
$$

$g_j$ 允许各通道重新选择合适尺度，$\epsilon$ 避免接近零时数值异常。这里归一化的是一个 token 的特征轴，一般不是在 batch 或时间轴上共同归一化。

对 $x=(3,4)$，均方根为 $\sqrt{12.5}\approx3.5355$；取 $g=(1,1)$、忽略很小的 $\epsilon$，结果约 $(0.8485,1.1314)$。LayerNorm 会先减均值 3.5，再除标准差 0.5，结果约 $(-1,1)$。两者都能调整尺度，但是否去均值这一点不同；仅用零均值向量测试会掩盖差异。[RMSNorm 原论文](https://arxiv.org/abs/1910.07467)。

Pre-Norm 和 Post-Norm 描述 Norm 与子层/残差的相对顺序；RMSNorm 和 LayerNorm 描述归一化算法。它们是不同维度的选择，不能把 Pre-Norm 当成 RMSNorm 的同义词。

<span id="mm-9563c95eb5ca" style="display:block;scroll-margin-top:6rem"></span>

## 7. FFN、SwiGLU 与 MoE：如何加工已融合的内容

普通 FFN 常写作 $\phi(xW_1)W_2$，$\phi$ 泛指激活函数，例如 GELU：先扩展通道、逐元素非线性、再投回原宽度。若中间不加非线性，两次线性映射会合并为一次，表达能力受限。SwiGLU 使用两个分支：

$$
u=xW_u,\quad g=xW_g,\quad
z=\operatorname{SiLU}(g)\odot u,\quad
\operatorname{FFN}(x)=zW_d,\qquad
\operatorname{SiLU}(a)=a\sigma(a).
$$

$\odot$ 表示逐元素相乘；这里 $\sigma(a)=1/(1+e^{-a})$ 特指 sigmoid，不是上一段的泛激活 $\phi$。$W_u,W_g:[d,m]$，$W_d:[m,d]$。两路都映射到中间宽度 $m$ 后才能逐元素相乘。输入分支名因代码而异，不能靠变量名判断哪边有激活。

<figure style="margin:1.5rem 0">
<a href="/notes-assets/transformer-attention-rope-gqa/05-SwiGLU.png" target="_blank" rel="noopener" aria-label="查看原图：SwiGLU"><img src="/notes-assets/transformer-attention-rope-gqa/05-SwiGLU.png" alt="SwiGLU" width="1686" height="1525" loading="lazy" decoding="async" style="height:auto;cursor:zoom-in" /></a>
<figcaption style="font-size:0.9em">SwiGLU（点击查看原图）</figcaption>
</figure>

**读图**：输入在底部分成两条线性分支；右支经 SiLU 后与左支做 Hadamard 乘积。图顶部展示的是门控乘积，完整 Transformer FFN 通常还要接回 $d$ 维的输出投影 $W_d$。SiLU 是连续函数，输出可为负，不能把它当只在 0 与 1 之间的开关。例：某通道 $g=1,u=2$，乘积 $2/(1+e^{-1})\approx1.4621$。

经典两矩阵 FFN 若 $m=4d$，约有 $8d^2$ 权重；三矩阵 SwiGLU 为约 $3dm$。若取 $m\approx8d/3$，参数量约相同，这说明比较激活函数时常会调整中间宽度。具体模型会按硬件对齐取整，不可声称所有 SwiGLU 的 $m$ 都相同。[GLU Variants 原论文](https://arxiv.org/abs/2002.05202)。

MoE 进一步把单个 FFN 换为多位专家和一个 router。router 为每个 token 打分，选 top-$k$ 专家，按门控权重合并结果。若选中两专家输出 2、10，归一权重 0.75、0.25，则输出为 4。专家数量决定很多总参数，但单个 token 只激活其中一部分；共享专家、负载均衡、容量及跨卡通信会改变实际成本。

Decoder-only 描述因果语言骨干；MoE 描述部分子层如何路由。一个模型可以同时满足两者。GQA 改 Attention 的 KV 共享，MoE 改 FFN 的专家选择，它们也能并用。将 MoE 与 Decoder-only 写成二选一，会把不同设计维度混在一起。

<span id="mm-c2d16908afce" style="display:block;scroll-margin-top:6rem"></span>

## 8. 把视觉接进来后，什么变了

典型视觉语言模型先将像素变成视觉特征，再通过 projector、merger 或查询压缩模块进入语言骨干。视觉特征的维度、数量、位置表示及训练目标都要对齐。图像占一个占位符，不等于进入 LLM 后只占一个视觉 token。

一个 $224\times224$ RGB 图像以 $16\times16$ patch 切分，得到 196 个 patch，每块原始向量宽度为 $16\cdot16\cdot3=768$。视觉塔可输出 $[196,d_v]$，线性 projector 可映射为 $[196,d]$；若另有 32-query 压缩器，则可能输出 $[32,d]$。后者节省上下文预算，也可能丢失细粒度信息。8 帧不压缩是 1568 个视觉 token，逐帧压到 32 个则为 256；真实模型还会加边界/时间信息，或在时空上联合合并。

这条维度链只解释“能接上”，并不解释“接上之后为何理解正确”。视觉预训练、图文对齐、指令数据和视频时间建模继续见[06-视觉视频算法面试专项](/notes/vision-video-algorithms/)；不同 Qwen/DeepSeek/Llama 版本的设计见[08-模型家族与论文精读路线](/notes/multimodal-models-paper-reading/)。

<span id="mm-d5ec7e3f0c6b" style="display:block;scroll-margin-top:6rem"></span>

## 面试口述与进一步追问

QKV 题可先用一段话讲计算路径，再展开一个具体形状和一个数值例子：“本层隐藏状态分别经 Q/K/V 投影，按 head 计算缩放点积，加可见关系的 mask，沿 key 维 softmax 后加权 V，再合并头和输出投影。GQA 让多组 Q 共用较少 K/V，但各 Q 的注意力仍不同。”

如果被追问“为什么”，依次解释缩放对应点积分布、mask 对应因果条件、softmax 对应归一化权重、V 对应被汇总内容。如果被追问“怎么证明没写错”，给出未来 token 不影响过去输出、缓存与整段 forward 一致的检查，见[04-推理效率与手撕考点](/notes/prefill-decode-video-tokens/)。完整练习与答案见[10-十四天练习与参考解答](/notes/multimodal-fourteen-day-workbook/)。

<span id="mm-94cf8ab39d29" style="display:block;scroll-margin-top:6rem"></span>

## 来源与图像说明

本章四幅原图来自 [LLM-RL-Visualized](https://github.com/changyeyu/LLM-RL-Visualized)，保留余昌叶及仓库署名，按其[非商业图片许可](https://github.com/changyeyu/LLM-RL-Visualized/blob/master/LICENSE)使用；注意力算例图为本教程自绘。原图是教学示意，本章已说明 FFN 输出投影、RoPE 作用位置等省略之处。技术依据另见 [Transformer](https://arxiv.org/abs/1706.03762)、[RoPE](https://arxiv.org/abs/2104.09864)、[GQA](https://arxiv.org/abs/2305.13245)、[RMSNorm](https://arxiv.org/abs/1910.07467)、[SwiGLU](https://arxiv.org/abs/2002.05202)。
