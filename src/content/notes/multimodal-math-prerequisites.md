---
title: 读懂大模型公式之前：数学与张量预备课
date: '2026-09-22'
tags:
- 多模态算法
- 数学基础
- 张量
- 概率
- 预备课
summary: 用具体数字学会张量、点积、条件概率、softmax、梯度和KL。
draft: false
obsidian: true
series: multimodal-interview
---
<span id="mm-1225e87274c4" style="display:block;scroll-margin-top:6rem"></span>

本章为从零阅读模型与训练教程准备共同语言。目标不是补完整门高等数学，而是让 Attention、SFT、DPO、PPO 的每个符号都能落到具体的数。你只需要会加减乘除；遇到求导时，先理解它描述“某个数变一点，结果怎样变”，再看推导。

<span id="mm-c6665c53a2e8" style="display:block;scroll-margin-top:6rem"></span>

## 1. 标量、向量、矩阵与张量：形状是数据的索引方式

一个数如 0.5 是标量。按顺序排几个数如 $(1,2,3)$ 是向量；按行列排是矩阵；再增加 batch、head、时间等索引轴，就是高阶张量。“张量维度”有两种常见意思：轴数（rank）与某个轴的长度。为了不混淆，本教程直接写形状。

$X:[2,3,4]$ 表示 2 条样本、每条 3 个位置、每个位置 4 个特征，共 $2\cdot3\cdot4=24$ 个数。它不是“24 维模型”，而是隐藏宽度 4 的一批数据。$X_{1,2,3}$ 是其中一个标量：第二条样本、第三个位置、第四个通道（若从 0 开始编号）。

模型权重和中间激活都可用张量保存，但生命周期不同。权重在训练中更新，推理时被不同输入复用；激活随本次输入计算产生。KV Cache 存的是激活，不是把 $W_K,W_V$ 又存一次。

reshape 改变索引组织但不改变元素数量；transpose 交换轴。$[B,T,Hd_h]$ 常先 reshape 为 $[B,T,H,d_h]$，再交换中间两轴得到 $[B,H,T,d_h]$。只写 reshape 直接变为 $[B,H,T,d_h]$ 可能把位置和头的元素排错，虽然总数仍正确。

<span id="mm-9803d7115ee7" style="display:block;scroll-margin-top:6rem"></span>

## 2. 点积、矩阵乘法、逐元素乘法解决不同问题

点积把两个同长度向量变成一个数：

$$
(1,2)\cdot(3,4)=1\cdot3+2\cdot4=11.
$$

它可以衡量方向相关性，但原始点积也受长度影响。余弦相似度先除以两向量范数：$\cos\theta=(u^\top v)/(\|u\|\|v\|)$。CLIP 中先做 L2 归一化后点积，就等于余弦；零向量需按实现稳定处理。

矩阵乘法是“左边每一行与右边每一列点积”。例如

$$
X=\begin{bmatrix}1&2\\3&4\end{bmatrix},\quad
W=\begin{bmatrix}1&0&-1\\0&2&1\end{bmatrix},\quad
XW=\begin{bmatrix}1&4&1\\3&8&1\end{bmatrix}.
$$

形状是 $[2,2]\times[2,3]\to[2,3]$；中间两维必须相同。每行输入通过同一个 $W$，得到三个新特征，线性层就是这样的通道混合。Attention 的 $QK^\top$ 则用每个 Q 与每个 K 点积：若 $Q:[T_q,d_h]$、$K:[T_k,d_h]$，结果为 $[T_q,T_k]$。

逐元素乘法 $\odot$ 不会沿一个轴求和：$(1,2)\odot(3,4)=(3,8)$。SwiGLU 两路相乘用它；把它误写成矩阵乘法，输出形状和机制都变了。

加权和把多个向量按权重合成，例如 $0.25(2,0)+0.75(0,4)=(0.5,3)$。Attention 的输出正是对 V 的加权和。权重非负且和为 1 时是凸组合，但后面的输出投影、残差和非线性会继续改变向量，不能据此说整个网络只能平均输入。

<span id="mm-8ca674ffd299" style="display:block;scroll-margin-top:6rem"></span>

## 3. 概率、条件概率与期望：一次结果不等于平均表现

概率分布中各事件概率非负、总和为 1。$P(y\mid x)$ 表示已经知道 $x$ 时 $y$ 的概率；条件符号后的内容不是再乘一次，而是定义当前考虑的情况。

语言模型通常采用链式分解：

$$
P(y_1,y_2\mid x)=P(y_1\mid x)P(y_2\mid x,y_1).
$$

若两项为 0.8 与 0.25，则整段概率为 0.2。第二项已知道第一个 token；不能把它换成无条件的 $P(y_2)$。多模态时 $x$ 可包括图像特征、采样帧和问题文本。

期望是按概率加权的平均。例如奖励取 1 的概率 0.4、取 0 的概率 0.6，则 $\mathbb E[R]=0.4$，但一次采样只得到 0 或 1。采 10 次的样本平均也不保证正好 0.4；样本数增大才在适当独立性条件下更稳定地接近期望。

条件期望 $\mathbb E[G\mid s]$ 是固定当前状态 $s$，对未来可能轨迹取平均。RL 中的 $V(s)$ 估计这个平均；一条轨迹算出的回报 $G$ 是随机样本。把一次 TD 目标当成精确价值，就是混淆了这两者。

方差描述围绕平均的波动：$\operatorname{Var}(R)=\mathbb E[(R-\mathbb E[R])^2]$。上面二值奖励的方差为 $0.4(0.6)^2+0.6(-0.4)^2=0.24$，标准差为平方根。GRPO 的“组内标准差”只来自同一问题的有限几条回答，还会受总体/样本标准差约定影响。

<span id="mm-fe86a88988b2" style="display:block;scroll-margin-top:6rem"></span>

## 4. 为什么模型爱用 log-prob

自然对数 $\log$ 是指数函数 $e^x$ 的逆函数。最有用的性质是 $\log(ab)=\log a+\log b$。因此一长串 token 概率的乘积可以变成 log-prob 的求和：

$$
\log P(y\mid x)=\sum_t\log P(y_t\mid x,y_{<t}).
$$

概率小于 1，所以 log-prob 通常为负；“更大”意味着更接近 0，对应更高概率。比如 $\log0.8\approx-0.2231$ 大于 $\log0.25\approx-1.3863$。长度较长的回答有更多负项，整段求和和平均不是同一个目标，比较时要说明长度处理。

两个概率的比率也可稳定地从 log-prob 得到：

$$
\frac{\pi_\theta(a\mid s)}{\pi_{\rm old}(a\mid s)}
=\exp(\log\pi_\theta(a\mid s)-\log\pi_{\rm old}(a\mid s)).
$$

例如新概率 0.3、旧概率 0.2，比率 1.5。这与“新策略的概率等于 1.5”不同：比率可以大于 1，概率不能。PPO 裁剪的是比率；DPO 用的是当前策略与参考策略的整段 log-prob 差，具体见对应章节。

<span id="mm-52644fac2b7d" style="display:block;scroll-margin-top:6rem"></span>

## 5. Softmax：把任意分数变成竞争关系

logit 是尚未归一化的实数分数；它可以大于 1 或为负。softmax 对每个候选取指数并除总和：

$$
p_j=\frac{e^{z_j}}{\sum_k e^{z_k}}.
$$

若 $z=(0,\log3)$，指数为 $(1,3)$，概率为 $(0.25,0.75)$。将两个 logits 都加 100，概率不变，因为共同的 $e^{100}$ 会在分子分母抵消。这允许用稳定写法先减最大值 $m$，再算 $e^{z_j-m}$，避免指数溢出。

调温度时用 $\operatorname{softmax}(z/\tau)$。例如 $z=(0,\log3)$、$\tau=0.5$，变成指数 $(1,9)$，概率 $(0.1,0.9)$，更尖锐。$\tau=2$ 时指数 $(1,\sqrt3)$，分布更平缓。CLIP 的温度和语言模型生成温度都影响分布尺度，但属于不同位置的参数，不可混为同一个超参数。

sigmoid 则把一个数压到 $(0,1)$：$\sigma(a)=1/(1+e^{-a})$。它等于二候选 softmax 中的概率 $\exp(a)/(\exp(0)+\exp(a))$，所以 DPO 可把两答案的相对得分差放进 sigmoid。

<span id="mm-bdff064d7df7" style="display:block;scroll-margin-top:6rem"></span>

## 6. 导数和梯度：解释“往哪里改”

导数描述输入很小变化时，输出变化率。若 $f(w)=w^2$，在 $w=3$ 附近增加 $\Delta w$，$f$ 大约增加 $6\Delta w$，故导数为 6。梯度是多参数情况下各个偏导数组成的向量。

训练最小化损失时做 $w\leftarrow w-\eta\nabla L$；最大化收益时做 $\theta\leftarrow\theta+\eta\nabla J$。大部分优化器执行下降，所以最大化 PPO surrogate 常在实现中取负号作为 actor loss。这只是优化方向的转换。

链式法则处理多层依赖。若 $z=wx$、$L=(z-y)^2/2$，则

$$
\frac{\partial L}{\partial w}
=\frac{\partial L}{\partial z}\frac{\partial z}{\partial w}
=(wx-y)x.
$$

取 $w=1,x=2,y=3$，梯度为 $-2$；学习率 0.1 时 $w$ 更新为 1.2，新输出 2.4 更接近目标 3。这解释反向传播沿计算图逐层乘局部导数。

冻结参数只是不更新这些参数；为更新其前面的可训练模块，往往仍须通过冻结模块计算输入梯度。LoRA 冻结底座权重，也不等于底座前向不存在。stop-gradient（detach）则明确阻止某条变量路径继续求导；在 PPO 中，采样时旧 log-prob 和已计算优势通常要当固定训练信号。

还有一个重要区别：$L(\theta)=\theta$ 在 $\theta=0$ 时值为 0，导数却是 1。因此“loss 的值为零”不能推出“梯度为零”。中心化优势可能让 GRPO 的正负项在数值上抵消，梯度仍能驱动学习。

<span id="mm-31ed1973508b" style="display:block;scroll-margin-top:6rem"></span>

## 7. 交叉熵、熵与 KL 各自比较什么

熵 $H(p)=-\sum_jp_j\log p_j$ 描述分布自身的不确定性。二类均匀分布熵为 $\log2\approx0.6931$；一个类别概率为 1 的确定分布熵为 0（按 $0\log0=0$ 的连续约定）。

交叉熵 $H(q,p)=-\sum_jq_j\log p_j$ 是用预测 $p$ 去匹配目标 $q$。如果目标是 one-hot 类别 $k$，只剩 $-\log p_k$。预测目标概率 0.25 时损失约 1.3863；预测为 0.8 时约 0.2231。这对应 SFT 的每 token 训练信号。

KL 散度为

$$
D_{\rm KL}(p\Vert q)=\sum_jp_j\log\frac{p_j}{q_j}
=H(p,q)-H(p).
$$

它是“按 $p$ 的概率加权，比较 $p$ 与 $q$”；方向不可随便交换。取 $p=(0.8,0.2),q=(0.5,0.5)$，有

$$
D_{\rm KL}(p\Vert q)=0.8\log1.6+0.2\log0.4\approx0.1927.
$$

反方向约为 $0.2231$，确实不同。若 $p_j>0$ 而 $q_j=0$，KL 可无穷大，涉及支持集条件。

单次从 $p$ 采到第 2 类时，$\log(p_2/q_2)=\log0.4<0$；单个样本的 log-ratio 可以为负，虽然完整 KL 非负。因此不能拿一个负的 token log-ratio 判断“KL 算错了”。近似 KL、无偏估计和训练时自动求导的目标还需看采样分布及是否 stop-gradient，见[从RL基础推到PPO与GRPO](/notes/policy-gradient-ppo-grpo/)。

<span id="mm-5ce0c9f8f446" style="display:block;scroll-margin-top:6rem"></span>

## 8. 用单位检查公式，不急着记数字

张量元素数乘“每元素字节数”才是存储量；BF16 常为 2 字节。$1\ {\rm GB}=10^9$ 字节，$1\ {\rm GiB}=2^{30}$ 字节；同一个缓存用这两个单位数字不同。

例如 $[2,8192,4096]$ 的 BF16 激活有 $2\cdot8192\cdot4096$ 个元素，占 $134{,}217{,}728$ 字节，即 128 MiB。第一项 2 是 batch，最后还乘一次 2 才是每元素字节数；少乘一个因子就会把量级估错。

复杂度 $O(N^2d)$ 描述变量增大时的增长，不含全部常数、硬件、调度和通信。把输入长度翻倍，某个二次项变四倍，并不意味着整体运行时间也一定四倍。用[Prefill、Decode与视频token的计算代价](/notes/prefill-decode-video-tokens/)的 Prefill/Decode 分项账单继续理解。

<span id="mm-c88b41f09856" style="display:block;scroll-margin-top:6rem"></span>

## 接下来怎样检验理解

在[十四天练习册：从手算到多模态面试](/notes/multimodal-fourteen-day-workbook/)第 1 天，先独立算矩阵乘法、softmax、条件概率和 KL，再对照过程。卡在某一步，就回到本章对应小节；无需一次把所有公式背下。

这些基础定义支撑 [Transformer](https://arxiv.org/abs/1706.03762)、[CLIP](https://arxiv.org/abs/2103.00020)、[DPO](https://arxiv.org/abs/2305.18290) 和 [PPO](https://arxiv.org/abs/1707.06347) 的目标与计算。本章算例均为自拟，公式采用自然对数。
