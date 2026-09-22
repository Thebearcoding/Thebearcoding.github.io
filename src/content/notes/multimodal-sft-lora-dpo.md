---
title: SFT与DPO：训练信号从哪里来
date: '2026-09-22'
tags:
- SFT
- DPO
- LoRA
- 多模态训练
- 数据分布
- 面试
summary: 从标签对齐和交叉熵推导，走到LoRA梯度、QLoRA和DPO偏好目标。
draft: false
obsidian: true
series: multimodal-interview
---
<span id="mm-a5813b6edb01" style="display:block;scroll-margin-top:6rem"></span>

这篇解释一个贯穿多模态后训练的问题：**模型究竟根据什么信号改变参数，为什么 loss 下降不一定代表看懂了视频？**读完后，你应能给一条样本画出监督位置、手算交叉熵、解释 LoRA 的第一步梯度，并从偏好假设推到 DPO，而不只记住算法名字。

先读 [从一个token理解Transformer与多模态入口](/notes/transformer-attention-rope-gqa/) 中的自回归预测与 attention mask。本文的算例用“先开门，再进屋”这个视频问答贯穿；“开门”被当成单个教学 token，真实 tokenizer 未必如此切分。SFT/DPO 是训练目标，LoRA/QLoRA 是参数更新与存储方式，二者可以组合。本章还讨论数据分布如何影响后续训练；推导、练习和视频案例均用于教学，不代表已完成的实验。

<span id="mm-fb4faa0cfd7f" style="display:block;scroll-margin-top:6rem"></span>

## 1. 从一句答案到一个概率

<span id="mm-efb10e8babce" style="display:block;scroll-margin-top:6rem"></span>

### 1.1 为什么整段概率变成一串 token 概率

令 $x$ 表示所有已知条件：问题文本、对话模板，以及图像/视频特征。令答案 $y=(y_1,\ldots,y_T)$，$\theta$ 是可训练参数。链式法则给出：

$$
p_\theta(y\mid x)=\prod_{t=1}^{T}p_\theta(y_t\mid x,y_{<t}).
$$

例如两 token 答案“开门、进屋”的概率，是“看到视频和问题后说开门”的概率，乘以“已给定真实前一个 token 开门后说进屋”的概率。不是两个互不相关的分类，也不是两 token 概率相加。

概率连乘容易非常小。取自然对数会把乘法变成求和：

$$
\log p_\theta(y\mid x)=\sum_{t=1}^{T}\log p_\theta(y_t\mid x,y_{<t}).
$$

最大化正确示范的概率等价于最小化负对数概率。这就是 SFT 的基本目标；模型不会因为名字里有“监督”而自动检查示范是否真实。

<span id="mm-bc01168ec439" style="display:block;scroll-margin-top:6rem"></span>

### 1.2 Teacher forcing 为什么能并行训练

训练时，完整真实答案已经存在。位置 $t$ 使用真实前缀 $y_{<t}$ 预测 $y_t$，称为 teacher forcing。因果 attention mask 保证它看不到待预测 token 及未来答案。各位置虽然预测不同条件概率，但可以在一次带因果 mask 的矩阵前向中一起算出。

推理时没有真实答案可喂，下一步只能接着模型自己刚生成的 token。这造成条件分布差异：训练通常见到正确前缀，生成可能被前面的错误带偏。它解释了为什么 token loss 很低，完整生成仍可能出错；但不能单凭这个现象认定某次失败就是 teacher forcing 所致。

<figure style="margin:1.5rem 0">
<a href="/notes-assets/multimodal-sft-lora-dpo/06-SFT%E4%BA%A4%E5%8F%89%E7%86%B5.png" target="_blank" rel="noopener" aria-label="查看原图：SFT交叉熵"><img src="/notes-assets/multimodal-sft-lora-dpo/06-SFT%E4%BA%A4%E5%8F%89%E7%86%B5.png" alt="SFT交叉熵" width="1334" height="1234" loading="lazy" decoding="async" style="height:auto;cursor:zoom-in" /></a>
<figcaption style="font-size:0.9em">SFT交叉熵（点击查看原图）</figcaption>
</figure>

**读图时沿着虚线看。**顶部黄色区域是 prompt，蓝色区域是参考回答。底部每一列是一整个词表的预测分布；粉色数字只取该位置真实目标 token 的概率。第一枚回答 token 的概率来自最后一个 prompt 位置的 logits，所以虚线向左错开一格。斜线覆盖的 prompt 区域不计 loss，但其内容仍能被回答位置读取。图中末尾还监督了结束符，因此有效目标数量要包含它。图用具体 tokenizer 的词表和切词作示意，不能把图中文字格数当成所有模型的 token 数。

<span id="mm-bd0ff5b87ace" style="display:block;scroll-margin-top:6rem"></span>

## 2. 三种 mask 与一次不能错的 shift

<span id="mm-639a8e42352d" style="display:block;scroll-margin-top:6rem"></span>

### 2.1 “能读什么”与“哪里评分”是两件事

因果 mask 遮住未来；padding mask 遮住填充位置；loss mask 决定哪个目标计入训练。一个 prompt 位置可以不被评分，却仍被后面的答案读取。把 prompt 的 label 设为忽略，不等于把 prompt 从 attention 中删除，更不等于它相关的所有参数梯度都为零：答案损失仍能通过上下文路径影响参数。

常见 PyTorch 交叉熵约定用 label 值 $-100$ 表示忽略目标。这是 loss API 的约定，不是词表里的特殊 token，也不是输入 ID；不能把输入中的问题 token 全改成 $-100$。

<span id="mm-1ee3f99588a4" style="display:block;scroll-margin-top:6rem"></span>

### 2.2 把一个序列逐位置对齐

用一个极简序列说明，已省略真实 chat template：

| 原序列位置 | 输入 token | 这一位置 logits 预测的下一个 token | 下一个目标是否计 loss |
|---|---|---|---|
| 0 | BOS | 问题 | 否 |
| 1 | 问题 | 助手边界 | 否 |
| 2 | 助手边界 | 开门 | 是 |
| 3 | 开门 | 进屋 | 是 |
| 4 | 进屋 | EOS | 是 |
| 5 | EOS | 当前样本没有提供下一个目标 | 否 |

若 labels 与输入等长、尚未 shift，可写成：

    输入：   [BOS, 问题, 助手边界, 开门, 进屋, EOS]
    labels：[-100, -100, -100, 开门, 进屋, EOS]

真正计算时，前 5 个位置的 logits 对齐后 5 个 labels：位置 2 的 logits 对 labels 位置 3，也就是“开门”。有些模型 forward 已在内部完成这个 shift；外部再 shift 一次，就变成“隔两个 token 预测”。因此“labels 等于 input_ids 再遮住 prompt”是否正确，要结合调用 API 是否内部移位。

多轮对话还要选定监督协议：只监督最后一轮，还是所有 assistant 轮次。两种都可定义，但不能把某一轮的 assistant 边界误判成用户内容。EOS 是否监督同样必须一致，否则可能影响停止行为。

<span id="mm-8142b35c3853" style="display:block;scroll-margin-top:6rem"></span>

### 2.3 Packing 为什么不能只拼起来

Packing 把短样本接在同一个长度槽里，提高有效 token 利用率。如果目标是让各样本保持独立，后一个样本不能通过 attention 读取前一个不相关样本；通常需要按样本边界隔离 attention，并处理 position IDs 与跨样本预测目标。

仅把一个边界 token 的 label 忽略，不能阻止后面答案读取前一视频的文字。反过来，也有训练协议故意允许串接文本；那是另一个目标，不应称为“等价于独立样本”。视频数据尤其应检查“这条答案实际条件于哪一个视频”。

<span id="mm-7e44abd1279e" style="display:block;scroll-margin-top:6rem"></span>

## 3. 交叉熵为何会提高目标词概率

<span id="mm-d67c6f895de5" style="display:block;scroll-margin-top:6rem"></span>

### 3.1 损失、求和与平均分别是什么意思

给每个回答目标一个 $m_t\in\{0,1\}$，求和损失为：

$$
\mathcal L_{\rm sum}=-\sum_{t=1}^{T}m_t\log p_\theta(y_t\mid x,y_{<t}).
$$

按有效 token 平均则为：

$$
\mathcal L_{\rm token}=
-\frac{\sum_t m_t\log p_\theta(y_t\mid x,y_{<t})}{\sum_t m_t}.
$$

分母必须非零。若全部 labels 被忽略，常见 mean reduction 会产生 NaN 或无效结果；它不是“模型全答对，所以 loss 为零”。

若两个监督 token 的概率为 0.8、0.25，则：

$$
\mathcal L_{\rm sum}=-\log0.8-\log0.25
=0.22314+1.38629=1.60944,
\qquad
\mathcal L_{\rm token}=0.80472.
$$

这里未纳入 EOS，仅为两 token 手算。如果加入 EOS，应加入它的负对数概率并修改平均分母。把用户消息也纳入损失，会增加“根据前文预测用户消息”的训练目标；它不等价于强迫模型在 assistant 回答里复述 prompt。

<span id="mm-6b693e5a1be0" style="display:block;scroll-margin-top:6rem"></span>

### 3.2 把梯度真的推出来

词表有 $V$ 项，模型输出 logits $z_1,\ldots,z_V$。logit 是未归一化分数，可以为负数；概率由 softmax 得到：

$$
p_j=\frac{e^{z_j}}{\sum_{\ell=1}^{V}e^{z_\ell}}.
$$

目标词编号为 $k$，于是：

$$
\ell=-\log p_k=-z_k+\log\sum_\ell e^{z_\ell}.
$$

第一项对 $z_j$ 求导是 $-\mathbf1[j=k]$。第二项先对 log 求导，再对指数求导：

$$
\frac{\partial}{\partial z_j}\log\sum_\ell e^{z_\ell}
=\frac{e^{z_j}}{\sum_\ell e^{z_\ell}}=p_j.
$$

合起来：

$$
\frac{\partial\ell}{\partial z_j}=p_j-\mathbf1[j=k].
$$

若三个词概率为 $(0.25,0.50,0.25)$，第一词是目标，梯度为 $(-0.75,0.50,0.25)$。假设只做一次对这些 logits 的梯度下降，第一词分数会上升，其他下降；真实网络更新的是共享参数，所以多个样本、位置的梯度会一起作用，不能保证每一个 token 的概率每一步都单调增加。

用 one-hot 标签写出的交叉熵 $\ell=-\sum_j y_j\log p_j$ 与上述公式相同，因为只有目标位置 $y_k=1$。交叉熵测的是对标签的拟合；若标签与画面矛盾，loss 下降可以意味着更牢固地学错。

<span id="mm-70250ec99414" style="display:block;scroll-margin-top:6rem"></span>

### 3.3 token 平均与样本平均不是同一件事

样本 A 有 2 个监督 token、总损失 2；样本 B 有 8 个监督 token、总损失 16。全 token 平均是 $18/10=1.8$；先按样本取均值再平均是 $(1+2)/2=1.5$。前者让长回答贡献更多 token，后者让每条样本权重相同。两者均可选择，但梯度累积、分布式训练和评估必须遵循同一约定，详见 [训练显存与实验排障：把机制变成可检查的量](/notes/training-memory-debugging/)。

<span id="mm-b7ee470fd875" style="display:block;scroll-margin-top:6rem"></span>

## 4. 冻结底座以后，LoRA 怎样还能学习

<span id="mm-b70b99f64320" style="display:block;scroll-margin-top:6rem"></span>

### 4.1 低秩更新约束的是什么

普通线性层用列向量记法 $h=W_0x$，$x\in\mathbb R^{d_{\rm in}}$，$W_0\in\mathbb R^{d_{\rm out}\times d_{\rm in}}$。全参微调会改变 $W_0$ 的所有元素。LoRA 保持 $W_0$ 不变，另加：

$$
h=W_0x+sBAx,\qquad
A\in\mathbb R^{r\times d_{\rm in}},\quad
B\in\mathbb R^{d_{\rm out}\times r},\quad s=\alpha/r.
$$

先用 $A$ 把输入映射到 $r$ 维，再用 $B$ 回到输出维度，更新矩阵 $\Delta W=sBA$ 的秩最多为 $r$。被限制的是**更新矩阵的秩**，不是要求原始 $W_0$ 本来就低秩。$\alpha/r$ 是经典 LoRA 的缩放；不同变体可能换缩放，讨论 rank 实验时需要一起说明。

一个 $4096\times4096$ 层有 $16{,}777{,}216$ 个权重。$r=8$ 时 LoRA 有 $8(4096+4096)=65{,}536$ 个参数，占该层原权重约 $0.390625\%$。整个模型节省比例依注入层、视觉塔、连接器、embedding 等是否可训练而不同。

<span id="mm-9e25b5ba2b5a" style="display:block;scroll-margin-top:6rem"></span>

### 4.2 为什么 A 随机、B 为零，而不是都为零

经典初始化令 A 随机、B 为零。此时 $BA=0$，初始输出与底座一致，但梯度仍能启动。设 $g=\partial\ell/\partial h$，则：

$$
\frac{\partial\ell}{\partial B}=s\,g(Ax)^\top,\qquad
\frac{\partial\ell}{\partial A}=s\,B^\top g x^\top.
$$

例如 $r=1,s=1,x=(1,2)^\top,A=(1,-1),B=(0,0)^\top,g=(3,4)^\top$。先算 $Ax=-1$，所以：

$$
\nabla_B\ell=(-3,-4)^\top,\qquad \nabla_A\ell=(0,0).
$$

学习率 0.1 时，B 更新为 $(0.3,0.4)^\top$；下一次 A 也能获得非零梯度。若 A、B 一开始都为零，两个梯度都为零，这条支路在普通梯度更新下无法启动。这里讨论经典初始化，不排除其他专门设计的非零初始化方案。[LoRA 原论文 §4.1](https://arxiv.org/abs/2106.09685)。

<span id="mm-3f9ee1021e49" style="display:block;scroll-margin-top:6rem"></span>

### 4.3 冻结、反向传播与 QLoRA

冻结 $W_0$ 意味着不为它更新参数、通常不保留它的梯度和优化器状态；并不表示该层可以被从计算图删除。输入梯度仍有：

$$
\frac{\partial\ell}{\partial x}=W_0^\top g+sA^\top B^\top g.
$$

前面若还有可训练连接器或 LoRA 层，就需要这个梯度。长视频的激活链因此仍可能占用很多显存。

QLoRA 进一步将冻结底座以低位形式存储。4 bit 只有 16 种编码，保存的可以是一个数值码表的索引，并配合每组权重的尺度恢复近似值。若用教学用的四级码表 $(-1,-0.3,0.3,1)$、某组尺度为 2，则编码对应近似权重 $(-2,-0.6,0.6,2)$；真实权重 0.7 会近似成 0.6。这是存储近似，不是把权重训练成四个离散类别。真实 NF4 使用 16 级、按预期近似正态权重分布设计的码值，本例的四级表不是 NF4 原表。

量化还必须保存各组尺度，不能只算“参数量除以二字节”。经典 QLoRA 的 double quantization 再压缩这些尺度常数；分页优化器则利用内存分页机制缓解优化状态的显存峰值，可能涉及 CPU/GPU 迁移。矩阵运算仍需把量化编码映射到计算用数值，LoRA 参数并非“全部用 4 bit 训练”。它节约的主要是冻结底座存储和全参优化状态，不能把 token 数引起的激活成本一起消除。[QLoRA 原论文](https://arxiv.org/abs/2305.14314)。

普通未量化 LoRA 可将 $sBA$ 合并进 $W_0$ 后推理；量化权重合并则涉及反量化、再量化与误差，不能自动宣称零精度损失。更多细节回链 LoRA、QLoRA。

<span id="mm-7a9030a2a652" style="display:block;scroll-margin-top:6rem"></span>

## 5. 为什么需要偏好数据：从示范到比较

SFT 告诉模型“这一条示范值得模仿”。偏好数据给同一条件 $x$ 下两条回答，并标出 chosen $y^+$ 优于 rejected $y^-$。例如视频明明是先开门再进屋，chosen 保留正确顺序，rejected 颠倒顺序；二者语言都很流畅，偏好区分的是视频事实。

“偏好”不天然等于真实正确。若标注者只看措辞、没看视频，优化会奖励好听的错误。好的偏好对还应控制明显的长度/格式混杂，记录判断依据，允许存在标注分歧。

DPO 的入口是离线的 $(x,y^+,y^-)$ 数据和冻结参考模型 $\pi_{\rm ref}$。经典离线 DPO 不需要在每个训练 step 重新 rollout，也不单独训练一个显式 reward model；但偏好数据本身可以来自先前的模型采样与标注。

<span id="mm-6cccc2407a1d" style="display:block;scroll-margin-top:6rem"></span>

## 6. 从一个受约束目标推到 DPO

<span id="mm-d7369284b334" style="display:block;scroll-margin-top:6rem"></span>

### 6.1 KL 为什么表示偏离参考策略

固定条件 $x$，令 $\pi_y=\pi(y\mid x)$、$q_y=\pi_{\rm ref}(y\mid x)$。从当前策略抽一个回答，平均对数概率比是：

$$
D_{\rm KL}(\pi\Vert q)=\sum_y\pi_y\log\frac{\pi_y}{q_y}.
$$

它不是对称距离，交换两分布通常不相等。KL 非负，两个分布相同取零；单个采到的 log-ratio 却可为负，不能把“单 token 的负值”直接判为实现错。

为了兼顾奖励与保留底座行为，考虑固定奖励函数 $r_y=r(x,y)$ 下：

$$
\max_\pi F(\pi)=\sum_y\pi_y r_y-\beta\sum_y\pi_y\log\frac{\pi_y}{q_y},
\qquad \sum_y\pi_y=1,\quad\beta>0.
$$

这是推导用的理想分布优化；实际神经网络只表示其中一部分策略，有限样本、有限优化步数也不等于这个精确最优解。还假定讨论的回答处 reference 有正概率，相关求和/归一化存在。

<span id="mm-fadc890d2261" style="display:block;scroll-margin-top:6rem"></span>

### 6.2 拉格朗日推导：归一化常数从哪里出现

加入概率和为 1 的约束，令乘子为 $\lambda$：

$$
\mathcal F=\sum_y\pi_y r_y-\beta\sum_y\pi_y\log\frac{\pi_y}{q_y}
+\lambda\left(\sum_y\pi_y-1\right).
$$

对每个 $\pi_y$ 求导。利用 $\frac{d}{du}(u\log u)=\log u+1$：

$$
\frac{\partial\mathcal F}{\partial\pi_y}
=r_y-\beta\left(\log\frac{\pi_y}{q_y}+1\right)+\lambda=0.
$$

移项、取指数：

$$
\log\frac{\pi_y}{q_y}=\frac{r_y}{\beta}+\frac{\lambda}{\beta}-1,
\qquad
\pi_y=q_y e^{r_y/\beta}e^{\lambda/\beta-1}.
$$

最后一个因子对同一 $x$ 下所有回答相同。由概率和等于 1，得到：

$$
Z(x)=\sum_y\pi_{\rm ref}(y\mid x)e^{r(x,y)/\beta},
\qquad
\pi^*(y\mid x)=\frac{\pi_{\rm ref}(y\mid x)e^{r(x,y)/\beta}}{Z(x)}.
$$

奖励线性项加上负 KL 构成凹目标，在上述支持条件下这个解是全局最优，不只是随便找到一个驻点。再整理为：

$$
r(x,y)=\beta\log\frac{\pi^*(y\mid x)}{\pi_{\rm ref}(y\mid x)}+\beta\log Z(x).
$$

这一行很关键：若用策略相对 reference 的概率变化表示奖励，就不一定需要另训练一张 reward 网络。

<span id="mm-0300e02a17dc" style="display:block;scroll-margin-top:6rem"></span>

### 6.3 从 Bradley–Terry 偏好假设得到分类损失

Bradley–Terry 模型假设，两个回答被偏好的概率由奖励差决定：

$$
P(y^+\succ y^-\mid x)
=\frac{e^{r(x,y^+)}}{e^{r(x,y^+)}+e^{r(x,y^-)}}
=\sigma\big(r(x,y^+)-r(x,y^-)\big).
$$

其中 $\sigma(u)=1/(1+e^{-u})$。它是一种偏好建模假设，不是“人类偏好必定满足”的自然规律。

把上一节的奖励表达代入。同一 prompt 的 $\beta\log Z(x)$ 相消；再用待学习的 $\pi_\theta$ 参数化策略，对观测偏好取负对数似然：

$$
\mathcal L_{\rm DPO}
=-\mathbb E_{(x,y^+,y^-)\sim\mathcal D}
\log\sigma\left[
\beta\left(
\log\frac{\pi_\theta(y^+\mid x)}{\pi_{\rm ref}(y^+\mid x)}
-\log\frac{\pi_\theta(y^-\mid x)}{\pi_{\rm ref}(y^-\mid x)}
\right)\right].
$$

这条推导连接了三件事：KL 正则化的奖励优化、基于奖励差的偏好概率、离线偏好数据上的分类损失。它不意味着有限数据 DPO 必然达到理想 RL 最优解，也不意味着没有偏好建模。[DPO 原论文及附录 A](https://arxiv.org/abs/2305.18290)。

<figure style="margin:1.5rem 0">
<a href="/notes-assets/multimodal-sft-lora-dpo/07-DPO%E8%AE%AD%E7%BB%83%E5%85%A8%E6%99%AF.png" target="_blank" rel="noopener" aria-label="查看原图：DPO训练全景"><img src="/notes-assets/multimodal-sft-lora-dpo/07-DPO%E8%AE%AD%E7%BB%83%E5%85%A8%E6%99%AF.png" alt="DPO训练全景" width="1424" height="1410" loading="lazy" decoding="async" style="height:auto;cursor:zoom-in" /></a>
<figcaption style="font-size:0.9em">DPO训练全景（点击查看原图）</figcaption>
</figure>

**按四条概率路径读图。**蓝色 chosen 与粉色 rejected 各经过策略模型和参考模型，形成四个条件概率。带锁的参考模型只提供基准，梯度最后回到策略模型。图里的 Gather 是从每个词表分布中取真实回答 token 的概率；对整段回答还需逐 token 求 log-prob 并相加。两个灰色“隐式奖励”框相减，进入最下方的 sigmoid 分类损失。直接把四条路径缩成“好答案做 SFT、坏答案负 SFT”，会漏掉 reference 和随偏好置信度变化的权重。

<span id="mm-cca67c691bfd" style="display:block;scroll-margin-top:6rem"></span>

## 7. 四个 log-prob、一个梯度与 beta 的真实含义

<span id="mm-1e49f901ea50" style="display:block;scroll-margin-top:6rem"></span>

### 7.1 手算 DPO

令四个整段回答 log-prob 如下；它们都只统计回答目标，包含哪些结束符由同一协议决定：

| 回答 | 当前策略 | 参考策略 | 相对参考变化 $u$ |
|---|---:|---:|---:|
| chosen | $-2$ | $-2.5$ | $u^+=0.5$ |
| rejected | $-3$ | $-2.5$ | $u^-=-0.5$ |

取 $\beta=0.2$，分类 logit 为 $z=\beta(u^+-u^-)=0.2$，所以：

$$
\ell=-\log\sigma(0.2)=\log(1+e^{-0.2})\approx0.59814.
$$

如果交换当前策略的好坏 log-prob，其余不变，$z=-0.2$，损失变为约 $0.79814$。若当前策略与参考模型完全相同，则 $u^+=u^-=0$，初始损失为 $\log2\approx0.69315$；这不是“没有梯度”。

令 $\Delta=u^+-u^-$，对差值求导：

$$
\frac{\partial\ell}{\partial\Delta}
=-\beta\,\sigma(-\beta\Delta).
$$

初始 $\Delta=0$ 时导数为 $-\beta/2$，梯度下降会推动相对偏好间隔增大。已经强烈满足偏好的样本权重较小；被模型错误排序的样本权重较大。由于共享参数和概率归一化，不能保证每一步 chosen 的绝对概率上升、rejected 的绝对概率下降；DPO 直接优化的是相对差值。

<span id="mm-7d41aa57cb2f" style="display:block;scroll-margin-top:6rem"></span>

### 7.2 beta 为什么看起来有两种相反解释

在本篇明确采用的 $\mathbb E r-\beta KL$ 约定中，对**固定 reward 的理想优化**，更大 $\beta$ 更重视靠近参考策略。例如 reference 对两答案各给 0.5，奖励为 1 和 0，则最优好答案概率：

$$
p^+=\frac{e^{1/\beta}}{e^{1/\beta}+1}=\sigma(1/\beta).
$$

$\beta=0.1,1,10$ 时依次约为 $0.99995,0.73106,0.52498$；更大 $\beta$ 更接近 0.5。

但在 DPO 分类损失上，固定当前 $\Delta=0$，梯度绝对值是 $\beta/2$，更大 $\beta$ 的**此刻梯度**更大。这不矛盾：一个讨论理想最优分布，另一个讨论特定参数点的优化尺度。学习率、偏好噪声、数据覆盖与参数化还会影响训练。因此面试不能把 beta 简化成“越大训练越稳”。

<span id="mm-21387febac63" style="display:block;scroll-margin-top:6rem"></span>

### 7.3 正确概率为何还会受长度和模板影响

整段概率是乘积，log-prob 是求和。长度增加通常会让整段 log-prob 更负。把求和偷偷换成 token 平均，会改变经典 DPO 的目标，不是纯数值技巧。不同长度偏好对、截断和 EOS 处理都会影响实际更新，应把采用的变体说清。

两回答必须条件于同一视频、采样帧、问题和模板。reference 若预先缓存 log-prob，可节省训练时前向；但若随机抽帧或数据增强改变输入，旧缓存就不再是当前条件下的概率。reference 通常冻结，并不代表“DPO 总共只加载一个模型”。

数值实现宜用 log-softmax 后 gather，再按回答 mask 求和；直接把小概率连乘后取 log 容易下溢。sigmoid 再取 log 也可用稳定的 log-sigmoid 实现。

<span id="mm-501b25eaf89d" style="display:block;scroll-margin-top:6rem"></span>

## 8. 数据分布怎样把 SFT 和后续 GRPO 连起来

SFT 改变了起始策略会生成什么。GRPO 随后在同一个问题下采多个答案，比较奖励；如果起始策略几乎从不生成正确证据链，二值奖励可能让整组全错。如果所有答案都正确、奖励完全相同，也没有组内优劣信号。公式与例题在 [从RL基础推到PPO与GRPO](/notes/policy-gradient-ppo-grpo/)。

这不是“同 prompt 做过 SFT，所以不能再做 RL”的定理。需要区分：

| 情况 | 真正的问题 |
|---|---|
| SFT 与 RL 训练 prompt 重合 | 两阶段数据设计，可有意义，需测量 |
| 训练与测试同源视频或近重复 | 泛化评估泄漏，必须在拆分时处理 |
| SFT 已把全部 reward 可区分行为学满 | 后续该奖励剩余改进空间小 |
| SFT 只学会答题模板、未学视觉证据 | loss 好看，生成事实仍可能错 |

视频数据应记录视频 ID、片段 ID、帧采样时间戳、问题、答案与证据区间；按源视频隔离 train/val/test，再检查相近片段、字幕和改写问题的重复。caption 数据主要描述静态场景；静态 VQA 训练问题条件下的取证；视频时序问答才直接训练先后、动作变化或时间定位。增加一种数据不代表自动覆盖另外两种能力。

“SFT 后 GRPO 没有提升”只能作为排障现象；缺少完整日志与对照实验时，不能确定“失败就是过拟合”。应当逐项验证数据分布、初始化、采样、奖励和优化条件。

<span id="mm-3ea9dfb6dcce" style="display:block;scroll-margin-top:6rem"></span>

## 9. 把概念用于面试，而不把推测写成真题

**问题：SFT 与 DPO 有什么区别？**  
SFT 最大化示范回答在给定条件下的似然。DPO 用同题两回答的偏好标签，优化它们相对冻结 reference 的 log-prob 差值；其公式可由 KL 正则化奖励优化和 Bradley–Terry 偏好模型联系起来。两者都依赖标注可靠性，DPO 不会自动发现视频证据矛盾。接着可以主动解释四个 log-prob 与 EOS/mask 处理。

**问题：为什么 LoRA 冻结了底座还需要反向传播？**  
冻结的是原权重更新；梯度仍需通过原线性变换传播到 LoRA 或上游模块。LoRA 省梯度/优化器状态，不保证激活很少。用 A 随机、B 为零的首步梯度举例，比只背“低秩”更完整。

**问题：SFT loss 降了，视频 QA 为什么反而差？**  
loss 只衡量训练标签拟合。先解释可能的目标错位：监督位置错误、视觉证据漏采、模板捷径、数据分布不同、标签噪声或泄漏；再用固定测试集的顺序题、OCR、字幕依赖等分桶验证。需要观测和对照，不能凭一条 loss 曲线归因。

这三题是根据知识点整理的扩展练习；更多口述题和项目追问见 [面试问题与项目深挖](/notes/multimodal-interview-questions/)。

<span id="mm-1fa2dedf5dec" style="display:block;scroll-margin-top:6rem"></span>

## 10. 自测：能算清，才算读懂

**练习 1。**只有两个回答 token 被监督，概率为 0.5 和 0.5，求和与均值损失各是多少？若另有 10 个 prompt token 不监督，分母是否变 12？

**答案。**求和为 $2\log2\approx1.38629$，均值为 $\log2\approx0.69315$；分母仍为 2。若把 12 当分母，会把梯度尺度压低六倍。

**练习 2。**A 的 reference log-prob 为 $-8$、当前为 $-7$；B 为 $-2$、$-1.5$。A 是 chosen，DPO logit 是什么？为什么 B 绝对概率更大不构成矛盾？

**答案。**$u_A=1,u_B=0.5$，logit 为 $0.5\beta$。DPO 比较的是相对 reference 的偏好变化；当前模型对 B 的绝对概率较大，不妨碍其对 A 的相对提升更大。

**练习 3。**一个 $1024\times2048$ 线性层用 rank 4 LoRA，有多少可训练矩阵参数？A、B 都为零会发生什么？

**答案。**$4(1024+2048)=12{,}288$。普通 LoRA 两矩阵都零时，彼此的梯度因子也零，支路不能启动；经典做法只让其中 B 为零、A 随机。

**练习 4。**reference log-prob 已缓存；第二天训练随机抽取同视频的不同帧，还能直接复用吗？

**答案。**不能默认复用。条件 $x$ 已改变，原 log-prob 不再对应当前输入。只有确定同样的完整输入、模板与概率计算协议时，缓存才匹配。

<span id="mm-1a00ac98b40b" style="display:block;scroll-margin-top:6rem"></span>

## 来源与进一步阅读

- 余昌叶：[图解仓库 SFT](https://github.com/changyeyu/LLM-RL-Visualized#header-14)、[DPO](https://github.com/changyeyu/LLM-RL-Visualized#header-19)。保留原图署名，使用条件见 [学习总览](/notes/multimodal-interview-guide/)。
- [DPO 原论文](https://arxiv.org/abs/2305.18290)：核对目标与附录推导；本文的数字例题为独立教学构造。
- [LoRA 原论文](https://arxiv.org/abs/2106.09685)、[QLoRA 原论文](https://arxiv.org/abs/2305.14314)：区分低秩更新、初始化和量化存储。
