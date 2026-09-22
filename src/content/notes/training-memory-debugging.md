---
title: 训练显存与实验排障：把机制变成可检查的量
date: '2026-09-22'
tags:
- 训练显存
- 数值稳定
- 排障
- LoRA
- 实验设计
- 大模型面试
summary: 解释AdamW、混精、梯度累积、激活与显存，建立可验证的排障路径。
draft: false
obsidian: true
series: multimodal-interview
---
<span id="mm-5ceedb771961" style="display:block;scroll-margin-top:6rem"></span>

“7B 的 BF16 权重只有 14 GB，为什么训练还会 OOM？”“loss 已下降，为什么视频问答变差？”这两类问题共同要求把一个抽象算法放回真实训练过程：**哪一项张量被保存，哪一条梯度被计算，哪个指标才支持结论。**

本文承接 [SFT与DPO：训练信号从哪里来](/notes/multimodal-sft-lora-dpo/) 的损失与 LoRA，以及 [从RL基础推到PPO与GRPO](/notes/policy-gradient-ppo-grpo/) 的 rollout、优势和新旧策略。目标是能解释和定位现象，不要求掌握分布式部署框架。下文通过教学构造的显存账单与排障案例，将训练机制变成可检查的量。

<span id="mm-2c2559ac4c6a" style="display:block;scroll-margin-top:6rem"></span>

## 1. 一次参数更新究竟发生了什么

一条视频样本先被抽帧、编码并组成序列，再由模型前向得到 loss。反向传播沿计算图应用链式法则，为可训练参数得到梯度。优化器根据梯度及历史状态修改参数。多次重复这一循环才形成训练。

这里有四种常被混为一谈的量：权重是模型当前参数；激活是某个样本在前向过程中的中间结果；梯度是 loss 对参数的导数；优化器状态是为了更新参数而保存的历史统计。它们可能形状相同，但生命周期、dtype 与用途不同。

<span id="mm-47f0c906c07a" style="display:block;scroll-margin-top:6rem"></span>

### 1.1 梯度下降与 AdamW 为什么需要额外内存

最简单的梯度下降是 $w_{t+1}=w_t-\eta g_t$，$\eta$ 为学习率，$g_t=\partial L/\partial w_t$。例如 $w=2,g=0.3,\eta=0.1$，更新后为 1.97。

Adam 对每个参数额外维护梯度的一阶矩 m 和平方梯度的二阶矩 v：

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\qquad
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2.
$$

从零初始化会导致早期统计偏小，所以使用偏差修正：

$$
\hat m_t=\frac{m_t}{1-\beta_1^t},\qquad
\hat v_t=\frac{v_t}{1-\beta_2^t}.
$$

一种常见 AdamW 更新写为：

$$
w_{t+1}=(1-\eta\lambda_w)w_t
-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon_{\rm opt}}.
$$

$\lambda_w$ 是独立的 weight decay 系数，$\epsilon_{\rm opt}$ 防分母过小，不能与 PPO 的 clip 参数混用。AdamW 的权重衰减与自适应梯度更新分开；它不等价于在任意自适应优化器里直接把 L2 项加进 loss。

首步令 $g=0.3,\beta_1=0.9,\beta_2=0.999$，则 $m_1=0.03,v_1=0.00009$；修正后 $\hat m_1=0.3,\hat v_1=0.09$。忽略极小 epsilon，归一化更新方向为 1。若 $\eta=0.001,\lambda_w=0,w=2$，新值约为 1.999。这个例子说明 Adam 的更新量不只是“学习率乘原梯度”；还说明每个参数为何通常要配两份状态。

<span id="mm-a4f2d309759c" style="display:block;scroll-margin-top:6rem"></span>

### 1.2 梯度裁剪约束的是整次更新前的梯度

全局 norm clipping 常对拼接后的梯度向量 g 做：

$$
g\leftarrow g\min\left(1,\frac{C}{\|g\|_2+\epsilon}\right).
$$

$C$ 是范数阈值。若 $g=(3,4)$，范数为 5，阈值为 1，则缩为 $(0.6,0.8)$。这保留方向、降低整体幅度；逐元素 clip 到 $[-1,1]$ 会得到 $(1,1)$，是另一种操作。

裁剪通常在梯度累积完成、混精梯度已 unscale 后、optimizer.step 前进行。它能限制有限大梯度，不能修复已经产生的 NaN，也不能使错误监督目标变正确。

<span id="mm-f2ac8bcbbe3f" style="display:block;scroll-margin-top:6rem"></span>

## 2. 显存账单必须分项目

<span id="mm-7c22a3704cb4" style="display:block;scroll-margin-top:6rem"></span>

### 2.1 权重小于训练状态总和

设参数量 $P=7\times10^9$。下表是一个教学情景：BF16 权重和梯度，FP32 Adam 两份状态，不做分片/卸载。

| 项目 | 每参数字节 | 7B 对应十进制容量 | 用途 |
|---|---:|---:|---|
| BF16 权重 | 2 | 14 GB | 前向及反向使用 |
| BF16 梯度 | 2 | 14 GB | 当前优化周期更新信号 |
| FP32 一阶矩 m | 4 | 28 GB | 平滑梯度 |
| FP32 二阶矩 v | 4 | 28 GB | 平滑平方梯度 |
| 四项合计 | 12 | 84 GB | 尚未计激活等 |
| 若另存 FP32 master 权重 | 再加 4 | 再加 28 GB | 高精度参数更新副本 |

master 权重让小更新先在较高精度下累积，再转换为计算用权重。并非每个 BF16/优化器实现都额外存它；有的梯度或状态 dtype 也不同。因此 84 GB、112 GB 都是**指定存储假设下的账单**，不是 7B 全参训练的普遍硬门槛。

GB 按 $10^9$ 字节，GiB 按 $2^{30}$ 字节。比较工具显示值前应先统一单位。实际峰值还包括激活、视觉塔、临时工作区、通信 buffer、缓存分配和碎片；“模型参数数×2”只算了一行。

<span id="mm-7d433eac19f6" style="display:block;scroll-margin-top:6rem"></span>

### 2.2 长视频为什么让激活迅速增大

设 batch 为 B、序列长度 T、隐藏维度 d。一个 BF16 的 $[B,T,d]$ 张量占：

$$
M=2BTd\ \text{bytes}.
$$

$B=2,T=8192,d=4096$ 时为 $134{,}217{,}728$ 字节，即 128 MiB。T 翻倍到 16384，这个张量就是 256 MiB。每层通常有不止一个需要反向的张量，MLP 中间维度还可能比 d 大。

朴素 attention 若保存 $[B,H,T,T]$ 分数或概率矩阵，则随 T 平方增长。例如 B=1、H=32、T=8192、每元素 2 字节，仅一个矩阵就占 4 GiB。改变内核、checkpointing 或保存策略可以避免这项完整物化，但不能改变“同一个张量有多少元素”的算术。

视频增加 T 的方式是帧数、分辨率、动态裁剪、patch 数或时间 token 增长。视觉编码器也有自己的激活和计算。必须同时理解视觉侧和语言侧，不能把所有 OOM 都归因于 LLM 参数太大。

<span id="mm-1a0c9c5cf5e4" style="display:block;scroll-margin-top:6rem"></span>

### 2.3 训练激活与推理 KV Cache 必须分开

普通 teacher-forced SFT 一次并行处理答案，通常关闭自回归生成用的持久 decode cache。它仍会产生 attention 的 K/V 激活，但不是“逐步追加、供下一次生成复用”的 KV Cache 生命周期。

rollout、评估生成与线上推理则会持有历史 KV。以标准 GQA cache 为例：

$$
M_{\rm KV}\approx2LBT H_{\rm kv}d_hs,
$$

L 为层数，$H_{\rm kv}$ 为 KV 头数，$d_h$ 为每头维度，s 为字节数，最前面的 2 对应 K 和 V。32 层、B=1、T=8192、8 KV 头、128 维、BF16 时约为 1 GiB。详细来源与边界见 [Prefill、Decode与视频token的计算代价](/notes/prefill-decode-video-tokens/)。

RL 会先 rollout，再进行策略训练，有时生成引擎和训练模型同时驻留。因此一次 RL OOM 要先定位发生在生成还是反向阶段；不能把所有内存都套进 SFT 的 12P 公式，也不能把所有峰值都套进 KV 公式。

<span id="mm-0e73da833113" style="display:block;scroll-margin-top:6rem"></span>

## 3. 各种省显存方法改的是哪一项

<span id="mm-7a72b4dcfdfb" style="display:block;scroll-margin-top:6rem"></span>

### 3.1 梯度累积：降低 micro-batch 才会降低激活峰值

设每次前向处理 b 条样本，累积 K 次才更新一次，D 个数据并行 worker，则在样本数一致的情形下，有效 batch 为 $bKD$。例如 $b=2,K=8,D=4$，一次参数更新使用 64 条样本。

每个 micro-batch 反向后可以释放大部分激活，参数梯度累积到同一份 buffer。因此用更小 b 加更大 K，可减少单次激活峰值。只增加 K、b 不变，不会让那个 micro-batch 的前向突然更省内存；单条长视频已放不下时，累积本身也不能解决。

在 loss 可加、参数在 K 次累积之间不更新、归一化一致的条件下，累积梯度对应大 batch 梯度。浮点加法顺序、dropout、BatchNorm 等可能使数值或训练行为并非逐位一致，所以“完全一样”需要条件。

<span id="mm-3219a166f168" style="display:block;scroll-margin-top:6rem"></span>

### 3.2 为什么只除以累积次数可能算错

若每个 micro-batch 有相同有效 token 数，对 mean loss 再除 K 通常匹配全 token 平均。但长度可变时，各 micro-batch 的有效 token 数不同。

例如第一个有 2 个 token，总损失 2；第二个有 8 个 token，总损失 16。分别 mean 再平均是 $(1+2)/2=1.5$，而全 token 平均是 $(2+16)/(2+8)=1.8$。两种目标不是同一个权重。

若目标就是全 token 平均，总有效 token 为 N，可以让每个 micro-batch 的 loss_sum 都除以 N 后再累加梯度。分布式框架若还自动平均各 rank 梯度，需把全局分母和该平均约定同时考虑。具体地，若 D 个 rank 的梯度最终取算术平均、全局有效 token 数为 $N_{\rm global}$，每个 rank 的局部 loss sum 应乘 $D/N_{\rm global}$，平均后才得到全局 sum 除以 $N_{\rm global}$；若框架已经完成这项缩放，不能再乘一次。概念关键是：**梯度对应哪个明确的总 loss**。

优化周期应在开始时清零梯度，在各 micro-batch 后只累积，结束时才 unscale、裁剪和更新。如果每个 micro-batch 都 optimizer.step，就不是梯度累积；若每次都 zero_grad，又丢掉前面结果。

<span id="mm-20c6d327fef0" style="display:block;scroll-margin-top:6rem"></span>

### 3.3 Activation checkpointing：少存结果，反向时重算

反向需要某些前向激活。checkpointing 只保存选定边界上的输入/必要状态，内部结果到反向时再做一次前向重建。例如四个连续 block 中，保存每一段入口，再在需要这段梯度时重新计算内部 attention/MLP 激活。

因此它节约的是激活存储，代价是额外计算；不会让参数、Adam 状态突然缩小。若重算涉及 dropout 等随机算子，框架还需要保持合适的随机状态，否则重算不再对应原来的前向。具体节省量依分段和哪些张量必须保存，没有普遍固定的“减半”。

<span id="mm-9575b4983e36" style="display:block;scroll-margin-top:6rem"></span>

### 3.4 LoRA/QLoRA：主要减少可训练状态和冻结底座存储

LoRA 冻结原权重，用低秩参数更新，所以原权重无需梯度和 Adam 两份状态；但原权重仍需参与前向，梯度仍可能穿过该层到上游模块。QLoRA 再压缩冻结权重存储，计算 dtype 与存储 bit 数不同。

因此低秩参数很少，不代表长视频激活很少。若 OOM 来自 T 太长，只减 LoRA rank 可能变化有限；若全参状态才是大头，LoRA 的帮助就很明显。首步梯度与参数数量手算见 [SFT与DPO：训练信号从哪里来](/notes/multimodal-sft-lora-dpo/)。

<span id="mm-d0085138cb71" style="display:block;scroll-margin-top:6rem"></span>

### 3.5 ZeRO：让多卡少存重复状态

普通数据并行常让各卡保存同一份参数、梯度和优化器状态。ZeRO 分阶段去掉这些冗余：

| 阶段 | 分片保存的部分 | 仍需要理解的代价 |
|---|---|---|
| ZeRO-1 | 优化器状态 | 更新相关通信 |
| ZeRO-2 | 再加梯度 | 梯度分片/归约 |
| ZeRO-3 | 再加参数 | 层计算时临时汇集参数，通信更多 |

沿用 12P 字节教学账单，D=4，不计 master 与临时开销，理想常驻状态每卡：普通数据并行为 12P；ZeRO-1 为 $2P+2P+8P/4=6P$；ZeRO-2 为 $2P+(2P+8P)/4=4.5P$；ZeRO-3 为 $12P/4=3P$。实际峰值还包括被临时汇集的参数、buffer、激活等，不能把理想分片量当作卡上总内存。

分片与 offload 不同：offload 将部分状态放到 CPU 或其他存储，又引入传输延迟。它们可组合，但“省显存最多”不自动等于“训练最快”。[DeepSpeed ZeRO 官方教程](https://www.deepspeed.ai/tutorials/zero/)。

<span id="mm-809280b34eaa" style="display:block;scroll-margin-top:6rem"></span>

## 4. FlashAttention 为什么既是精确注意力，又能省内存

<span id="mm-324c5ff84b17" style="display:block;scroll-margin-top:6rem"></span>

### 4.1 问题在于把整张分数表搬来搬去

普通 attention 先形成 $S=QK^\top/\sqrt{d_h}$，再 softmax，再乘 V。长序列时 $T\times T$ 表很大，在 GPU 高带宽显存 HBM 与片上存储之间反复读写也很贵。

FlashAttention 将计算分块，利用在线 softmax 的合并性质，不把完整分数/概率矩阵长期存入 HBM；反向也可根据保存的统计量重算。它改变计算调度与数据搬运，不是把精确 dense attention 改成稀疏近似，也没有把全部算术复杂度从平方变线性。[FlashAttention 原论文](https://arxiv.org/abs/2205.14135)。

<span id="mm-179d7366e527" style="display:block;scroll-margin-top:6rem"></span>

### 4.2 在线 softmax 为什么可以合并

对一个 query，把可见 key 的分数记为 $s_j$，值为向量 $v_j$。输出为：

$$
o=\frac{\sum_j e^{s_j}v_j}{\sum_j e^{s_j}}.
$$

为了稳定，已处理部分维护最大分数 m、分母 $\ell=\sum e^{s_j-m}$、未归一化分子 $n=\sum e^{s_j-m}v_j$。新块到达后，令 $m'=\max(m,\max_{\rm new}s_j)$，把旧统计缩放到同一个最大值：

$$
\ell'=e^{m-m'}\ell+\sum_{\rm new}e^{s_j-m'},\qquad
n'=e^{m-m'}n+\sum_{\rm new}e^{s_j-m'}v_j.
$$

最后 $o=n'/\ell'$。所有块只需维护这些统计，而不是保存每个 key 的概率。首个非空块初始化后再用此式，避免对空块做无意义的无穷运算。

手算两个 key：第一分数 0、值 2，初始 $m=0,\ell=1,n=2$。第二分数 $\log2$、值 8，新的最大值是 $\log2$，旧统计乘 $e^{-\log2}=1/2$。于是 $\ell'=0.5+1=1.5$，$n'=1+8=9$，输出 6。一次完整 softmax 的权重恰好是 $(1/3,2/3)$，输出 $2/3+16/3=6$，结果一致。

这只是在线归一化的核心直觉；完整 FlashAttention 还涉及矩阵分块、GPU 内存层次和反向重算。数学目标等价不代表浮点运算顺序相同，因此与朴素实现可能有很小数值差异。

<span id="mm-3c97477f5c48" style="display:block;scroll-margin-top:6rem"></span>

### 4.3 与 KV Cache 的关系

FlashAttention 主要减少 attention 计算的中间物化与 IO；KV Cache 保存已生成位置的 K/V，避免下一次 decode 重算旧位置。一个针对算子执行，一个针对自回归跨步复用，可以一起使用。

若抽帧根本没看到关键动作，这两种优化都不会创造视觉证据。效率改善必须同时看时序问答质量与 token 预算，不能仅以显存下降声称“视频理解增强”。

<span id="mm-01f775d6ac63" style="display:block;scroll-margin-top:6rem"></span>

## 5. FP16、BF16 与数值稳定究竟在防什么

<span id="mm-3796bd901ed0" style="display:block;scroll-margin-top:6rem"></span>

### 5.1 范围与精细度是两个维度

二进制浮点数用符号、指数和尾数表示近似数值。指数位决定能容纳的数量级范围，尾数位影响同一数量级附近能区分多细。

| 类型 | 指数位 | 小数尾数位 | 最大有限正数约值 | 1 附近的间隔 |
|---|---:|---:|---:|---:|
| FP16 | 5 | 10 | $6.55\times10^4$ | $2^{-10}\approx0.0009766$ |
| BF16 | 8 | 7 | $3.39\times10^{38}$ | $2^{-7}=0.0078125$ |
| FP32 | 8 | 23 | $3.40\times10^{38}$ | $2^{-23}\approx1.19\times10^{-7}$ |

例如 1.001 在 FP16 中可接近 1.0009766，在 BF16 中可能舍入到 1。另一方面，数值 100000 超出 FP16 有限范围，却可被 BF16 表示。因此“BF16 总是精度更高”不正确；它的主要优势之一是更大的动态范围。

模型权重存 BF16，不意味着每个 reduction、softmax、优化器状态都必须用 BF16。关键累积常提升到 FP32；具体实现决定内存和误差，不能只看配置里一个 dtype 字符串。

<span id="mm-dc54477268ae" style="display:block;scroll-margin-top:6rem"></span>

### 5.2 Loss scaling 为什么有效，又为什么不能包治 NaN

FP16 最小正次正规数约 $5.96\times10^{-8}$，某些硬件路径还会冲掉次正规数。很小的梯度可能舍入为 0。若将 loss 乘以 S，则梯度也乘 S；反向后在高精度中除回 S，再更新参数，可减少反向中小梯度下溢。

例如真实梯度为 $10^{-8}$，直接转 FP16 可能为 0；取 $S=65536$，缩放后约 $6.55\times10^{-4}$，可进入 FP16 正常数范围。随后高精度 unscale 恢复其量级。这个例子只说明机制，真实反向有很多中间运算，不能保证任意一处下溢都被同样修复。

S 太大也可能溢出，所以动态 loss scaling 会检测非有限梯度、调整尺度，并可能跳过该次参数更新。范数裁剪通常必须在 unscale 后做，否则阈值对应的是错误尺度。BF16 因范围大通常不依赖同样的 loss scaling，但仍会受舍入、无效运算和巨大更新影响。

<span id="mm-94024886c04b" style="display:block;scroll-margin-top:6rem"></span>

### 5.3 稳定 softmax 与全 mask 行

直接算 $e^{1000}$ 会溢出。softmax 对整行加减同一常数不变，因此取 $m=\max_j z_j$：

$$
p_j=\frac{e^{z_j-m}}{\sum_ke^{z_k-m}}.
$$

例如 logits $[1000,1001,999]$ 先减 1001，变为 $[-1,0,-2]$，得到约 $[0.24473,0.66524,0.09003]$。同样的概率，不需要表示天文量级的指数。

但若一整行都被 attention mask 成 $-\infty$，则减最大值会出现 $-\infty-(-\infty)$，普通表达产生 NaN。必须按模型和内核约定处理无效 query、有效长度与 padding，而不是寄希望于“稳定 softmax 自动修一切”。交叉熵全 labels 忽略是另一个零分母问题，应检查有效监督 token 数。

<span id="mm-921189a822ab" style="display:block;scroll-margin-top:6rem"></span>

## 6. 用三个故障案例串起数据、梯度和指标

<span id="mm-7b1fc64262a3" style="display:block;scroll-margin-top:6rem"></span>

### 6.1 案例 A：第一步正常，随后 loss 变 NaN

NaN 是非有限值传播的结果，优先定位最早出现的位置。若 logits 已非有限，问题早于 loss；若 logits 有限、交叉熵 NaN，则可能是全部目标忽略、错误归一化或 loss 实现；若 loss 有限但梯度非有限，需看反向和混精。

一个最小 batch 可使问题可重复。关键观测是输入/labels 是否有效、第一处异常激活、loss dtype、梯度范数、优化前后参数是否有限。若第一个异常来自全 mask 行，调低学习率并没有修复原因；若来自更新后参数爆炸，学习率、梯度尺度、optimizer 状态与 loss scaling 才是直接候选。

理论上 loss 不变也不一定是数值溢出：可能全部训练参数被冻结、未进入 optimizer param groups，或梯度被错误 detach。需要看“该参数本应可训练吗、其梯度是否有限且非零、step 后是否变化”，而不是只看日志打印的 loss。

<span id="mm-b26f3e129b6a" style="display:block;scroll-margin-top:6rem"></span>

### 6.2 案例 B：loss 下降，但模型只会读字幕

假设任务要求识别“杯子先放下还是先拿起”，字幕里常有一句顺序提示。模型可能只拟合语言规律，视觉输入没有真正贡献。合理证据来自同一测试集的控制比较：

| 输入条件 | 可以检验什么 |
|---|---|
| 完整视频与字幕 | 原始任务表现 |
| 仅视频 | 是否能从画面取得证据 |
| 仅字幕 | 字幕捷径能解释多少成绩 |
| 字幕打乱/屏蔽，画面保持 | 对语言提示变化有多敏感 |
| 帧时序打乱，文本保持 | 对时序证据是否敏感 |

若仅字幕和完整输入几乎同分，可以怀疑视觉贡献有限；但不能直接断言“视觉完全没用”，因为测试集本身可能不需要视觉。时序打乱也要选真正依赖顺序的题，否则分数不变并无诊断力。

同时还要检查视觉特征是否真正进入 forward、连接器是否在训练、采帧是否覆盖关键动作。若关键事件落在两帧之间，再强的优化器也没有足够输入证据。更多采帧与评测解释见 [为什么图像和视频能够进入语言模型](/notes/vision-video-algorithms/)。

<span id="mm-7bb808e6efba" style="display:block;scroll-margin-top:6rem"></span>

### 6.3 案例 C：GRPO loss 变化，验证准确率不涨

先看 reward 是否评价了目标任务。若只奖励格式，模型学到稳定输出 JSON 也能让 reward 上升，却未提高时序事实。若答案解析把所有结果判错，组内优势可能全部为零。人审若干同题回答与机器分数，能发现这类目标错误。

然后看信号与更新两个层面：同题组内方差、全对/全错比例说明有没有可比较的信号；梯度、ratio、clip 和新旧概率差说明信号有没有造成更新。中心化优势可能让 policy loss 数值为零而梯度非零，详见 [从RL基础推到PPO与GRPO](/notes/policy-gradient-ppo-grpo/)；所以不能用标量零值独立判定“没训练”。

如果训练 reward 上升、独立验证不上升，再检查同视频泄漏、题型分布、奖励投机、输出截断和长度变化。只有这些证据支持时，才进一步把问题归因于 SFT 初始化、LoRA rank 或 GRPO 超参数；“某次 LoRA+GRPO 失败”不能证明 LoRA 普遍不适合 RL。

<span id="mm-49b86d08b6b2" style="display:block;scroll-margin-top:6rem"></span>

## 7. 什么样的实验结论经得住面试追问

“我调了学习率，分数涨了”不能区分变化来自随机波动、采样预算还是学习率。一次对照应把 base、数据拆分、视觉编码器/采样、训练预算、评测脚本与解码设置说清，并记录真正改变的因素。

视频要按源视频隔离 train/val/test，继续检查近重复片段、字幕、同题改写。只有随机打散问答行，很可能把同一视频证据放进不同集合。报告总体成绩时，还应给顺序、动作、OCR、长视频定位等与目标相关的分桶，否则一个容易题型可能掩盖关键能力退步。

消融是在同一证据标准下比较组件贡献。例如只改“均匀采帧”与“事件采帧”，固定帧数/分辨率预算并报告证据覆盖和准确率；若多给一倍帧再上涨，就同时改变了算法与成本。另一个可解释的对照是同数据与预算下比较全参和不同 LoRA rank，但其算力/训练状态差异应明确。

小评测集里少数题的变化未必稳健。可记录多次运行的均值与波动、逐题成功/失败变化或合适的置信区间；不同模型对同一题的结果是配对数据，不能把它们当毫无关联的两堆样本。具体统计选择需配合题目是否独立、视频是否成组，不能用一个通用显著性口号代替分析。

<span id="mm-fb1771ce2eed" style="display:block;scroll-margin-top:6rem"></span>

## 8. 面试回答与自测

**问题：为什么 7B 的 BF16 权重 14 GB，训练可能要远多于 14 GB？**  
参数只是第一项。全参训练还有梯度、Adam 两份状态、可能的 FP32 master、反向激活和临时 buffer。按 BF16 权重/梯度与 FP32 两矩的假设，四项已有 12 字节/参数，即 84 GB；长视频再增加激活。分片、量化与 LoRA 改变的是不同账单项，所以不能用一个固定倍数套所有实现。

**问题：FlashAttention 为什么省显存，但 attention 仍可能是平方计算？**  
它利用分块与在线 softmax，避免在 HBM 中物化完整分数/概率矩阵，减少 IO；仍需处理稠密 query-key 相互作用。减少存储与减少算术运算不是同一件事。可以用两个 key 的合并例子证明归一化结果不变。

**问题：为什么不能把所有优化措施都称为“省显存”然后一起加？**  
因为目标和代价不同：累积配合小 micro-batch 减激活峰值，checkpointing 用计算换激活，LoRA 减可训练状态，QLoRA 压冻结权重，ZeRO 分片冗余，FlashAttention 减中间物化。先测出主要项，才知道哪一种解释和对照最有意义。

**练习 1。**$[2,8192,4096]$ BF16 激活增加到两倍序列长度，占用如何变？若另保存朴素 $T\times T$ attention 表又如何变？

**答案。**前者 128 MiB 到 256 MiB，后者元素数变为四倍。总峰值是否同倍变化还依赖其他项及保存策略。

**练习 2。**梯度 $g=(6,8)$，全局阈值 C=5。norm clipping 的结果是什么？若当前梯度还有 S=100 的 loss scale，应在哪一步裁剪？

**答案。**原范数 10，结果 $(3,4)$。先把缩放梯度除回 S，再按真正阈值裁剪；否则阈值与真实更新尺度不匹配。

**练习 3。**两 micro-batch 有效 token 数为 1 和 3，loss sum 为 2 和 3。逐 micro 平均再平均与全 token 平均各是多少？

**答案。**前者 $(2+1)/2=1.5$，后者 $(2+3)/4=1.25$。使用哪一个会改变样本权重，必须与想训练的目标一致。

**练习 4。**GRPO 生成阶段 OOM，但反向能正常跑，优先考虑哪些量？

**答案。**并发组采样数、输入/输出长度、历史 KV、生成引擎与训练模型是否共同驻留、视觉处理峰值。不能只依据 LoRA 可训练参数量判断生成显存。

<span id="mm-c626ee98c56c" style="display:block;scroll-margin-top:6rem"></span>

## 来源与回链

- [图解仓库训练流程](https://github.com/changyeyu/LLM-RL-Visualized#header-8)；本页与原图的 attention/SFT/PPO 部分对应，图片和授权总览见 [学习总览](/notes/multimodal-interview-guide/)。
- [DeepSpeed ZeRO 官方](https://www.deepspeed.ai/tutorials/zero/)、[FlashAttention 原论文](https://arxiv.org/abs/2205.14135)。
- [AdamW 原论文](https://arxiv.org/abs/1711.05101)、[PyTorch AMP 官方文档](https://docs.pytorch.org/docs/stable/amp.html)：核对优化器与混精实现时使用；硬件/框架版本须以实际环境为准。
- [SFT与DPO：训练信号从哪里来](/notes/multimodal-sft-lora-dpo/) · [从RL基础推到PPO与GRPO](/notes/policy-gradient-ppo-grpo/) · [Prefill、Decode与视频token的计算代价](/notes/prefill-decode-video-tokens/) · [面试问题与项目深挖](/notes/multimodal-interview-questions/) · 参数高效微调。
