---
title: Prefill、Decode与视频token的计算代价
date: '2026-09-22'
tags:
- 多模态算法
- Prefill
- Decode
- KVCache
- FlashAttention
- 采样
summary: 用缓存mask、显存账单和online softmax解释生成速度与视觉预算。
draft: false
obsidian: true
series: multimodal-interview
---
<span id="mm-7f62acc39ee0" style="display:block;scroll-margin-top:6rem"></span>

一段长视频的问题，可能等很久才出现第一个字，之后却流畅地逐字输出；也可能首字很快，后面每一步都慢。两种现象对应的计算工作不同。本章沿着一次真实生成的时间顺序，解释 Prefill、KV Cache、Decode、视觉 token 成本和采样策略。

先读[从一个token理解Transformer与多模态入口](/notes/transformer-attention-rope-gqa/)中的 Q/K/V 形状。本章从输入形状、缓存状态和硬件代价解释 Prefill/Decode，技术依据为原论文和官方文档；对应口述题见[面试问题与项目深挖](/notes/multimodal-interview-questions/)。

<span id="mm-463ec88d4b5d" style="display:block;scroll-margin-top:6rem"></span>

## 1. 一次生成到底先后做了什么

假设 prompt 有 $N$ 个 token，我们要生成 $M$ 个 token。Prefill 把已知的 $N$ 个输入一起送进因果 decoder。每个输入位置只能读自己和以前的位置，但这些位置在同一次矩阵运算里可以并行计算，因为输入已经全部知道。

Prefill 最后位置的 logits 给出第一个输出 token 的分布。选出这个 token 后，把它作为新输入再前向一步，得到第二个输出 token；重复至 EOS 或长度上限。这里有一个容易错位的细节：Prefill 返回第一枚新 token 时，缓存往往还只包含 prompt；下一次前向消费这枚新 token 后，它自己的 K/V 才追加进去。

Decode 的串行来自“后一个输入 token 取决于前一次生成结果”。训练时 teacher forcing 用真实答案作为前缀，可以同时算所有目标位置；生成时没有未来真实答案。二者都用因果注意力，但可用的输入不同。

视频请求还先经过视频解码、抽帧、视觉编码和连接器。用户看到的首 token 时间（TTFT）可能包括排队、预处理、视觉塔、LLM Prefill 和 token 选择；不能把 TTFT 全记成语言模型的 Prefill 耗时。TPOT 是后续输出 token 的时间间隔，通常还需要报告平均值或分位数和统计口径。

<span id="mm-47856eae717c" style="display:block;scroll-margin-top:6rem"></span>

## 2. KV Cache 为什么成立，缓存的是什么

对一个普通因果 decoder，在确定性推理下，旧位置 $i$ 的隐藏状态只取决于前缀 $\le i$。新增位置并不改变这个前缀，因此旧位置在每一层的 K/V 都不变。保留它们，就不必在每一步重新计算全部历史位置。

注意这里是**每层分别**保存 K/V，不是只存最后一层表示，也不是存注意力概率矩阵。新位置有新的 Q，因为它发出的查询与旧 token 不同；它需要读取所有历史 K/V。旧 Q 完成当时的查询后，通常没有在未来步骤中复用的需要。

这条论证依赖因果结构和输入不变。如果视觉编码器对整段视频做双向处理，新增帧可能改变旧视觉特征；不能在这种上游变化后无条件复用下游缓存。普通固定视频前缀的 decoder 生成，才直接适用这里的推导。

<span id="mm-346a152726d8" style="display:block;scroll-margin-top:6rem"></span>

### 缓存形状与非方形 mask

已有 $P=3$ 个位置，一次追加 $U=2$ 个位置。当前 $Q$ 的长度是 2，合并历史后的 $K,V$ 长度是 5，分数形状为 $[B,H_q,2,5]$。若新位置的绝对索引为 3 和 4，允许矩阵应为

$$
A=
\begin{bmatrix}
1&1&1&1&0\\
1&1&1&1&1
\end{bmatrix}.
$$

已有 3 个位置而只追加 1 个时，允许行是 $[1,1,1,1]$。把一个 $1\times4$ 的“左上对齐下三角”当作增量 mask，则只允许看第 0 列，完全错误。可见条件应该按绝对位置算：$k_{\rm pos}\le q_{\rm pos}$，再结合有效 key/padding。

<figure style="margin:1.5rem 0">
<a href="/notes-assets/prefill-decode-video-tokens/17-%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%89%8B%E7%AE%97%E4%B8%8E%E7%BC%93%E5%AD%98%E6%8E%A9%E7%A0%81.svg" target="_blank" rel="noopener" aria-label="查看原图：注意力手算与缓存掩码"><img src="/notes-assets/prefill-decode-video-tokens/17-%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%89%8B%E7%AE%97%E4%B8%8E%E7%BC%93%E5%AD%98%E6%8E%A9%E7%A0%81.svg" alt="注意力手算与缓存掩码" width="1200" height="780" loading="lazy" decoding="async" style="height:auto;cursor:zoom-in" /></a>
<figcaption style="font-size:0.9em">注意力手算与缓存掩码（点击查看原图）</figcaption>
</figure>

**读图**：下方绿色格子表示新 query 能读的列，粉色格子是同批新 token 中仍处未来的一列。上方两 token 手算提醒，mask 改变的是 softmax 的归一化集合。代码接口不一定都用同样的布尔语义；PyTorch SDPA 中布尔 True 表示允许参与注意力，其他接口可能相反，必须看实际 API。[SDPA 官方文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)。

RoPE 也要用全局位置。缓存中第 2 个 K 已按位置 2 旋转，追加 token 的 Q/K 应按自己的位置 3 或 4 旋转；不应把每次输入片段都从位置 0 编号。多 batch 的 padding、滑动窗口或缓存淘汰还要求明确“绝对位置”与“缓存槽位”的区别。

<span id="mm-38c0c6fb220f" style="display:block;scroll-margin-top:6rem"></span>

## 3. 两个阶段的计算量与硬件瓶颈

考虑一个 dense decoder 层，隐藏宽度 $d$、FFN 中间宽度与 $d$ 成比例，暂不计词表输出头、视觉塔和特殊稀疏结构。Prefill 的投影/FFN 工作量约 $O(Nd^2)$，Attention 的 QK 和 PV 约 $O(N^2d)$，因此总量约

$$
O(Nd^2+N^2d).
$$

有缓存的单步 Decode 对一个新 token 做投影和 FFN，约 $O(d^2)$。令 $j$ 为已经消费的生成 token 数，当前新 Q 读取含本 token 的 $N+j$ 个 K/V，约 $O((N+j)d)$。$j=1$ 时，消费首枚输出、缓存增至 $N+1$，预测第二枚输出；因此生成 $M$ 枚输出通常只需 Prefill 后再做 $M-1$ 次 Decode。单步总量约

$$
O(d^2+(N+j)d).
$$

这比每一步把整个前缀重算便宜，但 GPU 是否充分利用，还取决于算术强度：每读一个字节做多少浮点运算。Prefill 有较长矩阵，能让同一组权重服务多个输入位置；batch 1 Decode 每步只有一个新位置，反复读权重和 KV，算力可能等着数据。

一个只用于理解的硬件算例：7B 权重以 BF16 保存约 14 GB；假设可用显存带宽恰为 1 TB/s，完整读取一次权重的理想下界约 14 ms。batch 1、单 token 前向的稠密线性权重主项约为 $2P=14$ GFLOPs，暂忽略 Attention 等额外算子。若可用算力恰为 100 TFLOP/s，纯计算下界约 0.14 ms。理想重叠下以两个下界的最大值衡量约束，不把二者机械相加。它说明只看 FLOPs 无法判断延迟。真实模型存在权重复用、量化、并行、内核开销和带宽利用率，这些数字不是某款 GPU 或某个服务的性能承诺。

增大 batch 可让同一权重服务多个序列，提高吞吐；但同时增加 KV、排队和延迟压力。长上下文、大 batch、不同硬件下瓶颈可以改变。因此面试中更准确的说法是：“Prefill 常更有利于大矩阵计算，低 batch Decode 常受内存带宽影响；我会在具体工作负载上分段测量。”

<span id="mm-af14fb846732" style="display:block;scroll-margin-top:6rem"></span>

## 4. 视频 token 怎样同时影响首字、缓存和逐字速度

普通 MHA/GQA 的缓存近似公式为

$$
M_{\rm KV}=2LBT H_{kv}d_hs.
$$

$2$ 是 K 与 V 两份，$L$ 是层数，$B$ 是并发序列数，$T$ 是缓存长度，$H_{kv}$ 是独立 KV 头数，$d_h$ 是每头宽度，$s$ 是每元素字节数。MLA、量化 KV、滑窗层、跨层共享等需用对应的存储结构，不能照搬普通公式。

取 $L=32,B=1,T=8192,H_{kv}=8,d_h=128,s=2$，结果是 $1{,}073{,}741{,}824$ 字节，正好 1 GiB。新增 1024 个视觉 token，会增加 128 MiB KV；若并发为 8，增量约 1 GiB。这个账单不含模型权重、视觉编码器、分配块、通信和临时工作区。

如果总输入从 8192 增至 9216，dense attention 的 $N^2$ 项增加到约 $1.266$ 倍；FFN 的 $N$ 项只增到 $1.125$ 倍；随后每步访问的历史也更长。不能把其中任何一个比值直接当成整体时延倍率。

<figure style="margin:1.5rem 0">
<a href="/notes-assets/prefill-decode-video-tokens/15-%E8%A7%86%E9%A2%91%E9%87%87%E5%B8%A7%E4%B8%8E%E8%AF%81%E6%8D%AE.svg" target="_blank" rel="noopener" aria-label="查看原图：视频采帧与证据"><img src="/notes-assets/prefill-decode-video-tokens/15-%E8%A7%86%E9%A2%91%E9%87%87%E5%B8%A7%E4%B8%8E%E8%AF%81%E6%8D%AE.svg" alt="视频采帧与证据" width="1100" height="390" loading="lazy" decoding="async" style="height:auto;cursor:zoom-in" /></a>
<figcaption style="font-size:0.9em">视频采帧与证据（点击查看原图）</figcaption>
</figure>

**读图**：视频时间轴上的选帧先决定哪些事件证据进入视觉塔，之后才形成 visual tokens 和语言模型前缀。减少帧数或压缩 patch 可以省预算，却可能先把待回答事件删掉；缓存或 attention 内核优化只改变已输入信息的处理成本。图的算法含义和漏采反例在[为什么图像和视频能够进入语言模型](/notes/vision-video-algorithms/)中展开。

<span id="mm-54d47931bd03" style="display:block;scroll-margin-top:6rem"></span>

## 5. FlashAttention、分页与量化分别解决什么

朴素 Attention 可能把 $N\times N$ 的分数和概率矩阵写到显存，再读回来乘 V。FlashAttention 通过分块及片上存储组织，减少这些中间结果的物化与显存 IO；在实数数学上实现同一 dense attention 目标，浮点实现仍可能有微小差异。它改变计算组织，不意味着 dense attention 的一般算术复杂度变成线性。[FlashAttention 原论文](https://arxiv.org/abs/2205.14135)。

为何不能每块独立 softmax 后直接相加？因为 softmax 分母跨越所有 key。假设第一块分数 $[0,0]$，第二块分数 $[10]$，各块独立归一会给第一块总权重 1、第二块总权重 1；全局 softmax 却应几乎全给分数 10。正确分块要维护全局归一化统计。

设已处理块有最大值 $m$、指数和 $\ell=\sum e^{s_i-m}$、加权分子 $n=\sum e^{s_i-m}v_i$；新块计算 $m_b,\ell_b,n_b$。合并时令 $m'=\max(m,m_b)$，

$$
\ell'=e^{m-m'}\ell+e^{m_b-m'}\ell_b,\qquad
n'=e^{m-m'}n+e^{m_b-m'}n_b.
$$

最终输出 $n'/\ell'$。接着把上例算完：第一块分数 $[0,0]$、标量 V 为 $[1,3]$，统计为 $(m,\ell,n)=(0,2,4)$；第二块分数 $[10]$、V 为 $[5]$，统计为 $(10,1,5)$。合并得到 $\ell'=1+2e^{-10}$、$n'=5+4e^{-10}$，输出约为 $4.999727625$，与对三个分数一起 softmax 再加权相同。真实 V 有多维时，$n$ 也是同宽向量，缩放逐元素作用。

这个在线 softmax 恒等式说明可以不保存完整概率矩阵而保留正确归一化；全被 mask 的空块要单独跳过或使用安全初始化，不能直接计算 $-\infty-(-\infty)$。实际内核还涉及反向重算、布局与硬件优化。

KV Cache 解决历史 K/V 重算；分页 KV 管理解决缓存块的分配、碎片及复用；KV 量化降低每元素存储，但引入误差和反量化开销；GQA 改模型头共享；视觉压缩改输入内容量。它们可以组合，作用层次不同。标准 teacher-forced SFT 的显存重点是训练激活/梯度/优化器，生成缓存则主要出现在推理或 RL rollout，详见[训练显存与实验排障：把机制变成可检查的量](/notes/training-memory-debugging/)。

<span id="mm-d9647fdd28e1" style="display:block;scroll-margin-top:6rem"></span>

## 6. Greedy、温度、Top-k 与 Top-p

模型输出分布后，还要决定“选哪一枚 token”。Greedy 每次取最大概率；多项式采样按概率抽样。二者用同一组 logits，却产生不同轨迹；这直接影响 GRPO 能否在一组回答里采出差异。

温度 $\tau>0$ 把 logits 缩为 $z/\tau$ 再 softmax。低温通常更尖锐，高温通常更分散；它不创造模型原本没学到的正确知识。用概率写，可视为 $p_i^{1/\tau}$ 再归一。$\tau\to0$ 的极限接近选最大值，遇到并列最大值仍需定义选择规则。

Top-k 保留概率最高的 $k$ 个候选再归一；Top-p 保留累计概率达到阈值的最小前缀集合。本例明确采用“按概率降序，累计首次大于等于 $p$ 就停止”的规则：概率依次为 $[0.5,0.3,0.15,0.05]$，Top-k=2 或 Top-p=0.8 均留下前两个，归一后为 $[0.625,0.375]$。Top-p=0.9 则还需第三个，归一后约为 $[0.5263,0.3158,0.1579]$。浮点相等、过滤边界及与其他采样器的组合顺序可能因库不同而变化，需核查实际实现。

Beam search 保留若干条高分前缀，每步扩展并筛选。常用序列分数是 token log-prob 之和，长度不同会产生偏好，可能需要明确的长度归一；它并不是独立采样多次的同义词。评估 pass@G 时，用 beam、greedy 重复或随机采样得到的 $G$ 条结果不能不加说明地比较。

视频对照实验至少固定 temperature、top-p/top-k、最大输出长度和随机种子策略。否则“换了采帧方法后涨分”可能同时混入输出分布变化。

<span id="mm-c7a54d7876a7" style="display:block;scroll-margin-top:6rem"></span>

## 面试检查与练习入口

“QKV 如何实现”常追到形状、mask、缓存位置；“Prefill/Decode 为什么不同”常追到完整计算项、带宽与 KV 账单；“视频为什么慢”常追到视觉塔、帧数、分辨率和语言前缀。回答时给一个可检验的数字和一个适用条件，比只背结论更有用。

[十四天练习册：从手算到多模态面试](/notes/multimodal-fourteen-day-workbook/)安排了缓存可见矩阵、64 MiB 增量计算和采样归一练习。它还给出逐项检查路线：同一模型下，改变未来 token 不应影响过去 logits；关闭随机 dropout 后，整段 causal forward 与按 token 缓存 forward 应在合理浮点容差内一致。这些是实现验证的目标，不代表本文已训练或测试某款 7B 模型。

来源：[GQA](https://arxiv.org/abs/2305.13245)、[FlashAttention](https://arxiv.org/abs/2205.14135)、[PagedAttention](https://arxiv.org/abs/2309.06180)、[Top-p / Nucleus Sampling](https://arxiv.org/abs/1904.09751)、[PyTorch SDPA](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)。两幅图均为本教程自绘；仓库原图使用说明见[来源、图像许可与知识点覆盖](/notes/multimodal-sources-coverage/)。
