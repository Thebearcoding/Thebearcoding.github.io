---
title: 序列推荐与长序列建模：从 SASRec 到 TWIN、LONGER
date: '2026-10-05'
tags: [推荐系统, 序列推荐, SASRec, 长序列, SIM, TWIN]
summary: 先讲清 next-item 序列推荐的训练目标，再看行为序列从几十条变成上万条时，SIM、ETA、SDIM、TWIN、LONGER 各自把检索、注意力和服务成本换成了什么。
draft: false
---

用户历史在推荐里有两种用法。**序列推荐**把历史当输入，直接预测下一个物品，是召回和生成式推荐的基础任务。**行为序列特征**出现在精排里：给定一个候选，从历史中读出与它相关的兴趣，喂给 CTR 模型。两者都要回答“历史这么长，读哪一段、怎样读得起”。

阅读入口：[全覆盖教程](/notes/recsys-llm4rec-guide/)。DIN 与 SIM 的入门在 [排序笔记](/notes/wangshusen-recommender-ranking/) 第 10–11 节，本篇从那里接着讲。数值为教学构造。

## 1. 序列推荐：给定前 t 个，预测第 t+1 个

用户历史 $(i_1,\dots,i_T)$。训练时每个位置都做一次预测：用 $(i_1,\dots,i_t)$ 预测 $i_{t+1}$。评估常用留一法，最后一个物品当测试目标，指标是全库排序下的 HR@K、NDCG@K（见 [评估与偏差](/notes/recsys-eval-bias/)）。

**GRU4Rec** 用循环网络逐步更新隐状态，最早用于会话推荐。**SASRec** 用带因果 mask 的自注意力：位置 $t$ 只能看见 $1..t$，所以一次前向就得到所有位置的预测，训练高效。输出向量与物品 embedding 做内积得到分数。

**BERT4Rec** 改成双向注意力，训练时随机遮住一部分物品让模型补全（Cloze 目标），推理时在序列末尾放一个遮罩位预测下一个。双向让每个位置看得更全，但训练目标和“预测下一个”的推理目标不完全一致。

## 2. 损失函数比架构更常决定结果

SASRec 原论文对每个位置用二元交叉熵，只配一个随机负样本。后来的复现发现，**把损失换成全库 softmax 交叉熵（或足够多负样本的采样 softmax）**，SASRec 能明显变强，甚至超过 BERT4Rec，见 Klenitskiy 与 Vasilev 的 *Turning Dross Into Gold Loss: is BERT4Rec really better than SASRec?*（RecSys 2023）。

这件事对你有两个用处。第一，比较两个序列模型时，先确认损失、负样本数、训练轮数一致。第二，评价生成式推荐“胜过 SASRec”的说法时，要问基线用的是哪种损失，这是“生成式推荐是不是伪范式”讨论的核心之一，见 [LLM4Rec](/notes/recsys-llm4rec-paradigms/)。

| 训练目标 | 每个位置的负样本 | 代价 | 风险 |
|---|---|---|---|
| BCE + 1 个随机负样本 | 1 | 最低 | 负样本太简单，模型过度自信 |
| 采样 softmax | 几十到几千 | 中 | 需要 logQ 纠正采样分布 |
| 全库 softmax | 全部物品 | 物品数大时显存高 | 物品数上千万时不可行 |

## 3. 精排里的行为序列：从 DIN 到 DIEN

精排里，候选物品 $q$ 已经给定。DIN 用 $q$ 对每条历史打一个相关性权重，再加权求和：

$$
h_u(q)=\sum_{j=1}^{L}a(q,e_j)\,e_j .
$$

$h_u(q)$ 随候选变化，所以每个候选都要对整段历史算一遍，成本是 $O(L)$ 每候选。一个请求有 $N$ 个候选，就是 $O(NL)$。

**DIEN** 在 DIN 前加了一层兴趣抽取 GRU，并用辅助损失让每一步的隐状态能预测下一次行为；再用注意力控制 GRU 更新门（AUGRU），让与候选相关的兴趣演化得到更多更新。它刻画了兴趣随时间的变化，但循环结构难以并行，序列一长就慢。

## 4. 为什么要“长”，又为什么难

几十条最近行为只反映短期兴趣。一年的历史里有季节性、低频但稳定的爱好，对长尾和老用户召回很有价值。问题在成本：$L$ 从 50 变成 10,000，$O(NL)$ 的目标注意力放大 200 倍，线上精排通常只有几十毫秒。

于是几乎所有方案都是**两阶段**：先用便宜的方法从长历史里挑出几百条和候选相关的，再对这几百条做精细的目标注意力。工业论文常把前者叫 GSU（General Search Unit），后者叫 ESU（Exact Search Unit）。

<figure>
<a href="/notes-assets/recsys-llm4rec/long-seq.svg" target="_blank" rel="noopener" aria-label="查看长序列两阶段原图"><img src="/notes-assets/recsys-llm4rec/long-seq.svg" alt="长序列两阶段：上万条历史经 GSU 用类目、哈希或简化注意力挑出 Top-K，再由 ESU 对 Top-K 做目标注意力，得到兴趣向量送入 CTR 模型" width="1100" height="440" loading="lazy" style="height:auto;cursor:zoom-in" /></a>
</figure>

各方案的差别，就在 GSU 用什么相关性、和 ESU 是否一致。

## 5. SIM：先查，再读

**SIM** 的 GSU 有两种：hard search 按类目查，只保留与候选同类目的历史；soft search 用 embedding 内积查最相近的 Top-K。ESU 对 Top-K 做多头目标注意力，并把“这条行为距今多久”分桶做 embedding 拼进去，让模型区分昨天和半年前。

问题在 **GSU 与 ESU 不一致**：类目相同不代表 ESU 会认为它重要，soft search 的向量也和 ESU 的注意力分数不是同一个函数。GSU 漏掉的行为，ESU 无从补救。

## 6. ETA 与 SDIM：用哈希代替内积

**ETA** 把物品 embedding 用 SimHash 编成 $k$ 位二值签名：随机取 $k$ 个超平面 $r_1,\dots,r_k$，每位记录 $\mathrm{sign}(v\cdot r_i)$。两个向量夹角为 $\theta$ 时，单个位相同的概率是

$$
\Pr[h(u)=h(v)]=1-\frac{\theta}{\pi}.
$$

所以 Hamming 距离近似反映夹角。GSU 用 XOR 与 popcount 比较候选和每条历史的签名，取 Top-K；因为签名来自同一套 embedding，可以端到端训练，GSU 与 ESU 的相关性更接近。跨模态哈希里的 Hamming 检索，在这里就成了长序列的检索器。

**SDIM** 更进一步：不检索 Top-K，而是**采样**。对候选和每条历史计算多组哈希签名，把与候选签名相同的历史直接聚合（求和后归一化）。期望上，一条历史被聚合进来的概率随它和候选的夹角单调变化，所以聚合结果近似一个“按相似度加权”的注意力，却不需要逐条算 softmax。历史的签名可以离线算好，线上几乎只剩查表。

| 方法 | GSU 相关性 | 是否端到端 | 主要省掉的计算 |
|---|---|---|---|
| SIM hard | 类目相同 | 否 | 全部历史的注意力 |
| SIM soft | embedding 内积 | 部分 | 同上 |
| ETA | SimHash 的 Hamming 距离 | 是 | 高维内积换成位运算 |
| SDIM | 哈希碰撞直接聚合 | 是 | 连 Top-K 与 softmax 都省掉 |

## 7. TWIN：让 GSU 也用目标注意力

快手的 **TWIN** 的出发点是：既然不一致是问题，就让 GSU 直接用和 ESU 同一种目标注意力打分，只是想办法算得起。

关键是把每条行为的特征拆成两部分。**固有特征**只和物品有关（物品 ID、类目、作者、时长），它们投影到 Key 的结果可以离线按物品缓存。**交叉特征**和用户-物品对有关（观看时长、距今时间），无法缓存，就压缩成一个标量偏置加到注意力分数上。这样 GSU 对上万条行为打分时，主要计算是一次查表加一次内积。

**TWIN V2** 面对更长的终身序列，先离线做层次聚类：按类目分组，组内按 embedding 聚类，把每个簇用一个“虚拟物品”代表。线上对簇做注意力，并在分数上加 $\ln n$（$n$ 为簇大小），让包含更多行为的簇得到相应的权重：

$$
\alpha'_c=\alpha_c+\ln n_c .
$$

直觉：softmax 后 $e^{\alpha_c+\ln n_c}=n_c\,e^{\alpha_c}$，相当于簇里每条行为各贡献一份。

## 8. 用 Transformer 直接吃长序列

2025 年前后，字节、抖音等团队开始在精排里直接用 Transformer 处理几千到上万条行为，并报告了模型规模扩大带来的稳定收益。难点全在算力，常见的降本办法有四类：

1. **全局 token**。把候选和用户画像做成少数几个全局 token，放在序列前面，所有位置都能看到，也起到注意力“锚点”的作用（LONGER）。
2. **token 合并**。把相邻的若干条行为合并成一个 token，序列长度按比例缩短（LONGER 用一个小 Transformer 做组内合并）。
3. **只让少数位置当 Query**。第一层用全局 token 和最近的一部分行为做 Query、全部行为做 Key/Value，后续层只在这些位置上做自注意力；或者干脆只让候选当 Query、历史当 Key/Value，变成线性复杂度的交叉注意力（STCA）。
4. **请求内复用**。同一个请求的上千个候选共享同一段用户历史，把与候选无关的计算只做一次，再和每个候选组合；配合 KV cache 与混合精度。

这些方法的共同点：**把与候选无关的部分（用户侧）和与候选有关的部分（交叉）彻底拆开**，前者每个请求算一次，后者尽量轻。这和双塔“可分离计算”的思想一脉相承。

## 9. 多模态进入长序列

ID embedding 对长尾物品学得差，而终身序列里大部分是长尾。**MUSE**（淘宝）用多模态向量做长序列建模：GSU 直接用多模态 embedding 的余弦相似度挑行为，ESU 把基于 ID 的注意力和多模态相似度融合。它借鉴了把相似度分档统计成直方图特征（SimTier）的做法，让模型知道“历史里和候选很像的有多少条、一般像的有多少条”。

Meta 的 **DV365** 走了另一条路：把用户一年多的行为离线压成一个稳定的用户向量，由上游模型产出，下游各个推荐模型直接使用，而不是每个模型各自读长序列。

对你的意义：多模态表征在推荐里最直接的落点之一，就是**长序列的检索器与相似度特征**。这正是 CLIP 背景能发挥的地方，详见 [多模态推荐与 I2I](/notes/recsys-multimodal-i2i/)。

## 10. 怎样选

| 场景 | 推荐起点 | 理由 |
|---|---|---|
| 历史几十条，精排 | DIN | 简单，可解释 |
| 历史上千、延迟紧 | SIM hard 或 ETA | 实现成本低 |
| 历史上万、追求效果 | TWIN 类一致性 GSU | 检索与精读一致 |
| 算力充足、追求 scaling | Transformer 化长序列 | 规模扩大有收益，但工程量大 |
| 长尾占比高 | 多模态 GSU（MUSE 思路） | ID 向量在长尾上不可靠 |

## 闭卷验收

写出 SASRec 训练时一个长度为 5 的序列产生几个预测、因果 mask 长什么样；解释为什么换成全库 softmax 会让基线变强；画出 GSU/ESU 两阶段，并分别说出 SIM、ETA、SDIM、TWIN 的 GSU 用什么相关性；推一遍 TWIN V2 里加 $\ln n$ 的含义；最后说出 Transformer 化长序列的四种降本手段。

**参考。** [SASRec](https://arxiv.org/abs/1808.09781)；[BERT4Rec](https://arxiv.org/abs/1904.06690)；Klenitskiy & Vasilev, *Turning Dross Into Gold Loss*, RecSys 2023；[DIN](https://arxiv.org/abs/1706.06978)；[DIEN](https://arxiv.org/abs/1809.03672)；[SIM](https://arxiv.org/abs/2006.05639)；[ETA](https://arxiv.org/abs/2108.04468)；[SDIM](https://arxiv.org/abs/2205.10249)；[TWIN](https://arxiv.org/abs/2302.02352)；[TWIN V2](https://arxiv.org/abs/2407.16357)；[LONGER](https://arxiv.org/abs/2505.04421)；[MUSE](https://arxiv.org/abs/2512.07216)。选题参考 [算法小小怪下士的长序列建模笔记](https://github.com/cst20/LLM4Rec_xiaoxiaoguai)，正文依据原论文重写。
