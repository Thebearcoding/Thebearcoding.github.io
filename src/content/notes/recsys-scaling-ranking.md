---
title: 排序模型的 scaling：从 DLRM 到 HSTU、RankMixer、OneTrans
date: '2026-10-05'
tags: [推荐系统, 排序, scaling, HSTU, RankMixer, OneTrans]
summary: 推荐排序模型为什么长期“变不大”，以及序列路线（HSTU、MTGR）和特征 token 路线（RankMixer、OneTrans）分别怎样把参数和算力花在 GPU 擅长的地方。
draft: false
---

大语言模型的经验是：参数和算力按比例增加，效果按幂律稳定提升。推荐排序模型过去很少有这种现象，加深 MLP、加宽交叉层常常收益很快饱和。本篇回答两个问题：**传统排序模型卡在哪里；最近的工业工作分别用什么结构让模型“变大有用”。**

阅读入口：[全覆盖教程](/notes/recsys-llm4rec-guide/)。FM、DCN V2、MMoE 等经典结构见 [排序笔记](/notes/wangshusen-recommender-ranking/)。数值为教学构造。

## 1. DLRM 的参数在哪里

典型的深度推荐模型（DLRM 一类）由三部分组成：稀疏特征（用户 ID、物品 ID、类目等）各自查一张 embedding 表；稠密特征过一个小 MLP；所有向量做交叉（内积、FM、DCN 等）后送进顶层 MLP 输出概率。

参数绝大部分在 embedding 表里。一亿个物品 ID、64 维、float32，就是 $10^8\times64\times4$ 字节，约 25.6 GB；而稠密部分可能只有几千万参数。embedding 查表是**访存密集**的：每个样本只读少量行，几乎没有计算。这导致两个后果：

1. **加大 embedding 维度很快饱和**。长尾 ID 本来就没多少样本，维度越大越学不动。
2. **GPU 利用率低**。用 MFU（Model FLOPs Utilization，实际达到的浮点运算速率除以硬件峰值）衡量，传统排序模型在 GPU 上往往只有个位数百分比，算力大多在等内存。

要让模型变大有用，就要把参数和计算挪到稠密、规整、能用大矩阵乘的地方。

## 2. 两条路线

| 路线 | 代表 | 怎样组织输入 | 扩展的是什么 |
|---|---|---|---|
| 序列路线 | HSTU、MTGR | 把用户的“物品 + 行为”排成一条长序列 | 序列模型的层数、宽度与序列长度 |
| 特征 token 路线 | RankMixer、OneTrans | 把异构特征分组成若干 token | token 内与 token 间交互的宽度与深度 |

两条路线并不互斥：OneTrans 就把行为序列和非序列特征放进同一个 token 序列。

## 3. HSTU：把推荐写成序列转导

Meta 的 HSTU（*Actions Speak Louder than Words*，ICML 2024）把用户历史写成交替的序列：用户看了什么、对它做了什么动作。排序和召回都改写成“给定前面的序列，预测下一个物品或下一个动作”。

一层 HSTU 的核心计算（省略归一化与残差的细节）：

$$
U,V,Q,K=\mathrm{Split}\big(\phi(f_1(X))\big),\qquad
A=\phi\big(QK^\top+\mathrm{rab}\big),\qquad
Y=f_2\big(\mathrm{Norm}(AV)\odot U\big).
$$

$\phi$ 是 SiLU，$\mathrm{rab}$ 是基于位置和时间间隔的相对注意力偏置。和标准 Transformer 最大的不同是**注意力不做 softmax 归一化**。softmax 把每一行权重归一到和为 1，会抹掉“相关行为有多少条”这一强度信息：用户看过 50 条篮球视频和看过 1 条，归一化后对篮球候选的贡献可能差不多。逐点的 SiLU 保留了这种强度。$U$ 起门控作用，代替了 FFN。

**训练**用生成式目标：沿序列预测下一个物品，每个位置都产生监督，一条用户序列抵得上许多条独立样本。**推理**时排序要给上千个候选打分，论文提出 M-FALCON：把候选分成小批，共享用户序列部分的计算与缓存，只为候选额外算一小段。论文报告模型质量随训练算力在约三个数量级上呈幂律提升。

**边界。** HSTU 去掉了大量手工交叉特征。美团的 MTGR 认为交叉特征（用户-物品统计、上下文）在本地生活场景仍然重要，于是在类似架构上把交叉特征也作为 token 保留，并做了按用户聚合样本等训练加速。是否保留交叉特征取决于业务，不存在统一答案。

## 4. RankMixer：为 GPU 设计的特征交互

字节的 RankMixer 走特征 token 路线，出发点是：自注意力适合同质的 token（文字、行为），却不适合异构特征，几百个特征彼此算注意力既贵又没有必要。它由三件事组成。

**特征 token 化。** 把几百个特征按语义分组（用户画像、物品属性、上下文……），每组拼接后投影成一个 $D$ 维 token，得到 $T$ 个 token。

**多头 token mixing。** 把每个 token 切成 $H$ 段，第 $h$ 个新 token 由所有旧 token 的第 $h$ 段拼成。这个操作没有参数，只是一次重排，却让每个新 token 都含有所有特征组的信息。手算 shape：$T=4$、$D=8$、$H=4$，每个旧 token 切成 4 段、每段 2 维；新 token $h$ 由 4 个旧 token 的第 $h$ 段拼成，维度 $4\times2=8$，仍是 4 个 8 维 token。

**逐 token 的 FFN。** 每个 token 用自己的一套 FFN 参数，而不是所有 token 共享，因为不同特征子空间的分布差别很大。参数再大时用稀疏 MoE，每个 token 只激活少数专家。

这套结构几乎全是规整的矩阵乘，能把 GPU 吃满。后续的 TokenMixer-Large 沿这条路继续扩大规模。

## 5. OneTrans：一个 Transformer 同时做序列和交叉

传统精排是“先编码、再交互”：行为序列用 DIN/Transformer 编码成向量，再和其他特征做交叉。OneTrans 把两者合并：

1. **统一 token 化**。行为序列的每条行为是一个 token；非序列特征分组投影成若干 token；全部排成一条序列。
2. **混合参数**。同质的序列 token 共享参数；异质的非序列 token 各用自己的参数，这一点与 RankMixer 的逐 token FFN 思路一致。
3. **只让需要的位置继续往上走**。随着层数增加逐步减少参与计算的序列 token（金字塔式堆叠），控制长序列的成本。
4. **跨请求复用**。用户侧 token 与候选无关，算好的 Key/Value 可以缓存，被同一用户的多个候选、甚至相邻请求复用。

## 6. token 从哪来：三种思路

| 思路 | 例子 | 优点 | 风险 |
|---|---|---|---|
| 按字段语义分组 | RankMixer | 可解释，组内同质 | 分组靠人工，组数影响效果 |
| 投影或自动切分 | OneTrans 的非序列特征 | 少人工 | token 边界没有语义 |
| 语义 ID 作为 token | STORE 等 | 物品用少量离散码表示，长尾共享参数 | 依赖 tokenizer 质量，见 [语义 ID](/notes/recsys-semantic-id/) |

## 7. 怎样证明“变大有用”

自己做实验或讲论文时，scaling 的证据要满足三条：

1. **横轴是算力或参数，纵轴是离线指标，画在对数坐标上**，看是否接近直线，而不是只给两个点。
2. **固定数据与训练轮数，只改规模**；规模变大的同时改了特征或损失，就分不清来源。
3. **同时报告 MFU 与线上延迟**。离线涨了但推理时延翻倍，工业上不一定能上线。

## 8. 面试怎么讲

被问“推荐模型为什么不像 LLM 那样 scaling”，可以按这个顺序答：参数集中在 embedding 表，访存密集、MFU 低；特征异构，标准注意力不合适；于是出现两条路线，序列路线（HSTU）把推荐写成序列转导、用生成式训练获得密集监督，特征 token 路线（RankMixer）设计无参数的 token mixing 与逐 token FFN；两者都把与候选无关的计算拆出来复用。最后补一句边界：交叉特征在一些业务里仍然关键。

## 闭卷验收

估算一张一亿行、64 维 float32 的 embedding 表多大；写出 HSTU 注意力和标准注意力的区别，并举一个 softmax 抹掉强度信息的例子；手算 RankMixer 在 $T=6$、$D=12$、$H=6$ 时 token mixing 后的 shape；说出 OneTrans 的混合参数是什么意思；给出证明 scaling 的三条要求。

**参考。** [DLRM](https://arxiv.org/abs/1906.00091)；[HSTU](https://arxiv.org/abs/2402.17152)；[MTGR](https://arxiv.org/abs/2505.18654)；[RankMixer](https://arxiv.org/abs/2507.15551)；[TokenMixer-Large](https://arxiv.org/abs/2602.06563)；[OneTrans](https://arxiv.org/abs/2510.26104)。选题参考 [算法小小怪下士的可扩展 token 排序笔记](https://github.com/cst20/LLM4Rec_xiaoxiaoguai)，正文依据原论文重写。
