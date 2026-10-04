---
title: 推荐系统召回：从协同过滤到双塔和向量索引
date: '2026-10-04'
tags: [推荐系统, 王树森, 召回, 双塔, 对比学习]
summary: 用手算串起 ItemCF、Swing、UserCF、采样纠偏、自监督、Deep Retrieval 与线上索引。
draft: false
---

召回的输入是一个用户及其当前上下文，输出是一个规模可控的候选集。问题不只是“怎样算兴趣”，还包括**怎样避免给整个物品库逐条运行一个昂贵模型**。协同过滤把计算搬到离线邻居索引；双塔把物品提前编码进向量索引；Deep Retrieval 则直接学习可搜索的路径。

阅读入口：[课程与笔记来源](/notes/wangshusen-recommender-guide/)。本页所有具体数字为教学构造；沿用课程术语“矩阵补充”，也可在其他资料中看到矩阵分解、矩阵补全等相关表述。

## 1. ItemCF：沿着喜欢过的物品找邻居

记 $H_u$ 为用户最近交互的物品，$r_{uj}$ 为交互强度，$s(j,i)$ 为物品相似度。候选物品的分数为

$$
\hat r_{ui}=\sum_{j\in H_u}r_{uj}s(j,i).
$$

如果用户对两条历史内容的强度为 $(3,1)$，候选 A 与它们的相似度为 $(0.4,0.2)$，则 A 得分 $3\times0.4+1\times0.2=1.4$；候选 B 的相似度为 $(0.1,0.8)$，得分为 1.1。这里的强度是人为示意，真实系统需要决定点击、收藏、近期行为和负反馈怎样计权。

最简单的二值相似度把物品表示成“哪些用户喜欢它”的稀疏向量。记喜欢物品 $i$ 的用户集合为 $U_i$，则

$$
s(i,j)=\frac{|U_i\cap U_j|}{\sqrt{|U_i||U_j|}}.
$$

若两条内容分别被 16、25 个用户喜欢，共同用户有 8 个，相似度为 $8/\sqrt{16\times25}=0.4$。若采用非二值行为，余弦分母应是各行为向量的欧氏范数，不能把加权次数直接塞回集合人数公式。

线上维护两类索引：用户→近期交互物品，物品→Top-K 相似物品。先取最近 $n$ 条种子，再展开每条的 $k$ 个邻居，最多形成 $nk$ 条路径；合并同一候选的贡献，过滤已看或不可用内容，最后取 Top-M。候选数往往小于 $nk$，因为邻居重叠。[ItemCF 课件](https://github.com/wangshusen/RecommenderSystem/blob/main/Slides/02_Retrieval_01.pdf)

**边界。** 共现反映的是用户行为，不自动等于内容语义。两个内容可能因为一起进了群聊而共现；没有历史交互的新物品也没有稳定邻居。离线计算可以用共现倒排等稀疏方法避免朴素全库两两枚举，但热门用户、热门物品仍会放大计算量。

## 2. Swing：共同用户也可能来自同一个小圈子

若一群彼此高度相似的用户同时喜欢两条内容，几十次共现未必代表几十份独立证据。Swing 额外考虑共同用户之间喜欢物品的重合度。记 $V=U_i\cap U_j$，$I_a$ 为用户 $a$ 喜欢的物品，一种用户对求和写法是

$$
s_{\rm swing}(i,j)=\sum_{\{a,b\}\subset V,\ a\ne b}
\frac{1}{\alpha+|I_a\cap I_b|},\qquad\alpha>0.
$$

这里采用无序、不含自身的用户对，便于手算。课程 slides 写成双重求和；实际实现要统一有序/无序、自身项与附加权重的约定，不能直接比较不同约定的绝对分值。

取 $\alpha=1$。两条内容的某对共同用户还共同喜欢 19 条其他或相关物品，贡献为 $1/(1+19)=0.05$；另一对仅重合 1 条，贡献为 0.5。前一对高度重合，更像一个小圈子，因此权重低。这个计算不是在证明群聊关系，而是在用行为重合度作为代理。[Swing 课件](https://github.com/wangshusen/RecommenderSystem/blob/main/Slides/02_Retrieval_02.pdf)

## 3. UserCF：沿着相似用户找物品

UserCF 对调了邻居方向：用户→相似用户→这些用户喜欢的内容。记 $N_u$ 为用户邻居，预测分数为

$$
\hat r_{ui}=\sum_{v\in N_u}s(u,v)r_{vi}.
$$

邻居相似度为 $(0.8,0.3)$，对某物品的行为强度为 $(1,4)$，分数就是 2.0。用户相似度可以从共同喜欢的物品计算，但所有人都看过的热门内容区分力低，应考虑降低热门物品的贡献。

课程强调热门度修正；工程上仍要验证它是否伤害真实的共同兴趣。ItemCF 与 UserCF 的差异是索引方向和行为统计，不是一个负责短期、另一个天然负责长期。两者都受到行为窗口、更新频率和活跃度影响。[UserCF 课件](https://github.com/wangshusen/RecommenderSystem/blob/main/Slides/02_Retrieval_03.pdf)

## 4. 从 ID 查表到矩阵补充

类别编号本身不表达距离：城市 3 与城市 4 不比城市 3 与城市 100 更相似。Embedding 矩阵 $E\in\mathbb R^{N\times d}$ 将类别 ID 映射到可训练向量。查第 $i$ 行等价于 one-hot 向量乘矩阵，但实现通常直接查表。

一百万个 ID、每个 64 维、FP32 权重，需要 $10^6\times64\times4=256,000,000$ 字节，约 244 MiB；这是权重本身，训练还可能需要梯度、优化器状态和分布式存储。哈希桶减少表大小，却可能让不同 ID 冲突；OOV/default 让未知类别有入口，却不会自动学会新物品特性。[离散特征课件](https://github.com/wangshusen/RecommenderSystem/blob/main/Slides/02_Retrieval_04.pdf)

只用用户 ID 向量 $a_u$ 和物品 ID 向量 $b_i$，通过内积 $a_u^\top b_i$ 拟合交互，是矩阵分解类模型的基本形式。真实日志里未观察到的格子不是可靠的“用户不喜欢”，因此如何构造负样本是任务的一部分。纯 ID 模型还很难处理新 ID，侧信息正是双塔的重要补充。[矩阵补充课件](https://github.com/wangshusen/RecommenderSystem/blob/main/Slides/02_Retrieval_05.pdf)

## 5. 双塔的价值来自可分离计算

<figure>
<a href="/notes-assets/wangshusen-recommender/two-tower.svg" target="_blank" rel="noopener" aria-label="查看双塔召回原图"><img src="/notes-assets/wangshusen-recommender/two-tower.svg" alt="物品塔离线构建版本化索引，用户塔在线产生查询向量，通过近似检索得到候选" width="1100" height="420" loading="lazy" style="height:auto;cursor:zoom-in" /></a>
<figcaption>双塔的离线与在线分工。图为自绘，未表示具体平台的全部服务；点击查看原图。</figcaption>
</figure>

用户塔 $f_\theta(x_u)$ 编码 ID、画像、历史和场景；物品塔 $g_\phi(x_i)$ 编码 ID、内容与属性，输出同维向量。若采用归一化向量，兴趣分数可以写成

$$
a_u=\frac{f_\theta(x_u)}{\|f_\theta(x_u)\|_2},\qquad
b_i=\frac{g_\phi(x_i)}{\|g_\phi(x_i)\|_2},\qquad s(u,i)=a_u^\top b_i.
$$

物品向量可以预先算好，在线只算一次用户向量，再从索引取邻居。如果先把用户与每个物品拼接，再运行大 MLP，用户和物品的交互会更丰富，但通常不能把整次评分分解成“一次用户编码＋可索引的物品向量”。

内积与余弦不是随意互换：未归一化时，内积还受向量长度影响。训练、离线建库和查询必须采用一致的度量、归一化及维度。[双塔模型与训练课件](https://github.com/wangshusen/RecommenderSystem/blob/main/Slides/02_Retrieval_06.pdf)

## 6. 三种训练目标，分别要求什么

Pointwise 把用户—物品配对当作二分类，用标签 $y$ 和 logit $z$ 做二元交叉熵。采样后的类别比例与线上比例不同，输出 sigmoid 未必就是线上真实点击概率；如果目标主要是召回排序，需要避免把它误称为天然校准的 CTR。

Pairwise 要求正例高于负例。课程给出的 triplet hinge loss 为

$$
\ell=\max(0,m+s(u,i^-)-s(u,i^+)).
$$

$s^+=0.7$、$s^-=0.5$、$m=0.3$ 时损失为 0.1；正例提高到 0.9 后为 0。Margin 是希望正负分数拉开的距离，不是正例概率阈值。

Listwise 在一个正例与多个负例中选择正例。令 $z_j=s(u,i_j)/\tau$，一种 sampled softmax 目标是

$$
\ell=-\log\frac{e^{z_+}}{e^{z_+}+\sum_{j\in N}e^{z_j}}.
$$

logits 为 $(2,1,0)$，正例在第一位，则概率约为 0.6652，损失约为 0.4076。这和 CLIP 的对比分类机制相通，但这里的用户行为正例与多正确兴趣更复杂。以上温度是一般化写法；课程算例也可不显式使用温度。

## 7. 负样本改变模型学到的任务

| 候选来源 | 带来的训练信息 | 需要检查 |
|---|---|---|
| 全库随机物品 | 易负例，教模型区分大范围兴趣 | 未观察不等于确定不喜欢；采样热门度 |
| Batch 内其他用户的正例 | 编码可复用，负例多 | 热门物品高频、重复 ID、多正例冲突 |
| 召回后被排序淘汰的物品 | 较难的相近候选 | 旧排序器偏差、假负例比例 |
| 曝光但未点击 | 排序中的选择性反馈 | 实际可见性、位置、停留、点击定义 |

课程特别反对把“曝光未点击”直接作为召回负例：召回要宽泛地寻找可能感兴趣的内容；已经进入曝光列表的物品可能只是这次没有被选中。这是该课程的建模建议，不能推导成所有召回训练永远不得使用曝光日志。更一般的原则是：**明确负标签对应的事件，再处理曝光选择偏差和假负例。** [正负样本课件](https://github.com/wangshusen/RecommenderSystem/blob/main/Slides/02_Retrieval_07.pdf)

### Batch 内负例与采样纠偏

$B$ 对用户—物品正例构成 $B\times B$ 分数矩阵，对角线正例，非对角线有 $B(B-1)$ 个候选负配对。来自点击日志的物品按其点击频率出现，热门物品也更常成为负例。课程采用训练 logit 修正

$$
z'_j=z_j-\log q(i_j),
$$

其中 $q$ 是物品在当前采样方案中的概率。更精确的 sampled-softmax 实现可能使用期望采样次数，且正例、重复样本、采样过程都需要一致处理。**修正的是进入 softmax 的 logit**；若 $z=s/\tau$，就修正 $s/\tau-\log q$，不能把减项的位置随意移到温度之前。

两个物品原 logit 都为 1，$q_A=0.1$、$q_B=0.01$，修正后为 3.3026、5.6052。稀有物品在 sampled 分类中获得更大的补偿。这是对采样分布的校正，不是建议在线对所有物品额外加一个稀有度奖励。[课程引用的采样纠偏论文](https://research.google/pubs/sampling-bias-corrected-neural-modeling-for-large-corpus-item-recommendations/)

## 8. 自监督为什么能帮助长尾物品

行为损失按点击日志学习，头部物品监督多，长尾监督少。课程增加物品侧自监督：从物品库取样，为同一物品生成两个特征视图，通过共享物品塔编码，让同物品视图靠近，不同物品视图区分开。

$$
\mathcal L=\mathcal L_{u-i}+\lambda\mathcal L_{i-i}.
$$

视图可采用特征遮挡、多值字段 dropout、互补字段分组或联合遮挡相关字段。联合遮挡的原因是：只藏类目，却保留几乎能完全推回类目的关键词，模型可能轻易走捷径。但遮挡若毁掉了决定兴趣的关键信息，也会制造错误不变性。

验证时固定行为样本、总更新预算和采样方案，比较去掉自监督、随机遮挡、相关字段遮挡；按曝光量分桶看长尾 Recall@K，并看头部是否回退。只报整体提升无法证明是长尾表征改善。[自监督课件](https://github.com/wangshusen/RecommenderSystem/blob/main/Slides/02_Retrieval_09.pdf)

## 9. 模型、物品向量和索引必须成套更新

在线部署至少要关联四个版本：特征 schema、用户塔、物品塔、物品向量索引。旧索引里的向量不一定与新用户塔处在同一个空间；模型文件成功加载并不等于检索服务可用。

课程提出全量训练与增量更新的分工：全量训练更新整体网络和物品向量，部分在线学习更新用户 ID embedding，以捕捉近期兴趣并避免频繁重建全库。这是一个部署方案；若你的增量训练同时改动物品塔或共享空间，就必须重新考虑向量兼容性，不能照搬“只发用户参数”。[线上服务课件](https://github.com/wangshusen/RecommenderSystem/blob/main/Slides/02_Retrieval_08.pdf)

先在一个固定小库中比较精确点积 Top-K 与 ANN Top-K，测索引召回率；再看标签上的用户兴趣召回率。新物品入库延迟、删除后能否搜到、空查询与索引不可用时的退路，也属于召回系统的质量。

## 10. Deep Retrieval：把物品编码成路径

Deep Retrieval 用有限深度、宽度的路径表示物品，维护物品→多条路径与路径→物品列表。模型根据用户与此前节点预测下一个节点，路径概率分解为

$$
P(a,b,c\mid u)=P(a\mid u)P(b\mid a,u)P(c\mid a,b,u).
$$

在线用 beam search 找较优路径，再从路径倒排取物品。第一步概率最高的前缀未必通向总概率最高的完整路径；beam size=1 是贪心，而不是保证精确最优。增加 beam 能缓解早期剪枝，但也增加计算。物品路径本身需要学习和维护，因此它改变的不只是模型，也是可检索结构。[Deep Retrieval 课件](https://github.com/wangshusen/RecommenderSystem/blob/main/Slides/02_Retrieval_10.pdf)

## 11. 通道组合、缓存与曝光过滤

关注作者、有交互作者、相似作者、同城、新内容、热门内容都可以成为召回通道。精排高分但尚未曝光的候选可以放入用户缓存，后续复用，但需退场规则：已曝光移除、时间过期、反复无效召回后移除、容量淘汰。

增加通道不应只报“它自己的命中率”。如果它完全重复其他通道，合并候选不变，却多花了算力。应看独有候选、独有点击覆盖、替换了哪些旧候选，以及同总候选预算下的最终结果。[其他召回通道课件](https://github.com/wangshusen/RecommenderSystem/blob/main/Slides/02_Retrieval_11.pdf)

曝光过滤可用精确集合，也可用 Bloom filter 节约空间。标准插入型 Bloom filter 对已插入元素没有假阴性，却可能把没看过的内容误判为已看过，降低候选覆盖。这个保证依赖实现与保留周期；重置后遗忘历史不是数学保证失效，而是系统主动丢掉旧记录。误报率近似为

$$
p_{\rm fp}\approx(1-e^{-kn/m})^k,
$$

$m$ 是 bit 数、$n$ 是插入数、$k$ 是 hash 数。应按可接受误报率和记录寿命设计，并测误过滤的损失。[曝光过滤课件](https://github.com/wangshusen/RecommenderSystem/blob/main/Slides/02_Retrieval_12.pdf)

## 闭卷验收

手算一个 ItemCF 候选的多条路径贡献；画出双塔哪段离线、哪段在线；解释 $q(i)$ 为什么出现在训练 logit 里；再分别说出“ANN 丢邻居”和“模型不懂兴趣”的诊断方法。下一篇：[排序、特征交叉与行为序列](/notes/wangshusen-recommender-ranking/)。

**来源与许可。** 参考 Cheer-ego 的 [上篇](https://blog.csdn.net/qq_43629945/article/details/134109883)，以王树森原始召回课件校对，并重写解释、补充算例与边界。本页改编文字与自绘图采用 [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)。完整来源、检索快照与改写说明见 [系列入口](/notes/wangshusen-recommender-guide/)。
