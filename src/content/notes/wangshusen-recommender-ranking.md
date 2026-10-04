---
title: 推荐系统排序：概率口径、特征交叉与行为序列
date: '2026-10-04'
tags: [推荐系统, 王树森, 排序, MMoE, DIN, 特征交叉]
summary: 从一次曝光的价值推到多目标融分，讲清三塔、FM、DCN V2、LHUC、FiBiNET、DIN 与 SIM 的分工。
draft: false
---

召回已经找到几千条可能相关的内容，排序要决定谁值得进入接下来几十个位置。模型输出不是一个含义模糊的“推荐分”，而是对点击、交互、观看等事件的预估。**先把事件口径写清，再讨论网络结构，最后才谈融合和调参。**

前置阅读：[召回与双塔](/notes/wangshusen-recommender-retrieval/)；完整来源见 [系列入口](/notes/wangshusen-recommender-guide/)。本页数值均为教学构造，网络结构收益需要在具体数据上验证。

## 1. 共享底座和多任务头

用户、物品、统计和场景特征拼接成 $x$，shared bottom 产生 $h=F(x)$，多个任务头分别输出

$$
p_t=\sigma(w_t^\top h+b_t),\qquad
\mathcal L=\sum_t\lambda_t\mathcal L_t.
$$

共享表示可以让任务利用共同信号，但任务数据量、噪声和梯度方向可能不同。点击多、点赞少时，点击损失可能主导训练；某个特征促进点击却不促进满意度时，也可能出现负迁移。权重 $\lambda_t$ 是训练目标的取舍，不等于线上融分的业务权重。

如果点赞目标定义为点击后的条件事件，训练可以只在有效点击样本上计算对应损失；若定义为曝光后的点赞事件，则使用曝光样本及相应标签。是否使用 mask、是否建联合概率都属于任务定义，不能因为“都是 sigmoid 头”就不加区分。[多目标排序课件](https://github.com/wangshusen/RecommenderSystem/blob/main/Slides/03_Rank_01.pdf)

## 2. MMoE：每个任务自己选择怎样共享

设 $E$ 个专家输出 $e_1(x),\ldots,e_E(x)$，任务 $t$ 的门输出 softmax 权重 $g_t(x)$，其表示为

$$
h_t=\sum_{j=1}^{E}g_{t,j}(x)e_j(x),\qquad
p_t=\operatorname{Tower}_t(h_t).
$$

专家是共享的，门与输出塔是任务特定的。同一组专家向量 $(2,0)$、$(0,2)$，点击门权重 $(0.75,0.25)$ 得到 $(1.5,0.5)$，点赞门权重 $(0.25,0.75)$ 得到 $(0.5,1.5)$。这比所有任务拿同一个底座向量更灵活。

门输出若长期几乎 one-hot，部分专家可能很少被使用，梯度也少，课程称之为极化。应看每个任务的专家占用、门分布熵、专家梯度和分桶表现。课程介绍对门输出做 dropout 来缓解；实现还要统一训练与推理的缩放规则。极化不必然使模型等同于 shared bottom：多个任务可能分别使用不同专家，真正需要判断的是共享是否有效、专家是否浪费。

MMoE 不是“用了就会涨”的模块。比较时应尽量控制总参数、特征、训练预算与任务损失，确认增益来自共享方式而不是单纯多了容量。[MMoE 课件](https://github.com/wangshusen/RecommenderSystem/blob/main/Slides/03_Rank_02.pdf)

## 3. 融分要和概率定义对齐

假设点开后才可能点赞、收藏。$p_c$ 是曝光后点击概率，$p_l,p_f$ 是点击后的点赞、收藏概率。每次曝光的期望加权价值可以写成

$$
S=p_c(1+w_lp_l+w_fp_f).
$$

其中 1 表示一次点击的单位价值。A 的 $(p_c,p_l,p_f)=(0.30,0.04,0.02)$，B 为 $(0.18,0.10,0.08)$，$w_l=3,w_f=5$。A 得 $0.30(1+0.12+0.10)=0.366$，B 得 $0.18(1+0.30+0.40)=0.306$。B 的点赞和收藏概率较高，但在这组价值权重下仍由 A 排前。改权重可能改顺序，解释时必须说清优化了哪种价值。

课程还介绍直接加权和、乘法融合、排名变换等方式。它们可以作为实用排序规则，但未必都是同一个概率期望。点击后点赞率直接与 CTR 相加，是混合不同条件口径的启发式；不是数学错误，却需要说清业务含义与实验依据。

一个头的 AUC 好，不保证其概率与另一个头处在可比较尺度。融合前要看 logloss、可靠性曲线、分桶校准和取值分布；否则权重可能主要在补偿预测尺度，而不是表达业务偏好。[预估分数融合课件](https://github.com/wangshusen/RecommenderSystem/blob/main/Slides/03_Rank_03.pdf)

## 4. 时长建模：代数等价不等于统计等价

课程把实际时长 $t\ge0$ 变换为 $y=t/(1+t)$，令 $p=\sigma(z)$，使用 soft-label BCE：

$$
\ell=-\frac{t}{1+t}\log p-\frac{1}{1+t}\log(1-p).
$$

对于单个固定目标，若 $p=y$，则 $p/(1-p)=t$，而 sigmoid 的 odds 是 $e^z$，所以推理输出可取 $e^z$。例如 $t=3$，$y=0.75$，$z=\log3$。

但给定同样特征，真实时长仍有随机性。这个**未加权**损失的总体最优值是 $p^*(x)=E[t/(1+t)\mid x]$，由它得到的 odds 通常不等于 $E[t\mid x]$。若同条件下 $t$ 等概率为 0 和 3，$p^*=0.375$，odds 为 0.6，而平均时长为 1.5。这是理解该变换必须保留的边界。

若再给每条样本乘权重 $(1+t)$，损失变为

$$
\ell_w=-t\log p-\log(1-p).
$$

在该理想化目标下，条件期望损失的最优 odds 等于 $E[t\mid x]$。因此去掉原式分母实际改变了样本权重与拟合目标，不能说只是删掉一个无关常数。还有零时长定义、截断、重复播放与视频长度效应需要明确。[视频播放建模课件](https://github.com/wangshusen/RecommenderSystem/blob/main/Slides/03_Rank_04.pdf)

完播也有两种目标：预测实际观看比例的期望，或预测“观看超过 80%”的概率。模型输出 0.7 在前者表示预计看七成，在后者表示七成概率达到阈值。短视频通常更容易高完播，按视频长度做基线修正时，还要检查短桶样本量与小分母稳定性。

## 5. 特征为什么比换网络更容易改变结果

| 类别 | 示例 | 时间与使用边界 |
|---|---|---|
| 用户特征 | ID、画像、长期偏好 | 更新周期与 OOV 处理 |
| 物品特征 | 类目、作者、标题图像、长度 | 上线时确实可用；新内容也有 |
| 统计特征 | 近 1 小时、1 天、7 天曝光与交互 | 只能使用请求前的计数 |
| 交叉特征 | 用户在某类目的行为、用户与作者关系 | 通常随候选变化 |
| 场景特征 | 时间、页面、设备、地理上下文 | 请求级信息与缺失值 |

长尾计数常用 $\log(1+x)$ 压缩；CTR 可采用平滑估计 $(c+\alpha p_0)/(n+\alpha)$，其中 $p_0$ 是先验概率、$\alpha$ 是等效样本量。1 次曝光 1 次点击不应直接被解释为稳定的 100% CTR。

最危险的错误是用未来信息训练：例如给上午的曝光样本拼接全天点击数，其中包含了这次曝光之后的行为。离线分数会很好，但上线时这些特征根本不存在。需要 point-in-time join，固定事件时间、统计窗口和延迟规则。[特征课件](https://github.com/wangshusen/RecommenderSystem/blob/main/Slides/03_Rank_05.pdf)

## 6. 三塔粗排：把可复用和不可复用计算拆开

用户塔在一次请求中算一次；物品塔处理相对静态的物品特征，其向量可缓存；交叉塔处理动态统计与用户—物品交叉，随候选逐条算。三个塔的输出再拼接、交叉，经过较小上层网络输出多任务分数。

有 $M$ 条候选时，粗略成本是

$$
C\approx C_u+M(1-h)C_i+M(C_{\rm cross}+C_{\rm head}),
$$

$h$ 是物品缓存命中率。$M=3,000$、$h=0.99$ 时，物品塔期望仅运行 30 次，交叉塔和上层仍需处理 3,000 个候选。放进物品塔的特征越动态，缓存越容易过期；缓存键还要包含模型版本。网络小不一定服务快，特征远程读取、批处理与缓存命中也会决定时延。[粗排三塔课件](https://github.com/wangshusen/RecommenderSystem/blob/main/Slides/03_Rank_06.pdf)

## 7. FM：怎样少参数地表示二阶交叉

普通线性模型只计算 $\sum_iw_ix_i$，无法直接表达“某年龄用户对某类目更感兴趣”的组合。FM 加入二阶项：

$$
\hat y=w_0+\sum_iw_ix_i+\sum_{i<j}\langle v_i,v_j\rangle x_ix_j.
$$

每个特征有 $k$ 维向量，用内积表示交叉系数，相比每对特征独立存权重，参数从 $O(d^2)$ 变为 $O(dk)$。交叉项还能改写为

$$
\frac12\sum_{f=1}^k\left[\left(\sum_i v_{if}x_i\right)^2-\sum_i v_{if}^2x_i^2\right],
$$

避免显式遍历全部特征对。对于稀疏输入，主要成本随非零特征数乘 $k$ 增长。

例如两个特征 $x_1=1,x_2=2$，$v_1=(1,0),v_2=(0.5,1)$，交叉贡献为 $0.5\times1\times2=1$。这个低秩约束意味着各交叉系数通过向量共享参数，不可能任意独立设置，节省参数也同时引入了结构假设。[FM 课件](https://github.com/wangshusen/RecommenderSystem/blob/main/Slides/04_Cross_01.pdf)

## 8. DCN V2：用原始输入显式参与每层交叉

课程使用矩阵形式的 DCN V2 cross layer：

$$
x_{l+1}=x_l+x_0\odot(W_lx_l+b_l).
$$

第一层把 $x_0$ 与其线性变换逐元素相乘，产生显式二阶关系；后续层继续与原始 $x_0$ 相乘，增加交叉阶数，同时残差保留已有信号。令 $x_0=(1,2)$、$W=I$、$b=0$，第一层为 $(1,2)+(1,4)=(2,6)$。把“逐元素乘”误写成内积会把向量变标量，shape 已经不对。

早期 DCN 的向量参数形式和 V2 的矩阵参数形式不同。解释参数量与表达能力时应标版本；全矩阵层有 $d^2$ 参数，低秩版本或专家结构会再改变计算。DCN 可在某个双塔的单塔内部使用；但若让用户与候选特征先交叉，就会改变可分离召回条件。[DCN V2 课件](https://github.com/wangshusen/RecommenderSystem/blob/main/Slides/04_Cross_02.pdf)

## 9. LHUC、SENet、FiBiNET：先确定权重作用在哪

LHUC 用条件信息生成隐藏维度的缩放，例如 $a(c)=2\sigma(G(c))$，然后 $h'=h\odot a(c)$。一个用户的偏好可以改变隐藏单元贡献，缩放范围在 $(0,2)$。它控制隐藏层，不能简单等同于对输入字段做一个重要性排序。[LHUC 课件](https://github.com/wangshusen/RecommenderSystem/blob/main/Slides/04_Cross_03.pdf)

推荐中的 SENet 对每个字段 embedding 做池化，生成 field-wise 权重。用户 ID embedding 的 64 个维度作为一个 field，共享一个缩放权重；有 $m$ 个字段就有 $m$ 个权重。得到的是随当前样本变化的字段重标定，不是每个维度独立门控。

FiBiNET 再组合字段重标定与 bilinear interaction，例如向量交叉 $e_i\odot(W_{ij}e_j)$。内积给一个标量，Hadamard 乘给向量，双线性变换引入可学习混合。字段多时交叉数量和参数共享方式很重要；不是在任何系统里把所有交叉都堆上去就划算。[SENet 与 FiBiNET 课件](https://github.com/wangshusen/RecommenderSystem/blob/main/Slides/04_Cross_04.pdf)

## 10. Last-N 与 DIN：候选改变你读取哪段历史

把最近 $N$ 个交互 embedding 简单平均，得到一个与当前候选无关的用户向量，便宜且可复用。缺点是美食、汽车、摄影兴趣混在一起，同一平均向量无法针对本次候选选择相关行为。

DIN 让候选 $q$ 参与历史权重计算：$a_j=A(q,e_j)$，兴趣表示为

$$
h_u(q)=\sum_j a_j e_j.
$$

原始 DIN 的局部激活权重不要求是标准 softmax；教学演示也可以用归一化权重，但要标明实现。若美食与汽车行为向量分别为 $(1,0)$、$(0,1)$，美食候选下权重 $(0.9,0.1)$ 得到 $(0.9,0.1)$，汽车候选则可能相反。

因为 $h_u(q)$ 随候选变化，它不能直接作为经典双塔里“一个请求只算一次”的用户塔输出。**候选相关 attention 不可复用，不代表所有 attention 都不适合召回。** 用户历史自身的 self-attention 若不依赖候选，仍能产生一次性用户向量。[Last-N 课件](https://github.com/wangshusen/RecommenderSystem/blob/main/Slides/05_LastN_01.pdf)、[DIN 课件](https://github.com/wangshusen/RecommenderSystem/blob/main/Slides/05_LastN_02.pdf)

DIN 的加权池化自身不必表达历史顺序。若行为时间和先后顺序是重要信号，应明确加入时间、位置或序列编码，不能看到 attention 就自动称为 Transformer 时序建模。

## 11. SIM：先从长历史里查，再精细读取

若用户有数千次历史，而每个候选都对完整历史计算 DIN，成本会扩大。SIM 先检索相关历史 Top-K，再对这个小集合做候选相关注意力。Hard search 可按类目查；soft search 可用向量邻居查。再加入距今时间的分桶 embedding，让模型区分昨天和半年前的行为。

有 500 个候选、5,000 条历史，朴素逐候选匹配有 250 万个配对；若历史检索每次保留 50 条，精细 attention 只剩 25,000 个配对。但这还没算历史检索的成本，不能据此声称服务必然快 100 倍。检索若漏掉相关历史，后面的注意力也救不回来。[SIM 课件](https://github.com/wangshusen/RecommenderSystem/blob/main/Slides/05_LastN_03.pdf)

## 闭卷验收

写出一个任务从标签、样本 mask 到预测概率再到融分的完整路径；说明 MMoE 门与 SENet 门各控制什么；手算一层 FM/DCN；最后画出 DIN 为什么随候选变化。下一篇：[MMR 与 DPP 重排](/notes/wangshusen-recommender-reranking/)。

**来源与许可。** 参考 Cheer-ego 的 [上篇](https://blog.csdn.net/qq_43629945/article/details/134109883) 与 [下篇](https://blog.csdn.net/qq_43629945/article/details/138551391)，并以王树森排序、特征交叉、Last-N 原始课件校对。补充了条件概率与时长目标的推导、版本边界和独立算例。本页改编文字采用 [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)；详情见 [系列入口](/notes/wangshusen-recommender-guide/)。
