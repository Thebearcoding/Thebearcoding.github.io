---
title: 推荐与 LLM4Rec 面试题库：九十问
date: '2026-10-05'
tags: [推荐系统, LLM4Rec, 复习问答, 面试, 手撕代码]
summary: 九十道题按系统评估、召回、排序、长序列、scaling、语义 ID、LLM4Rec、多模态、广告、手撕代码和项目追问分组，每题附参考解答和对应章节；末尾另有一节取自公开面经的补充问题清单。
draft: false
---

每题先用两分钟口述，再展开参考解答。答得出“是什么”只算一半，另一半是“为什么这样做、什么时候会失败、怎样验证”。这些题是本站按教程内容设计的复习题，不代表某家公司的题库或出现频率。

阅读入口：[全覆盖教程](/notes/recsys-llm4rec-guide/)。手算类题目另见 [王树森复习题](/notes/wangshusen-recommender-workbook/)，多模态大模型题目另见 [多模态复习问答](/notes/multimodal-interview-questions/)。

## A. 系统与评估

### A1. 推荐链路为什么分召回、粗排、精排、重排？

<details><summary>参考解答</summary>

每一段的单条打分成本递增、候选数递减。全库千万级只能用可分离的向量检索或倒排；几千个候选可以用三塔或小模型粗排；几百个用重特征、多目标精排；最后几十个按整个列表的多样性与规则重排。把精排直接用于全库，成本乘以几万倍。分段的代价是各段目标不一致，上一段漏掉的物品下一段救不回来。[入口](/notes/wangshusen-recommender-guide/)

</details>

### A2. AUC 和 GAUC 有什么区别，什么时候更信 GAUC？

<details><summary>参考解答</summary>

AUC 是任取一对正负样本、正样本得分更高的概率；GAUC 先算每个用户内的 AUC 再按曝光或点击加权。全局 AUC 会奖励“知道谁是活跃用户”这种跨用户信息，而精排只需要在同一用户的候选间排序，所以精排更信 GAUC。要报告剔除的单类用户比例。[评估](/notes/recsys-eval-bias/)

</details>

### A3. 离线 AUC 涨了，线上 CTR 没涨，可能是什么原因？

<details><summary>参考解答</summary>

一是样本分布不同：离线评估的曝光样本来自旧模型，新模型改变了展示的东西。二是特征不一致：时间穿越、线上缺失值处理不同。三是校准与融分：AUC 不看绝对值，但融分和出价看，概率尺度变了会打乱多目标排序。四是瓶颈不在这一层：召回没有好候选。五是实验问题：流量小、有干扰、新奇效应。排查时先对齐特征快照，再看分桶校准，最后看分层漏斗。

</details>

### A4. 写出 NDCG@K，用户有多个相关物品时 IDCG 怎么算？

<details><summary>参考解答</summary>

$\mathrm{DCG@}K=\sum_{i=1}^{K}\mathrm{rel}_i/\log_2(i+1)$，NDCG 除以理想排序的 DCG。多个相关物品时，IDCG 是把它们全部排在最前面得到的 DCG，最多算 $\min(|R|,K)$ 个位置。例：两个相关物品排第 1、4 位，NDCG@5 约 0.877。[评估](/notes/recsys-eval-bias/)

</details>

### A5. 为什么不建议只用“1 个正样本 + 100 个采样负样本”的 HR@10？

<details><summary>参考解答</summary>

采样排名是全量排名的粗糙压缩：全库排第 500 的目标，在 101 个候选里期望约排第 6，很可能算命中。指标对顶部的区分度被抹平，两个模型的优劣甚至可能反转（Krichene 与 Rendle, KDD 2020）。能全量就全量；只能采样时固定种子，并用全量结果核对。

</details>

### A6. 留一法和全局时间切分各有什么问题？

<details><summary>参考解答</summary>

留一法实现简单，但不同用户的“最后一次”时间不同，训练集可能包含比某些测试交互更晚的数据，泄漏未来的流行趋势。全局时间切分更接近线上，但新物品多，指标偏低。两种结果不能混比，论文和项目里都要写明。

</details>

### A7. 负样本只保留了 10%，模型输出 0.5，真实点击率约是多少？

<details><summary>参考解答</summary>

负采样率 $w$ 下，模型赔率是真实赔率的 $1/w$ 倍，还原 $p=wp'/(1-p'+wp')$。$w=0.1$、$p'=0.5$ 时 $p\approx0.091$。融分或出价前不还原，这个目标会被放大约 10 倍。

</details>

### A8. 位置偏差怎么处理？

<details><summary>参考解答</summary>

点击可近似为“被看到 × 看到后点击”。训练时把位置作为特征或单独的浅层塔建模“被看到”，推理时位置置为统一默认值，只用相关性部分打分（PAL 一类）。也可以用随机打散的小流量估计位置效应。验证：在随机流量上比较纠偏前后的排序质量。

</details>

### A9. ESMM 解决什么问题，怎么做？

<details><summary>参考解答</summary>

CVR 只在点击样本上有标签，训练空间（点击）和推理空间（曝光）不同，而且样本稀疏。ESMM 在全部曝光上监督 pCTR 和 pCTCVR $=$ pCTR $\times$ pCVR，pCVR 作为中间量学习，两塔共享 embedding。IPW 是另一种思路：点击样本按 $1/\mathrm{pCTR}$ 加权，要截断以控制方差。

</details>

### A10. A/B 实验有哪些常见坑？

<details><summary>参考解答</summary>

分桶不独立（多层实验互相干扰）；样本单位与指标单位不一致（按用户分流，按曝光算方差）；同时看很多指标导致假阳性；新奇效应让前几天偏高；作者侧、广告主侧实验存在流量竞争，组间不独立。要有预先确定的主指标、保护指标和足够时长。[冷启动与实验](/notes/wangshusen-recommender-coldstart/)

</details>

## B. 召回

### B1. ItemCF 的相似度怎么算，为什么要惩罚热门？

<details><summary>参考解答</summary>

二值情形 $s(i,j)=|U_i\cap U_j|/\sqrt{|U_i||U_j|}$。热门物品和什么都共现，不惩罚就会到处被召回，失去个性化；对活跃用户也可降权，因为他们的共现信息量低。[召回](/notes/wangshusen-recommender-retrieval/)

</details>

### B2. Swing 比 ItemCF 好在哪里？

<details><summary>参考解答</summary>

ItemCF 把每个共同用户当独立证据。若一群用户本身高度重合（同一个群聊），他们的共现不代表物品相似。Swing 对共同用户两两计算 $1/(\alpha+|I_a\cap I_b|)$，重合越多贡献越小，压低“小圈子”的影响。

</details>

### B3. 双塔为什么做不了用户与物品的特征交叉？

<details><summary>参考解答</summary>

双塔的价值在可分离：物品向量离线算好建索引，用户向量每个请求算一次，打分只是内积，才能在全库上做近似检索。一旦用户和物品特征在塔内交叉，每个物品都要和当前用户一起过网络，就退化成精排。交叉放到粗排、精排去做。

</details>

### B4. batch 内负样本有什么偏差，logQ 怎么纠正？

<details><summary>参考解答</summary>

batch 内负样本按物品出现频率被抽中，热门物品作为负样本出现过多，被过度打压。训练时把 logit 改为 $s(u,i)-\log q_i$，$q_i$ 用流式频率估计；线上检索仍用原始分数。[评估](/notes/recsys-eval-bias/)

</details>

### B5. 召回的负样本为什么通常不用“曝光未点击”？

<details><summary>参考解答</summary>

曝光过的物品已经通过了召回和排序，是“用户可能感兴趣”的物品；召回的任务是从全库里区分相关与不相关，大部分全库物品远比曝光未点击的更不相关。用曝光未点击做召回负样本，会让模型过度关注细微偏好而丢掉大方向。召回常用全库随机负样本、batch 内负样本，再混入少量难负例；曝光未点击更适合排序。

</details>

### B6. 对比损失里的温度起什么作用？

<details><summary>参考解答</summary>

$\mathrm{softmax}(s/\tau)$ 中，$\tau$ 小时分布更尖，梯度集中在最难的负样本上，表征更有区分度，但对假负例更敏感、训练更不稳；$\tau$ 大时各负样本权重接近。CLIP 把 $\tau$ 设为可学习参数。[CLIP](/notes/vision-video-algorithms/)

</details>

### B7. HNSW 和 IVF-PQ 怎么选？

<details><summary>参考解答</summary>

HNSW 是多层近邻图，查询从顶层贪心走到底层，召回率高、延迟低，但内存大（存原向量和图），增删需要维护。IVF-PQ 先用粗聚类分桶，只搜最近的若干桶，桶内用乘积量化压缩向量并查表算距离，内存小、适合十亿级，但量化有精度损失。内存够、规模中等选 HNSW；规模极大选 IVF-PQ，必要时用原向量重排。

</details>

### B8. 多路召回怎么融合？

<details><summary>参考解答</summary>

常见做法是每路设配额后合并去重，再交给粗排统一打分；配额按各路历史贡献（最终曝光和互动的来源占比）和业务目标（冷启动、多样性）调整。要监控每一路的覆盖率和被排序选中的比例，某路长期贡献低就该优化或下线。

</details>

### B9. 树索引、Deep Retrieval 和向量检索有什么不同？

<details><summary>参考解答</summary>

向量检索要求打分可分离成内积。树索引（TDM）和 Deep Retrieval 把物品编码到树或路径上，检索时逐层 beam search，每层可以用任意复杂的模型打分，突破内积限制；代价是索引结构要和模型一起学，更新更难。语义 ID 的生成式召回可以看作同一思路的延续。

</details>

## C. 排序与特征交叉

### C1. MMoE 解决什么问题？

<details><summary>参考解答</summary>

多任务共享底座时，相关性弱的任务互相拖累（跷跷板）。MMoE 用多个专家网络，每个任务一个门控对专家加权，任务可以选择不同的共享方式。常见问题是门控极化：某个门几乎只选一个专家。可以对门加 dropout 或熵正则。[排序](/notes/wangshusen-recommender-ranking/)

</details>

### C2. 多目标怎么融合成一个排序分？

<details><summary>参考解答</summary>

常见形式是加权乘积 $\prod_k p_k^{w_k}$ 或加权和；权重先用业务目标与线上实验确定。要求各个预估值口径清楚（曝光后还是点击后）、经过校准。权重调整本身就是一种实验，要看主指标与保护指标。

</details>

### C3. FM 的二阶交叉怎么从 $O(kn^2)$ 降到 $O(kn)$？

<details><summary>参考解答</summary>

$\sum_{i<j}\langle v_i,v_j\rangle x_ix_j=\frac12\sum_{f=1}^{k}\big[(\sum_i v_{i,f}x_i)^2-\sum_i v_{i,f}^2x_i^2\big]$。每一维只需求和与平方和。

</details>

### C4. 写出 DCN V2 的一层交叉。

<details><summary>参考解答</summary>

$x_{l+1}=x_0\odot(W_lx_l+b_l)+x_l$。$x_0$ 是原始输入，每层都显式地和原始输入做逐元素乘，堆 $L$ 层得到最高 $L+1$ 阶交叉；残差保留低阶信息。$W_l$ 可以用低秩分解降参数。

</details>

### C5. SENet 在推荐里做什么？

<details><summary>参考解答</summary>

把每个特征字段的 embedding 压缩成一个标量，经两层 MLP 得到每个字段的权重，再乘回原 embedding，相当于按样本动态调整字段重要性。和 MMoE 的门不同：SENet 的门作用在字段上，MMoE 的门作用在专家上。

</details>

### C6. 粗排为什么用三塔？

<details><summary>参考解答</summary>

用户塔每个请求算一次，物品塔可以缓存，交叉塔只放少量必须实时计算的交叉特征，几千个候选都跑一个轻量交叉塔仍然可控。目标是在双塔与精排之间找到成本和效果的平衡，常配合精排蒸馏。

</details>

### C7. 播放时长怎么建模？

<details><summary>参考解答</summary>

一种做法是把时长当权重的加权逻辑回归：正样本权重为时长 $t$，负样本权重为 1，训练后 $e^{z}$ 近似期望时长。也可以预测完播率或分桶后分类。要注意长视频天然时长长，常需要按视频长度归一化。

</details>

### C8. 特征穿越是什么，怎么避免？

<details><summary>参考解答</summary>

训练样本使用了事件发生时还拿不到的信息，例如用当天全天的点击数做当天上午样本的特征。离线效果虚高，上线就掉。做法是所有特征按样本时间点做快照（point-in-time join），线上线下用同一份特征生成逻辑。

</details>

### C9. 精排怎样帮助粗排？

<details><summary>参考解答</summary>

把精排对候选的打分或排序当作粗排的软标签（蒸馏），让粗排学会精排的偏好；常用 listwise 或 pairwise 损失。注意精排只见过粗排筛过的候选，蒸馏样本的分布要包含粗排阶段的候选。

</details>

## D. 序列与长序列

### D1. SASRec 训练时一个长度为 T 的序列产生几个预测？mask 是什么样？

<details><summary>参考解答</summary>

产生 $T-1$ 个（每个位置预测下一个）。因果 mask 是上三角为 $-\infty$ 的矩阵，位置 $t$ 只能看见 $1..t$。一次前向就得到所有位置的预测，比 RNN 高效。[序列](/notes/recsys-sequence-long/)

</details>

### D2. BERT4Rec 一定比 SASRec 强吗？

<details><summary>参考解答</summary>

不一定。早期比较中 SASRec 用 BCE 加一个负样本；换成全库 softmax 交叉熵后，SASRec 可以追平甚至超过 BERT4Rec（Klenitskiy 与 Vasilev, RecSys 2023）。比较模型前要先统一损失、负样本数和训练轮数。

</details>

### D3. DIN 的注意力和 Transformer 的注意力有何不同？

<details><summary>参考解答</summary>

DIN 的 Query 是候选物品，Key/Value 是历史行为，权重由一个小 MLP 根据 $[e_u,e_v,e_u-e_v,e_u\odot e_v]$ 计算，原文不要求 softmax 归一化，以保留兴趣强度。它是目标注意力，不建模历史内部的顺序关系。

</details>

### D4. SIM 的两阶段分别做什么，有什么问题？

<details><summary>参考解答</summary>

GSU 从上万条历史里按类目（hard）或 embedding 内积（soft）挑出 Top-K；ESU 对 Top-K 做目标注意力，并加入时间间隔 embedding。问题是 GSU 与 ESU 的相关性不一致，GSU 漏掉的 ESU 无法补救。

</details>

### D5. ETA 为什么用 SimHash？

<details><summary>参考解答</summary>

SimHash 把向量编成 $k$ 位二值码，两向量夹角为 $\theta$ 时单位碰撞概率为 $1-\theta/\pi$，Hamming 距离近似反映相似度。比较只需 XOR 和 popcount，比高维内积便宜得多；签名来自同一套 embedding，可以端到端训练。

</details>

### D6. SDIM 和 ETA 有什么不同？

<details><summary>参考解答</summary>

ETA 用哈希检索 Top-K 再做注意力；SDIM 不检索，直接把与候选哈希签名相同的历史聚合起来。期望上每条历史被聚合的概率随相似度单调变化，聚合结果近似按相似度加权的注意力，省掉 Top-K 和 softmax。

</details>

### D7. TWIN 怎样让 GSU 也算得起目标注意力？

<details><summary>参考解答</summary>

把行为特征拆成固有特征（只与物品有关，Key 投影可按物品离线缓存）和交叉特征（与用户-物品对有关，压缩成一个标量偏置加到分数上）。GSU 对上万条行为打分时主要是查表加内积，于是能和 ESU 用同一种注意力，保持一致。

</details>

### D8. 用 Transformer 处理上万条行为，有哪些降本手段？

<details><summary>参考解答</summary>

全局 token（候选与用户画像，兼作注意力锚点）；token 合并（相邻行为压成一个 token）；只让少数位置当 Query，或只让候选当 Query 做线性复杂度的交叉注意力；同一请求内复用与候选无关的用户侧计算和 KV cache。核心是把用户侧与候选侧拆开。

</details>

### D9. 时间信息怎么加进行为序列？

<details><summary>参考解答</summary>

最常见的是把“距当前请求的时间间隔”分桶后做 embedding，与行为 embedding 拼接或相加；也可以作为注意力的相对偏置（HSTU 的 rab 包含时间）。分桶通常取对数尺度，因为一小时和两小时的差别远大于一百天和一百零一天。

</details>

## E. 排序模型的 scaling

### E1. 为什么传统排序模型难以 scaling？MFU 是什么？

<details><summary>参考解答</summary>

参数集中在 embedding 表，查表是访存密集操作，稠密计算很少；加大维度对长尾 ID 收益有限。MFU 是实际达到的浮点运算速率占硬件峰值的比例，传统模型在 GPU 上很低。要 scaling 就要把计算挪到规整的大矩阵乘上。[scaling](/notes/recsys-scaling-ranking/)

</details>

### E2. HSTU 的注意力为什么不用 softmax？

<details><summary>参考解答</summary>

softmax 把每行权重归一到和为 1，会抹掉“相关行为有多少条”的强度信息：看过 50 条篮球视频和 1 条，对篮球候选的贡献可能接近。HSTU 用逐点的 SiLU 激活代替，保留强度；再用 $U$ 门控代替 FFN。

</details>

### E3. HSTU 怎样训练、怎样给大量候选打分？

<details><summary>参考解答</summary>

训练用生成式目标，沿用户序列预测下一个物品或动作，每个位置都有监督。推理用 M-FALCON：候选分成小批，共享用户序列部分的计算与缓存，只为候选额外计算一小段。论文报告质量随训练算力在约三个数量级上呈幂律提升。

</details>

### E4. RankMixer 的 token mixing 怎么做，shape 怎样变化？

<details><summary>参考解答</summary>

$T$ 个 $D$ 维特征 token，每个切成 $H$ 段；第 $h$ 个新 token 由所有旧 token 的第 $h$ 段拼成，维度 $TD/H$。取 $H=T$ 时仍是 $T$ 个 $D$ 维 token。无参数，只是重排，但每个新 token 都含所有特征组的信息；之后每个 token 用自己的 FFN。

</details>

### E5. OneTrans 统一了什么？

<details><summary>参考解答</summary>

传统精排先编码行为序列、再与其他特征交叉；OneTrans 把行为 token 和非序列特征 token 放进同一个 Transformer。同质的序列 token 共享参数，异质的非序列 token 各用自己的参数；逐层减少参与计算的序列 token；用户侧 Key/Value 可缓存复用。

</details>

### E6. 怎样证明一个推荐模型“变大有用”？

<details><summary>参考解答</summary>

固定数据与训练轮数，只改规模；横轴取算力或参数、纵轴取离线指标，在对数坐标下看是否接近直线，至少三到四个点；同时报告 MFU 与推理延迟，说明可上线。

</details>

## F. 语义 ID 与生成式推荐

### F1. 为什么需要语义 ID？

<details><summary>参考解答</summary>

物品 ID 千万级、无结构，不能当生成词表；标题太长且不唯一。语义 ID 把内容向量量化成几个离散码，词表只有几百到几千，相似物品共享前缀，新物品有内容就有 ID。[语义 ID](/notes/recsys-semantic-id/)

</details>

### F2. VQ-VAE 的三项损失各更新什么？

<details><summary>参考解答</summary>

重建项更新编码器和解码器（梯度经 STE 到编码器）；码本项 $\|\mathrm{sg}[z]-e\|^2$ 只更新码字，把它拉向编码器输出；承诺项 $\beta\|z-\mathrm{sg}[e]\|^2$ 只更新编码器，防止输出漂移。$\beta$ 常取 0.25。

</details>

### F3. 直通估计（STE）是怎么回事？

<details><summary>参考解答</summary>

$z_q=z+\mathrm{sg}[e-z]$。前向等于 $e$，反向时 $\mathrm{sg}$ 内没有梯度，$\partial z_q/\partial z=I$，重建梯度原样传给编码器，绕过不可导的 argmin。它是近似：假装量化不存在。

</details>

### F4. RQ-VAE 和 RQ-KMeans 怎么选？

<details><summary>参考解答</summary>

RQ-VAE 编码器与码本联合训练，可以学到更适合量化的表示，但容易坍塌、要调参。RQ-KMeans 对内容向量逐层聚类残差，简单稳定、易扩展，没有可学习的编码器。做基线先用 RQ-KMeans；追求效果、有时间调参再用 RQ-VAE，并报告利用率与冲突率。

</details>

### F5. 为什么说 PQ/OPQ 不太适合逐 token 生成？

<details><summary>参考解答</summary>

PQ 把向量切段独立量化，各段地位平等，第一个码没有“粗类别”的含义；生成第一个 token 时模型缺少有意义的粗目标，前缀树的分支也没有语义。残差量化从粗到细，更匹配自回归生成。RQ-OPQ 用前几层残差量化保层级、最后用 OPQ 增区分度。

</details>

### F6. FSQ 为什么不会码本坍塌？

<details><summary>参考解答</summary>

FSQ 没有显式码本，每一维压到有界区间后四舍五入到固定的若干档，隐式码本是规则网格，所有格点都可达，也就不存在“从未被选中所以从未更新的码字”。

</details>

### F7. 怎样度量和缓解码本坍塌？

<details><summary>参考解答</summary>

度量：每层利用率、使用分布的熵与困惑度、基尼系数。缓解：用数据做 K-means 初始化（TIGER 采用）；重置长期未用的码字；EMA 更新；向量归一化或降维；平衡分配。

</details>

### F8. 语义 ID 冲突怎么处理？

<details><summary>参考解答</summary>

内容几乎相同的物品会得到相同 SID。常见做法是末尾追加去重位，冲突物品依次编号；下游层数参数要加 1。去重位没有语义，冲突率越高模型越要“死记”，所以冲突率要报告。

</details>

### F9. 生成式召回为什么需要约束解码？

<details><summary>参考解答</summary>

模型可能生成不存在的 SID 组合。把所有合法 SID 建成前缀树，每一步只保留当前前缀下存在的子节点，其余 token 的 logit 置为 $-\infty$。这样 beam 里的每条结果都对应真实物品。

</details>

### F10. beam size 怎样影响效果和成本？

<details><summary>参考解答</summary>

beam 大小 $B$ 决定一次能召回的物品数，也决定每步保留的假设数；解码 $L$ 步，成本约正比于 $L\times B$。$B$ 太小召回不足，太大延迟线性上升，且后排结果质量下降。还要处理不同 beam 映射到同一物品时的去重。

</details>

### F11. 语义 ID 当特征和当生成目标，成本各在哪？

<details><summary>参考解答</summary>

当特征：只需离线产出 SID，排序模型为 SID 前缀或组合学 embedding，几乎不增加线上成本，长尾共享参数（Google 在 YouTube 排序上的做法）。当生成目标：要训练生成模型、做 beam search 与约束解码，线上成本和召回架构都要变，但能统一召回与排序。

</details>

### F12. tokenizer 重建误差更小，推荐一定更好吗？

<details><summary>参考解答</summary>

不一定。SID 既要有足够语义信息，也要容易被生成模型学会：码本越大、层数越多，信息越多，但每个 token 越难预测，前缀分布可能越偏。实验要固定下游、只换 tokenizer，同时报告 tokenizer 指标和下游指标。

</details>

## G. LLM4Rec

### G1. LLM 在推荐里有哪几种用法？

<details><summary>参考解答</summary>

编码器（NoteLLM、HLLM，离线产出向量）；文本推荐器（P5、TALLRec、LLaRA）；生成式推荐器（LC-Rec、PLUM、OneRec、MiniOneRec，生成 SID）；老师与评审（蒸馏、奖励模型）；数据与内容（补全描述、生成文案、领域后训练）。另有 HSTU 这类只借架构、不用 LLM 权重的推荐大模型。[LLM4Rec](/notes/recsys-llm4rec-paradigms/)

</details>

### G2. P5 用数字表示物品 ID 有什么问题？

<details><summary>参考解答</summary>

数字 ID 会被分词器切成子词，“7391”和“7392”共享子词却毫无关系，模型被迫从无意义的子词组合里学物品；编号方式（随机、顺序、聚类）会显著影响效果。这也是后来转向语义 ID 的动机之一。

</details>

### G3. TALLRec 的局限是什么？

<details><summary>参考解答</summary>

它把推荐变成“是否喜欢这个物品”的二分类，只能逐个打分，不能从全库检索；每个候选一次 LLM 推理，成本高；物品只用标题表示，丢掉协同信号。

</details>

### G4. LLaRA 的混合表示是什么？

<details><summary>参考解答</summary>

把传统序列推荐模型学到的物品 ID embedding 投影到 LLM 的 token 空间，与物品标题一起放进 prompt，让 LLM 同时获得语义和协同信号；训练用课程学习，先纯文本、再逐步加入混合表示。

</details>

### G5. LC-Rec 为什么需要多种对齐任务？

<details><summary>参考解答</summary>

新加入词表的 SID token 对 LLM 是陌生的。只训练“历史 SID → 下一个 SID”，模型学不到 SID 与语言语义的联系。LC-Rec 同时训练 SID 与标题互译、意图推断等任务，在文本和 SID 两个世界之间建立映射。

</details>

### G6. OneRec 的核心设计有哪些？

<details><summary>参考解答</summary>

用平衡 K-means 生成语义 ID；编码器读用户历史，带 MoE 的解码器一次生成一个会话要展示的多个物品；用奖励模型对生成结果打分、构造偏好对，迭代做偏好对齐。目标是用一个生成模型替代召回与排序的级联。

</details>

### G7. MiniOneRec 的三个阶段是什么？RL 为什么用约束 beam search 采样？

<details><summary>参考解答</summary>

SID 构建（冻结文本编码器 + RQ-VAE）→ SFT（下一个 SID 预测 + 语言与 SID 对齐任务）→ 推荐导向的 GRPO。动作空间是有限的物品集合，随机采样会产生重复或非法 SID，组内差异小；约束 beam search 保证一组候选合法且互不相同，组内比较更有信息量。

</details>

### G8. 推荐场景的 RL 奖励怎么设计？

<details><summary>参考解答</summary>

基础是命中奖励；加排名感知项，对高概率但错误的候选惩罚更重；可加入协同过滤分数作为稠密奖励；可加多样性或新颖性约束。要防 reward hacking：奖励模型被讨好、只推热门。组内归一化与 KL 惩罚保持训练稳定。

</details>

### G9. 生成式推荐是不是伪范式？

<details><summary>参考解答</summary>

支持：词表小，参数花在序列模型上，出现 scaling 趋势；共享前缀帮助长尾；统一多阶段目标；可接入 LLM 知识。质疑：不少论文基线偏弱（SASRec 用 BCE 加一个负样本）；SID 不天然含协同信号；beam 限制召回数且成本线性；部分收益来自更多特征或更好的训练配方。评价时看基线是否公平、增益是否随规模持续、线上成本是否可接受。

</details>

### G10. LLM 在推荐里最容易真正上线的方式是什么？

<details><summary>参考解答</summary>

离线使用：用 LLM 编码物品得到向量或 SID 特征，供线上检索和排序使用；或用 LLM 生成标注、做蒸馏，线上跑小模型。这样不增加请求延迟。直接用 LLM 逐个打分或在线生成，受 QPS 与延迟限制，多用于低流量或异步场景。

</details>

## H. 多模态与 I2I

### H1. CLIP 的温度系数为什么设成可学习？

<details><summary>参考解答</summary>

温度决定 softmax 的尖锐程度：太大区分度弱，太小训练不稳定、对假负例敏感。可学习的温度让模型随训练调整，通常会被截断在合理范围内。[CLIP](/notes/vision-video-algorithms/)

</details>

### H2. SigLIP 和 CLIP 的目标有什么不同？

<details><summary>参考解答</summary>

CLIP 对每行、每列做 softmax 交叉熵，需要整个 batch 的相似度矩阵归一化；SigLIP 把每个图文对当成独立的二分类，用 sigmoid 损失，不需要全局归一化，对 batch 大小更友好，分布式实现更省通信。

</details>

### H3. 难负例和假负例是什么关系？

<details><summary>参考解答</summary>

难负例是与正样本相似、模型容易混淆的负样本，能提供更强梯度；但越难的负样本越可能其实是正样本（假负例），例如标签不全、用户没曝光过。只用最难的负例会伤害训练。常见做法是混合简单与难负例、对疑似假负例降权或过滤，并用验证集检查。

</details>

### H4. 冻结的多模态向量直接给推荐用，有什么问题？QARM 怎么改？

<details><summary>参考解答</summary>

表征不匹配：预训练目标是图文匹配，推荐目标是用户行为。表征学不动：离线缓存，推荐梯度传不回去。QARM 先用行为上高相关的物品对微调多模态模型（物品对齐），再把向量量化成码，下游为每个码学 embedding（量化编码）。[多模态](/notes/recsys-multimodal-i2i/)

</details>

### H5. NoteLLM-2 发现了什么问题？

<details><summary>参考解答</summary>

把视觉编码器接到 LLM 上端到端微调做 I2I 表征时，表征几乎只由文字决定，视觉被忽视。修正：多模态上下文学习，把视觉和文本内容分开压缩；后期融合，把视觉特征直接融入最终表征。

</details>

### H6. VLM2Vec 为什么取最后一个 token 作表征？

<details><summary>参考解答</summary>

解码器式模型用因果注意力，只有最后一个位置能看到整个输入，自然汇总了全部信息；配合任务指令，同一模型可以为不同任务产出不同向量。代价是表征依赖 prompt 格式，训练与推理的模板要一致。

</details>

### H7. item2vec 和 EGES 分别利用什么信息？

<details><summary>参考解答</summary>

item2vec 把行为序列当句子，用 skip-gram 学物品向量，只用共现。EGES 在共现图上随机游走，并加入类目、品牌等边信息，每种边信息有 embedding，用可学习权重融合，新物品没有共现时还能靠边信息得到向量。

</details>

### H8. 多模态怎样帮助冷启动？怎样证明？

<details><summary>参考解答</summary>

新物品没有交互，ID 向量是随机初始化；多模态内容向量让它从第一天就有可用表示，并能通过语义 ID 共享相似物品的参数。证明要按物品交互数分桶，单独报告冷启动桶的指标，并用时间切分保证这些物品在训练期确实没有交互。

</details>

### H9. 多卡训练对比学习时，为什么要 all-gather 负样本？

<details><summary>参考解答</summary>

每张卡的 batch 有限，只用本卡样本做负样本，负样本少、对比信号弱。all-gather 把各卡的向量收集起来，负样本数乘以卡数。注意默认的 all-gather 不传梯度，常见做法是只让本卡的那一份保留梯度，或使用支持梯度的 gather 实现。[训练显存](/notes/training-memory-debugging/)

</details>

## I. 广告与业务

### I1. 写出 CPC、CPA 与 oCPM 的 eCPM。

<details><summary>参考解答</summary>

CPC：$\mathrm{bid}\times\mathrm{pCTR}\times1000$。CPA 与 oCPM：$\mathrm{bid}_{\mathrm{CPA}}\times\mathrm{pCTR}\times\mathrm{pCVR}\times1000$，oCPM 按展示计费、由平台代为出价。[广告](/notes/recsys-ads-business/)

</details>

### I2. 举例说明 GSP 为什么不鼓励如实出价。

<details><summary>参考解答</summary>

价值 A=10、B=8、C=6，两个位置点击率 1.0 与 0.8。如实出价 A 拿第一位付 8，剩余 2；A 出 7 落到第二位付 6，剩余 $4\times0.8=3.2$。压价更赚。VCG 下 A 付 6.4，如实出价是占优策略。

</details>

### I3. oCPM 下 pCVR 高估会怎样？

<details><summary>参考解答</summary>

平台替广告主出更高的价、拿更多展示，实际转化不足，广告主转化成本超标，平台要么赔付、要么流失客户；不同广告偏差不一致时还会错排。所以广告模型要盯 PCOC 与分组校准。

</details>

### I4. 预算平滑要解决什么？

<details><summary>参考解答</summary>

避免预算在流量高峰前就花完、或错过优质流量。常按流量预期分配节奏，用 PID 类控制器调出价系数或参与竞价概率，使实际花费曲线贴近计划。

</details>

### I5. 业务规则和模型冲突时怎么办？

<details><summary>参考解答</summary>

先用实验量化规则的收益与代价；规则反映的是日志里学不到的偏好或约束（如连续同类内容的疲劳），可以考虑把它变成模型目标或重排约束，而不是两边各自加码；规则改动同样要 A/B。

</details>

### I6. 转化的延迟反馈是什么问题？

<details><summary>参考解答</summary>

点击后几天才转化，训练时被当成负样本。等久数据准但模型旧，等短负样本有噪声。常设等待窗口，或显式建模转化延迟分布并修正标签。

</details>

## J. 手撕代码

### J1. 不用双重循环实现 AUC（处理并列分数）。

<details><summary>参考解答</summary>

```python
import numpy as np

def auc(y, s):
    y, s = np.asarray(y), np.asarray(s, dtype=float)
    order = np.argsort(s, kind="mergesort")
    s_sorted, y_sorted = s[order], y[order]
    ranks = np.empty(len(s))
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks[i:j + 1] = (i + j) / 2 + 1  # 平均秩，从 1 开始
        i = j + 1
    n_pos = y_sorted.sum()
    n_neg = len(y) - n_pos
    return (ranks[y_sorted == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
```

正样本 $\{0.9,0.6,0.4\}$、负样本 $\{0.7,0.3,0.2\}$ 时，正样本秩和为 13，减 6 后除以 9，得 $7/9$。

</details>

### J2. 实现 NDCG@K（二值相关性）。

<details><summary>参考解答</summary>

```python
import math

def ndcg_at_k(ranked, relevant, k):
    dcg = sum(1 / math.log2(i + 2) for i, x in enumerate(ranked[:k]) if x in relevant)
    idcg = sum(1 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0.0
```

</details>

### J3. 实现对称 InfoNCE。

<details><summary>参考解答</summary>

```python
import torch
import torch.nn.functional as F

def info_nce(q, k, tau=0.07):
    q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
    logits = q @ k.t() / tau                 # [B, B]，对角线是正样本
    labels = torch.arange(q.size(0), device=q.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) / 2
```

</details>

### J4. 实现带因果 mask 的多头注意力。

<details><summary>参考解答</summary>

```python
import math
import torch

def mha(x, Wq, Wk, Wv, Wo, h, causal=True):
    B, T, D = x.shape
    d = D // h
    q = (x @ Wq).view(B, T, h, d).transpose(1, 2)    # [B, h, T, d]
    k = (x @ Wk).view(B, T, h, d).transpose(1, 2)
    v = (x @ Wv).view(B, T, h, d).transpose(1, 2)
    att = q @ k.transpose(-2, -1) / math.sqrt(d)      # [B, h, T, T]
    if causal:
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), 1)
        att = att.masked_fill(mask, float("-inf"))
    att = att.softmax(dim=-1)
    out = (att @ v).transpose(1, 2).reshape(B, T, D)
    return out @ Wo
```

</details>

### J5. 实现残差量化的编码。

<details><summary>参考解答</summary>

```python
import numpy as np

def rq_encode(z, codebooks):
    """z: [N, D]；codebooks: L 个 [K, D] 数组。返回 SID [N, L] 与最终残差。"""
    r = z.copy()
    codes = []
    for C in codebooks:
        dist = ((r[:, None, :] - C[None, :, :]) ** 2).sum(-1)   # [N, K]
        c = dist.argmin(axis=1)
        codes.append(c)
        r = r - C[c]
    return np.stack(codes, axis=1), r
```

用 [语义 ID](/notes/recsys-semantic-id/) 第 2 节的例子验证：输入 $(0.9,0.2)$ 应得到 SID $(0,0)$、残差 $(-0.1,0)$。

</details>

### J6. 用前缀树实现约束解码的合法 token 查询。

<details><summary>参考解答</summary>

```python
def build_trie(sids):
    root = {}
    for sid in sids:
        node = root
        for t in sid:
            node = node.setdefault(t, {})
    return root

def allowed_next(trie, prefix):
    node = trie
    for t in prefix:
        node = node.get(t)
        if node is None:
            return []
    return list(node.keys())
```

解码每一步，对 beam 里每条前缀取合法 token 集合，其余 logit 置为 $-\infty$，再进入 beam 排序。

</details>

## K. 项目与研究追问

### K1. 你的项目基线是什么，为什么比较是公平的？

<details><summary>参考解答</summary>

回答模板：基线来自哪个开源实现与版本；复现数字与原文差多少、原因是什么；所有对照固定数据切分、下游模型、训练步数，只改一个因素；报告多个随机种子的均值与标准差。说不出复现差距，面试官会怀疑后面所有结论。

</details>

### K2. 结果提升不显著怎么办？

<details><summary>参考解答</summary>

先看方差：多种子的标准差有多大，提升是否超过它。再做分桶：总体不显著，冷启动或长尾桶可能显著，这本身就是结论。最后诚实写负结果，并分析原因（例如“哈希 tokenizer 输在层级而非信息量”），一个控制变量干净的负结果比含糊的正结果更有说服力。

</details>

### K3. 从检索或多模态方向转推荐，你会怎样迁移经验？

<details><summary>参考解答</summary>

讲三个对应关系：假负例 ↔ 隐式反馈里“未交互不等于不喜欢”；难负例挖掘 ↔ 召回负样本与 logQ；Hamming 检索与量化 ↔ ETA、语义 ID、IVF-PQ。再说一个差异：推荐的标签来自曝光与行为，分母与偏差更复杂，所以会先把评估协议和纠偏做扎实。[教程入口第 5 节](/notes/recsys-llm4rec-guide/)

</details>

### K4. 如果给你一周时间提升某个推荐场景的线上指标，你怎么做？

<details><summary>参考解答</summary>

先诊断瓶颈：看漏斗，好内容在哪一层消失（召回覆盖、排序误判、重排规则）。再挑成本低、可验证的改动（加一路召回、修正一个特征穿越、调整融分权重）。设计 A/B：主指标、保护指标、样本量与时长。最后准备回滚方案。不要第一天就换模型结构。[涨指标](/notes/wangshusen-recommender-coldstart/)

</details>

## L. 面经补充：2025–2026 年公开面经里的高频问题

检索日期：**2026 年 10 月 5 日**。渠道是牛客网公开帖子；小红书因验证码拦截，本次没有读到正文，所以不在来源里。下面 51 个问题是读完帖子后用自己的话概括的，按主题合并了同类问法，已去掉与上面九十问重复的题。每条只列出能核对到的帖子链接（标题与日期与题目相符），不转录原文，也**不附参考解答**：这一节只是题目清单，用来查漏补缺。有“相关教程”链接的题，可以先去对应章节读机制，再自己写答案。

出现次数只说明有多少篇公开帖子提到，不代表某家公司的真实题库，也不代表面试官一定会问。

### 召回

- **L1.** 双塔召回的固有缺陷有哪些？长尾物品表征学不好怎么办？不换范式、只改双塔结构怎么提升长尾质量？ 来源:[2913652](https://www.nowcoder.com/discuss/2913652)
- **L2.** 双塔 pointwise、pairwise、listwise 三种训练范式的区别和对应 loss？ 相关教程:[入口](/notes/wangshusen-recommender-retrieval/)。 来源:[2913652](https://www.nowcoder.com/discuss/2913652)、[2833239](https://www.nowcoder.com/discuss/2833239)
- **L3.** 数据预处理时直接过滤交互很少的新物品有什么问题？如何打压热门物品？ 来源:[2833249](https://www.nowcoder.com/discuss/2833249)
- **L4.** 图召回解决什么问题？节点、边、多跳怎么设计？如何避免热门偏置？为何不用双塔或矩阵分解？ 相关教程:[入口](/notes/wangshusen-recommender-retrieval/)。 来源:[2907543](https://www.nowcoder.com/discuss/2907543)
- **L5.** 召回阶段做多兴趣建模，多兴趣双塔训练的标签怎么设？SIM 的原理与 top-k 召回改进思路？ 相关教程:[入口](/notes/wangshusen-recommender-retrieval/)。 来源:[2917383](https://www.nowcoder.com/discuss/2917383)
- **L6.** UserCF 与 ItemCF 的核心区别；协同过滤的局限？ 来源:[2753479](https://www.nowcoder.com/discuss/2753479)
- **L7.** 热度召回怎么做，会不会引入未来信息？Word2Vec 用物品 ID 序列训练的做法与理由？ 来源:[2751519](https://www.nowcoder.com/discuss/2751519)
- **L8.** 新上线的独播剧/新商品冷启动，如何快速找到目标用户？ 相关教程:[入口](/notes/wangshusen-recommender-coldstart/)。 来源:[2846992](https://www.nowcoder.com/discuss/2846992)、[2839719](https://www.nowcoder.com/discuss/2839719)

### 排序（粗排/精排/重排）

- **L9.** 粗排样本怎么构造？能否直接用精排的 embedding？怎样让粗排感知未曝光数据？粗精排一致性越高越好吗？ 相关教程:[入口](/notes/wangshusen-recommender-ranking/)。 来源:[2883759](https://www.nowcoder.com/discuss/2883759)
- **L10.** 精排训练的样本、标签、特征与分组怎么构造？静态分与动态分作用？LightGBM 与 LambdaRank 为何同时用？ 相关教程:[入口](/notes/wangshusen-recommender-ranking/)。 来源:[2907543](https://www.nowcoder.com/discuss/2907543)
- **L11.** Wide&Deep、DeepFM、DCN、DIN 分别适合什么场景，各自优缺点？ 来源:[2820690](https://www.nowcoder.com/discuss/2820690)、[2801431](https://www.nowcoder.com/discuss/2801431)
- **L12.** 重排与多样性：MMR 原理；如何从召回到重排全链路提升多样性？ 相关教程:[入口](/notes/wangshusen-recommender-reranking/)。 来源:[2833249](https://www.nowcoder.com/discuss/2833249)、[2913652](https://www.nowcoder.com/discuss/2913652)

### 多任务与评估指标

- **L13.** 多任务跷跷板现象如何发现、如何解决？专家坍缩/同质化的原因与对策？ 相关教程:[入口](/notes/wangshusen-recommender-ranking/)。 来源:[2917383](https://www.nowcoder.com/discuss/2917383)、[2652615](https://www.nowcoder.com/discuss/2652615)
- **L14.** 训练用多个目标（点赞、收藏、评论、关注），推理只用其中几个，这样设计的理由？ 来源:[2837321](https://www.nowcoder.com/discuss/2837321)
- **L15.** AUC 的定义、物理意义、工业计算方法；分数整体放大两倍 AUC 变吗？加一批全正样本 AUC 怎么变？ 相关教程:[入口](/notes/recsys-eval-bias/)。 来源:[2883759](https://www.nowcoder.com/discuss/2883759)、[2870581](https://www.nowcoder.com/discuss/2870581)、[2801431](https://www.nowcoder.com/discuss/2801431)
- **L16.** 正负样本极不均衡（1:1000 或更极端）会带来什么问题？为什么用 Focal Loss？ 相关教程:[入口](/notes/recsys-eval-bias/)。 来源:[2823593](https://www.nowcoder.com/discuss/2823593)、[2917383](https://www.nowcoder.com/discuss/2917383)
- **L17.** CTR 校准用的 listwise 模型结构，如何评估校准效果？ 相关教程:[入口](/notes/recsys-eval-bias/)。 来源:[2652615](https://www.nowcoder.com/discuss/2652615)

### 序列建模与 Scaling

- **L18.** 长短期行为序列的时间窗口与序列长度怎么设，多种行为序列怎么融合？ 相关教程:[入口](/notes/recsys-sequence-long/)。 来源:[2891309](https://www.nowcoder.com/discuss/2891309)、[2823593](https://www.nowcoder.com/discuss/2823593)
- **L19.** 异构特征 Token 化的方案有哪些？非序列特征为何用异构 FFN/QKV，序列特征为何可同构？causal mask 如何支撑一个用户多个物品的推理优化？ 相关教程:[入口](/notes/recsys-scaling-ranking/)。 来源:[2839272](https://www.nowcoder.com/discuss/2839272)
- **L20.** 多模态 embedding 接入精排的融合方式？离线如何评估 embedding？ 相关教程:[入口](/notes/recsys-multimodal-i2i/)。 来源:[2883914](https://www.nowcoder.com/discuss/2883914)

### 生成式推荐与语义 ID

- **L21.** RQ-VAE 的流程、模块、损失与梯度传播？stop-gradient 项代码怎么写？除 STE 外还有哪些办法（如 rotation trick）？ 相关教程:[入口](/notes/recsys-semantic-id/)。 来源:[2917383](https://www.nowcoder.com/discuss/2917383)
- **L22.** 构建 SID 时如何加曝光容量约束？约束带来的量化误差怎么权衡？ 相关教程:[入口](/notes/recsys-semantic-id/)。 来源:[2913652](https://www.nowcoder.com/discuss/2913652)
- **L23.** 生成式召回的 SFT 数据怎么构造与筛选？LoRA 的核心优势？ 相关教程:[入口](/notes/recsys-llm4rec-paradigms/)。 来源:[2913652](https://www.nowcoder.com/discuss/2913652)
- **L24.** 生成式召回延迟高，从哪些维度降？线上延迟多少？ 相关教程:[入口](/notes/recsys-llm4rec-paradigms/)。 来源:[2913652](https://www.nowcoder.com/discuss/2913652)、[2807968](https://www.nowcoder.com/discuss/2807968)
- **L25.** RQ-Kmeans 聚类召回做了什么改进？生成式召回与传统通路如何融合、兜底、配比？拓展到多目标多场景时框架怎么改？ 相关教程:[入口](/notes/recsys-semantic-id/)。 来源:[2917077](https://www.nowcoder.com/discuss/2917077)
- **L26.** 生成式召回为什么优于传统方法，训练有无上限，如何突破？ 相关教程:[入口](/notes/recsys-llm4rec-paradigms/)。 来源:[2807968](https://www.nowcoder.com/discuss/2807968)

### 多模态与表征

- **L27.** BERT 的核心设计思想与训练输入输出；ViT 与文本 Transformer 结构差异及各自做分类的方式？ 来源:[2918287](https://www.nowcoder.com/discuss/2918287)
- **L28.** Qwen-VL、LLaVA、DeepSeek 等多模态模型各自特点？ 相关教程:[入口](/notes/multimodal-models-paper-reading/)。 来源:[2858047](https://www.nowcoder.com/discuss/2858047)

### 搜索与 RAG

- **L29.** 传统检索方式有哪些？RAG 的作用与流程？大模型结合 RAG 为何仍会幻觉，怎么缓解？ 来源:[2822391](https://www.nowcoder.com/discuss/2822391)、[2768568](https://www.nowcoder.com/discuss/2768568)、[2763740](https://www.nowcoder.com/discuss/2763740)
- **L30.** 如何量化检索质量；召回率与准确率；chunk 大小、滑动窗口、知识库动态更新？ 来源:[2818415](https://www.nowcoder.com/discuss/2818415)
- **L31.** 讲讲 Deep Search；ReAct 架构；Agent 如何判断信息已足够、陷入死循环怎么办？ 来源:[2768568](https://www.nowcoder.com/discuss/2768568)

### LLM 基础与后训练

- **L32.** MHA / MQA / GQA / MLA 的区别，KV Cache 的作用与优化，FlashAttention？ 相关教程:[入口](/notes/transformer-attention-rope-gqa/)。 来源:[2907543](https://www.nowcoder.com/discuss/2907543)、[2870581](https://www.nowcoder.com/discuss/2870581)
- **L33.** MHA 的时间复杂度，怎么降？Adam 存什么状态，显存占用与 SGD 对比，一阶二阶矩与偏差修正公式？ 来源:[2822196](https://www.nowcoder.com/discuss/2822196)、[2868452](https://www.nowcoder.com/discuss/2868452)
- **L34.** PPO、DPO、GRPO（含 GSPO、DAPO）的区别；DPO 为什么省掉 critic、DPO 自己有什么问题？ 相关教程:[入口](/notes/policy-gradient-ppo-grpo/)。 来源:[2891972](https://www.nowcoder.com/discuss/2891972)、[2820690](https://www.nowcoder.com/discuss/2820690)、[2814199](https://www.nowcoder.com/discuss/2814199)
- **L35.** GAE 原理，λ、γ 的含义；MC 与 TD 的偏差方差权衡？ 相关教程:[入口](/notes/policy-gradient-ppo-grpo/)。 来源:[2891972](https://www.nowcoder.com/discuss/2891972)
- **L36.** 为什么先 SFT 再 RL？SFT 混入通用数据的比例？SFT 与 DPO 的数据制作差异？ 相关教程:[入口](/notes/multimodal-sft-lora-dpo/)。 来源:[2836470](https://www.nowcoder.com/discuss/2836470)
- **L37.** LoRA 原理与公式、两个矩阵的初始化（A 全零 B 随机行不行）、rank 与 alpha 怎么选、灾难性遗忘怎么办？ 相关教程:[入口](/notes/multimodal-sft-lora-dpo/)。 来源:[2893356](https://www.nowcoder.com/discuss/2893356)、[2838681](https://www.nowcoder.com/discuss/2838681)、[2814199](https://www.nowcoder.com/discuss/2814199)
- **L38.** 从大模型蒸馏到小模型有哪些方法？大模型打标签的延迟、幻觉、冲突怎么解？ 来源:[2907543](https://www.nowcoder.com/discuss/2907543)

### 机器学习基础

- **L39.** 交叉熵与 KL 散度的关系；Dropout 训练与测试的差别及"预估偏高"问题；BN 训练与推理差异？ 来源:[2907543](https://www.nowcoder.com/discuss/2907543)、[2883759](https://www.nowcoder.com/discuss/2883759)
- **L40.** L1 与 L2 正则的区别，为何 L1 产生稀疏解；过拟合怎么判断、怎么缓解？ 来源:[2883759](https://www.nowcoder.com/discuss/2883759)、[2801431](https://www.nowcoder.com/discuss/2801431)、[2804764](https://www.nowcoder.com/discuss/2804764)
- **L41.** GBDT、XGBoost、随机森林的区别；为什么树模型常优于 LR？ 来源:[2751519](https://www.nowcoder.com/discuss/2751519)、[2804764](https://www.nowcoder.com/discuss/2804764)

### 系统设计与业务场景

- **L42.** 结合具体产品（B站、爱奇艺），分析推荐问题；如何把 70% 流量扶持中小创作者？ 来源:[2913652](https://www.nowcoder.com/discuss/2913652)、[2917077](https://www.nowcoder.com/discuss/2917077)、[2846992](https://www.nowcoder.com/discuss/2846992)
- **L43.** 一键三连这类稀疏高价值信号放在链路哪里、怎么建模？弹幕如何变成特征？ 相关教程:[入口](/notes/recsys-ads-business/)。 来源:[2917077](https://www.nowcoder.com/discuss/2917077)
- **L44.** 如何度量同质化、信息茧房、多样性，如何把长期生态健康度放进优化目标？ 来源:[2917077](https://www.nowcoder.com/discuss/2917077)
- **L45.** 广告主价值（advv）怎么定义，转化怎么定义？商业价值与用户体验如何平衡？ 相关教程:[入口](/notes/recsys-ads-business/)。 来源:[2837321](https://www.nowcoder.com/discuss/2837321)

### 代码题（高频）

- **L46.** 手写 MHA / self-attention / cross-attention / MMoE / DeepFM（PyTorch）。 相关教程:[入口](/notes/recsys-interview-bank/)。 来源:[2833239](https://www.nowcoder.com/discuss/2833239)、[2870581](https://www.nowcoder.com/discuss/2870581)、[2893356](https://www.nowcoder.com/discuss/2893356)、[2804764](https://www.nowcoder.com/discuss/2804764)
- **L47.** 第 K 大数（O(N)）、前 K 小、前 K 高频、中位数、快排。 来源:[2837321](https://www.nowcoder.com/discuss/2837321)、[2833249](https://www.nowcoder.com/discuss/2833249)
- **L48.** 括号类：括号生成、最长有效括号；链表类：K 个一组翻转、合并 K 个升序链表。 来源:[2883914](https://www.nowcoder.com/discuss/2883914)、[2891309](https://www.nowcoder.com/discuss/2891309)、[2801888](https://www.nowcoder.com/discuss/2801888)、[2753479](https://www.nowcoder.com/discuss/2753479)
- **L49.** DP/图类：零钱兑换、最长回文子序列、岛屿数量/面积、不同路径变形（最大乘积路径）。 来源:[2896097](https://www.nowcoder.com/discuss/2896097)
- **L50.** 同时在线人数最多的时刻（区间扫描线）。 来源:[2822196](https://www.nowcoder.com/discuss/2822196)
- **L51.** 智力题：12 球称重找异常球、25 马选前 3、圆上三点成锐角三角形概率、掷骰期望。 来源:[2870581](https://www.nowcoder.com/discuss/2870581)、[2801888](https://www.nowcoder.com/discuss/2801888)、[2883759](https://www.nowcoder.com/discuss/2883759)
