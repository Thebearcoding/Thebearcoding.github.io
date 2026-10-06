---
title: 搜索算法入门：从 query 理解到相关性模型
date: '2026-10-06'
tags: [搜索, 搜推, BM25, DSSM, ColBERT, 相关性, NDCG, 面试]
summary: 对照推荐讲清搜索链路：query 理解、BM25 手算、双塔与 cross-encoder 与 ColBERT 的形状和成本、蒸馏、相关性与个性化的融合，以及分档标注与线上指标。
draft: false
---

搜推岗的面试官常会先问一句“搜索和推荐有什么不一样”，再顺着往下问 BM25 的参数、双塔和 cross-encoder 的取舍、相关性怎样标注。推荐的链路你已经学过，这一篇只讲搜索多出来的部分：用户给了一个 **query**，系统必须先读懂它，再保证返回的东西和它相关，然后才谈个性化。

阅读入口：[全覆盖教程](/notes/recsys-llm4rec-guide/)。前置：[召回与双塔](/notes/wangshusen-recommender-retrieval/)、[评估与偏差](/notes/recsys-eval-bias/)（NDCG 在那里推导过，本篇只用不推）。数值算例为教学构造。

## 1. 搜索和推荐差在哪

| 维度 | 推荐 | 搜索 |
|---|---|---|
| 输入 | 用户、上下文、历史行为，没有显式需求 | 多一个显式 query，通常很短，可能有错字 |
| 首要目标 | 猜兴趣，最大化点击、时长、转化 | **相关性是硬约束**：不相关的结果点击率再高也不该排前 |
| 离线评估 | 行为日志上的 AUC、Recall@K、NDCG@K | 人工相关性分档标注上的 NDCG@K，加上行为日志 |
| 线上指标 | CTR、时长、留存 | CTR，加上无点击率、改写率等“没找到”的信号 |
| 失败的样子 | 推荐不够有趣 | 搜“苹果手机壳”出来苹果（水果） |

链路对照：推荐是“召回 → 粗排 → 精排 → 重排”，搜索在最前面加一段 **query 理解**，召回则多一条词法通道：

$$
\text{query} \to \text{QU} \to \underbrace{\text{BM25} \cup \text{ANN}}_{\text{retrieval}} \to \text{pre-rank} \to \text{rank} \to \text{rerank}.
$$

可以共用的模块：ANN 向量索引（HNSW、IVF-PQ）、粗排和精排的特征框架与多目标头、重排的多样性与打散（[MMR、DPP](/notes/wangshusen-recommender-reranking/)）、A/B 平台、物品侧的内容表征。搜索特有的：倒排索引、query 理解、相关性模型与相关性标注体系。推荐里的“精排分”在搜索里通常要拆成两部分：**相关性分**和**个性化/转化分**，第 8 节讲怎么合。

## 2. Query 理解：一个 query 走一遍

**输入**：原始字符串（几个到十几个字）。**输出**：一个结构化的 query 表示，供召回和排序使用：规范化后的词序列、改写候选、意图、实体及其类型、类目分布。

用一个电商 query 走一遍（各步输出均为教学构造，不代表任何公司的实现）：

| 步骤 | 输入 | 输出 | 常见做法 |
|---|---|---|---|
| 归一化 | `苹果手几壳 防摔!!` | `苹果手几壳 防摔` | 全半角、大小写、去符号 |
| 纠错 | `苹果手几壳 防摔` | `苹果手机壳 防摔`（置信度 0.97） | 拼音/字形混淆集 + 语言模型打分 |
| 分词 | `苹果手机壳 防摔` | `[苹果, 手机壳, 防摔]` | 词典 + 序列标注，领域词典要保住“手机壳”不被切开 |
| 实体识别 | 分词结果 | 苹果=品牌，手机壳=产品词，防摔=属性 | 序列标注（BIO） |
| 意图分类 | query 与实体 | 商品购买（而非资讯、导航） | 文本分类 |
| 类目预测 | query 与实体 | 手机配件/手机壳 0.92，水果 0.01 | 多分类，标签来自点击日志的类目分布 |
| 改写与扩展 | 结构化 query | `iPhone 手机壳`、`苹果 手机套` | 同义词表、点击二部图、生成式改写 |

注意“苹果”的歧义靠**上下文与类目预测**消解：单独搜“苹果”时类目分布可能是水果与手机各占一部分，这时结果页需要混排而不是二选一。

**边界与失败**：纠错过度会把冷门的正确词（新品牌名、型号）改掉；改写会引入**语义漂移**（“苹果手机套”扩展成“手机套”后召回了别的品牌）。所以改写结果一般只作为额外召回通道，并在后面的相关性模型里重新把关。

**怎样验证**：每个子模块有自己的离线标注集（纠错准确率与误纠率分开报、实体识别 F1、类目预测 Top-1 准确率）；端到端看 query 级的 NDCG 与线上的无结果率、改写率。误纠率比召回率更要紧，因为一次误纠会让整页结果偏掉。

## 3. 词法检索：BM25

倒排索引把每个词映射到包含它的文档列表。query 分词后取各词的倒排链合并，得到候选，再用 BM25 打分。**输入**：query 词集合 $q$、文档 $d$ 的词频 $\mathrm{tf}(t,d)$、文档长度 $|d|$、平均长度 $\mathrm{avgdl}$、文档总数 $N$、文档频率 $n_t$。**输出**：一个标量分。Lucene 实现的形式是

$$
\mathrm{BM25}(q,d)=\sum_{t\in q}\mathrm{idf}(t)\cdot\frac{\mathrm{tf}(t,d)}{\mathrm{tf}(t,d)+k_1\left(1-b+b\,\frac{|d|}{\mathrm{avgdl}}\right)},\qquad
\mathrm{idf}(t)=\ln\!\left(1+\frac{N-n_t+0.5}{n_t+0.5}\right).
$$

Robertson 与 Zaragoza 的综述（2009，式 3.12–3.15）给出的经典式在分子上多一个 $(k_1+1)$；综述指出这个因子对所有词相同，不改变排序。两个参数：

- $k_1$ 控制**词频饱和**：分数随 $\mathrm{tf}$ 增长但有上限，$k_1$ 越小饱和越快。重复堆砌关键词收益递减，这是 BM25 优于线性 TF-IDF 的关键。
- $b\in[0,1]$ 控制**长度归一化**：$b=1$ 完全按长度归一，$b=0$ 关闭。长文档可能只是啰嗦（应当惩罚），也可能覆盖更多内容（不该惩罚），$b$ 在两者间折中。

默认值：Lucene `BM25Similarity` 取 $k_1=1.2$、$b=0.75$。综述说大量实验表明 $0.5<b<0.8$、$1.2<k_1<2$ 在多数情况下都不错，但最优值依赖文档和 query 类型，模型本身不给设定方法，要在标注集上调。

**手算**（教学构造）。语料 $N=10$，$\mathrm{avgdl}=10$，query 为“降噪 耳机”。“降噪”出现在 2 篇文档里，idf 为 $\ln(1+8.5/2.5)=\ln 4.4\approx1.482$；“耳机”出现在 5 篇里，idf 为 $\ln(1+5.5/5.5)=\ln2\approx0.693$。取 $k_1=1.2$、$b=0.75$。

- 文档 A：长 5，“耳机”出现 2 次，没有“降噪”。长度因子 $0.25+0.75\times0.5=0.625$，$k_1$ 乘以它为 0.75。得分 $0.693\times 2/2.75\approx0.504$。
- 文档 B：长 20，两个词各出现 1 次。长度因子 $0.25+0.75\times2=1.75$，乘 $k_1$ 为 2.1。得分 $(1.482+0.693)\times 1/3.1\approx0.702$。

B 更长、“耳机”更少，仍然胜出，因为它命中了稀有词“降噪”。若把 $b$ 设为 0，A 变为 $0.693\times2/3.2\approx0.433$，B 变为 $2.175\times1/2.2\approx0.989$，差距拉大。再看饱和：A 的长度因子下，$\mathrm{tf}=1$ 时比例项为 0.571，$\mathrm{tf}=10$ 时只到 0.930，上限是 1。

**边界**：BM25 只认字面，“手机套”和“手机壳”没有重叠就得 0 分（词汇鸿沟）；对型号、品牌、编号这类精确匹配需求它反而比向量更可靠。所以工业搜索的召回通常是**词法 + 向量**多路并行。多路结果的合并，一个不需要分数校准的基线是 RRF（Cormack 等，SIGIR 2009）：$\mathrm{RRF}(d)=\sum_{r}1/(k+r(d))$，论文固定 $k=60$。

## 4. 双塔：DSSM 与稠密检索

DSSM（Huang 等，CIKM 2013）是搜索里最早的深度双塔之一。**输入**：query 和文档标题的词袋，先经“词哈希”变成字母三元组向量（如 `#good#` 切成 `#go, goo, ood, od#`），50 万词表压到 30,621 维（论文表 1）。**结构**：词哈希层约 3 万维，两层 300 维隐层，输出 128 维（论文 3.2 节）。**打分**：两塔输出的余弦相似度 $R(q,d)=\cos(y_q,y_d)$。**训练**：对每个（query，点击文档）对，随机取 4 个未点击文档，做带平滑因子 $\gamma$ 的 softmax：

$$
P(d^+\mid q)=\frac{\exp(\gamma R(q,d^+))}{\sum_{d'\in\{d^+\}\cup D^-}\exp(\gamma R(q,d'))},\qquad \mathcal{L}=-\log\prod_{(q,d^+)}P(d^+\mid q).
$$

这和 InfoNCE 是同一个形状，$\gamma$ 相当于温度的倒数，见 [对比学习](/notes/contrastive-learning/)。论文表 2 在 5 档人工标注的评估集上报告：BM25 的 NDCG@1 为 0.308，最好的 DSSM（L-WH DNN）为 0.362。

今天的双塔把词袋 MLP 换成 BERT 类编码器，形状不变：

| 模型 | 在线输入 | 中间形状 | 输出 | 文档侧能否离线 | 每 query 在线成本 |
|---|---|---|---|---|---|
| 双塔 | query 的 $L_q$ 个 token | $[L_q,h]\to[d]$ | 点积或余弦 | 能，$[N,d]$ 建 ANN | 1 次 query 编码 + ANN 检索 |
| cross-encoder | $[\mathrm{CLS}]\,q\,[\mathrm{SEP}]\,d\,[\mathrm{SEP}]$ | $[L_q+L_d+3,h]$ | 标量 | 不能 | $k$ 次完整前向 |
| ColBERT | query 的 $N_q$ 个 token | $E_q:[N_q,m]$，$E_d:[L_d,m]$ | MaxSim 求和 | 能，存 $[N,L_d,m]$ | 1 次 query 编码 + $N_q\times L_d$ 点积/候选 |

双塔的问题是 query 和文档在编码时互相看不见，整篇文档被压成一个 $d$ 维向量，细粒度匹配（“不含酒精”里的否定、型号里的一位数字）容易丢。

## 5. 交互式：cross-encoder 精排

monoBERT（Nogueira 与 Cho，arXiv 1901.04085）是 BERT 重排的标准基线。按论文第 2 节：query 作为句子 A、段落作为句子 B 拼接，query 截断到 64 个 token，总长不超过 512；用 BERT-Large，取 $[\mathrm{CLS}]$ 向量过单层网络输出“相关”的概率 $s_j$，用二元交叉熵训练（式 1），负例来自 BM25 前 1000 中的非相关段落。论文表 1：MS MARCO 开发集 MRR@10，BM25 为 16.7，BERT-Large 重排后为 36.5。

**成本**：每个候选都要跑一次完整前向，注意力对序列长度是平方复杂度，候选数 $k$ 线性放大。ColBERT 论文表 1 在单张 V100 上测得，BERT-base 重排 BM25 前 1000 个候选需约 10,700 ms，约 97T FLOPs。所以 cross-encoder 只能放在候选已经很少的精排或相关性打分环节，或者作为离线老师（第 7 节）。

**失败场景**：截断会切掉长文档后半部分的证据；对标注分布外的 query（长尾、新词）过于自信；线上用作相关性门槛时，分数要先校准才能设阈值。

## 6. ColBERT：晚交互

ColBERT（Khattab 与 Zaharia，SIGIR 2020）在两者之间折中：query 和文档仍然**分开编码**，但每个 token 都保留一个向量，打分时再交互。按论文式 (1)–(3)：

$$
E_q=\mathrm{Normalize}(\mathrm{CNN}(\mathrm{BERT}(q))),\quad
E_d=\mathrm{Filter}(\mathrm{Normalize}(\mathrm{CNN}(\mathrm{BERT}(d)))),\quad
S_{q,d}=\sum_{i=1}^{\lvert E_q\rvert}\max_{j}\;E_{q_i}\cdot E_{d_j}^{\top}.
$$

这里 CNN 指一个线性层，把维度降到 $m$；向量做 L2 归一化，所以点积就是余弦。query 不足 $N_q$ 个 token 时用 $[\mathrm{mask}]$ 补齐（论文称 query augmentation），Filter 去掉文档里的标点向量。实验设置 $N_q=32$、$m=128$（4.1.2 节）。训练用 $(q,d^+,d^-)$ 三元组上的 pairwise softmax 交叉熵；MaxSim 本身没有可训练参数。

**手算**（教学构造）。query 有 2 个 token，两篇文档各 3 个 token，相似度矩阵（行是 query token，列是文档 token）为

$$
S^{(1)}=\begin{pmatrix}0.9&0.1&0.3\\0.2&0.4&0.8\end{pmatrix},\qquad
S^{(2)}=\begin{pmatrix}0.6&0.6&0.5\\0.5&0.6&0.5\end{pmatrix}.
$$

每行取最大再相加：文档 1 得 $0.9+0.8=1.7$，文档 2 得 $0.6+0.6=1.2$。文档 1 对两个 query 词各有一个强匹配；文档 2 处处“有点像”，但没有一个词被真正对上。若把每篇文档矩阵的 6 个数取平均（类似单向量的粗粒度相似），文档 1 为 0.45、文档 2 为 0.55，顺序反过来。

**成本**：ColBERT 论文表 1 中重排 BM25 前 1000，MRR@10（开发集）34.9，延迟 61 ms，约 7B FLOPs；同表 BERT-base 为 34.7（引自 monoBERT）或 36.0（ColBERT 作者重训），延迟 10,700 ms。代价转移到存储：每篇文档要存 $L_d\times m$ 个数而不是 $d$ 个。端到端检索时，论文用 faiss IVFPQ 对每个 query 向量各取近邻，合并候选后再精确算 MaxSim（3.6 节）。

## 7. 蒸馏：把 cross-encoder 教给双塔

线上召回只能用双塔，效果最好的却是 cross-encoder。常见做法是离线用 cross-encoder 给训练三元组打分，再让学生模仿。Hofstätter 等（arXiv 2010.02666）提出 Margin-MSE（式 11）：不要求学生复现老师的绝对分数，只要求**正负例的分差**一致，

$$
\mathcal{L}(q,p^+,p^-)=\mathrm{MSE}\big(M_s(q,p^+)-M_s(q,p^-),\;M_t(q,p^+)-M_t(q,p^-)\big).
$$

理由是不同结构的输出尺度差别很大（拼接式 BERT 与点积双塔的分数分布不同），对齐分差比对齐绝对值更现实。论文的老师是拼接式 BERT（BERT_CAT）及其三模型平均的集成，学生包括 BERT 点积双塔、ColBERT、PreTT、TK。

**手算**（教学构造）。老师给正例 3.1、负例 1.0，分差 2.1；学生给 0.9 与 0.4，分差 0.5。损失 $(0.5-2.1)^2=2.56$。学生只要把分差拉到 2.1 即可，绝对值是 0.9 还是 5.0 都无所谓。

**边界**：老师的错误会被学生继承；老师打分的候选分布要和学生线上面对的候选分布接近，否则只学会了区分“容易的负例”。验证方法：在同一份人工标注集上比较“只用点击训练的双塔”和“蒸馏双塔”的 Recall@K 与 NDCG@K，再看线上相关性抽检的坏结果率。

## 8. 相关性和个性化怎样合在一起

搜索的排序分一般包含两类信号：相关性分 $r(q,d)$（模型拟合人工标注）和个性化/转化分（模型拟合点击、购买，用到用户特征）。合法的融合方式有多种，取舍在于相关性是否被当作硬约束。

**手算**（教学构造）。三个结果的 $(r,\ \mathrm{pCTR})$：A $(0.90,0.05)$，B $(0.50,0.20)$，C $(0.85,0.08)$。

- **线性加权** $s=r+3\,\mathrm{pCTR}$：A 1.05，B 1.10，C 1.09，排序 B、C、A。B 只有半相关，却因为点击率高排第一，这正是搜索要避免的“标题党上位”。
- **分档门控**：先按 $r\ge0.7$ 分出相关档，档内按 pCTR 排，档外整体放后或截断。结果 C、A、B。相关性决定“能不能上”，个性化决定“同档谁先”。

门控的失败场景：阈值依赖相关性分的校准，模型一更新分数分布变了，阈值就失效；门槛太严会让长尾 query 结果太少。乘法融合 $s=r^{\alpha}\cdot\mathrm{pCTR}^{\beta}$ 介于两者之间，$r$ 接近 0 时整体分趋近 0。无论哪种，都要在**相关性评估集**和**线上行为指标**两边同时报告，只看 CTR 会系统性地奖励不相关但吸引眼球的结果。

## 9. 标注、NDCG 与线上指标

**相关性标注分档**。人工按 query 判断每个结果满足需求的程度，分成若干档。DSSM 论文的评估集用 0–4 五档。Google 公开的《Search Quality Rater Guidelines》（2025-09-11 版，第 13 节）的 Needs Met 量表为 Fully Meets、Highly Meets、Moderately Meets、Slightly Meets、Fails to Meet；该文档第 0.1 节说明单个评分不会直接改变某个结果的排名，评分用于衡量搜索系统的效果，并提供好坏结果的样例。分档要配标注说明和样例，并抽检标注一致性。

**分级增益的 NDCG**。公式见 [评估与偏差](/notes/recsys-eval-bias/) 第 5 节，分级时常用增益 $2^{\mathrm{rel}}-1$。例子（教学构造）：三个结果标注为 2、3、0 档，理想顺序是 3、2、0。

$$
\mathrm{DCG}=3+\frac{7}{\log_2 3}\approx7.417,\qquad \mathrm{IDCG}=7+\frac{3}{\log_2 3}\approx8.893,\qquad \mathrm{NDCG@3}\approx0.834.
$$

若用线性增益（直接用档位），同一排序得 $3.893/4.262\approx0.913$。指数增益让“把最高档放错位置”的惩罚更重，适合头部结果最要紧的搜索场景。

**离线评估**：固定一份 query 抽样（覆盖头部、腰部、长尾）与标注，报 NDCG@K；新模型召回到旧标注里没有的结果时，要补标，否则未标注结果会被当作 0 分。

**线上指标**：

| 指标 | 含义 | 陷阱 |
|---|---|---|
| CTR | 点击 / 曝光 | 标题党可以拉高它 |
| 无点击率 | 有结果却没点任何结果的搜索比例 | 答案直接显示在结果页时，不点击也可能是满意 |
| 改写率 | 用户短时间内修改 query 重搜的比例 | 用户主动细化需求（加属性词）不一定是坏事 |
| 无结果率 | 召回为空的 query 比例 | 改写放宽可以降低它，但可能引入漂移 |

单一指标都能被“刷”，所以要组合看，并和人工相关性抽检互相印证。

## 10. 和跨模态检索的对照

你做过跨模态检索，搜索的大部分结构可以直接迁移：

| 你熟悉的 | 文本搜索里的对应 | 差异 |
|---|---|---|
| CLIP 式双塔 + InfoNCE | DSSM、稠密检索双塔 | 搜索有词法通道，型号、品牌这类精确匹配 BM25 更稳 |
| 单流融合的图文匹配模型 | cross-encoder 精排 | 成本结构相同：每个候选一次前向，只能放在后段 |
| FILIP 的 token 级最大相似 | ColBERT 的 MaxSim | 都是“分开编码、晚交互”；文本侧存储随 $L_d$ 线性增长 |
| mAP、Recall@K | 分档标注 + NDCG@K | 搜索的标签是多档人工判断，不是类别是否重叠 |
| 难负例与假负例 | 未点击 ≠ 不相关；BM25 前 1000 里也有漏标的正例 | 蒸馏老师可以帮忙识别假负例 |

最大的差别在 query 一侧：跨模态检索的 query 通常是完整句子或整张图，搜索 query 只有几个词、带错字、有歧义，所以多了一整套 query 理解；另外搜索要把相关性当硬约束，和行为信号分开建模、分开评估。

## 闭卷验收

不看资料，说出搜索与推荐在输入、目标、评估上的三处差别，并画出“query 理解 → 词法与向量召回 → 粗排/精排 → 重排”的链路，标出哪些模块能和推荐共用；拿一个歧义 query 讲清纠错、分词、实体识别、意图与类目预测各自的输入输出；写出 BM25 公式，说出 $k_1$、$b$ 的作用和 Lucene 默认值，手算一次两文档打分；写出 DSSM 的训练 softmax、cross-encoder 的输入格式、ColBERT 的 MaxSim，并比较三者的张量形状、能否离线、在线成本；写出 Margin-MSE；手算一次线性融合与分档门控的排序差异；用 $2^{\mathrm{rel}}-1$ 手算一次分级 NDCG；列出无点击率与改写率各自的陷阱。

**参考。** [Robertson & Zaragoza, The Probabilistic Relevance Framework: BM25 and Beyond, FnTIR 2009](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf)；[Lucene BM25Similarity 文档](https://lucene.apache.org/core/9_9_0/core/org/apache/lucene/search/similarities/BM25Similarity.html)；[Cormack et al., Reciprocal Rank Fusion, SIGIR 2009](https://dl.acm.org/doi/10.1145/1571941.1572114)；[Huang et al., DSSM, CIKM 2013](https://www.microsoft.com/en-us/research/publication/learning-deep-structured-semantic-models-for-web-search-using-clickthrough-data/)；[Nogueira & Cho, Passage Re-ranking with BERT](https://arxiv.org/abs/1901.04085)；[Khattab & Zaharia, ColBERT, SIGIR 2020](https://arxiv.org/abs/2004.12832)；[Hofstätter et al., Cross-Architecture Knowledge Distillation (Margin-MSE)](https://arxiv.org/abs/2010.02666)；[Järvelin & Kekäläinen, Cumulated Gain-based Evaluation of IR Techniques, TOIS 2002](https://dl.acm.org/doi/10.1145/582415.582418)；[Google Search Quality Rater Guidelines](https://guidelines.raterhub.com/searchqualityevaluatorguidelines.pdf)；[FILIP](https://arxiv.org/abs/2111.07783)。
