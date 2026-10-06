---
title: 图推荐：从 GCN 到 LightGCN
date: '2026-10-06'
tags: [推荐系统, 图神经网络, GCN, NGCF, LightGCN, PinSage, 召回]
summary: 把交互日志写成二部图，从 GCN 的对称归一化讲到 NGCF 的消息构造，再讲 LightGCN 为什么删掉变换和非线性反而更好，手算一层传播，最后讲过平滑、PinSage 的工业做法和面试追问。
draft: false
---

协同过滤的核心假设是“相似的人喜欢相似的东西”。矩阵分解只用一阶信号：用户 $u$ 点过物品 $i$，就把两个向量拉近。可是“和我点过同一件物品的人，还点过什么”是二阶、三阶的信号，矩阵分解只能靠训练间接学到。**图推荐**把交互日志显式写成一张用户-物品二部图，用图卷积把多跳邻居的信息直接“传”进嵌入。这一篇讲清这条线上的四个模型：GCN 给出传播公式，NGCF 把它搬进推荐，LightGCN 发现其中两样东西是累赘，PinSage 讲怎样在几十亿节点上落地。

阅读入口：[全覆盖教程](/notes/recsys-llm4rec-guide/)。前置：[召回：协同过滤、双塔与向量索引](/notes/wangshusen-recommender-retrieval/)。本页数值算例为教学构造，论文数字均注明表号。

## 1. 推荐为什么能看成二部图

**输入**：$M$ 个用户、$N$ 个物品，隐式反馈（点击、购买）。交互矩阵 $R\in\{0,1\}^{M\times N}$，$R_{ui}=1$ 表示 $u$ 和 $i$ 有交互。用户只连物品、物品只连用户，同类节点之间没有边，所以这是一张**二部图**。

把两类节点排在一起编号，邻接矩阵是 $(M+N)\times(M+N)$ 的分块矩阵（LightGCN 论文式 (6)）：

$$
A=\begin{pmatrix}0 & R\\ R^\top & 0\end{pmatrix},\qquad D=\mathrm{diag}(A\mathbf{1}).
$$

左上 $M\times M$ 与右下 $N\times N$ 全为零，正好表达“同类不相连”。度矩阵 $D$ 的对角元是节点的邻居数：用户的度是交互过的物品数 $\lvert\mathcal{N}_u\rvert$，物品的度是交互过它的用户数 $\lvert\mathcal{N}_i\rvert$，也就是物品的热度。

**为什么要图**：在图上走两步 $u\to i\to v$ 就是 UserCF 的“相似用户”，走三步 $u\to i\to v\to j$ 就是“相似用户喜欢的物品”。图卷积把这类路径编码进嵌入，而不是在线上临时查邻居。

**形状约定**：嵌入表 $E^{(0)}\in\mathbb{R}^{(M+N)\times d}$（LightGCN 论文把嵌入维度记作 $T$，实验取 64）。注意 NGCF 论文的记号相反，用 $N$ 表示用户数、$M$ 表示物品数，读原文时别混。

## 2. GCN：对称归一化与自环

Kipf & Welling 的 GCN（ICLR 2017）用于有节点特征的半监督节点分类。逐层传播规则（原文第 2 节式 (2)）：

$$
H^{(l+1)}=\sigma\!\left(\tilde D^{-1/2}\tilde A\tilde D^{-1/2}H^{(l)}W^{(l)}\right),\qquad \tilde A=A+I_N,\quad \tilde D_{ii}=\textstyle\sum_j\tilde A_{ij}.
$$

$H^{(l)}\in\mathbb{R}^{N\times D}$ 是第 $l$ 层激活，$H^{(0)}=X$ 是节点特征；$W^{(l)}$ 是逐层可训练矩阵，$\sigma$ 如 ReLU。三个零件各有分工：

- **自环** $A+I$：让节点聚合时也带上自己，否则表示只由邻居决定。
- **对称归一化** $\tilde D^{-1/2}\cdot\tilde D^{-1/2}$：边 $(v,w)$ 的权重是 $1/\sqrt{\tilde d_v\tilde d_w}$，防止高度节点的表示越加越大。原文称这一步为 renormalization trick：一阶近似得到的算子 $I+D^{-1/2}AD^{-1/2}$ 特征值在 $[0,2]$，反复作用会数值不稳定，于是替换成 $\tilde D^{-1/2}\tilde A\tilde D^{-1/2}$（原文 2.2 节、式 (8)）。
- **变换 $W$ 与非线性 $\sigma$**：节点有词袋这类语义特征时，多层非线性变换有助于学特征。第 4 节会看到，推荐里这两样恰恰是问题。

深度上，GCN 附录 B 在 Cora、Citeseer、Pubmed 上的实验里 2 到 3 层最好，超过 7 层不加残差就难训练。

## 3. NGCF：把 GCN 搬进协同过滤

NGCF（*Neural Graph Collaborative Filtering*，SIGIR 2019）输入只有 ID 嵌入。一层传播分两步。

**消息构造**（原文式 (3)）：物品 $i$ 发给用户 $u$ 的消息

$$
m_{u\leftarrow i}=\frac{1}{\sqrt{\lvert\mathcal{N}_u\rvert\lvert\mathcal{N}_i\rvert}}\Big(W_1e_i+W_2(e_i\odot e_u)\Big),\qquad W_1,W_2\in\mathbb{R}^{d'\times d}.
$$

系数 $p_{ui}=1/\sqrt{\lvert\mathcal{N}_u\rvert\lvert\mathcal{N}_i\rvert}$ 就是 GCN 的对称归一化。新东西是 $e_i\odot e_u$：逐元素乘积让消息依赖 $u$ 与 $i$ 的相似度，作者称之为类似注意力的作用，相似的物品传更多信息。

**消息聚合**（原文式 (4)）：

$$
e_u^{(1)}=\mathrm{LeakyReLU}\Big(m_{u\leftarrow u}+\sum_{i\in\mathcal{N}_u}m_{u\leftarrow i}\Big),\qquad m_{u\leftarrow u}=W_1e_u.
$$

$m_{u\leftarrow u}$ 是自环，和消息共用 $W_1$。第 $l$ 层每层有自己的 $W_1^{(l)},W_2^{(l)}\in\mathbb{R}^{d_l\times d_{l-1}}$（式 (5)(6)）。$L$ 层后把各层**拼接**：$e_u^*=e_u^{(0)}\Vert\cdots\Vert e_u^{(L)}$（式 (9)），打分 $\hat y=e_u^{*\top}e_i^*$，用 BPR 训练，另有 message dropout 与 node dropout 两种正则。

**手算**（教学构造，沿用第 6 节的小图，$d=2$，取 $W_1=W_2=I$）：$e_{u_1}=(1,0)$，邻居 $i_1=(1,0)$、$i_2=(1,1)$，系数分别为 $1/\sqrt2\approx0.7071$ 与 $0.5$。来自 $i_1$ 的消息 $0.7071\cdot\big((1,0)+(1,0)\big)=(1.4142,0)$，来自 $i_2$ 的消息 $0.5\cdot\big((1,1)+(1,0)\big)=(1,0.5)$，加自环 $(1,0)$ 得 $(3.4142,0.5)$，全为正，LeakyReLU 不改变它。对比第 6 节 LightGCN 同一位置的 $(1.2071,0.5)$：$\odot$ 项放大了 $u$ 与邻居“对得上”的那一维。

**论文证据**：NGCF 表 4 中，NGCF-1 在三个数据集上都优于把消息换成 GC-MC、PinSage 式线性消息的变体；表 3 中 3 层在三个数据集上都优于 1 层（Gowalla 上 2 层略低于 1 层），作者认为 NGCF-4 在 Yelp2018 上过拟合。

## 4. LightGCN：删掉变换和非线性

LightGCN（*Simplifying and Powering Graph Convolution Network for Recommendation*，SIGIR 2020）先对 NGCF 做消融，再从头设计。

**消融**（LightGCN 表 1，2 层，NGCF 的最终嵌入从拼接改为求和以便比较嵌入质量）：

| 变体 | 去掉了什么 | Gowalla recall@20 | Amazon-Book recall@20 |
|---|---|---|---|
| NGCF | 无 | 0.1547 | 0.0330 |
| NGCF-f | 特征变换 $W_1,W_2$ | 0.1686 | 0.0368 |
| NGCF-n | 非线性 $\sigma$ | 0.1536 | 0.0336 |
| NGCF-fn | 两者都去掉 | 0.1742 | 0.0399 |

结论三条（论文 2.2 节）：去掉变换稳定变好；只去非线性影响不大，但在已去掉变换的基础上再去非线性，提升明显；两者一起是负作用。**为什么去掉反而更好？** 论文的解释分两层。一是输入：节点分类里每个节点有标题、摘要这类语义特征，多层非线性变换能学到更好的特征；协同过滤里节点只有一个 ID，没有可变换的语义。二是训练：图 1 中 NGCF-fn 的训练损失在全程都更低，且低训练损失转化成了更高的测试 recall，所以 NGCF 变差**源于训练困难而不是过拟合**。理论上 NGCF 表达力更强（把 $W$ 设成单位阵就退化为 NGCF-f），实践中却训练得更差。

**传播规则**（LightGCN 式 (3)），不加自环、不变换、不激活：

$$
e_u^{(k+1)}=\sum_{i\in\mathcal{N}_u}\frac{1}{\sqrt{\lvert\mathcal{N}_u\rvert}\sqrt{\lvert\mathcal{N}_i\rvert}}e_i^{(k)},\qquad
e_i^{(k+1)}=\sum_{u\in\mathcal{N}_i}\frac{1}{\sqrt{\lvert\mathcal{N}_i\rvert}\sqrt{\lvert\mathcal{N}_u\rvert}}e_u^{(k)}.
$$

矩阵形式（式 (7)）：$E^{(k+1)}=\tilde AE^{(k)}$，$\tilde A=D^{-1/2}AD^{-1/2}$，注意这里的 $A$ 不加单位阵。

**层组合**（式 (4)(8)）：

$$
e_u=\sum_{k=0}^{K}\alpha_ke_u^{(k)},\qquad E=\sum_{k=0}^{K}\alpha_k\tilde A^kE^{(0)},\qquad \alpha_k=\frac{1}{K+1}.
$$

论文说均匀的 $1/(K+1)$ 普遍效果好，所以不去学它；在训练集上学 $\alpha$ 没有提升，在验证集上学只提升不到 1%。层组合有三个理由：深层会过平滑，只用最后一层有问题；不同层捕捉不同阶的相似性；它等价于自环的效果（3.2.1 节证明 $(A+I)^K$ 展开就是各阶 $A^k$ 的加权和），所以传播时不必再加自环。

**预测与损失**（式 (5)(15)）：

$$
\hat y_{ui}=e_u^\top e_i,\qquad
L_{\mathrm{BPR}}=-\sum_{u=1}^{M}\sum_{i\in\mathcal{N}_u}\sum_{j\notin\mathcal{N}_u}\ln\sigma(\hat y_{ui}-\hat y_{uj})+\lambda\lVert E^{(0)}\rVert^2.
$$

唯一可训练参数是 $E^{(0)}$，参数量与矩阵分解相同；没有 $W$ 也就不需要 dropout，$L_2$ 正则足够（3.3 节）。官方 PyTorch 实现里，BPR 写成 `softplus(neg - pos)` 的均值，层组合写成各层嵌入的 `mean`，两者分别等价于上面的 $-\ln\sigma$ 与 $1/(K+1)$。

**对 NGCF 的提升**（表 3）：1 到 4 层的每个设置下 LightGCN 都优于 NGCF；Gowalla 上 4 层 recall@20 为 0.1830 对 0.1570。正文报告三个数据集平均 recall 提升 16.52%、ndcg 提升 16.87%。超参数：嵌入维度 64，Adam，学习率 0.001，$\lambda$ 多数情况取 $10^{-4}$，$K$ 取 3 时普遍令人满意（4.1 节）。

**归一化消融**（表 5，3 层）：两侧开方的对称归一化最好；只保留一侧会大幅下降；完全去掉归一化会训练出 NaN，论文没有列这一行。

## 5. 二阶平滑：LightGCN 到底在做什么

展开两层（3.2.3 节），另一个用户 $v$ 对 $u$ 的平滑强度是

$$
c_{v\to u}=\frac{1}{\sqrt{\lvert\mathcal{N}_u\rvert}\sqrt{\lvert\mathcal{N}_v\rvert}}\sum_{i\in\mathcal{N}_u\cap\mathcal{N}_v}\frac{1}{\lvert\mathcal{N}_i\rvert}.
$$

三点读法：共同交互的物品越多，影响越大；共同物品越冷门（$\lvert\mathcal{N}_i\rvert$ 小），越能说明个性化偏好，影响越大；$v$ 越不活跃，影响越大。这和 ItemCF、Swing 里对热门物品、活跃用户降权是同一个直觉（见 [召回](/notes/wangshusen-recommender-retrieval/)）。表 6 用一个平滑度损失验证：2 层 LightGCN-single 的用户、物品嵌入都比 MF 平滑得多，作者认为嵌入平滑是 LightGCN 有效的关键原因。

## 6. 手算一层 LightGCN

**教学构造**：2 个用户、3 个物品。$u_1$ 交互 $i_1,i_2$，$u_2$ 交互 $i_2,i_3$。

$$
R=\begin{pmatrix}1&1&0\\0&1&1\end{pmatrix}\in\mathbb{R}^{2\times3},\qquad A\in\mathbb{R}^{5\times5},\qquad \mathrm{diag}(D)=(2,2,1,2,1).
$$

节点顺序 $(u_1,u_2,i_1,i_2,i_3)$。非零边权 $1/\sqrt{d_ud_i}$：$u_1$–$i_1$ 为 $1/\sqrt{2}\approx0.7071$，$u_1$–$i_2$ 为 $1/2$，$u_2$–$i_2$ 为 $1/2$，$u_2$–$i_3$ 为 $0.7071$。

初始嵌入（$d=2$）：$u_1=(1,0)$，$u_2=(0,1)$，$i_1=(1,0)$，$i_2=(1,1)$，$i_3=(0,1)$。

| 节点 | $e^{(0)}$ | $e^{(1)}=$ 邻居加权和 | $e=\tfrac12(e^{(0)}+e^{(1)})$ |
|---|---|---|---|
| $u_1$ | $(1,0)$ | $0.7071(1,0)+0.5(1,1)=(1.2071,0.5)$ | $(1.1036,0.25)$ |
| $u_2$ | $(0,1)$ | $0.5(1,1)+0.7071(0,1)=(0.5,1.2071)$ | $(0.25,1.1036)$ |
| $i_1$ | $(1,0)$ | $0.7071(1,0)=(0.7071,0)$ | $(0.8536,0)$ |
| $i_2$ | $(1,1)$ | $0.5(1,0)+0.5(0,1)=(0.5,0.5)$ | $(0.75,0.75)$ |
| $i_3$ | $(0,1)$ | $0.7071(0,1)=(0,0.7071)$ | $(0,0.8536)$ |

$K=1$，$\alpha_0=\alpha_1=1/2$。$u_1$ 对三个物品的分数：$i_1$ 为 $1.1036\times0.8536\approx0.942$，$i_2$ 为 $0.75\times(1.1036+0.25)\approx1.015$，$i_3$ 为 $0.25\times0.8536\approx0.213$。直接用 $E^{(0)}$ 做内积（即 MF）时 $u_1\cdot i_3=0$；传播一层后 $u_1$ 经由共同物品 $i_2$ 和 $u_2$ 拿到了一点 $i_3$ 的方向，这就是“高阶协同信号”的最小例子。

BPR 一项：正样本 $i_1$、负样本 $i_3$，差 $0.942-0.213=0.729$，$\sigma(0.729)\approx0.674$，损失 $-\ln0.674\approx0.394$。

再验算第 5 节的系数：$u_1,u_2$ 只共享 $i_2$，$c_{u_2\to u_1}=\frac{1}{\sqrt2\sqrt2}\cdot\frac12=0.25$。直接算 $E^{(2)}=\tilde A^2E^{(0)}$ 得 $e_{u_1}^{(2)}=(0.75,0.25)$，其中第二维的 0.25 正是 $0.25\times e_{u_2}^{(0)}$。

## 7. 过平滑：层数越多越像

$\tilde A$ 的特征值都在 $[-1,1]$。反复乘 $\tilde A$，嵌入会被推向少数几个主特征向量，不同节点的表示越来越像，这就是**过平滑**。上面的小图里（教学构造，自行计算）：$u_1,u_2$ 初始余弦相似度为 0，传播 1 层后为 0.707，传播 10 层后为 0.998，两个口味完全不同的用户几乎分不开了。二部图的 $\tilde A$ 还有特征值 $-1$（本例特征值为 $\pm1,\pm0.7071,0$），奇偶层会来回摆动，所以相似度不是单调上升的，这也是只取某一层不稳的一个原因。

论文证据：LightGCN 图 4 中，只用最后一层的 LightGCN-single 在多数情况下 2 层最好，到 4 层降到最差；加了层组合的 LightGCN 随层数增加逐渐变好，4 层也不退化。3.2.2 节说明层组合可以恢复 APPNP 的形式，而 APPNP 借个性化 PageRank 的“回到起点”来对抗过平滑。

**边界**：层组合缓解但不消除过平滑；Amazon-Book 和 Yelp2018 上 2 层 LightGCN-single 反而最好（图 4 讨论），说明均匀的 $\alpha$ 不一定最优。**怎样验证**：画层数-指标曲线，同时画嵌入两两余弦相似度的分布随层数的变化；分用户活跃度分桶看，冷用户和重度用户的最佳层数可能不同。

## 8. 工业落地：全图传播的代价与 PinSage

**代价**。$\tilde AE^{(k)}$ 是稀疏乘稠密，$A$ 有 $2\lvert R^+\rvert$ 个非零元，一层大约 $O(\lvert R^+\rvert d)$（自行估算，论文未单独给出）。问题不在一次乘法，而在训练：官方 PyTorch 实现每个 batch 都调用一次全图传播再取出 batch 内的用户和物品。图上亿节点时，全图嵌入放不进单卡，每步都传播也不现实。另一个限制是**直推式**：$E^{(0)}$ 只是 ID 表，新用户、新物品没有行，必须重新训练。

**PinSage**（*Graph Convolutional Neural Networks for Web-Scale Recommender Systems*，KDD 2018）在 Pinterest 的 pin-board 二部图上部署：摘要给出训练用 75 亿样本，图有 30 亿节点、180 亿条边。要点：

1. **按需卷积**：不乘全图拉普拉斯，而是围绕每个目标节点动态构造局部计算图（引言贡献列表）。
2. **随机游走定义邻居**：从 $u$ 出发模拟随机游走，按 $L_1$ 归一化的访问次数取前 $T$ 个节点作为邻居（3.2 节），访问次数在无限次模拟下逼近个性化 PageRank。好处是邻居数固定，内存可控，且访问次数可直接当聚合权重，论文称为 **importance pooling**。
3. **GraphSAGE 式卷积**（算法 1）：邻居表示过一层 $\mathrm{ReLU}(Qh_v+q)$，按权重聚合成 $n_u$，与自身表示**拼接**后再过一层 $\mathrm{ReLU}(W\cdot\mathrm{concat}(z_u,n_u)+w)$，最后 $L_2$ 归一化。输入是图像和文本特征，所以是**归纳式**：没见过的 pin 也能算嵌入。
4. **训练**：max-margin 排序损失 $\mathbb{E}_{n_k\sim P_n(q)}\max\{0,z_q\cdot z_{n_k}-z_q\cdot z_i+\Delta\}$；难负例取对查询 pin 的个性化 PageRank 排名第 2000–5000 的物品，按课程学习逐轮加入，第 $n$ 轮每个样本加 $n-1$ 个难负例（3.3 节）。
5. **推理**：用 MapReduce 一次算全量嵌入，避免 $K$ 跳邻域重叠带来的重复计算（3.4 节）。

论文数字：表 1 中 PinSage 的 hit-rate 67%、MRR 0.59，最好的非图内容基线为 27% 与 0.37；表 4 中邻居数 $T$ 从 10、20 到 50，hit-rate 为 60%、63%、67%，训练时间为 20、33、78 小时，收益递减。实验用 $K=2$ 层。

## 9. 和矩阵分解、双塔的关系；评估协议

**与 MF**：$K=0$ 时 LightGCN 就是 BPR-MF。$K>0$ 时最终嵌入 $E=\sum_k\alpha_k\tilde A^kE^{(0)}$ 是对 MF 嵌入做了一个固定的线性图滤波，可训练参数仍然只有 $E^{(0)}$。所以 LightGCN 本质上是**被交互图平滑过的 MF 嵌入**。LightGCN 表 4 拿 GRMF 作对照：GRMF 在 BPR 损失里加图拉普拉斯正则来平滑嵌入，三个数据集上都不如 LightGCN（Gowalla recall@20 为 0.1477 对 0.1830）。作者据此对比“在损失里平滑”和“在模型前向里平滑”两种做法。

**与双塔**：打分仍是内积，训练完把用户、物品嵌入导出建 ANN 索引，线上用法和双塔召回相同（索引必须与模型成套更新，见 [召回](/notes/wangshusen-recommender-retrieval/)）。差别在“塔”是什么：双塔的塔是特征上的神经网络，可以吃画像、上下文、内容特征，新物品有特征就有向量；LightGCN 的塔是 ID 表加固定的图传播，强在高阶协同信号，弱在冷启动和特征利用。

**评估协议**。NGCF 与 LightGCN 用同一套：每个用户随机取 80% 交互作训练、其余作测试；对每个测试用户，**所有未交互物品都是候选**（全排序，all-ranking），训练集正例排除在外；报告 recall@20、ndcg@20 的用户平均（NGCF 4.2.1 节，LightGCN 4.1 节）。两点要警惕：这是按用户随机切分，不是按时间切分，存在“用未来预测过去”的风险；论文之间的数字只在同一切分、同一指标实现下可比，NGCF 脚注提到它修正过 ndcg 的实现并重跑了所有方法。指标定义、采样评测为什么会弄反排序、时间切分，见 [评估与偏差](/notes/recsys-eval-bias/)。

## 10. 面试常问

**图召回解决什么问题？** 把多跳协同信号显式编码进嵌入，主要帮交互少的用户：NGCF 图 4 按活跃度分组，Gowalla 上交互数小于 24 和小于 50 两组相对最佳基线提升 8.49% 和 7.79%，最活跃一组只有 1.29%。顺带一句：图召回常作为多路召回中的一路，和 ItemCF、双塔互补。

**怎样避免热门偏置？** 先说来源（这里是推理，不是上面论文的实验结论）：高度物品出现在大量用户的邻域里，传播后容易和很多用户都相似，BPR 若均匀采负样本，热门物品作为负例的频率又低于它在曝光中的频率。能说出处的手段：对称归一化本身会对热门物品降权（第 5 节的 $1/\lvert\mathcal{N}_i\rvert$），且 LightGCN 表 5 说明归一化方式对效果影响很大。工程上常见的做法（经验，非上面论文的结论）：调节归一化里度的指数；按热度做负采样或 logQ 纠偏；评估时按物品热度分桶报告 recall，并看推荐结果的覆盖率和长尾占比。

**为什么不直接用双塔？** 不是二选一。双塔吃特征、能处理冷启动、训练按 batch 可扩展；LightGCN 只吃 ID、直推式、全图传播成本高，但在纯 ID 场景下把高阶信号用得更充分。工业上要图信号又要可扩展，就走 PinSage 路线：采样邻居、用内容特征做归纳式卷积；或者把图嵌入离线算好，当作双塔或排序的输入特征。

**追问：为什么 LightGCN 不加自环？** 层组合 $\sum_k\alpha_k\tilde A^k$ 已经包含 $k=0$ 的自身项，$(A+I)^K$ 展开后也只是各阶的加权和，二者效果相同（3.2.1 节）。

**追问：NGCF 的 $e_i\odot e_u$ 有用吗？** NGCF 表 4 说有用；LightGCN 认为 NGCF-fn 仍保留的自环、$\odot$ 项和 dropout “可能也没用”，因为删掉之后的 LightGCN 比 NGCF-fn 更好（4.2 节），但它没有逐项单独消融 $\odot$ 项。回答时要把这两个结论的实验条件说清楚。

## 闭卷验收

写出 $R$ 与 $A$ 的分块形状和度矩阵含义；写出 GCN 的 renormalization 传播式，说出自环和对称归一化各防什么；写出 NGCF 的消息与聚合式，指出 $e_i\odot e_u$、自环、$W_1,W_2$、LeakyReLU 各在哪；说出 LightGCN 删了哪三样（变换、非线性、自环）以及各自的理由，引用表 1 和图 1 说明“训练困难而非过拟合”；手算第 6 节的一层传播与一个 BPR 项；推出 $c_{v\to u}$ 并解读三个因素；解释过平滑和层组合的关系；说出 PinSage 的随机游走邻居、importance pooling、难负例课程学习各解决什么；最后说清 LightGCN、MF、双塔三者的关系，以及全排序 recall@20 的协议细节。

**参考。** [GCN（Kipf & Welling, ICLR 2017）](https://arxiv.org/abs/1609.02907)；[NGCF（SIGIR 2019）](https://arxiv.org/abs/1905.08108)、[官方代码](https://github.com/xiangwang1223/neural_graph_collaborative_filtering)；[LightGCN（SIGIR 2020）](https://arxiv.org/abs/2002.02126)、[ACM DL](https://dl.acm.org/doi/10.1145/3397271.3401063)、[官方 TensorFlow 代码](https://github.com/kuandeng/LightGCN)、[官方 PyTorch 代码](https://github.com/gusye1234/LightGCN-PyTorch)；[PinSage（KDD 2018）](https://arxiv.org/abs/1806.01973)；[GraphSAGE](https://arxiv.org/abs/1706.02216)；[APPNP](https://arxiv.org/abs/1810.05997)；[BPR](https://arxiv.org/abs/1205.2618)。
