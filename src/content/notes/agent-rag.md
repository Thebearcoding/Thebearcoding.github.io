---
title: RAG 全链路：切块、检索、重排、生成与评估
date: '2026-10-06'
tags: [Agent, RAG, 检索增强, DPR, Self-RAG, ColPali, RAGAS]
summary: 从 RAG 原论文的两种边缘化讲起，过一遍切块、稠密/稀疏/混合检索、重排与生成，再讲 Self-RAG 和 CRAG 的自适应检索、ColPali 的文档图像检索，以及 Recall@k、MRR 与 faithfulness 的算法和坑。
draft: false
---

面试问 RAG，常见的追问有三层：一是链路，文档怎样切、用什么检索、要不要重排、怎样拼进提示；二是原理，RAG 原论文到底优化的是什么概率，DPR 怎样训练；三是评估和改进，检索错了怎么办，怎样判断答案是“检索来的”而不是“编的”。这一篇按这三层走。BM25、双塔、cross-encoder、ColBERT 和 RRF 在 [搜索笔记](/notes/recsys-search/) 里已经推过，这里只链接、不重复；让模型自己决定“查什么、查几次”的 RL 训练（Search-R1）见 [Agent RL 预备](/notes/recsys-agent-rl/)。

阅读入口：[Agent 专题总览](/notes/agent-guide/)。数值算例为教学构造；论文数字均注明表号或章节。

## 1. 全链路一图：离线建索引，在线四步

**离线**：原始文档 → 解析（PDF、HTML、表格）→ 切块 → 编码 → 写入索引（倒排索引给 BM25，向量索引给稠密检索）。**在线**：query → 检索 top-$K$（可多路）→ 重排取 top-$k$ → 拼提示生成 → 引用与后处理。

| 环节 | 输入 | 输出 | 常见实现 | 主要失败 |
|---|---|---|---|---|
| 切块 | 文档 token 序列，长 $L$ | $n$ 个块，每块不超过 $c$ token | 定长滑窗、按段落/标题 | 答案被切断；块太长稀释向量 |
| 检索 | query 文本 | top-$K$ 块 ID 与分数 | BM25、双塔 + ANN、混合 | 词汇鸿沟；稠密模型不认型号编号 |
| 重排 | query 与 $K$ 个块 | top-$k$ 块（$k\ll K$） | cross-encoder | 成本随 $K$ 线性增长 |
| 生成 | 指令 + 问题 + $k$ 个块 | 答案（可带引用） | LLM | 不用上下文、编造、引用错位 |

这张表也是排错的顺序：答错了，先看正确块有没有进 top-$K$（召回问题），再看有没有进 top-$k$（重排问题），最后看模型拿到正确块后有没有用对（生成问题）。三处各有指标，第 8 节讲。

## 2. RAG 原论文：把文档当隐变量

RAG（Lewis et al., arXiv 2005.11401，NeurIPS 2020）由两部分组成：检索器 $p_\eta(z\mid x)$ 给出文档 $z$ 的分布，生成器 $p_\theta(y_i\mid x,z,y_{1:i-1})$ 是 BART-large（论文 §2.3 写 400M 参数），输入是把 $x$ 和 $z$ 直接拼接。检索器沿用 DPR 双塔（§2.2）：

$$
p_\eta(z\mid x)\propto\exp\big(\mathbf d(z)^\top\mathbf q(x)\big),\qquad \mathbf d(z)=\mathrm{BERT}_d(z),\ \mathbf q(x)=\mathrm{BERT}_q(x).
$$

文档对生成是隐变量，用 top-$K$ 近似做边缘化。两种模型的区别只在**在哪一层求和**（§2.1）：

$$
p_{\text{RAG-Seq}}(y\mid x)\approx\sum_{z\in\text{top-}k}p_\eta(z\mid x)\prod_{i=1}^{N}p_\theta(y_i\mid x,z,y_{1:i-1}),
\qquad
p_{\text{RAG-Tok}}(y\mid x)\approx\prod_{i=1}^{N}\sum_{z\in\text{top-}k}p_\eta(z\mid x)\,p_\theta(y_i\mid x,z,y_{1:i-1}).
$$

RAG-Sequence 整句用同一篇文档，最后对文档加权平均；RAG-Token 每个 token 都可以换一篇文档。分类任务（目标长度为 1）时两者等价。

**训练**（§2.4）：最小化 $-\sum_j\log p(y_j\mid x_j)$，不给“该检索哪篇”的监督。文档编码器 $\mathrm{BERT}_d$ 和索引**固定**，只微调查询编码器和 BART，理由是更新文档编码器就要周期性重建索引，代价大且作者认为没必要。**语料**（§3）：2018 年 12 月维基百科，每篇切成不重叠的 100 词块，共 21M 个文档；FAISS 的 HNSW 近似索引；训练时 $k\in\{5,10\}$。**解码**（§2.5）：RAG-Token 可以直接塞进普通 beam search；RAG-Sequence 的似然不能逐 token 分解，要对每篇文档各跑一次 beam，再对没出现在某篇 beam 里的候选补跑前向（Thorough Decoding），或者把它们的概率近似为 0（Fast Decoding）。

**手算：两种边缘化的差别**（教学构造）。top-2 文档先验 $p_\eta(z_1\mid x)=0.6$、$p_\eta(z_2\mid x)=0.4$；答案两个 token。$z_1$ 下两步概率为 $0.9,\ 0.2$；$z_2$ 下为 $0.3,\ 0.9$（$z_1$ 只含答案的前半，$z_2$ 只含后半）。

- RAG-Sequence：$0.6\times0.9\times0.2+0.4\times0.3\times0.9=0.108+0.108=0.216$。
- RAG-Token：第 1 步 $0.6\times0.9+0.4\times0.3=0.66$，第 2 步 $0.6\times0.2+0.4\times0.9=0.48$，乘积 $0.3168$。

答案要拼两篇文档的信息时，RAG-Token 给的似然更高，因为它允许每个 token 从更合适的文档里取。反过来，若答案整句都由同一篇支撑，两者差别很小。

**结论与边界。** Table 1 的开放域 QA（EM）：NQ 上 RAG-Token 44.1、RAG-Seq 44.5，DPR（抽取式）41.5，T5-11B 闭卷 34.5。作者的消融还冻结检索器做了对照（Table 6 的 “Frozen” 行），NQ 上 RAG-Seq 由 44.0 降到 41.2（开发集）。注意 $p_\eta(z\mid x)$ 在整个序列里不变，RAG-Token 换的只是“每步用哪篇”的混合，而不是重新检索；真正“边生成边重新检索”是 Self-RAG、Search-R1 这类方法的事。

## 3. 切块：块数、重叠与粒度

**输入输出。** 文档长 $L$ token，块长 $c$，重叠 $o$（$0\le o<c$），步长 $s=c-o$。当 $L\le c$ 时只有 1 块；否则

$$
n=1+\left\lceil\frac{L-c}{s}\right\rceil,
$$

第 $i$ 块（从 0 计）覆盖 $[is,\ \min(is+c,L))$。总存储的 token 数约为 $L+(n-1)o$，冗余倍数约 $1+(n-1)o/L$。

**手算**（教学构造）。$L=1000$，$c=256$，$o=64$，$s=192$：$n=1+\lceil744/192\rceil=1+\lceil3.875\rceil=5$。起点为 0、192、384、576、768；最后一块是 768 到 1000，只有 232 token。存储的 token 数 $4\times256+232=1256$，冗余 1.256 倍。不重叠（$o=0$）时 $n=\lceil1000/256\rceil=4$，最后一块同样是 232 token。重叠多出 1 个块、25.6% 的存储和编码成本，换来的是：任何不超过 $o=64$ token 的片段都至少完整落在一个块里（不重叠时，跨在 256 边界上的句子会被切开）。

**规模估算。** DPR 的语料（§3）是 21,015,324 个不重叠 100 词段落，每个 768 维；fp32 平铺存储为 $21{,}015{,}324\times768\times4\approx64.6$ GB（约 60.1 GiB），fp16 减半。块切得越碎，向量数、索引内存和检索延迟一起涨。

**边界。** ① 重叠不是越大越好：DPR 论文脚注 5 说他们试过重叠切分，没有发现比不重叠更好。② 块太短，单块缺上下文（代词、表格表头丢失）；块太长，一个向量要概括多个话题，稠密检索变差，且生成时占上下文。常见补救：块前加文档标题（DPR 每个段落都前置了维基标题，§4.1），或“小块检索、返回所在大块”。③ 表格、代码、PDF 版式不适合按 token 硬切，第 7 节的 ColPali 干脆按页检索。

**怎样验证。** 固定检索器和评测集，只改 $(c,o)$ 和是否加标题，比较检索 Recall@k 与端到端答案准确率，同时记录索引大小。要报告 token 数相同条件下的对比：$k$ 个短块和 $k$ 个长块喂给生成器的 token 数不同，不对齐预算的比较不公平。

## 4. 检索：稀疏、稠密与混合

**稀疏**：BM25，公式与手算见 [搜索笔记第 3 节](/notes/recsys-search/)。对型号、人名、编号这类精确匹配可靠，对同义改写无能为力。

**稠密：DPR**（Karpukhin et al., arXiv 2004.04906，EMNLP 2020）。两个独立的 BERT-base，取 [CLS] 向量，$d=768$，相似度为点积 $\mathrm{sim}(q,p)=E_Q(q)^\top E_P(p)$（式 1）。训练目标是正段落的负对数似然（式 2）：

$$
\mathcal L=-\log\frac{e^{\mathrm{sim}(q_i,p_i^+)}}{e^{\mathrm{sim}(q_i,p_i^+)}+\sum_{j=1}^{n}e^{\mathrm{sim}(q_i,p_{i,j}^-)}}.
$$

负样本用 in-batch：batch 内 $B$ 个问题与 $B$ 个正段落组成 $S=QP^\top\in\mathbb R^{B\times B}$，对角线是正例，其余 $B-1$ 个是负例（§3.2）。主实验用 batch 128，外加每个问题 1 个 BM25 难负例（§5）。这和 InfoNCE 同形，温度、假负例等细节见 [对比学习](/notes/contrastive-learning/) 和 [多模态 embedding 检索](/notes/multimodal-embedding-retrieval/)。Table 2（NQ 测试集，top-20 准确率，即 top-20 里有含答案段落的问题比例）：BM25 59.1，DPR 78.4；但 SQuAD 上 BM25 68.8 高于 DPR 63.2，作者认为原因之一是 SQuAD 的问题由看过段落的标注者写成，与段落字面重叠大（§5.1）。

**混合。** 方式一是 DPR 论文的线性融合（§5）：BM25 与 DPR 各取 top-2000，对并集按 $\mathrm{BM25}(q,p)+\lambda\cdot\mathrm{sim}(q,p)$ 重排，$\lambda=1.1$ 由开发集选定。方式二是不需要分数校准的 RRF，见 [搜索笔记第 3 节末](/notes/recsys-search/)。

**手算：线性融合对尺度敏感**（教学构造）。$\lambda=1.1$。文档 A：BM25 14.2、sim 78.0，融合分 $14.2+85.8=100.0$；文档 B：BM25 6.0、sim 84.0，融合分 $6.0+92.4=98.4$。A 排第一。若把 BM25 换成另一种实现、分数整体放大 2 倍，A 为 $128.6$、B 为 $104.4$，差距拉大；若稠密模型换成余弦相似度（取值在 $[-1,1]$），sim 项几乎不起作用。所以 $\lambda$ 只对特定的“检索器对”有效，换模型必须重调；RRF 只用名次，避开了这一问题，代价是丢掉分数差的信息。

**HyDE：先写一篇假答案再检索**（Gao et al., arXiv 2212.10496，ACL 2023）。没有相关性标注时，用 InstructGPT（text-davinci-003）按指令“写一段回答这个问题的文字”生成 $N$ 篇假文档，用无监督的 Contriever 编码，取平均作为查询向量，论文还把原 query 的向量也算进去（式 8）：

$$
\hat{\mathbf v}_q=\frac{1}{N+1}\Big[\sum_{k=1}^{N}f(\hat d_k)+f(q)\Big].
$$

假文档可能事实错误，论文的论点是编码器会把细节“压掉”，只保留相关性模式。Table 1：TREC DL19 上 nDCG@10，Contriever 44.5，HyDE 61.3，BM25 50.6。边界：多一次 LLM 生成，查询延迟和成本上去；LLM 对冷门领域写出的假文档可能把检索带偏，论文自己也把歧义 query 留作未来工作（§3.2）。

**怎样验证。** 分 query 类型（实体/编号类、改写类、长问题）报 Recall@k，混合检索的收益通常集中在某一类；再看两路的重叠率，重叠很高时混合意义不大。

## 5. 重排与生成

**重排。** 检索取 $K$（几十到几百），cross-encoder 逐对打分后取 $k$（个位数）喂给生成器。结构、训练和成本见 [搜索笔记第 5 节](/notes/recsys-search/)；cross-encoder 也可当老师蒸馏双塔（同篇第 7 节）。ColBERT 的晚交互是二者之间的折中（同篇第 6 节），本文第 7 节的 ColPali 就是它在页面图像上的版本。

**生成的输入格式**（通用写法，非某篇论文规定）。

```
system: 只根据给定资料回答；资料不足时说明无法回答；每句话后用 [编号] 标出处。
user:   资料：
        [1] (标题 A) 块文本 …
        [2] (标题 B) 块文本 …
        问题：…
```

**失败场景。** ① 模型忽略资料、用参数知识作答（资料与常识冲突时尤甚）；② 资料都不相关时仍硬答，而不是拒答；③ 引用编号和内容对不上；④ 块太多，有用信息淹没在中间。对应的验证：构造“资料与常识冲突”“资料全无关”两类测试集，分别看是否跟随资料、拒答率；引用正确率可以人工抽检，或让一个 NLI/LLM 判官检查“被引块是否支持该句”。

## 6. 自适应检索：Self-RAG 与 CRAG

固定“每次都检索 $k$ 篇”有两个问题：不需要检索时引入噪声，检索错了也照单全收。

**Self-RAG**（Asai et al., arXiv 2310.11511，ICLR 2024）。把“要不要检索、检索得好不好、答案有没有依据”都做成词表里的**反思 token**，让生成器自己输出（Table 1）：

| 类型 | 输入 | 取值 | 含义 |
|---|---|---|---|
| Retrieve | $x$ 或 $x,y$ | yes / no / continue | 是否检索 |
| IsRel | $x,d$ | relevant / irrelevant | 段落 $d$ 对解 $x$ 是否有用 |
| IsSup | $x,d,y$ | fully / partially / no support | $y$ 中需验证的陈述是否被 $d$ 支持 |
| IsUse | $x,y$ | 5 到 1 | $y$ 是否是有用的回答 |

**训练**（§3.2）：先用 GPT-4 给样本标反思 token，蒸馏成一个 Llama2-7B 判官模型 $C$（与 GPT-4 的一致率在多数类别上超过 90%，附录 Table 5）；再用 $C$ 和检索器离线给 150k 条指令数据插入反思 token 与检索段落，训练生成器（Llama2 7B/13B），检索段落在损失里掩掉。推理时不再需要判官。**推理**（§3.3）：Retrieve=yes 的归一化概率超过阈值就检索（§4.3 默认阈值 0.2）；对每篇段落并行生成一个片段（论文以句子为片段），按式 (3)(4) 打分做片段级 beam search（宽度 2）：

$$
f(y_t,d)=p(y_t\mid x,d,y_{<t})+\sum_{G\in\{\text{IsRel},\text{IsSup},\text{IsUse}\}}w_G\,s_t^G,
$$

其中 $s_t^G$ 是该组里“最理想取值”（如 IsRel=relevant）的概率在组内归一化后的值，默认 $w=(1.0,1.0,0.5)$（§4.3）。

**手算**（教学构造）。两篇段落各生成一个候选句。$d_1$：$p=0.40$，$s^{\text{Rel}}=0.9$、$s^{\text{Sup}}=0.3$、$s^{\text{Use}}=0.8$，得分 $0.40+0.9+0.3+0.5\times0.8=2.00$。$d_2$：$p=0.30$，$s^{\text{Rel}}=0.8$、$s^{\text{Sup}}=0.9$、$s^{\text{Use}}=0.6$，得分 $0.30+0.8+0.9+0.3=2.30$。生成概率低一些、但证据支持更强的 $d_2$ 胜出。把 $w_{\text{Sup}}$ 调到 0，两者变为 1.70 和 1.40，结论反转：权重是推理时可调的旋钮，不用重新训练。

**CRAG**（Yan et al., arXiv 2401.15884；会议信息未核实）。在检索和生成之间插一个轻量**检索评估器**：T5-large（0.77B）对每个“问题–文档”对打 $[-1,1]$ 的相关分（§4.2、附录 B）。按上下两个阈值触发三种动作（§4.3）：至少一篇高于上阈值为 **Correct**，对文档做“分解–过滤–重组”的知识精炼，切成小条、用评估器去掉低分条；全部低于下阈值为 **Incorrect**，丢掉检索结果，让 ChatGPT 把问题改写成关键词去调网页搜索；其余为 **Ambiguous**，两种知识都用。阈值是经验设定的，PopQA 上为 (0.59, −0.99)（附录 B）。Table 4：在 PopQA 的检索结果上，这个评估器判断准确率 84.3%，直接提示 ChatGPT 为 58.0%。Table 1：以 SelfRAG-LLaMA2-7b 为生成器，PopQA 准确率 RAG 52.8、CRAG 59.8、Self-RAG 54.9、Self-CRAG 61.8。

**边界。** Self-RAG 要改词表、重新训练生成器；CRAG 是即插即用，但多了评估器调用、可能的网页搜索，阈值随数据集不同。两者的评测都只在 PopQA、PubHealth、ARC、Biography 等少数集合上，迁移到私有知识库时阈值与权重都要重调。**验证**：报告检索触发率与准确率的曲线（Self-RAG 的 Figure 3c 画了阈值 δ 变化时 PubHealth、PopQA 上的检索频率与准确率，§5.2），以及三种动作各自的占比和各自子集上的准确率。

## 7. 多模态 RAG：ColPali 与 ViDoRe

**问题。** 企业文档多是 PDF：表格、图表、版式本身就是信息。传统做法是版面检测 + OCR + 图片说明 + 切块，再走文本检索，离线处理慢，且视觉信息丢失。

**ColPali**（Faysse et al., arXiv 2407.01449，ICLR 2025）直接把**页面截图**当文档（§4.1）。PaliGemma-3B 的语言模型对图像 patch token 和文本 token 都输出隐向量，加一层投影降到 $D=128$；查询 $E_q\in\mathbb R^{N_q\times D}$，页面 $E_d\in\mathbb R^{N_d\times D}$，用 ColBERT 式晚交互打分（式 1）：

$$
\mathrm{LI}(q,d)=\sum_{i=1}^{N_q}\max_{1\le j\le N_d}\langle E_q^{(i)},E_d^{(j)}\rangle .
$$

损失（式 2）对每个 query 取正例分 $s_k^+$ 和 batch 内最难负例分 $s_k^-=\max_{l\ne k}\mathrm{LI}(q_k,d_l)$，写成 $\frac1b\sum_k\mathrm{softplus}(s_k^- - s_k^+)$。训练集 118,695 个 query–页面对（63% 来自学术数据集，37% 是网页 PDF 加 Claude-3 Sonnet 生成的伪问题），1 个 epoch，LoRA $r=\alpha=32$，batch 32，学习率 5e-5（§4.2）；query 后追加 5 个 `<unused0>` 作查询扩展。

**ViDoRe 基准**（§3，Table 1）：页面级检索，10 个任务。学术类 5 个由 VQA 数据集改造：DocVQA、InfoVQA、arXivQA 各 500 query/500 页，TAT-DQA 1600/1600，法语 TabFQuAD 210/210；实用类 5 个（能源、政府、医疗、AI、法语的 Shift Project），每个 1000 页、100 个由 Claude-3 Sonnet 生成并经人工筛选的 query。主指标 nDCG@5。Table 2：ColPali 平均 81.3；最好的文本管线 Unstructured + 图片说明 + BGE-M3 为 67.0，Unstructured + OCR + BM25 为 65.5；原版 SigLIP 51.4。消融链条也在同表：SigLIP 微调成双塔 58.6，换成 PaliGemma 单向量双塔 58.8，加晚交互到 81.3，提升主要来自多向量。Figure 2：每页离线索引，PDF 解析管线 7.22 s，ColPali 0.39 s。

**手算：存储**（教学构造，数字可与论文对照）。每页存 1024 个图像 patch 向量加 6 个文本提示 token（§5.2 说的 “Describe the image”），共 1030 个 128 维 fp16 向量：$1030\times128\times2=263{,}680$ 字节 $=257.5$ KiB，与附录 Table 4 的 257.5 KB 一致；同表 BGE-M3 为 8.60 KB。100 万页约 263.7 GB。打分时一个 query 若有 25 个向量，每页要做 $25\times1030=25{,}750$ 次 128 维点积。论文 §5.2 报告 token pooling 池化因子 3 时向量数减少 66.7%，保留 97.8% 的性能，但文字最密的 Shift 数据集掉得最多。

**边界与验证。** ① 存储与检索成本比单向量高一到两个数量级，大库要配合压缩或先用单向量粗召回；② 检索到的是整页图像，生成器必须是能读图的 VLM，答案定位到页内哪个区域要另做；③ 学术任务的 query 来自 VQA 数据，实用任务的 query 是模型生成的，和真实用户 query 有分布差。验证时在自有文档上抽一批真实问题做页级标注，对比“OCR + 文本检索”与 ColPali 的 nDCG@5 和 Recall@k，并记录离线索引时长与单页存储。ViDoRe 之后的版本本文未核对。

**GraphRAG 一句话**（Edge et al., arXiv 2404.16130）：用 LLM 从语料抽实体知识图谱，再为紧密相关的实体社区预先生成摘要；回答“这批文档的主要主题是什么”这类全局问题时，先用各社区摘要生成部分答案再汇总，摘要报告在百万 token 规模的数据上比普通 RAG 更全面、更多样。

## 8. 评估：检索指标与生成指标

**检索指标。** 对 query $q$，相关集合 $R_q$，检索列表前 $k$ 个为 $T_q^k$：

$$
\mathrm{Recall@}k=\frac{\lvert R_q\cap T_q^k\rvert}{\lvert R_q\rvert},\qquad
\mathrm{MRR}=\frac{1}{\lvert Q\rvert}\sum_{q}\frac{1}{\mathrm{rank}_q},
$$

$\mathrm{rank}_q$ 是第一个相关项的名次，前若干名都没有就记 0。DPR 报告的 “top-$k$ accuracy” 是“前 $k$ 个里至少有一个含答案”，即 Hit@k，和多相关文档时的 Recall@k 不是一回事，读论文时要看定义。

**手算**（教学构造）。三个 query，$k=5$，MRR 截到 10。Q1 相关 $\{d_3\}$，$d_3$ 排第 2；Q2 相关 $\{d_7,d_9\}$，$d_9$ 排第 1、$d_7$ 排第 6；Q3 相关 $\{d_4\}$，前 10 都没有。

- Recall@5：Q1 为 1，Q2 为 $1/2$，Q3 为 0，均值 $0.5$。
- Hit@5：1、1、0，均值 $2/3\approx0.667$。
- MRR@10：$1/2$、$1$、$0$，均值 $0.5$。

**生成指标：RAGAS**（Es et al., arXiv 2309.15217，EACL 2024 Demo）。不依赖参考答案，三个量都靠 LLM（论文用 gpt-3.5-turbo-16k）计算（§3）：

- **Faithfulness**：先让 LLM 把答案拆成若干陈述 $S$，再逐条判断能否由上下文推出，支持的集合为 $V$，$F=\lvert V\rvert/\lvert S\rvert$。
- **Answer relevance**：让 LLM 根据答案反向生成 $n$ 个问题 $q_i$，用 text-embedding-ada-002 算它们与原问题的余弦相似度，$\mathrm{AR}=\frac1n\sum_i\mathrm{sim}(q,q_i)$。它不管事实对错，惩罚答非所问、不完整和冗余。
- **Context relevance**：让 LLM 从上下文中抽出回答所必需的句子，$\mathrm{CR}=$ 抽出句数 / 上下文总句数。

**手算**（教学构造）。问题“埃菲尔铁塔在哪座城市，何时建成”，上下文 8 句，答案“埃菲尔铁塔位于巴黎。它于 1889 年建成。它高 330 米。它由古斯塔夫·埃菲尔设计。”拆成 4 条陈述；上下文写了地点、年份、设计者，没写高度。$F=3/4=0.75$。反向生成 3 个问题，与原问题的余弦为 0.92、0.85、0.71，$\mathrm{AR}=2.48/3\approx0.827$。8 句上下文里必需的只有 2 句，$\mathrm{CR}=2/8=0.25$。注意高度那条可能恰好是对的，但 faithfulness 只问“能否由上下文推出”，所以照样扣分；它衡量的是“有据”，不是“正确”。

**边界。** ① 判官本身会错：RAGAS Table 1 在其 WikiEval 数据上与人工成对比较的一致率为 faithfulness 0.95、answer relevance 0.78、context relevance 0.70，后两项明显更不可靠。② 拆陈述的粒度会改变分母：同一答案拆成 2 条或 6 条，分数不同。③ 换判官模型、换 prompt、换嵌入模型，分数不可直接比较；开源库的实现可能已和论文不同，用前要锁版本。④ 有参考答案时，EM/F1 和人工评测仍是主指标，RAGAS 适合做无标注时的回归监控。

**怎样组织一次评测。** 检索层：Recall@k、MRR、nDCG（有分级标注时）。生成层：答案准确率（EM 或人工）、faithfulness、拒答正确率、引用正确率。再按第 1 节的顺序做归因：答错样本里，正确块不在 top-$K$、在 top-$K$ 不在 top-$k$、在 top-$k$ 仍答错，三类各占多少，决定该改检索、重排还是生成。

## 9. 面试常问

**RAG-Sequence 和 RAG-Token 有什么区别？** 都把文档当隐变量用 top-$k$ 近似边缘化。前者整句条件在同一文档上、句级求和，后者每个 token 位置先对文档求和再连乘，所以能在一句话里组合多篇文档。前者解码要对每篇文档各跑 beam，后者能直接用标准 beam search。原论文只训练查询编码器和生成器，文档编码器与索引固定。

**稠密检索为什么还要 BM25？** 稠密模型擅长语义改写，但对型号、编号、罕见实体不稳；DPR 自己在 SQuAD 上就输给了 BM25（Table 2）。混合检索可以线性融合（需要调 $\lambda$，对分数尺度敏感），也可以用只看名次的 RRF。

**切块大小和重叠怎么定？** 没有通用值，用评测定：固定检索器，扫 $(c,o)$，比较 Recall@k、端到端准确率和索引大小，且要对齐喂给生成器的 token 预算。重叠的代价可以手算（第 3 节），DPR 论文发现重叠切分没有收益。

**检索结果不可靠时怎么办？** 加一个相关性判断再决定动作：CRAG 用轻量评估器分 Correct/Incorrect/Ambiguous，错了就改写 query 去网页搜索；Self-RAG 让模型自己输出 Retrieve/IsRel/IsSup 并据此挑选片段；再往前一步是用 RL 让模型学会何时查、查什么（见 [Agent RL 预备](/notes/recsys-agent-rl/) 和 [Agent RL 进阶](/notes/agent-rl-advanced/)）。

**faithfulness 高就说明答案对吗？** 不是。它只衡量答案陈述能否由检索到的上下文推出；上下文本身错了，或答案漏了关键内容，faithfulness 照样可以是 1。所以要和答案准确率、answer relevance、检索召回一起看。

## 闭卷验收

不看资料写出 RAG-Sequence 与 RAG-Token 的公式，说清求和与连乘的顺序，并用两篇文档、两个 token 的例子各算一次似然；说出 RAG 原论文冻结了哪个模块、语料怎样切、Thorough 与 Fast Decoding 的区别；给定 $L$、$c$、$o$ 算块数、每块起止与冗余倍数，并估算 2100 万个 768 维向量的 fp32 存储；写出 DPR 的相似度、损失和 in-batch 负样本矩阵形状，说出它的混合公式与 $\lambda$；写出 HyDE 的查询向量公式；列出 Self-RAG 四类反思 token 的输入与取值，写出片段打分式并手算一次权重调整的影响；讲清 CRAG 三种动作的触发条件与各自做什么；写出 ColPali 的晚交互打分与损失，说出 ViDoRe 的任务构成和主指标，从 1030 个 128 维 fp16 向量算出 257.5 KB；对一组排序结果手算 Recall@k、Hit@k、MRR；写出 RAGAS 三个指标的定义并手算一次，说出它们与人工一致率的差别和使用时的坑。

**参考。** [Lewis et al., Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)；[Karpukhin et al., Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906)；[Gao et al., Precise Zero-Shot Dense Retrieval without Relevance Labels (HyDE)](https://arxiv.org/abs/2212.10496)（[ACL Anthology](https://aclanthology.org/2023.acl-long.99/)）；[Asai et al., Self-RAG](https://arxiv.org/abs/2310.11511)（[官方仓库](https://github.com/AkariAsai/self-rag)）；[Yan et al., Corrective Retrieval Augmented Generation](https://arxiv.org/abs/2401.15884)；[Faysse et al., ColPali](https://arxiv.org/abs/2407.01449)（[ViDoRe 资源](https://huggingface.co/vidore)）；[Es et al., RAGAS](https://arxiv.org/abs/2309.15217)（[ACL Anthology](https://aclanthology.org/2024.eacl-demo.16/)）；[Edge et al., From Local to Global: A Graph RAG Approach](https://arxiv.org/abs/2404.16130)；[Khattab & Zaharia, ColBERT](https://arxiv.org/abs/2004.12832)；[Cormack et al., Reciprocal Rank Fusion, SIGIR 2009](https://dl.acm.org/doi/10.1145/1571941.1572114)。
