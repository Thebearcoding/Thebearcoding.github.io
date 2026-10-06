---
title: 推荐与 LLM4Rec 全覆盖教程：从一条请求到生成式推荐
date: '2026-10-05'
tags: [推荐系统, LLM4Rec, 生成式推荐, 学习路线, 面试]
summary: 给多模态背景、没系统学过推荐的人：二十篇笔记串起链路、评估、长序列、多任务与多场景、图推荐、搜索、语义 ID、LLM4Rec、多模态 I2I、广告、系统设计与 Agent RL，附覆盖清单和逐日看板。
draft: false
---

这组教程写给一种具体的读者：做过多模态或检索研究，熟悉对比学习和 CLIP，但没有系统学过推荐系统，现在要在一两个月内准备推荐、搜索或多模态表征方向的算法实习。目标不是背下一百个模型名，而是能回答三类问题：**这个模块的输入和输出是什么，它在省哪一部分计算或纠正哪一种偏差，怎样用实验证明它真的有用。**

配套的逐日计划在 [备战看板](/plan/)：每天的时间块都链接到下面的具体章节。第一次读，先看本页的地图和顺序；之后按看板每天读一节，读完做文末的“闭卷验收”。

## 1. 资料从哪里来

检索日期：**2026 年 10 月 5 日**。这组教程的骨架来自三类材料，作用不同：

| 材料 | 用途 | 取舍 |
|---|---|---|
| [王树森《工业界的推荐系统》](https://github.com/wangshusen/RecommenderSystem) | 经典链路：召回、排序、交叉、行为序列、重排、冷启动、涨指标 | 本站已有六篇笔记覆盖，见下表 0–5 号 |
| [算法小小怪下士的 LLM4Rec 学习笔记仓库](https://github.com/cst20/LLM4Rec_xiaoxiaoguai) | 新范式的主题索引：语义 ID 编码对比、text+SID、SID 作特征、长序列、可扩展 token 排序、AIGC、I2I、广告与策略 | 仓库未声明许可证，这里只参考它的选题和阅读清单，正文全部依据原论文重写 |
| 原论文与开源代码 | 公式、机制与实验结论的依据 | 每篇文末列出链接；数值算例均为教学构造 |

两个原则贯穿全系列。第一，**论文报告的线上收益不能直接当成你项目的预期**：工业论文的增益依赖它的流量、特征和基线。第二，**新范式要和强基线比较**：生成式推荐和 SASRec 比，就要用同样的负样本和评估协议，否则比较的是训练配方而不是范式。

## 2. 一张地图

<figure>
<a href="/notes-assets/recsys-llm4rec/map.svg" target="_blank" rel="noopener" aria-label="查看知识地图原图"><img src="/notes-assets/recsys-llm4rec/map.svg" alt="知识地图：左侧是经典链路的召回、排序、重排，中间是序列建模和可扩展排序，右侧是语义 ID 和 LLM4Rec，底部是评估、多模态表征与广告业务" width="1100" height="560" loading="lazy" style="height:auto;cursor:zoom-in" /></a>
</figure>

读图的方法：从左往右是“同一个问题越来越大”。经典链路解决的是全库太大、只能分段计算；序列建模解决的是用户历史越来越长；可扩展排序和生成式推荐解决的是模型想变大、但 embedding 表和特征交叉不好扩展。底部三块贯穿所有层：评估决定你信不信一个改动，多模态表征决定新物品和长尾物品有没有信号，广告业务决定分数最后怎样变成钱。

## 3. 二十篇怎样连起来读

| 顺序 | 笔记 | 读完应当能回答 | 看板对应 |
|---|---|---|---|
| 0 | [王树森系列入口：系统与评价](/notes/wangshusen-recommender-guide/) | 为什么点击率上涨仍可能是坏改动？ | W1 |
| 1 | [召回：协同过滤、双塔与向量索引](/notes/wangshusen-recommender-retrieval/) | batch 内负样本为什么要做 logQ 纠偏？ | W1 |
| 2 | [排序：多目标、交叉与行为序列](/notes/wangshusen-recommender-ranking/) | DIN 的用户表示为什么随候选变化？ | W1 |
| 3 | [重排：MMR 与 DPP](/notes/wangshusen-recommender-reranking/) | 两个高分物品放在一起为什么可能更差？ | W1 |
| 4 | [冷启动与涨指标](/notes/wangshusen-recommender-coldstart/) | 作者侧 A/B 为什么可能制造假增益？ | W1 |
| 5 | [复习题：十六个算例](/notes/wangshusen-recommender-workbook/) | 能否手算并指出前提错误？ | W1 周末 |
| 6 | [评估与偏差：指标的分母、切分与纠偏](/notes/recsys-eval-bias/) | 采样 100 个负样本评测，为什么可能把模型排序弄反？ | W1–W2 |
| 7 | [序列推荐与长序列建模](/notes/recsys-sequence-long/) | SIM、ETA、SDIM、TWIN 分别把“检索”换成了什么？ | W2–W3 |
| 8 | [排序模型的 scaling](/notes/recsys-scaling-ranking/) | HSTU 和 RankMixer 都想扩展模型，扩展的方向有何不同？ | W3 |
| 9 | [语义 ID：把物品写成 token](/notes/recsys-semantic-id/) | RQ-VAE 的残差量化、码本坍塌、去重位分别是什么？ | W2 |
| 10 | [LLM4Rec 的五种用法](/notes/recsys-llm4rec-paradigms/) | P5、TALLRec、LC-Rec、OneRec、MiniOneRec 在哪一层用了 LLM？ | W2–W4 |
| 11 | [多模态推荐与 I2I](/notes/recsys-multimodal-i2i/) | 冻结的 CLIP 向量为什么不能直接当推荐特征用到最好？ | W2–W4 |
| 12 | [广告机制与业务直觉](/notes/recsys-ads-business/) | GSP 为什么不鼓励如实出价？oCPM 依赖哪两个预估？ | W4 |
| 13 | [面试题库：九十问](/notes/recsys-interview-bank/) | 能否在两分钟内讲清一个机制并给出验证方案？ | W4 起反复做 |
| 14 | [CLIP 与多模态检索](/notes/vision-video-algorithms/) | 难负例和假负例在图文匹配与推荐里有何异同？ | 随时回看 |
| 15 | [多任务与多场景：PLE、STAR](/notes/recsys-multitask-scenario/) | 怎样从实验里看出跷跷板？CGC 比 MMoE 少了哪些连接？ | W3 |
| 16 | [图推荐：从 GCN 到 LightGCN](/notes/recsys-graph/) | LightGCN 为什么删掉特征变换和非线性反而更好？ | W3 |
| 17 | [搜索算法入门](/notes/recsys-search/) | 搜索和推荐在输入、目标、评估上差在哪？BM25 的 $k_1$、$b$ 各管什么？ | W4 |
| 18 | [推荐系统设计题](/notes/recsys-system-design/) | 能否从 DAU 推到峰值 QPS、精排算力和延迟预算？ | W4 起反复练 |
| 19 | [Agent RL 预备：Search-R1 与 verl](/notes/recsys-agent-rl/) | 为什么检索返回的 token 不计入损失？检索器怎样作为环境接入？ | 面试期 |

第 0–5 篇是推荐的地基，第 6 篇是判断力，第 7–12 篇是你这一届面试最常被追问的新内容。第 13 篇不是最后才做，从 W1 周末开始每周抽一节。第 15–16 篇补排序与召回的进阶结构，第 17 篇给投搜索岗的人，第 18 篇是开放题的答题框架，第 19 篇是下一个项目方向的预备。

## 4. 覆盖清单

把下面每一行讲给别人听，能讲出输入、输出、一个公式或算例、一个失败场景，才算掉。分三档：**必会**是任何推荐岗都会问的；**常问**是生成式推荐、多模态岗大概率追问的；**加分**是能拉开差距的。

**必会：链路与评估**

- [ ] 召回、粗排、精排、重排各自的输入规模和计算预算 → [入口](/notes/wangshusen-recommender-guide/)
- [ ] CTR、点击后转化率、曝光后转化率的分母 → [入口](/notes/wangshusen-recommender-guide/)
- [ ] AUC 的排序定义与手算；GAUC 为什么按用户加权 → [评估](/notes/recsys-eval-bias/)
- [ ] Recall@K、NDCG@K、MRR 的公式；多个正样本时的 IDCG → [评估](/notes/recsys-eval-bias/)
- [ ] full-ranking 与采样评测；留一法与全局时间切分；数据泄漏的三种来源 → [评估](/notes/recsys-eval-bias/)
- [ ] A/B 测试的分桶、holdout、指标冲突 → [入口](/notes/wangshusen-recommender-guide/)

**必会：召回、排序、重排**

- [ ] ItemCF、Swing、UserCF 的相似度公式 → [召回](/notes/wangshusen-recommender-retrieval/)
- [ ] 双塔的三种训练目标；batch 内负样本与 logQ 纠偏 → [召回](/notes/wangshusen-recommender-retrieval/)
- [ ] ANN 索引（HNSW、IVF-PQ）与模型版本必须成套更新 → [召回](/notes/wangshusen-recommender-retrieval/)
- [ ] 多目标、MMoE、融分公式 → [排序](/notes/wangshusen-recommender-ranking/)
- [ ] FM、DCN V2、SENet 的交叉方式与 shape → [排序](/notes/wangshusen-recommender-ranking/)
- [ ] DIN 与 SIM → [排序](/notes/wangshusen-recommender-ranking/)
- [ ] MMR 与 DPP 的手算 → [重排](/notes/wangshusen-recommender-reranking/)
- [ ] 跷跷板与负迁移；CGC、PLE 的门控与专家划分；多任务 loss 的样本空间与加权 → [多任务](/notes/recsys-multitask-scenario/)
- [ ] 多场景建模：STAR 的星形 FCN 与分区归一化 → [多任务](/notes/recsys-multitask-scenario/)
- [ ] 系统设计题：约束澄清、指标体系、容量与延迟估算 → [设计题](/notes/recsys-system-design/)
- [ ] 冷启动的召回通道与流量扶持；实验干扰 → [冷启动](/notes/wangshusen-recommender-coldstart/)

**必会：偏差**

- [ ] 曝光偏差、位置偏差、流行度偏差分别从哪来 → [评估](/notes/recsys-eval-bias/)
- [ ] CVR 的样本选择偏差；ESMM 与 IPW 两种做法 → [评估](/notes/recsys-eval-bias/)
- [ ] 负采样后的概率校准公式 → [评估](/notes/recsys-eval-bias/)

**常问：搜索与图**

- [ ] 搜索与推荐的区别；query 理解各步的输入输出 → [搜索](/notes/recsys-search/)
- [ ] BM25 公式与手算；双塔、cross-encoder、ColBERT 的形状与成本 → [搜索](/notes/recsys-search/)
- [ ] 相关性与个性化的融合；分级 NDCG 与线上指标的陷阱 → [搜索](/notes/recsys-search/)
- [ ] GCN 的对称归一化；NGCF 的消息构造；LightGCN 的传播与层组合 → [图推荐](/notes/recsys-graph/)
- [ ] 过平滑；PinSage 的随机游走邻居与难负例 → [图推荐](/notes/recsys-graph/)

**常问：序列与长序列**

- [ ] SASRec 的因果 mask 与训练目标；BERT4Rec 的 Cloze 目标 → [序列](/notes/recsys-sequence-long/)
- [ ] SIM 的 GSU/ESU；ETA 用 SimHash 检索；SDIM 用哈希采样替代检索 → [序列](/notes/recsys-sequence-long/)
- [ ] TWIN 为什么让 GSU 与 ESU 一致；TWIN V2 的聚类压缩 → [序列](/notes/recsys-sequence-long/)
- [ ] LONGER 等 Transformer 化长序列的降本手段 → [序列](/notes/recsys-sequence-long/)

**常问：可扩展排序**

- [ ] DLRM 为什么难以 scaling；MFU 是什么 → [scaling](/notes/recsys-scaling-ranking/)
- [ ] HSTU 把推荐改写成什么问题；它的注意力为什么不用 softmax → [scaling](/notes/recsys-scaling-ranking/)
- [ ] RankMixer 的特征 token 化与 token mixing；OneTrans 统一序列与非序列特征 → [scaling](/notes/recsys-scaling-ranking/)

**常问：语义 ID 与生成式推荐**

- [ ] VQ-VAE 的三项损失与直通估计（STE） → [语义 ID](/notes/recsys-semantic-id/)
- [ ] RQ-VAE、RQ-KMeans、PQ/OPQ、FSQ 的区别 → [语义 ID](/notes/recsys-semantic-id/)
- [ ] 码本利用率、冲突率、熵；码本坍塌的成因与对策 → [语义 ID](/notes/recsys-semantic-id/)
- [ ] TIGER 的训练与 beam search；约束解码为什么需要前缀树 → [语义 ID](/notes/recsys-semantic-id/)
- [ ] 语义 ID 作为排序特征（不生成，只替换或补充 item ID） → [语义 ID](/notes/recsys-semantic-id/)

**常问：LLM4Rec**

- [ ] LLM 当编码器、当推荐器、当生成式推荐器、当老师、当数据生成器 → [LLM4Rec](/notes/recsys-llm4rec-paradigms/)
- [ ] P5、TALLRec、LLaRA 各自怎样把物品放进 prompt → [LLM4Rec](/notes/recsys-llm4rec-paradigms/)
- [ ] LC-Rec、PLUM：语义 ID 与文本怎样对齐 → [LLM4Rec](/notes/recsys-llm4rec-paradigms/)
- [ ] OneRec 的端到端生成与偏好对齐；MiniOneRec 的 SFT + GRPO → [LLM4Rec](/notes/recsys-llm4rec-paradigms/)
- [ ] “生成式推荐是不是伪范式”的正反两面 → [LLM4Rec](/notes/recsys-llm4rec-paradigms/)

**常问：多模态**

- [ ] CLIP/SigLIP 目标、温度、难负例与假负例 → [CLIP](/notes/vision-video-algorithms/)
- [ ] 冻结多模态向量的两个问题与 QARM 的两步修正 → [多模态](/notes/recsys-multimodal-i2i/)
- [ ] NoteLLM-2 的视觉忽视问题；VLM2Vec 取最后一个 token 做表征 → [多模态](/notes/recsys-multimodal-i2i/)
- [ ] I2I：item2vec、EGES、PDN 与内容 I2I → [多模态](/notes/recsys-multimodal-i2i/)

**加分：业务与工程**

- [ ] eCPM 与 CPM、CPC、CPA、oCPM 的关系 → [广告](/notes/recsys-ads-business/)
- [ ] GSP、VCG、保留价；为什么出价要校准 → [广告](/notes/recsys-ads-business/)
- [ ] 用 CTR 训练文案生成模型时的奖励设计与 reward hacking → [LLM4Rec](/notes/recsys-llm4rec-paradigms/)
- [ ] 训练显存估算与 LoRA、DDP、all-gather 负样本 → [训练显存](/notes/training-memory-debugging/)
- [ ] Search-R1 的多轮 rollout、检索 token 掩码与 EM 奖励；verl 的角色划分与多轮工具配置 → [Agent RL](/notes/recsys-agent-rl/)

## 5. 你的背景怎样接上推荐

做过多模态或跨模态检索的人，其实已经接触过推荐里的好几个核心问题，只是名字不同。面试时把这张对照表讲出来，比说“我也了解推荐”有说服力得多。

| 你熟悉的概念 | 推荐里的对应 | 要补的差异 |
|---|---|---|
| 对比学习里的假负例 | 隐式反馈里“没交互 ≠ 不喜欢”；batch 内负样本撞到用户其实喜欢的物品 | 推荐的负样本来自曝光与采样分布，需要 logQ 纠偏 |
| 难负例挖掘 | 召回的难负例、粗排蒸馏中的难样本 | 太难的负例常常是假负例，要混合简单负例 |
| 向量检索与 ANN 索引 | 双塔召回、语义 ID、IVF-PQ | 索引要和模型版本成套更新；语义 ID 还要能逐 token 生成 |
| CLIP 图文对齐 | 内容侧 item 表征、冷启动、多模态 I2I | 内容相似不等于行为相似，需要用行为对齐 |
| mAP 与 PR 曲线 | Recall@K、NDCG@K、AUC、GAUC | 推荐评估的分母和切分方式更容易出错 |

## 6. 三个常见误区

**把论文名当知识点。** 面试官问 TWIN，真正想听的是“GSU 与 ESU 不一致会损失什么，怎样在预算内让 GSU 也用目标注意力”。每读一篇，用一句话写出它在省什么或纠正什么。

**把离线提升当结论。** 采样评测、留一法和时间泄漏都可能让离线数字好看。做项目时把评估协议写进 README 第一段。

**把“大模型”当成目的。** 很多工业 LLM4Rec 工作最后上线的是小模型、语义 ID 特征或蒸馏后的打分器。能讲清“为什么不直接用 LLM 打分”比会背 prompt 模板更重要。

## 闭卷验收

不看任何资料，画出一条请求从召回到曝光反馈的路径，在每个箭头上写一个可能的偏差；然后选一个新范式（语义 ID、HSTU 或多模态 I2I），说出它替换了图上的哪一块、新增了什么成本、用什么实验证明有效。答不顺的地方，回到上面对应的那一篇。

**来源与说明。** 经典部分沿用本站 [王树森系列](/notes/wangshusen-recommender-guide/)（CC BY-SA 4.0 改编）；新范式部分的选题参考 [算法小小怪下士的笔记仓库](https://github.com/cst20/LLM4Rec_xiaoxiaoguai) 与其阅读清单，内容依据各篇文末列出的原论文重写。图为本站绘制。
