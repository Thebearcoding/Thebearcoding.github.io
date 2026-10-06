---
title: 表征项目面试题库：从 VLM embedding 到工业落地
date: '2026-10-06'
tags: [多模态算法, 表征学习, 复习问答, 面试, 推荐系统]
summary: 围绕“用 VLM 训 embedding”的项目，按面试官深挖的顺序整理四十道题：动机、方法、数据与评测、消融、工业落地、排障和穿插的八股，每题附参考解答和出处，重点标出最容易被抓住的漏洞，比如官方权重见过 COCO、拿零样本 CLIP 当唯一基线。
draft: false
---

项目面试一般 20–30 分钟，路线基本固定：为什么做 → 怎么做 → 怎样证明 → 哪里不行 → 能不能用。面试官最想听到“我做了 X，因为 Y，我用 Z 验证了它”，最怕听到“官方代码就是这么写的”。每题先口述 2 分钟，再展开参考解答。这些题是本站按项目内容设计的复习题，不代表某家公司的题库。

阅读入口：[多模态学习手册](/notes/multimodal-interview-guide/)。方法细节见 [多模态 embedding](/notes/multimodal-embedding-retrieval/)，工业做法见 [工业界多模态表征](/notes/multimodal-industrial-representation/)，推荐侧的题目见 [推荐面试题库](/notes/recsys-interview-bank/)。

## A. 动机与定位

### A1. 用一分钟介绍你的项目。

<details><summary>参考解答</summary>

四句话：问题、方法、结果、发现。例如：“CLIP 双塔读不了指令、不能图文混合输入，也和推荐要的‘行为相似’对不上。我用对比学习把 Qwen2-VL-2B 训成 embedding 模型，在 MicroLens 上用用户共现对训练，和 CLIP、官方 VLM2Vec 权重比较 I2I 召回和下游推荐。主要发现是行为对训练对冷启动视频提升最大，但会损失一部分内容检索能力，我画出了这条权衡曲线。”说不出“发现”，就只是在复述做了什么。

</details>

### A2. 为什么不直接用 CLIP？VLM 做 embedding 好在哪，代价是什么？

<details><summary>参考解答</summary>

好处：能读指令，同一个模型可以服务多种检索任务；图文可以混合输入；语言理解更强（长文本、否定、计数）。代价：参数量大一个数量级以上，图像会被切成很多视觉 token，编码慢、部署贵。加分的说法是“在纯图文检索上 CLIP 的性价比可能更高，我的项目就是要量化这个差距值不值”。VLM2Vec 论文 Table 4 的数字可以用来说明指令的作用：加任务指令让 VLM2Vec 提升 49.4%，却让 CLIP 下降 29.4%。[多模态 embedding](/notes/multimodal-embedding-retrieval/)

</details>

### A3. 你的贡献是什么？这不就是跑了一遍官方代码吗？

<details><summary>参考解答</summary>

这是最重要的压力题。能说成贡献的东西：①训练信号换成了用户行为，复现了工业界“行为对齐表征”的链路；②四组消融，尤其是假负例屏蔽和流行度纠偏是自己实现的；③模态诊断和错误分析，回答了“模型有没有真的用图”；④三层评测：表征指标、下游推荐、内容保持；⑤可复现的工程：多种子、标准差、单测、开源。只说“复现了 VLM2Vec”是最差的回答。

</details>

### A4. VLM2Vec 和 CLIP、SigLIP、E5-V 有什么区别？

<details><summary>参考解答</summary>

CLIP：双塔，batch 内 softmax 对比，图文各自编码。SigLIP：把 softmax 换成逐对的 sigmoid，不需要全局归一化，大 batch 更省。E5-V：只用文本对训练 VLM，靠提示把图像映射到文本空间。VLM2Vec：用 VLM 当编码器，在多任务数据上做带指令的对比训练，取最后一个 token。[多模态 embedding](/notes/multimodal-embedding-retrieval/)

</details>

### A5. 学术图文检索和工业表征差在哪？

<details><summary>参考解答</summary>

四点：正样本（人工描述对 vs 用户行为：共同观看、搜后点击、看后购买）；用途（直接检索 vs 喂给召回、排序、冷启动等下游）；评测（Recall@K vs 下游 AUC 和线上 LT）；约束（单机 vs 十亿级库、毫秒延迟）。抖音 SAIL-Embedding 的训练数据里，用户共同消费构造的物品对就有 29 亿对。[工业界多模态表征](/notes/multimodal-industrial-representation/)

</details>

## B. 方法细节

### B1. 向量是怎么取的？为什么取最后一个 token？

<details><summary>参考解答</summary>

因果注意力下只有最后一个位置看到了全部输入。官方代码的 `_pooling` 用 `attention_mask` 找每行最后一个值为 1 的位置，左右填充都对；自己写 `hidden_states[:, -1]` 在右填充时会取到 pad 的向量。mean pooling 会混进大量只看到前缀的位置。要用你自己的池化消融结果来回答，而不是只讲道理。

</details>

### B2. 写出你的损失函数。

<details><summary>参考解答</summary>

$$
\mathcal L=-\frac1B\sum_i\log\frac{\exp(s_{ii}/\tau)}{\sum_j\exp(s_{ij}/\tau)},
$$

$s$ 是 L2 归一化后的余弦相似度，标签在对角线。追问单向还是双向：VLM2Vec 只算查询到候选的方向，因为两端不对称；CLIP 两个方向取平均。I2I 场景里两端都是物品，可以对称地算。

</details>

### B3. 温度 0.02 是什么意思？调大调小会怎样？

<details><summary>参考解答</summary>

对 logit 的梯度是 $(p-y)/\tau$。温度小，softmax 尖，梯度集中在最难的负例上，对假负例也更敏感；温度大，所有负例被均匀地推，区分度下降。可学习温度要设上下界，CLIP 把 logit 的缩放截断在 100 以内。

</details>

### B4. 训练刚开始 loss 应该是多少？

<details><summary>参考解答</summary>

从随机或未对齐的状态开始，每行 softmax 接近均匀分布，loss 约为 $\ln B$：$B=256$ 时是 5.55，$B=1024$ 时是 6.93。从已经训好的 VLM2Vec 权重继续训，起点会远低于这个值。第一步离理论值很远，先查归一化、温度和标签对齐。

</details>

### B5. 为什么 batch 要大？为什么不能用梯度累积代替？

<details><summary>参考解答</summary>

负例数是 batch 减 1。InfoNCE 每个样本的损失依赖整个 batch，切块累积会让负例变少，目标函数本身就变了。GradCache 先无梯度前向拿到全部 embedding，在 embedding 上算 loss 和梯度，再分块重算前向并反传，得到精确梯度，代价是多一次前向。但 batch 不是越大越好：抖音 DME 的实验显示 batch 从 128 到 8192 一直有收益，再大就收益递减，因为假负例变多、难负例的梯度被稀释。

</details>

### B6. 4 卡怎样扩大负例？梯度怎么回传？

<details><summary>参考解答</summary>

各卡把查询和候选向量 `all_gather` 到一起，拼成全局 batch；`all_gather` 本身不回传梯度，所以官方代码把本卡那一段替换回带梯度的原张量，每个查询和全局候选比较，梯度只流回本卡样本；loss 再乘以卡数，抵消 DDP 对梯度的平均。单测：断言相似度矩阵的形状是（全局 batch × 全局 batch）。只答“all_gather 之后直接算”，被问“all_gather 有梯度吗”就会卡住。

</details>

### B7. LoRA 加在哪？为什么不全参？视觉编码器冻不冻？

<details><summary>参考解答</summary>

官方 2B 配置是 LoRA r=16，加在注意力和 MLP 的线性层。理由是显存和训练时间，VLM2Vec 论文里 LoRA 版本比全参版本略好。视觉编码器是否冻结，要在代码里确认过再答；NoteLLM-2 的做法是冻结视觉编码器、训练连接器和 LLM，以便用更大的 batch。

</details>

### B8. 一个视觉 token 对应多大的图像区域？封面图编码要多少 token？

<details><summary>参考解答</summary>

Qwen2-VL 系列 patch 是 14×14，相邻 2×2 合并，一个视觉 token 对应 28×28 像素。一张缩放到 448×448 的封面是 $16\times16=256$ 个 token。抖音 DME 把每张图限制在 1280 个 token，实验显示从 256 增加到 1280 各项指标都涨。token 预算直接决定吞吐和能用的 batch。[视频理解](/notes/multimodal-video-understanding/)

</details>

## C. 数据与评测

### C1. R@K 怎么算？图搜文和文搜图有什么不同？

<details><summary>参考解答</summary>

图搜文：每张图有 5 条正确描述，任意一条进 top-K 就算命中，分母是图片数。文搜图：每条描述只对应 1 张图，分母是描述数。I2I 召回：用 trigger 物品检索，目标物品进 top-K 算命中，分母是测试对数。说不清分母是最常见的失分。

</details>

### C2. 有没有数据泄漏？（必问）

<details><summary>参考解答</summary>

COCO 方向有个大坑：VLM2Vec 的训练集 MMEB 包含 MSCOCO_i2t（11.3 万）和 MSCOCO_t2i（10 万），官方权重在 COCO 上的结果不能叫零样本，还要检查它的训练图片和 Karpathy 测试集有没有重叠；Flickr30K 不在 MMEB 训练集里，是更干净的分布外测试。行为数据方向：共现对只能用训练期交互构造，否则测试期的共现会泄漏进训练。

</details>

### C3. 和 CLIP 的比较公平吗？

<details><summary>参考解答</summary>

零样本 CLIP 对比你微调过的模型不公平。公平的对照是：CLIP 在同样数据上微调；或者只在分布外数据上比零样本。VLM2Vec 论文同时报告了零样本 CLIP（37.8）和在 MMEB 上全参微调的 CLIP（45.4）。只拿零样本 CLIP 当基线，然后说“大幅超过”，是容易被抓住的漏洞。

</details>

### C4. 提升是不是噪声？

<details><summary>参考解答</summary>

至少 3 个种子，报告均值±标准差，差距小于两三倍标准差不说显著；同一测试集上可以做配对检验，方法见 [Agent 评测](/notes/agent-eval/) 第 6 节。评测脚本要用手算的小例子做单测，基线要和公开数字对齐。

</details>

### C5. 只看 Recall@K 够吗？

<details><summary>参考解答</summary>

不够。抖音 SAIL 的技术报告明确写道：embedding 自身指标可能很强，却对下游用户互动贡献很小。工业上的离线评测分三层：表征指标；接入下游模型后的 AUC 或 HR/NDCG；按流行度分桶、单独看冷启动。你的项目至少要有前两层。

</details>

## D. 消融

### D1. 难负例怎么挖？为什么跳过第 1 名？

<details><summary>参考解答</summary>

用一个已有模型检索候选库，取排第 2–20 名里的随机一个；第 1 名很可能就是正例或近似正例。用正在训练的模型自己挖会自我强化，离线挖要定期刷新。SAIL 用 F1 选每个数据集的相似度阈值，再在阈值以下取最难的样本当难负例。

</details>

### D2. 什么是假负例？你怎么处理的？

<details><summary>参考解答</summary>

batch 里的“负例”其实也和查询匹配。描述数据里，来自同一张图的另一条描述、语义几乎相同的不同图；行为数据里，同一用户连着看的视频、同一搜索会话里被点的结果。处理方式：不处理 / 硬阈值屏蔽 / 软降权，判断用冻结的老模型以免自我强化。工业例子：抖音 LEMUR 在 batch 内屏蔽同一搜索会话的样本，去掉这个屏蔽 QAUC 从 +0.81% 降到 +0.71%。追问阈值怎么定、会不会误伤真正的难负例，要能讲阈值扫描和人工抽查。

</details>

### D3. 行为对训练为什么要做流行度纠偏？

<details><summary>参考解答</summary>

行为对里热门物品出现得多，batch 内负例被热门物品占满，模型会过度推开热门物品。sampled softmax 的纠正是 logit 减去被采样的对数概率：$s_j/\tau-\log q_j$。热门 $q=0.01$、长尾 $q=0.0001$ 时修正量相差 4.61 个 logit 单位，温度 0.05 下相当于余弦差 0.23。[推荐评估与偏差](/notes/recsys-eval-bias/) 第 9 节

</details>

### D4. 你怎么知道模型真的用了图？

<details><summary>参考解答</summary>

分别只给图像、只给文本、图文都给，测同一个指标。小红书 NoteLLM-2 的实验里，一个微调后的模型只给文本 R@100 是 71.49，只给图像只有 0.60，几乎完全没用图；VLM2Rec 也发现对比 SFT 会放大模态失衡。修正方法有模态分开压缩（mICL）、视觉后期融合、模态 dropout。要在短标题子集上单独报告，那是图像最有用的地方。

</details>

### D5. 向量要压缩，为什么选语义 ID 而不是二值化或降维？

<details><summary>参考解答</summary>

三种方式解决的问题不同。二值化（SimHash）省存储，汉明距离近似夹角，碰撞概率 $\theta/\pi$；Matryoshka 降维可以按成本选维度；语义 ID 把向量变成离散码，下游可以给每个码学 embedding，随推荐模型更新，还方便做打散之类的规则。SAIL 的线上实验发现语义 ID 的收益比稠密向量更大。[语义 ID](/notes/recsys-semantic-id/)

</details>

## E. 工业落地

### E1. 为什么不直接把 CLIP 向量喂给推荐模型？

<details><summary>参考解答</summary>

快手 QARM 总结的两个问题：表征不匹配，图文相似不等于行为相似；表征学不动，缓存的固定向量接不到推荐梯度。解法：用召回模型认定的相似物品对或 Swing 对做对齐，再量化成 VQ/RQ 码当可训练特征。上线后广告收入在冷启动物品组提升最高约 9.7%。

</details>

### E2. 内容向量能替代物品 ID 吗？

<details><summary>参考解答</summary>

一般不能。YouTube 的实验显示直接替换会掉效果，因为丢了 ID 的记忆能力，语义 ID 能在新视频和长尾上泛化又不牺牲整体；冻结 LVLM 的系统研究也发现与 ID 融合严格好于替换。内容向量的主要价值在冷启动和长尾。

</details>

### E3. 两阶段（先预训练再冻结使用）和端到端各有什么问题？

<details><summary>参考解答</summary>

两阶段：目标不一致、推荐模型优先学 ID 导致内容向量欠拟合、更新频率不一致、长序列向量传输贵（LEMUR 的总结）。端到端：算力太大，LEMUR 的多模态塔只有 2–4 层、128 维，用几十亿参数的 VLM 做端到端目前不现实。所以 QARM 走“对齐 + 量化”的折中路线。

</details>

### E4. 内容和协同信号冲突怎么办？

<details><summary>参考解答</summary>

只用行为训，向量会丢内容；只用内容，和推荐目标对不上。OneRec 的 tokenizer 同时用物品对比损失和描述生成损失；SAIL 的协同增强让 GID-i2i 涨了、Copair-i2i 跌了，作者认为可以接受。项目里可以用加权系数扫一条权衡曲线，看下游最好的点在哪。

</details>

### E5. 新物品冷启动，多模态表征怎么帮？

<details><summary>参考解答</summary>

新物品没有交互，但有内容。I2I 召回可以直接用内容向量（NoteLLM 上线后新笔记当日评论 +3.58%，NoteLLM-2 新笔记前 24 小时互动 +8.08%）；排序模型可以用对齐到 ID 空间的代理向量替代缺失的 ID 向量（小红书 IDProxy）。评估时要单独报告冷启动桶。

</details>

### E6. 线上 LT 只涨 0.1%，算多吗？

<details><summary>参考解答</summary>

算多。快手在 OneRec 报告中写明，停留时长 0.1%、LT7 0.01% 的提升在快手就已经统计显著。抖音 SAIL 在精选场景 LT7 +0.5%、DME 在抖音搜索 LT +0.1%，都是很大的收益。把 0.1% 说成“提升不大”会显得不懂业务。

</details>

### E7. 你的模型上线，瓶颈在哪？

<details><summary>参考解答</summary>

延迟：VLM 编码比 CLIP 慢得多，候选端离线编码，查询端可以蒸馏到小模型。存储：1536 维 fp16 每个 3 KB，十亿物品 3 TB，要降维或量化。更新：换模型要重编码全库、重建索引。视觉 token 预算：直接决定吞吐。DME 把潜在推理 token 放在同一次前向里，每次查询只多花不到 1 毫秒，这是“能上线”的设计取向。

</details>

### E8. 内容理解岗还做哪些事，你的项目和它们有什么关系？

<details><summary>参考解答</summary>

推荐表征之外，还有审核、打标签、质量判断。快手 KuaiMod 用 VLM 加思维链做审核，上线后用户举报率降低 20%；小红书 UniNote 用 I2I 检索把新笔记和风险样本库匹配。它们都依赖同一个能力：把内容编码成可比较、可检索的表示，并验证模型真的看了画面。你的模态诊断和假负例处理可以直接迁移过去。

</details>

## F. 失败与排障

### F1. 训练中遇到过什么问题？

<details><summary>参考解答</summary>

必须准备两三个真实例子，每个按“现象 → 假设 → 验证 → 结论”讲，例如 OOM、loss 不降、某张卡特别慢、评测分数异常。从项目第一天起就在 `notes/` 里记录，到面试时才有素材。

</details>

### F2. loss 在降，检索指标不涨，可能是什么原因？

<details><summary>参考解答</summary>

评测脚本错（分母、候选库、归一化）；池化位置取错，训练和评测不一致；训练的正样本和评测的任务不一致；假负例太多，模型学的是“记住 batch”；温度太小只拟合最难的负例。先用手算单测和固定小样本排除评测问题，再看训练。

</details>

### F3. 检索失败的样例有什么规律？

<details><summary>参考解答</summary>

分类统计：细粒度属性（颜色、品种）、计数、图中文字、标注或行为噪声。各类占多少，最大的一类怎么改进。这张分类表比主表更有说服力。

</details>

## G. 穿插的八股

### G1. InfoNCE 和交叉熵是什么关系？

<details><summary>参考解答</summary>

InfoNCE 就是把“B 个候选里选出正例”当成一个 B 分类问题的交叉熵，logit 是相似度除以温度。最小化它等价于最大化查询和正例互信息的一个下界，负例越多下界越紧。

</details>

### G2. LoRA 的原理是什么，为什么低秩够用？

<details><summary>参考解答</summary>

冻结原权重 $W$，只训练 $\Delta W=BA$，$A\in\mathbb R^{r\times d}$、$B\in\mathbb R^{d\times r}$，$r\ll d$；$B$ 初始化为 0，训练开始时模型不变。经验上微调所需的权重更新本身是低秩的，所以少量参数就能逼近全参效果。[SFT 与 DPO](/notes/multimodal-sft-lora-dpo/) 第 4 节

</details>

### G3. CLIP 和 SigLIP 的损失有什么区别？

<details><summary>参考解答</summary>

CLIP 用 batch 内 softmax，每个样本要和所有候选做归一化；SigLIP 把每一对当成独立的二分类，用 sigmoid，不需要全局归一化，跨卡通信更省，小 batch 也更稳。[视觉进入语言模型](/notes/vision-video-algorithms/) 第 4 节

</details>

### G4. Qwen2-VL 怎样处理任意分辨率的图？

<details><summary>参考解答</summary>

动态分辨率：图像按原始比例缩放到 28 的倍数，切成 14×14 的 patch，相邻 2×2 合并成一个视觉 token；位置编码用 M-RoPE，分别编码时间、高、宽。分辨率越高 token 越多，要用 `min_pixels`、`max_pixels` 控制预算。

</details>

### G5. 什么是模态鸿沟（modality gap）？

<details><summary>参考解答</summary>

对比学习训出的图像向量和文本向量，在共享空间里往往各自聚在一个锥形区域，两个区域之间有明显间隔，即使配对的图文也不完全重合。它会影响跨模态检索的阈值设定和混合模态检索的公平性。

</details>

### G6. 语义 ID 的 RQ-Kmeans 怎么做？

<details><summary>参考解答</summary>

第一层对所有向量做 K-means，每个向量取最近的中心作为第一个码，减去中心得到残差；第二层对残差再做 K-means，依此类推。快手 OneRec 用 3 层、每层 8192 个码，重建误差低于 RQ-VAE，码本利用率 100%。[语义 ID](/notes/recsys-semantic-id/) 第 3 节

</details>

## 准备清单

1. **记住关键数字**：数据规模、batch、温度、LoRA rank、训练时长、显存、主表和每组消融的结论。数字张口就来，可信度完全不同。
2. **准备三张图**：权衡曲线、模态诊断表、错误分类表。
3. **准备两三个排障故事**：按“现象 → 假设 → 验证 → 结论”讲。
4. **准备一句话的工业对照**：你的设计分别对应 SAIL、QARM、LEMUR、NoteLLM-2 的哪一点。

## 闭卷验收

不看资料用四句话介绍项目；回答“这不就是跑了官方代码吗”；写出 InfoNCE、说清温度的作用和初始 loss；讲清 GradCache 和 all-gather 的梯度回传；说出 COCO 上的泄漏陷阱和公平对比的做法；讲清假负例、流行度纠偏和模态诊断三组消融；说出内容向量为什么不能直接替代 ID、两阶段和端到端各自的问题；解释为什么 LT 涨 0.1% 是大收益；最后按准备清单把自己的数字填进去。

**参考。** [VLM2Vec](https://arxiv.org/abs/2410.05160)；[SAIL-Embedding](https://arxiv.org/abs/2510.12709)；[DME](https://arxiv.org/abs/2608.02148)；[LEMUR](https://arxiv.org/abs/2511.10962)；[QARM](https://arxiv.org/abs/2411.11739)；[OneRec Technical Report](https://arxiv.org/abs/2506.13695)；[NoteLLM](https://arxiv.org/abs/2403.01744)；[NoteLLM-2](https://arxiv.org/abs/2405.16789)；[IDProxy](https://arxiv.org/abs/2603.01590)；[KuaiMod](https://arxiv.org/abs/2504.14904)；[UniNote](https://arxiv.org/abs/2605.29287)；[Better Generalization with Semantic IDs](https://arxiv.org/abs/2306.08121)；[VLM2Rec](https://arxiv.org/abs/2603.17450)；[GradCache](https://arxiv.org/abs/2101.06983)；[CLIP](https://arxiv.org/abs/2103.00020)；[SigLIP](https://arxiv.org/abs/2303.15343)；[LoRA](https://arxiv.org/abs/2106.09685)。
