---
title: 工业界多模态表征：已上线的系统怎样训练、怎样接入、怎样评测
date: '2026-10-06'
tags: [多模态算法, 表征学习, 推荐系统, 语义 ID, 内容理解, 工业实践]
summary: 读了字节、快手、小红书、阿里、Pinterest、YouTube 已经上线的十几个多模态表征系统，归纳出五条和学术检索不同的规律：正样本来自用户行为，内容与协同要折中，向量常被量化成语义 ID，模态会失衡，工程预算决定设计；每条配论文出处、线上数字和手算。最后给出用公开数据 MicroLens 把你的表征项目升级成“有业务场景”的方案。
draft: false
---

在 COCO 上训一个图文检索模型，和抖音训一个给推荐用的视频 embedding，用的可以是同一套代码：VLM、最后一个 token、InfoNCE、大 batch。差别在别处：**正样本从哪来、向量交给谁用、用什么指标判断好坏、能花多少算力**。这一篇读的是 2022–2026 年各家公开、并写明已经上线的多模态表征系统，把它们和学术做法的差别讲清楚，最后落到你的项目怎样升级。

[多模态 embedding](/notes/multimodal-embedding-retrieval/) 讲了 VLM2Vec 这类模型怎样训；[多模态推荐与 I2I](/notes/recsys-multimodal-i2i/) 讲了 QARM、NoteLLM-2 的基本思路和 item2vec 等经典 I2I；[语义 ID](/notes/recsys-semantic-id/) 讲了量化方法。本篇不重复这些基础，专讲已上线系统的具体做法和数字。

阅读入口：[多模态学习手册](/notes/multimodal-interview-guide/)。论文数字均来自原文，可按 arXiv 编号核对；手算例子为教学构造。

## 1. 为什么说 COCO 图文检索是“玩具”

先说清楚：在 COCO 上做对比学习**不是没用**，它能证明你会训 embedding。但它和工业场景有四个差距，面试官一问就会暴露：

| | 学术图文检索 | 工业表征 |
|---|---|---|
| 正样本 | 人工写的图文描述对 | 用户行为：一起看过、搜了又点、看了又买 |
| 向量的用途 | 直接拿来检索，算 Recall@K | 喂给召回、粗排、精排、冷启动、风控等下游模型 |
| 怎样算好 | 检索指标涨了 | 下游 AUC 涨了，线上停留时长、LT（用户留存天数）涨了 |
| 约束 | 一张卡跑完就行 | 十亿级物品库、毫秒级延迟、每天新增上千万条内容 |

下面每一节都对应其中一个差距，并给出已上线系统的具体做法。

## 2. 一张地图：这些系统都在做什么

| 公司 | 系统 | 业务 | 训练信号 | 怎样接入下游 | 公开的线上结果 |
|---|---|---|---|---|---|
| 字节·抖音 | SAIL-Embedding（2510.12709） | 推荐：Feed、消息推送、冷启动、抖音精选 | 超过 100 亿对样本，主体是用户行为对 | 稠密向量 + 语义 ID，进召回、粗排、精排、重排 | 精选场景 LT7 +0.5%；Feed 精排 AUC +0.1% |
| 字节·抖音 | DME（2608.02148） | 搜索：生成式搜索、图搜、AI 搜索 | 约 2500 万弱监督对 + 指令数据 + 生成式重建 | 检索，也作为排序特征 | 离线相对提升 2.92%；搜索 LT +0.1% |
| 字节 | LEMUR（2511.10962） | 抖音搜索、广告 | 搜索会话里的查询–文档对，和排序模型联合训练 | 端到端：多模态塔和排序模型一起训 | QAUC +0.81%，已全量 |
| 快手 | QARM（2411.11739） | 广告、电商 | 召回模型认定的相似物品对、Swing 物品对 | 量化成 VQ/RQ 码，当作可训练的 ID 特征 | 广告收入最高 +9.7%（冷启动物品组），电商 GMV +1.6%~2.3% |
| 快手 | OneRec tokenizer（2506.13695） | 主站短视频推荐 | 协同相似的视频对 + 描述生成损失 | RQ-Kmeans 三层语义 ID，供生成式推荐 | 承接约 25% 流量，停留时长 +0.54%/+1.24% |
| 快手 | KuaiMod（2504.14904） | 内容审核 | 用户举报、审核员反馈 + 思维链 | VLM 直接判断，结果也用于推荐 | 用户举报率 −20% |
| 小红书 | NoteLLM（2403.01744） | 笔记 I2I 推荐 | 一周内的共现点击对 | I2I 向量召回 | CTR +16.2%，新笔记当日评论 +3.58% |
| 小红书 | NoteLLM-2（2405.16789） | 笔记 I2I 推荐 | 同上，加视觉增强 | 替换原 I2I 召回通路 | 新笔记前 24 小时互动 +8.08% |
| 小红书 | UniNote（2605.29287） | 内容安全：风险样本匹配、回查 | 对比 SFT + GRPO 相关性强化学习 | I2I 检索，Matryoshka 可变维度 | 召回保留 85.6%–93.6%，原方案要多花 9.2 倍存储与计算 |
| 小红书 | IDProxy（2603.01590） | 冷启动 CTR | 对齐到排序模型的 ID embedding 空间 | 新物品用代理向量替代 ID 向量 | 2025 年底全量，覆盖信息流和广告 |
| 阿里·淘宝 | MIM（2502.00321） | CTR 预估 | 拍立淘“搜图后购买”的图–商品对 | 向量进用户行为建模 | CTR +14.14%，RPM +4.12% |
| Pinterest | ItemSage（2205.11728） | 购物推荐 | 多种用户互动，多任务 | 一套商品向量服务所有场景 | GMV/用户 +7%，点击 +11% |
| Pinterest | OmniSearchSage（2404.16260） | 搜索：pin、商品、查询的召回与排序 | 一年搜索日志里的查询–互动对：收藏、长点击、加购、下单 | 一个查询向量检索三类对象，并兼容旧的 PinSage、ItemSage 向量 | 相关性 +8% 以上，互动 +7% 以上，广告 CTR +5% 以上 |
| YouTube | Semantic IDs（2306.08121） | 推荐排序 | 冻结的内容向量做 RQ-VAE | 用语义 ID 替代随机哈希的视频 ID | 新视频、长尾视频泛化更好，整体不掉 |

表里的百分比看起来很小，但规模摆在那里。快手在 OneRec 报告中写明，**停留时长 0.1%、LT7 0.01% 的提升在快手就已经统计显著**。所以 SAIL 的 LT7 +0.5% 是很大的收益。面试时如果把 0.1% 说成“提升不大”，会显得不懂业务。

## 3. 规律一：正样本来自用户行为，不来自描述

### 3.1 各家怎样构造正样本

**抖音 SAIL-Embedding** 的训练数据（论文 Table 1）最能说明问题。物品对物品的数据里，按用户共同消费构造的 Copair-i2i 有 29 亿对，按搜索行为构造的 Search-i2i 有 6 亿对，直播共现的 Live-i2i 有 14 亿对；查询对物品的数据里，“搜索词 → 被点击的视频”有 18 亿对；另有 31 亿条“视频 → 多级标签”的分类数据。按视频摘要文本经 n-gram 过滤构造的内容类 Summary-i2i 有 11 亿对，只是其中一部分。

**快手 QARM** 有两个来源（§2.2）：一是对每个用户正向点击的目标物品，在他最近 50 个正向点击里找 ID 空间最相似的一个，配成对；二是直接导出 Swing 召回模型认定的高相似物品对。OneRec 的 tokenizer 用的是同一种数据。

**小红书 NoteLLM** 统计一周的用户行为（§4.3）：用户看了笔记 A 之后点了笔记 B，记一次共现。每个用户的贡献按他点过的笔记数取倒数：

$$
s_{A\to B}=\sum_{u=1}^{U}\frac{1}{N_u},
$$

其中 $N_u$ 是用户 $u$ 点过的笔记数。然后去掉共现分过高或过低的异常对，每篇笔记取分数最高的若干篇作为正样本。整体框架见 [NoteLLM Figure 2](https://arxiv.org/html/2403.01744v2#S4.F2)。

**手算**（教学构造）：三个用户都有“看 A 后点 B”的行为，他们一周分别点了 2、10、4 篇笔记。加权共现分是 $1/2+1/10+1/4=0.85$；不加权是 3。如果第四个用户是“什么都点”的重度用户，一周点了 50 篇，他只贡献 $1/50=0.02$。**倒数加权是为了不让重度用户的随手点击主导“相关”的定义**，和 [ItemCF 惩罚热门](/notes/wangshusen-recommender-retrieval/) 是同一个思路。

**淘宝 MIM**（§3.3）用的是拍立淘（以图搜商品）里“搜了这张图、最后买了这个商品”的对，并在 §4.3.2 对比了点击和购买两种信号，理由是购买比点击更可靠地反映兴趣。整体框架见 [MIM Figure 2](https://arxiv.org/html/2502.00321v4#S1.F2)。

**Pinterest OmniSearchSage**（§4.1）从一年的搜索日志里抽“查询–互动对象”对：收藏，以及长点击（点出去浏览超过 10 秒再回来）；商品还加上站外的加购和下单。为了压住热门偏差，同一个 pin 最多配 50 对，商品最多 200 对。

### 3.2 为什么不用描述对

[多模态推荐与 I2I](/notes/recsys-multimodal-i2i/) 第 2 节讲过 QARM 的判断：用图文匹配训出的向量，和推荐模型用用户行为学到的东西目标不一致。两件外观相似的衣服在 CLIP 空间里很近，但一件是爆款、一件无人问津；两个画面毫不相关的视频，可能总被同一批人连着看。[QARM Figure 1](https://arxiv.org/html/2411.11739v1#S1.F1) 的 (a) 画的就是这种“表征不匹配”，(b) 是第 5 节要讲的“表征学不动”。

在 SAIL 的评测（Table 2）里，同样在 Copair-i2i（行为共现对）上测 Recall@50，CLIP 结构的模型是 53.47，普通 VLM 是 66.06，SAIL 是 69.17。三者都在同一份训练数据上微调过，差距来自模型和训练方法。三种结构的对比见 [SAIL Figure 2](https://arxiv.org/html/2510.12709v3#S1.F2)。

### 3.3 行为数据里的假负例更多

行为对比描述对噪声大得多，假负例问题也更严重：

- **LEMUR** 的对比损失（§3.3）在 batch 里屏蔽“同一个搜索会话的其他样本”：同一个查询下用户点过的多个视频，不应该互相当负例。消融（Table 5）显示，去掉这个会话屏蔽，QAUC 从 +0.81% 降到 +0.71%。
- **DME**（§4.1）在挖难负例时，去掉和正例相似度过高的候选；它的 batch 实验（§5.4）显示，batch 从 128 增大到 8192 一直有收益，再大就收益递减甚至略降。作者的解释是 batch 越大，batch 里混进假负例的概率越高，而真正难的负例的梯度被大量容易负例稀释。他们的结论是：单纯放大 batch 只在一定范围内有用，要配合假负例过滤和难负例构造。

<figure style="margin:1.5rem 0">
<a href="/notes-assets/multimodal-industrial-representation/dme-fig5-batch-size.png" target="_blank" rel="noopener" aria-label="查看原图：DME 的 batch 大小实验"><img src="/notes-assets/multimodal-industrial-representation/dme-fig5-batch-size.png" alt="折线图：batch 从 128 增大到 8192，Overall 从 0.6437 升到 0.7083；batch 为 16384 时降到 0.7000" width="1404" height="993" loading="lazy" decoding="async" style="height:auto;cursor:zoom-in" /></a>
<figcaption style="font-size:0.9em">DME 的 batch 实验：128 到 1024 涨得快，2048 以后接近饱和，8192 最高（0.7083），16384 反而降到 0.7000。图源：Douyin Multimodal Embedding（arXiv 2608.02148）Figure 5，CC BY 4.0，未修改（点击查看原图）</figcaption>
</figure>

**这正是你项目里“假负例处理”那组消融的工业版本**，见第 11 节。

## 4. 规律二：内容和协同信号要折中

只用行为对训练，向量会越来越像“ID 向量”，把内容信息丢掉。可只用内容，又和推荐目标对不上。各家都在找平衡：

- **OneRec 的 tokenizer**（§2.1）同时用两个损失：物品对比损失对齐协同相似的视频；再加一个描述生成损失，用 LLaMA3 当解码器，从表征生成视频描述，论文说这是为了“防止幻觉、保留内容理解能力”。流程图见 [OneRec Figure 3](https://arxiv.org/html/2506.13695v4#S2.F3)。
- **SAIL 的协同增强训练**（§3.3.5，Table 4）：用户序列到物品的蒸馏、ID 向量到物品的蒸馏（示意见 [SAIL Figure 7](https://arxiv.org/html/2510.12709v3#S3.F7)），让 GID-i2i 的 Recall@50 从 52.46 升到 59.69；但 Copair-i2i 反而从 69.17 降到 66.83。作者明确说这种变化是可以接受的，是为了平衡内容导向和行为导向的不同需求。**同一个改动，一个指标涨、另一个跌**，这就是“折中”的具体样子。
- **YouTube**（2306.08121）发现，直接用内容向量替换视频 ID，整体质量会下降，因为失去了 ID 的“记忆”能力；语义 ID 能在新视频和长尾视频上泛化得更好，又不牺牲整体。
- **IDProxy**（2603.01590，[Figure 1](https://arxiv.org/html/2603.01590v2#S1.F1)）指出，用 Swing 物品对训出的多模态向量（QARM 的做法）和线上排序模型的 ID 向量分布差别很大，所以不适合直接当冷启动物品的 ID 替身。它的做法是把 MLLM 的输出对齐到 ID 向量空间，并和 CTR 模型端到端一起训。
- **冻结 LVLM 的系统研究**（2512.21863）在 MicroLens 等公开数据上得到三条结论：中间层隐状态比生成的描述更好用（[Figure 2](https://arxiv.org/html/2512.21863v2#S4.F2)）；**和 ID 向量融合严格好于替换 ID**；不同层的效果差别很大（[Figure 3](https://arxiv.org/html/2512.21863v2#S4.F3)）。

YouTube 那组实验的原图如下。模型输入里不含用户历史，纵轴是相对基线（随机哈希、8K 行嵌入表）的 CTR AUC 提升百分比；左图是全部视频，右图只看测试当天新上传的视频（CTR/1D，即冷启动）。直接把内容向量当输入（Dense Input，层数 ×1）在两张图里都低于基线；把排序模型加深到 1.5 倍、2 倍能追回来，但服务成本明显上升。语义 ID（SPM-SID）在嵌入表够大时整体提升约 0.13%–0.15%，冷启动提升约 0.5%。原文说 CTR AUC 变化 0.1% 对他们的排序模型就算显著。

<figure style="margin:1.5rem 0">
<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,260px),1fr));gap:0 1rem;align-items:start">
<a href="/notes-assets/multimodal-industrial-representation/youtube-sid-fig2a-overall-auc.png" target="_blank" rel="noopener" aria-label="查看原图：全部视频的 CTR AUC 提升" style="display:block;min-width:0"><img src="/notes-assets/multimodal-industrial-representation/youtube-sid-fig2a-overall-auc.png" alt="全部视频：SPM-SID 在大嵌入表下提升约 0.15%，直接用内容向量（层数 ×1）约为 -0.045%" width="1524" height="1241" loading="lazy" decoding="async" style="height:auto;cursor:zoom-in" /></a>
<a href="/notes-assets/multimodal-industrial-representation/youtube-sid-fig2b-coldstart-auc.png" target="_blank" rel="noopener" aria-label="查看原图：当天新视频的 CTR AUC 提升" style="display:block;min-width:0"><img src="/notes-assets/multimodal-industrial-representation/youtube-sid-fig2b-coldstart-auc.png" alt="当天新视频：SPM-SID 提升约 0.47% 到 0.55%，直接用内容向量（层数 ×1）约为 -0.07%" width="1468" height="1241" loading="lazy" decoding="async" style="height:auto;cursor:zoom-in" /></a>
</div>
<figcaption style="font-size:0.9em">左：全部视频（Overall CTR AUC）；右：当天新视频（Cold-start CTR/1D AUC）。图源：Better Generalization with Semantic IDs（arXiv 2306.08121）Figure 2(a)(b)，CC BY 4.0，未修改（点击查看原图）</figcaption>
</figure>

**一句话总结**：内容向量负责泛化（新物品、长尾），ID 负责记忆（热门物品的细节），工业上几乎都是融合，很少直接替换。

## 5. 规律三：向量怎样接入下游，有五种用法

| 用法 | 怎样做 | 代表 | 为什么这样做 |
|---|---|---|---|
| I2I 向量召回 | 用户点过的物品当 trigger，近邻检索相关物品 | NoteLLM、NoteLLM-2、SAIL | 最直接；新物品有内容就能召回 |
| 排序特征 | 向量或“用户历史与目标的相似度”作为精排特征 | SAIL、DME | 不改结构，增量上线 |
| 语义 ID | 量化成离散码，下游给每个码学 embedding | QARM、OneRec、SAIL、YouTube | 码可以被推荐梯度更新；规则也好用 |
| 冷启动代理 | 新物品用内容生成的向量替代 ID 向量 | IDProxy | 新物品没有 ID 统计量 |
| 端到端 | 多模态塔和排序模型一起训 | LEMUR | 消除两阶段的目标不一致 |

**语义 ID 为什么常常比稠密向量更好用。** SAIL 的线上实验（Table 6）发现，语义 ID 带来的收益比稠密向量更大，作者给出两个原因：语义 ID 更容易用于规则类策略，例如打散（同一个码的视频不要连续推）；语义 ID 可以像物品 ID 一样被编码成可训练的 embedding，跟着推荐模型一起更新。QARM 用“表征学不动”描述的是同一个问题：稠密向量缓存起来作为固定输入，推荐模型的梯度传不回去。QARM 的“对齐 → 量化 → 码当 ID 特征训练”流程见 [QARM Figure 3](https://arxiv.org/html/2411.11739v1#S2.F3)，线上部署见 [Figure 4](https://arxiv.org/html/2411.11739v1#S2.F4)。

**RQ-Kmeans 手算**（教学构造，二维）。物品向量 $e=(0.9,0.2)$。第一层码本 $\{(1,0),(0,1)\}$，最近的是第 1 个（距离 0.224，另一个是 1.204），残差 $r=e-(1,0)=(-0.1,0.2)$。第二层码本 $\{(0,0.2),(-0.2,0)\}$，残差离第 1 个更近（0.100 对 0.224）。语义 ID 是 $(1,1)$，重建为 $(1,0)+(0,0.2)=(1,0.2)$，误差 0.1。

<figure style="margin:1.5rem 0">
<a href="/notes-assets/multimodal-industrial-representation/youtube-sid-fig1-rqvae.png" target="_blank" rel="noopener" aria-label="查看原图：RQ-VAE 残差量化"><img src="/notes-assets/multimodal-industrial-representation/youtube-sid-fig1-rqvae.png" alt="RQ-VAE：编码器把 x 压成 z，四层依次找最近码字并减去得到残差，码字 1、4、6、2 相加得到重建向量，再由解码器还原" width="2490" height="1112" loading="lazy" decoding="async" style="height:auto;cursor:zoom-in" /></a>
<figcaption style="font-size:0.9em">RQ-VAE 的残差量化，图中这件物品的语义 ID 是 (1, 4, 6, 2)。图源：Better Generalization with Semantic IDs（arXiv 2306.08121）Figure 1，CC BY 4.0，未修改（点击查看原图）</figcaption>
</figure>

**读图**：每一层在码本里找离当前残差最近的码字，减掉它，剩下的残差交给下一层；四个码字相加得到 $\hat z$，再由解码器重建 $\hat x$。和上面的手算是同一个过程，区别在于 RQ-VAE 多了可训练的编码器和解码器，码本和它们一起训；RQ-Kmeans 直接对残差逐层做 k-means，没有编解码器。

OneRec 用的是 3 层、每层 8192 个码，组合数 $8192^3\approx5.5\times10^{11}$，足够区分亿级视频。它对比了两种量化（Table 11）：RQ-Kmeans 的重建误差 0.0410，低于 RQ-VAE 的 0.0548，码本利用率三层都是 100%。原理见 [语义 ID](/notes/recsys-semantic-id/) 第 2–3 节。

语义 ID 的前缀是由粗到细的概念。YouTube 论文附录画了一棵体育视频的子树：

<figure style="margin:1.5rem 0">
<a href="/notes-assets/multimodal-industrial-representation/youtube-sid-fig6-sports-trie.png" target="_blank" rel="noopener" aria-label="查看原图：语义 ID 的体育视频子树"><img src="/notes-assets/multimodal-industrial-representation/youtube-sid-fig6-sports-trie.png" alt="语义 ID 子树：前缀 1723 是体育，下分力量举与健美、格斗、户外运动；户外运动 (1723, 541) 再分出冲浪、铁人三项、沙滩排球" width="1600" height="798" loading="lazy" decoding="async" style="height:auto;cursor:zoom-in" /></a>
<figcaption style="font-size:0.9em">前缀 1723 是体育，(1723, 541) 是户外运动，再往下分出冲浪、铁人三项、沙滩排球。图源：Better Generalization with Semantic IDs（arXiv 2306.08121）Figure 6，CC BY 4.0；为控制页面体积，缩小到 1600 像素宽并压缩了颜色，内容未改（点击查看原图）</figcaption>
</figure>

同一篇的附录 Table 1 给了定量结果：共享前缀长度为 1、2、3、4 的视频，内容向量的平均余弦相似度分别是 0.41、0.68、0.91、0.97，对应的子树一般有 15 万–45 万、20–150、1–5、1 个视频。**同前缀的视频内容相近**，所以“同一个码的视频不要连续推”这类打散规则可以直接按前缀写。

**两阶段与端到端。** LEMUR（§1）总结了“先预训练多模态模型、再把冻结向量交给推荐模型”的四个问题：内容向量没有被用户行为引导；推荐模型优先学 ID 向量，内容向量欠拟合；推荐模型实时更新，多模态模型训完就不动；长序列的多模态向量在线传输成本高。它的办法是端到端联合训练，并用“记忆库”缓存历史物品的向量，避免每步重新编码整个用户序列（框架见 [LEMUR Figure 1](https://arxiv.org/html/2511.10962v2#S1.F1)）。对比实验（Table 3）里，两阶段方法的 QAUC 只涨 0.12%，端到端涨 0.81%。

**但要注意 LEMUR 的多模态塔很小**：查询端 2 层、文档端 4 层 Transformer，向量 128 维（§4.1.4），不是几十亿参数的 VLM。端到端的代价是算力，用 VLM 做端到端目前不现实，这也是 QARM 选择“对齐 + 量化”这条折中路线的原因。

**和已有向量兼容。** 线上换一版向量，不能一夜之间把所有索引和下游模型都重建。Pinterest 的 OmniSearchSage（§3.4.3）训练新的查询向量时，除了和新的 pin、商品编码器对齐，还加了两个“兼容”任务，让查询向量也能直接检索旧的 PinSage、ItemSage 向量，这两个旧模型冻结不动。论文说兼容旧向量是为了节省成本、简化迁移。

<figure style="margin:1.5rem 0">
<a href="/notes-assets/multimodal-industrial-representation/omnisearchsage-fig1-multitask.png" target="_blank" rel="noopener" aria-label="查看原图：OmniSearchSage 多任务结构"><img src="/notes-assets/multimodal-industrial-representation/omnisearchsage-fig1-multitask.png" alt="一个查询编码器同时连五个损失：与冻结的 ItemSage、PinSage 向量兼容的两个任务，与从头训练的统一 pin–商品编码器对齐的两个任务，以及查询对查询" width="738" height="597" loading="lazy" decoding="async" style="height:auto;cursor:zoom-in" /></a>
<figcaption style="font-size:0.9em">一个查询编码器同时学五个任务：带雪花的 ItemSage、PinSage 是冻结的旧向量，对应 L(query, product_c)、L(query, pin_c) 两个兼容任务；统一的 pin–商品编码器从头训练；L(query, query) 学查询之间的相似。图源：OmniSearchSage（arXiv 2404.16260）Figure 1，CC BY 4.0，未修改（点击查看原图）</figcaption>
</figure>

## 6. 规律四：模态会失衡，图像容易被忽视

**NoteLLM-2**（§2.4，Table 3）做了一个很有说服力的实验：同一个微调后的模型，分别只输入图像、只输入文本、图文都输入，测 I2I 召回。以 MTomato-Base 为例，只给文本 R@100 是 71.49，图文都给是 71.94，**只给图像只有 0.60**。也就是说，这个模型几乎完全没用图。注意力分析（§2.5，[Figure 3](https://arxiv.org/html/2405.16789v2#S2.F3)）也显示，最终表征主要从文本 token 获取信息。

它的两个修正：mICL，在提示里把视觉和文本内容分开，各自压缩成一个 token，分别做对比学习；late fusion，把视觉编码器的输出直接融合进最终表征，绕过 LLM 对视觉的削弱。两种机制的框架图见 [NoteLLM-2 Figure 2](https://arxiv.org/html/2405.16789v2#S1.F2)。线上，它在新笔记上的提升最大：前 24 小时的互动数 +8.08%。

**VLM2Rec**（2603.17450）在序列推荐上得到同样的结论：对比 SFT 会放大模态失衡，优化被强势模态（通常是文本）主导，弱势模态越训越差。它的 [Figure 1](https://arxiv.org/html/2603.17450v2#S2.F1) 用模态 dropout 测试和梯度分析展示了这一点。

**为什么这对内容理解岗重要。** 短视频的标题常常很短、甚至没有，真正的信息在画面里。模型如果只会读标题，在“标题党”和“无标题视频”上就会失灵。所以**分别测“只给文本、只给图像、图文都给”**，是这个方向的标准诊断，见 [多模态 RL](/notes/multimodal-rl/) 第 4 节的“视觉依赖度”。

## 7. 规律五：工程预算决定设计

- **视觉 token 预算**：DME（§5.4）把每张图限制在 1280 个视觉 token、每个视频均匀采 32 帧。实验显示，图像 token 从 256 增加到 1280 各项指标都涨；视频训练和推理都用 32 帧最好，推理加到 64 帧几乎不再提升。
- **推理延迟**：DME 的“潜在推理 token”只在编码器的一次前向里加几个 token，查询端每次多花不到 1 毫秒（Table 8），不需要在线生成思维链。思维链类方法效果好，但上线成本太高，这是它们在工业界难以直接落地的原因。
- **向量维度**：UniNote 用 Matryoshka 表征，维度可以按成本选（§4.4）：512 维时多数任务已接近全维度；降到 64 维，在原子对齐、从属检索、语义抽取等任务上仍保留约 70% 的能力（维度与效率、召回的关系见 [UniNote Figure 4](https://arxiv.org/html/2605.29287v2#S4.F4)）。**手算**：以 1536 维为例，fp16 每个向量 3 KB，64 维只要 128 字节；十亿个物品分别是 3 TB 和 128 GB。
- **记忆库的陈旧问题**：LEMUR 的记忆库里，缓存的向量和当前模型输出之间的余弦相似度最终稳定在约 0.95（§4.5，[Figure 3](https://arxiv.org/html/2511.10962v2#S4.F3)），所以陈旧不是大问题；序列物品在记忆库里的覆盖率超过 90%。

DME 的两张图把“预算决定设计”画得很直观。第一张对比三种做法：

<figure style="margin:1.5rem 0">
<a href="/notes-assets/multimodal-industrial-representation/dme-fig2-compare.png" target="_blank" rel="noopener" aria-label="查看原图：DME 与已有方法的对比"><img src="/notes-assets/multimodal-industrial-representation/dme-fig2-compare.png" alt="三栏对比：VLM2Vec 直接出向量，低延迟、粗粒度；CoT-VLM2Vec 先写显式思维链，高延迟、细粒度；DME 用潜在思维链加生成损失，低延迟、细粒度" width="1236" height="333" loading="lazy" decoding="async" style="height:auto;cursor:zoom-in" /></a>
<figcaption style="font-size:0.9em">(a) VLM2Vec：一次前向出向量，低延迟、粗粒度；(b) 先生成显式思维链再出向量：细粒度、高延迟；(c) DME：潜在思维链加生成损失，低延迟、细粒度。图源：Douyin Multimodal Embedding（arXiv 2608.02148）Figure 2，CC BY 4.0，未修改（点击查看原图）</figcaption>
</figure>

第二张是两阶段训练流程。昂贵的部分都放在训练里：教师模型生成的证据记录（Anchor）和推理轨迹（Trajectory）只用来监督，重建分支也只在训练时使用；上线时仍是一次前向、按向量相似度检索。

<figure style="margin:1.5rem 0">
<a href="/notes-assets/multimodal-industrial-representation/dme-fig3-pipeline.png" target="_blank" rel="noopener" aria-label="查看原图：DME 两阶段训练流程"><img src="/notes-assets/multimodal-industrial-representation/dme-fig3-pipeline.png" alt="第一阶段用 2500 万对数据做大规模对比学习；第二阶段用 500 万条数据加教师生成的 Anchor 与 Trajectory，联合优化对比损失和语义充分性损失" width="904" height="199" loading="lazy" decoding="async" style="height:auto;cursor:zoom-in" /></a>
<figcaption style="font-size:0.9em">第一阶段：约 2500 万对数据，只用对比损失；第二阶段：500 万条更高质量的数据，加教师生成的思维链监督，对比损失和语义充分性损失联合优化。图源：Douyin Multimodal Embedding（arXiv 2608.02148）Figure 3，CC BY 4.0，未修改（点击查看原图）</figcaption>
</figure>

## 8. 工业界怎样评测表征

**离线有三层。**

1. **表征本身**：在行为对上算 I2I Recall@50/100（SAIL）、在数十万物品的库上算 R@100/1k/10k（NoteLLM-2 的测试库有 53 万篇笔记、2.1 万个测试对）。SAIL 还用了几种分布指标：正负样本相似度分布的可分性、聚类一致性（NMI）、排序一致性（Kendall τ）、双向检索能否互相找回。
2. **下游模型**：把向量或语义 ID 接进召回、排序模型，看 AUC、QAUC、GAUC。
3. **分人群、分物品**：按物品的曝光量或交互数分桶，单独报告冷启动和长尾。NoteLLM-2 专门在“短查询”子集（文本很短的笔记）上报告结果，这正是图像最有用的地方。

**线上**：LT7、LT30（用户留存）、停留时长、互动数；冷启动场景看新内容的首日互动、首千次曝光的点击（NoteLLM-2）。广告和电商看收入、GMV。

**一个常被忽视的点**：SAIL 在 §4.2 明确写道：一个多模态 embedding 模型自身指标可能很强，却对下游用户互动贡献很小。所以只报告 Recall@K 的项目，在工业面试官眼里是不完整的。

## 9. 内容理解的另一半：审核与治理

内容理解岗不只做推荐表征，还有审核、打标签、质量判断。这部分的代表工作：

**快手 KuaiMod**（2504.14904）：用 VLM 加思维链判断视频是否有害，训练数据来自用户举报和审核员反馈，策略可以快速更新。上线后用户举报率降低 20%，用在推荐里时 DAU 和使用时长都上涨。论文把它和人工审核、传统 AI 审核放在一起对比：

<figure style="margin:1.5rem 0">
<a href="/notes-assets/multimodal-industrial-representation/kuaimod-fig1a-paradigms.png" target="_blank" rel="noopener" aria-label="查看原图：三种内容审核范式"><img src="/notes-assets/multimodal-industrial-representation/kuaimod-fig1a-paradigms.png" alt="三种审核范式：人工审核成本高、效率低、有偏差；传统规则或网络模型不准、难解释、难泛化；KuaiMod 把 VLM 当作策略，可适应、可解释、可泛化" width="1013" height="683" loading="lazy" decoding="async" style="height:auto;cursor:zoom-in" /></a>
<figcaption style="font-size:0.9em">(a) 人工审核：审核员学习平台规则后逐条判断，成本高、效率低、有主观偏差；(b) 传统 AI：规则或小模型单独或组合使用，不准、难解释、难泛化；(c) KuaiMod 把 VLM 当作审核策略，判断并给出解释，再根据平台反馈更新策略。图源：KuaiMod（arXiv 2504.14904）Figure 1(a)，CC BY 4.0，未修改（点击查看原图）</figcaption>
</figure>

上线后的迭代方式是论文所说的 RLUF（从用户反馈中强化学习）：

<figure style="margin:1.5rem 0">
<a href="/notes-assets/multimodal-industrial-representation/kuaimod-fig3-online.png" target="_blank" rel="noopener" aria-label="查看原图：KuaiMod 线上迭代流程"><img src="/notes-assets/multimodal-industrial-representation/kuaimod-fig3-online.png" alt="快手平台的用户反馈和人工复核把视频分成高质量和低质量两类；收集旧策略判错的样本，经 YuanQi-20B 的 Tag2CoT 与 CoT2Tag 转成状态转移数据，后训练得到新策略，再上线服务" width="2039" height="392" loading="lazy" decoding="async" style="height:auto;cursor:zoom-in" /></a>
<figcaption style="font-size:0.9em">用户反馈和人工复核把视频分成高质量、低质量两类；收集旧策略判错的样本，由 YuanQi-20B 通过 Tag2CoT、CoT2Tag 转成状态转移格式的训练数据，后训练得到新策略，再上线服务。图源：KuaiMod（arXiv 2504.14904）Figure 3，CC BY 4.0，未修改；图中缩略图来自快手平台（点击查看原图）</figcaption>
</figure>

**小红书 UniNote**（2605.29287）：每天新发的笔记和一个风险样本库做 I2I 匹配（§5）。对比线上 CLIP 方案，在 Note2Image、Note2Video、Note2Text 三类匹配上保留了 85.6%、91.2%、93.6% 的召回，而原方案要多花 9.2 倍的存储和计算；回查场景的相关样本召回提升 23.5%。它在 embedding 训练里用了 GRPO 强化学习，直接优化排序。训练流程如下：

<figure style="margin:1.5rem 0">
<a href="/notes-assets/multimodal-industrial-representation/uninote-fig2-pipeline.png" target="_blank" rel="noopener" aria-label="查看原图：UniNote 训练流程"><img src="/notes-assets/multimodal-industrial-representation/uninote-fig2-pipeline.png" alt="三栏：数据构造与难负例挖掘；第一阶段对比 SFT 与 Matryoshka 表征；第二阶段用 GRPO 做基于强化学习的重排" width="1434" height="685" loading="lazy" decoding="async" style="height:auto;cursor:zoom-in" /></a>
<figcaption style="font-size:0.9em">(a) 数据：用描述生成、模态替换、OCR 抽取合成三类正样本对，按相似度区间和规则挖多粒度难负例；(b) 第一阶段：对比 SFT 加 Matryoshka（MRL），取最后一个 token 作表征，并用软标签监督；(c) 第二阶段：用 GRPO 对候选排序做强化学习，奖励包括无关笔记惩罚、相关命中、绝对位置和相对顺序。图源：UniNote（arXiv 2605.29287）Figure 2，CC BY 4.0，未修改（点击查看原图）</figcaption>
</figure>

线上分两种模式：

<figure style="margin:1.5rem 0">
<a href="/notes-assets/multimodal-industrial-representation/uninote-fig5-deployment.png" target="_blank" rel="noopener" aria-label="查看原图：UniNote 线上部署"><img src="/notes-assets/multimodal-industrial-representation/uninote-fig5-deployment.png" alt="在线模式：新笔记经 UniNote 编码后入库并查询风险样本库，相似度超过阈值就处置；离线模式：更新风险样本库后，用 ANN 从历史库召回 Top-K 笔记" width="935" height="264" loading="lazy" decoding="async" style="height:auto;cursor:zoom-in" /></a>
<figcaption style="font-size:0.9em">在线模式：每天新发的笔记一边入库，一边作为查询去匹配风险样本库，相似度超过阈值就交给处置；离线模式：运营更新风险样本库后，用 ANN 从历史库里召回 Top-K 笔记回查。图源：UniNote（arXiv 2605.29287）Figure 5，CC BY 4.0，未修改（点击查看原图）</figcaption>
</figure>

**TikTok 的内容治理预训练**（2509.21486，EMNLP 2025 Industry）：用描述、问答、思维链三种预训练任务，让一个 MLLM 统一识别多种违规内容，并能泛化到新出现的违规类型。摘要没有写线上部署结果。

**快手的长尾搜索重排**（2603.24975）：用多模态 LLM 估计内容质量，在用户行为稀疏的长尾查询上压制标题党、扶持优质但曝光少的视频。

共同点是：**用多模态大模型的理解能力，弥补用户行为稀疏或有偏的地方**。

## 10. 公开数据能做什么

工业数据拿不到，但有几个公开数据集同时带原始内容和用户行为：

| 数据集 | 内容 | 规模 | 适合做什么 |
|---|---|---|---|
| MicroLens-100K（2309.15379） | 标题、封面、音频、评论 | 10 万用户、19,738 个视频、71.9 万次交互，平均时长 161 秒 | 短视频 I2I、序列推荐、冷启动 |
| MicroLens-1M | 同上，2026 年 3 月起开放全部视频 | 100 万用户、91,402 个视频、910 万次交互 | 规模扩展实验 |
| Amazon Reviews | 商品标题、图片、评论 | 按品类划分 | VLM2Rec、LEARN 都在上面评测 |
| MIM 公开数据 | 淘宝场景 | 见论文链接 | CTR 方向 |

MicroLens 的论文（2309.15379）还给了一个有用的发现：在它的基准上，只用文本特征和只用 ID 效果差不多，因为不少视频的标题只有几个词；用视频特征的模型能超过 ID。这正好说明**画面信息在短视频里不可替代**。数据集里的视频样例见 [MicroLens Figure 2](https://arxiv.org/html/2309.15379v1#S2.F2)。

Spotify 的 Describe What You See（2508.09789，RecSys 2025）在 MicroLens-100K 上换了个思路：不微调 MLLM，只让现成的 MLLM 给每个视频写一段描述，再用文本编码器转成向量，喂给常规推荐模型。双塔模型的 HR@10，用元数据是 0.0414，用视频特征是 0.0393，用 MLLM 对视频写的描述是 0.0489（Table 1）。论文第一张图说明了描述里多出来的是什么：

<figure style="margin:1.5rem 0">
<a href="/notes-assets/multimodal-industrial-representation/describe-fig1-naruto.jpg" target="_blank" rel="noopener" aria-label="查看原图：MLLM 对一段手游视频的描述"><img src="/notes-assets/multimodal-industrial-representation/describe-fig1-naruto.jpg" alt="一段火影手游视频的四个关键帧，以及 MLLM 生成的描述，其中绿色是屏幕文字，红色是世界知识，蓝色是细粒度情节描述" width="1600" height="1061" loading="lazy" decoding="async" style="height:auto;cursor:zoom-in" /></a>
<figcaption style="font-size:0.9em">MLLM（图中为 Qwen-VL）从一段火影手游视频里读出三类信息：屏幕文字（绿，OCR）、世界知识（红，认出角色和招式）、细粒度的情节和语气（蓝）。图源：Describe What You See（arXiv 2508.09789）Figure 1，CC0；为控制体积重新压缩为 JPEG，内容未改；图中游戏画面的权利属于原视频与游戏的权利人（点击查看原图）</figcaption>
</figure>

## 11. 把你的项目升级成“有业务场景”的版本

你现在的计划：Qwen2-VL-2B + VLM2Vec 方法，在 COCO/Flickr 上做图文检索，加四组消融。下面的升级**保留全部训练代码和工程基础**（环境、LoRA、GradCache、all-gather、评测脚本的写法），只换数据、正样本和评测，对应工业链路：

**数据**：MicroLens-100K 的封面加标题。按用户时间顺序留一切分：最后一次交互做测试，倒数第二次做验证，其余做训练（MicroLens 基准的标准做法）。

**正样本**：用训练期序列构造共现对。窗口内（例如相邻 3 步）出现的两个视频记一次，按 NoteLLM 的 $1/N_u$ 加权，汇总后每个视频保留分数最高的若干个。**只用训练期交互，否则测试期的共现会泄漏进训练**。

**对照组**：
- B0：CLIP 零样本；
- B1：VLM2Vec-V2.0 零样本；
- B2：只用“标题 ↔ 封面”训练，也就是你原来的图文对齐目标，现在变成基线；
- M1：只用行为共现对训练；
- M2：两种损失加权（$\lambda$ 取三个值），对应 OneRec“行为对比 + 内容保持”的思路。

**评测分三层**，和第 8 节一一对应：
1. **I2I 召回**：对每个测试用户，用倒数第二个视频当 trigger，看能否在全部 19,738 个视频里召回他最后一个视频，报告 Recall@50/100；按目标视频的流行度分桶，并单独报告训练期从未出现过的冷启动视频；
2. **下游推荐**：把向量或语义 ID 接进 SASRec（MicroLens 仓库有现成代码），对比“只用 ID / ID + 多模态向量融合 / 只用多模态向量 / 语义 ID”，报告 HR@10、NDCG@10，同样分冷启动；
3. **内容保持**：标题搜封面的检索指标，监控行为训练有没有把内容信息训丢。

**四组消融，对应原来的四个位置**：

| 原计划 | 升级后 | 对应的工业问题 | 预期产出 |
|---|---|---|---|
| 难负例 | 正样本来源：内容 / 行为 / 两者加权 | 第 4 节：内容与协同的折中 | 一条“内容检索 vs 行为召回 vs 下游效果”的权衡曲线 |
| 假负例处理 | 同用户共现屏蔽 + 流行度纠偏（logQ） | 第 3.3 节：行为数据的假负例与热门偏差 | 冷启动与长尾桶的提升是否来自纠偏 |
| 池化方式 | 模态诊断：只给图 / 只给文 / 都给，加模态 dropout | 第 6 节：模态失衡 | 视觉依赖度，以及短标题子集上的变化 |
| 二值化 | 稠密 / Matryoshka 降维 / RQ-Kmeans 语义 ID | 第 5、7 节：接入方式与存储 | 哪种形式在下游推荐里最好用 |

**logQ 纠偏的手算**（教学构造）。行为对里热门视频出现得多，batch 内负例就被热门视频占满，模型会过度“推开”热门视频。sampled softmax 的纠正是在每个候选的 logit 上减去它被采样的对数概率：$s_j/\tau-\log q_j$。热门视频 $q=0.01$，长尾视频 $q=0.0001$，两者的修正量相差 $\ln(0.01/0.0001)=4.61$ 个 logit 单位；温度 $\tau=0.05$ 时，相当于热门视频的余弦相似度要比长尾视频高 0.23，才会挨到同样大小的推力。原理见 [推荐评估与偏差](/notes/recsys-eval-bias/) 第 9 节。工业上的例子：Pinterest 的 OmniSearchSage（§3.5）训练查询向量时用的就是带 logQ 纠正的 sampled softmax。

**这个版本的“创新”怎样讲**。它不是新的 SOTA 方法，而是三件可验证的事：①在公开数据上完整复现“行为对齐的 MLLM 表征 → 语义 ID → 下游推荐”这条工业链路，开源可复现；②量化内容与协同的权衡曲线，SAIL 在线上只观察到“一个涨、一个跌”，你给出完整的曲线；③在行为数据上验证假负例屏蔽和流行度纠偏对冷启动的作用。面试时说“我在公开数据上复现了抖音 SAIL 和快手 QARM 的核心思路，并量化了它们论文里没展开的权衡”，比“我在 COCO 上调了参”有说服力得多。

**时间上怎么排**：10/6–10/8 的“COCO + CLIP 基线”换成“MicroLens 下载、切分、共现对、零样本基线”；10/10–10/13 的单卡、4 卡、首个完整训练不变；10/14 的首个结果表换成 I2I 召回表；SASRec 下游实验放在科研周之后（10/25 前后）。10/17、10/31 两个节点都不需要推迟。

## 12. 系统学习的阅读顺序

| 顺序 | 论文 | 重点看 |
|---|---|---|
| 1 | NoteLLM（2403.01744） | §4.3 共现对的构造与加权；§5.8 线上结果 |
| 2 | NoteLLM-2（2405.16789） | §2.4–2.5 视觉被忽视的证据；§4.5 线上部署 |
| 3 | QARM（2411.11739） | §2.2 物品对齐的两种数据源；§2.3 VQ/RQ 码；§2.4 三种特征用法 |
| 4 | SAIL-Embedding（2510.12709） | Table 1 训练数据；§3.3.5 协同增强；§4.2 评测体系；Table 6 线上 |
| 5 | YouTube Semantic IDs（2306.08121） | 为什么内容向量不能直接替换 ID |
| 6 | OneRec 技术报告（2506.13695） | tokenizer 部分：对比 + 描述损失、RQ-Kmeans |
| 7 | LEMUR（2511.10962） | §1 两阶段的四个问题；§3.3 会话屏蔽；§3.4 记忆库 |
| 8 | DME（2608.02148） | §4.1 数据与假负例过滤；§5.4 batch、帧数、token 预算的消融 |
| 9 | IDProxy（2603.01590） | 冷启动代理向量为什么要对齐到 ID 空间 |
| 10 | KuaiMod（2504.14904）、UniNote（2605.29287） | 内容审核与治理怎样用多模态大模型 |

每读一篇，回答同样四个问题：正样本从哪来？向量交给谁用？离线和线上看什么指标？为了上线做了哪些妥协？

## 13. 面试常问

**你的 embedding 和工业界用的有什么区别？** 正样本来源（描述对 vs 用户行为）、用途（直接检索 vs 喂给下游）、评测（Recall@K vs 下游 AUC 和线上 LT）、约束（单机 vs 十亿级库和毫秒延迟）。再说你在项目里怎样逼近其中哪几点。

**为什么不直接用 CLIP 向量做推荐？** 表征不匹配：图文相似不等于行为相似；表征学不动：缓存的固定向量接不到推荐梯度。工业解法是行为对齐 + 量化成可训练的码（QARM），或端到端（LEMUR）。

**语义 ID 和稠密向量怎么选？** 语义 ID 能像 ID 一样被下游训练，也方便做打散等规则；稠密向量信息更多，但作为固定特征难以更新。SAIL 的线上结果是语义 ID 收益更大。

**内容向量能替代 ID 吗？** 一般不能。YouTube 的实验显示直接替换会掉效果，因为丢了记忆能力；冻结 LVLM 的系统研究也发现融合严格好于替换。内容向量的主要价值在冷启动和长尾。

**行为数据训练有什么坑？** 共现对泄漏测试期、重度用户主导共现（用 $1/N_u$ 加权）、batch 内假负例（同会话、同用户屏蔽）、热门偏差（logQ 纠偏）、内容信息被训丢（加内容保持损失）。

## 闭卷验收

不看资料说出学术图文检索和工业表征的四个差距；画出第 2 节的系统地图，至少说出五个系统各自的业务、训练信号、接入方式和线上结果；写出 NoteLLM 的共现分公式并手算一个例子；解释 SAIL 协同增强后 GID-i2i 涨、Copair-i2i 跌的含义；列出向量接入下游的五种用法及各自的理由；手算一个两层 RQ-Kmeans 的编码和重建误差；用 NoteLLM-2 的数字说明视觉被忽视；说出 DME 关于 batch、帧数、图像 token 的三个结论；描述工业表征的三层离线评测；最后讲出把你的项目升级到 MicroLens 的数据、正样本、对照组、三层评测和四组消融，以及 logQ 纠偏的手算。

**图片说明。** 本文转载的 12 幅图来自采用 CC BY 4.0 或 CC0 的论文，图注写明了论文、arXiv 编号、图号、许可和修改情况；图中出现的视频画面和缩略图，权利仍属于原作者，论文的许可只覆盖论文作者自己的部分。正文里其余的“Figure N”链接指向 arXiv 原图，这些论文多数采用 arXiv 默认许可或禁止演绎的许可，不能转载。

**参考。** [SAIL-Embedding](https://arxiv.org/abs/2510.12709)；[Douyin Multimodal Embedding（DME）](https://arxiv.org/abs/2608.02148)；[LEMUR](https://arxiv.org/abs/2511.10962)；[QARM](https://arxiv.org/abs/2411.11739)；[QARM V2](https://arxiv.org/abs/2602.08559)；[OneRec Technical Report](https://arxiv.org/abs/2506.13695)；[KuaiMod](https://arxiv.org/abs/2504.14904)；[RGCD-Rep](https://arxiv.org/abs/2606.04448)；[RecoReward](https://arxiv.org/abs/2607.25901)；[TikTok 内容治理预训练](https://arxiv.org/abs/2509.21486)；[快手长尾搜索多模态重排](https://arxiv.org/abs/2603.24975)；[NoteLLM](https://arxiv.org/abs/2403.01744)；[NoteLLM-2](https://arxiv.org/abs/2405.16789)；[UniNote](https://arxiv.org/abs/2605.29287)；[IDProxy](https://arxiv.org/abs/2603.01590)；[MIM](https://arxiv.org/abs/2502.00321)；[ItemSage](https://arxiv.org/abs/2205.11728)；[OmniSearchSage](https://arxiv.org/abs/2404.16260)；[Better Generalization with Semantic IDs](https://arxiv.org/abs/2306.08121)；[LEARN](https://arxiv.org/abs/2405.03988)；[AdaSID](https://arxiv.org/abs/2604.23522)；[VLM2Rec](https://arxiv.org/abs/2603.17450)；[Frozen LVLMs for Micro-Video Recommendation](https://arxiv.org/abs/2512.21863)；[Describe What You See](https://arxiv.org/abs/2508.09789)；[MicroLens](https://arxiv.org/abs/2309.15379)（[数据与代码](https://github.com/westlake-repl/MicroLens)）；[Kwai Keye-VL](https://arxiv.org/abs/2507.01949)。
